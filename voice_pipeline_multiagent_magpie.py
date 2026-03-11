#!/usr/bin/env python3
"""Multi-agent voice pipeline (WebRTC) with Magpie TTS.

Supports multiple scenarios — each with its own agents, voices, tools,
and domain context.  The user picks a scenario before starting.

Pipeline:
  Browser Mic → AssemblyAI STT → LangGraph Orchestrator → Voice Switch → Magpie TTS (Riva NIM) → Browser Speaker

Prerequisites:
  1. cd multiagent && langgraph dev --n-jobs-per-worker 10
  2. source .venv/bin/activate && python voice_pipeline_multiagent_magpie.py
  3. Open http://localhost:7871 in your browser
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Dict, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from loguru import logger
from starlette.responses import StreamingResponse

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from assemblyai_multilingual_stt import AssemblyAIMultilingualSTTService
from riva_nim_realtime_tts import RivaNimRealtimeTTSService
from multiagent.voice.langgraph_processor import LangGraphProcessor, VoiceMap
from multiagent.voice.voice_switch_processor import VoiceSwitchProcessor
from multiagent.voice.frames import VoiceSwitchFrame
from multiagent.src.scenarios import get_scenario, list_scenarios

load_dotenv(override=True)
load_dotenv("multiagent/.env", override=True)

MAGPIE_SERVER = os.getenv("MAGPIE_TTS_SERVER", "192.168.7.203:9000")
DEFAULT_VOICE = "Magpie-Multilingual.EN-US.Aria"

LANGUAGE_PRESETS: dict[str, dict] = {
    "en": {"label": "English", "flag": "\U0001F1EC\U0001F1E7", "tts_language": "en-US"},
    "fr": {"label": "Fran\u00e7ais", "flag": "\U0001F1EB\U0001F1F7", "tts_language": "fr-FR"},
    "de": {"label": "Deutsch", "flag": "\U0001F1E9\U0001F1EA", "tts_language": "de-DE"},
    "es": {"label": "Espa\u00f1ol", "flag": "\U0001F1EA\U0001F1F8", "tts_language": "es-US"},
    "it": {"label": "Italiano", "flag": "\U0001F1EE\U0001F1F9", "tts_language": "it-IT"},
}


# ---------------------------------------------------------------------------
# Frame processors for SSE events (synced with TTS audio)
# ---------------------------------------------------------------------------

class TextEventBuffer:
    """Holds per-speaker SSE event groups until TTS starts speaking.

    Each group is paired with an asyncio.Event that is set when the
    group is complete (all token/end events added).  This prevents
    a race where TTSSyncCapture pops a group before it is fully
    populated — which happens for the *first* speaker because TTS
    opens its session (firing TTSStartedFrame) faster than the LLM
    response finishes flowing through the pipeline.
    """

    def __init__(self):
        self._groups: list[tuple[list[dict], asyncio.Event]] = []
        self._current: list[dict] | None = None
        self._current_ready: asyncio.Event | None = None

    def start_group(self, speaker_event: dict):
        self._current = [speaker_event]
        self._current_ready = asyncio.Event()
        self._groups.append((self._current, self._current_ready))
        logger.info(
            f"[TextEventBuffer] start_group speaker={speaker_event.get('speaker')} "
            f"(total groups={len(self._groups)})"
        )

    def mark_complete(self):
        if self._current_ready is not None:
            self._current_ready.set()
            logger.info(
                f"[TextEventBuffer] mark_complete "
                f"(group has {len(self._current) if self._current else 0} events, "
                f"pending groups={len(self._groups)})"
            )

    def add_event(self, event: dict):
        if self._current is not None:
            self._current.append(event)
            logger.debug(
                f"[TextEventBuffer] add_event type={event.get('event')} "
                f"(group now has {len(self._current)} events)"
            )

    def pop_group(self) -> tuple[list[dict], asyncio.Event | None]:
        if self._groups:
            group, ready = self._groups.pop(0)
            speaker = next((e.get('speaker') for e in group if e.get('event') == 'speaker'), '?')
            logger.info(
                f"[TextEventBuffer] pop_group speaker={speaker} "
                f"events={len(group)} ready={ready.is_set() if ready else 'N/A'} "
                f"(remaining groups={len(self._groups)})"
            )
            return group, ready
        logger.warning("[TextEventBuffer] pop_group called but NO groups available!")
        return [], None


class MultiAgentTextCapture(FrameProcessor):
    """Buffers LLM text/speaker events per speaker group.

    Streaming tokens are accumulated and emitted as a single token
    event when LLMFullResponseEndFrame arrives, so the UI receives
    the complete text for word-by-word reveal.

    On InterruptionFrame, any accumulated partial text is flushed
    so interrupted responses still appear in the UI.
    """

    def __init__(self, buffer: TextEventBuffer, **kwargs):
        super().__init__(**kwargs)
        self._buffer = buffer
        self._text_acc = ""

    def _flush_acc(self):
        if self._text_acc:
            logger.info(
                f"[MultiAgentTextCapture] _flush_acc: "
                f"{len(self._text_acc)} chars [{self._text_acc[:60]}...]"
            )
            self._buffer.add_event({"event": "token", "text": self._text_acc})
            self._buffer.add_event({"event": "end"})
            self._text_acc = ""
        else:
            logger.info("[MultiAgentTextCapture] _flush_acc: empty (no text accumulated)")
        self._buffer.mark_complete()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            pass
        elif isinstance(frame, InterruptionFrame):
            logger.info("[MultiAgentTextCapture] InterruptionFrame → flushing")
            self._flush_acc()
        elif isinstance(frame, VoiceSwitchFrame):
            logger.info(f"[MultiAgentTextCapture] VoiceSwitchFrame: speaker={frame.speaker}")
            self._text_acc = ""
            self._buffer.start_group({
                "event": "speaker",
                "speaker": frame.speaker,
                "voice": frame.voice,
            })
        elif isinstance(frame, LLMFullResponseStartFrame):
            logger.info("[MultiAgentTextCapture] LLMFullResponseStartFrame → adding 'start' event")
            self._buffer.add_event({"event": "start"})
        elif isinstance(frame, TextFrame):
            self._text_acc += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            logger.info("[MultiAgentTextCapture] LLMFullResponseEndFrame → flushing")
            self._flush_acc()

        await self.push_frame(frame, direction)


class TTSSyncCapture(FrameProcessor):
    """Placed AFTER TTS — flushes one buffered speaker group to the SSE
    queue each time TTS emits TTSStartedFrame (= audio is about to play).
    """

    def __init__(self, queue: asyncio.Queue, buffer: TextEventBuffer, **kwargs):
        super().__init__(**kwargs)
        self._queue = queue
        self._buffer = buffer

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            logger.info("[TTSSyncCapture] TTSStartedFrame received → popping group")
            group, ready = self._buffer.pop_group()
            if ready is not None:
                if not ready.is_set():
                    logger.info("[TTSSyncCapture] Group not ready yet — waiting...")
                try:
                    await asyncio.wait_for(ready.wait(), timeout=5.0)
                    logger.info("[TTSSyncCapture] Group ready — flushing to SSE queue")
                except asyncio.TimeoutError:
                    logger.warning("[TTSSyncCapture] Timed out waiting for group completion")
            events_summary = [e.get('event') for e in group]
            logger.info(
                f"[TTSSyncCapture] Sending {len(group)} events to SSE: {events_summary}"
            )
            for event in group:
                await self._queue.put(event)
        elif isinstance(frame, TTSStoppedFrame):
            logger.info("[TTSSyncCapture] TTSStoppedFrame received")
        elif isinstance(frame, InterruptionFrame):
            discarded = 0
            while True:
                group, ready = self._buffer.pop_group()
                if not group:
                    break
                if ready is not None:
                    ready.set()
                discarded += 1
            if discarded:
                logger.info(f"[TTSSyncCapture] Discarded {discarded} unspoken group(s) on interruption")
            logger.info("[TTSSyncCapture] InterruptionFrame passing through → transport")

        await self.push_frame(frame, direction)


class TranscriptionCapture(FrameProcessor):
    """Captures STT frames for the UI hearing bar."""

    def __init__(self, queue: asyncio.Queue, **kwargs):
        super().__init__(**kwargs)
        self._queue = queue

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterimTranscriptionFrame):
            await self._queue.put({"event": "stt_interim", "text": frame.text})
        elif isinstance(frame, TranscriptionFrame):
            await self._queue.put({"event": "stt_final", "text": frame.text})

        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pcs_map: Dict[str, SmallWebRTCConnection] = {}
active_tasks: Dict[str, tuple] = {}
pending_languages: Dict[str, str] = {}
pending_scenarios: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

async def run_voice_agent(webrtc_connection: SmallWebRTCConnection):
    langgraph_url = os.getenv("LANGGRAPH_URL", "http://localhost:2024")
    text_queue: asyncio.Queue = asyncio.Queue()

    pc_id = webrtc_connection.pc_id
    lang_code = pending_languages.pop(pc_id, "en")
    scenario_id = pending_scenarios.pop(pc_id, "ecommerce")
    preset = LANGUAGE_PRESETS.get(lang_code, LANGUAGE_PRESETS["en"])
    tts_language = preset["tts_language"]

    scenario = get_scenario(scenario_id)
    voice_map_dict = scenario.agent_voices
    first_voice = next(iter(voice_map_dict.values()), DEFAULT_VOICE)

    logger.info(
        f"[{pc_id}] Scenario={scenario.title} ({scenario_id}) "
        f"Language={preset['label']} ({lang_code}) "
        f"agents={list(voice_map_dict.keys())}"
    )

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
    )

    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                confidence=0.5,
                start_secs=0.2,
                stop_secs=0.8,
                min_volume=0.5,
            )
        ),
    )

    user_turn = UserTurnProcessor(
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy()],
            stop=[SpeechTimeoutUserTurnStopStrategy(timeout=1.0)],
        ),
    )

    stt = AssemblyAIMultilingualSTTService(
        language=os.getenv("ASSEMBLYAI_LANGUAGE", "multi"),
        sample_rate=16000,
    )

    tts = RivaNimRealtimeTTSService(
        server=MAGPIE_SERVER,
        voice_id=first_voice,
        language=tts_language,
        sample_rate=24000,
    )

    vm = VoiceMap(voices=dict(voice_map_dict))

    lg_processor = LangGraphProcessor(
        langgraph_url=langgraph_url,
        graph_name="orchestrator",
        voice_map=vm,
        scenario_id=scenario_id,
        tts=tts,
    )

    voice_switch = VoiceSwitchProcessor(tts=tts)
    stt_capture = TranscriptionCapture(text_queue)
    event_buffer = TextEventBuffer()
    text_capture = MultiAgentTextCapture(buffer=event_buffer)
    tts_sync = TTSSyncCapture(queue=text_queue, buffer=event_buffer)

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
            stt_capture,
            user_turn,
            lg_processor,
            text_capture,
            voice_switch,
            tts,
            tts_sync,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    active_tasks[pc_id] = (task, text_queue, tts, vm, scenario)

    @transport.event_handler("on_client_connected")
    async def on_connected(transport_obj, connection):
        logger.info(f"Client connected: {pc_id}")

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport_obj, connection):
        logger.info(f"Client disconnected: {pc_id}")
        active_tasks.pop(pc_id, None)

    runner = PipelineRunner(handle_sigint=False)
    logger.info("Voice pipeline running — speak in the browser!")
    await runner.run(task)
    active_tasks.pop(pc_id, None)


# ---------------------------------------------------------------------------
# WebRTC signalling endpoints
# ---------------------------------------------------------------------------

@app.post("/api/offer")
async def api_offer(req: Request, background_tasks: BackgroundTasks):
    try:
        body = await req.json()
        pc_id = body.get("pc_id")
        sdp = body.get("sdp")
        offer_type = body.get("type")

        if not sdp or not offer_type:
            return JSONResponse({"error": "Missing sdp/type"}, status_code=400)

        if pc_id and pc_id in pcs_map:
            conn = pcs_map[pc_id]
            await conn.renegotiate(
                sdp=sdp, type=offer_type,
                restart_pc=body.get("restart_pc", False),
            )
        else:
            conn = SmallWebRTCConnection()
            await conn.initialize(sdp=sdp, type=offer_type)

            lang = body.get("language", "en")
            if lang not in LANGUAGE_PRESETS:
                lang = "en"
            pending_languages[conn.pc_id] = lang
            pending_scenarios[conn.pc_id] = body.get("scenario", "ecommerce")

            @conn.event_handler("closed")
            async def on_closed(c: SmallWebRTCConnection):
                pcs_map.pop(c.pc_id, None)
                pending_languages.pop(c.pc_id, None)
                pending_scenarios.pop(c.pc_id, None)

            background_tasks.add_task(run_voice_agent, conn)

        answer = conn.get_answer()
        pcs_map[answer["pc_id"]] = conn
        return JSONResponse(answer)

    except Exception as e:
        logger.exception(f"offer error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.patch("/api/offer")
async def api_offer_patch(req: Request):
    body = await req.json()
    pc_id = body.get("pc_id")

    if not pc_id or pc_id not in pcs_map:
        return JSONResponse({"error": "Connection not found"}, status_code=404)

    conn = pcs_map[pc_id]

    if "candidate" in body:
        try:
            await conn.add_ice_candidate(body["candidate"])
        except (AttributeError, TypeError):
            pass

    if "sdp" in body:
        await conn.renegotiate(
            sdp=body["sdp"], type=body.get("type", "offer"),
            restart_pc=body.get("restart_pc", False),
        )
        return JSONResponse(conn.get_answer())

    return JSONResponse({"status": "ok"})


@app.get("/api/events/{pc_id}")
async def events(pc_id: str):
    if pc_id not in active_tasks:
        return JSONResponse({"error": "No active session"}, status_code=404)

    _, text_queue, _, _, _ = active_tasks[pc_id]

    async def generate():
        while True:
            try:
                msg = await asyncio.wait_for(text_queue.get(), timeout=120)
                yield f"data: {json.dumps(msg)}\n\n"
            except asyncio.TimeoutError:
                yield "data: {}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/voices")
async def list_voices():
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"http://{MAGPIE_SERVER}/v1/audio/list_voices")
            resp.raise_for_status()
            data = resp.json()
            voices = []
            for lang_voices in data.values():
                if isinstance(lang_voices, dict) and "voices" in lang_voices:
                    voices.extend(lang_voices["voices"])
            return sorted(set(voices))
    except Exception as e:
        logger.error(f"Failed to fetch voices: {e}")
        return JSONResponse({"error": str(e)}, status_code=502)


@app.post("/api/voice")
async def set_agent_voice(req: Request):
    body = await req.json()
    pc_id = body.get("pc_id", "")
    agent = body.get("agent", "")
    voice = body.get("voice", "").strip()

    if not agent or not voice:
        return JSONResponse({"error": "Need agent and voice"}, status_code=400)

    if pc_id and pc_id in active_tasks:
        _, _, _, voice_map, _ = active_tasks[pc_id]
        voice_map.set(agent, voice)
        logger.info(f"[{pc_id}] {agent} voice → {voice}")

    return JSONResponse({"status": "ok", "agent": agent, "voice": voice})


@app.get("/nvidia_icon.png")
async def nvidia_icon():
    return FileResponse("nvidia_icon.png", media_type="image/png")


@app.get("/api/scenarios")
async def api_list_scenarios():
    scenarios = list_scenarios()
    result = []
    for s in scenarios:
        agents = []
        for a in s.agents:
            fast = [{"name": t.name, "description": t.description or ""} for t in a.fast_tools]
            slow = [{"name": t.name, "description": t.description or ""} for t in a.slow_launchers]
            agents.append({
                "id": a.id, "name": a.name, "role": a.role, "color": a.color,
                "tools_instant": fast,
                "tools_long_running": slow,
            })
        result.append({
            "id": s.id,
            "title": s.title,
            "description": s.description,
            "guide_overview": s.guide_overview,
            "suggested_questions": s.suggested_questions,
            "agents": agents,
        })
    return result


@app.get("/api/languages")
async def list_languages():
    return [
        {"code": code, **{k: v for k, v in preset.items() if k != "tts_language"}}
        for code, preset in LANGUAGE_PRESETS.items()
    ]


@app.get("/health")
async def health():
    return {"status": "healthy", "mode": "multi_agent_voice_magpie"}


@app.post("/start")
async def start_agent(request: Request):
    proto = request.headers.get("x-forwarded-proto", "http")
    host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("host")
        or str(request.base_url.hostname)
    )
    offer_url = f"{proto}://{host}/api/offer"
    return {"url": offer_url, "offerUrl": offer_url, "webrtcUrl": offer_url}


# ---------------------------------------------------------------------------
# Custom Chat UI
# ---------------------------------------------------------------------------

CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NVIDIA Multi-Agent Voice</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --nv-green: #76B900;
    --nv-green-dim: #5a8f00;
    --nv-green-glow: rgba(118,185,0,0.25);
    --nv-black: #000000;
    --nv-bg: #0a0a0a;
    --nv-surface: #141414;
    --nv-surface2: #1e1e1e;
    --nv-border: #2a2a2a;
    --nv-text: #e8e8e8;
    --nv-text-dim: #999999;
    --nv-text-muted: #666666;
    --nv-user: #76B900;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Inter', -apple-system, system-ui, sans-serif; background: var(--nv-bg); color: var(--nv-text); height: 100vh; display: flex; flex-direction: column; }

  /* ── Header ── */
  .header { padding: 14px 24px; background: var(--nv-black); border-bottom: 1px solid var(--nv-border); display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
  .brand { display: flex; align-items: center; gap: 10px; }
  .brand .nv-logo { height: 28px; width: auto; }
  .brand h1 { font-size: 16px; font-weight: 600; letter-spacing: -0.2px; }
  .brand h1 .nv { color: var(--nv-green); font-weight: 700; }
  .status { font-size: 11px; padding: 4px 10px; border-radius: 4px; background: var(--nv-surface2); color: var(--nv-text-muted); font-weight: 500; letter-spacing: 0.3px; text-transform: uppercase; }
  .status.connected { background: rgba(118,185,0,0.12); color: var(--nv-green); border: 1px solid rgba(118,185,0,0.25); }
  .status.connecting { background: rgba(250,204,21,0.1); color: #facc15; border: 1px solid rgba(250,204,21,0.2); }

  #mic-btn { width: 36px; height: 36px; border-radius: 4px; border: 1px solid var(--nv-border); background: var(--nv-surface); color: var(--nv-text-dim); cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.15s ease; }
  #mic-btn:hover { border-color: var(--nv-text-muted); color: var(--nv-text); }
  #mic-btn.active { border-color: var(--nv-green); color: var(--nv-green); }
  #mic-btn.active.listening { border-color: var(--nv-green); box-shadow: 0 0 0 2px var(--nv-green-glow); animation: pulse 1.5s ease-in-out infinite; }
  #mic-btn.muted { border-color: #ef4444; color: #ef4444; }
  @keyframes pulse { 0%,100% { box-shadow: 0 0 0 2px var(--nv-green-glow); } 50% { box-shadow: 0 0 0 6px var(--nv-green-glow); } }

  .voice-row { display: flex; gap: 14px; align-items: center; margin-left: auto; }
  .voice-group { display: flex; align-items: center; gap: 6px; }
  .voice-group label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
  .voice-select { padding: 4px 8px; border-radius: 4px; border: 1px solid var(--nv-border); background: var(--nv-surface); color: var(--nv-text); font-size: 11px; font-family: 'Inter', sans-serif; outline: none; max-width: 150px; transition: border-color 0.15s; }
  .voice-select:focus { border-color: var(--nv-green); }
  #speaking { display: none; color: var(--nv-green); font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }

  /* ── STT bar ── */
  #stt-bar { padding: 8px 24px; background: var(--nv-surface); border-bottom: 1px solid var(--nv-border); font-size: 13px; min-height: 34px; display: flex; align-items: center; gap: 10px; }
  #stt-bar .label { color: var(--nv-text-muted); font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; }
  #stt-bar .text { color: var(--nv-text-dim); font-style: italic; }
  #stt-bar .text.final { color: var(--nv-text); font-style: normal; font-weight: 500; }

  /* ── Chat ── */
  .chat { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 14px; }
  .chat::-webkit-scrollbar { width: 6px; }
  .chat::-webkit-scrollbar-track { background: transparent; }
  .chat::-webkit-scrollbar-thumb { background: var(--nv-border); border-radius: 3px; }

  .msg { max-width: 78%; padding: 12px 16px; border-radius: 8px; font-size: 14px; line-height: 1.6; }
  .msg.user { align-self: flex-end; background: var(--nv-green); color: var(--nv-black); border-bottom-right-radius: 2px; font-weight: 500; }
  .msg.bot { align-self: flex-start; border-bottom-left-radius: 2px; }
  .msg.bot { background: var(--nv-surface2); border-left: 3px solid var(--nv-border); }
  .msg.system { align-self: center; background: transparent; color: var(--nv-text-muted); font-size: 13px; padding: 8px 0; }

  .speaker-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }

  /* ── Overlay ── */
  #start-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.92); display: flex; align-items: center; justify-content: center; z-index: 200; flex-direction: column; gap: 20px; }
  #start-overlay .nv-badge { margin-bottom: 8px; }
  #start-overlay h2 { color: var(--nv-text); font-size: 26px; font-weight: 600; letter-spacing: -0.3px; }
  #start-overlay p { color: var(--nv-text-dim); font-size: 14px; max-width: 420px; text-align: center; line-height: 1.6; }

  .scenario-picker { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; margin: 8px 0; max-width: 700px; }
  .scenario-btn { padding: 14px 18px; border-radius: 8px; border: 2px solid var(--nv-border); background: var(--nv-surface); color: var(--nv-text-dim); cursor: pointer; font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 500; transition: all 0.15s ease; text-align: left; min-width: 160px; max-width: 220px; }
  .scenario-btn:hover { border-color: var(--nv-text-muted); color: var(--nv-text); }
  .scenario-btn.selected { border-color: var(--nv-green); color: var(--nv-green); background: rgba(118,185,0,0.08); box-shadow: 0 0 12px var(--nv-green-glow); }
  .scenario-btn .sc-title { font-weight: 600; font-size: 13px; display: block; margin-bottom: 4px; }
  .scenario-btn .sc-agents { font-size: 11px; color: var(--nv-text-muted); display: block; }

  .lang-picker { display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; margin: 4px 0; }
  .lang-btn { padding: 10px 18px; border-radius: 8px; border: 2px solid var(--nv-border); background: var(--nv-surface); color: var(--nv-text-dim); cursor: pointer; font-family: 'Inter', sans-serif; font-size: 14px; font-weight: 500; transition: all 0.15s ease; display: flex; align-items: center; gap: 8px; }
  .lang-btn:hover { border-color: var(--nv-text-muted); color: var(--nv-text); }
  .lang-btn.selected { border-color: var(--nv-green); color: var(--nv-green); background: rgba(118,185,0,0.08); box-shadow: 0 0 12px var(--nv-green-glow); }
  .lang-btn .flag { font-size: 20px; }

  #start-overlay button#start-btn { padding: 16px 48px; font-size: 16px; border-radius: 6px; border: none; background: var(--nv-green); color: var(--nv-black); cursor: pointer; font-weight: 700; font-family: 'Inter', sans-serif; letter-spacing: 0.3px; transition: all 0.15s ease; }
  #start-overlay button#start-btn:hover { background: #8cd100; box-shadow: 0 0 20px var(--nv-green-glow); }

  /* ── Guide panel ── */
  #guide-btn { width: 36px; height: 36px; border-radius: 4px; border: 1px solid var(--nv-border); background: var(--nv-surface); color: var(--nv-text-dim); cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.15s ease; font-family: 'Inter', sans-serif; font-size: 15px; font-weight: 700; }
  #guide-btn:hover { border-color: var(--nv-green); color: var(--nv-green); }
  #guide-btn.open { border-color: var(--nv-green); color: var(--nv-green); background: rgba(118,185,0,0.08); }

  .guide-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.5); z-index: 150; opacity: 0; pointer-events: none; transition: opacity 0.2s ease; }
  .guide-overlay.visible { opacity: 1; pointer-events: auto; }

  .guide-panel { position: fixed; top: 0; right: 0; bottom: 0; width: 420px; max-width: 90vw; background: var(--nv-surface); border-left: 1px solid var(--nv-border); z-index: 160; transform: translateX(100%); transition: transform 0.25s ease; display: flex; flex-direction: column; }
  .guide-panel.open { transform: translateX(0); }

  .guide-header { padding: 16px 20px; border-bottom: 1px solid var(--nv-border); display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
  .guide-header h2 { font-size: 15px; font-weight: 600; color: var(--nv-green); }
  .guide-close { width: 32px; height: 32px; border: none; background: transparent; color: var(--nv-text-dim); cursor: pointer; font-size: 20px; display: flex; align-items: center; justify-content: center; border-radius: 4px; }
  .guide-close:hover { background: var(--nv-surface2); color: var(--nv-text); }

  .guide-body { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 22px; }
  .guide-body::-webkit-scrollbar { width: 5px; }
  .guide-body::-webkit-scrollbar-track { background: transparent; }
  .guide-body::-webkit-scrollbar-thumb { background: var(--nv-border); border-radius: 3px; }

  .guide-section h3 { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: var(--nv-text-muted); margin-bottom: 10px; }
  .guide-overview { font-size: 13px; line-height: 1.7; color: var(--nv-text-dim); }

  .guide-questions { list-style: none; padding: 0; display: flex; flex-direction: column; gap: 6px; }
  .guide-questions li { font-size: 13px; padding: 8px 12px; background: var(--nv-surface2); border-radius: 6px; color: var(--nv-text-dim); border: 1px solid transparent; cursor: default; line-height: 1.5; display: flex; align-items: flex-start; gap: 8px; }
  .guide-questions li::before { content: '\201C'; color: var(--nv-green); font-size: 18px; font-weight: 700; line-height: 1; flex-shrink: 0; margin-top: 1px; }

  .guide-agent { background: var(--nv-surface2); border-radius: 8px; padding: 14px; display: flex; flex-direction: column; gap: 10px; }
  .guide-agent-header { display: flex; align-items: center; gap: 8px; }
  .guide-agent-name { font-size: 13px; font-weight: 700; }
  .guide-agent-role { font-size: 11px; color: var(--nv-text-muted); }
  .guide-tools-label { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--nv-text-muted); margin-top: 2px; }
  .guide-tool { font-size: 12px; color: var(--nv-text-dim); padding: 4px 0; display: flex; flex-direction: column; gap: 2px; }
  .guide-tool-name { font-weight: 600; color: var(--nv-text); font-size: 12px; }
  .guide-tool-desc { font-size: 11px; color: var(--nv-text-muted); line-height: 1.4; }
  .guide-tool-badge { display: inline-block; font-size: 9px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; padding: 1px 5px; border-radius: 3px; margin-left: 6px; }
  .guide-tool-badge.instant { background: rgba(118,185,0,0.15); color: var(--nv-green); }
  .guide-tool-badge.slow { background: rgba(250,204,21,0.12); color: #facc15; }
</style>
</head>
<body>
<div id="start-overlay">
  <div class="nv-badge">
    <img src="/nvidia_icon.png" alt="NVIDIA" style="height:64px;width:auto;">
  </div>
  <h2>Multi-Agent Voice Collaboration</h2>
  <p id="scenario-desc">Select a scenario to start your conversation.</p>
  <p style="color:var(--nv-text);font-weight:500;margin-top:8px;">Choose a scenario</p>
  <div class="scenario-picker" id="scenario-picker"></div>
  <p style="color:var(--nv-text);font-weight:500;margin-top:8px;">Choose your language</p>
  <div class="lang-picker" id="lang-picker"></div>
  <button id="start-btn">Start Conversation</button>
</div>
<div class="header">
  <div class="brand">
    <img class="nv-logo" src="/nvidia_icon.png" alt="NVIDIA">
    <h1>Multi-Agent Voice <span id="header-scenario" style="color:var(--nv-text-muted);font-weight:400;font-size:12px">Magpie</span></h1>
  </div>
  <span id="status" class="status">Disconnected</span>
  <button id="mic-btn" title="Mute / Unmute">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
  </button>
  <span id="speaking">Speaking...</span>
  <div class="voice-row" id="voice-row"></div>
  <button id="guide-btn" title="Scenario Guide">?</button>
  <audio id="audio" autoplay playsinline></audio>
</div>
<div class="guide-overlay" id="guide-overlay"></div>
<div class="guide-panel" id="guide-panel">
  <div class="guide-header">
    <h2 id="guide-title">Scenario Guide</h2>
    <button class="guide-close" id="guide-close-btn">&times;</button>
  </div>
  <div class="guide-body" id="guide-body"></div>
</div>
<div id="stt-bar">
  <span class="label">Hearing:</span>
  <span id="stt-text" class="text"></span>
</div>
<div class="chat" id="chat">
  <div class="msg system" id="chat-welcome">Select a scenario and click Start to begin.</div>
</div>

<script>
let AGENT_NAMES = {};
let AGENT_COLORS = {};
let selectedScenario = '';
let scenarioData = [];

let pc = null, pcId = null, localStream = null, micMuted = false;
let eventSource = null, sttClearTimer = null;
let selectedLang = 'en';
let langPresets = {};

/* ── Scenario picker ── */
async function loadScenarios() {
  try {
    const resp = await fetch('/api/scenarios');
    scenarioData = await resp.json();
    const picker = document.getElementById('scenario-picker');
    picker.innerHTML = '';
    scenarioData.forEach((sc, idx) => {
      const btn = document.createElement('button');
      btn.className = 'scenario-btn' + (idx === 0 ? ' selected' : '');
      btn.innerHTML = '<span class="sc-title">' + sc.title + '</span>' +
        '<span class="sc-agents">' + sc.agents.map(a => a.name + ' (' + a.role + ')').join(' & ') + '</span>';
      btn.addEventListener('click', () => {
        selectedScenario = sc.id;
        picker.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
        document.getElementById('scenario-desc').textContent = sc.description;
      });
      picker.appendChild(btn);
      if (idx === 0) {
        selectedScenario = sc.id;
        document.getElementById('scenario-desc').textContent = sc.description;
      }
    });
  } catch (e) { console.error('Failed to load scenarios', e); }
}

function applyScenario() {
  const sc = scenarioData.find(s => s.id === selectedScenario);
  if (!sc) return;
  AGENT_NAMES = {};
  AGENT_COLORS = {};
  sc.agents.forEach(a => {
    AGENT_NAMES[a.id] = a.name;
    AGENT_COLORS[a.id] = a.color;
  });
  document.getElementById('header-scenario').textContent = sc.title;
  const names = sc.agents.map(a => a.name).join(' and ');
  document.getElementById('chat-welcome').textContent = 'Speak to ' + names + '. They will collaborate to help you.';
  buildVoiceRow(sc.agents);
}

function buildVoiceRow(agents) {
  const row = document.getElementById('voice-row');
  row.innerHTML = '';
  agents.forEach(a => {
    const group = document.createElement('div');
    group.className = 'voice-group';
    const lbl = document.createElement('label');
    lbl.textContent = a.name + ':';
    lbl.style.color = a.color;
    const sel = document.createElement('select');
    sel.className = 'voice-select';
    sel.id = 'voice-' + a.id;
    sel.disabled = true;
    sel.innerHTML = '<option>Loading...</option>';
    sel.addEventListener('change', () => setAgentVoice(a.id, sel.value));
    group.appendChild(lbl);
    group.appendChild(sel);
    row.appendChild(group);
  });
}

/* ── Language picker ── */
async function loadLanguages() {
  try {
    const resp = await fetch('/api/languages');
    const langs = await resp.json();
    const picker = document.getElementById('lang-picker');
    picker.innerHTML = '';
    langs.forEach(lang => {
      langPresets[lang.code] = lang;
      const btn = document.createElement('button');
      btn.className = 'lang-btn' + (lang.code === selectedLang ? ' selected' : '');
      btn.setAttribute('data-lang', lang.code);
      btn.innerHTML = '<span class="flag">' + lang.flag + '</span>' + lang.label;
      btn.addEventListener('click', () => {
        selectedLang = lang.code;
        picker.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
      });
      picker.appendChild(btn);
    });
  } catch (e) { console.error('Failed to load languages', e); }
}

loadScenarios();
loadLanguages();

const statusEl = document.getElementById('status');
const chatEl = document.getElementById('chat');
const audioEl = document.getElementById('audio');
const micBtn = document.getElementById('mic-btn');
const sttTextEl = document.getElementById('stt-text');

/* ── Text display state ── */
let currentSpeaker = '';
let currentBotMsg = null;
let currentContentEl = null;

function createBotBubble(speaker) {
  currentBotMsg = document.createElement('div');
  currentBotMsg.className = 'msg bot';
  const color = AGENT_COLORS[speaker] || '#666';
  currentBotMsg.style.borderLeftColor = color;
  if (speaker && AGENT_NAMES[speaker]) {
    const label = document.createElement('div');
    label.className = 'speaker-label';
    label.style.color = color;
    label.textContent = AGENT_NAMES[speaker];
    currentBotMsg.appendChild(label);
  }
  currentContentEl = document.createElement('span');
  currentContentEl.className = 'bot-content';
  currentBotMsg.appendChild(currentContentEl);
}

function addMessage(text, cls) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = 'status ' + (cls || '');
}

/* ── Voice dropdowns ── */
async function loadVoices() {
  try {
    const resp = await fetch('/api/voices');
    const voices = await resp.json();
    const names = voices.map(v => typeof v === 'string' ? v : (v.name || v));
    const sc = scenarioData.find(s => s.id === selectedScenario);
    if (!sc) return;
    sc.agents.forEach(a => {
      const sel = document.getElementById('voice-' + a.id);
      if (!sel) return;
      sel.innerHTML = '';
      names.forEach(n => {
        const opt = document.createElement('option');
        opt.value = n; opt.textContent = n;
        sel.appendChild(opt);
      });
      sel.disabled = false;
    });
  } catch (e) {
    console.error('Failed to load voices', e);
  }
}

async function setAgentVoice(agent, voice) {
  if (!pcId) return;
  try {
    await fetch('/api/voice', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pc_id: pcId, agent, voice }),
    });
  } catch (e) { console.error('Failed to set voice', e); }
}

/* ── Mic ── */
micBtn.addEventListener('click', () => {
  if (!localStream) return;
  micMuted = !micMuted;
  localStream.getAudioTracks().forEach(t => { t.enabled = !micMuted; });
  micBtn.classList.toggle('muted', micMuted);
});

/* ── WebRTC ── */
async function connect() {
  setStatus('Connecting...', 'connecting');
  try {
    localStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
  } catch (err) {
    addMessage('Microphone access denied.', 'system');
  }

  pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });

  pc.ontrack = (event) => {
    audioEl.srcObject = event.streams[0];
    audioEl.play().catch(() => {
      document.addEventListener('click', () => { audioEl.play(); }, { once: true });
    });
    monitorAudio();
  };

  pc.oniceconnectionstatechange = () => {
    if (pc.iceConnectionState === 'connected') {
      setStatus('Connected', 'connected');
      micBtn.classList.add('active');
      connectSSE();
    } else if (pc.iceConnectionState === 'disconnected' || pc.iceConnectionState === 'failed') {
      setStatus('Disconnected');
      micBtn.classList.remove('active', 'listening');
      if (eventSource) { eventSource.close(); eventSource = null; }
    }
  };

  if (localStream) {
    localStream.getAudioTracks().forEach(track => pc.addTrack(track, localStream));
  } else {
    pc.addTransceiver('audio', { direction: 'recvonly' });
  }

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  await new Promise((resolve) => {
    if (pc.iceGatheringState === 'complete') return resolve();
    const timeout = setTimeout(resolve, 3000);
    pc.onicegatheringstatechange = () => {
      if (pc.iceGatheringState === 'complete') { clearTimeout(timeout); resolve(); }
    };
  });

  const resp = await fetch('/api/offer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type, pc_id: pcId, language: selectedLang, scenario: selectedScenario }),
  });
  const answer = await resp.json();
  pcId = answer.pc_id;
  await pc.setRemoteDescription(answer);
}

/* ── SSE events ── */
function connectSSE(retries) {
  if (!pcId) return;
  if (eventSource) eventSource.close();
  retries = retries || 0;
  eventSource = new EventSource('/api/events/' + encodeURIComponent(pcId));

  let pendingUserText = '';

  eventSource.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (!msg.event) return;

    console.log('[SSE] event=' + msg.event + (msg.speaker ? ' speaker=' + msg.speaker : '') + (msg.text ? ' text=[' + msg.text.substring(0, 50) + '...]' : ''));

    if (msg.event === 'stt_interim') {
      sttTextEl.textContent = msg.text;
      sttTextEl.className = 'text';
      micBtn.classList.add('listening');
      if (sttClearTimer) clearTimeout(sttClearTimer);

    } else if (msg.event === 'stt_final') {
      if (pendingUserText) {
        addMessage(pendingUserText, 'user');
      }
      pendingUserText = msg.text;
      sttTextEl.textContent = msg.text;
      sttTextEl.className = 'text final';
      micBtn.classList.remove('listening');
      if (sttClearTimer) clearTimeout(sttClearTimer);
      sttClearTimer = setTimeout(() => {
        if (pendingUserText) {
          addMessage(pendingUserText, 'user');
          pendingUserText = '';
        }
        sttTextEl.textContent = '';
      }, 3000);

    } else if (msg.event === 'speaker') {
      currentSpeaker = msg.speaker || '';
      console.log('[SSE] Speaker set to: ' + currentSpeaker);

    } else if (msg.event === 'start') {
      if (pendingUserText) {
        addMessage(pendingUserText, 'user');
        pendingUserText = '';
        sttTextEl.textContent = '';
        if (sttClearTimer) clearTimeout(sttClearTimer);
      }
      console.log('[SSE] Creating bot bubble for: ' + currentSpeaker);
      createBotBubble(currentSpeaker);
      chatEl.appendChild(currentBotMsg);
      chatEl.scrollTop = chatEl.scrollHeight;

    } else if (msg.event === 'token') {
      if (!currentBotMsg) {
        createBotBubble(currentSpeaker);
        chatEl.appendChild(currentBotMsg);
      }
      if (currentContentEl) {
        currentContentEl.textContent = msg.text;
        chatEl.scrollTop = chatEl.scrollHeight;
      }

    } else if (msg.event === 'end') {
      if (currentBotMsg && currentContentEl && !currentContentEl.textContent) {
        currentBotMsg.remove();
      }
      currentBotMsg = null;
      currentContentEl = null;
    }
  };

  eventSource.onerror = () => {
    eventSource.close(); eventSource = null;
    if (retries < 5) setTimeout(() => connectSSE(retries + 1), 500);
  };
}

/* ── Audio level monitor ── */
function monitorAudio() {
  if (!audioEl.srcObject) return;
  const ctx = new AudioContext();
  const src = ctx.createMediaStreamSource(audioEl.srcObject);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 256;
  src.connect(analyser);
  const data = new Uint8Array(analyser.frequencyBinCount);
  const speakEl = document.getElementById('speaking');
  (function poll() {
    analyser.getByteFrequencyData(data);
    const vol = data.reduce((a,b) => a+b, 0) / data.length;
    speakEl.style.display = vol > 2 ? 'inline' : 'none';
    requestAnimationFrame(poll);
  })();
}

document.getElementById('start-btn').addEventListener('click', () => {
  document.getElementById('start-overlay').style.display = 'none';
  applyScenario();
  loadVoices();
  connect();
  buildGuide();
});

/* ── Guide panel ── */
const guideBtn = document.getElementById('guide-btn');
const guidePanel = document.getElementById('guide-panel');
const guideOverlay = document.getElementById('guide-overlay');

function toggleGuide() {
  const open = guidePanel.classList.toggle('open');
  guideOverlay.classList.toggle('visible', open);
  guideBtn.classList.toggle('open', open);
}
guideBtn.addEventListener('click', toggleGuide);
document.getElementById('guide-close-btn').addEventListener('click', toggleGuide);
guideOverlay.addEventListener('click', toggleGuide);

function buildGuide() {
  const sc = scenarioData.find(s => s.id === selectedScenario);
  if (!sc) return;

  document.getElementById('guide-title').textContent = sc.title + ' — Guide';
  const body = document.getElementById('guide-body');
  body.innerHTML = '';

  if (sc.guide_overview) {
    const sec = document.createElement('div');
    sec.className = 'guide-section';
    sec.innerHTML = '<h3>About this scenario</h3><p class="guide-overview">' + escapeHtml(sc.guide_overview) + '</p>';
    body.appendChild(sec);
  }

  if (sc.suggested_questions && sc.suggested_questions.length) {
    const sec = document.createElement('div');
    sec.className = 'guide-section';
    sec.innerHTML = '<h3>Try asking</h3>';
    const ul = document.createElement('ul');
    ul.className = 'guide-questions';
    sc.suggested_questions.forEach(q => {
      const li = document.createElement('li');
      li.textContent = q;
      ul.appendChild(li);
    });
    sec.appendChild(ul);
    body.appendChild(sec);
  }

  if (sc.agents && sc.agents.length) {
    const sec = document.createElement('div');
    sec.className = 'guide-section';
    sec.innerHTML = '<h3>Agent tools</h3>';
    sc.agents.forEach(a => {
      const card = document.createElement('div');
      card.className = 'guide-agent';
      const color = a.color || '#666';

      let html = '<div class="guide-agent-header">'
        + '<span class="guide-agent-name" style="color:' + color + '">' + escapeHtml(a.name) + '</span>'
        + '<span class="guide-agent-role">' + escapeHtml(a.role) + '</span></div>';

      if (a.tools_instant && a.tools_instant.length) {
        html += '<div class="guide-tools-label">Instant checks</div>';
        a.tools_instant.forEach(t => {
          html += '<div class="guide-tool"><span class="guide-tool-name">' + formatToolName(t.name) + '<span class="guide-tool-badge instant">instant</span></span>';
          if (t.description) html += '<span class="guide-tool-desc">' + escapeHtml(t.description) + '</span>';
          html += '</div>';
        });
      }

      if (a.tools_long_running && a.tools_long_running.length) {
        html += '<div class="guide-tools-label">Long-running actions</div>';
        a.tools_long_running.forEach(t => {
          html += '<div class="guide-tool"><span class="guide-tool-name">' + formatToolName(t.name) + '<span class="guide-tool-badge slow">long</span></span>';
          if (t.description) html += '<span class="guide-tool-desc">' + escapeHtml(t.description) + '</span>';
          html += '</div>';
        });
      }

      card.innerHTML = html;
      sec.appendChild(card);
    });
    body.appendChild(sec);
  }
}

function formatToolName(name) {
  return escapeHtml(name.replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase()));
}

function escapeHtml(str) {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return CHAT_HTML


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Voice Pipeline")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7871)
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()

    logger.remove(0)
    if args.verbose >= 2:
        logger.add(sys.stderr, level="TRACE")
    elif args.verbose >= 1:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    scenarios = list_scenarios()
    logger.info("=" * 60)
    logger.info("  Multi-Agent Voice Pipeline — Magpie TTS (WebRTC)")
    logger.info("=" * 60)
    logger.info(f"  STT:  AssemblyAI ({os.getenv('ASSEMBLYAI_LANGUAGE', 'multi')})")
    logger.info(f"  LLM:  LangGraph orchestrator @ {os.getenv('LANGGRAPH_URL', 'http://localhost:2024')}")
    logger.info(f"  TTS:  Magpie (Riva NIM) @ {MAGPIE_SERVER}")
    logger.info(f"  Scenarios: {len(scenarios)}")
    for sc in scenarios:
        agents = ", ".join(f"{a.name} ({a.role})" for a in sc.agents)
        logger.info(f"    {sc.id}: {sc.title} [{agents}]")
    logger.info("=" * 60)
    logger.info(f"  Open http://localhost:{args.port} in your browser")
    logger.info("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port)
