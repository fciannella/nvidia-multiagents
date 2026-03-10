#!/usr/bin/env python3
"""Multi-agent voice pipeline (WebRTC) with custom chat UI.

Two specialized agents (Data Scientist + IT Infrastructure) collaborate
with a human via voice. Each agent has a distinct TTS voice and is
shown with a colored label in the chat transcript.

Pipeline:
  Browser Mic → AssemblyAI STT → LangGraph Orchestrator → Voice Switch → Qwen3 TTS → Browser Speaker

Prerequisites:
  1. cd multiagent && langgraph dev --n-jobs-per-worker 10
  2. source .venv/bin/activate && python voice_pipeline_multiagent.py
  3. Open http://localhost:7870 in your browser
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
from qwen3_tts import Qwen3TTSService
from multiagent.voice.langgraph_processor import LangGraphProcessor, VoiceMap
from multiagent.voice.voice_switch_processor import VoiceSwitchProcessor
from multiagent.voice.frames import VoiceSwitchFrame

load_dotenv(override=True)
load_dotenv("multiagent/.env", override=True)

TTS_SERVER = os.getenv("TTS_SERVER", "http://192.168.7.163:8100")
# TUNNEL: TTS_SERVER = os.getenv("TTS_SERVER", "http://localhost:18100")
DEFAULT_VOICE = os.getenv("MA_DS_VOICE", "qwen3-data-scientist-demo-en-en")

DS_NAME = os.getenv("MA_DS_NAME", "Sarah")
IT_NAME = os.getenv("MA_IT_NAME", "Mike")
DS_VOICE = os.getenv("MA_DS_VOICE", "qwen3-data-scientist-demo-en-en")
IT_VOICE = os.getenv("MA_IT_VOICE", "qwen3-male-it-expert-en")

LANGUAGE_PRESETS: dict[str, dict] = {
    "en": {
        "label": "English",
        "flag": "\U0001F1EC\U0001F1E7",
        "tts_language": "English",
        "ds_voice": os.getenv("MA_DS_VOICE_EN", "qwen3-data-scientist-demo-en-en"),
        "it_voice": os.getenv("MA_IT_VOICE_EN", "qwen3-male-it-expert-en"),
    },
    "fr": {
        "label": "Fran\u00e7ais",
        "flag": "\U0001F1EB\U0001F1F7",
        "tts_language": "French",
        "ds_voice": os.getenv("MA_DS_VOICE_FR", "qwen3-french-woman-news-speaker-fr"),
        "it_voice": os.getenv("MA_IT_VOICE_FR", "qwen3-french-male-news-speaker-fr"),
    },
    "de": {
        "label": "Deutsch",
        "flag": "\U0001F1E9\U0001F1EA",
        "tts_language": "German",
        "ds_voice": os.getenv("MA_DS_VOICE_DE", "qwen3-german-female-newscaster-de"),
        "it_voice": os.getenv("MA_IT_VOICE_DE", "qwen3-german-male-newscaster-de"),
    },
    "es": {
        "label": "Espa\u00f1ol",
        "flag": "\U0001F1EA\U0001F1F8",
        "tts_language": "Spanish",
        "ds_voice": os.getenv("MA_DS_VOICE_ES", "qwen3-newscaster-female-spanish-es"),
        "it_voice": os.getenv("MA_IT_VOICE_ES", "qwen3-newscaster-male-spanish-es"),
    },
    "it": {
        "label": "Italiano",
        "flag": "\U0001F1EE\U0001F1F9",
        "tts_language": "Italian",
        "ds_voice": os.getenv("MA_DS_VOICE_IT", "qwen3-italian-newscaster-female-it"),
        "it_voice": os.getenv("MA_IT_VOICE_IT", "qwen3-italian-newscaster-male-it"),
    },
}


# ---------------------------------------------------------------------------
# Frame processors for SSE events (synced with TTS audio)
# ---------------------------------------------------------------------------

class TextEventBuffer:
    """Holds per-speaker SSE event groups until TTS starts speaking."""

    def __init__(self):
        self._groups: list[list[dict]] = []
        self._current: list[dict] | None = None

    def start_group(self, speaker_event: dict):
        self._current = [speaker_event]
        self._groups.append(self._current)

    def add_event(self, event: dict):
        if self._current is not None:
            self._current.append(event)

    def pop_group(self) -> list[dict]:
        if self._groups:
            return self._groups.pop(0)
        return []


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
            self._buffer.add_event({"event": "token", "text": self._text_acc})
            self._buffer.add_event({"event": "end"})
            self._text_acc = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            pass
        elif isinstance(frame, InterruptionFrame):
            self._flush_acc()
        elif isinstance(frame, VoiceSwitchFrame):
            self._text_acc = ""
            self._buffer.start_group({
                "event": "speaker",
                "speaker": frame.speaker,
                "voice": frame.voice,
            })
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._buffer.add_event({"event": "start"})
        elif isinstance(frame, TextFrame):
            self._text_acc += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
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
            group = self._buffer.pop_group()
            for event in group:
                await self._queue.put(event)
        elif isinstance(frame, InterruptionFrame):
            group = self._buffer.pop_group()
            if group:
                for event in group:
                    await self._queue.put(event)
                logger.info(f"[TTSSyncCapture] Flushed interrupted group ({len(group)} events)")
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


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

async def run_voice_agent(webrtc_connection: SmallWebRTCConnection):
    langgraph_url = os.getenv("LANGGRAPH_URL", "http://localhost:2024")
    text_queue: asyncio.Queue = asyncio.Queue()

    pc_id = webrtc_connection.pc_id
    lang_code = pending_languages.pop(pc_id, "en")
    preset = LANGUAGE_PRESETS.get(lang_code, LANGUAGE_PRESETS["en"])
    ds_voice = preset["ds_voice"]
    it_voice = preset["it_voice"]
    tts_language = preset["tts_language"]
    logger.info(
        f"[{pc_id}] Language={preset['label']} ({lang_code}) "
        f"ds_voice={ds_voice} it_voice={it_voice}"
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

    tts = Qwen3TTSService(
        server=TTS_SERVER,
        voice=ds_voice,
        language=tts_language,
    )

    lg_processor = LangGraphProcessor(
        langgraph_url=langgraph_url,
        graph_name="orchestrator",
        voice_map=VoiceMap(data_scientist=ds_voice, it=it_voice),
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

    voice_map = lg_processor._voice_map
    active_tasks[pc_id] = (task, text_queue, tts, voice_map)

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

            @conn.event_handler("closed")
            async def on_closed(c: SmallWebRTCConnection):
                pcs_map.pop(c.pc_id, None)
                pending_languages.pop(c.pc_id, None)

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

    _, text_queue, _, _ = active_tasks[pc_id]

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
            resp = await client.get(f"{TTS_SERVER}/v1/voices")
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch voices: {e}")
        return JSONResponse({"error": str(e)}, status_code=502)


@app.post("/api/voice")
async def set_agent_voice(req: Request):
    body = await req.json()
    pc_id = body.get("pc_id", "")
    agent = body.get("agent", "")
    voice = body.get("voice", "").strip()

    if agent not in ("data_scientist", "it") or not voice:
        return JSONResponse({"error": "Need agent and voice"}, status_code=400)

    if pc_id and pc_id in active_tasks:
        _, _, _, voice_map = active_tasks[pc_id]
        if agent == "data_scientist":
            voice_map.data_scientist = voice
        else:
            voice_map.it = voice
        logger.info(f"[{pc_id}] {agent} voice → {voice}")

    return JSONResponse({"status": "ok", "agent": agent, "voice": voice})


@app.get("/nvidia_icon.png")
async def nvidia_icon():
    return FileResponse("nvidia_icon.png", media_type="image/png")


@app.get("/api/languages")
async def list_languages():
    return [
        {"code": code, **{k: v for k, v in preset.items() if k != "tts_language"}}
        for code, preset in LANGUAGE_PRESETS.items()
    ]


@app.get("/health")
async def health():
    return {"status": "healthy", "mode": "multi_agent_voice"}


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
    --nv-ds: #76B900;
    --nv-it: #00B4D8;
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
  .voice-group label.ds { color: var(--nv-ds); }
  .voice-group label.it { color: var(--nv-it); }
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
  .msg.bot.data_scientist { background: var(--nv-surface2); border-left: 3px solid var(--nv-ds); }
  .msg.bot.it { background: var(--nv-surface2); border-left: 3px solid var(--nv-it); }
  .msg.bot.unknown { background: var(--nv-surface2); border-left: 3px solid var(--nv-border); }
  .msg.system { align-self: center; background: transparent; color: var(--nv-text-muted); font-size: 13px; padding: 8px 0; }

  .speaker-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }
  .speaker-label.data_scientist { color: var(--nv-ds); }
  .speaker-label.it { color: var(--nv-it); }

  /* ── Overlay ── */
  #start-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.92); display: flex; align-items: center; justify-content: center; z-index: 200; flex-direction: column; gap: 20px; }
  #start-overlay .nv-badge { margin-bottom: 8px; }
  #start-overlay h2 { color: var(--nv-text); font-size: 26px; font-weight: 600; letter-spacing: -0.3px; }
  #start-overlay p { color: var(--nv-text-dim); font-size: 14px; max-width: 420px; text-align: center; line-height: 1.6; }

  .lang-picker { display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; margin: 4px 0; }
  .lang-btn { padding: 10px 18px; border-radius: 8px; border: 2px solid var(--nv-border); background: var(--nv-surface); color: var(--nv-text-dim); cursor: pointer; font-family: 'Inter', sans-serif; font-size: 14px; font-weight: 500; transition: all 0.15s ease; display: flex; align-items: center; gap: 8px; }
  .lang-btn:hover { border-color: var(--nv-text-muted); color: var(--nv-text); }
  .lang-btn.selected { border-color: var(--nv-green); color: var(--nv-green); background: rgba(118,185,0,0.08); box-shadow: 0 0 12px var(--nv-green-glow); }
  .lang-btn .flag { font-size: 20px; }

  #start-overlay button#start-btn { padding: 16px 48px; font-size: 16px; border-radius: 6px; border: none; background: var(--nv-green); color: var(--nv-black); cursor: pointer; font-weight: 700; font-family: 'Inter', sans-serif; letter-spacing: 0.3px; transition: all 0.15s ease; }
  #start-overlay button#start-btn:hover { background: #8cd100; box-shadow: 0 0 20px var(--nv-green-glow); }
</style>
</head>
<body>
<div id="start-overlay">
  <div class="nv-badge">
    <img src="/nvidia_icon.png" alt="NVIDIA" style="height:64px;width:auto;">
  </div>
  <h2>Multi-Agent Voice Collaboration</h2>
  <p>Talk with """ + DS_NAME + """ (Data Scientist) and """ + IT_NAME + """ (IT Infrastructure) to plan your ML deployment project.</p>
  <p style="color:var(--nv-text);font-weight:500;margin-top:4px;">Choose your language</p>
  <div class="lang-picker" id="lang-picker"></div>
  <button id="start-btn">Start Conversation</button>
</div>
<div class="header">
  <div class="brand">
    <img class="nv-logo" src="/nvidia_icon.png" alt="NVIDIA">
    <h1>Multi-Agent Voice</h1>
  </div>
  <span id="status" class="status">Disconnected</span>
  <button id="mic-btn" title="Mute / Unmute">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
  </button>
  <span id="speaking">Speaking...</span>
  <div class="voice-row">
    <div class="voice-group">
      <label class="ds">""" + DS_NAME + """:</label>
      <select id="voice-ds" class="voice-select" disabled><option>Loading...</option></select>
    </div>
    <div class="voice-group">
      <label class="it">""" + IT_NAME + """:</label>
      <select id="voice-it" class="voice-select" disabled><option>Loading...</option></select>
    </div>
  </div>
  <audio id="audio" autoplay playsinline></audio>
</div>
<div id="stt-bar">
  <span class="label">Hearing:</span>
  <span id="stt-text" class="text"></span>
</div>
<div class="chat" id="chat">
  <div class="msg system">Speak to """ + DS_NAME + """ and """ + IT_NAME + """. They'll collaborate to help plan your ML project.</div>
</div>

<script>
const AGENT_NAMES = {
  'data_scientist': '""" + DS_NAME + """',
  'it': '""" + IT_NAME + """'
};
const DS_DEFAULT_VOICE = '""" + DS_VOICE + """';
const IT_DEFAULT_VOICE = '""" + IT_VOICE + """';

let pc = null, pcId = null, localStream = null, micMuted = false;
let eventSource = null, sttClearTimer = null;
let selectedLang = 'en';
let langPresets = {};

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
loadLanguages();

const statusEl = document.getElementById('status');
const chatEl = document.getElementById('chat');
const audioEl = document.getElementById('audio');
const micBtn = document.getElementById('mic-btn');
const sttTextEl = document.getElementById('stt-text');
const voiceDsEl = document.getElementById('voice-ds');
const voiceItEl = document.getElementById('voice-it');

/* ── Streaming text reveal state ── */
let currentSpeaker = '';
let revealQueue = [];
let revealTimer = null;
let currentBotMsg = null;
let currentContentEl = null;
let revealedText = '';
let pendingFullText = '';

const WORDS_PER_SECOND = 2.6;
const MIN_REVEAL_MS = 40;

function flushReveal() {
  if (revealTimer) { clearInterval(revealTimer); revealTimer = null; }
  if (currentContentEl && pendingFullText) {
    currentContentEl.textContent = pendingFullText;
    revealedText = pendingFullText;
    chatEl.scrollTop = chatEl.scrollHeight;
  }
}

function startReveal(fullText) {
  pendingFullText = fullText;
  revealedText = '';
  const words = fullText.split(/\\s+/);
  const estimatedAudioMs = (words.length / WORDS_PER_SECOND) * 1000;
  const intervalMs = Math.max(MIN_REVEAL_MS, estimatedAudioMs / words.length);
  let idx = 0;
  if (revealTimer) clearInterval(revealTimer);
  revealTimer = setInterval(() => {
    if (idx >= words.length) {
      clearInterval(revealTimer);
      revealTimer = null;
      return;
    }
    revealedText += (idx > 0 ? ' ' : '') + words[idx];
    idx++;
    if (currentContentEl) {
      currentContentEl.textContent = revealedText;
      chatEl.scrollTop = chatEl.scrollHeight;
    }
  }, intervalMs);
}

function createBotBubble(speaker) {
  flushReveal();
  currentBotMsg = document.createElement('div');
  currentBotMsg.className = 'msg bot ' + (speaker || 'unknown');
  if (speaker && AGENT_NAMES[speaker]) {
    const label = document.createElement('div');
    label.className = 'speaker-label ' + speaker;
    label.textContent = AGENT_NAMES[speaker];
    currentBotMsg.appendChild(label);
  }
  currentContentEl = document.createElement('span');
  currentContentEl.className = 'bot-content';
  currentBotMsg.appendChild(currentContentEl);
  pendingFullText = '';
  revealedText = '';
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
    [voiceDsEl, voiceItEl].forEach(sel => {
      sel.innerHTML = '';
      names.forEach(n => {
        const opt = document.createElement('option');
        opt.value = n; opt.textContent = n;
        sel.appendChild(opt);
      });
      sel.disabled = false;
    });
    voiceDsEl.value = DS_DEFAULT_VOICE;
    voiceItEl.value = IT_DEFAULT_VOICE;
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

voiceDsEl.addEventListener('change', () => setAgentVoice('data_scientist', voiceDsEl.value));
voiceItEl.addEventListener('change', () => setAgentVoice('it', voiceItEl.value));

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
    body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type, pc_id: pcId, language: selectedLang }),
  });
  const answer = await resp.json();
  pcId = answer.pc_id;
  await pc.setRemoteDescription(answer);

  /* Set the voice dropdowns to match the language preset */
  const preset = langPresets[selectedLang];
  if (preset) {
    voiceDsEl.value = preset.ds_voice;
    voiceItEl.value = preset.it_voice;
  }
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

    } else if (msg.event === 'start') {
      if (pendingUserText) {
        addMessage(pendingUserText, 'user');
        pendingUserText = '';
        sttTextEl.textContent = '';
        if (sttClearTimer) clearTimeout(sttClearTimer);
      }
      createBotBubble(currentSpeaker);
      chatEl.appendChild(currentBotMsg);
      chatEl.scrollTop = chatEl.scrollHeight;

    } else if (msg.event === 'token') {
      if (!currentBotMsg) {
        createBotBubble(currentSpeaker);
        chatEl.appendChild(currentBotMsg);
      }
      startReveal(msg.text);

    } else if (msg.event === 'end') {
      flushReveal();
      if (currentBotMsg && !pendingFullText && !revealedText) {
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
  loadVoices();
  connect();
});
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
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()

    logger.remove(0)
    if args.verbose >= 2:
        logger.add(sys.stderr, level="TRACE")
    elif args.verbose >= 1:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    logger.info("=" * 60)
    logger.info("  Multi-Agent Voice Pipeline (WebRTC)")
    logger.info("=" * 60)
    logger.info(f"  STT:  AssemblyAI ({os.getenv('ASSEMBLYAI_LANGUAGE', 'multi')})")
    logger.info(f"  LLM:  LangGraph orchestrator @ {os.getenv('LANGGRAPH_URL', 'http://localhost:2024')}")
    logger.info(f"  TTS:  Qwen3 @ {TTS_SERVER}")
    logger.info(f"  {DS_NAME} voice: {DS_VOICE}")
    logger.info(f"  {IT_NAME} voice: {IT_VOICE}")
    logger.info("=" * 60)
    logger.info(f"  Open http://localhost:{args.port} in your browser")
    logger.info("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port)
