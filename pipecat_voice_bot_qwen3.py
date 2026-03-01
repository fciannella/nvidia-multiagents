"""Pipecat voice bot: voice + text in, audio out via WebRTC (Qwen3-TTS edition).

Pipeline: Mic -> STT (Riva Parakeet) -> LLM (Nemotron) -> TTS (Qwen3-TTS) -> Speaker
          Text input kept as fallback.

Open http://localhost:7861 in your browser.

Usage:
    python pipecat_voice_bot_qwen3.py
    python pipecat_voice_bot_qwen3.py --host 0.0.0.0 --port 7861
"""

import argparse
import asyncio
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from starlette.responses import StreamingResponse

sys.path.insert(0, ".")
sys.path.insert(0, "pipecat/src")

from pipecat.audio.vad.silero import SileroVADAnalyzer

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.nvidia.stt import NvidiaSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from hybrid_turn_strategy import HybridUserTurnStartStrategy  # noqa: F401 - keep available

from qwen3_tts import Qwen3TTSService


class TextStreamCapture(FrameProcessor):
    """Captures LLM text tokens and pushes them to an async queue for SSE."""

    def __init__(self, queue: asyncio.Queue, **kwargs):
        super().__init__(**kwargs)
        self._queue = queue

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            await self._queue.put({"event": "start"})
        elif isinstance(frame, TextFrame):
            await self._queue.put({"event": "token", "text": frame.text})
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._queue.put({"event": "end"})

        await self.push_frame(frame, direction)


class TranscriptionCapture(FrameProcessor):
    """Captures STT transcription frames and pushes them to an async queue for SSE."""

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


class PipelineDebugLogger(FrameProcessor):
    """Logs key frames flowing through a specific pipeline point."""

    def __init__(self, tag: str, **kwargs):
        super().__init__(**kwargs)
        self._tag = tag
        self._audio_bytes = 0
        self._audio_chunks = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            self._audio_bytes = 0
            self._audio_chunks = 0
            logger.info(f"[{self._tag}] TTSStarted")
        elif isinstance(frame, TTSAudioRawFrame):
            self._audio_chunks += 1
            self._audio_bytes += len(frame.audio)
        elif isinstance(frame, TTSStoppedFrame):
            dur = self._audio_bytes / (24000 * 2) if self._audio_bytes else 0
            logger.info(
                f"[{self._tag}] TTSStopped "
                f"chunks={self._audio_chunks} bytes={self._audio_bytes} "
                f"audio={dur:.2f}s"
            )
            self._audio_bytes = 0
            self._audio_chunks = 0
        elif isinstance(frame, LLMFullResponseStartFrame):
            logger.info(f"[{self._tag}] LLMResponseStart")
        elif isinstance(frame, LLMFullResponseEndFrame):
            logger.info(f"[{self._tag}] LLMResponseEnd")

        await self.push_frame(frame, direction)


load_dotenv(override=True)

STT_SERVER = "192.168.7.163:50052"
LLM_BASE_URL = "http://192.168.7.203:8008/v1"
LLM_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
TTS_SERVER = "http://192.168.7.163:8100"
TTS_VOICE = "hekate-en"
TTS_LANGUAGE = "English"

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Your responses will be spoken aloud, "
    "so keep them concise (2-4 sentences), avoid bullet points, code blocks, "
    "special characters, and emojis. Speak naturally and conversationally."
)

active_tasks: Dict[str, tuple[PipelineTask, LLMContext, asyncio.Queue, Qwen3TTSService]] = {}

ice_servers = [IceServer(urls="stun:stun.l.google.com:19302")]

app = FastAPI()


async def run_bot(webrtc_connection: SmallWebRTCConnection):
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
        ),
    )

    stt = NvidiaSTTService(
        api_key="not-used",
        server=STT_SERVER,
        use_ssl=False,
        model_function_map={
            "function_id": "",
            "model_name": "parakeet-1-1b-rnnt-multilingual",
        },
    )
    stt._stop_history_eou = 100
    stt._stop_threshold_eou = 0.2
    stt._stop_history = 200
    stt._stop_threshold = 0.8

    llm = OpenAILLMService(
        model=LLM_MODEL,
        api_key="none",
        base_url=LLM_BASE_URL,
        params=OpenAILLMService.InputParams(
            temperature=0.8,
            max_tokens=256,
            extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        ),
    )

    tts = Qwen3TTSService(
        server=TTS_SERVER,
        voice=TTS_VOICE,
        language=TTS_LANGUAGE,
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = LLMContext(messages)
    text_queue: asyncio.Queue = asyncio.Queue()

    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
        assistant_params=LLMAssistantAggregatorParams(),
    )

    stt_capture = TranscriptionCapture(text_queue)
    text_capture = TextStreamCapture(text_queue)
    pre_transport_log = PipelineDebugLogger("PRE-TRANSPORT")

    pipeline = Pipeline([
        transport.input(),
        stt,
        stt_capture,
        user_agg,
        llm,
        text_capture,
        tts,
        pre_transport_log,
        transport.output(),
        assistant_agg,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    pc_id = webrtc_connection.pc_id

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected: {pc_id}")
        active_tasks[pc_id] = (task, context, text_queue, tts)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected: {pc_id}")
        active_tasks.pop(pc_id, None)
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)

    active_tasks.pop(pc_id, None)


pcs_map: Dict[str, SmallWebRTCConnection] = {}


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        conn = pcs_map[pc_id]
        await conn.renegotiate(
            sdp=request["sdp"], type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        conn = SmallWebRTCConnection(ice_servers)
        await conn.initialize(sdp=request["sdp"], type=request["type"])

        @conn.event_handler("closed")
        async def handle_closed(wc: SmallWebRTCConnection):
            pcs_map.pop(wc.pc_id, None)

        background_tasks.add_task(run_bot, conn)

    answer = conn.get_answer()
    pcs_map[answer["pc_id"]] = conn
    return answer


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    text = body.get("text", "").strip()
    pc_id = body.get("pc_id", "")

    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    if pc_id not in active_tasks:
        return JSONResponse({"error": "No active session. Connect via WebRTC first."}, status_code=404)

    task, context, _queue, _tts = active_tasks[pc_id]

    context.messages.append({"role": "user", "content": text})
    await task.queue_frames([LLMMessagesFrame(context.messages)])

    logger.info(f"[{pc_id}] User: {text}")
    return JSONResponse({"status": "ok", "text": text})


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
async def set_voice(request: Request):
    body = await request.json()
    pc_id = body.get("pc_id", "")
    voice = body.get("voice", "").strip()

    if not voice:
        return JSONResponse({"error": "No voice provided"}, status_code=400)

    if pc_id not in active_tasks:
        return JSONResponse({"error": "No active session"}, status_code=404)

    _, _, _, tts = active_tasks[pc_id]
    tts._voice = voice
    logger.info(f"[{pc_id}] Voice changed to: {voice}")
    return JSONResponse({"status": "ok", "voice": voice})


@app.get("/api/events/{pc_id}")
async def events(pc_id: str):
    if pc_id not in active_tasks:
        return JSONResponse({"error": "No active session"}, status_code=404)

    _, _, text_queue, _ = active_tasks[pc_id]

    async def generate():
        import json
        while True:
            try:
                msg = await asyncio.wait_for(text_queue.get(), timeout=120)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("event") == "end":
                    continue
            except asyncio.TimeoutError:
                yield "data: {}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Voice Bot (Qwen3-TTS)</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, system-ui, sans-serif; background: #0f0f0f; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
  .header { padding: 12px 24px; border-bottom: 1px solid #222; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .header h1 { font-size: 18px; font-weight: 600; }
  .status { font-size: 12px; padding: 4px 10px; border-radius: 12px; background: #333; }
  .status.connected { background: #1a3a1a; color: #4ade80; }
  .status.connecting { background: #3a3a1a; color: #facc15; }
  #mic-btn { width: 36px; height: 36px; border-radius: 50%; border: 2px solid #444; background: #1a1a1a; color: #e0e0e0; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.2s; position: relative; }
  #mic-btn.active { border-color: #4ade80; color: #4ade80; }
  #mic-btn.active.listening { border-color: #7c3aed; box-shadow: 0 0 0 3px rgba(124,58,237,0.3); animation: pulse 1.5s ease-in-out infinite; }
  #mic-btn.muted { border-color: #ef4444; color: #ef4444; }
  @keyframes pulse { 0%,100% { box-shadow: 0 0 0 3px rgba(124,58,237,0.2); } 50% { box-shadow: 0 0 0 8px rgba(124,58,237,0.4); } }
  #stt-bar { padding: 6px 24px; background: #111; border-bottom: 1px solid #1a1a1a; font-size: 13px; color: #888; min-height: 30px; display: flex; align-items: center; gap: 8px; }
  #stt-bar .label { color: #555; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
  #stt-bar .text { color: #aaa; font-style: italic; }
  #stt-bar .text.final { color: #e0e0e0; font-style: normal; }
  .chat { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 80%; padding: 10px 16px; border-radius: 16px; font-size: 15px; line-height: 1.5; }
  .msg.user { align-self: flex-end; background: #7c3aed; color: white; border-bottom-right-radius: 4px; }
  .msg.bot { align-self: flex-start; background: #222; border-bottom-left-radius: 4px; }
  .msg.system { align-self: center; background: transparent; color: #666; font-size: 13px; }
  .input-bar { padding: 12px 24px; border-top: 1px solid #222; display: flex; gap: 12px; }
  .input-bar input { flex: 1; padding: 10px 16px; border-radius: 12px; border: 1px solid #333; background: #1a1a1a; color: #e0e0e0; font-size: 15px; outline: none; }
  .input-bar input:focus { border-color: #7c3aed; }
  .input-bar button { padding: 10px 20px; border-radius: 12px; border: none; background: #7c3aed; color: white; font-size: 14px; cursor: pointer; font-weight: 500; }
  .input-bar button:hover { background: #6d28d9; }
  .input-bar button:disabled { opacity: 0.5; cursor: not-allowed; }
  #voice-select { padding: 4px 8px; border-radius: 8px; border: 1px solid #444; background: #1a1a1a; color: #e0e0e0; font-size: 12px; outline: none; max-width: 180px; }
  #voice-select:focus { border-color: #7c3aed; }
  #debug-panel { position: fixed; bottom: 70px; right: 16px; width: 380px; max-height: 260px; overflow-y: auto; background: #111; border: 1px solid #333; border-radius: 8px; padding: 8px 12px; font-family: monospace; font-size: 11px; color: #8f8; z-index: 100; display: none; }
  #debug-panel.visible { display: block; }
  #debug-toggle { position: fixed; bottom: 74px; right: 16px; z-index: 101; background: #333; color: #ccc; border: none; padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 11px; }
  #start-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.85); display: flex; align-items: center; justify-content: center; z-index: 200; }
  #start-overlay button { padding: 20px 48px; font-size: 20px; border-radius: 16px; border: none; background: #7c3aed; color: white; cursor: pointer; font-weight: 600; }
  #start-overlay button:hover { background: #6d28d9; }
</style>
</head>
<body>
<div id="start-overlay">
  <button id="start-btn">Start Voice Bot</button>
</div>
<div class="header">
  <h1>Voice Bot <span style="color:#7c3aed">(Qwen3-TTS)</span></h1>
  <span id="status" class="status">Disconnected</span>
  <button id="mic-btn" title="Mute / Unmute microphone">
    <svg id="mic-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
    <svg id="mic-off-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:none"><line x1="1" y1="1" x2="23" y2="23"/><path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6"/><path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2c0 .76-.13 1.49-.35 2.17"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
  </button>
  <select id="voice-select" disabled><option>Loading voices...</option></select>
  <span id="speaking" style="display:none; color:#4ade80; font-size:12px;">Speaking...</span>
  <audio id="audio" autoplay playsinline></audio>
</div>
<div id="stt-bar">
  <span class="label">Hearing:</span>
  <span id="stt-text" class="text"></span>
</div>
<div class="chat" id="chat">
  <div class="msg system">Speak or type a message. The AI will respond with voice.</div>
</div>
<div class="input-bar">
  <input id="input" type="text" placeholder="Or type here..." autocomplete="off" disabled />
  <button id="send" onclick="sendMessage()" disabled>Send</button>
</div>
<button id="debug-toggle" onclick="toggleDebug()">Debug</button>
<div id="debug-panel"></div>

<script>
let pc = null;
let pcId = null;
let localStream = null;
let micMuted = false;

const statusEl = document.getElementById('status');
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const audioEl = document.getElementById('audio');
const micBtn = document.getElementById('mic-btn');
const micIcon = document.getElementById('mic-icon');
const micOffIcon = document.getElementById('mic-off-icon');
const sttTextEl = document.getElementById('stt-text');

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

micBtn.addEventListener('click', () => {
  if (!localStream) return;
  micMuted = !micMuted;
  localStream.getAudioTracks().forEach(t => { t.enabled = !micMuted; });
  micBtn.classList.toggle('muted', micMuted);
  micIcon.style.display = micMuted ? 'none' : 'block';
  micOffIcon.style.display = micMuted ? 'block' : 'none';
  dbg('Mic ' + (micMuted ? 'MUTED' : 'UNMUTED'));
});

async function connect() {
  setStatus('Connecting...', 'connecting');

  try {
    localStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    dbg('Mic access granted');
  } catch (err) {
    dbg('Mic access denied: ' + err);
    addMessage('Microphone access denied. You can still use text input.', 'system');
  }

  pc = new RTCPeerConnection({
    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
  });

  pc.ontrack = (event) => {
    dbg('ontrack fired, setting srcObject');
    audioEl.srcObject = event.streams[0];
    audioEl.play().then(() => {
      dbg('audioEl.play() resolved OK');
      monitorAudio();
    }).catch((err) => {
      dbg('audioEl.play() rejected: ' + err);
      addMessage('Click anywhere on the page to enable audio playback.', 'system');
      document.addEventListener('click', () => {
        audioEl.play().then(() => { dbg('audioEl.play() after click OK'); monitorAudio(); });
      }, { once: true });
    });
  };

  pc.oniceconnectionstatechange = () => {
    if (pc.iceConnectionState === 'connected') {
      setStatus('Connected', 'connected');
      inputEl.disabled = false;
      sendBtn.disabled = false;
      micBtn.classList.add('active');
      inputEl.focus();
      connectSSE();
    } else if (pc.iceConnectionState === 'disconnected' || pc.iceConnectionState === 'failed') {
      setStatus('Disconnected');
      inputEl.disabled = true;
      sendBtn.disabled = true;
      micBtn.classList.remove('active', 'listening');
      if (eventSource) { eventSource.close(); eventSource = null; }
    }
  };

  if (localStream) {
    localStream.getAudioTracks().forEach(track => {
      pc.addTrack(track, localStream);
    });
  } else {
    pc.addTransceiver('audio', { direction: 'recvonly' });
  }

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  await new Promise((resolve) => {
    if (pc.iceGatheringState === 'complete') return resolve();
    const timeout = setTimeout(resolve, 3000);
    pc.onicegatheringstatechange = () => {
      if (pc.iceGatheringState === 'complete') {
        clearTimeout(timeout);
        resolve();
      }
    };
  });

  const resp = await fetch('/api/offer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      sdp: pc.localDescription.sdp,
      type: pc.localDescription.type,
      pc_id: pcId,
    }),
  });

  const answer = await resp.json();
  pcId = answer.pc_id;
  await pc.setRemoteDescription(answer);
}

const voiceSelect = document.getElementById('voice-select');

async function loadVoices() {
  try {
    const resp = await fetch('/api/voices');
    const data = await resp.json();
    const voices = data.voices || data;
    voiceSelect.innerHTML = '';
    voices.forEach(v => {
      const name = typeof v === 'string' ? v : (v.voice_id || v.name || v);
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      voiceSelect.appendChild(opt);
    });
    voiceSelect.value = 'hekate-en';
    voiceSelect.disabled = false;
  } catch (e) {
    voiceSelect.innerHTML = '<option>Error loading</option>';
  }
}

voiceSelect.addEventListener('change', async () => {
  if (!pcId) return;
  const voice = voiceSelect.value;
  try {
    await fetch('/api/voice', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pc_id: pcId, voice }),
    });
    addMessage('Voice changed to: ' + voice, 'system');
  } catch (e) {
    addMessage('Failed to change voice: ' + e.message, 'system');
  }
});

let eventSource = null;
let currentBotMsg = null;
let currentBotText = '';
let sttClearTimer = null;

function connectSSE(retries) {
  if (!pcId) return;
  if (eventSource) eventSource.close();
  retries = retries || 0;

  const url = '/api/events/' + encodeURIComponent(pcId);
  eventSource = new EventSource(url);

  let pendingUserChunks = [];
  let runningUserText = '';

  eventSource.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (!msg.event) return;

    if (msg.event === 'stt_interim') {
      sttTextEl.textContent = runningUserText + msg.text;
      sttTextEl.className = 'text';
      micBtn.classList.add('listening');
      if (sttClearTimer) clearTimeout(sttClearTimer);
    } else if (msg.event === 'stt_final') {
      pendingUserChunks.push(msg.text);
      runningUserText = pendingUserChunks.join(' ');
      sttTextEl.textContent = runningUserText;
      sttTextEl.className = 'text final';
      micBtn.classList.remove('listening');
      sttClearTimer = setTimeout(() => { sttTextEl.textContent = ''; }, 3000);
    } else if (msg.event === 'start') {
      if (pendingUserChunks.length > 0) {
        addMessage(pendingUserChunks.join(' '), 'user');
        pendingUserChunks = [];
        runningUserText = '';
        sttTextEl.textContent = '';
        if (sttClearTimer) clearTimeout(sttClearTimer);
      }
      currentBotText = '';
      currentBotMsg = document.createElement('div');
      currentBotMsg.className = 'msg bot';
      currentBotMsg.textContent = '';
      chatEl.appendChild(currentBotMsg);
    } else if (msg.event === 'token' && currentBotMsg) {
      currentBotText += msg.text;
      currentBotMsg.textContent = currentBotText;
      chatEl.scrollTop = chatEl.scrollHeight;
    } else if (msg.event === 'end') {
      currentBotMsg = null;
      currentBotText = '';
    }
  };

  eventSource.onerror = () => {
    eventSource.close();
    eventSource = null;
    if (retries < 5) {
      setTimeout(() => connectSSE(retries + 1), 500);
    }
  };
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || !pcId) return;

  addMessage(text, 'user');
  inputEl.value = '';
  inputEl.focus();

  try {
    await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, pc_id: pcId }),
    });
  } catch (e) {
    addMessage('Failed to send: ' + e.message, 'system');
  }
}

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

const debugPanel = document.getElementById('debug-panel');
const debugLines = [];
function dbg(msg) {
  const ts = new Date().toISOString().slice(11, 23);
  const line = ts + ' ' + msg;
  debugLines.push(line);
  if (debugLines.length > 200) debugLines.shift();
  debugPanel.textContent = debugLines.join('\\n');
  debugPanel.scrollTop = debugPanel.scrollHeight;
  console.log('[DBG] ' + line);
}
function toggleDebug() {
  debugPanel.classList.toggle('visible');
}

['play', 'pause', 'waiting', 'stalled', 'ended', 'error', 'suspend'].forEach(evt => {
  audioEl.addEventListener(evt, () => dbg('audio.' + evt + ' paused=' + audioEl.paused + ' readyState=' + audioEl.readyState));
});

function monitorAudio() {
  if (!audioEl.srcObject) return;
  const ctx = new AudioContext();
  const src = ctx.createMediaStreamSource(audioEl.srcObject);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 256;
  src.connect(analyser);
  const data = new Uint8Array(analyser.frequencyBinCount);
  const speakEl = document.getElementById('speaking');
  let wasSpeaking = false;
  (function poll() {
    analyser.getByteFrequencyData(data);
    const vol = data.reduce((a,b) => a+b, 0) / data.length;
    const speaking = vol > 2;
    speakEl.style.display = speaking ? 'inline' : 'none';
    if (speaking && !wasSpeaking) dbg('SPEECH_START vol=' + vol.toFixed(1));
    if (!speaking && wasSpeaking) dbg('SPEECH_STOP vol=' + vol.toFixed(1));
    wasSpeaking = speaking;
    requestAnimationFrame(poll);
  })();
}

let prevStats = { bytesReceived: 0, packetsReceived: 0, packetsLost: 0, ts: Date.now() };
async function pollStats() {
  if (!pc) return;
  try {
    const stats = await pc.getStats();
    stats.forEach(report => {
      if (report.type === 'inbound-rtp' && report.kind === 'audio') {
        const dt = (Date.now() - prevStats.ts) / 1000;
        const dBytes = report.bytesReceived - prevStats.bytesReceived;
        const dPkts = report.packetsReceived - prevStats.packetsReceived;
        const dLost = report.packetsLost - prevStats.packetsLost;
        dbg(
          'RTP in: ' + (dBytes/1024).toFixed(1) + 'kB ' +
          dPkts + 'pkts ' +
          (dLost > 0 ? 'LOST=' + dLost + ' ' : '') +
          'jitter=' + (report.jitter*1000).toFixed(1) + 'ms ' +
          'total=' + (report.bytesReceived/1024).toFixed(0) + 'kB'
        );
        prevStats = { bytesReceived: report.bytesReceived, packetsReceived: report.packetsReceived, packetsLost: report.packetsLost, ts: Date.now() };
      }
    });
  } catch(e) {}
}
setInterval(pollStats, 2000);

document.getElementById('start-btn').addEventListener('click', () => {
  document.getElementById('start-overlay').style.display = 'none';
  loadVoices();
  connect();
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return CHAT_HTML


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat voice bot (Qwen3-TTS): text in, audio out")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
