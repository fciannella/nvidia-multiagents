"""Generic voice bot with dual-thread architecture.

Pipeline: Mic → STT → GenericVoiceFrontend → TTS → Speaker

The GenericVoiceFrontend manages two LangGraph threads:
  - Main thread: full agent with tools (can be long-running)
  - Secondary thread: chitchat-only while main is busy

Requires: langgraph dev server running
    cd generic_agent && langgraph dev --no-browser --n-jobs-per-worker 10

Usage:
    python pipecat_voice_bot_generic.py
    python pipecat_voice_bot_generic.py --host 0.0.0.0 --port 7865
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
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from starlette.responses import StreamingResponse

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
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
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from assemblyai_multilingual_stt import AssemblyAIMultilingualSTTService
from qwen3_tts import Qwen3TTSService
from generic_voice_frontend import GenericVoiceFrontend

load_dotenv()

TTS_SERVER = os.getenv("TTS_SERVER", "http://192.168.7.163:8100")
TTS_VOICE = "hekate-en"
TTS_LANGUAGE = "English"


# ── SSE for chat transcript ──────────────────────────────────────────

sse_queues: Dict[str, asyncio.Queue] = {}


async def sse_generator(q: asyncio.Queue):
    try:
        while True:
            msg = await q.get()
            yield f"data: {json.dumps(msg)}\n\n"
    except asyncio.CancelledError:
        return


class TranscriptionCapture(FrameProcessor):
    """Send user transcription events (interim + final) to SSE."""

    def __init__(self, sse_q: asyncio.Queue, **kw):
        super().__init__(**kw)
        self._q = sse_q

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            await self._q.put({
                "event": "user",
                "text": frame.text.strip(),
            })
        elif isinstance(frame, InterimTranscriptionFrame) and frame.text.strip():
            await self._q.put({
                "event": "user_interim",
                "text": frame.text.strip(),
            })
        await self.push_frame(frame, direction)


# ── Pipecat pipeline ─────────────────────────────────────────────────

active_tasks: Dict[str, tuple] = {}
ice_servers = [IceServer(urls="stun:stun.l.google.com:19302")]
pcs_map: Dict[str, SmallWebRTCConnection] = {}

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

    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                confidence=0.5, start_secs=0.2, stop_secs=0.8, min_volume=0.5,
            )
        ),
    )

    pc_id = webrtc_connection.pc_id
    sse_q: asyncio.Queue = asyncio.Queue()
    sse_queues[pc_id] = sse_q

    stt = AssemblyAIMultilingualSTTService(
        language=os.getenv("ASSEMBLYAI_LANGUAGE", "multi"),
        sample_rate=16000,
    )

    tts = Qwen3TTSService(
        server=TTS_SERVER,
        voice=TTS_VOICE,
        language=TTS_LANGUAGE,
    )

    langgraph_url = os.getenv("LANGGRAPH_URL", "http://localhost:2024")

    frontend = GenericVoiceFrontend(
        server_url=langgraph_url,
        graph_name="agent",
        ui_queue=sse_q,
        tts=tts,
    )

    transcription_capture = TranscriptionCapture(sse_q)

    user_turn = UserTurnProcessor(
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy()],
            stop=[SpeechTimeoutUserTurnStopStrategy(timeout=1.5)],
        ),
    )

    pipeline = Pipeline([
        transport.input(),
        vad,
        stt,
        transcription_capture,
        user_turn,
        frontend,
        tts,
        transport.output(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_connected(t, client):
        logger.info(f"Client connected: {pc_id}")
        active_tasks[pc_id] = (task, sse_q, tts)

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(t, client):
        logger.info(f"Client disconnected: {pc_id}")
        active_tasks.pop(pc_id, None)
        sse_queues.pop(pc_id, None)
        pcs_map.pop(pc_id, None)
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
    active_tasks.pop(pc_id, None)


# ── HTTP endpoints ───────────────────────────────────────────────────

@app.get("/")
async def index():
    return HTMLResponse(HTML_PAGE)


@app.post("/api/offer")
async def offer(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    pc_id = body.get("pc_id")

    if pc_id and pc_id in pcs_map:
        conn = pcs_map[pc_id]
        await conn.renegotiate(
            sdp=body["sdp"], type=body["type"],
            restart_pc=body.get("restart_pc", False),
        )
    else:
        conn = SmallWebRTCConnection(ice_servers)
        await conn.initialize(sdp=body["sdp"], type=body["type"])

        @conn.event_handler("closed")
        async def handle_closed(wc: SmallWebRTCConnection):
            pcs_map.pop(wc.pc_id, None)

        background_tasks.add_task(run_bot, conn)

    answer = conn.get_answer()
    pcs_map[answer["pc_id"]] = conn
    return answer


@app.get("/api/events/{pc_id}")
async def events(pc_id: str):
    q = sse_queues.get(pc_id)
    if not q:
        return JSONResponse({"error": "not found"}, 404)
    return StreamingResponse(
        sse_generator(q),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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

    _, _, tts = active_tasks[pc_id]
    tts._voice = voice
    logger.info(f"[{pc_id}] Voice changed to: {voice}")
    return JSONResponse({"status": "ok", "voice": voice})


# ── Inline HTML/JS ───────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Generic Voice Agent</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0d1117; color: #e6edf3; height: 100vh; display: flex;
         flex-direction: column; }
  .header { background: #161b22; padding: 12px 24px; border-bottom: 1px solid #30363d;
            display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .header h1 { font-size: 18px; font-weight: 600; }
  .status { font-size: 12px; padding: 4px 10px; border-radius: 12px;
            background: #1f2937; color: #9ca3af; }
  .status.connected { background: #064e3b; color: #6ee7b7; }
  .status.busy { background: #78350f; color: #fbbf24; }
  .agent-badge { font-size: 11px; padding: 4px 10px; border-radius: 12px;
                 font-weight: 600; letter-spacing: 0.3px; text-transform: uppercase;
                 transition: all 0.3s ease; }
  .agent-badge.primary { background: #1e3a5f; color: #58a6ff; }
  .agent-badge.chitchat { background: #3b1f6e; color: #c084fc; }
  #mic-btn { width: 36px; height: 36px; border-radius: 50%; border: 2px solid #444;
             background: #161b22; color: #e6edf3; cursor: pointer; display: flex;
             align-items: center; justify-content: center; transition: all 0.2s;
             padding: 0; font-size: 0; }
  #mic-btn.active { border-color: #4ade80; color: #4ade80; }
  #mic-btn.muted { border-color: #ef4444; color: #ef4444; }
  #voice-select { padding: 4px 8px; border-radius: 8px; border: 1px solid #444;
                  background: #161b22; color: #e6edf3; font-size: 12px; outline: none;
                  max-width: 200px; }
  #chat { flex: 1; overflow-y: auto; padding: 24px; display: flex;
          flex-direction: column; gap: 12px; }
  .msg { max-width: 75%; padding: 12px 16px; border-radius: 16px;
         line-height: 1.5; font-size: 15px; word-wrap: break-word; }
  .msg.user { align-self: flex-end; background: #1f6feb; color: #fff;
              border-bottom-right-radius: 4px; }
  .msg.user.interim { opacity: 0.5; font-style: italic; }
  .msg.bot { align-self: flex-start; background: #21262d; color: #e6edf3;
             border-bottom-left-radius: 4px; }
  .msg .label { font-size: 11px; font-weight: 600; color: #8b949e;
                margin-bottom: 4px; text-transform: uppercase; }
  .msg.system { align-self: center; background: transparent; color: #6b7280;
                font-size: 12px; font-style: italic; padding: 4px 12px; }
  .controls { background: #161b22; padding: 16px 24px; border-top: 1px solid #30363d;
              display: flex; justify-content: center; }
  button.start { padding: 12px 32px; border-radius: 24px; border: none; cursor: pointer;
                 font-size: 15px; font-weight: 600; transition: all 0.2s;
                 background: #238636; color: #fff; }
  button.start:hover { background: #2ea043; }
  button.start:disabled { background: #1f2937; color: #484f58; cursor: not-allowed; }
  #start-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.85);
                   display: flex; align-items: center; justify-content: center;
                   z-index: 200; flex-direction: column; gap: 16px; }
  #start-overlay h2 { color: #58a6ff; font-size: 24px; }
  #start-overlay p { color: #8b949e; font-size: 14px; }
</style>
</head>
<body>
<div id="start-overlay">
  <h2>Generic Voice Agent</h2>
  <p>Dual-thread architecture: chat while the agent works on long tasks</p>
  <button class="start" id="startBtn" onclick="startSession()">Start Conversation</button>
</div>
<div class="header">
  <h1>Voice Agent</h1>
  <span class="status" id="statusBadge">Disconnected</span>
  <span class="agent-badge primary" id="agentBadge">Primary Agent</span>
  <button id="mic-btn" title="Mute / Unmute">
    <svg id="mic-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
    <svg id="mic-off-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:none"><line x1="1" y1="1" x2="23" y2="23"/><path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6"/><path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2c0 .76-.13 1.49-.35 2.17"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
  </button>
  <select id="voice-select" disabled><option>Loading voices...</option></select>
  <audio id="audio" autoplay playsinline></audio>
</div>
<div id="chat"></div>
<script>
const chatEl = document.getElementById('chat');
const audioEl = document.getElementById('audio');
const statusBadge = document.getElementById('statusBadge');
const startBtn = document.getElementById('startBtn');
const overlay = document.getElementById('start-overlay');
const micBtn = document.getElementById('mic-btn');
const micIcon = document.getElementById('mic-icon');
const micOffIcon = document.getElementById('mic-off-icon');
const voiceSelect = document.getElementById('voice-select');
const agentBadge = document.getElementById('agentBadge');

let pc, eventSource, pcId, localStream, micMuted = false;

function addSystemMsg(text) {
  const div = document.createElement('div');
  div.className = 'msg system';
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

micBtn.addEventListener('click', () => {
  if (!localStream) return;
  micMuted = !micMuted;
  localStream.getAudioTracks().forEach(t => { t.enabled = !micMuted; });
  micBtn.classList.toggle('muted', micMuted);
  micBtn.classList.toggle('active', !micMuted);
  micIcon.style.display = micMuted ? 'none' : 'block';
  micOffIcon.style.display = micMuted ? 'block' : 'none';
});

async function loadVoices() {
  try {
    const resp = await fetch('/api/voices');
    const data = await resp.json();
    const voices = data.voices || data;
    voiceSelect.innerHTML = '';
    voices.forEach(v => {
      const name = typeof v === 'string' ? v : (v.voice_id || v.name || v);
      const opt = document.createElement('option');
      opt.value = name; opt.textContent = name;
      voiceSelect.appendChild(opt);
    });
    voiceSelect.value = 'hekate-en';
    voiceSelect.disabled = false;
  } catch (e) { voiceSelect.innerHTML = '<option>Error loading</option>'; }
}

voiceSelect.addEventListener('change', async () => {
  if (!pcId) return;
  try {
    await fetch('/api/voice', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pc_id: pcId, voice: voiceSelect.value }),
    });
    addSystemMsg('Voice changed to: ' + voiceSelect.value);
  } catch (e) { addSystemMsg('Failed to change voice'); }
});

async function startSession() {
  startBtn.disabled = true;

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert('Microphone access requires HTTPS or localhost.\\nTry: chrome://flags/#unsafely-treat-insecure-origin-as-secure\\nand add this origin.');
    startBtn.disabled = false;
    return;
  }

  try {
    localStream = await navigator.mediaDevices.getUserMedia({
      audio: {echoCancellation: true, noiseSuppression: true, autoGainControl: true}
    });
  } catch (err) {
    alert('Microphone access denied: ' + err.message);
    startBtn.disabled = false;
    return;
  }

  pc = new RTCPeerConnection({iceServers: [{urls: 'stun:stun.l.google.com:19302'}]});
  pc.addTransceiver('audio', {direction: 'sendrecv'});
  localStream.getTracks().forEach(t => pc.addTrack(t, localStream));

  pc.ontrack = e => {
    audioEl.srcObject = e.streams[0];
    statusBadge.textContent = 'Connected';
    statusBadge.className = 'status connected';
    micBtn.classList.add('active');
  };

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  await new Promise(resolve => {
    const timeout = setTimeout(resolve, 2000);
    pc.onicegatheringstatechange = () => {
      if (pc.iceGatheringState === 'complete') { clearTimeout(timeout); resolve(); }
    };
  });

  const resp = await fetch('/api/offer', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({sdp: pc.localDescription.sdp, type: pc.localDescription.type, pc_id: pcId}),
  });
  const answer = await resp.json();
  pcId = answer.pc_id;
  await pc.setRemoteDescription(answer);

  overlay.style.display = 'none';
  loadVoices();
  connectSSE();
}

let currentBotMsg = null, currentContentEl = null;
let pendingUserDiv = null;
let currentAgent = 'primary';

function connectSSE(retries = 0) {
  eventSource = new EventSource('/api/events/' + encodeURIComponent(pcId));
  eventSource.onmessage = e => {
    const msg = JSON.parse(e.data);

    if (msg.event === 'user_interim') {
      if (!pendingUserDiv) {
        pendingUserDiv = document.createElement('div');
        pendingUserDiv.className = 'msg user interim';
        chatEl.appendChild(pendingUserDiv);
      }
      pendingUserDiv.textContent = msg.text;
      chatEl.scrollTop = chatEl.scrollHeight;

    } else if (msg.event === 'user') {
      if (pendingUserDiv) {
        pendingUserDiv.className = 'msg user';
        pendingUserDiv.textContent = msg.text;
        pendingUserDiv = null;
      } else {
        const div = document.createElement('div');
        div.className = 'msg user';
        div.textContent = msg.text;
        chatEl.appendChild(div);
      }
      chatEl.scrollTop = chatEl.scrollHeight;

    } else if (msg.event === 'start') {
      if (currentBotMsg) {
        if (currentContentEl && !currentContentEl.textContent) currentBotMsg.remove();
        currentBotMsg = null; currentContentEl = null;
      }
      const div = document.createElement('div');
      div.className = 'msg bot';
      const label = document.createElement('div');
      label.className = 'label';
      label.textContent = currentAgent === 'primary' ? 'Primary Agent' : 'Chitchat Agent';
      label.style.color = currentAgent === 'primary' ? '#58a6ff' : '#c084fc';
      const content = document.createElement('div');
      div.appendChild(label);
      div.appendChild(content);
      chatEl.appendChild(div);
      currentBotMsg = div;
      currentContentEl = content;
      chatEl.scrollTop = chatEl.scrollHeight;

    } else if (msg.event === 'token') {
      if (currentContentEl) {
        currentContentEl.textContent += msg.text;
        chatEl.scrollTop = chatEl.scrollHeight;
      }

    } else if (msg.event === 'end') {
      if (currentBotMsg && currentContentEl && !currentContentEl.textContent) {
        currentBotMsg.remove();
      }
      currentBotMsg = null; currentContentEl = null;

    } else if (msg.event === 'status') {
      statusBadge.textContent = msg.text || 'Connected';
      statusBadge.className = 'status ' + (msg.state || 'connected');

    } else if (msg.event === 'agent') {
      currentAgent = msg.agent;
      const isPrimary = msg.agent === 'primary';
      agentBadge.textContent = isPrimary ? 'Primary Agent' : 'Chitchat Agent';
      agentBadge.className = 'agent-badge ' + (isPrimary ? 'primary' : 'chitchat');
    }
  };
  eventSource.onerror = () => {
    eventSource.close(); eventSource = null;
    if (retries < 5) setTimeout(() => connectSSE(retries + 1), 500);
  };
}
</script>
</body>
</html>
"""


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic Voice Agent Pipeline")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7865)
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
    logger.info("  Generic Voice Agent — Dual-Thread Architecture")
    logger.info("=" * 60)
    logger.info(f"  STT:  AssemblyAI ({os.getenv('ASSEMBLYAI_LANGUAGE', 'multi')})")
    logger.info(f"  LLM:  LangGraph @ {os.getenv('LANGGRAPH_URL', 'http://localhost:2024')}")
    logger.info(f"  TTS:  Qwen3 @ {TTS_SERVER} (voice={TTS_VOICE})")
    logger.info("=" * 60)
    logger.info(f"  Open http://localhost:{args.port} in your browser")
    logger.info("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port)
