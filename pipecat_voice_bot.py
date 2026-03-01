"""Pipecat voice bot: text in, audio out via WebRTC.

Pipeline: User text -> LLM (Qwen3-8B via vLLM) -> TTS (Riva NIM) -> WebRTC audio

Open http://localhost:7860 in your browser. Type a question, hear the answer.

Usage:
    python pipecat_voice_bot.py
    python pipecat_voice_bot.py --host 0.0.0.0 --port 7860
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
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from loguru import logger
from starlette.responses import StreamingResponse

sys.path.insert(0, ".")
sys.path.insert(0, "pipecat/src")

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
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
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from riva_nim_realtime_tts import RivaNimRealtimeTTSService


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


SENTENCE_ENDINGS = frozenset(".!?;:")
MIN_FIRST_CHUNK_CHARS = 80


class SmartChunkAggregator(FrameProcessor):
    """Aggregates LLM tokens and sends text to TTS in at most 2 chunks.

    The first chunk is emitted once a sentence boundary is found AND the
    accumulated text is at least MIN_FIRST_CHUNK_CHARS long.  This avoids
    sending very short fragments whose prosody sounds wrong in isolation.
    Everything after the first chunk is buffered and sent as one piece when
    the LLM response ends.
    """

    def __init__(self, min_first_chars: int = MIN_FIRST_CHUNK_CHARS, **kwargs):
        super().__init__(**kwargs)
        self._min_first_chars = min_first_chars
        self._buffer = ""
        self._in_response = False
        self._first_chunk_sent = False
        self._last_sentence_end = -1

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer = ""
            self._in_response = True
            self._first_chunk_sent = False
            self._last_sentence_end = -1
            await self.push_frame(frame, direction)

        elif isinstance(frame, TextFrame) and self._in_response:
            self._buffer += frame.text

            if not self._first_chunk_sent:
                stripped = self._buffer.rstrip()
                if stripped and stripped[-1] in SENTENCE_ENDINGS:
                    self._last_sentence_end = len(self._buffer)

                if (
                    self._last_sentence_end > 0
                    and self._last_sentence_end >= self._min_first_chars
                ):
                    first = self._buffer[: self._last_sentence_end]
                    self._buffer = self._buffer[self._last_sentence_end :]
                    self._first_chunk_sent = True
                    self._last_sentence_end = -1
                    logger.info(
                        f"[AGGREGATOR] First chunk ({len(first)} chars): "
                        f"[{first.strip()[:80]}]"
                    )
                    await self.push_frame(TextFrame(text=first), direction)

        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._buffer.strip():
                logger.info(
                    f"[AGGREGATOR] Final chunk ({len(self._buffer)} chars): "
                    f"[{self._buffer.strip()[:80]}...]"
                )
                await self.push_frame(TextFrame(text=self._buffer), direction)
            self._buffer = ""
            self._in_response = False
            self._first_chunk_sent = False
            self._last_sentence_end = -1
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


class PipelineDebugLogger(FrameProcessor):
    """Logs key frames flowing through a specific pipeline point."""

    def __init__(self, tag: str, **kwargs):
        super().__init__(**kwargs)
        self._tag = tag
        self._audio_bytes = 0
        self._audio_chunks = 0
        self._current_ctx = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            self._audio_bytes = 0
            self._audio_chunks = 0
            self._current_ctx = (getattr(frame, "context_id", None) or "?")[:8]
            logger.info(f"[{self._tag}] TTSStarted ctx={self._current_ctx}")
        elif isinstance(frame, TTSAudioRawFrame):
            self._audio_chunks += 1
            self._audio_bytes += len(frame.audio)
        elif isinstance(frame, TTSStoppedFrame):
            ctx = (getattr(frame, "context_id", None) or "?")[:8]
            dur = self._audio_bytes / (24000 * 2) if self._audio_bytes else 0
            logger.info(
                f"[{self._tag}] TTSStopped ctx={ctx} "
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

LLM_BASE_URL = "http://192.168.7.203:8008/v1"
LLM_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
TTS_SERVER = "http://192.168.7.163:9000"
TTS_VOICE = "Magpie-Multilingual.EN-US.Aria"

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Your responses will be spoken aloud, "
    "so keep them concise (2-4 sentences), avoid bullet points, code blocks, "
    "special characters, and emojis. Speak naturally and conversationally."
)

# Active pipeline tasks by pc_id: (task, context, text_queue)
active_tasks: Dict[str, tuple[PipelineTask, LLMContext, asyncio.Queue]] = {}

ice_servers = [IceServer(urls="stun:stun.l.google.com:19302")]

app = FastAPI()


async def run_bot(webrtc_connection: SmallWebRTCConnection):
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=False,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
        ),
    )

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

    tts = RivaNimRealtimeTTSService(
        server=TTS_SERVER,
        voice_id=TTS_VOICE,
        sample_rate=24000,
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = LLMContext(messages)
    text_queue: asyncio.Queue = asyncio.Queue()

    _user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(),
        assistant_params=LLMAssistantAggregatorParams(),
    )

    text_capture = TextStreamCapture(text_queue)
    pre_transport_log = PipelineDebugLogger("PRE-TRANSPORT")

    pipeline = Pipeline([
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
        active_tasks[pc_id] = (task, context, text_queue)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected: {pc_id}")
        active_tasks.pop(pc_id, None)
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)

    active_tasks.pop(pc_id, None)


# -- WebRTC signaling --

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


# -- Text input endpoint --

@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    text = body.get("text", "").strip()
    pc_id = body.get("pc_id", "")

    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    if pc_id not in active_tasks:
        return JSONResponse({"error": "No active session. Connect via WebRTC first."}, status_code=404)

    task, context, _queue = active_tasks[pc_id]

    context.messages.append({"role": "user", "content": text})
    await task.queue_frames([LLMMessagesFrame(context.messages)])

    logger.info(f"[{pc_id}] User: {text}")
    return JSONResponse({"status": "ok", "text": text})


# -- SSE stream for LLM text tokens --

@app.get("/api/events/{pc_id}")
async def events(pc_id: str):
    if pc_id not in active_tasks:
        return JSONResponse({"error": "No active session"}, status_code=404)

    _, _, text_queue = active_tasks[pc_id]

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


# -- Chat UI --

CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Voice Bot</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, system-ui, sans-serif; background: #0f0f0f; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
  .header { padding: 16px 24px; border-bottom: 1px solid #222; display: flex; align-items: center; gap: 12px; }
  .header h1 { font-size: 18px; font-weight: 600; }
  .status { font-size: 12px; padding: 4px 10px; border-radius: 12px; background: #333; }
  .status.connected { background: #1a3a1a; color: #4ade80; }
  .status.connecting { background: #3a3a1a; color: #facc15; }
  .chat { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 80%; padding: 10px 16px; border-radius: 16px; font-size: 15px; line-height: 1.5; }
  .msg.user { align-self: flex-end; background: #2563eb; color: white; border-bottom-right-radius: 4px; }
  .msg.bot { align-self: flex-start; background: #222; border-bottom-left-radius: 4px; }
  .msg.system { align-self: center; background: transparent; color: #666; font-size: 13px; }
  .input-bar { padding: 16px 24px; border-top: 1px solid #222; display: flex; gap: 12px; }
  .input-bar input { flex: 1; padding: 12px 16px; border-radius: 12px; border: 1px solid #333; background: #1a1a1a; color: #e0e0e0; font-size: 15px; outline: none; }
  .input-bar input:focus { border-color: #2563eb; }
  .input-bar button { padding: 12px 24px; border-radius: 12px; border: none; background: #2563eb; color: white; font-size: 15px; cursor: pointer; font-weight: 500; }
  .input-bar button:hover { background: #1d4ed8; }
  .input-bar button:disabled { opacity: 0.5; cursor: not-allowed; }
  #debug-panel { position: fixed; bottom: 80px; right: 16px; width: 380px; max-height: 260px; overflow-y: auto; background: #111; border: 1px solid #333; border-radius: 8px; padding: 8px 12px; font-family: monospace; font-size: 11px; color: #8f8; z-index: 100; display: none; }
  #debug-panel.visible { display: block; }
  #debug-toggle { position: fixed; bottom: 84px; right: 16px; z-index: 101; background: #333; color: #ccc; border: none; padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 11px; }
</style>
</head>
<body>
<div class="header">
  <h1>Voice Bot</h1>
  <span id="status" class="status">Disconnected</span>
  <span id="speaking" style="display:none; color:#4ade80; font-size:12px;">Speaking...</span>
  <audio id="audio" autoplay playsinline></audio>
</div>
<div class="chat" id="chat">
  <div class="msg system">Type a message to hear the AI respond with voice.</div>
</div>
<div class="input-bar">
  <input id="input" type="text" placeholder="Ask something..." autocomplete="off" disabled />
  <button id="send" onclick="sendMessage()" disabled>Send</button>
</div>
<button id="debug-toggle" onclick="toggleDebug()">Debug</button>
<div id="debug-panel"></div>

<script>
let pc = null;
let pcId = null;

const statusEl = document.getElementById('status');
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const audioEl = document.getElementById('audio');

function addMessage(text, cls) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = 'status ' + (cls || '');
}

async function connect() {
  setStatus('Connecting...', 'connecting');

  pc = new RTCPeerConnection({
    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
  });

  pc.ontrack = (event) => {
    audioEl.srcObject = event.streams[0];
    audioEl.play().then(() => monitorAudio()).catch(() => {
      addMessage('Click anywhere on the page to enable audio playback.', 'system');
      document.addEventListener('click', () => { audioEl.play(); monitorAudio(); }, { once: true });
    });
  };

  pc.oniceconnectionstatechange = () => {
    if (pc.iceConnectionState === 'connected') {
      setStatus('Connected', 'connected');
      inputEl.disabled = false;
      sendBtn.disabled = false;
      inputEl.focus();
      connectSSE();
    } else if (pc.iceConnectionState === 'disconnected' || pc.iceConnectionState === 'failed') {
      setStatus('Disconnected');
      inputEl.disabled = true;
      sendBtn.disabled = true;
      if (eventSource) { eventSource.close(); eventSource = null; }
    }
  };

  // Need at least one transceiver for audio
  pc.addTransceiver('audio', { direction: 'recvonly' });

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  // Wait for ICE gathering (with timeout)
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

let eventSource = null;
let currentBotMsg = null;
let currentBotText = '';

function connectSSE(retries) {
  if (!pcId) return;
  if (eventSource) eventSource.close();
  retries = retries || 0;

  const url = '/api/events/' + encodeURIComponent(pcId);
  eventSource = new EventSource(url);

  eventSource.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (!msg.event) return;

    if (msg.event === 'start') {
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

// --- Debug infrastructure ---
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

// Monitor audio element events
['play', 'pause', 'waiting', 'stalled', 'ended', 'error', 'suspend'].forEach(evt => {
  audioEl.addEventListener(evt, () => dbg('audio.' + evt + ' paused=' + audioEl.paused + ' readyState=' + audioEl.readyState));
});

// Monitor audio activity
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

// Poll WebRTC inbound audio stats every 2s
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

connect();
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
    parser = argparse.ArgumentParser(description="Pipecat voice bot: text in, audio out")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
