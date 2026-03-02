# AssemblyAI Multilingual Support in Pipecat

This document explains **why** the built-in pipecat AssemblyAI integration doesn't support automatic multilingual detection, **what** we changed to make it work, and **how** to replicate the approach.

---

## The Problem

AssemblyAI's [v3 streaming WebSocket API](https://www.assemblyai.com/docs/speech-to-text/streaming) supports a `language` query parameter that can be set to `"multi"` for automatic language detection. When `language=multi` is used, AssemblyAI automatically detects the spoken language and transcribes accordingly — no need to specify the language in advance.

However, the built-in pipecat `AssemblyAISTTService` (as of pipecat 0.0.102) does **not** expose this capability. Here's why.

---

## What the Built-in Service Does

The built-in service lives at `pipecat.services.assemblyai.stt.AssemblyAISTTService`. Its constructor accepts:

```python
class AssemblyAISTTService(WebsocketSTTService):
    def __init__(
        self,
        *,
        api_key: str,
        language: Language = Language.EN,  # <-- pipecat Language enum, NOT a string
        api_endpoint_base_url: str = "wss://streaming.assemblyai.com/v3/ws",
        connection_params: AssemblyAIConnectionParams = AssemblyAIConnectionParams(),
        vad_force_turn_endpoint: bool = True,
        ...
    ):
```

**Issue 1: The `language` parameter uses pipecat's `Language` enum.**

Pipecat defines a `Language` enum with standard codes (`Language.EN`, `Language.ES`, `Language.FR`, etc.). This enum does **not** include `"multi"` — that's an AssemblyAI-specific value, not a standard language code.

**Issue 2: The `language` parameter is never sent to AssemblyAI.**

The WebSocket URL is built from `AssemblyAIConnectionParams`, not from the `language` parameter:

```python
def _build_ws_url(self) -> str:
    params = {}
    for k, v in self._connection_params.model_dump().items():
        if v is not None:
            params[k] = v
    query_string = urlencode(params)
    return f"{self._api_endpoint_base_url}?{query_string}"
```

The `language` constructor parameter is stored internally for pipecat metadata, but it's **never added to the query string** sent to AssemblyAI's WebSocket.

**Issue 3: `AssemblyAIConnectionParams` uses `speech_model`, not `language`.**

```python
class AssemblyAIConnectionParams(BaseModel):
    sample_rate: int = 16000
    encoding: Literal["pcm_s16le", "pcm_mulaw"] = "pcm_s16le"
    formatted_finals: bool = True
    end_of_turn_confidence_threshold: Optional[float] = None
    max_turn_silence: Optional[int] = None
    keyterms_prompt: Optional[List[str]] = None
    speech_model: Literal["universal-streaming-english", "universal-streaming-multilingual"] = (
        "universal-streaming-english"
    )
    # NOTE: no 'language' field!
```

The `speech_model` field can be set to `"universal-streaming-multilingual"`, which selects AssemblyAI's multilingual-capable model. But this is **not the same** as `language=multi`:

| Parameter | What it does |
|-----------|-------------|
| `speech_model=universal-streaming-multilingual` | Selects a model that *can* handle multiple languages |
| `language=multi` | Tells AssemblyAI to *auto-detect* the language on the fly |

You need **both** for true multilingual auto-detection. The built-in service only exposes the first.

---

## Our Solution: Custom STT Service

We wrote `AssemblyAIMultilingualSTTService` — a custom pipecat `STTService` that connects directly to AssemblyAI's v3 WebSocket API and passes `language=multi` in the query string.

**File:** `pipelines/assemblyasr/assemblyai_multilingual_stt.py`

### Key Differences from Built-in

| Aspect | Built-in `AssemblyAISTTService` | Our `AssemblyAIMultilingualSTTService` |
|--------|------|------|
| Base class | `WebsocketSTTService` | `STTService` |
| Language parameter | `Language` enum (no `"multi"`) | Plain `str` (accepts `"multi"`) |
| Language in URL | Not sent | Sent as `language=multi` query param |
| Connection config | `AssemblyAIConnectionParams` (pydantic) | Direct constructor args |
| VAD integration | Built-in `vad_force_turn_endpoint` | Manual turn detection params |
| WebSocket library | `websockets` (async) | `websockets` (async) |

### The Critical Change: Passing `language` to the WebSocket

The core of the fix is in `_build_ws_url()`:

```python
def _build_ws_url(self) -> str:
    params = {
        "sample_rate": self._sample_rate,
        "language": self._language,  # <-- This is the key line. "multi" goes here.
        "format_turns": str(self._format_turns).lower(),
        "end_of_turn_confidence_threshold": self._end_of_turn_confidence_threshold,
        "min_end_of_turn_silence_when_confident": self._min_end_of_turn_silence_when_confident,
        "max_turn_silence": self._max_turn_silence,
    }

    if self._keyterms_prompt:
        params["keyterms_prompt"] = json.dumps(self._keyterms_prompt)

    query_string = urlencode(params)
    return f"{self._api_endpoint_base_url}?{query_string}"
```

This produces a URL like:

```
wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&language=multi&format_turns=true&end_of_turn_confidence_threshold=0.7&min_end_of_turn_silence_when_confident=160&max_turn_silence=2400
```

The `language=multi` parameter is what enables automatic language detection.

---

## How to Build a Custom Service Like This

### Step 1: Subclass `STTService`

```python
from pipecat.services.stt_service import STTService

class MyCustomSTTService(STTService):
    def __init__(self, *, api_key: str, language: str = "multi", sample_rate: int = 16000, **kwargs):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._api_key = api_key
        self._language = language
        self._sample_rate = sample_rate
```

Use `STTService` (not `WebsocketSTTService`) for full control over the WebSocket connection.

### Step 2: Implement Lifecycle Methods

Pipecat calls these when the pipeline starts, stops, or is cancelled:

```python
async def start(self, frame: StartFrame):
    """Pipeline is starting. Open the WebSocket connection."""
    await super().start(frame)
    await self._connect()

async def stop(self, frame: EndFrame):
    """Pipeline is stopping gracefully. Close the connection."""
    await self._disconnect()
    await super().stop(frame)

async def cancel(self, frame: CancelFrame):
    """Pipeline is being cancelled. Close immediately."""
    await self._disconnect()
    await super().cancel(frame)
```

### Step 3: Implement the WebSocket Connection

Use the `websockets` library (async) to connect:

```python
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

async def _connect(self):
    ws_url = self._build_ws_url()  # URL with language=multi in query string
    self._websocket = await websocket_connect(
        ws_url,
        additional_headers={"Authorization": self._api_key},
    )
    self._receive_task = asyncio.create_task(self._receive_messages())
```

### Step 4: Build the URL with `language=multi`

```python
from urllib.parse import urlencode

def _build_ws_url(self) -> str:
    params = {
        "sample_rate": self._sample_rate,
        "language": self._language,  # "multi" for auto-detection
        "format_turns": "true",
        "end_of_turn_confidence_threshold": 0.7,
        "max_turn_silence": 2400,
    }
    query_string = urlencode(params)
    return f"wss://streaming.assemblyai.com/v3/ws?{query_string}"
```

### Step 5: Implement `run_stt()` — the Main Audio Processing Loop

This method is called by pipecat every time an audio chunk arrives:

```python
async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
    # Buffer audio into 50ms chunks (AssemblyAI requirement)
    self._audio_buffer.extend(audio)

    # Send complete chunks to AssemblyAI
    while len(self._audio_buffer) >= self._chunk_size_bytes:
        chunk = bytes(self._audio_buffer[:self._chunk_size_bytes])
        self._audio_buffer = self._audio_buffer[self._chunk_size_bytes:]
        await self._websocket.send(chunk)

    # Yield any transcription that arrived from the receive task
    if self._pending_frame:
        frame = self._pending_frame
        self._pending_frame = None
        yield frame
    else:
        yield None
```

### Step 6: Handle AssemblyAI Responses

A background task receives WebSocket messages and converts them to pipecat frames:

```python
async def _receive_messages(self):
    async for message in self._websocket:
        data = json.loads(message)
        msg_type = data.get("type")

        if msg_type == "Turn":
            transcript = data.get("transcript", "")
            end_of_turn = data.get("end_of_turn", False)
            turn_is_formatted = data.get("turn_is_formatted", False)

            if not transcript:
                continue

            if end_of_turn and turn_is_formatted:
                # Final transcription — user finished speaking
                self._pending_frame = TranscriptionFrame(
                    text=transcript, user_id="", timestamp=""
                )
            else:
                # Interim transcription — user still speaking
                self._pending_frame = InterimTranscriptionFrame(
                    text=transcript, user_id="", timestamp=""
                )

        elif msg_type == "Begin":
            # Session started
            pass

        elif msg_type == "Termination":
            # Session ended
            self._termination_event.set()

        elif msg_type == "Error":
            logger.error(f"AssemblyAI error: {data.get('error')}")
```

### Step 7: Graceful Disconnection

```python
async def _disconnect(self):
    if not self._connected:
        return

    # Flush remaining audio
    if self._audio_buffer and self._websocket.state is State.OPEN:
        await self._websocket.send(bytes(self._audio_buffer))
        self._audio_buffer.clear()

    # Tell AssemblyAI we're done
    await self._websocket.send(json.dumps({"type": "Terminate"}))

    # Wait for acknowledgment
    try:
        await asyncio.wait_for(self._termination_event.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        pass

    # Cleanup
    if self._receive_task:
        self._receive_task.cancel()
    await self._websocket.close()
```

---

## AssemblyAI v3 WebSocket Protocol Summary

### Connection

```
Client → AssemblyAI:  Open WebSocket to wss://streaming.assemblyai.com/v3/ws?language=multi&sample_rate=16000&...
                       Header: Authorization: <api_key>

AssemblyAI → Client:  {"type": "Begin", "id": "session-id", "expires_at": 1234567890}
```

### Streaming Audio

```
Client → AssemblyAI:  Raw PCM audio bytes (16-bit signed LE, mono, 16kHz)
                       Send in 50ms-1000ms chunks (50ms = 1600 bytes at 16kHz)

AssemblyAI → Client:  {"type": "Turn", "transcript": "Hello", "end_of_turn": false}
                       {"type": "Turn", "transcript": "Hello world", "end_of_turn": false}
                       {"type": "Turn", "transcript": "Hello world.", "end_of_turn": true, "turn_is_formatted": true}
```

### Termination

```
Client → AssemblyAI:  {"type": "Terminate"}
AssemblyAI → Client:  {"type": "Termination", "audio_duration_seconds": 12.5, "session_duration_seconds": 15.2}
```

### Key `Turn` Message Fields

| Field | Type | Description |
|-------|------|-------------|
| `transcript` | string | Current transcription text |
| `end_of_turn` | boolean | `true` when user finished speaking |
| `turn_is_formatted` | boolean | `true` when transcript has proper formatting (punctuation, capitalization) |
| `words` | array | Word-level details with timestamps and confidence |

### Language Auto-Detection with `language=multi`

When `language=multi` is set in the query string:
- AssemblyAI analyzes the audio and detects the spoken language automatically
- Works across all supported languages without configuration changes
- The detected language can vary turn-by-turn (e.g., user switches from English to Spanish mid-conversation)
- Slightly higher latency than single-language mode due to detection overhead

Supported languages for multi: English, Spanish, French, German, Italian, Portuguese, Dutch, Hindi, Japanese, Chinese, Finnish, Korean, Polish, Russian, Turkish, Ukrainian, Vietnamese.

---

## Usage

```python
# Create the service with multi-language auto-detection
stt = AssemblyAIMultilingualSTTService(
    api_key="your-assemblyai-api-key",
    language="multi",          # Auto-detect language
    sample_rate=16000,         # Must match your audio input
)

# Or with a specific language (skips auto-detection, lower latency)
stt = AssemblyAIMultilingualSTTService(
    api_key="your-assemblyai-api-key",
    language="en",             # English only
    sample_rate=16000,
)

# Wire into a pipecat pipeline
pipeline = Pipeline([
    transport.input(),
    stt,
    transport.output(),
])
```

Set via environment variable:

```bash
ASSEMBLYAI_API_KEY=your-key
ASSEMBLYAI_LANGUAGE=multi    # or "en", "es", "fr", etc.
```
