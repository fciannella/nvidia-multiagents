# Qwen3 TTS Streaming API

Streaming text-to-speech service powered by Qwen3-TTS. Audio begins streaming to the client within ~170ms, so playback can start before the full utterance is synthesized.

**Base URL:** `http://192.168.7.163:8100`

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/tts/stream` | Synthesize text to streaming WAV audio |
| `GET` | `/v1/voices` | List available voice profiles |
| `POST` | `/v1/voices` | Upload a new voice (multipart form) |
| `POST` | `/v1/voices/reload` | Re-fetch voices from the MongoDB database |
| `GET` | `/health` | Server readiness check |

---

## Synthesize speech

```
POST /v1/tts/stream
Content-Type: application/json
```

### Request body

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | *required* | Text to speak (1–5000 chars) |
| `language` | string | `"Auto"` | Language hint. `Auto`, `English`, `Italian`, `French`, `Chinese`, `Spanish`, `German`, `Japanese`, `Korean`, etc. |
| `voice` | string | `"default"` | Voice profile name (from `/v1/voices`) |
| `emit_every_frames` | int | `2` | Emit audio chunk every N codec frames. Lower = faster first audio, higher = fewer HTTP chunks. Range: 1–32. |
| `decode_window_frames` | int | `80` | Decoder window size in frames. Larger = better quality at edges, more decode latency. Range: 16–256. |

### Response

- **Content-Type:** `audio/wav`
- **Transfer-Encoding:** chunked
- **Format:** 24 kHz, 16-bit PCM, mono

The first bytes are a standard 44-byte WAV header (with data length set to a placeholder for streaming). PCM audio chunks follow immediately as they are generated.

### Example: curl (save to file)

```bash
curl -X POST http://192.168.7.163:8100/v1/tts/stream \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello, how are you?", "language": "English", "voice": "default"}' \
  -o output.wav
```

### Example: curl (stream to speakers on Linux)

```bash
curl -sN -X POST http://192.168.7.163:8100/v1/tts/stream \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello, how are you?", "language": "English", "voice": "default"}' \
  | aplay -q
```

### Example: Python (save to file)

```python
import requests

resp = requests.post(
    "http://192.168.7.163:8100/v1/tts/stream",
    json={"text": "Hello, how are you?", "language": "English", "voice": "default"},
    stream=True,
)
resp.raise_for_status()

with open("output.wav", "wb") as f:
    for chunk in resp.iter_content(chunk_size=4096):
        f.write(chunk)
```

### Example: Python (real-time streaming playback)

```python
import requests
import subprocess

resp = requests.post(
    "http://192.168.7.163:8100/v1/tts/stream",
    json={"text": "Hello, how are you?", "language": "English", "voice": "default"},
    stream=True,
)
resp.raise_for_status()

player = subprocess.Popen(["aplay", "-q"], stdin=subprocess.PIPE)
for chunk in resp.iter_content(chunk_size=4096):
    player.stdin.write(chunk)
    player.stdin.flush()
player.stdin.close()
player.wait()
```

### Example: JavaScript (Web Audio API streaming)

```javascript
const res = await fetch("/v1/tts/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "Hello!", language: "English", voice: "default" }),
});

const ctx = new AudioContext({ sampleRate: 24000 });
if (ctx.state === "suspended") await ctx.resume();
const reader = res.body.getReader();

let headerSkipped = false;
let leftover = new Uint8Array(0);
let scheduledTime = 0;

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  // Combine with any leftover bytes
  const combined = new Uint8Array(leftover.length + value.length);
  combined.set(leftover);
  combined.set(value, leftover.length);

  let offset = 0;
  if (!headerSkipped) {
    if (combined.length < 44) { leftover = combined; continue; }
    offset = 44;
    headerSkipped = true;
  }

  // Convert Int16 PCM to Float32 and schedule playback
  const pcmLen = combined.length - offset;
  const usable = pcmLen - (pcmLen % 2);
  if (usable > 0) {
    const pcm = combined.slice(offset, offset + usable);
    const int16 = new Int16Array(pcm.buffer, pcm.byteOffset, pcm.byteLength / 2);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

    const buf = ctx.createBuffer(1, float32.length, 24000);
    buf.copyToChannel(float32, 0);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    const startAt = Math.max(ctx.currentTime, scheduledTime);
    src.start(startAt);
    scheduledTime = startAt + buf.duration;
  }
  leftover = combined.slice(offset + usable);
}
```

---

## List voices

```
GET /v1/voices
```

### Response

```json
[
  {
    "name": "default",
    "has_transcript": true,
    "source": "local",
    "language": "",
    "actor_name": "default"
  },
  {
    "name": "franco-battiato-it",
    "has_transcript": true,
    "source": "mongodb",
    "language": "it-IT",
    "actor_name": "Franco Battiato"
  }
]
```

| Field | Description |
|-------|-------------|
| `name` | The identifier to pass as `voice` in `/v1/tts/stream` |
| `has_transcript` | `true` = ICL mode (reference audio + transcript), `false` = x-vector only (speaker embedding) |
| `source` | Where the voice was loaded from: `local` (filesystem), `mongodb`, or `upload` (runtime API) |
| `language` | Language code from the voice database (e.g. `it-IT`, `en-US`) |
| `actor_name` | Human-readable display name |

---

## Upload a voice

```
POST /v1/voices
Content-Type: multipart/form-data
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Voice profile name |
| `audio` | file | yes | Reference audio file (WAV, MP3, FLAC, OGG) |
| `transcript` | string | no | Text spoken in the reference audio. If provided, enables ICL (in-context learning) mode for higher voice fidelity. If omitted, uses x-vector speaker embedding only. |

### Example: curl

```bash
curl -X POST http://192.168.7.163:8100/v1/voices \
  -F "name=my-voice" \
  -F "transcript=This is what I said in the recording." \
  -F "audio=@reference.wav"
```

### Example: Python

```python
import requests

with open("reference.wav", "rb") as f:
    resp = requests.post(
        "http://192.168.7.163:8100/v1/voices",
        data={"name": "my-voice", "transcript": "This is what I said in the recording."},
        files={"audio": ("reference.wav", f, "audio/wav")},
    )
print(resp.json())
# {"status": "ok", "voice": "my-voice"}
```

After uploading, use `"voice": "my-voice"` in `/v1/tts/stream` requests.

> **Note:** Uploaded voices are stored in memory and lost when the server restarts. For persistent voices, add them to the MongoDB database or the voice directory on disk.

---

## Reload voices from MongoDB

```
POST /v1/voices/reload
```

Re-reads the `audio_prompts_a2flow` collection and loads any new voices that aren't already in memory. Existing voices are not replaced.

### Response

```json
{
  "status": "ok",
  "voices": ["default", "franco-battiato-it", "..."],
  "count": 13
}
```

---

## Health check

```
GET /health
```

### Response

```json
{
  "status": "ok",
  "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "voices": ["default", "franco-battiato-it", "..."],
  "gpu": "NVIDIA GB10"
}
```

---

## Audio format details

| Property | Value |
|----------|-------|
| Sample rate | 24,000 Hz |
| Bit depth | 16-bit signed integer (PCM) |
| Channels | 1 (mono) |
| Container | WAV (RIFF) |
| Streaming | Chunked transfer encoding; data size in WAV header is a placeholder (`0x7FFFFFFF`) |

To compute audio duration from byte count:

```
duration_seconds = (total_bytes - 44) / (24000 * 2)
```

---

## Latency characteristics

| Metric | Typical value |
|--------|---------------|
| WAV header arrival | < 10 ms |
| First audio chunk | 170–210 ms |
| Chunk interval | ~120 ms |
| Real-time factor | ~0.5x (generates 2s of audio per second of wall time) |

The server uses `torch.compile` with a warmup pass at startup, so the first request after a server restart incurs no extra compilation delay.

---

## Error handling

All errors return JSON with a `detail` field:

```json
{"detail": "Voice 'nonexistent' not found. Available: ['default', 'franco-battiato-it']"}
```

| Status | Meaning |
|--------|---------|
| 400 | Bad request (invalid audio file, etc.) |
| 404 | Voice not found |
| 503 | Model not loaded yet (server still starting) |
| 500 | Internal generation error |

---

## Concurrency

The server processes one TTS request at a time (GPU lock). Concurrent requests queue up and are served in order. The health and voice listing endpoints are always available regardless of GPU load.
