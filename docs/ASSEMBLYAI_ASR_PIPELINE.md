# AssemblyAI ASR Pipeline - Connection Guide

This document explains how to connect to the AssemblyAI ASR-only pipeline for real-time multilingual speech-to-text transcription.

## Overview

The AssemblyAI ASR pipeline provides **transcription-only** service with multilingual support:
- Real-time audio input via WebRTC
- Speech recognition using AssemblyAI's v3 streaming API
- **Automatic language detection** with `language="multi"`
- Transcription streaming back to client via RTVI protocol
- **No LLM** - no AI responses
- **No TTS** - no voice output

## Key Features

| Feature | Description |
|---------|-------------|
| **Multilingual** | Automatic language detection across 20+ languages |
| **Real-time** | Streaming transcription with low latency |
| **Turn Detection** | Intelligent end-of-turn detection |
| **Interim Results** | See transcription as you speak |
| **Formatting** | Automatic punctuation and capitalization |

## Quick Start

### 1. Start the Pipeline

**Using Docker Compose (recommended):**

```bash
# Make sure ASSEMBLYAI_API_KEY is set in your .env file
docker-compose up pipecat-assemblyai-asr
```

**Running directly:**

```bash
cd pipelines/assemblyasr
python pipeline_assembly_asr.py --port 7868 --language multi
```

### 2. Access the Web UI

Open your browser and navigate to:

```
http://localhost:7868
```

This loads the Pipecat prebuilt WebRTC UI where you can:
- Grant microphone access
- Start speaking in any supported language
- See transcriptions in real-time

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ASSEMBLYAI_API_KEY` | (required) | Your AssemblyAI API key |
| `ASSEMBLYAI_LANGUAGE` | `multi` | Language code or "multi" for auto-detect |
| `ASSEMBLYAI_SAMPLE_RATE` | `16000` | Audio sample rate in Hz |
| `ASSEMBLYAI_SPEECH_MODEL` | (auto) | Speech model (optional) |
| `ASSEMBLYAI_END_OF_TURN_CONFIDENCE` | `0.7` | End-of-turn confidence threshold |
| `ASSEMBLYAI_MAX_TURN_SILENCE` | `2400` | Max silence before turn ends (ms) |
| `ASSEMBLYAI_KEYTERMS` | (none) | Comma-separated terms to boost |

### Language Options

**Auto-detection (recommended):**
```bash
ASSEMBLYAI_LANGUAGE=multi
```

**Specific language:**
```bash
ASSEMBLYAI_LANGUAGE=en      # English
ASSEMBLYAI_LANGUAGE=es      # Spanish
ASSEMBLYAI_LANGUAGE=fr      # French
ASSEMBLYAI_LANGUAGE=de      # German
ASSEMBLYAI_LANGUAGE=it      # Italian
ASSEMBLYAI_LANGUAGE=pt      # Portuguese
ASSEMBLYAI_LANGUAGE=ja      # Japanese
ASSEMBLYAI_LANGUAGE=zh      # Chinese
ASSEMBLYAI_LANGUAGE=ko      # Korean
ASSEMBLYAI_LANGUAGE=ru      # Russian
```

### Supported Languages

The full list of supported languages:

| Code | Language |
|------|----------|
| `multi` | Auto-detect (Multi-language) |
| `en` | English |
| `en-US` | English (US) |
| `en-GB` | English (UK) |
| `en-AU` | English (Australia) |
| `es` | Spanish |
| `fr` | French |
| `de` | German |
| `it` | Italian |
| `pt` | Portuguese |
| `nl` | Dutch |
| `hi` | Hindi |
| `ja` | Japanese |
| `zh` | Chinese |
| `fi` | Finnish |
| `ko` | Korean |
| `pl` | Polish |
| `ru` | Russian |
| `tr` | Turkish |
| `uk` | Ukrainian |
| `vi` | Vietnamese |

## API Endpoints

### WebRTC Connection

**POST `/api/offer`**

Initiate WebRTC connection with SDP offer.

```javascript
const response = await fetch('http://localhost:7868/api/offer', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    sdp: offer.sdp,
    type: offer.type,
    language: 'multi'  // optional language override
  })
});

const answer = await response.json();
// Use answer.sdp and answer.type to complete WebRTC handshake
```

**PATCH `/api/offer`**

Add ICE candidates or renegotiate.

```javascript
await fetch('http://localhost:7868/api/offer', {
  method: 'PATCH',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    pc_id: connectionId,
    candidate: iceCandidate
  })
});
```

### Utility Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check, returns status and config |
| `/rtc-config` | GET | Get ICE server configuration |
| `/languages` | GET | List supported languages |
| `/models` | GET | List available speech models |
| `/start` | POST | Bootstrap endpoint for Pipecat UI |

## Connecting from a Custom Client

### JavaScript/TypeScript Example

```javascript
// 1. Get ICE configuration
const rtcConfig = await fetch('http://localhost:7868/rtc-config').then(r => r.json());

// 2. Create RTCPeerConnection
const pc = new RTCPeerConnection({ iceServers: rtcConfig.iceServers });

// 3. Get microphone access
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
stream.getTracks().forEach(track => pc.addTrack(track, stream));

// 4. Set up data channel for RTVI messages (transcriptions)
const dataChannel = pc.createDataChannel('rtvi');

dataChannel.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  if (message.type === 'user-transcription') {
    // Final transcription - end of turn detected
    console.log('Final:', message.data.text);
  } else if (message.type === 'user-interim-transcription') {
    // Interim (partial) transcription - still speaking
    console.log('Interim:', message.data.text);
  }
};

// 5. Create and send offer
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);

const response = await fetch('http://localhost:7868/api/offer', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    sdp: offer.sdp,
    type: offer.type,
    language: 'multi'  // Enable auto-detection
  })
});

const answer = await response.json();
await pc.setRemoteDescription(new RTCSessionDescription(answer));

// 6. Handle ICE candidates
pc.onicecandidate = async (event) => {
  if (event.candidate) {
    await fetch('http://localhost:7868/api/offer', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        pc_id: answer.pc_id,
        candidate: event.candidate
      })
    });
  }
};
```

### Python Example

```python
import asyncio
import json
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
import aiohttp

async def connect_to_assemblyai_asr():
    # Create peer connection
    pc = RTCPeerConnection()
    
    # Add audio track from microphone (or file)
    player = MediaPlayer('/dev/audio0')  # or use a file
    pc.addTrack(player.audio)
    
    # Create data channel for receiving transcriptions
    channel = pc.createDataChannel('rtvi')
    
    @channel.on('message')
    def on_message(message):
        data = json.loads(message)
        if data.get('type') == 'user-transcription':
            print(f"Final: {data['data']['text']}")
        elif data.get('type') == 'user-interim-transcription':
            print(f"Interim: {data['data']['text']}")
    
    # Create offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    
    # Send to server with language preference
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'http://localhost:7868/api/offer',
            json={
                'sdp': pc.localDescription.sdp, 
                'type': pc.localDescription.type,
                'language': 'multi'  # Auto-detect language
            }
        ) as resp:
            answer = await resp.json()
    
    # Set remote description
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=answer['sdp'], type=answer['type'])
    )
    
    # Keep connection alive
    await asyncio.sleep(300)  # 5 minutes
    await pc.close()

asyncio.run(connect_to_assemblyai_asr())
```

## RTVI Protocol Messages

The pipeline uses the RTVI (Real-Time Voice Interface) protocol for transcription events.

### Transcription Events

**Final Transcription (end of turn detected):**
```json
{
  "type": "user-transcription",
  "data": {
    "text": "Hello, how are you today?",
    "user_id": "",
    "timestamp": ""
  }
}
```

**Interim Transcription (still speaking):**
```json
{
  "type": "user-interim-transcription", 
  "data": {
    "text": "Hello, how are you",
    "user_id": "",
    "timestamp": ""
  }
}
```

### Bot Ready Event

When the pipeline is ready to receive audio:
```json
{
  "type": "bot-ready"
}
```

## Turn Detection

AssemblyAI uses intelligent turn detection to determine when a speaker has finished. You can tune this behavior:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `end_of_turn_confidence_threshold` | 0.7 | Confidence level required (0.0-1.0) |
| `min_end_of_turn_silence_when_confident` | 160ms | Silence duration when confident |
| `max_turn_silence` | 2400ms | Maximum silence before forcing end |

**Low latency (faster, may cut off):**
```bash
ASSEMBLYAI_END_OF_TURN_CONFIDENCE=0.5
ASSEMBLYAI_MAX_TURN_SILENCE=1500
```

**High accuracy (slower, fewer interruptions):**
```bash
ASSEMBLYAI_END_OF_TURN_CONFIDENCE=0.9
ASSEMBLYAI_MAX_TURN_SILENCE=3000
```

## Keyword Boosting

Boost recognition of specific terms (product names, jargon, etc.):

```bash
ASSEMBLYAI_KEYTERMS="Pipecat,AssemblyAI,FastCosyVoice,WebRTC"
```

## Docker Compose Configuration

The service is defined in `docker-compose.yml`:

```yaml
pipecat-assemblyai-asr:
  build:
    context: .
    dockerfile: Dockerfile
  container_name: pipecat-assemblyai-asr-server
  network_mode: host
  working_dir: /app/pipelines/assemblyasr
  command: >
    python pipeline_assembly_asr.py
    --host 0.0.0.0
    --port 7868
  environment:
    - PYTHONPATH=/app:/app/pipelines/assemblyasr
    - ASSEMBLYAI_API_KEY=${ASSEMBLYAI_API_KEY}
    - ASSEMBLYAI_LANGUAGE=${ASSEMBLYAI_LANGUAGE:-multi}
    - ASSEMBLYAI_SAMPLE_RATE=${ASSEMBLYAI_SAMPLE_RATE:-16000}
    - TURN_SERVER_URL=${TURN_SERVER_URL:-}
    - TURN_USERNAME=${TURN_USERNAME:-}
    - TURN_PASSWORD=${TURN_PASSWORD:-}
    - TWILIO_ACCOUNT_SID=${TWILIO_ACCOUNT_SID:-}
    - TWILIO_AUTH_TOKEN=${TWILIO_AUTH_TOKEN:-}
```

## Comparison: AssemblyAI vs OpenAI ASR

| Feature | AssemblyAI | OpenAI |
|---------|------------|--------|
| **Auto Language Detection** | Yes (`multi`) | No |
| **Languages** | 20+ | 50+ |
| **Interim Results** | Yes | Yes |
| **Turn Detection** | Built-in, configurable | VAD-based |
| **Keyword Boosting** | Yes | Via prompt |
| **Best For** | Multilingual, turn-based | English, high accuracy |

## Troubleshooting

### No Audio / Connection Failed

1. **Check ICE connectivity**: Ensure TURN servers are configured if behind NAT
   ```bash
   # Add to .env
   TWILIO_ACCOUNT_SID=your_sid
   TWILIO_AUTH_TOKEN=your_token
   ```

2. **Check browser permissions**: Microphone access must be granted

3. **Check network**: WebRTC requires UDP connectivity

### No Transcriptions

1. **Verify API key**: Check `ASSEMBLYAI_API_KEY` is set correctly
2. **Check logs**: Run with `-v` for verbose output
   ```bash
   python pipeline_assembly_asr.py --port 7868 -v
   ```

### Wrong Language Detected

1. **Use specific language**: If auto-detect fails, specify the language
   ```bash
   ASSEMBLYAI_LANGUAGE=en
   ```

2. **Use keyword boosting**: Help recognition with domain-specific terms

### Transcription Cuts Off Too Early

1. **Increase turn silence threshold**:
   ```bash
   ASSEMBLYAI_MAX_TURN_SILENCE=3000
   ```

2. **Increase confidence threshold**:
   ```bash
   ASSEMBLYAI_END_OF_TURN_CONFIDENCE=0.9
   ```

## Use Cases

- **Multilingual captioning** - Auto-detect language in real-time
- **International meetings** - Participants can speak any supported language
- **Voice notes/dictation** - Quick transcription in any language
- **Accessibility** - Real-time captions for hearing-impaired
- **Call center** - Transcribe customer calls in multiple languages

## Related Pipelines

| Pipeline | Port | Description |
|----------|------|-------------|
| `pipecat-assemblyai-asr` | 7868 | AssemblyAI ASR (this pipeline) |
| `pipecat-openai-asr` | 7867 | OpenAI ASR (English-focused) |
| `pipecat-assemblyai-echo` | 7870 | STT → TTS echo (no LLM) |
| `pipecat-assemblyai-voice-agent` | 7869 | Full voice agent (STT + LLM + TTS) |
| `pipecat-fastcosyvoice-agent` | 7860 | Full voice agent with NVCF TTS |

## API Reference: AssemblyAI v3 Streaming

The pipeline uses AssemblyAI's v3 streaming WebSocket API:

**Endpoint:** `wss://streaming.assemblyai.com/v3/ws`

**Query Parameters:**
- `sample_rate` - Audio sample rate (e.g., 16000)
- `language` - Language code or "multi"
- `format_turns` - Enable turn formatting
- `end_of_turn_confidence_threshold` - Confidence threshold
- `min_end_of_turn_silence_when_confident` - Silence duration (ms)
- `max_turn_silence` - Max silence (ms)
- `keyterms_prompt` - JSON array of terms to boost

**Message Types (Server → Client):**
- `Begin` - Session started
- `Turn` - Transcription (interim or final)
- `Termination` - Session ended
- `Error` - Error occurred

For more details, see [AssemblyAI Streaming Documentation](https://www.assemblyai.com/docs/speech-to-text/streaming).
