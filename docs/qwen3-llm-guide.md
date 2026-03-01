# Using Qwen3-8B as a Local Filler LLM

## Overview

This project uses a locally hosted **Qwen/Qwen3-8B-FP8** model served via **vLLM** as a fast filler-phrase generator. When a user sends a query, the filler LLM generates a short, natural-sounding phrase (e.g., "That's a really interesting question, let me think about it") that the TTS speaks aloud while the main LLM (Grok Fast 4.1 via OpenRouter) prepares the real answer. This dramatically reduces perceived latency.

> **Current status**: The dynamic filler approach is preserved in `server.py` as `_handle_llm_tts_dynamic_filler()` but is currently inactive. The active path uses pre-synthesized fillers from MongoDB. The dynamic approach can be re-enabled by swapping the handler — see [Switching Between Approaches](#switching-between-approaches).

## Architecture

```
User Query
    │
    ├──► Local Qwen3-8B (filler)  ──► TTS ──► Audio (plays immediately)
    │         ~65ms TTFT
    │
    └──► OpenRouter Grok 4.1 (answer) ──► TTS ──► Audio (plays after filler)
              ~500-1000ms TTFT
```

Both LLMs run in parallel. The filler LLM streams tokens directly into the TTS synthesizer, so the user hears audio within ~300ms of pressing Send. The main LLM buffers its tokens until the filler finishes, then seamlessly continues into the answer.

## vLLM Server Setup

### Prerequisites

- NVIDIA GPU with sufficient VRAM (Qwen3-8B-FP8 requires ~8-10 GB)
- vLLM installed (`pip install vllm`)

### Starting the Server

```bash
vllm serve Qwen/Qwen3-8B-FP8 \
    --host 0.0.0.0 \
    --port 8001 \
    --max-model-len 40960 \
    --dtype auto
```

The model is downloaded from HuggingFace on first run. Subsequent starts use the cached weights.

### Verifying the Endpoint

```bash
# Check the model is loaded
curl http://192.168.7.203:8001/v1/models | python3 -m json.tool
```

Expected response:

```json
{
    "object": "list",
    "data": [
        {
            "id": "Qwen/Qwen3-8B-FP8",
            "object": "model",
            "owned_by": "vllm",
            "max_model_len": 40960
        }
    ]
}
```

### Testing with curl

```bash
curl -s http://192.168.7.203:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B-FP8",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "max_tokens": 50,
    "temperature": 0.8,
    "chat_template_kwargs": {"enable_thinking": false}
  }' | python3 -m json.tool
```

## Configuration

### Environment Variables

In `.env`:

```bash
FILLER_LLM_BASE_URL=http://192.168.7.203:8001/v1
FILLER_LLM_MODEL=Qwen/Qwen3-8B-FP8
```

These are read by `server.py` at startup:

```python
FILLER_LLM_MODEL = os.environ.get("FILLER_LLM_MODEL", "Qwen/Qwen3-8B-FP8")
FILLER_LLM_BASE_URL = os.environ.get("FILLER_LLM_BASE_URL", "http://localhost:8001/v1")
```

### Disabling Thinking Mode

Qwen3 is a "reasoning" model that by default emits `<think>...</think>` tokens before answering. This internal reasoning consumes tokens and adds latency. For filler generation, thinking must be disabled.

This is done via the `chat_template_kwargs` parameter in the vLLM API:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="Qwen/Qwen3-8B-FP8",
    base_url="http://192.168.7.203:8001/v1",
    api_key="none",           # vLLM doesn't require auth
    streaming=True,
    temperature=0.8,
    max_tokens=80,
    model_kwargs={
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False}
        },
    },
)
```

Key points:

- `api_key="none"` — vLLM's OpenAI-compatible endpoint doesn't require authentication
- `extra_body` is LangChain's way of passing non-standard parameters through to the API
- `chat_template_kwargs: {"enable_thinking": False}` tells the Qwen3 chat template to skip the reasoning phase
- Without this, Qwen3 will emit `<think>` tokens that waste the entire `max_tokens` budget on internal reasoning before producing any useful output

### System Prompt for Filler Generation

The prompt is designed to produce short, natural phrases:

```python
FILLER_SYSTEM_PROMPT = (
    "You are a voice assistant filler generator. Given the user's question, "
    "produce a single short filler phrase (8-15 words) that sounds natural, "
    "warm, and conversational — something a person would say while thinking "
    "before answering. Do NOT answer the question. Output ONLY the filler "
    "phrase, nothing else. Start with real words, NEVER with filler sounds "
    "like Hmm, Uh, Oh, Ah, Well, or similar hesitation sounds."
)
```

Important constraints in the prompt:

- **8-15 words**: Keeps the filler short (~2-4 seconds of speech). Longer fillers delay the real answer.
- **Do NOT answer the question**: Prevents the filler from leaking the actual answer.
- **No hesitation sounds**: "Hmm", "Uh", etc. sound unnatural when synthesized by TTS.
- **Start with real words**: Ensures TTS produces clean audio from the first syllable.

## Performance Characteristics

Measured on a local GPU (192.168.7.203):

| Metric | Value |
|--------|-------|
| Time to First Token (TTFT) | **65-109 ms** |
| Full filler generation | **500-800 ms** |
| Model memory (FP8) | ~8-10 GB VRAM |
| Context window | 40,960 tokens |

The first request after a cold start may be slightly slower (~1.3s TTFT) due to KV cache warmup. Subsequent requests consistently hit 65-109ms.

### Example Outputs

| User Query | Generated Filler |
|---|---|
| "Tell me a joke about cats" | "That's a really fun topic, give me just a second to think." |
| "What is the capital of Japan?" | "Great geography question, let me recall that for you." |
| "Write me a poem about the ocean" | "Creative writing, I love it — let me craft something nice." |
| "How do I sort a list in Python?" | "That's a great question — let me think about how to explain that." |
| "What are your thoughts on climate change?" | "That's a thoughtful question. Let me consider it carefully." |

## Integration with the TTS Pipeline

### How It Works (Dynamic Filler Mode)

The dynamic filler flow in `_handle_llm_tts_dynamic_filler()`:

1. User sends a query via WebSocket
2. Two async tasks start **in parallel**:
   - `stream_filler()`: Calls the local Qwen3-8B, streams tokens into a shared `TokenBuffer`, and sends `filler_token` messages to the frontend
   - `stream_main_llm()`: Calls the OpenRouter Grok API, buffers its tokens until the filler finishes, then flushes them into the same `TokenBuffer`
3. The `TokenBuffer` feeds into `synthesizer.stream_tts()`, which produces audio chunks
4. Audio chunks stream to the client via WebSocket

The `TokenBuffer` is an async queue that bridges the LLM streaming and the TTS:

```python
class TokenBuffer:
    async def put(self, token: str):  # LLM pushes tokens
    async def finish(self):            # signals end of text
    async def __anext__(self) -> str:  # TTS pulls tokens
```

### Compared to Pre-Synthesized Fillers (Current Active Mode)

The currently active `_handle_llm_tts()` uses pre-synthesized fillers from MongoDB:

- Filler audio is already generated (no TTS processing needed)
- Selected by semantic similarity using embeddings
- Audio starts within ~10-50ms (just sending pre-existing bytes)
- 142 fillers across 15 categories, rotated round-robin

The trade-off:

| | Pre-synthesized (active) | Dynamic Qwen3 (preserved) |
|---|---|---|
| Time to first audio | ~10-50ms | ~300-700ms |
| Variety | Fixed set (142 phrases) | Infinite |
| Context relevance | Category-level matching | Query-specific |
| Dependencies | MongoDB + embeddings | Local GPU + vLLM |
| Reliability | Very high | Depends on LLM availability |

## Switching Between Approaches

To re-enable the dynamic filler approach, edit `server.py`:

1. In the WebSocket handler dispatch (around line 766), change the `llm_tts` case:

```python
elif msg_type == "llm_tts":
    await _handle_llm_tts_dynamic_filler(ws, msg)  # dynamic fillers
    # await _handle_llm_tts(ws, msg)               # pre-synthesized fillers
```

2. Restart the container:

```bash
docker restart eartts
```

## Troubleshooting

### Model outputs `<think>` tags

The thinking mode is not disabled. Ensure `chat_template_kwargs: {"enable_thinking": False}` is passed in the `extra_body` parameter. This must go through LangChain's `model_kwargs` wrapping.

### High TTFT on first request

Normal — the first request warms up the KV cache. Subsequent requests will be 65-109ms.

### vLLM endpoint not reachable from Docker

If the EAR-TTS server runs in Docker and vLLM runs on the host, use the host's IP address (not `localhost`). Set `FILLER_LLM_BASE_URL=http://192.168.7.203:8001/v1` in `.env`.

### Filler leaks the actual answer

Refine the system prompt. Adding "Do NOT answer the question" and "Output ONLY the filler phrase" helps, but some factual questions may still leak. This is a known limitation of the approach.

### LangChain UserWarning about unsupported parameters

The `extra_body` parameter triggers a LangChain warning. This is suppressed in the code with:

```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    return ChatOpenAI(...)
```

The warning is harmless — `extra_body` is correctly passed through to the underlying OpenAI client.
