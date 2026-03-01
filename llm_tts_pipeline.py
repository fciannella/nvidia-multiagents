"""Real-time LLM -> TTS pipeline.

Streams a question to a local Qwen3-8B LLM (via vLLM OpenAI-compatible API),
pipes each token into the Riva NIM realtime TTS WebSocket, and plays the
synthesized audio as it arrives.

The full chain is:
  User question -> LLM (streaming) -> TTS WebSocket (token-by-token) -> Audio playback

Usage:
    python llm_tts_pipeline.py "What is the theory of relativity?"
    python llm_tts_pipeline.py "Tell me a joke" --voice Magpie-Multilingual.EN-US.Ray
    python llm_tts_pipeline.py --dry-run "Hello world, this is a test."
"""

import argparse
import asyncio
import base64
import json
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd
import websockets
from openai import AsyncOpenAI

# -- Server configuration --
LLM_BASE_URL = "http://192.168.7.203:8001/v1"
LLM_MODEL = "Qwen/Qwen3-8B-FP8"
TTS_SERVER = "192.168.7.163:9000"
TTS_VOICE = "Magpie-Multilingual.EN-US.Aria"
SAMPLE_RATE = 22050

SYSTEM_PROMPT = "You are a helpful, concise assistant. Answer in 2-4 sentences."


@dataclass
class PipelineStats:
    pipeline_start: float = 0.0

    # LLM timing
    llm_request_sent: float = 0.0
    llm_first_token: float = 0.0
    llm_last_token: float = 0.0
    llm_tokens: int = 0
    llm_text: str = ""

    # TTS timing
    tts_connected: float = 0.0
    tts_session_ready: float = 0.0
    tts_first_commit: float = 0.0
    tts_first_audio: float = 0.0
    tts_last_audio: float = 0.0
    tts_synthesis_done: float = 0.0
    tts_commits: int = 0
    tts_audio_chunks: int = 0
    tts_audio_bytes: int = 0

    chunk_times: list[float] = field(default_factory=list)

    @property
    def llm_ttft_ms(self) -> float:
        return (self.llm_first_token - self.llm_request_sent) * 1000

    @property
    def llm_total_ms(self) -> float:
        return (self.llm_last_token - self.llm_request_sent) * 1000

    @property
    def llm_tokens_per_sec(self) -> float:
        dur = self.llm_last_token - self.llm_first_token
        return self.llm_tokens / dur if dur > 0 else 0

    @property
    def e2e_first_audio_ms(self) -> float:
        """End-to-end: from pipeline start to first audio out."""
        return (self.tts_first_audio - self.pipeline_start) * 1000

    @property
    def e2e_total_ms(self) -> float:
        end = self.tts_synthesis_done or self.tts_last_audio
        return (end - self.pipeline_start) * 1000

    @property
    def audio_duration_s(self) -> float:
        return (self.tts_audio_bytes // 2) / SAMPLE_RATE

    def report(self, question: str, voice: str):
        print("\n" + "=" * 70)
        print("  LLM -> TTS REALTIME PIPELINE STATS")
        print("=" * 70)
        print(f"  Question:         {question[:65]}{'...' if len(question) > 65 else ''}")
        print(f"  Voice:            {voice}")
        print(f"  LLM answer:       {self.llm_text[:65]}{'...' if len(self.llm_text) > 65 else ''}")
        print("-" * 70)
        print("  LLM")
        print(f"    TTFT:           {self.llm_ttft_ms:>10.1f} ms")
        print(f"    Total gen:      {self.llm_total_ms:>10.1f} ms")
        print(f"    Tokens:         {self.llm_tokens:>10d}")
        print(f"    Throughput:     {self.llm_tokens_per_sec:>10.1f} tok/s")
        print("-" * 70)
        print("  TTS")
        tts_ttfat = (self.tts_first_audio - self.tts_first_commit) * 1000 if self.tts_first_commit else 0
        print(f"    TTFAT (commit): {tts_ttfat:>10.1f} ms")
        print(f"    Commits:        {self.tts_commits:>10d}")
        print(f"    Audio chunks:   {self.tts_audio_chunks:>10d}")
        print(f"    Audio bytes:    {self.tts_audio_bytes:>10,d}")
        print(f"    Audio duration: {self.audio_duration_s:>10.2f} s")
        print("-" * 70)
        print("  END-TO-END")
        print(f"    First audio:    {self.e2e_first_audio_ms:>10.1f} ms")
        print(f"    Total:          {self.e2e_total_ms:>10.1f} ms")
        print("=" * 70)


def make_event(event_type: str, **kwargs) -> str:
    event = {"event_id": f"event_{uuid.uuid4().hex[:8]}", "type": event_type, **kwargs}
    return json.dumps(event)


async def stream_llm(
    question: str,
    llm_base_url: str,
    llm_model: str,
    stats: PipelineStats,
    system_prompt: str = SYSTEM_PROMPT,
):
    """Yield tokens from the LLM as they arrive."""
    client = AsyncOpenAI(base_url=llm_base_url, api_key="none")

    stats.llm_request_sent = time.perf_counter()

    stream = await client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        max_tokens=256,
        temperature=0.8,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            token = delta.content
            now = time.perf_counter()

            if stats.llm_tokens == 0:
                stats.llm_first_token = now

            stats.llm_tokens += 1
            stats.llm_last_token = now
            stats.llm_text += token

            yield token


async def stream_llm_dry_run(text: str, stats: PipelineStats):
    """Simulate LLM output from canned text (for testing without an LLM)."""
    words = text.split(" ")
    stats.llm_request_sent = time.perf_counter()

    for i, word in enumerate(words):
        token = word + (" " if i < len(words) - 1 else "")
        now = time.perf_counter()

        if stats.llm_tokens == 0:
            stats.llm_first_token = now

        stats.llm_tokens += 1
        stats.llm_last_token = now
        stats.llm_text += token

        yield token
        await asyncio.sleep(0.03)  # simulate ~30ms/token LLM


async def run_pipeline(
    question: str,
    tts_server: str,
    tts_voice: str,
    sample_rate: int,
    llm_base_url: str,
    llm_model: str,
    dry_run: bool = False,
    system_prompt: str = SYSTEM_PROMPT,
):
    stats = PipelineStats()
    stats.pipeline_start = time.perf_counter()

    audio_stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="int16", blocksize=1024)
    audio_stream.start()

    ws_url = f"ws://{tts_server}/v1/realtime?intent=synthesize"
    print(f"  Connecting to TTS: {ws_url}")

    leftover = b""

    async with websockets.connect(ws_url, max_size=16 * 1024 * 1024) as ws:
        stats.tts_connected = time.perf_counter()

        # Wait for conversation.created
        msg = json.loads(await ws.recv())
        assert msg["type"] == "conversation.created", f"Unexpected: {msg}"

        # Configure TTS session
        await ws.send(make_event(
            "synthesize_session.update",
            session={
                "input_text_synthesis": {
                    "language_code": "en-US",
                    "voice_name": tts_voice,
                },
                "output_audio_params": {
                    "sample_rate_hz": sample_rate,
                    "num_channels": 1,
                    "audio_format": "LINEAR_PCM",
                },
            },
        ))
        msg = json.loads(await ws.recv())
        assert msg["type"] == "synthesize_session.updated", f"Unexpected: {msg}"
        stats.tts_session_ready = time.perf_counter()
        print(f"  TTS session ready ({(stats.tts_session_ready - stats.tts_connected)*1000:.0f}ms)")

        # -- Sender: LLM tokens -> TTS --
        async def send_llm_to_tts():
            if dry_run:
                token_source = stream_llm_dry_run(question, stats)
            else:
                print(f"  Sending question to LLM: {question}")
                token_source = stream_llm(question, llm_base_url, llm_model, stats, system_prompt)

            sentence_buf = ""
            token_idx = 0

            async for token in token_source:
                if token_idx == 0:
                    print(f"\n  LLM TTFT: {stats.llm_ttft_ms:.0f}ms\n")
                    print("  LLM output: ", end="", flush=True)

                print(token, end="", flush=True)

                await ws.send(make_event("input_text.append", text=token))
                sentence_buf += token
                token_idx += 1

                # Commit at sentence boundaries for lower latency
                stripped = sentence_buf.rstrip()
                if any(stripped.endswith(p) for p in ".!?"):
                    if stats.tts_first_commit == 0:
                        stats.tts_first_commit = time.perf_counter()
                    await ws.send(make_event("input_text.commit"))
                    stats.tts_commits += 1
                    sentence_buf = ""

            # Commit any remaining text
            if sentence_buf.strip():
                if stats.tts_first_commit == 0:
                    stats.tts_first_commit = time.perf_counter()
                await ws.send(make_event("input_text.commit"))
                stats.tts_commits += 1

            await ws.send(make_event("input_text.done"))
            print("\n")

        # -- Receiver: TTS audio -> speaker --
        async def recv_and_play():
            nonlocal leftover
            done = False
            while not done:
                raw = await ws.recv()
                msg = json.loads(raw)
                etype = msg["type"]

                if etype == "conversation.item.speech.data":
                    now = time.perf_counter()
                    audio_bytes = base64.b64decode(msg["audio"])

                    if stats.tts_first_audio == 0:
                        stats.tts_first_audio = now
                        e2e = stats.e2e_first_audio_ms
                        tts_lat = (now - stats.tts_first_commit) * 1000 if stats.tts_first_commit else 0
                        print(f"  << FIRST AUDIO  e2e={e2e:.0f}ms  tts_latency={tts_lat:.0f}ms")

                    stats.chunk_times.append(now)
                    stats.tts_audio_bytes += len(audio_bytes)
                    stats.tts_audio_chunks += 1
                    stats.tts_last_audio = now

                    pcm = leftover + audio_bytes
                    usable = len(pcm) - (len(pcm) % 2)
                    leftover = pcm[usable:]
                    if usable > 0:
                        samples = np.frombuffer(pcm[:usable], dtype=np.int16)
                        audio_stream.write(samples)

                elif etype == "conversation.item.speech.completed":
                    is_last = msg.get("is_last_result", False)
                    meta = msg.get("synthesis_metadata") or {}
                    print(f"  << SYNTH DONE  chunks={msg.get('total_audio_chunks', '?')}"
                          f"  audio_dur={meta.get('audio_duration_ms', '?')}ms"
                          f"{'  [FINAL]' if is_last else ''}")
                    if is_last:
                        stats.tts_synthesis_done = time.perf_counter()
                        done = True

                elif etype == "input_text.committed":
                    pass

                elif etype == "error":
                    err = msg.get("error", {})
                    print(f"  !! ERROR: {err.get('message', msg)}")
                    done = True

        await asyncio.gather(send_llm_to_tts(), recv_and_play())

    # Wait for playback to drain
    if stats.tts_audio_bytes > 0:
        audio_dur = stats.audio_duration_s
        elapsed = time.perf_counter() - stats.tts_first_audio
        wait = max(0, audio_dur - elapsed + 0.2)
        if wait > 0:
            print(f"  Waiting {wait:.1f}s for playback to finish ...")
            await asyncio.sleep(wait)

    audio_stream.stop()
    audio_stream.close()

    stats.report(question, tts_voice)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time LLM -> TTS pipeline: ask a question, hear the answer"
    )
    parser.add_argument("question", nargs="?",
                        default="Explain quantum entanglement in simple terms.")
    parser.add_argument("--voice", default=TTS_VOICE)
    parser.add_argument("--tts-server", default=TTS_SERVER)
    parser.add_argument("--llm-url", default=LLM_BASE_URL)
    parser.add_argument("--llm-model", default=LLM_MODEL)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--dry-run", action="store_true",
                        help="Use canned text instead of calling the LLM (for testing TTS only)")
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    args = parser.parse_args()

    print("=" * 70)
    print("  LLM -> TTS REALTIME PIPELINE")
    print("=" * 70)
    if args.dry_run:
        print(f"  Mode:   DRY RUN (no LLM, using text as-is)")
        print(f"  Text:   {args.question[:60]}")
    else:
        print(f"  LLM:    {args.llm_model} @ {args.llm_url}")
        print(f"  Question: {args.question}")
    print(f"  TTS:    {args.voice} @ {args.tts_server}")
    print("-" * 70)

    asyncio.run(run_pipeline(
        question=args.question,
        tts_server=args.tts_server,
        tts_voice=args.voice,
        sample_rate=args.sample_rate,
        llm_base_url=args.llm_url,
        llm_model=args.llm_model,
        dry_run=args.dry_run,
        system_prompt=args.system_prompt,
    ))


if __name__ == "__main__":
    main()
