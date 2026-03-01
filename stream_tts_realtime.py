"""Realtime WebSocket TTS streaming with token-by-token text input.

Simulates an LLM streaming tokens into the Riva NIM TTS server via the
realtime WebSocket API, plays audio as it arrives, and reports latency stats.

Protocol:
  1. Connect to ws://<server>/v1/realtime?intent=synthesize
  2. Send synthesize_session.update to configure voice/sample_rate
  3. Stream tokens via input_text.append (simulating LLM output)
  4. Send input_text.commit after each sentence to trigger synthesis
  5. Send input_text.done when all text is sent
  6. Receive conversation.item.speech.data with base64 audio chunks

Usage:
    python stream_tts_realtime.py
    python stream_tts_realtime.py --voice Magpie-Multilingual.EN-US.Ray
    python stream_tts_realtime.py --token-delay 0.05
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

SERVER = "192.168.7.163:9000"
SAMPLE_RATE = 22050
VOICE = "Magpie-Multilingual.EN-US.Aria"

TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "This sentence is being streamed token by token, "
    "just like an LLM would produce it. "
    "Each word arrives with a small delay, simulating real inference."
)

# Simulated LLM token delay in seconds
TOKEN_DELAY = 0.03


@dataclass
class RealtimeStats:
    ws_connected_at: float = 0.0
    session_ready_at: float = 0.0
    first_token_sent_at: float = 0.0
    first_commit_at: float = 0.0
    first_audio_at: float = 0.0
    last_audio_at: float = 0.0
    all_text_done_at: float = 0.0
    synthesis_done_at: float = 0.0

    audio_chunks: int = 0
    audio_bytes: int = 0
    tokens_sent: int = 0
    commits_sent: int = 0

    chunk_times: list[float] = field(default_factory=list)
    chunk_sizes: list[int] = field(default_factory=list)

    server_synthesis_time_ms: float = 0.0
    server_audio_duration_ms: float = 0.0

    @property
    def ttfat_from_connect_ms(self) -> float:
        return (self.first_audio_at - self.ws_connected_at) * 1000

    @property
    def ttfat_from_first_token_ms(self) -> float:
        return (self.first_audio_at - self.first_token_sent_at) * 1000

    @property
    def ttfat_from_commit_ms(self) -> float:
        return (self.first_audio_at - self.first_commit_at) * 1000

    @property
    def total_wall_ms(self) -> float:
        end = self.synthesis_done_at or self.last_audio_at
        return (end - self.ws_connected_at) * 1000

    @property
    def audio_duration_s(self) -> float:
        return (self.audio_bytes // 2) / SAMPLE_RATE

    @property
    def inter_chunk_latencies_ms(self) -> list[float]:
        if len(self.chunk_times) < 2:
            return []
        return [
            (self.chunk_times[i] - self.chunk_times[i - 1]) * 1000
            for i in range(1, len(self.chunk_times))
        ]

    def report(self, text: str, voice: str, token_delay: float):
        icl = self.inter_chunk_latencies_ms
        print("\n" + "=" * 70)
        print("  RIVA NIM REALTIME WEBSOCKET TTS BENCHMARK")
        print("=" * 70)
        print(f"  Text length:      {len(text)} chars, {len(text.split())} words")
        print(f"  Voice:            {voice}")
        print(f"  Token delay:      {token_delay*1000:.0f} ms (simulated LLM)")
        print(f"  Sample rate:      {SAMPLE_RATE} Hz")
        print("-" * 70)
        print(f"  Session setup:    {(self.session_ready_at - self.ws_connected_at)*1000:>10.1f} ms")
        print(f"  TTFAT (connect):  {self.ttfat_from_connect_ms:>10.1f} ms")
        print(f"  TTFAT (1st tok):  {self.ttfat_from_first_token_ms:>10.1f} ms")
        print(f"  TTFAT (commit):   {self.ttfat_from_commit_ms:>10.1f} ms")
        print(f"  Total wall time:  {self.total_wall_ms:>10.1f} ms")
        print(f"  Audio duration:   {self.audio_duration_s:>10.2f} s")
        if self.server_synthesis_time_ms:
            print(f"  Server synth:     {self.server_synthesis_time_ms:>10.1f} ms (server-reported)")
        if self.server_audio_duration_ms:
            print(f"  Server audio dur: {self.server_audio_duration_ms:>10.1f} ms (server-reported)")
        print("-" * 70)
        print(f"  Tokens sent:      {self.tokens_sent:>10d}")
        print(f"  Commits sent:     {self.commits_sent:>10d}")
        print(f"  Audio chunks:     {self.audio_chunks:>10d}")
        print(f"  Audio bytes:      {self.audio_bytes:>10,d}")
        if self.chunk_sizes:
            print(f"  Avg chunk:        {np.mean(self.chunk_sizes):>10.0f} bytes")
            print(f"  Min chunk:        {min(self.chunk_sizes):>10,d} bytes")
            print(f"  Max chunk:        {max(self.chunk_sizes):>10,d} bytes")
        if icl:
            print("-" * 70)
            print(f"  Avg inter-chunk:  {np.mean(icl):>10.1f} ms")
            print(f"  Med inter-chunk:  {np.median(icl):>10.1f} ms")
            print(f"  Min inter-chunk:  {min(icl):>10.1f} ms")
            print(f"  Max inter-chunk:  {max(icl):>10.1f} ms")
        print("=" * 70)


def make_event(event_type: str, **kwargs) -> str:
    event = {"event_id": f"event_{uuid.uuid4().hex[:8]}", "type": event_type, **kwargs}
    return json.dumps(event)


def tokenize(text: str) -> list[str]:
    """Split text into word-level tokens, preserving trailing spaces."""
    words = text.split(" ")
    return [w + " " for w in words[:-1]] + [words[-1]]


async def stream_realtime(text: str, voice: str, server: str, sample_rate: int, token_delay: float):
    stats = RealtimeStats()

    stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="int16", blocksize=1024)
    stream.start()

    ws_url = f"ws://{server}/v1/realtime?intent=synthesize"
    print(f"Connecting to {ws_url} ...")

    leftover = b""

    async with websockets.connect(ws_url, max_size=16 * 1024 * 1024) as ws:
        stats.ws_connected_at = time.perf_counter()
        print("  WebSocket connected.")

        # Wait for conversation.created
        msg = json.loads(await ws.recv())
        assert msg["type"] == "conversation.created", f"Unexpected: {msg}"
        print(f"  Conversation created: {msg['conversation']['id']}")

        # Configure session
        await ws.send(make_event(
            "synthesize_session.update",
            session={
                "input_text_synthesis": {
                    "language_code": "en-US",
                    "voice_name": voice,
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
        stats.session_ready_at = time.perf_counter()
        print(f"  Session configured: voice={voice}, rate={sample_rate}")

        tokens = tokenize(text)
        print(f"\n  Streaming {len(tokens)} tokens with {token_delay*1000:.0f}ms delay ...\n")

        # -- Sender task: stream tokens and commit at sentence boundaries --
        async def send_tokens():
            buffer = ""
            for i, token in enumerate(tokens):
                if stats.first_token_sent_at == 0:
                    stats.first_token_sent_at = time.perf_counter()

                await ws.send(make_event("input_text.append", text=token))
                stats.tokens_sent += 1
                buffer += token

                is_sentence_end = any(buffer.rstrip().endswith(p) for p in ".!?")
                is_last = i == len(tokens) - 1

                if is_sentence_end or is_last:
                    if stats.first_commit_at == 0:
                        stats.first_commit_at = time.perf_counter()
                    await ws.send(make_event("input_text.commit"))
                    stats.commits_sent += 1
                    print(f"  >> COMMIT ({len(buffer.split())} words): "
                          f"{buffer[:60]}{'...' if len(buffer) > 60 else ''}")
                    buffer = ""

                if not is_last:
                    await asyncio.sleep(token_delay)

            stats.all_text_done_at = time.perf_counter()
            await ws.send(make_event("input_text.done"))
            print(f"\n  >> ALL TEXT DONE (sent input_text.done)\n")

        # -- Receiver task: collect audio and play it --
        async def recv_audio():
            nonlocal leftover
            done = False
            while not done:
                raw = await ws.recv()
                msg = json.loads(raw)
                etype = msg["type"]

                if etype == "conversation.item.speech.data":
                    now = time.perf_counter()
                    audio_b64 = msg["audio"]
                    audio_bytes = base64.b64decode(audio_b64)

                    if stats.first_audio_at == 0:
                        stats.first_audio_at = now
                        print(f"  << FIRST AUDIO  TTFAT(commit)={stats.ttfat_from_commit_ms:.1f}ms  "
                              f"size={len(audio_bytes):,d} bytes")
                    else:
                        delta = (now - stats.chunk_times[-1]) * 1000
                        print(f"  << audio chunk #{stats.audio_chunks:>3d}  "
                              f"+{delta:>6.1f}ms  size={len(audio_bytes):,d} bytes"
                              f"{'  LAST' if msg.get('is_last_chunk') else ''}")

                    stats.chunk_times.append(now)
                    stats.chunk_sizes.append(len(audio_bytes))
                    stats.audio_bytes += len(audio_bytes)
                    stats.audio_chunks += 1
                    stats.last_audio_at = now

                    pcm = leftover + audio_bytes
                    usable = len(pcm) - (len(pcm) % 2)
                    leftover = pcm[usable:]
                    if usable > 0:
                        samples = np.frombuffer(pcm[:usable], dtype=np.int16)
                        stream.write(samples)

                elif etype == "conversation.item.speech.completed":
                    meta = msg.get("synthesis_metadata") or {}
                    stats.server_synthesis_time_ms += meta.get("synthesis_time_ms", 0)
                    stats.server_audio_duration_ms += meta.get("audio_duration_ms", 0)
                    is_last = msg.get("is_last_result", False)
                    print(f"  << SYNTHESIS COMPLETED  "
                          f"chunks={msg.get('total_audio_chunks', '?')}  "
                          f"synth={meta.get('synthesis_time_ms', '?')}ms  "
                          f"audio_dur={meta.get('audio_duration_ms', '?')}ms"
                          f"{'  [FINAL]' if is_last else ''}")
                    if is_last:
                        stats.synthesis_done_at = time.perf_counter()
                        done = True

                elif etype == "input_text.committed":
                    pass  # expected ack

                elif etype == "error":
                    err = msg.get("error", {})
                    print(f"  !! ERROR: {err.get('message', msg)}")
                    done = True

                else:
                    print(f"  ?? {etype}: {json.dumps(msg)[:120]}")

        # Run sender and receiver concurrently
        await asyncio.gather(send_tokens(), recv_audio())

    # Wait for playback to drain
    if stats.audio_bytes > 0:
        audio_dur = stats.audio_duration_s
        elapsed = time.perf_counter() - stats.first_audio_at
        wait = max(0, audio_dur - elapsed + 0.2)
        if wait > 0:
            print(f"  Waiting {wait:.1f}s for playback to finish ...")
            await asyncio.sleep(wait)

    stream.stop()
    stream.close()

    stats.report(text, voice, token_delay)


def main():
    parser = argparse.ArgumentParser(description="Realtime WebSocket TTS streaming benchmark")
    parser.add_argument("--text", default=TEXT)
    parser.add_argument("--voice", default=VOICE)
    parser.add_argument("--server", default=SERVER)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--token-delay", type=float, default=TOKEN_DELAY,
                        help="Delay between tokens in seconds (simulates LLM)")
    args = parser.parse_args()

    asyncio.run(stream_realtime(args.text, args.voice, args.server, args.sample_rate, args.token_delay))


if __name__ == "__main__":
    main()
