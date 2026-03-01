"""Stream TTS audio from Riva NIM and play it in real-time.

Measures time-to-first-audio-token (TTFAT), per-chunk latencies,
total synthesis wall-clock time, and throughput.

Usage:
    python stream_tts.py
    python stream_tts.py --voice Magpie-Multilingual.EN-US.Ray
    python stream_tts.py --sample-rate 44100
"""

import argparse
import asyncio
import time
from dataclasses import dataclass, field

import aiohttp
import numpy as np
import sounddevice as sd

RIVA_SERVER = "http://192.168.7.163:9000"
SAMPLE_RATE = 22050
VOICE = "Magpie-Multilingual.EN-US.Aria"
TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "This sentence is being streamed directly from the NVIDIA Riva "
    "text to speech server, and played back in real time as audio chunks arrive."
)


@dataclass
class StreamStats:
    request_sent_at: float = 0.0
    first_byte_at: float = 0.0
    last_byte_at: float = 0.0
    chunk_times: list[float] = field(default_factory=list)
    chunk_sizes: list[int] = field(default_factory=list)
    total_audio_bytes: int = 0
    num_chunks: int = 0

    @property
    def ttfat_ms(self) -> float:
        """Time to first audio token in milliseconds."""
        return (self.first_byte_at - self.request_sent_at) * 1000

    @property
    def total_wall_time_ms(self) -> float:
        return (self.last_byte_at - self.request_sent_at) * 1000

    @property
    def total_transfer_time_ms(self) -> float:
        return (self.last_byte_at - self.first_byte_at) * 1000

    @property
    def audio_duration_s(self) -> float:
        num_samples = self.total_audio_bytes // 2  # 16-bit PCM
        return num_samples / SAMPLE_RATE

    @property
    def real_time_factor(self) -> float:
        """Audio duration / wall time. >1 means faster than real-time."""
        wall_s = self.total_wall_time_ms / 1000
        return self.audio_duration_s / wall_s if wall_s > 0 else 0

    @property
    def avg_chunk_size(self) -> float:
        return self.total_audio_bytes / self.num_chunks if self.num_chunks else 0

    @property
    def inter_chunk_latencies_ms(self) -> list[float]:
        if len(self.chunk_times) < 2:
            return []
        return [
            (self.chunk_times[i] - self.chunk_times[i - 1]) * 1000
            for i in range(1, len(self.chunk_times))
        ]

    def report(self, text: str, voice: str, sample_rate: int):
        icl = self.inter_chunk_latencies_ms
        sizes = self.chunk_sizes

        print("\n" + "=" * 65)
        print("  RIVA NIM TTS STREAMING BENCHMARK")
        print("=" * 65)
        print(f"  Text:           {text[:70]}{'...' if len(text) > 70 else ''}")
        print(f"  Text length:    {len(text)} chars")
        print(f"  Voice:          {voice}")
        print(f"  Sample rate:    {sample_rate} Hz")
        print("-" * 65)
        print(f"  TTFAT:          {self.ttfat_ms:>10.1f} ms")
        print(f"  Total wall:     {self.total_wall_time_ms:>10.1f} ms")
        print(f"  Transfer time:  {self.total_transfer_time_ms:>10.1f} ms")
        print(f"  Audio duration: {self.audio_duration_s:>10.2f} s")
        print(f"  RTF:            {self.real_time_factor:>10.2f}x real-time")
        print("-" * 65)
        print(f"  Chunks:         {self.num_chunks:>10d}")
        print(f"  Total bytes:    {self.total_audio_bytes:>10,d}")
        print(f"  Avg chunk:      {self.avg_chunk_size:>10.0f} bytes")
        if sizes:
            print(f"  Min chunk:      {min(sizes):>10,d} bytes")
            print(f"  Max chunk:      {max(sizes):>10,d} bytes")
        if icl:
            print("-" * 65)
            print(f"  Avg inter-chunk:{np.mean(icl):>10.1f} ms")
            print(f"  Med inter-chunk:{np.median(icl):>10.1f} ms")
            print(f"  Min inter-chunk:{min(icl):>10.1f} ms")
            print(f"  Max inter-chunk:{max(icl):>10.1f} ms")
            print(f"  Std inter-chunk:{np.std(icl):>10.1f} ms")
        print("=" * 65)


async def stream_and_play(text: str, voice: str, server: str, sample_rate: int):
    stats = StreamStats()

    stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=1024,
    )
    stream.start()

    url = f"{server}/v1/audio/synthesize_online"
    form = aiohttp.FormData()
    form.add_field("text", text)
    form.add_field("language", "en-US")
    form.add_field("voice", voice)
    form.add_field("sample_rate_hz", str(sample_rate))
    form.add_field("encoding", "LINEAR_PCM")

    print(f"Sending request to {server} ...")
    print(f"Voice: {voice}")
    print(f"Text:  {text}\n")

    async with aiohttp.ClientSession() as session:
        stats.request_sent_at = time.perf_counter()

        async with session.post(url, data=form) as resp:
            if resp.status != 200:
                body = await resp.text()
                print(f"Error {resp.status}: {body}")
                stream.stop()
                stream.close()
                return

            chunk_idx = 0
            leftover = b""
            async for chunk in resp.content.iter_any():
                now = time.perf_counter()

                if chunk_idx == 0:
                    stats.first_byte_at = now
                    print(f"  [chunk {chunk_idx:>3d}]  FIRST AUDIO  "
                          f"TTFAT={stats.ttfat_ms:.1f}ms  "
                          f"size={len(chunk):,d} bytes")
                else:
                    delta = (now - stats.chunk_times[-1]) * 1000
                    print(f"  [chunk {chunk_idx:>3d}]  +{delta:>6.1f}ms  "
                          f"size={len(chunk):,d} bytes")

                stats.chunk_times.append(now)
                stats.chunk_sizes.append(len(chunk))
                stats.total_audio_bytes += len(chunk)
                stats.num_chunks += 1

                pcm = leftover + chunk
                # 16-bit samples need even byte count; stash any trailing byte
                usable = len(pcm) - (len(pcm) % 2)
                leftover = pcm[usable:]
                if usable > 0:
                    samples = np.frombuffer(pcm[:usable], dtype=np.int16)
                    stream.write(samples)

                chunk_idx += 1

        stats.last_byte_at = time.perf_counter()

    remaining_samples = stats.total_audio_bytes // 2
    remaining_duration = remaining_samples / sample_rate
    playback_elapsed = time.perf_counter() - stats.first_byte_at
    wait_time = max(0, remaining_duration - playback_elapsed + 0.1)
    if wait_time > 0:
        print(f"\n  Waiting {wait_time:.1f}s for playback to finish ...")
        await asyncio.sleep(wait_time)

    stream.stop()
    stream.close()

    stats.report(text, voice, sample_rate)


def main():
    parser = argparse.ArgumentParser(description="Stream Riva NIM TTS with real-time playback")
    parser.add_argument("--text", default=TEXT)
    parser.add_argument("--voice", default=VOICE)
    parser.add_argument("--server", default=RIVA_SERVER)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    args = parser.parse_args()

    asyncio.run(stream_and_play(args.text, args.voice, args.server, args.sample_rate))


if __name__ == "__main__":
    main()
