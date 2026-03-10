#!/usr/bin/env python3
"""CLI client for testing the multi-agent orchestrator without voice.

Usage:
  source .venv/bin/activate && python multiagent/client.py

Requires the LangGraph dev server to be running:
  cd multiagent && langgraph dev --n-jobs-per-worker 10
"""

import asyncio
import os
import time

from dotenv import load_dotenv

load_dotenv("multiagent/.env", override=True)

LANGGRAPH_URL = os.getenv("LANGGRAPH_URL", "http://localhost:2024")
GRAPH_NAME = "orchestrator"


async def main():
    from langgraph_sdk import get_client

    client = get_client(url=LANGGRAPH_URL)
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    print(f"\n{'='*60}")
    print(f"  Multi-Agent CLI (thread={thread_id[:8]}…)")
    print(f"  LangGraph @ {LANGGRAPH_URL}")
    print(f"{'='*60}")
    print("  Type your message and press Enter. Ctrl+C to quit.\n")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not text:
            continue

        t0 = time.perf_counter()
        current_speaker = None

        try:
            async for chunk in client.runs.stream(
                thread_id,
                GRAPH_NAME,
                input={"human_input": text},
                stream_mode=["messages", "updates"],
            ):
                event_type = chunk.event if hasattr(chunk, "event") else ""
                data = chunk.data if hasattr(chunk, "data") else {}

                if event_type == "messages/partial":
                    if isinstance(data, list):
                        for item in data:
                            msg_chunk = item[0] if isinstance(item, (list, tuple)) else item
                            metadata = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else {}

                            node = metadata.get("langgraph_node", "")
                            if node not in ("invoke_primary", "invoke_secondary"):
                                continue

                            content = ""
                            if isinstance(msg_chunk, dict):
                                content = msg_chunk.get("content", "")
                            elif hasattr(msg_chunk, "content"):
                                content = msg_chunk.content or ""

                            if not content:
                                continue

                            speaker = metadata.get("langgraph_node", "agent")
                            if speaker != current_speaker:
                                if current_speaker is not None:
                                    print()
                                current_speaker = speaker
                                label = "Primary" if "primary" in speaker else "Secondary"
                                elapsed = (time.perf_counter() - t0) * 1000
                                print(f"\n[{label} | {elapsed:.0f}ms] ", end="", flush=True)

                            print(content, end="", flush=True)

                elif event_type == "updates" and isinstance(data, dict):
                    for node_name, updates in data.items():
                        if node_name == "route":
                            r = updates.get("routing", {})
                            if r:
                                elapsed = (time.perf_counter() - t0) * 1000
                                print(
                                    f"  [Router {elapsed:.0f}ms] "
                                    f"primary={r.get('primary_agent')} "
                                    f"secondary={r.get('secondary_agent', 'none')} "
                                    f"topic={r.get('topic')}"
                                )
                            continue

                        if node_name not in ("invoke_primary", "invoke_secondary"):
                            continue

                        for msg in updates.get("messages", []):
                            speaker = msg.get("speaker", "")
                            if not speaker or speaker == "human":
                                continue
                            if current_speaker and "invoke" in current_speaker:
                                continue

                            elapsed = (time.perf_counter() - t0) * 1000
                            print(f"\n[{speaker} | {elapsed:.0f}ms] {msg['text']}")

        except Exception as e:
            print(f"\nError: {e}")

        total = (time.perf_counter() - t0) * 1000
        print(f"\n  [{total:.0f}ms total]\n")


if __name__ == "__main__":
    asyncio.run(main())
