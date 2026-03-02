"""CLI client for the LangGraph voice agent platform.

Connects to the LangGraph dev server and orchestrates:
  - Chitchat agent (fast conversational response)
  - Router agent (parallel intent classification)
  - Worker agent (background tool execution)

Usage:
    cd langgraph_agents
    source ../.venv/bin/activate && python cli_client.py
"""

import asyncio
import json
import sys
import uuid

from langgraph_sdk import get_client
from loguru import logger

SERVER_URL = "http://127.0.0.1:2024"


class Session:
    """Manages a user session with chat thread, router, and worker dispatching."""

    def __init__(self, client):
        self.client = client
        self.session_id = uuid.uuid4().hex[:8]
        self.chat_thread_id: str | None = None
        self.pending_jobs: dict[str, dict] = {}
        self.recent_history: list[dict] = []
        self._user_busy = False
        self._queued_completions: list[str] = []

    async def start(self):
        thread = await self.client.threads.create()
        self.chat_thread_id = thread["thread_id"]
        logger.info(f"Session {self.session_id} started, chat thread: {self.chat_thread_id}")

    async def _recover_session(self):
        """Re-create the chat thread after a server reload."""
        thread = await self.client.threads.create()
        self.chat_thread_id = thread["thread_id"]
        self.pending_jobs.clear()
        logger.info(f"Session recovered, new chat thread: {self.chat_thread_id}")

    def _add_to_history(self, role: str, content: str):
        self.recent_history.append({"role": role, "content": content})
        if len(self.recent_history) > 20:
            self.recent_history = self.recent_history[-20:]

    def _pending_jobs_list(self) -> list[dict]:
        """Build a list of pending job summaries for the chitchat agent."""
        return [
            {"job_id": jid, "tool": info["tool"]}
            for jid, info in self.pending_jobs.items()
        ]

    async def _run_chitchat(
        self, user_input: str, input_type: str = "user_message",
        *, print_stream: bool = True, **extra,
    ) -> str:
        """Stream a chitchat run token-by-token and return the full response."""
        payload = {
            "session_id": self.session_id,
            "type": input_type,
            "message": user_input,
            "pending_jobs": self._pending_jobs_list(),
            **extra,
        }

        full_response = ""
        prev_len = 0

        try:
            async for chunk in self.client.runs.stream(
                self.chat_thread_id,
                assistant_id="chitchat",
                input=payload,
                stream_mode="messages",
            ):
                if chunk.event == "messages/partial":
                    data = chunk.data
                    if isinstance(data, list) and data:
                        content = data[0].get("content", "")
                        if content and len(content) > prev_len:
                            new_text = content[prev_len:]
                            if print_stream:
                                print(new_text, end="", flush=True)
                            prev_len = len(content)
                            full_response = content
        except Exception as e:
            err = str(e)
            if "thread" in err.lower() or "not found" in err.lower() or "404" in err:
                logger.warning("Server reloaded — creating new chat thread")
                await self._recover_session()
                return await self._run_chitchat(
                    user_input, input_type, print_stream=print_stream, **extra
                )
            raise

        if print_stream and full_response:
            print()  # newline after streaming

        return full_response

    async def _run_router(self, user_input: str) -> dict:
        """Run the router (stateless) and return the classification."""
        try:
            result = await self.client.runs.wait(
                None,
                assistant_id="router",
                input={
                    "message": user_input,
                    "recent_history": self.recent_history[-10:],
                },
            )
        except Exception as e:
            logger.warning(f"Router call failed (server may have reloaded): {e}")
            return {"action": "none"}
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"action": "none"}
        return result if isinstance(result, dict) else {"action": "none"}

    async def _launch_worker(self, tool: str, params: dict, context_summary: str, original_question: str):
        """Launch a worker as a background run."""
        worker_thread = await self.client.threads.create()
        job_id = f"job-{uuid.uuid4().hex[:6]}"

        run = await self.client.runs.create(
            worker_thread["thread_id"],
            assistant_id="worker",
            input={
                "tool": tool,
                "params": params,
                "session_id": self.session_id,
                "job_id": job_id,
                "context_summary": context_summary,
            },
        )

        self.pending_jobs[job_id] = {
            "worker_thread_id": worker_thread["thread_id"],
            "run_id": run["run_id"],
            "tool": tool,
            "original_question": original_question,
        }

        logger.debug(f"Worker launched: job_id={job_id} tool={tool} run_id={run['run_id']}")
        return job_id

    async def _check_pending_jobs(self):
        """Check if any pending jobs have completed. Queue delivery if user is busy."""
        completed = []
        for job_id, info in self.pending_jobs.items():
            try:
                run = await self.client.runs.get(
                    info["worker_thread_id"],
                    info["run_id"],
                )
                if run["status"] == "success":
                    completed.append(job_id)
                elif run["status"] in ("error", "interrupted"):
                    logger.error(f"Job {job_id} failed: {run.get('status')}")
                    completed.append(job_id)
            except Exception as e:
                logger.debug(f"Could not check job {job_id}: {e}")

        for job_id in completed:
            info = self.pending_jobs.pop(job_id)
            logger.debug(f"Job {job_id} completed!")

            if self._user_busy:
                self._queued_completions.append(job_id + "|" + info["original_question"])
                logger.debug(f"User busy — queued delivery for {job_id}")
            else:
                await self._deliver_result(job_id, info["original_question"])

    async def _deliver_result(self, job_id: str, original_question: str):
        """Deliver a completed job's result via chitchat."""
        print(f"\n🔔 [Job Complete] ", end="", flush=True)
        response = await self._run_chitchat(
            "",
            input_type="job_result",
            print_stream=True,
            job_id=job_id,
            original_question=original_question,
        )
        self._add_to_history("assistant", response)

    async def _deliver_queued(self):
        """Deliver any queued job completions."""
        while self._queued_completions:
            entry = self._queued_completions.pop(0)
            job_id, original_question = entry.split("|", 1)
            await self._deliver_result(job_id, original_question)

    async def handle_message(self, user_input: str):
        """Handle a user message: chitchat + router in parallel.

        If a job completed while the user was typing, merge the delivery
        with the user's message into a single chitchat call.
        """
        self._user_busy = True
        self._add_to_history("user", user_input)

        has_queued = bool(self._queued_completions)

        if has_queued:
            entry = self._queued_completions.pop(0)
            job_id, original_question = entry.split("|", 1)
            print("\n🔔 ", end="", flush=True)
            response = await self._run_chitchat(
                user_input,
                input_type="job_result_with_followup",
                print_stream=True,
                job_id=job_id,
                original_question=original_question,
            )
            self._add_to_history("assistant", response)
            await self._deliver_queued()
        else:
            print("\n🤖 ", end="", flush=True)
            chitchat_task = asyncio.create_task(self._run_chitchat(user_input, print_stream=True))
            router_task = asyncio.create_task(self._run_router(user_input))

            chat_response = await chitchat_task
            self._add_to_history("assistant", chat_response)

            router_result = await router_task
            logger.debug(f"Router: {router_result}")

            if router_result.get("action") == "launch_tool":
                tool = router_result["tool"]
                params = router_result.get("params", {})
                context = router_result.get("context_summary", "")
                job_id = await self._launch_worker(tool, params, context, user_input)
                print(f"   ⚙️  Tool '{tool}' launched as {job_id}")

        self._user_busy = False
        await self._check_pending_jobs()
        await self._deliver_queued()

    async def poll_jobs(self):
        """Background poller for pending jobs."""
        while True:
            await asyncio.sleep(3)
            if self.pending_jobs:
                await self._check_pending_jobs()


async def main():
    client = get_client(url=SERVER_URL)
    session = Session(client)
    await session.start()

    poll_task = asyncio.create_task(session.poll_jobs())

    print("\n" + "="*50)
    print("  LangGraph Voice Agent CLI")
    print("  Type a message and press Enter.")
    print("  Type 'quit' or 'exit' to leave.")
    print("  Type 'jobs' to check pending jobs.")
    print("="*50 + "\n")

    try:
        while True:
            session._user_busy = True
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("You: ").strip()
                )
            except EOFError:
                break
            session._user_busy = False

            if not user_input:
                await session._deliver_queued()
                continue
            if user_input.lower() in ("quit", "exit"):
                break
            if user_input.lower() == "jobs":
                if session.pending_jobs:
                    for jid, info in session.pending_jobs.items():
                        print(f"  ⏳ {jid}: {info['tool']} (running)")
                else:
                    print("  No pending jobs.")
                continue

            await session.handle_message(user_input)
    except KeyboardInterrupt:
        pass
    finally:
        poll_task.cancel()
        print("\nGoodbye!")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<dim>{time:HH:mm:ss}</dim> | <level>{message}</level>")
    asyncio.run(main())
