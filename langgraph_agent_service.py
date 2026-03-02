"""Pipecat FrameProcessor that uses a LangGraph agent as the 'LLM'.

Replaces OpenAILLMService in the pipeline. Receives LLMContextFrame
from the user aggregator, streams the chitchat response as TextFrames,
and runs the router in the background to launch tools when needed.
A background poller delivers tool results proactively, but only when
the user is NOT speaking. If a result arrives mid-speech, it is queued
and merged with the user's next message.

IMPORTANT: process_frame MUST return quickly so that system frames
(StartInterruptionFrame) can flow through and reach the pipeline sink.
All long-running work is done in asyncio background tasks.
"""

import asyncio
import json
import uuid
from typing import Optional

from langgraph_sdk import get_client
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    StartInterruptionFrame,
    TextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

LANGGRAPH_SERVER = "http://127.0.0.1:2024"


class LangGraphAgentService(FrameProcessor):
    """Streams LangGraph chitchat responses as LLM text frames."""

    def __init__(
        self,
        *,
        server_url: str = LANGGRAPH_SERVER,
        ui_queue: Optional[asyncio.Queue] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._server_url = server_url
        self._client = get_client(url=server_url)
        self._session_id = uuid.uuid4().hex[:8]
        self._chat_thread_id: Optional[str] = None
        self._pending_jobs: dict[str, dict] = {}
        self._recent_history: list[dict] = []
        self._initialized = False
        self._poll_task: Optional[asyncio.Task] = None
        self._llm_task: Optional[asyncio.Task] = None
        self._router_task: Optional[asyncio.Task] = None
        self._cancelled = False
        self._generating = False
        self._user_speaking = False
        self._queued_completions: list[dict] = []
        self._delivered_jobs: list[dict] = []
        self._ui_queue: Optional[asyncio.Queue] = ui_queue

    async def _ensure_initialized(self):
        if self._initialized:
            return
        thread = await self._client.threads.create()
        self._chat_thread_id = thread["thread_id"]
        self._initialized = True
        self._poll_task = asyncio.create_task(self._poll_jobs())
        logger.info(
            f"LangGraph session={self._session_id} "
            f"chat_thread={self._chat_thread_id}"
        )

    def _pending_jobs_list(self) -> list[dict]:
        jobs = [
            {"job_id": jid, "tool": info["tool"], "status": "running"}
            for jid, info in self._pending_jobs.items()
        ]
        jobs.extend(self._delivered_jobs)
        return jobs

    def _cancel_current(self):
        """Cancel all in-flight LLM/router tasks immediately."""
        self._cancelled = True
        for task in (self._llm_task, self._router_task):
            if task and not task.done():
                task.cancel()
        self._llm_task = None
        self._router_task = None
        self._generating = False
        logger.debug("[LangGraph] Cancelled current generation")

    async def _stream_chitchat(
        self, message: str, input_type: str = "user_message", **extra
    ) -> str:
        """Stream a chitchat response, pushing TextFrames for each token."""
        payload = {
            "session_id": self._session_id,
            "type": input_type,
            "message": message,
            "pending_jobs": self._pending_jobs_list(),
            **extra,
        }

        await self.push_frame(LLMFullResponseStartFrame())

        full_response = ""
        prev_len = 0

        try:
            async for chunk in self._client.runs.stream(
                self._chat_thread_id,
                assistant_id="chitchat",
                input=payload,
                stream_mode="messages",
            ):
                if self._cancelled:
                    logger.debug("[LangGraph] Stream cancelled mid-flight")
                    break

                if chunk.event == "messages/partial":
                    data = chunk.data
                    if isinstance(data, list) and data:
                        content = data[0].get("content", "")
                        if content and len(content) > prev_len:
                            new_text = content[prev_len:]
                            await self.push_frame(TextFrame(text=new_text))
                            prev_len = len(content)
                            full_response = content
        except asyncio.CancelledError:
            logger.debug("[LangGraph] Stream task cancelled")
        except Exception as e:
            if self._cancelled:
                logger.debug("[LangGraph] Stream error after cancel (expected)")
            else:
                logger.error(f"Chitchat stream error: {e}")
                if "not found" in str(e).lower() or "404" in str(e):
                    await self._recover()
                    await self.push_frame(
                        TextFrame(text="I had a brief hiccup. Could you say that again?")
                    )

        await self.push_frame(LLMFullResponseEndFrame())
        return full_response

    async def _run_router(self, message: str) -> dict:
        try:
            result = await self._client.runs.wait(
                None,
                assistant_id="router",
                input={
                    "message": message,
                    "recent_history": self._recent_history[-10:],
                },
            )
        except asyncio.CancelledError:
            return {"action": "none"}
        except Exception as e:
            logger.warning(f"Router failed: {e}")
            return {"action": "none"}
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"action": "none"}
        return result if isinstance(result, dict) else {"action": "none"}

    async def _launch_worker(
        self, tool: str, params: dict, context_summary: str, original_question: str
    ) -> str:
        worker_thread = await self._client.threads.create()
        job_id = f"job-{uuid.uuid4().hex[:6]}"

        run = await self._client.runs.create(
            worker_thread["thread_id"],
            assistant_id="worker",
            input={
                "tool": tool,
                "params": params,
                "session_id": self._session_id,
                "job_id": job_id,
                "context_summary": context_summary,
            },
        )

        self._pending_jobs[job_id] = {
            "worker_thread_id": worker_thread["thread_id"],
            "run_id": run["run_id"],
            "tool": tool,
            "original_question": original_question,
        }
        logger.info(f"Worker launched: {job_id} tool={tool}")
        return job_id

    async def _poll_jobs(self):
        """Background poller that checks for completed jobs.

        If the user is speaking or the bot is generating, the completed
        job is queued and delivered later (merged with the user's next turn).
        """
        while True:
            await asyncio.sleep(3)
            if not self._pending_jobs:
                continue

            completed = []
            for job_id, info in list(self._pending_jobs.items()):
                try:
                    run = await self._client.runs.get(
                        info["worker_thread_id"],
                        info["run_id"],
                    )
                    if run["status"] == "success":
                        completed.append(job_id)
                    elif run["status"] in ("error", "interrupted"):
                        logger.error(f"Job {job_id} failed")
                        completed.append(job_id)
                except Exception:
                    pass

            for job_id in completed:
                info = self._pending_jobs.pop(job_id)
                logger.info(f"Job {job_id} completed")

                if self._user_speaking or self._generating:
                    logger.debug(f"[LangGraph] User busy — queuing {job_id}")
                    self._queued_completions.append({
                        "job_id": job_id,
                        "original_question": info["original_question"],
                    })
                else:
                    await self._deliver_result(job_id, info["original_question"])

    async def _deliver_result(self, job_id: str, original_question: str):
        """Proactively deliver a completed job's results."""
        self._generating = True
        self._cancelled = False
        response = await self._stream_chitchat(
            "",
            input_type="job_result",
            job_id=job_id,
            original_question=original_question,
        )
        if not self._cancelled:
            self._recent_history.append({"role": "assistant", "content": response})
            self._delivered_jobs.append({
                "job_id": job_id,
                "tool": "long_analysis",
                "status": "delivered",
                "original_question": original_question,
            })
        self._generating = False

    async def _deliver_queued(self):
        """Deliver any queued completions that built up while user was busy."""
        while self._queued_completions and not self._user_speaking:
            entry = self._queued_completions.pop(0)
            logger.info(f"[LangGraph] Delivering queued result: {entry['job_id']}")
            await self._deliver_result(entry["job_id"], entry["original_question"])

    async def _recover(self):
        thread = await self._client.threads.create()
        self._chat_thread_id = thread["thread_id"]
        self._pending_jobs.clear()
        logger.warning("LangGraph session recovered")

    async def _handle_llm_frame(self, frame: Frame, direction: FrameDirection):
        """Handle an LLM context/messages frame (user turn complete).

        Runs as a background task so process_frame returns immediately.
        """
        try:
            await self._ensure_initialized()
            self._user_speaking = False

            if isinstance(frame, LLMContextFrame):
                messages = frame.context.messages
            else:
                messages = frame.messages

            user_msg = ""
            for msg in reversed(messages):
                role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
                content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                if role == "user":
                    if isinstance(content, list):
                        content = " ".join(
                            p.get("text", "") for p in content
                            if isinstance(p, dict) and p.get("type") == "text"
                        )
                    user_msg = str(content)
                    break

            if not user_msg:
                return

            stripped = user_msg.strip()
            word_count = len(stripped.split())
            if len(stripped) < 3 or (word_count <= 2 and len(stripped) < 12):
                logger.warning(f"[LangGraph] Ignoring fragment: '{stripped}' (words={word_count})")
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(LLMFullResponseEndFrame())
                return

            logger.info(f"[LangGraph] User: {user_msg[:100]}")

            self._generating = True
            self._cancelled = False

            if self._queued_completions:
                entry = self._queued_completions.pop(0)
                logger.info(
                    f"[LangGraph] Delivering queued result {entry['job_id']} "
                    f"before responding to user"
                )
                result_response = await self._stream_chitchat(
                    "",
                    input_type="job_result",
                    job_id=entry["job_id"],
                    original_question=entry["original_question"],
                )
                if not self._cancelled:
                    self._recent_history.append(
                        {"role": "assistant", "content": result_response}
                    )
                    self._delivered_jobs.append({
                        "job_id": entry["job_id"],
                        "tool": "long_analysis",
                        "status": "delivered",
                        "original_question": entry["original_question"],
                    })
                    logger.info(f"[LangGraph] Result delivered: {result_response[:100]}")

                if self._cancelled:
                    self._generating = False
                    return

            self._recent_history.append({"role": "user", "content": user_msg})
            response = await self._stream_chitchat(user_msg)

            if not self._cancelled:
                self._recent_history.append({"role": "assistant", "content": response})
                logger.info(f"[LangGraph] Response: {response[:100]}")

            if self._cancelled:
                self._generating = False
                return

            router_task = asyncio.create_task(self._run_router(user_msg))
            self._router_task = router_task

            try:
                router_result = await router_task
            except asyncio.CancelledError:
                router_result = {"action": "none"}

            if not self._cancelled and router_result.get("action") == "launch_tool":
                tool = router_result["tool"]
                already_running = any(
                    info["tool"] == tool for info in self._pending_jobs.values()
                )
                if already_running:
                    logger.info(f"[LangGraph] Skipping duplicate launch: {tool} already running")
                else:
                    params = router_result.get("params", {})
                    context = router_result.get("context_summary", "")
                    await self._launch_worker(tool, params, context, user_msg)

            if len(self._recent_history) > 20:
                self._recent_history = self._recent_history[-20:]

            self._generating = False
            self._llm_task = None
            self._router_task = None

            if self._queued_completions and not self._user_speaking:
                await self._deliver_queued()

        except asyncio.CancelledError:
            logger.debug("[LangGraph] _handle_llm_frame cancelled")
            self._generating = False
        except Exception as e:
            logger.error(f"[LangGraph] Error: {e}", exc_info=True)
            self._generating = False
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(TextFrame(text="Sorry, could you try again?"))
                await self.push_frame(LLMFullResponseEndFrame())
            except Exception:
                pass

    @staticmethod
    def _extract_user_text(frame: Frame) -> str:
        """Quick extraction of user text from an LLM frame for fragment check."""
        if isinstance(frame, LLMContextFrame):
            messages = frame.context.messages
        else:
            messages = getattr(frame, "messages", [])
        for msg in reversed(messages or []):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                return str(content).strip()
        return ""

    @staticmethod
    def _is_fragment(text: str) -> bool:
        word_count = len(text.split())
        return len(text) < 3 or (word_count <= 2 and len(text) < 12)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Route frames. Returns IMMEDIATELY for LLM frames by spawning
        background tasks, so system frames (interruptions) are never blocked.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, (LLMMessagesFrame, LLMContextFrame)):
            user_text = self._extract_user_text(frame)
            if user_text and self._is_fragment(user_text):
                logger.warning(f"[LangGraph] Dropping fragment in process_frame: '{user_text}'")
                return

            self._cancel_current()
            self._cancelled = False
            task = asyncio.create_task(self._handle_llm_frame(frame, direction))
            self._llm_task = task

        elif isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
            await self.push_frame(frame, direction)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            await self.push_frame(frame, direction)

        elif isinstance(frame, StartInterruptionFrame):
            self._cancel_current()
            await self.push_frame(frame, direction)

        elif isinstance(frame, (CancelFrame, EndFrame)):
            self._cancel_current()
            if isinstance(frame, EndFrame) and self._poll_task:
                self._poll_task.cancel()
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
