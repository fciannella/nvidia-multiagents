"""Pipecat FrameProcessor that uses LangGraph agents as the 'LLM'.

Manages a state machine with three states:
  IDLE             — normal chitchat + router flow
  WORKFLOW_ACTIVE  — a workflow is running in the background
  AWAITING_INPUT   — workflow paused (interrupt), waiting for user answer

In IDLE state, user messages go to chitchat (immediate response) then
the router decides whether to launch a workflow. In WORKFLOW_ACTIVE
state, user messages go to chitchat only (casual chat while waiting).
In AWAITING_INPUT state, the user's next message resumes the workflow.

The workflow agent's questions and results are always delivered through
chitchat so the user hears a consistent voice (Ron).

IMPORTANT: process_frame MUST return quickly so that system frames
(StartInterruptionFrame) can flow through. All long-running work is
done in asyncio background tasks.
"""

import asyncio
import enum
import random
import uuid
from typing import Optional

from langgraph_sdk import get_client
from langgraph_sdk.schema import Command
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    StartFrame,
    StartInterruptionFrame,
    TextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

LANGGRAPH_SERVER = "http://127.0.0.1:2024"


class AgentState(enum.Enum):
    IDLE = "idle"
    WORKFLOW_ACTIVE = "workflow_active"
    AWAITING_INPUT = "awaiting_input"


class LangGraphAgentService(FrameProcessor):
    """Streams LangGraph chitchat responses as LLM text frames.
    Coordinates workflow agent for complex multi-step tasks.
    """

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
        self._recent_history: list[dict] = []
        self._initialized = False
        self._warmup_task: Optional[asyncio.Task] = None

        # State machine
        self._state = AgentState.IDLE
        self._workflow_thread_id: Optional[str] = None
        self._workflow_run_id: Optional[str] = None
        self._workflow_original_question: Optional[str] = None
        self._workflow_tool: Optional[str] = None
        self._workflow_assistant_id: str = "workflow"
        self._workflow_task: Optional[asyncio.Task] = None
        self._pending_question: Optional[str] = None
        self._question_delivered: bool = False
        self._pending_result: Optional[str] = None
        self._fake_interrupt: bool = False

        self._persistent_threads: dict[str, str] = {}

        # Legacy background jobs (kept for backward compat)
        self._pending_jobs: dict[str, dict] = {}
        self._delivered_jobs: list[dict] = []
        self._poll_task: Optional[asyncio.Task] = None

        # Frame management
        self._llm_task: Optional[asyncio.Task] = None
        self._router_task: Optional[asyncio.Task] = None
        self._cancelled = False
        self._generating = False
        self._user_speaking = False
        self._ui_queue: Optional[asyncio.Queue] = ui_queue

    async def _ensure_initialized(self):
        if self._initialized:
            return
        thread = await self._client.threads.create()
        self._chat_thread_id = thread["thread_id"]
        self._initialized = True
        self._poll_task = asyncio.create_task(self._poll_legacy_jobs())
        logger.info(
            f"LangGraph session={self._session_id} "
            f"chat_thread={self._chat_thread_id}"
        )

    def _cancel_current(self):
        self._cancelled = True
        for task in (self._llm_task, self._router_task):
            if task and not task.done():
                task.cancel()
        self._llm_task = None
        self._router_task = None
        self._generating = False
        logger.debug("[LangGraph] Cancelled current generation")

    # ---- Chitchat streaming ----

    async def _stream_chitchat(
        self, message: str, input_type: str = "user_message", **extra
    ) -> str:
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
        interrupted = False

        try:
            async for chunk in self._client.runs.stream(
                self._chat_thread_id,
                assistant_id="chitchat",
                input=payload,
                stream_mode="messages",
            ):
                if self._cancelled:
                    interrupted = True
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
            interrupted = True
            logger.debug("[LangGraph] Chitchat stream cancelled")
        except Exception as e:
            if not self._cancelled:
                logger.error(f"Chitchat stream error: {e}")
                if "not found" in str(e).lower() or "404" in str(e):
                    await self._recover()
                    await self.push_frame(
                        TextFrame(text="I had a brief hiccup. Could you say that again?")
                    )

        if not interrupted:
            await self.push_frame(LLMFullResponseEndFrame())
        return full_response

    # ---- Router ----

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
            import json
            try:
                return json.loads(result)
            except Exception:
                return {"action": "none"}
        return result if isinstance(result, dict) else {"action": "none"}

    # ---- Workflow management ----

    TELCO_TOOLS = {"telco_account"}

    def _assistant_id_for_tool(self, tool: str) -> str:
        return "telco_workflow" if tool in self.TELCO_TOOLS else "workflow"

    async def _launch_workflow(self, tool: str, params: dict, context: str, original_question: str):
        """Launch a workflow agent, reusing an existing thread if available."""
        assistant_id = self._assistant_id_for_tool(tool)

        if assistant_id in self._persistent_threads:
            self._workflow_thread_id = self._persistent_threads[assistant_id]
            logger.info(
                f"[LangGraph] Reusing {assistant_id} thread "
                f"{self._workflow_thread_id[:8]} (auth/state preserved)"
            )
        else:
            thread = await self._client.threads.create()
            self._workflow_thread_id = thread["thread_id"]
            self._persistent_threads[assistant_id] = self._workflow_thread_id
            logger.info(
                f"[LangGraph] Created new {assistant_id} thread "
                f"{self._workflow_thread_id[:8]}"
            )

        self._workflow_original_question = original_question
        self._workflow_tool = tool
        self._workflow_assistant_id = assistant_id

        enriched_question = self._enrich_with_context(original_question)

        logger.info(
            f"[LangGraph] Launching {assistant_id} on "
            f"thread={self._workflow_thread_id[:8]} tool={tool} "
            f"question='{enriched_question[:100]}'"
        )

        self._state = AgentState.WORKFLOW_ACTIVE
        self._workflow_task = asyncio.create_task(
            self._run_workflow_background(enriched_question)
        )

    async def _run_workflow_background(self, message: str):
        """Background task: run workflow until interrupt or completion."""
        try:
            result = await self._client.runs.wait(
                self._workflow_thread_id,
                assistant_id=self._workflow_assistant_id,
                input={"message": message, "session_id": self._session_id},
            )
            await self._handle_workflow_result(result)
        except asyncio.CancelledError:
            logger.debug("[LangGraph] Workflow task cancelled")
            self._state = AgentState.IDLE
        except Exception as e:
            logger.error(f"[LangGraph] Workflow error: {e}", exc_info=True)
            self._state = AgentState.IDLE

    async def _resume_workflow_background(self, answer: str):
        """Background task: resume an interrupted workflow with user's answer."""
        try:
            result = await self._client.runs.wait(
                self._workflow_thread_id,
                assistant_id=self._workflow_assistant_id,
                command=Command(resume=answer),
            )
            await self._handle_workflow_result(result)
        except asyncio.CancelledError:
            logger.debug("[LangGraph] Workflow resume cancelled")
            self._state = AgentState.IDLE
        except Exception as e:
            logger.error(f"[LangGraph] Workflow resume error: {e}", exc_info=True)
            self._state = AgentState.IDLE

    async def _handle_workflow_result(self, result: dict | str):
        """Process workflow run result: either interrupt or completion."""
        if isinstance(result, dict) and result.get("__interrupt__"):
            interrupts = result["__interrupt__"]
            if interrupts:
                question = interrupts[0].get("value", {}).get("question", "Could you clarify?")
                logger.info(f"[LangGraph] Workflow interrupted, question: {question}")
                self._pending_question = question
                self._question_delivered = False
                self._fake_interrupt = False

                if not self._user_speaking and not self._cancelled:
                    await self._deliver_workflow_question(question)
                    if not self._cancelled:
                        self._question_delivered = True
                        self._state = AgentState.AWAITING_INPUT
                        logger.info("[LangGraph] State → AWAITING_INPUT (question delivered)")
                    else:
                        logger.info("[LangGraph] Question delivery cancelled, staying WORKFLOW_ACTIVE")
                else:
                    logger.info(
                        f"[LangGraph] Question deferred (user_speaking={self._user_speaking} "
                        f"cancelled={self._cancelled}), staying WORKFLOW_ACTIVE"
                    )
            return

        final_text = result if isinstance(result, str) else str(result)
        logger.info(f"[LangGraph] Workflow completed: {final_text[:100]}")

        if self._looks_like_question(final_text):
            logger.warning(
                f"[LangGraph] Workflow returned a question as result — "
                f"treating as follow-up question instead"
            )
            self._pending_question = final_text
            self._question_delivered = False
            self._fake_interrupt = True
            if not self._user_speaking and not self._cancelled:
                await self._deliver_workflow_question(final_text)
                if not self._cancelled:
                    self._question_delivered = True
                    self._state = AgentState.AWAITING_INPUT
                    logger.info("[LangGraph] State → AWAITING_INPUT (question-as-result, fake_interrupt)")
                else:
                    self._state = AgentState.WORKFLOW_ACTIVE
            else:
                self._state = AgentState.WORKFLOW_ACTIVE
                logger.info("[LangGraph] Question-as-result deferred")
            return

        self._state = AgentState.IDLE

        if not self._user_speaking and not self._cancelled:
            await self._deliver_workflow_result(final_text)
        else:
            self._pending_result = final_text
            logger.info(
                f"[LangGraph] Result deferred (user_speaking={self._user_speaking} "
                f"cancelled={self._cancelled})"
            )

    def _last_assistant_response(self) -> str:
        for entry in reversed(self._recent_history):
            if entry.get("role") == "assistant":
                return entry.get("content", "")
        return ""

    def _enrich_with_context(self, answer: str) -> str:
        """Append relevant conversation context so the workflow LLM knows
        what the caller mentioned earlier (e.g. destination country)."""
        user_turns = [
            e["content"] for e in self._recent_history
            if e.get("role") == "user" and e["content"] != answer
        ]
        if not user_turns:
            return answer
        context_snippet = " | ".join(user_turns[-3:])
        return (
            f"{answer}\n\n"
            f"[Additional context from earlier conversation: {context_snippet}]"
        )

    async def _deliver_workflow_question(self, question: str):
        """Deliver workflow's follow-up question through chitchat."""
        self._generating = True
        self._cancelled = False

        response = await self._stream_chitchat(
            question,
            input_type="workflow_question",
            question=question,
            original_request=self._workflow_original_question or "",
            last_response=self._last_assistant_response(),
        )

        if not self._cancelled:
            self._recent_history.append({"role": "assistant", "content": response})

        self._generating = False

    async def _deliver_workflow_result(self, result_text: str):
        """Deliver workflow's final result through chitchat."""
        self._generating = True
        self._cancelled = False

        response = await self._stream_chitchat(
            result_text,
            input_type="workflow_result",
            result=result_text,
            original_question=self._workflow_original_question or "",
            last_response=self._last_assistant_response(),
        )

        if not self._cancelled:
            self._recent_history.append({"role": "assistant", "content": response})

        self._generating = False
        self._pending_result = None
        self._workflow_original_question = None
        self._workflow_tool = None
        self._pending_question = None
        self._question_delivered = False

    # ---- Legacy job support ----

    def _pending_jobs_list(self) -> list[dict]:
        jobs = [
            {"job_id": jid, "tool": info["tool"], "status": "running"}
            for jid, info in self._pending_jobs.items()
        ]
        if self._state in (AgentState.WORKFLOW_ACTIVE, AgentState.AWAITING_INPUT):
            jobs.append({
                "job_id": f"workflow-{self._workflow_thread_id or 'active'}",
                "tool": getattr(self, "_workflow_tool", "workflow"),
                "status": "running",
                "original_question": self._workflow_original_question or "",
            })
        jobs.extend(self._delivered_jobs)
        return jobs

    async def _poll_legacy_jobs(self):
        while True:
            await asyncio.sleep(3)
            if not self._pending_jobs:
                continue

            completed = []
            for job_id, info in list(self._pending_jobs.items()):
                try:
                    run = await self._client.runs.get(
                        info["worker_thread_id"], info["run_id"],
                    )
                    if run["status"] in ("success", "error", "interrupted"):
                        completed.append(job_id)
                except Exception:
                    pass

            for job_id in completed:
                info = self._pending_jobs.pop(job_id)
                logger.info(f"Legacy job {job_id} completed")

    async def _recover(self):
        thread = await self._client.threads.create()
        self._chat_thread_id = thread["thread_id"]
        self._pending_jobs.clear()
        logger.warning("LangGraph session recovered")

    # ---- Main frame handler ----

    async def _handle_llm_frame(self, frame: Frame, direction: FrameDirection):
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
            if self._state != AgentState.AWAITING_INPUT:
                if len(stripped) < 3 or (word_count <= 2 and len(stripped) < 12):
                    logger.warning(f"[LangGraph] Ignoring fragment: '{stripped}'")
                    return

            logger.info(f"[LangGraph] User: {user_msg[:100]} (state={self._state.value})")

            self._generating = True
            self._cancelled = False

            # ---- State machine routing ----

            if self._state == AgentState.AWAITING_INPUT:
                self._recent_history.append({"role": "user", "content": user_msg})
                self._state = AgentState.WORKFLOW_ACTIVE
                self._pending_question = None

                if self._fake_interrupt:
                    enriched = self._enrich_with_context(user_msg)
                    logger.info(f"[LangGraph] Re-running workflow with answer: {enriched[:80]} (fake_interrupt)")
                    self._fake_interrupt = False
                    self._workflow_task = asyncio.create_task(
                        self._run_workflow_background(enriched)
                    )
                else:
                    logger.info(f"[LangGraph] Resuming workflow with: {user_msg[:80]}")
                    self._workflow_task = asyncio.create_task(
                        self._resume_workflow_background(user_msg)
                    )

                _ACKS = [
                    "Got it, one moment.",
                    "Thanks, let me check.",
                    "Perfect, one second.",
                    "Great, working on it.",
                    "Noted, just a moment.",
                ]
                ack = random.choice(_ACKS)
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(TextFrame(text=ack))
                await self.push_frame(LLMFullResponseEndFrame())
                self._recent_history.append({"role": "assistant", "content": ack})

            elif self._state == AgentState.WORKFLOW_ACTIVE:
                self._recent_history.append({"role": "user", "content": user_msg})

                if self._pending_question and not self._question_delivered:
                    logger.info("[LangGraph] Pending question ready — delivering directly, skipping chitchat")
                    await self._deliver_workflow_question(self._pending_question)
                    if not self._cancelled:
                        self._question_delivered = True
                        self._state = AgentState.AWAITING_INPUT
                        logger.info("[LangGraph] State → AWAITING_INPUT (deferred question delivered)")
                else:
                    logger.info("[LangGraph] Workflow active — chitchat only")
                    response = await self._stream_chitchat(user_msg)
                    if not self._cancelled:
                        self._recent_history.append({"role": "assistant", "content": response})

            else:
                self._recent_history.append({"role": "user", "content": user_msg})

                if self._pending_result:
                    logger.info("[LangGraph] Delivering deferred workflow result")
                    await self._deliver_workflow_result(self._pending_result)
                else:
                    response = await self._stream_chitchat(user_msg)
                    if not self._cancelled:
                        self._recent_history.append({"role": "assistant", "content": response})

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
                        params = router_result.get("params", {})
                        context = router_result.get("context_summary", "")
                        await self._launch_workflow(tool, params, context, user_msg)

            if len(self._recent_history) > 20:
                self._recent_history = self._recent_history[-20:]

            self._generating = False
            self._llm_task = None
            self._router_task = None

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

    # ---- Heuristics ----

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        """Detect when a workflow 'result' is actually a question the LLM
        produced without using the ask_user tool."""
        stripped = text.strip()
        if not stripped:
            return False
        if stripped.endswith("?"):
            word_count = len(stripped.split())
            if word_count < 30:
                return True
        question_starters = [
            "what is your", "could you", "can you", "please provide",
            "what's your", "may i have", "i need your",
        ]
        lower = stripped.lower()
        return any(lower.startswith(q) for q in question_starters)

    # ---- Fragment detection ----

    @staticmethod
    def _extract_user_text(frame: Frame) -> str:
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

    # ---- Frame processing ----

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._warmup_task = asyncio.create_task(self._ensure_initialized())
            logger.info("[LangGraph] Pre-warming client on pipeline start")
            await self.push_frame(frame, direction)

        elif isinstance(frame, (LLMMessagesFrame, LLMContextFrame)):
            user_text = self._extract_user_text(frame)
            if self._state != AgentState.AWAITING_INPUT and user_text and self._is_fragment(user_text):
                logger.warning(f"[LangGraph] Dropping fragment: '{user_text}'")
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
            if isinstance(frame, EndFrame):
                if self._poll_task:
                    self._poll_task.cancel()
                if self._workflow_task and not self._workflow_task.done():
                    self._workflow_task.cancel()
                if self._warmup_task and not self._warmup_task.done():
                    self._warmup_task.cancel()
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
