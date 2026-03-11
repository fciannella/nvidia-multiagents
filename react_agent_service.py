"""Pipecat FrameProcessor for the React-agent architecture.

Thin coordinator between three components:
  1. Starter matcher  — instant pre-synth audio on user utterance
  2. Front agent      — conversational ReAct agent (LangGraph)
  3. Executor agent   — background tool runner (LangGraph)

On user utterance:
  - Play matching pre-synthesised starter audio instantly
  - Resume executor if there is a pending interrupt
  - Drain queued executor updates into the front agent payload
  - Invoke front agent, stream the continuation after the starter
  - After the run, check the Store for new delegation requests
  - Start executor background tasks for any new delegations

On executor event (interrupt / completion):
  - Queue the update
  - If idle, invoke front agent immediately to deliver it
  - Otherwise deliver on the next gap

process_frame returns immediately; all LLM work runs in background
asyncio tasks so system frames (interruption, cancel) flow through.
"""

import asyncio
import os
import time
import uuid
from typing import Optional

import httpx
from dotenv import load_dotenv
from langgraph_sdk import get_client
from langgraph_sdk.schema import Command
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartFrame,
    StartInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

load_dotenv()


class ReactAgentService(FrameProcessor):
    """Coordinates starter audio, front agent, and executor."""

    def __init__(
        self,
        *,
        server_url: str = "http://127.0.0.1:2024",
        ui_queue: Optional[asyncio.Queue] = None,
        tts_server: str = "http://192.168.7.163:8100",
        voice: str = "hekate-en",
        language: str = "English",
        tts: Optional["FrameProcessor"] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._server_url = server_url
        self._client = get_client(url=server_url)
        self._session_id = uuid.uuid4().hex[:8]
        self._ui_queue = ui_queue
        self._tts_server = tts_server
        self._voice = voice
        self._language = language
        self._tts = tts

        # LangGraph threads
        self._front_thread_id: Optional[str] = None
        self._initialized = False
        self._warmup_task: Optional[asyncio.Task] = None

        # Executor state
        self._active_tasks: dict[str, dict] = {}
        self._executor_updates: list[dict] = []
        self._pending_interrupt: Optional[dict] = None
        self._executor_bg_tasks: dict[str, asyncio.Task] = {}

        # Frame management
        self._is_streaming = False
        self._cancelled = False
        self._llm_task: Optional[asyncio.Task] = None
        self._current_task: Optional[asyncio.Task] = None
        self._last_front_response: str = ""
        self._recent_turns: list[dict] = []  # last few turns for filler context

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def _ensure_initialized(self):
        if self._initialized:
            return
        thread = await self._client.threads.create()
        self._front_thread_id = thread["thread_id"]
        self._initialized = True

        logger.info(
            f"[react] session={self._session_id} "
            f"front_thread={self._front_thread_id}"
        )

    # ------------------------------------------------------------------
    # Filler LLM (replaces pre-synth starter matcher)
    # ------------------------------------------------------------------

    _FILLER_SYSTEM = (
        "You are a phone agent. Output a SHORT acknowledgment only.\n\n"
        "Pick ONE of these patterns based on what the caller said:\n\n"
        "GREETING → greet back:\n"
        "  'Hey there!' / 'Oh hey, good morning!' / 'Hi, doing great!'\n\n"
        "REQUEST → confirm you heard:\n"
        "  'Sure thing.' / 'Got it, one sec.' / 'Absolutely.'\n\n"
        "INFO (number, code, name) → confirm receipt:\n"
        "  'Perfect.' / 'Alright.' / 'Okay, thanks.'\n\n"
        "QUESTION → acknowledge:\n"
        "  'Good question.' / 'Let me check.' / 'One moment.'\n\n"
        "RULES:\n"
        "- 2-6 words MAXIMUM\n"
        "- ONLY pick from the patterns above or very close variations\n"
        "- NEVER answer questions, give information, or explain anything\n"
        "- NEVER mention topics, places, names, or details from what they said\n"
        "- NEVER tell jokes, stories, or make up facts\n"
        "- Check your recent messages and don't repeat the same phrase"
    )

    def _build_filler_messages(self, utterance: str) -> list[dict]:
        """Build message list for the filler LLM including recent history."""
        msgs = [{"role": "system", "content": self._FILLER_SYSTEM}]
        for turn in self._recent_turns[-6:]:
            msgs.append(turn)
        msgs.append({"role": "user", "content": utterance})
        return msgs

    def _record_turn(self, role: str, content: str):
        """Track recent turns for filler context (keep last 8)."""
        self._recent_turns.append({"role": role, "content": content})
        if len(self._recent_turns) > 8:
            self._recent_turns = self._recent_turns[-8:]

    async def _play_starter(self, utterance: str) -> Optional[str]:
        """Call the filler LLM for a quick contextual acknowledgment.

        Fires on every user turn to minimize perceived latency. The generated
        text is spoken via TTS immediately AND injected into the front agent's
        conversation history so the front agent continues seamlessly.

        Currently disabled — set FILLER_ENABLED=1 to re-enable.
        """
        if not os.environ.get("FILLER_ENABLED"):
            return None
        filler_url = os.environ.get("FILLER_LLM_BASE_URL", "http://192.168.6.55:8444/v1")
        filler_model = os.environ.get("FILLER_LLM_MODEL", "unsloth/Ministral-3-3B-Instruct-2512-FP8")

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.post(
                    f"{filler_url}/chat/completions",
                    json={
                        "model": filler_model,
                        "messages": self._build_filler_messages(utterance),
                        "max_tokens": 15,
                        "temperature": 0.5,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                filler_text = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
        except Exception as e:
            logger.warning(f"[react] Filler LLM failed ({e}), skipping")
            return None

        if not filler_text:
            return None

        filler_text = filler_text.replace("\n", " ").replace("*", "").replace("#", "").strip()
        first_sentence = filler_text.split(".")[0] if "." in filler_text else filler_text
        if len(first_sentence.split()) > 8:
            first_sentence = " ".join(first_sentence.split()[:6])
        filler_text = first_sentence.strip().rstrip(".") + "."

        if not filler_text or filler_text == ".":
            return None

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[react] event=filler_generated latency={latency_ms:.0f}ms "
            f"text={filler_text!r}"
        )

        if self._ui_queue:
            await self._ui_queue.put({"event": "start"})
            await self._ui_queue.put({"event": "token", "text": filler_text + " "})

        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(TextFrame(text=filler_text))
        await self.push_frame(LLMFullResponseEndFrame())

        return filler_text

    # ------------------------------------------------------------------
    # Front agent invocation
    # ------------------------------------------------------------------

    async def _stream_front_agent(
        self, payload: dict, *, ui_started: bool = False,
    ) -> str:
        """Invoke front agent with streaming, push TextFrames to TTS.

        Args:
            ui_started: If True, a UI bubble is already open (starter was played).
        """
        if not ui_started and self._ui_queue:
            await self._ui_queue.put({"event": "start"})

        await self.push_frame(LLMFullResponseStartFrame())

        full_response = ""
        prev_len = 0

        try:
            async for chunk in self._client.runs.stream(
                self._front_thread_id,
                assistant_id="front_agent",
                input=payload,
                stream_mode="messages",
                multitask_strategy="interrupt",
            ):
                if self._cancelled:
                    break

                if chunk.event == "messages/partial":
                    data = chunk.data
                    if isinstance(data, list) and data:
                        content = data[0].get("content", "")
                        if content and len(content) > prev_len:
                            delta = content[prev_len:]
                            await self.push_frame(TextFrame(text=delta))
                            if self._ui_queue:
                                await self._ui_queue.put(
                                    {"event": "token", "text": delta}
                                )
                            prev_len = len(content)
                            full_response = content

        except asyncio.CancelledError:
            logger.debug("[react] Front agent stream cancelled")
            raise
        except Exception as e:
            logger.error(f"[react] Front agent error: {e}", exc_info=True)
            if "not found" in str(e).lower() or "404" in str(e):
                await self._recover()

        if not self._cancelled:
            await self.push_frame(LLMFullResponseEndFrame())
            if self._ui_queue:
                await self._ui_queue.put({"event": "end"})

        self._last_front_response = full_response
        return full_response

    # ------------------------------------------------------------------
    # Delegation detection (reads Store after front agent run)
    # ------------------------------------------------------------------

    async def _check_delegations(self):
        """Check the Store for new delegation requests from front agent."""
        try:
            response = await self._client.store.search_items(
                ["delegations", self._session_id],
            )
            # search_items returns SearchItemsResponse; items are in .items (not iterating response)
            if isinstance(response, dict):
                items = response.get("items", [])
            else:
                items = getattr(response, "items", None) or []
            if items:
                logger.info(f"[react] Delegation check: {len(items)} item(s) from store")
        except Exception as e:
            logger.warning(f"[react] Store search failed: {e}")
            return

        for item in items:
            val = item.value if hasattr(item, "value") else (item.get("value") if isinstance(item, dict) else {})
            status = val.get("status", "")
            if status != "requested":
                continue

            task_id = val.get("task_id", item.key if hasattr(item, "key") else getattr(item, "key", "?"))
            description = val.get("description", "")
            logger.info(
                f"[react] event=delegation_started task_id={task_id} "
                f"description={description[:60]!r}"
            )

            await self._client.store.put_item(
                ["delegations", self._session_id],
                key=task_id,
                value={**val, "status": "running"},
            )

            await self._start_executor(task_id, description)

    # ------------------------------------------------------------------
    # Executor lifecycle
    # ------------------------------------------------------------------

    async def _start_executor(self, task_id: str, description: str):
        """Create an executor thread and start a background run."""
        thread = await self._client.threads.create()
        thread_id = thread["thread_id"]

        self._active_tasks[task_id] = {
            "task_id": task_id,
            "description": description,
            "status": "running",
            "thread_id": thread_id,
        }

        logger.info(
            f"[react] event=executor_started task_id={task_id} "
            f"thread_id={thread_id[:8]} description={description[:60]!r}"
        )

        bg = asyncio.create_task(
            self._run_executor_bg(task_id, thread_id, description)
        )
        self._executor_bg_tasks[task_id] = bg

    async def _run_executor_bg(
        self, task_id: str, thread_id: str, description: str
    ):
        """Background: run executor until interrupt or completion."""
        try:
            result = await self._client.runs.wait(
                thread_id,
                assistant_id="executor",
                input={"message": description, "session_id": self._session_id},
            )
            await self._handle_executor_result(task_id, thread_id, result)
        except asyncio.CancelledError:
            logger.debug(f"[react] Executor {task_id} cancelled")
        except Exception as e:
            logger.error(f"[react] Executor {task_id} error: {e}", exc_info=True)
            self._executor_updates.append({
                "type": "error",
                "task_id": task_id,
                "error": str(e),
            })
            if task_id in self._active_tasks:
                self._active_tasks[task_id]["status"] = "failed"
            try:
                await self._client.store.put_item(
                    ["delegations", self._session_id],
                    key=task_id,
                    value={"status": "failed", "task_id": task_id},
                )
            except Exception:
                pass
            await self._deliver_updates_if_idle()

    async def _resume_executor_bg(self, task_id: str, thread_id: str, answer: str):
        """Background: resume an interrupted executor with the user's answer."""
        try:
            result = await self._client.runs.wait(
                thread_id,
                assistant_id="executor",
                command=Command(resume=answer),
            )
            await self._handle_executor_result(task_id, thread_id, result)
        except asyncio.CancelledError:
            logger.debug(f"[react] Executor resume {task_id} cancelled")
        except Exception as e:
            logger.error(f"[react] Executor resume {task_id} error: {e}", exc_info=True)

    async def _handle_executor_result(
        self, task_id: str, thread_id: str, result
    ):
        """Process executor run result: interrupt or completion."""
        if isinstance(result, dict) and result.get("__interrupt__"):
            interrupts = result["__interrupt__"]
            if interrupts:
                question = (
                    interrupts[0].get("value", {}).get("question", "Could you clarify?")
                )
                logger.info(
                    f"[react] event=executor_interrupted task_id={task_id} "
                    f"question={question[:120]!r}"
                )

                # Do NOT set _pending_interrupt here — set it only after we've delivered
                # and spoken the question, so the next user message is the actual answer.
                self._executor_updates.append({
                    "type": "question",
                    "task_id": task_id,
                    "thread_id": thread_id,
                    "question": question,
                })
                if task_id in self._active_tasks:
                    self._active_tasks[task_id]["status"] = "waiting_for_input"

                await self._deliver_updates_if_idle()
            return

        final_text = result if isinstance(result, str) else str(result)
        logger.info(
            f"[react] event=executor_completed task_id={task_id} "
            f"result_preview={final_text[:100]!r}"
        )

        self._executor_updates.append({
            "type": "result",
            "task_id": task_id,
            "result": final_text,
        })
        if task_id in self._active_tasks:
            self._active_tasks[task_id]["status"] = "completed"
        self._pending_interrupt = None

        try:
            await self._client.store.put_item(
                ["delegations", self._session_id],
                key=task_id,
                value={"status": "completed", "task_id": task_id},
            )
        except Exception as e:
            logger.warning(f"[react] Could not update store delegation status: {e}")

        await self._deliver_updates_if_idle()

    # ------------------------------------------------------------------
    # Update delivery
    # ------------------------------------------------------------------

    async def _deliver_updates_if_idle(self):
        """If not streaming and user not speaking, deliver queued updates."""
        if self._is_streaming or self._cancelled:
            return
        if not self._executor_updates:
            return

        updates = list(self._executor_updates)
        self._executor_updates.clear()

        # If the front agent's last response already asked a question and the
        # executor is also asking a question, skip the LLM delivery — the user
        # already heard the question. Just wire up the interrupt so their next
        # answer goes to the executor.
        has_question_update = any(u.get("type") == "question" for u in updates)
        if has_question_update and self._last_front_response and "?" in self._last_front_response:
            for u in updates:
                if u.get("type") == "question":
                    tid = u.get("task_id", "")
                    q = u.get("question", "")
                    self._pending_interrupt = {
                        "task_id": tid,
                        "thread_id": u.get("thread_id", ""),
                        "question": q,
                    }
                    logger.info(
                        f"[react] event=skipped_duplicate_question task_id={tid} "
                        f"question={q[:80]!r} reason='front agent already asked'"
                    )
                    break
            return

        payload = {
            "session_id": self._session_id,
            "type": "executor_update",
            "message": "",
            "active_tasks": await self._active_tasks_list(),
            "executor_updates": updates,
        }

        logger.info(
            f"[react] event=delivering_updates count={len(updates)} "
            f"types={[u.get('type') for u in updates]}"
        )
        self._is_streaming = True
        self._cancelled = False
        try:
            response = await self._stream_front_agent(payload)
            if not self._cancelled:
                await self._check_delegations()
                for u in updates:
                    if u.get("type") == "question":
                        tid = u.get("task_id", "")
                        q = u.get("question", "")
                        self._pending_interrupt = {
                            "task_id": tid,
                            "thread_id": u.get("thread_id", ""),
                            "question": q,
                        }
                        logger.info(
                            f"[react] event=delivered_question task_id={tid} "
                            f"question={q[:80]!r}"
                        )
                        break
                for u in updates:
                    if u.get("type") == "result":
                        tid = u.get("task_id", "")
                        r = u.get("result", "")[:80]
                        logger.info(
                            f"[react] event=delivered_result task_id={tid} "
                            f"result_preview={r!r}"
                        )
        finally:
            self._is_streaming = False

        for u in updates:
            if u.get("type") == "result":
                self._active_tasks.pop(u.get("task_id", ""), None)

    # ------------------------------------------------------------------
    # Main user utterance handler
    # ------------------------------------------------------------------

    async def _handle_user_utterance(self, user_msg: str):
        """Full flow: starter → resume executor → invoke front agent → check delegations."""
        try:
            await self._ensure_initialized()

            logger.info(f"[react] User: {user_msg[:100]}")
            self._is_streaming = True
            self._cancelled = False

            # 0. Record the user turn for filler history
            self._record_turn("user", user_msg)

            # 1. Starter audio
            starter_text = await self._play_starter(user_msg)

            # 2. Resume executor if pending interrupt
            answer_forwarded = False
            if self._pending_interrupt:
                task_id = self._pending_interrupt["task_id"]
                thread_id = self._pending_interrupt["thread_id"]
                self._pending_interrupt = None
                answer_forwarded = True

                if task_id in self._active_tasks:
                    self._active_tasks[task_id]["status"] = "running"

                bg = asyncio.create_task(
                    self._resume_executor_bg(task_id, thread_id, user_msg)
                )
                self._executor_bg_tasks[task_id] = bg
                logger.info(
                    f"[react] event=forwarded_answer task_id={task_id} "
                    f"answer_preview={user_msg[:80]!r}"
                )

            # 3. Drain executor updates
            updates = list(self._executor_updates)
            self._executor_updates.clear()

            # 4. Build front agent payload
            payload = {
                "session_id": self._session_id,
                "type": "user_message",
                "message": user_msg,
                "starter_played": starter_text or "",
                "active_tasks": await self._active_tasks_list(),
                "executor_updates": updates,
                "answer_forwarded": answer_forwarded,
            }

            # 5. Stream front agent response (starter already opened a UI bubble)
            response = await self._stream_front_agent(
                payload, ui_started=bool(starter_text),
            )

            # Record combined assistant response for filler history
            full_said = f"{starter_text} {response}".strip() if starter_text else response
            if full_said:
                self._record_turn("assistant", full_said)

            if self._cancelled:
                logger.info("[react] Skipping delegation check — cancelled")
                self._is_streaming = False
                return

            # 6. Check Store for new delegations
            logger.info("[react] Front agent done, checking delegations...")
            await self._check_delegations()

            self._is_streaming = False

            # 7. Deliver any updates that arrived during streaming
            if self._executor_updates:
                await self._deliver_updates_if_idle()

            # Clean up completed tasks
            completed = [
                tid for tid, t in self._active_tasks.items()
                if t["status"] == "completed"
            ]
            for tid in completed:
                self._active_tasks.pop(tid, None)

        except asyncio.CancelledError:
            logger.debug("[react] User utterance handler cancelled")
            self._is_streaming = False
        except Exception as e:
            logger.error(f"[react] Error: {e}", exc_info=True)
            self._is_streaming = False
            try:
                error_text = "Sorry, could you try that again?"
                if self._ui_queue:
                    await self._ui_queue.put({"event": "start"})
                    await self._ui_queue.put({"event": "token", "text": error_text})
                    await self._ui_queue.put({"event": "end"})
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(TextFrame(text=error_text))
                await self.push_frame(LLMFullResponseEndFrame())
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _active_tasks_list(self) -> list[dict]:
        """Build the active tasks list, enriched with live progress from the Store."""
        tasks = []
        for t in self._active_tasks.values():
            info = {
                "task_id": t["task_id"],
                "description": t["description"],
                "status": t["status"],
            }
            tasks.append(info)

        if not tasks:
            return tasks

        try:
            response = await self._client.store.search_items(
                ["jobs", self._session_id],
            )
            if isinstance(response, dict):
                items = response.get("items", [])
            else:
                raw = getattr(response, "items", None)
                items = raw if isinstance(raw, list) else []
            best = None
            for item in items:
                val = item.value if hasattr(item, "value") else (item.get("value") if isinstance(item, dict) else {})
                if val.get("status") == "running":
                    step = val.get("step", 0)
                    if best is None or step > best.get("step", 0):
                        best = val
            logger.info(f"[react] Job progress query: {len(items)} item(s) from store")
            if best:
                progress_msg = best.get("message", "working...")
                step = best.get("step", "?")
                total = best.get("total", "?")
                progress_str = f"step {step}/{total}: {progress_msg}"
                logger.info(f"[react] event=progress_found {progress_str}")
                for info in tasks:
                    if info["status"] == "running":
                        info["progress"] = progress_str
            else:
                logger.info("[react] No running job progress found in store")
        except Exception as e:
            logger.warning(f"[react] Could not fetch job progress: {e}")

        return tasks

    def _cancel_current(self):
        self._cancelled = True
        if self._llm_task and not self._llm_task.done():
            self._llm_task.cancel()
        self._llm_task = None
        self._is_streaming = False
        if self._tts and hasattr(self._tts, '_cancel_all_tts'):
            self._tts._cancel_all_tts("Cancelled from ReactAgentService")
        elif self._tts:
            self._tts._interrupted = True
        logger.debug("[react] Cancelled current generation")

    async def _recover(self):
        thread = await self._client.threads.create()
        self._front_thread_id = thread["thread_id"]
        logger.warning("[react] Session recovered with new thread")

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._warmup_task = asyncio.create_task(self._ensure_initialized())
            logger.info("[react] Pre-warming on pipeline start")
            await self.push_frame(frame, direction)

        elif isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if not text:
                return

            if len(text) < 3 and not self._pending_interrupt:
                logger.warning(f"[react] Dropping fragment: '{text}'")
                return

            if self._current_task and not self._current_task.done():
                self._current_task.cancel()
                logger.info("[react] Cancelled in-flight task for new utterance")

            self._cancel_current()
            self._cancelled = False
            await self.push_frame(InterruptionFrame())

            self._current_task = asyncio.create_task(
                self._handle_user_utterance(text)
            )
            self._llm_task = self._current_task

        elif isinstance(frame, InterimTranscriptionFrame):
            pass

        elif isinstance(frame, StartInterruptionFrame):
            logger.info("[react] StartInterruptionFrame — stopping audio only")
            self._is_streaming = False
            if self._tts and hasattr(self._tts, '_cancel_all_tts'):
                self._tts._cancel_all_tts("StartInterruptionFrame from VAD")
            elif self._tts:
                self._tts._interrupted = True
            await self.push_frame(frame, direction)

        elif isinstance(frame, UserStartedSpeakingFrame):
            logger.debug("[react] UserStartedSpeakingFrame")
            await self.push_frame(frame, direction)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.debug("[react] UserStoppedSpeakingFrame")
            await self.push_frame(frame, direction)

        elif isinstance(frame, (CancelFrame, EndFrame)):
            self._cancel_current()
            if isinstance(frame, EndFrame):
                for bg in self._executor_bg_tasks.values():
                    if not bg.done():
                        bg.cancel()
                if self._warmup_task and not self._warmup_task.done():
                    self._warmup_task.cancel()
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
