"""Generic dual-thread voice frontend for any LangGraph agent.

Architecture:
  - Main Thread:      runs the real agent (with tools, long tasks)
  - Secondary Thread: spun up when main is busy, chitchat only

The client (this FrameProcessor) manages thread lifecycle:
  1. User speaks + main idle   → send to main, stream response
  2. User speaks + main busy   → create/reuse secondary, stream chitchat
  3. Main finishes              → relay result through secondary, tear down
  4. Main hits interrupt        → relay question through secondary, collect answer

All LLM work runs in background asyncio tasks so Pipecat system frames
(interruption, cancel) flow through without blocking.
"""

import asyncio
import time
import uuid
from typing import Optional

from langgraph_sdk import get_client
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
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class GenericVoiceFrontend(FrameProcessor):
    """Dual-thread coordinator: routes speech to main or secondary thread."""

    def __init__(
        self,
        *,
        server_url: str = "http://127.0.0.1:2024",
        graph_name: str = "agent",
        ui_queue: Optional[asyncio.Queue] = None,
        tts: Optional[FrameProcessor] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._server_url = server_url
        self._graph_name = graph_name
        self._client = get_client(url=server_url)
        self._ui_queue = ui_queue
        self._tts = tts

        # Main thread state
        self._main_thread_id: Optional[str] = None
        self._main_busy = False
        self._main_task_description = ""
        self._main_bg_stream: Optional[asyncio.Task] = None
        self._main_result_buffer: Optional[str] = None

        # Secondary thread state
        self._secondary_thread_id: Optional[str] = None

        # Frame management
        self._is_streaming = False
        self._cancelled = False
        self._current_task: Optional[asyncio.Task] = None
        self._initialized = False
        self._merge_event = asyncio.Event()

        # Only the "active" thread pushes frames/SSE to the pipeline
        self._active_stream_thread: Optional[str] = None

        # Merge guard
        self._merge_in_progress = False
        self._pending_utterance: Optional[str] = None

        # Cross-thread store session — shared between main & secondary
        self._session_id = str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def _ensure_initialized(self):
        if self._initialized:
            return
        thread = await self._client.threads.create()
        self._main_thread_id = thread["thread_id"]
        self._initialized = True
        logger.info(
            f"[frontend] Initialized — main_thread={self._main_thread_id} "
            f"session={self._session_id}"
        )

    # ------------------------------------------------------------------
    # Stream a LangGraph run → push frames to TTS (if active)
    # ------------------------------------------------------------------

    async def _stream_to_tts(
        self,
        thread_id: str,
        input_payload: dict,
        *,
        assistant_id: Optional[str] = None,
    ) -> str:
        """Run a LangGraph invocation and stream text to TTS.

        Only pushes frames and SSE events if this thread is the active one.
        Always captures and returns the full response text.
        """
        aid = assistant_id or self._graph_name
        is_active = lambda: thread_id == self._active_stream_thread

        if is_active():
            if self._ui_queue:
                await self._ui_queue.put({"event": "start"})
            await self.push_frame(LLMFullResponseStartFrame())

        full_response = ""
        prev_len = 0
        current_msg_id = None

        try:
            async for chunk in self._client.runs.stream(
                thread_id,
                assistant_id=aid,
                input=input_payload,
                config={"configurable": {"session_id": self._session_id}},
                stream_mode="messages",
                multitask_strategy="interrupt",
            ):
                if self._cancelled and is_active():
                    break

                if chunk.event == "messages/partial":
                    data = chunk.data
                    if isinstance(data, list) and data:
                        msg = data[0]
                        content = msg.get("content", "")
                        msg_id = msg.get("id", "")
                        msg_type = msg.get("type", "")

                        if msg_type and msg_type not in (
                            "ai", "AIMessage", "AIMessageChunk",
                        ):
                            continue

                        if msg_id and msg_id != current_msg_id:
                            logger.debug(
                                f"[frontend] New message in stream: "
                                f"id={msg_id} (prev={current_msg_id})"
                            )
                            current_msg_id = msg_id
                            prev_len = 0

                        if content and len(content) > prev_len:
                            delta = content[prev_len:]
                            if is_active():
                                await self.push_frame(TextFrame(text=delta))
                                if self._ui_queue:
                                    await self._ui_queue.put(
                                        {"event": "token", "text": delta}
                                    )
                            prev_len = len(content)
                            full_response = content

        except asyncio.CancelledError:
            logger.debug("[frontend] Stream cancelled")
            raise
        except Exception as e:
            logger.error(f"[frontend] Stream error: {e}", exc_info=True)

        if not self._cancelled and is_active():
            await self.push_frame(LLMFullResponseEndFrame())
            if self._ui_queue:
                await self._ui_queue.put({"event": "end"})

        return full_response

    # ------------------------------------------------------------------
    # Secondary thread management
    # ------------------------------------------------------------------

    async def _ensure_secondary(self):
        if self._secondary_thread_id:
            return
        thread = await self._client.threads.create()
        self._secondary_thread_id = thread["thread_id"]
        logger.info(
            f"[frontend] Created secondary_thread={self._secondary_thread_id}"
        )

    async def _teardown_secondary(self):
        if self._secondary_thread_id:
            try:
                await self._client.threads.delete(self._secondary_thread_id)
            except Exception:
                pass
            logger.info(
                f"[frontend] Torn down secondary_thread={self._secondary_thread_id}"
            )
            self._secondary_thread_id = None

    # ------------------------------------------------------------------
    # Merge: relay main result through secondary, then switch back
    # ------------------------------------------------------------------

    async def _do_merge(self):
        result_text = self._main_result_buffer
        self._main_result_buffer = None
        self._merge_event.clear()

        if not result_text:
            self._main_busy = False
            self._main_bg_stream = None
            return

        self._merge_in_progress = True
        logger.info("[frontend] Starting merge — relaying result")

        try:
            if self._ui_queue:
                await self._ui_queue.put(
                    {"event": "status", "text": "Task complete", "state": "connected"}
                )

            if self._secondary_thread_id:
                relay_msg = (
                    f"[SYSTEM] The background task has just completed. "
                    f"Here is the result:\n\n"
                    f'"{result_text}"\n\n'
                    f"Deliver this result to the user naturally and conversationally. "
                    f"Transition smoothly from whatever you were just talking about. "
                    f"This is your final message on this thread."
                )

                self._active_stream_thread = self._secondary_thread_id
                response = await self._stream_to_tts(
                    self._secondary_thread_id,
                    {
                        "messages": [{"role": "user", "content": relay_msg}],
                        "mode": "chitchat",
                    },
                )
                self._active_stream_thread = None
                logger.info(f"[frontend] Merge delivered: {response[:100]!r}")
                await self._teardown_secondary()
            else:
                if self._ui_queue:
                    await self._ui_queue.put({"event": "start"})
                    await self._ui_queue.put({"event": "token", "text": result_text})
                    await self._ui_queue.put({"event": "end"})
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(TextFrame(text=result_text))
                await self.push_frame(LLMFullResponseEndFrame())

        except asyncio.CancelledError:
            logger.warning("[frontend] Merge was cancelled — falling back to direct delivery")
            if self._ui_queue:
                await self._ui_queue.put({"event": "start"})
                await self._ui_queue.put({"event": "token", "text": result_text})
                await self._ui_queue.put({"event": "end"})
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(TextFrame(text=result_text))
            await self.push_frame(LLMFullResponseEndFrame())
            await self._teardown_secondary()
        finally:
            self._main_busy = False
            self._main_task_description = ""
            self._main_bg_stream = None
            self._merge_in_progress = False
            if self._ui_queue:
                await self._ui_queue.put({"event": "agent", "agent": "primary"})
            logger.info("[frontend] Merge complete — back to single-thread mode")

            pending = self._pending_utterance
            self._pending_utterance = None
            if pending:
                logger.info(
                    f"[frontend] Processing buffered utterance: {pending[:80]!r}"
                )
                self._current_task = asyncio.create_task(
                    self._handle_user_utterance(pending)
                )

    # ------------------------------------------------------------------
    # Main user utterance handler
    # ------------------------------------------------------------------

    async def _handle_user_utterance(self, text: str):
        try:
            await self._ensure_initialized()
            self._cancelled = False

            if self._main_result_buffer:
                logger.info(
                    "[frontend] Main result ready — merging before handling"
                )
                await self._do_merge()

            if not self._main_busy:
                # --- SINGLE THREAD MODE: send to main ---
                logger.info(f"[frontend] → Main thread: {text[:80]!r}")
                self._is_streaming = True
                self._active_stream_thread = self._main_thread_id
                if self._ui_queue:
                    await self._ui_queue.put({"event": "agent", "agent": "primary"})

                payload = {
                    "messages": [{"role": "user", "content": text}],
                    "mode": "normal",
                }

                response_task = asyncio.create_task(
                    self._stream_to_tts(self._main_thread_id, payload)
                )

                try:
                    response = await asyncio.wait_for(
                        asyncio.shield(response_task), timeout=8.0
                    )
                    self._is_streaming = False
                    self._active_stream_thread = None
                    logger.info(
                        f"[frontend] Main responded quickly: {response[:80]!r}"
                    )
                    return

                except asyncio.TimeoutError:
                    logger.info(
                        "[frontend] Main taking long — switching to dual-thread"
                    )
                    self._main_busy = True
                    self._main_task_description = text
                    self._is_streaming = False

                    # Close the current response frame cleanly
                    await self.push_frame(LLMFullResponseEndFrame())
                    if self._ui_queue:
                        await self._ui_queue.put({"event": "end"})

                    # Stop main stream from pushing any more frames
                    self._active_stream_thread = None

                    if self._ui_queue:
                        await self._ui_queue.put(
                            {
                                "event": "status",
                                "text": "Working on task...",
                                "state": "busy",
                            }
                        )
                        await self._ui_queue.put(
                            {"event": "agent", "agent": "chitchat"}
                        )

                    self._main_bg_stream = asyncio.create_task(
                        self._wait_for_main_response(response_task)
                    )

            else:
                # --- DUAL THREAD MODE: send to secondary ---
                await self._ensure_secondary()
                status = f"Working on: {self._main_task_description}"

                logger.info(f"[frontend] → Secondary thread: {text[:80]!r}")
                self._is_streaming = True
                self._active_stream_thread = self._secondary_thread_id
                if self._ui_queue:
                    await self._ui_queue.put({"event": "agent", "agent": "chitchat"})

                payload = {
                    "messages": [{"role": "user", "content": text}],
                    "mode": "chitchat",
                    "main_thread_status": status,
                }

                response = await self._stream_to_tts(
                    self._secondary_thread_id, payload
                )
                self._is_streaming = False
                self._active_stream_thread = None

                logger.info(f"[frontend] Secondary responded: {response[:80]!r}")

                if self._main_result_buffer:
                    await self._do_merge()

        except asyncio.CancelledError:
            logger.debug("[frontend] Utterance handler cancelled")
            self._is_streaming = False
            self._active_stream_thread = None
        except Exception as e:
            logger.error(f"[frontend] Error: {e}", exc_info=True)
            self._is_streaming = False
            self._active_stream_thread = None
            try:
                error_text = "Sorry, something went wrong. Could you try again?"
                if self._ui_queue:
                    await self._ui_queue.put({"event": "start"})
                    await self._ui_queue.put({"event": "token", "text": error_text})
                    await self._ui_queue.put({"event": "end"})
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(TextFrame(text=error_text))
                await self.push_frame(LLMFullResponseEndFrame())
            except Exception:
                pass

    async def _wait_for_main_response(self, response_task: asyncio.Task):
        """Wait for a still-running main thread stream to complete."""
        try:
            full_response = await response_task
            if full_response:
                logger.info(
                    f"[frontend] Main bg completed — "
                    f"result_preview={full_response[:100]!r}"
                )
                self._main_result_buffer = full_response
                self._merge_event.set()

                if not self._is_streaming:
                    await self._do_merge()
        except asyncio.CancelledError:
            logger.debug("[frontend] Main bg stream cancelled")
        except Exception as e:
            logger.error(f"[frontend] Main bg error: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Cancel / cleanup
    # ------------------------------------------------------------------

    def _cancel_current(self):
        self._cancelled = True
        self._active_stream_thread = None
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        self._current_task = None
        self._is_streaming = False
        if self._tts and hasattr(self._tts, "_cancel_all_tts"):
            self._tts._cancel_all_tts("Cancelled from GenericVoiceFrontend")
        elif self._tts:
            self._tts._interrupted = True
        logger.debug("[frontend] Cancelled current generation")

    async def _recover(self):
        thread = await self._client.threads.create()
        self._main_thread_id = thread["thread_id"]
        self._main_busy = False
        self._main_result_buffer = None
        self._active_stream_thread = None
        await self._teardown_secondary()
        logger.warning("[frontend] Session recovered with new main thread")

    # ------------------------------------------------------------------
    # Frame processing (Pipecat integration)
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            asyncio.create_task(self._ensure_initialized())
            logger.info("[frontend] Pre-warming on pipeline start")
            await self.push_frame(frame, direction)

        elif isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if not text or len(text) < 3:
                return

            if self._merge_in_progress:
                logger.info(
                    f"[frontend] Merge in progress — buffering utterance: "
                    f"{text[:80]!r}"
                )
                self._pending_utterance = text
                return

            if self._current_task and not self._current_task.done():
                self._current_task.cancel()
                logger.info("[frontend] Cancelled in-flight task for new utterance")

            self._cancel_current()
            self._cancelled = False
            await self.push_frame(InterruptionFrame())

            self._current_task = asyncio.create_task(
                self._handle_user_utterance(text)
            )

        elif isinstance(frame, InterimTranscriptionFrame):
            pass

        elif isinstance(frame, UserStartedSpeakingFrame):
            await self.push_frame(frame, direction)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self.push_frame(frame, direction)

        elif isinstance(frame, (CancelFrame, EndFrame)):
            self._cancel_current()
            if isinstance(frame, EndFrame):
                if self._main_bg_stream and not self._main_bg_stream.done():
                    self._main_bg_stream.cancel()
                await self._teardown_secondary()
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
