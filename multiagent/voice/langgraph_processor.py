"""Pipecat FrameProcessor bridging STT to the LangGraph orchestrator.

Receives TranscriptionFrames from STT, calls the LangGraph server via
the SDK using stream_mode=["messages","updates"], and pushes
VoiceSwitchFrame + TextFrame pairs for each agent response.

STREAMING STRATEGY:
  - stream_mode="messages" gives us token-by-token AI message chunks
    with metadata including which graph node emitted them.
  - We filter for tokens from invoke_agent nodes and push them
    downstream immediately so the TTS sentence buffer can fire as
    soon as a sentence boundary is detected.
  - With ReAct tool calling, a single node may produce multiple LLM
    calls (before/after tool execution). Each call gets a separate
    run_id. We track run_id changes to reset content accumulation.

EXECUTOR LIFECYCLE:
  Agents call tools via structured tool calling (bind_tools / ReAct loop).
  Fast tools execute inline within the graph. Slow tools use launchers
  that return immediately; their names appear in ``tool_launches`` in the
  graph state update. The processor reads these and starts background
  executors for the actual long-running work.
"""

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from multiagent.voice.frames import VoiceSwitchFrame

try:
    from langgraph_sdk import get_client as get_lg_client
    from langgraph_sdk.schema import Command
except ModuleNotFoundError as exc:
    raise ImportError("langgraph-sdk is required: pip install langgraph-sdk") from exc


INVOKE_NODES = {"invoke_agent"}


@dataclass
class VoiceMap:
    """Maps agent speaker IDs to TTS voice names.

    Works with any number of agents — backed by a simple dict.
    Legacy fields (data_scientist, it) are kept for backward compat
    but the dict-based ``voices`` is the canonical source.
    """

    voices: dict[str, str] = field(default_factory=dict)

    data_scientist: str = field(
        default_factory=lambda: os.getenv("MA_DS_VOICE", "qwen3-data-scientist-demo-en-en")
    )
    it: str = field(
        default_factory=lambda: os.getenv("MA_IT_VOICE", "qwen3-male-it-expert-en")
    )

    def __post_init__(self):
        if not self.voices:
            self.voices = {
                "data_scientist": self.data_scientist,
                "it": self.it,
            }

    def get(self, speaker: str) -> str:
        if speaker in self.voices:
            return self.voices[speaker]
        if self.voices:
            return next(iter(self.voices.values()))
        return self.data_scientist

    def set(self, speaker: str, voice: str):
        self.voices[speaker] = voice


class LangGraphProcessor(FrameProcessor):
    """Bridges STT to the LangGraph orchestrator with token streaming
    and per-agent executor lifecycle management.
    """

    def __init__(
        self,
        *,
        langgraph_url: Optional[str] = None,
        graph_name: str = "orchestrator",
        voice_map: Optional[VoiceMap] = None,
        scenario_id: str = "ecommerce",
        tts=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._url = langgraph_url or os.getenv("LANGGRAPH_URL", "http://localhost:2024")
        self._graph_name = graph_name
        self._voice_map = voice_map or VoiceMap()
        self._scenario_id = scenario_id
        self._tts = tts
        self._client = None
        self._thread_id: Optional[str] = None
        self._current_task: Optional[asyncio.Task] = None
        self._responding = False
        self._cancelled = False
        self._is_streaming = False
        self._session_id = uuid.uuid4().hex[:8]

        # Message combining: when user self-interrupts before routing finishes,
        # combine the messages so the first one's context isn't lost.
        self._last_user_text: str = ""
        self._last_user_time: float = 0.0
        self._agent_has_spoken: bool = False

        # Executor state (per-agent)
        self._executor_threads: dict[str, str] = {}
        self._executor_bg_tasks: dict[str, asyncio.Task] = {}
        self._executor_updates: list[dict] = []
        self._pending_interrupt: Optional[dict] = None
        self._active_tasks: dict[str, dict] = {}

        # Track executors started from the current (possibly interrupted) response.
        # If the user interrupts before TTS finishes, these get cancelled.
        self._current_response_executor_ids: list[str] = []
        self._stream_completed_at: float = 0.0
        self._stream_text_len: int = 0

        # Background poller that retries delivery when executor results arrive
        # while the pipeline is busy.
        self._update_poller: Optional[asyncio.Task] = None

    async def _ensure_client(self):
        if self._client is None:
            self._client = get_lg_client(url=self._url)
            thread = await self._client.threads.create()
            self._thread_id = thread["thread_id"]
            logger.info(
                f"LangGraph client ready — url={self._url} "
                f"thread={self._thread_id[:8]}… session={self._session_id}"
            )
            if self._update_poller is None:
                self._update_poller = asyncio.create_task(
                    self._poll_executor_updates()
                )

    def _cancel_current(self):
        """Stop all in-flight generation: LLM task + TTS chain."""
        self._cancelled = True
        self._is_streaming = False
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        self._current_task = None
        if self._tts and hasattr(self._tts, "_cancel_all_tts"):
            self._tts._cancel_all_tts("Cancelled from LangGraphProcessor")
        elif self._tts:
            self._tts._interrupted = True

        # Executors are only started when tools_allowed=true (explicit user
        # request), so interrupting the agent's speech should NOT cancel the
        # background work the user asked for.
        self._current_response_executor_ids.clear()

        logger.debug("[multiagent] Cancelled current generation")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if not text:
                return

            if len(text) < 3:
                logger.warning(f"[multiagent] Dropping fragment: '{text}'")
                return

            logger.info(f"User said: {text}")

            now = time.perf_counter()
            should_combine = (
                self._current_task
                and not self._current_task.done()
                and not self._agent_has_spoken
                and self._last_user_text
                and (now - self._last_user_time) < 4.0
            )

            self._cancel_current()
            self._cancelled = False
            await self.push_frame(InterruptionFrame())

            if should_combine:
                text = f"{self._last_user_text} {text}"
                logger.info(f"[multiagent] Combined with interrupted message: {text[:120]}")

            self._last_user_text = text
            self._last_user_time = now
            self._agent_has_spoken = False

            self._current_task = asyncio.create_task(
                self._handle_utterance_bg(text)
            )

        elif isinstance(frame, StartInterruptionFrame):
            logger.info("[multiagent] StartInterruptionFrame — stopping audio")
            self._is_streaming = False
            if self._tts and hasattr(self._tts, "_cancel_all_tts"):
                self._tts._cancel_all_tts("StartInterruptionFrame from VAD")
            elif self._tts:
                self._tts._interrupted = True
            await self.push_frame(frame, direction)

        elif isinstance(frame, UserStartedSpeakingFrame):
            logger.debug("[multiagent] UserStartedSpeakingFrame")
            await self.push_frame(frame, direction)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.debug("[multiagent] UserStoppedSpeakingFrame")
            await self.push_frame(frame, direction)

        elif isinstance(frame, InterimTranscriptionFrame):
            pass

        elif isinstance(frame, (CancelFrame, EndFrame)):
            self._cancel_current()
            self._cancel_all_executors()
            if self._update_poller and not self._update_poller.done():
                self._update_poller.cancel()
                self._update_poller = None
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    # ------------------------------------------------------------------
    # Main utterance handling
    # ------------------------------------------------------------------

    async def _handle_utterance_bg(self, text: str):
        """Background wrapper that sets _responding flag."""
        try:
            self._responding = True
            self._is_streaming = True
            self._stream_completed_at = 0.0
            self._stream_text_len = 0
            self._current_response_executor_ids.clear()
            await self._handle_utterance(text)
        except asyncio.CancelledError:
            logger.info("[INTERRUPT] LangGraph stream cancelled")
        except Exception as exc:
            logger.error(f"LangGraph invocation error: {exc}", exc_info=True)
        finally:
            self._responding = False
            self._is_streaming = False

    async def _handle_utterance(self, text: str):
        """Stream tokens from the orchestrator to TTS.

        Tool launches are detected from the graph state (``tool_launches``
        field in ``updates`` events) rather than regex scanning.
        """
        await self._ensure_client()
        t_start = time.perf_counter()

        if self._pending_interrupt:
            await self._resume_executor(self._pending_interrupt, text)
            self._pending_interrupt = None

        status_context = self._build_status_context()

        human_input = text
        if status_context:
            human_input = f"{text}\n\n[System: {status_context}]"

        stream = self._client.runs.stream(
            self._thread_id,
            self._graph_name,
            input={"human_input": human_input, "scenario_id": self._scenario_id},
            stream_mode=["messages", "updates"],
            multitask_strategy="interrupt",
        )

        run_nodes: dict[str, str] = {}
        current_run_id: str | None = None
        current_speaker: str | None = None
        pending_speaker: str | None = None
        agent_started = False
        prev_content_len = 0
        invoke_count = 0
        agent_responses: dict[str, str] = {}
        all_tool_launches: list[dict] = []

        async for chunk in stream:
            if self._cancelled:
                logger.info("[multiagent] _cancelled flag set — breaking stream loop")
                break
            event = chunk.event
            data = chunk.data

            if event == "messages/metadata" and isinstance(data, dict):
                for run_id, meta in data.items():
                    node = meta.get("metadata", {}).get("langgraph_node", "")
                    if node:
                        run_nodes[run_id] = node

            elif event == "messages/partial" and isinstance(data, (list, tuple)) and data:
                msg = data[0] if isinstance(data[0], dict) else {}
                run_id = msg.get("id", "")
                node = run_nodes.get(run_id, "")
                content = msg.get("content", "")

                if node not in INVOKE_NODES or not content:
                    continue

                if run_id != current_run_id:
                    current_run_id = run_id
                    prev_content_len = 0

                if pending_speaker and pending_speaker != current_speaker:
                    if agent_started:
                        await self.push_frame(LLMFullResponseEndFrame())

                    current_speaker = pending_speaker
                    pending_speaker = None

                    voice = self._voice_map.get(current_speaker)
                    t_ms = (time.perf_counter() - t_start) * 1000
                    logger.info(
                        f"[STREAM] {current_speaker} first token "
                        f"(voice={voice}) t={t_ms:.0f}ms"
                    )

                    await self.push_frame(
                        VoiceSwitchFrame(voice=voice, speaker=current_speaker)
                    )
                    await self.push_frame(LLMFullResponseStartFrame())
                    agent_started = True
                    self._agent_has_spoken = True
                    invoke_count += 1

                delta = content[prev_content_len:]
                prev_content_len = len(content)
                if delta:
                    await self.push_frame(TextFrame(text=delta))
                    if current_speaker:
                        agent_responses[current_speaker] = content

            elif event == "updates" and isinstance(data, dict):
                for node_name, updates in data.items():
                    if not isinstance(updates, dict):
                        continue

                    if node_name == "route":
                        r = updates.get("routing", {})
                        na = updates.get("next_agent", "")
                        if r:
                            t_ms = (time.perf_counter() - t_start) * 1000
                            logger.info(
                                f"[TIMING] routing: "
                                f"primary={r.get('primary_agent')} "
                                f"secondary={r.get('secondary_agent', 'none')} "
                                f"topic={r.get('topic')} "
                                f"t={t_ms:.0f}ms"
                            )
                        if na:
                            pending_speaker = na

                    elif node_name == "check_handoff":
                        na = updates.get("next_agent", "")
                        if na:
                            pending_speaker = na
                            logger.info(
                                f"[HANDOFF] Next speaker: {na}"
                            )

                    elif node_name in INVOKE_NODES:
                        if agent_started:
                            t_ms = (time.perf_counter() - t_start) * 1000
                            logger.info(
                                f"[TIMING] {current_speaker} complete "
                                f"t={t_ms:.0f}ms"
                            )
                            await self.push_frame(LLMFullResponseEndFrame())
                            agent_started = False
                            current_run_id = None
                            prev_content_len = 0

                        launches = updates.get("tool_launches", [])
                        if launches:
                            logger.info(
                                f"[REACT-LAUNCH] launched "
                                f"{[l['executor_tool'] for l in launches]}"
                            )
                            all_tool_launches.extend(launches)

        if agent_started:
            await self.push_frame(LLMFullResponseEndFrame())

        total = (time.perf_counter() - t_start) * 1000
        logger.info(f"[TIMING] utterance_complete total={total:.0f}ms")

        total_text_len = sum(len(t) for t in agent_responses.values())
        self._stream_completed_at = time.perf_counter()
        self._stream_text_len = total_text_len

        if not self._cancelled and all_tool_launches:
            await self._start_tool_launches(all_tool_launches)

    # ------------------------------------------------------------------
    # Delegation detection and executor management
    # ------------------------------------------------------------------

    async def _start_tool_launches(self, launches: list[dict]):
        """Start background executors for slow tool launches from the ReAct loop."""
        for launch in launches:
            agent = launch["agent"]
            executor_tool = launch["executor_tool"]
            tool_args = launch.get("args", {})
            task_id = f"{agent}-{executor_tool}-{uuid.uuid4().hex[:6]}"

            self._active_tasks[task_id] = {
                "agent": agent,
                "tool": executor_tool,
                "status": "running",
                "task_id": task_id,
                "started_at": time.perf_counter(),
            }

            executor_graph = "executor"

            logger.info(
                f"[EXECUTOR] Starting {executor_graph} tool={executor_tool} "
                f"agent={agent} scenario={self._scenario_id} task_id={task_id}"
            )

            self._current_response_executor_ids.append(task_id)

            bg = asyncio.create_task(
                self._run_executor_bg(
                    task_id, agent, executor_graph, executor_tool, tool_args
                )
            )
            self._executor_bg_tasks[task_id] = bg

    async def _run_executor_bg(
        self,
        task_id: str,
        agent: str,
        executor_graph: str,
        tool_name: str,
        tool_args: dict,
    ):
        """Run an executor graph in the background."""
        try:
            await self._ensure_client()

            thread = await self._client.threads.create()
            thread_id = thread["thread_id"]
            self._executor_threads[task_id] = thread_id

            result = await self._client.runs.wait(
                thread_id,
                assistant_id=executor_graph,
                input={
                    "session_id": self._session_id,
                    "scenario_id": self._scenario_id,
                    "agent": agent,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                },
            )

            await self._handle_executor_result(task_id, agent, thread_id, result)

        except asyncio.CancelledError:
            logger.info(f"[EXECUTOR] {task_id} cancelled")
        except Exception as exc:
            logger.error(f"[EXECUTOR] {task_id} error: {exc}", exc_info=True)
            self._executor_updates.append({
                "type": "error",
                "task_id": task_id,
                "agent": agent,
                "error": str(exc),
            })
            if task_id in self._active_tasks:
                self._active_tasks[task_id]["status"] = "failed"
            await self._deliver_updates_if_idle()

    async def _handle_executor_result(
        self, task_id: str, agent: str, thread_id: str, result
    ):
        """Process executor result: completion or interrupt (needs input)."""
        if isinstance(result, dict) and result.get("__interrupt__"):
            interrupts = result["__interrupt__"]
            if interrupts:
                question = (
                    interrupts[0].get("value", {}).get("question", "Could you clarify?")
                )
                logger.info(
                    f"[EXECUTOR] {task_id} interrupted — question={question[:80]!r}"
                )
                self._executor_updates.append({
                    "type": "question",
                    "task_id": task_id,
                    "agent": agent,
                    "thread_id": thread_id,
                    "question": question,
                })
                if task_id in self._active_tasks:
                    self._active_tasks[task_id]["status"] = "waiting_for_input"
                await self._deliver_updates_if_idle()
                return

        result_str = str(result) if not isinstance(result, str) else result

        elapsed = 0.0
        task_info = self._active_tasks.get(task_id)
        if task_info and task_info.get("started_at"):
            elapsed = time.perf_counter() - task_info["started_at"]

        logger.info(
            f"[EXECUTOR] {task_id} completed in {elapsed:.1f}s — "
            f"result={result_str[:100]!r}"
        )

        if task_info:
            task_info["status"] = "completed"

        FAST_TOOL_THRESHOLD = 5.0
        if elapsed < FAST_TOOL_THRESHOLD:
            logger.info(
                f"[EXECUTOR] {task_id} was a fast tool ({elapsed:.1f}s < "
                f"{FAST_TOOL_THRESHOLD}s) — skipping delivery (agent already "
                f"spoke the answer)"
            )
            return

        self._executor_updates.append({
            "type": "result",
            "task_id": task_id,
            "agent": agent,
            "result": result_str,
        })
        await self._deliver_updates_if_idle()

    async def _resume_executor(self, interrupt_info: dict, user_answer: str):
        """Resume an interrupted executor with the user's answer."""
        task_id = interrupt_info["task_id"]
        thread_id = interrupt_info["thread_id"]
        agent = interrupt_info["agent"]

        logger.info(
            f"[EXECUTOR] Resuming {task_id} with answer={user_answer[:80]!r}"
        )

        if task_id in self._active_tasks:
            self._active_tasks[task_id]["status"] = "running"

        bg = asyncio.create_task(
            self._resume_executor_bg(task_id, agent, thread_id, user_answer)
        )
        self._executor_bg_tasks[task_id] = bg

    async def _resume_executor_bg(
        self, task_id: str, agent: str, thread_id: str, answer: str
    ):
        """Background: resume an interrupted executor."""
        try:
            result = await self._client.runs.wait(
                thread_id,
                assistant_id="executor",
                command=Command(resume=answer),
            )
            await self._handle_executor_result(task_id, agent, thread_id, result)
        except asyncio.CancelledError:
            logger.info(f"[EXECUTOR] Resume {task_id} cancelled")
        except Exception as exc:
            logger.error(f"[EXECUTOR] Resume {task_id} error: {exc}", exc_info=True)

    async def _poll_executor_updates(self):
        """Background loop: retry delivering executor results when pipeline goes idle.

        Runs every 3 seconds.  If there are pending updates and nobody is
        streaming, triggers delivery.  Silently exits on cancellation.
        """
        try:
            while True:
                await asyncio.sleep(3)
                if (
                    self._executor_updates
                    and not self._is_streaming
                    and not self._cancelled
                ):
                    logger.info(
                        f"[POLLER] {len(self._executor_updates)} pending update(s) "
                        "— pipeline idle, delivering now"
                    )
                    await self._deliver_updates_if_idle()
        except asyncio.CancelledError:
            logger.debug("[POLLER] Update poller stopped")

    async def _deliver_updates_if_idle(self):
        """If not currently streaming, deliver executor updates via the front agent.

        Only the PRIMARY agent speaks — no secondary. This prevents
        executor deliveries from triggering multi-turn conversations.
        """
        if self._is_streaming or self._cancelled:
            return
        if not self._executor_updates:
            return

        updates = list(self._executor_updates)
        self._executor_updates.clear()

        parts = []
        for u in updates:
            agent = u.get("agent", "unknown")
            if u["type"] == "result":
                parts.append(
                    f"[System: {agent}'s tool completed — {u.get('result', '')[:200]}]"
                )
            elif u["type"] == "question":
                parts.append(
                    f"[System: {agent}'s tool needs input — {u.get('question', '')}]"
                )
            elif u["type"] == "error":
                parts.append(
                    f"[System: {agent}'s tool failed — {u.get('error', '')}]"
                )

        if not parts:
            return

        status_msg = "\n".join(parts)
        logger.info(f"[EXECUTOR] Delivering updates: {status_msg[:150]!r}")

        self._is_streaming = True
        self._cancelled = False
        primary_delivered = False
        try:
            await self._ensure_client()
            stream = self._client.runs.stream(
                self._thread_id,
                self._graph_name,
                input={"human_input": status_msg, "scenario_id": self._scenario_id},
                stream_mode=["messages", "updates"],
                multitask_strategy="interrupt",
            )

            run_nodes: dict[str, str] = {}
            current_speaker: str | None = None
            pending_speaker: str | None = None
            agent_started = False
            prev_content_len = 0

            async for chunk in stream:
                if self._cancelled:
                    break
                event = chunk.event
                data = chunk.data

                if event == "messages/metadata" and isinstance(data, dict):
                    for run_id, meta in data.items():
                        node = meta.get("metadata", {}).get("langgraph_node", "")
                        if node:
                            run_nodes[run_id] = node

                elif event == "messages/partial" and isinstance(data, (list, tuple)) and data:
                    msg = data[0] if isinstance(data[0], dict) else {}
                    run_id = msg.get("id", "")
                    node = run_nodes.get(run_id, "")
                    content = msg.get("content", "")

                    if node not in INVOKE_NODES or not content:
                        continue

                    if primary_delivered:
                        continue

                    if pending_speaker and pending_speaker != current_speaker:
                        if agent_started:
                            await self.push_frame(LLMFullResponseEndFrame())

                        current_speaker = pending_speaker
                        pending_speaker = None
                        prev_content_len = 0

                        voice = self._voice_map.get(current_speaker)
                        logger.info(
                            f"[EXECUTOR-DELIVERY] {current_speaker} speaking "
                            f"(voice={voice})"
                        )

                        await self.push_frame(
                            VoiceSwitchFrame(voice=voice, speaker=current_speaker)
                        )
                        await self.push_frame(LLMFullResponseStartFrame())
                        agent_started = True

                    delta = content[prev_content_len:]
                    prev_content_len = len(content)
                    if delta:
                        await self.push_frame(TextFrame(text=delta))

                elif event == "updates" and isinstance(data, dict):
                    for node_name, node_updates in data.items():
                        if not isinstance(node_updates, dict):
                            continue
                        if node_name == "route":
                            na = node_updates.get("next_agent", "")
                            if na:
                                pending_speaker = na
                        elif node_name in INVOKE_NODES and agent_started:
                            await self.push_frame(LLMFullResponseEndFrame())
                            agent_started = False
                            primary_delivered = True
                            prev_content_len = 0

            if agent_started:
                await self.push_frame(LLMFullResponseEndFrame())

            for u in updates:
                if u["type"] == "question":
                    self._pending_interrupt = {
                        "task_id": u["task_id"],
                        "agent": u["agent"],
                        "thread_id": u.get("thread_id", ""),
                        "question": u.get("question", ""),
                    }
                    logger.info(
                        f"[EXECUTOR] Pending interrupt set — task={u['task_id']}"
                    )
                    break

        except asyncio.CancelledError:
            logger.info("[EXECUTOR-DELIVERY] Cancelled during update delivery")
        except Exception as exc:
            logger.error(f"[EXECUTOR-DELIVERY] Error: {exc}", exc_info=True)
        finally:
            self._is_streaming = False

        for u in updates:
            if u["type"] == "result":
                self._active_tasks.pop(u.get("task_id", ""), None)
                self._executor_bg_tasks.pop(u.get("task_id", ""), None)

    def _build_status_context(self) -> str:
        """Build status context from queued executor updates."""
        if not self._executor_updates:
            return ""

        updates = list(self._executor_updates)
        self._executor_updates.clear()

        parts = []
        for u in updates:
            agent = u.get("agent", "unknown")
            if u["type"] == "result":
                parts.append(
                    f"{agent}'s tool completed: {u.get('result', '')[:200]}"
                )
            elif u["type"] == "question":
                parts.append(
                    f"{agent}'s tool needs input: {u.get('question', '')}"
                )
                self._pending_interrupt = {
                    "task_id": u["task_id"],
                    "agent": u["agent"],
                    "thread_id": u.get("thread_id", ""),
                    "question": u.get("question", ""),
                }

        return "; ".join(parts) if parts else ""

    def _cancel_all_executors(self):
        """Cancel all running executor background tasks."""
        for task_id, bg_task in list(self._executor_bg_tasks.items()):
            if not bg_task.done():
                bg_task.cancel()
                logger.info(f"[EXECUTOR] Cancelled executor {task_id}")
        self._executor_bg_tasks.clear()
        self._active_tasks.clear()
        self._executor_updates.clear()
        self._pending_interrupt = None
