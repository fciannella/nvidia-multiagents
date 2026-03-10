"""Per-agent executor graphs for background tool execution.

Each executor:
  - Receives a tool request (tool name + args)
  - Runs the tool (fast or slow)
  - Writes progress updates to the LangGraph Store
  - Can interrupt to ask the user a question (e.g. Kafka approval)
  - Returns the final result

Two separate compiled graphs are exported:
  - ds_executor_graph  (Sarah's tools)
  - it_executor_graph  (Mike's tools)
"""

import asyncio
import time
import uuid
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from loguru import logger

from src.tools.ds_tools import DS_FAST_TOOLS, DS_SLOW_TOOLS
from src.tools.it_tools import IT_FAST_TOOLS, IT_SLOW_TOOLS


# ---------------------------------------------------------------------------
# Executor state
# ---------------------------------------------------------------------------

class ExecutorState(TypedDict):
    """State for an executor run."""
    session_id: str
    task_id: str
    tool_name: str
    tool_args: dict
    agent: str
    status: str
    result: dict | None
    error: str | None


class ExecutorInput(TypedDict):
    """Input to kick off an executor run."""
    session_id: str
    tool_name: str
    tool_args: dict


# ---------------------------------------------------------------------------
# Generic executor node factory
# ---------------------------------------------------------------------------

def _make_executor_node(
    agent_key: str,
    fast_tools: list,
    slow_tools: dict,
):
    """Build the main executor node for an agent.

    Fast tools are called synchronously via their langchain @tool interface.
    Slow tools are async functions that write progress to the Store.
    """
    fast_tool_map = {t.name: t for t in fast_tools}

    async def run_tool(state: ExecutorState, *, store) -> dict:
        tool_name = state["tool_name"]
        tool_args = state.get("tool_args", {})
        session_id = state["session_id"]
        task_id = state.get("task_id") or f"{agent_key}-{tool_name}-{uuid.uuid4().hex[:6]}"

        logger.info(
            f"[EXECUTOR-{agent_key}] Running tool={tool_name} "
            f"task_id={task_id} args={tool_args}"
        )

        await store.aput(
            namespace=("tasks", session_id),
            key=task_id,
            value={
                "agent": agent_key,
                "tool": tool_name,
                "status": "started",
                "progress": 0.0,
                "message": f"Starting {tool_name}...",
                "updated_at": time.time(),
            },
        )

        try:
            if tool_name in fast_tool_map:
                result = fast_tool_map[tool_name].invoke(tool_args)
                await store.aput(
                    namespace=("tasks", session_id),
                    key=task_id,
                    value={
                        "agent": agent_key,
                        "tool": tool_name,
                        "status": "completed",
                        "progress": 1.0,
                        "message": f"{tool_name} complete",
                        "result": result,
                        "updated_at": time.time(),
                    },
                )
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": result,
                }

            elif tool_name in slow_tools:
                fn = slow_tools[tool_name]
                result = await fn(store, session_id, task_id)

                if isinstance(result, dict) and result.get("status") == "needs_input":
                    question = result.get("question", "Could you provide more info?")
                    answer = interrupt({"question": question, "task_id": task_id})
                    logger.info(
                        f"[EXECUTOR-{agent_key}] Resumed with answer: {str(answer)[:80]}"
                    )
                    resume_fn_name = f"{tool_name}_resume"
                    if resume_fn_name in slow_tools:
                        result = await slow_tools[resume_fn_name](
                            store, session_id, task_id, answer
                        )
                    return {
                        "task_id": task_id,
                        "status": result.get("status", "completed") if isinstance(result, dict) else "completed",
                        "result": result,
                    }

                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": result,
                }

            else:
                error = f"Unknown tool: {tool_name}"
                logger.warning(f"[EXECUTOR-{agent_key}] {error}")
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": error,
                }

        except Exception as exc:
            error = str(exc)
            logger.error(f"[EXECUTOR-{agent_key}] Tool {tool_name} failed: {error}")
            await store.aput(
                namespace=("tasks", session_id),
                key=task_id,
                value={
                    "agent": agent_key,
                    "tool": tool_name,
                    "status": "failed",
                    "progress": 0.0,
                    "message": f"Failed: {error}",
                    "updated_at": time.time(),
                },
            )
            return {
                "task_id": task_id,
                "status": "failed",
                "error": error,
            }

    return run_tool


# ---------------------------------------------------------------------------
# Build executor graphs
# ---------------------------------------------------------------------------

def _build_executor_graph(agent_key: str, fast_tools: list, slow_tools: dict):
    builder = StateGraph(ExecutorState, input=ExecutorInput)
    node = _make_executor_node(agent_key, fast_tools, slow_tools)
    builder.add_node("run_tool", node)
    builder.add_edge(START, "run_tool")
    builder.add_edge("run_tool", END)
    return builder.compile()


ds_executor_graph = _build_executor_graph(
    "data_scientist", DS_FAST_TOOLS, DS_SLOW_TOOLS
)

it_executor_graph = _build_executor_graph(
    "it", IT_FAST_TOOLS, IT_SLOW_TOOLS
)
