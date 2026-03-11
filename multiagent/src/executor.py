"""Generic executor graph for background tool execution.

A single executor handles ALL agents across ALL scenarios. It receives
the scenario_id and agent_id at runtime and looks up the appropriate
tools from the scenario registry.

The executor:
  - Receives a tool request (scenario, agent, tool name + args)
  - Resolves the tool from the scenario's agent config
  - Runs the tool (fast or slow)
  - Writes progress updates to the LangGraph Store
  - Can interrupt to ask the user a question
  - Returns the final result
"""

import time
import uuid
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from loguru import logger

from src.scenarios import get_scenario


# ---------------------------------------------------------------------------
# Executor state
# ---------------------------------------------------------------------------

class ExecutorState(TypedDict):
    session_id: str
    scenario_id: str
    agent: str
    task_id: str
    tool_name: str
    tool_args: dict
    status: str
    result: dict | None
    error: str | None


class ExecutorInput(TypedDict):
    session_id: str
    scenario_id: str
    agent: str
    tool_name: str
    tool_args: dict


# ---------------------------------------------------------------------------
# Generic executor node
# ---------------------------------------------------------------------------

async def run_tool(state: ExecutorState, *, store) -> dict:
    """Execute a tool for any agent in any scenario."""
    scenario_id = state.get("scenario_id", "ecommerce")
    agent_id = state.get("agent", "")
    tool_name = state["tool_name"]
    tool_args = state.get("tool_args", {})
    session_id = state["session_id"]
    task_id = state.get("task_id") or f"{agent_id}-{tool_name}-{uuid.uuid4().hex[:6]}"

    scenario = get_scenario(scenario_id)
    agent_cfg = scenario.get_agent(agent_id)

    fast_tool_map = {t.name: t for t in agent_cfg.fast_tools}
    slow_tools = agent_cfg.slow_tools

    logger.info(
        f"[EXECUTOR] scenario={scenario_id} agent={agent_id} "
        f"tool={tool_name} task_id={task_id}"
    )

    await store.aput(
        namespace=("tasks", session_id),
        key=task_id,
        value={
            "agent": agent_id,
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
                    "agent": agent_id,
                    "tool": tool_name,
                    "status": "completed",
                    "progress": 1.0,
                    "message": f"{tool_name} complete",
                    "result": result,
                    "updated_at": time.time(),
                },
            )
            return {"task_id": task_id, "status": "completed", "result": result}

        elif tool_name in slow_tools:
            fn = slow_tools[tool_name]
            result = await fn(store, session_id, task_id)

            if isinstance(result, dict) and result.get("status") == "needs_input":
                question = result.get("question", "Could you provide more info?")
                answer = interrupt({"question": question, "task_id": task_id})
                logger.info(
                    f"[EXECUTOR] {agent_id} resumed with answer: {str(answer)[:80]}"
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

            return {"task_id": task_id, "status": "completed", "result": result}

        else:
            error = f"Unknown tool: {tool_name}"
            logger.warning(f"[EXECUTOR] {error}")
            return {"task_id": task_id, "status": "failed", "error": error}

    except Exception as exc:
        error = str(exc)
        logger.error(f"[EXECUTOR] {agent_id} tool {tool_name} failed: {error}")
        await store.aput(
            namespace=("tasks", session_id),
            key=task_id,
            value={
                "agent": agent_id,
                "tool": tool_name,
                "status": "failed",
                "progress": 0.0,
                "message": f"Failed: {error}",
                "updated_at": time.time(),
            },
        )
        return {"task_id": task_id, "status": "failed", "error": error}


# ---------------------------------------------------------------------------
# Build the single generic executor graph
# ---------------------------------------------------------------------------

_builder = StateGraph(ExecutorState, input=ExecutorInput)
_builder.add_node("run_tool", run_tool)
_builder.add_edge(START, "run_tool")
_builder.add_edge("run_tool", END)

executor_graph = _builder.compile()
