"""Worker agent: executes long-running tools and writes results to Store.

Each job gets its own thread (for checkpointing / resumability). The worker
receives fully-resolved parameters from the router and an optional context
summary. It writes progress updates and the final result to the Store so
the chitchat agent (and the client) can track status.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.store.memory import InMemoryStore
from loguru import logger

from tools.mock_tools import run_tool, TOOL_REGISTRY


@task(retry_policy={"max_attempts": 2})
async def execute_tool(tool_name: str, *, session_id: str, job_id: str, params: dict) -> dict:
    """Run the tool. Retries once on failure."""
    logger.info(f"[{job_id}] executing tool={tool_name} params={params}")
    return await run_tool(tool_name, session_id=session_id, job_id=job_id, params=params)


@entrypoint()
async def graph(inputs: dict) -> dict:
    """Execute a tool job. Writes progress to Store throughout.

    Expected inputs:
        tool: str           - tool name from TOOL_REGISTRY
        params: dict        - tool parameters (resolved by router)
        session_id: str     - session owning this job
        job_id: str         - unique job identifier (optional, generated if missing)
        context_summary: str - optional conversation context
    """
    tool_name = inputs["tool"]
    params = inputs.get("params", {})
    session_id = inputs["session_id"]
    job_id = inputs.get("job_id", f"job-{uuid.uuid4().hex[:8]}")
    context_summary = inputs.get("context_summary", "")

    if tool_name not in TOOL_REGISTRY:
        return {
            "job_id": job_id,
            "status": "failed",
            "error": f"Unknown tool: {tool_name}",
        }

    logger.info(
        f"[{job_id}] starting tool={tool_name} session={session_id} "
        f"context='{context_summary[:60]}'"
    )

    try:
        result = await execute_tool(
            tool_name,
            session_id=session_id,
            job_id=job_id,
            params=params,
        ).result()

        return {
            "job_id": job_id,
            "status": "complete",
            "tool": tool_name,
            "result": result,
        }

    except Exception as e:
        logger.error(f"[{job_id}] tool failed: {e}")
        from langgraph.config import get_store
        store = get_store()
        await store.aput(("jobs", session_id), job_id, {
            "status": "failed",
            "tool": tool_name,
            "error": str(e),
            "message": f"Tool '{tool_name}' failed: {e}",
        })
        return {
            "job_id": job_id,
            "status": "failed",
            "tool": tool_name,
            "error": str(e),
        }
