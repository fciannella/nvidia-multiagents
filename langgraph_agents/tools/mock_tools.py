"""Mock tools that simulate long-running operations.

Each tool sleeps in steps, writing progress to the LangGraph Store at each
step so the chitchat agent (and the client) can report status to the user.

IMPORTANT: All tools use asyncio.sleep() (not time.sleep()) so the
single-worker langgraph dev runtime can interleave other runs during waits.
"""

import asyncio
from typing import Any

from langgraph.config import get_store
from loguru import logger

TOOL_REGISTRY: dict[str, dict[str, Any]] = {}


def _register(name: str, description: str, duration_secs: int, steps: int):
    """Decorator to register a tool function in the global registry."""

    def decorator(fn):
        TOOL_REGISTRY[name] = {
            "function": fn,
            "description": description,
            "duration_secs": duration_secs,
            "steps": steps,
        }
        return fn

    return decorator


async def _write_progress(session_id: str, job_id: str, data: dict):
    """Write job progress to the Store."""
    store = get_store()
    await store.aput(("jobs", session_id), job_id, data)
    logger.info(f"[{job_id}] progress: {data.get('status')} step={data.get('step')}/{data.get('total')}")


@_register("quick_lookup", "Fast database query (~5 seconds)", duration_secs=5, steps=5)
async def quick_lookup(*, session_id: str, job_id: str, params: dict) -> dict:
    """Simulate a quick database lookup."""
    total = 5
    await _write_progress(session_id, job_id, {
        "status": "running",
        "tool": "quick_lookup",
        "step": 0,
        "total": total,
        "message": "Starting database lookup...",
    })

    for step in range(1, total + 1):
        await asyncio.sleep(1)
        await _write_progress(session_id, job_id, {
            "status": "running",
            "tool": "quick_lookup",
            "step": step,
            "total": total,
            "message": f"Querying table {step} of {total}...",
        })

    result = {
        "records_found": 142,
        "top_match": f"Record matching '{params.get('query', 'unknown')}'",
        "confidence": 0.94,
    }
    await _write_progress(session_id, job_id, {
        "status": "complete",
        "tool": "quick_lookup",
        "step": total,
        "total": total,
        "message": "Lookup complete.",
        "result": result,
        "summary": f"Found 142 records. Top match for '{params.get('query', 'unknown')}' with 94% confidence.",
    })
    return result


@_register("long_analysis", "Data analysis job (~30 seconds)", duration_secs=30, steps=10)
async def long_analysis(*, session_id: str, job_id: str, params: dict) -> dict:
    """Simulate a long-running data analysis."""
    total = 10
    await _write_progress(session_id, job_id, {
        "status": "running",
        "tool": "long_analysis",
        "step": 0,
        "total": total,
        "message": "Initializing analysis pipeline...",
    })

    phase_names = [
        "Loading dataset",
        "Validating schema",
        "Computing statistics",
        "Detecting anomalies",
        "Running regression",
        "Cross-validating",
        "Generating clusters",
        "Ranking features",
        "Building summary",
        "Finalizing report",
    ]

    for step in range(1, total + 1):
        await asyncio.sleep(3)
        await _write_progress(session_id, job_id, {
            "status": "running",
            "tool": "long_analysis",
            "step": step,
            "total": total,
            "message": f"{phase_names[step - 1]}...",
        })

    dataset = params.get("dataset", "unknown")
    focus = params.get("focus", "general")
    result = {
        "dataset": dataset,
        "anomalies_found": 3,
        "top_anomaly": {"column": focus, "month": "August", "deviation": "+40%"},
        "secondary_anomalies": [
            {"column": focus, "month": "July", "deviation": "+12%"},
            {"column": focus, "month": "September", "deviation": "+8%"},
        ],
        "overall_trend": "upward",
    }
    await _write_progress(session_id, job_id, {
        "status": "complete",
        "tool": "long_analysis",
        "step": total,
        "total": total,
        "message": "Analysis complete.",
        "result": result,
        "summary": (
            f"Analysis of '{dataset}' found 3 anomalies in '{focus}'. "
            f"The biggest was a 40% spike in August. "
            f"July and September also showed smaller deviations. Overall trend is upward."
        ),
    })
    return result


@_register("very_long_training", "Model training job (~60 seconds)", duration_secs=60, steps=12)
async def very_long_training(*, session_id: str, job_id: str, params: dict) -> dict:
    """Simulate a very long model training job."""
    total = 12
    await _write_progress(session_id, job_id, {
        "status": "running",
        "tool": "very_long_training",
        "step": 0,
        "total": total,
        "message": "Preparing training environment...",
    })

    for step in range(1, total + 1):
        await asyncio.sleep(5)
        epoch_loss = round(2.5 - (step * 0.18), 3)
        await _write_progress(session_id, job_id, {
            "status": "running",
            "tool": "very_long_training",
            "step": step,
            "total": total,
            "message": f"Training epoch {step}/{total}, loss={epoch_loss}",
        })

    model_name = params.get("model_name", "custom-model")
    result = {
        "model_name": model_name,
        "final_loss": 0.34,
        "accuracy": 0.91,
        "epochs_completed": total,
        "training_time_secs": 60,
    }
    await _write_progress(session_id, job_id, {
        "status": "complete",
        "tool": "very_long_training",
        "step": total,
        "total": total,
        "message": "Training complete.",
        "result": result,
        "summary": (
            f"Model '{model_name}' trained for {total} epochs. "
            f"Final loss: 0.34, accuracy: 91%. Ready for deployment."
        ),
    })
    return result


def get_available_tools() -> list[dict]:
    """Return a list of tool descriptions for the router's system prompt."""
    return [
        {"name": name, "description": info["description"], "duration_secs": info["duration_secs"]}
        for name, info in TOOL_REGISTRY.items()
    ]


async def run_tool(name: str, *, session_id: str, job_id: str, params: dict) -> dict:
    """Execute a registered tool by name."""
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}. Available: {list(TOOL_REGISTRY.keys())}")
    return await TOOL_REGISTRY[name]["function"](session_id=session_id, job_id=job_id, params=params)
