"""Dummy tools for the generic agent demo.

These simulate real work with asyncio.sleep so we can demonstrate
the dual-thread pattern (chitchat while the agent is busy).

Each long-running tool writes live progress to the LangGraph Store
so the secondary (chitchat) agent can report status to the user.
"""

import asyncio
import json
import random
import uuid
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from loguru import logger


def _session_ns(config: RunnableConfig) -> tuple:
    sid = config.get("configurable", {}).get("session_id", "default")
    return ("session", sid)


@tool
async def run_data_analysis(
    dataset: str,
    focus: str = "general",
    *,
    config: RunnableConfig,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """Run a comprehensive data analysis on a dataset.

    This is a long-running operation (20-30 seconds). Use it when the user
    asks for analysis, reports, deep dives, or investigations on data.

    Args:
        dataset: Name or description of the dataset to analyze.
        focus: Specific area to focus on (e.g., "anomalies", "trends", "revenue").
    """
    ns = _session_ns(config)
    job_id = f"analysis-{uuid.uuid4().hex[:6]}"
    duration = random.randint(20, 30)
    logger.info(
        f"[tools] Starting {job_id}: dataset={dataset!r} focus={focus!r} duration={duration}s"
    )

    phases = [
        "Loading dataset",
        "Validating schema",
        "Computing statistics",
        "Detecting anomalies",
        "Running regression",
        "Cross-validating",
        "Building summary",
    ]

    for i, phase in enumerate(phases):
        pct = int((i + 1) / len(phases) * 100)
        await store.aput(
            ns,
            "task_progress",
            {
                "tool": "run_data_analysis",
                "status": "running",
                "dataset": dataset,
                "focus": focus,
                "phase": phase,
                "progress": f"{pct}%",
                "step": f"{i + 1}/{len(phases)}",
            },
        )
        await asyncio.sleep(duration / len(phases))
        logger.info(f"[tools] {job_id}: {phase} ({i+1}/{len(phases)})")

    result = {
        "job_id": job_id,
        "dataset": dataset,
        "focus": focus,
        "records_analyzed": random.randint(10000, 500000),
        "anomalies_found": random.randint(1, 8),
        "top_finding": (
            f"Significant {focus} deviation detected in Q3 "
            f"({random.randint(15, 45)}% above baseline)"
        ),
        "trend": random.choice(["upward", "stable", "declining"]),
        "confidence": round(random.uniform(0.85, 0.98), 2),
        "recommendation": (
            f"Further investigation recommended on {focus} patterns "
            f"in the {dataset} dataset."
        ),
    }

    await store.aput(
        ns,
        "task_progress",
        {
            "tool": "run_data_analysis",
            "status": "complete",
            "result_summary": result["top_finding"],
        },
    )

    logger.info(f"[tools] {job_id}: Complete")
    return json.dumps(result)


@tool
async def check_system_status() -> str:
    """Quick health check of all monitored systems.

    Returns current status of CPU, memory, GPU, and services.
    This is a fast operation (2-3 seconds).
    """
    await asyncio.sleep(random.uniform(2, 3))

    result = {
        "cpu_usage": f"{random.randint(15, 85)}%",
        "memory_usage": f"{random.randint(30, 75)}%",
        "gpu_utilization": f"{random.randint(10, 95)}%",
        "active_services": random.randint(8, 15),
        "alerts": random.randint(0, 3),
        "uptime_hours": random.randint(100, 5000),
        "status": random.choice(["healthy", "healthy", "healthy", "degraded"]),
    }

    return json.dumps(result)


@tool
async def train_model(
    model_name: str,
    dataset: str,
    *,
    config: RunnableConfig,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """Train a machine learning model on a dataset.

    This is a very long-running operation (40-60 seconds). Use when the user
    explicitly asks to train, fine-tune, or build a model.

    Args:
        model_name: Name for the model being trained.
        dataset: Dataset to train on.
    """
    ns = _session_ns(config)
    job_id = f"train-{uuid.uuid4().hex[:6]}"
    epochs = 10
    duration = random.randint(40, 60)
    logger.info(
        f"[tools] Starting {job_id}: model={model_name!r} dataset={dataset!r}"
    )

    await store.aput(
        ns,
        "task_progress",
        {
            "tool": "train_model",
            "status": "running",
            "model_name": model_name,
            "dataset": dataset,
            "phase": "Initializing",
            "epoch": "0/10",
            "progress": "0%",
        },
    )

    for epoch in range(1, epochs + 1):
        await asyncio.sleep(duration / epochs)
        loss = round(2.5 - (epoch * 0.2) + random.uniform(-0.05, 0.05), 3)
        pct = int(epoch / epochs * 100)
        await store.aput(
            ns,
            "task_progress",
            {
                "tool": "train_model",
                "status": "running",
                "model_name": model_name,
                "dataset": dataset,
                "phase": "Training",
                "epoch": f"{epoch}/{epochs}",
                "loss": str(loss),
                "progress": f"{pct}%",
            },
        )
        logger.info(f"[tools] {job_id}: Epoch {epoch}/{epochs} loss={loss}")

    accuracy = round(random.uniform(0.88, 0.95), 3)
    final_loss = round(0.34 + random.uniform(-0.05, 0.05), 3)
    result = {
        "job_id": job_id,
        "model_name": model_name,
        "dataset": dataset,
        "epochs_completed": epochs,
        "final_loss": final_loss,
        "accuracy": accuracy,
        "training_time_secs": duration,
        "status": "complete",
        "summary": (
            f"Model '{model_name}' trained on '{dataset}' for {epochs} epochs. "
            f"Ready for deployment."
        ),
    }

    await store.aput(
        ns,
        "task_progress",
        {
            "tool": "train_model",
            "status": "complete",
            "model_name": model_name,
            "result_summary": (
                f"Training complete — accuracy {accuracy}, loss {final_loss}"
            ),
        },
    )

    logger.info(f"[tools] {job_id}: Training complete")
    return json.dumps(result)


ALL_TOOLS = [run_data_analysis, check_system_status, train_model]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
