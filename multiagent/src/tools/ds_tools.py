"""Mock tools for the Data Scientist agent (Sarah).

Each tool simulates real work with asyncio.sleep and writes progress
updates to the LangGraph Store so the front agent can report status.

Fast tools return immediately. Slow tools yield progress updates.
"""

import asyncio
import time
from typing import Any

from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from loguru import logger


# ---------------------------------------------------------------------------
# Fast tools (instant lookup)
# ---------------------------------------------------------------------------

@tool
def check_training_status() -> dict:
    """Check the current status of the model training job.

    Returns the current epoch, validation loss, and estimated time remaining.
    """
    return {
        "status": "idle",
        "last_run": "3 experimental runs completed on subset",
        "best_val_ndcg": 0.38,
        "baseline_ndcg": 0.29,
        "full_run_needed": True,
        "compute_required": "4 x A100, ~16 hours",
    }


@tool
def get_model_metrics() -> dict:
    """Get the latest offline evaluation metrics for RecDeep v2.

    Returns CTR, NDCG, and revenue-per-session compared to baseline.
    """
    return {
        "model": "RecDeep v2",
        "baseline_model": "RecLite v1",
        "ctr": {"v2": 0.038, "v1": 0.032, "lift": "+18.7%"},
        "ndcg_at_20": {"v2": 0.38, "v1": 0.29, "lift": "+31.0%"},
        "revenue_per_session": {"v2": "$0.58", "v1": "$0.47", "lift": "+23.4%"},
        "note": "Metrics from experimental runs on subset — full eval pending",
    }


@tool
def check_data_pipeline() -> dict:
    """Check the health of the feature data pipeline including Kafka streams.

    Returns pipeline status, throughput, and any lag issues.
    """
    return {
        "overall_status": "degraded",
        "kafka_throughput": "12k events/sec",
        "click_stream": {"lag": "< 1 min", "status": "healthy"},
        "search_signals": {"lag": "< 2 min", "status": "healthy"},
        "wishlist_pipeline": {
            "lag": "6 hours",
            "status": "degraded",
            "fix_needed": "Dedicated Kafka consumer group (needs IT provisioning)",
        },
    }


# ---------------------------------------------------------------------------
# Slow tools (background execution with progress updates)
# ---------------------------------------------------------------------------

async def run_training_job_async(
    store: BaseStore, session_id: str, task_id: str
) -> dict:
    """Simulate a training run with periodic progress updates."""
    epochs = 20
    for epoch in range(1, epochs + 1):
        await asyncio.sleep(2)
        progress = epoch / epochs
        val_loss = round(0.42 - (0.18 * progress) + 0.02 * (0.5 - progress), 3)
        msg = f"Epoch {epoch}/{epochs}, val_loss={val_loss}"
        logger.info(f"[DS-TOOL] run_training_job: {msg}")
        await store.aput(
            namespace=("tasks", session_id),
            key=task_id,
            value={
                "agent": "data_scientist",
                "tool": "run_training_job",
                "status": "running",
                "progress": round(progress, 2),
                "message": msg,
                "updated_at": time.time(),
            },
        )
    result = {
        "status": "completed",
        "epochs": epochs,
        "final_val_loss": 0.22,
        "final_val_ndcg": 0.41,
        "training_time": "15h 42m",
        "model_checkpoint": "/models/recdeep_v2_full.pt",
    }
    await store.aput(
        namespace=("tasks", session_id),
        key=task_id,
        value={
            "agent": "data_scientist",
            "tool": "run_training_job",
            "status": "completed",
            "progress": 1.0,
            "message": "Training complete — val NDCG=0.41, val_loss=0.22",
            "result": result,
            "updated_at": time.time(),
        },
    )
    return result


async def run_offline_eval_async(
    store: BaseStore, session_id: str, task_id: str
) -> dict:
    """Simulate running the offline evaluation suite."""
    phases = [
        ("Computing CTR on test set...", 0.25),
        ("Computing NDCG@20...", 0.50),
        ("Computing revenue-per-session lift...", 0.75),
        ("Generating evaluation report...", 0.90),
    ]
    for msg, progress in phases:
        await asyncio.sleep(3)
        logger.info(f"[DS-TOOL] run_offline_eval: {msg}")
        await store.aput(
            namespace=("tasks", session_id),
            key=task_id,
            value={
                "agent": "data_scientist",
                "tool": "run_offline_eval",
                "status": "running",
                "progress": progress,
                "message": msg,
                "updated_at": time.time(),
            },
        )
    await asyncio.sleep(2)
    result = {
        "status": "completed",
        "ctr_lift": "+18.7%",
        "ndcg_lift": "+31.0%",
        "revenue_lift": "+23.4%",
        "cold_start_coverage": "92% (up from 0%)",
        "p99_latency": "52ms (within 80ms SLA)",
        "recommendation": "PASS — ready for shadow deployment",
    }
    await store.aput(
        namespace=("tasks", session_id),
        key=task_id,
        value={
            "agent": "data_scientist",
            "tool": "run_offline_eval",
            "status": "completed",
            "progress": 1.0,
            "message": "Evaluation complete — all metrics pass",
            "result": result,
            "updated_at": time.time(),
        },
    )
    return result


async def export_model_to_triton_async(
    store: BaseStore, session_id: str, task_id: str
) -> dict:
    """Simulate exporting the model to TensorRT/Triton format."""
    phases = [
        ("Converting PyTorch model to ONNX...", 0.30),
        ("Optimizing with TensorRT (FP16)...", 0.60),
        ("Validating exported model accuracy...", 0.85),
    ]
    for msg, progress in phases:
        await asyncio.sleep(2)
        logger.info(f"[DS-TOOL] export_model_to_triton: {msg}")
        await store.aput(
            namespace=("tasks", session_id),
            key=task_id,
            value={
                "agent": "data_scientist",
                "tool": "export_model_to_triton",
                "status": "running",
                "progress": progress,
                "message": msg,
                "updated_at": time.time(),
            },
        )
    await asyncio.sleep(2)
    result = {
        "status": "completed",
        "format": "TensorRT FP16",
        "model_path": "/models/recdeep_v2_trt.plan",
        "inference_latency": {"p50": "16ms", "p99": "48ms"},
        "accuracy_drift": "< 0.1% (within tolerance)",
    }
    await store.aput(
        namespace=("tasks", session_id),
        key=task_id,
        value={
            "agent": "data_scientist",
            "tool": "export_model_to_triton",
            "status": "completed",
            "progress": 1.0,
            "message": "Model exported to TensorRT — ready for deployment",
            "result": result,
            "updated_at": time.time(),
        },
    )
    return result


@tool
def start_training_job() -> dict:
    """Start a full training run of RecDeep v2 on the complete dataset.

    This is a LONG-RUNNING task (~16 hours, 20 epochs). Call this only
    when the user explicitly asks to start or kick off training.
    """
    return {
        "status": "submitted",
        "message": "Training job submitted — 20 epochs on full dataset, estimated 16 hours.",
    }


@tool
def start_offline_eval() -> dict:
    """Start the offline evaluation suite (CTR, NDCG, revenue lift).

    This is a LONG-RUNNING task (~15 minutes). Call this when the user
    asks to run or start the evaluation.
    """
    return {
        "status": "submitted",
        "message": "Offline evaluation started — computing CTR, NDCG, and revenue metrics.",
    }


@tool
def start_model_export() -> dict:
    """Start exporting the model to TensorRT/Triton format.

    This is a LONG-RUNNING task (~10 minutes). Call this when the user
    asks to export or convert the model.
    """
    return {
        "status": "submitted",
        "message": "Model export started — converting to TensorRT FP16 for Triton.",
    }


DS_FAST_TOOLS = [check_training_status, get_model_metrics, check_data_pipeline]

DS_SLOW_LAUNCHERS = [start_training_job, start_offline_eval, start_model_export]

DS_ALL_TOOLS = DS_FAST_TOOLS + DS_SLOW_LAUNCHERS

DS_LAUNCHER_TO_EXECUTOR = {
    "start_training_job": "run_training_job",
    "start_offline_eval": "run_offline_eval",
    "start_model_export": "export_model_to_triton",
}

DS_SLOW_TOOLS = {
    "run_training_job": run_training_job_async,
    "run_offline_eval": run_offline_eval_async,
    "export_model_to_triton": export_model_to_triton_async,
}
