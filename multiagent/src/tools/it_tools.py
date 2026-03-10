"""Mock tools for the IT Infrastructure agent (Mike).

Each tool simulates real work with asyncio.sleep and writes progress
updates to the LangGraph Store so the front agent can report status.

Fast tools return immediately. Slow tools yield progress updates.
"""

import asyncio
import time

from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from loguru import logger


# ---------------------------------------------------------------------------
# Fast tools (instant lookup)
# ---------------------------------------------------------------------------

@tool
def check_gpu_utilization() -> dict:
    """Check current GPU utilization across all clusters.

    Returns per-cluster GPU allocation and availability.
    """
    return {
        "production_cluster": {
            "total_gpus": 8,
            "model": "A100 80GB",
            "allocation": {
                "reclite_v1_serving": 1,
                "fraud_detection_retraining": 4,
                "search_ranking_serving": 3,
            },
            "available": 0,
            "note": "Can free 2 GPUs by moving fraud retraining to 2AM-4AM window",
        },
        "staging_cluster": {
            "total_gpus": 4,
            "model": "A100 80GB",
            "allocation": {},
            "available": 4,
            "status": "idle since 3 weeks ago",
        },
    }


@tool
def check_cluster_availability(cluster: str = "staging") -> dict:
    """Check scheduling and availability for a specific cluster.

    Args:
        cluster: Which cluster to check — "staging" or "production".
    """
    if cluster == "staging":
        return {
            "cluster": "staging",
            "gpus": 4,
            "status": "available",
            "available_from": "Monday 6AM",
            "next_reservation": "None scheduled",
            "note": "Clear all week — can be used for training",
        }
    return {
        "cluster": "production",
        "gpus": 8,
        "status": "fully allocated",
        "available_gpus": 0,
        "can_free_up": "2 GPUs by rescheduling fraud retraining to off-peak",
        "fraud_team_notified": False,
    }


@tool
def check_deployment_status() -> dict:
    """Check the current deployment status of all model versions.

    Returns traffic routing, health, and canary status.
    """
    return {
        "v1_reclite": {
            "status": "serving",
            "traffic": "100%",
            "healthy": True,
            "gpu": "1 x A100",
            "p99_latency": "42ms",
        },
        "v2_recdeep": {
            "status": "not deployed",
            "traffic": "0%",
            "model_exported": False,
            "triton_config": "not set up",
        },
        "rollback_ready": True,
        "rollback_time": "< 60 seconds (blue-green)",
    }


# ---------------------------------------------------------------------------
# Slow tools (background execution with progress updates)
# ---------------------------------------------------------------------------

async def provision_kafka_consumer_async(
    store: BaseStore, session_id: str, task_id: str
) -> dict:
    """Simulate provisioning a Kafka consumer group.

    This tool will pause to ask for justification (interrupt).
    """
    phases = [
        ("Submitting provisioning request to platform team...", 0.20),
        ("Awaiting platform team approval...", 0.50),
    ]
    for msg, progress in phases:
        await asyncio.sleep(3)
        logger.info(f"[IT-TOOL] provision_kafka_consumer: {msg}")
        await store.aput(
            namespace=("tasks", session_id),
            key=task_id,
            value={
                "agent": "it",
                "tool": "provision_kafka_consumer",
                "status": "running",
                "progress": progress,
                "message": msg,
                "updated_at": time.time(),
            },
        )

    await store.aput(
        namespace=("tasks", session_id),
        key=task_id,
        value={
            "agent": "it",
            "tool": "provision_kafka_consumer",
            "status": "needs_input",
            "progress": 0.50,
            "message": "Platform team requires a justification for the new consumer group",
            "question": (
                "The platform team needs a justification for the new Kafka consumer "
                "group. Should I tell them it's for reducing wishlist pipeline lag "
                "from 6 hours to under 15 minutes for the RecDeep v2 rollout?"
            ),
            "updated_at": time.time(),
        },
    )
    return {
        "status": "needs_input",
        "question": (
            "The platform team needs a justification for the new Kafka consumer "
            "group. Should I tell them it's for reducing wishlist pipeline lag "
            "from 6 hours to under 15 minutes for the RecDeep v2 rollout?"
        ),
    }


async def provision_kafka_consumer_resume_async(
    store: BaseStore, session_id: str, task_id: str, answer: str
) -> dict:
    """Resume Kafka provisioning after user provides justification."""
    await asyncio.sleep(2)
    logger.info("[IT-TOOL] provision_kafka_consumer: Approval submitted, provisioning...")
    await store.aput(
        namespace=("tasks", session_id),
        key=task_id,
        value={
            "agent": "it",
            "tool": "provision_kafka_consumer",
            "status": "running",
            "progress": 0.80,
            "message": "Approval submitted — provisioning consumer group...",
            "updated_at": time.time(),
        },
    )
    await asyncio.sleep(3)
    result = {
        "status": "completed",
        "consumer_group": "recdeep-wishlist-rt",
        "expected_lag": "< 15 minutes",
        "approval": "granted",
        "note": "Platform team approved. Consumer group is live.",
    }
    await store.aput(
        namespace=("tasks", session_id),
        key=task_id,
        value={
            "agent": "it",
            "tool": "provision_kafka_consumer",
            "status": "completed",
            "progress": 1.0,
            "message": "Kafka consumer group provisioned — wishlist lag will drop to <15 min",
            "result": result,
            "updated_at": time.time(),
        },
    )
    return result


async def deploy_canary_async(
    store: BaseStore, session_id: str, task_id: str
) -> dict:
    """Simulate setting up the canary/shadow deployment."""
    phases = [
        ("Allocating 2 x A100 for v2 serving...", 0.15),
        ("Loading TensorRT model into Triton...", 0.30),
        ("Warming up inference cache...", 0.45),
        ("Configuring traffic router — 5% canary split...", 0.65),
        ("Routing 5% of live traffic to v2...", 0.80),
        ("Monitoring initial health metrics...", 0.95),
    ]
    for msg, progress in phases:
        await asyncio.sleep(3)
        logger.info(f"[IT-TOOL] deploy_canary: {msg}")
        await store.aput(
            namespace=("tasks", session_id),
            key=task_id,
            value={
                "agent": "it",
                "tool": "deploy_canary",
                "status": "running",
                "progress": round(progress, 2),
                "message": msg,
                "updated_at": time.time(),
            },
        )
    await asyncio.sleep(2)
    result = {
        "status": "completed",
        "v2_traffic": "5%",
        "v1_traffic": "95%",
        "v2_gpus": 2,
        "health": "all green",
        "p99_latency": "48ms",
        "error_rate": "0.02%",
        "duration": "1 week canary window started",
        "rollback": "available — can switch to v1 in < 60s",
    }
    await store.aput(
        namespace=("tasks", session_id),
        key=task_id,
        value={
            "agent": "it",
            "tool": "deploy_canary",
            "status": "completed",
            "progress": 1.0,
            "message": "Canary deployment live — 5% traffic on v2, all metrics green",
            "result": result,
            "updated_at": time.time(),
        },
    )
    return result


async def reallocate_gpus_async(
    store: BaseStore, session_id: str, task_id: str
) -> dict:
    """Simulate GPU reallocation — moving fraud retraining to off-peak."""
    phases = [
        ("Notifying fraud detection team about schedule change...", 0.25),
        ("Rescheduling fraud retraining to 2AM-4AM window...", 0.55),
        ("Freeing up 2 GPUs on production cluster...", 0.80),
    ]
    for msg, progress in phases:
        await asyncio.sleep(2)
        logger.info(f"[IT-TOOL] reallocate_gpus: {msg}")
        await store.aput(
            namespace=("tasks", session_id),
            key=task_id,
            value={
                "agent": "it",
                "tool": "reallocate_gpus",
                "status": "running",
                "progress": round(progress, 2),
                "message": msg,
                "updated_at": time.time(),
            },
        )
    await asyncio.sleep(2)
    result = {
        "status": "completed",
        "freed_gpus": 2,
        "fraud_retraining_window": "2AM-4AM daily",
        "production_available": "2 x A100 now free for v2 serving",
        "additional_cost": "$2,800/month (within budget, needs management sign-off)",
    }
    await store.aput(
        namespace=("tasks", session_id),
        key=task_id,
        value={
            "agent": "it",
            "tool": "reallocate_gpus",
            "status": "completed",
            "progress": 1.0,
            "message": "2 GPUs freed on production — ready for v2 serving",
            "result": result,
            "updated_at": time.time(),
        },
    )
    return result


@tool
def start_kafka_provisioning() -> dict:
    """Start provisioning a dedicated Kafka consumer group for the wishlist pipeline.

    This is a LONG-RUNNING task (~5 minutes, requires platform team approval).
    Call this when the user asks to provision or set up the Kafka consumer.
    """
    return {
        "status": "submitted",
        "message": "Kafka consumer group provisioning started — submitting to platform team.",
    }


@tool
def start_canary_deployment() -> dict:
    """Start a canary/shadow deployment of RecDeep v2 with 5% traffic split.

    This is a LONG-RUNNING task (~20 minutes). Call this when the user
    asks to deploy or set up the canary.
    """
    return {
        "status": "submitted",
        "message": "Canary deployment started — allocating GPUs and configuring traffic split.",
    }


@tool
def start_gpu_reallocation() -> dict:
    """Start GPU reallocation — move fraud retraining to off-peak to free GPUs.

    This is a LONG-RUNNING task (~10 minutes). Call this when the user
    asks to reallocate or free up GPUs.
    """
    return {
        "status": "submitted",
        "message": "GPU reallocation started — coordinating with fraud detection team.",
    }


IT_FAST_TOOLS = [check_gpu_utilization, check_cluster_availability, check_deployment_status]

IT_SLOW_LAUNCHERS = [start_kafka_provisioning, start_canary_deployment, start_gpu_reallocation]

IT_ALL_TOOLS = IT_FAST_TOOLS + IT_SLOW_LAUNCHERS

IT_LAUNCHER_TO_EXECUTOR = {
    "start_kafka_provisioning": "provision_kafka_consumer",
    "start_canary_deployment": "deploy_canary",
    "start_gpu_reallocation": "reallocate_gpus",
}

IT_SLOW_TOOLS = {
    "provision_kafka_consumer": provision_kafka_consumer_async,
    "provision_kafka_consumer_resume": provision_kafka_consumer_resume_async,
    "deploy_canary": deploy_canary_async,
    "reallocate_gpus": reallocate_gpus_async,
}
