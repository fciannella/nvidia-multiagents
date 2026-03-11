"""E-commerce Recommendation System v2 Rollout scenario.

Two agents — a Data Scientist and an IT Infrastructure Lead — collaborate
to roll out RecDeep v2, replacing the old RecLite v1 recommendation model.
"""

import asyncio
import time
from typing import Any

from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from loguru import logger

from . import AgentConfig, ScenarioConfig, register_scenario

# ---------------------------------------------------------------------------
# Data Scientist tools
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


@tool
def start_training_job() -> dict:
    """Start a full training run of RecDeep v2 on the complete dataset.
    This is a LONG-RUNNING task (~16 hours, 20 epochs).
    """
    return {
        "status": "submitted",
        "message": "Training job submitted — 20 epochs on full dataset, estimated 16 hours.",
    }


@tool
def start_offline_eval() -> dict:
    """Start the offline evaluation suite (CTR, NDCG, revenue lift).
    This is a LONG-RUNNING task (~15 minutes).
    """
    return {
        "status": "submitted",
        "message": "Offline evaluation started — computing CTR, NDCG, and revenue metrics.",
    }


@tool
def start_model_export() -> dict:
    """Start exporting the model to TensorRT/Triton format.
    This is a LONG-RUNNING task (~10 minutes).
    """
    return {
        "status": "submitted",
        "message": "Model export started — converting to TensorRT FP16 for Triton.",
    }


async def run_training_job_async(
    store: BaseStore, session_id: str, task_id: str
) -> dict:
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


# ---------------------------------------------------------------------------
# IT Infrastructure tools
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


@tool
def start_kafka_provisioning() -> dict:
    """Start provisioning a dedicated Kafka consumer group for the wishlist pipeline.
    This is a LONG-RUNNING task (~5 minutes, requires platform team approval).
    """
    return {
        "status": "submitted",
        "message": "Kafka consumer group provisioning started — submitting to platform team.",
    }


@tool
def start_canary_deployment() -> dict:
    """Start a canary/shadow deployment of RecDeep v2 with 5% traffic split.
    This is a LONG-RUNNING task (~20 minutes).
    """
    return {
        "status": "submitted",
        "message": "Canary deployment started — allocating GPUs and configuring traffic split.",
    }


@tool
def start_gpu_reallocation() -> dict:
    """Start GPU reallocation — move fraud retraining to off-peak to free GPUs.
    This is a LONG-RUNNING task (~10 minutes).
    """
    return {
        "status": "submitted",
        "message": "GPU reallocation started — coordinating with fraud detection team.",
    }


async def provision_kafka_consumer_async(
    store: BaseStore, session_id: str, task_id: str
) -> dict:
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


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

DS_SITUATION = """\
SCENARIO — RecDeep v2 ROLLOUT:

You lead the ML side of replacing the old recommendation model (RecLite v1)
with a new one (RecDeep v2). v2 is a bigger, smarter model that handles
new users better and uses more signals (search, wishlists, browsing).

YOUR 3 ACTION ITEMS:
1. RUN FINAL TRAINING — one full training run on the complete dataset.
   Needs 4 x A100 GPUs for about 16 hours. You need {{other_name}}'s staging cluster.
2. RUN EVALUATION — after training, run the offline eval suite (~2 hours).
   Expecting +18% CTR and +23% revenue lift over v1.
3. FIX WISHLIST LAG — the wishlist data pipeline has a 6-hour lag; needs to be
   under 15 minutes. You need {{other_name}} to provision a Kafka consumer group.

KEY FACTS YOU KNOW:
- v1 serves 14M daily users, CTR stuck at 3.2%, revenue $0.47/session
- v2 is ~340M params (v1 was ~8M) — needs GPU inference (Triton/TensorRT)
- 3 test training runs already done, best val NDCG = 0.38 (v1 was 0.29)
- After eval, need a 1-week shadow deployment before going live

THINGS YOU NEED FROM {{other_name}}:
- "Is the staging cluster available for training?"
- "Can you provision the Kafka consumer group for the wishlist pipeline?"
- "How are we handling serving — the model is 40x bigger than v1?"
"""

IT_SITUATION = """\
SCENARIO — RecDeep v2 ROLLOUT:

You lead the infrastructure and deployment side of replacing the old
recommendation model (RecLite v1) with RecDeep v2.

YOUR 3 ACTION ITEMS:
1. ALLOCATE STAGING CLUSTER — give {{other_name}} the 4-GPU staging cluster for
   training. It's available starting Monday.
2. SET UP SERVING — v2 needs 2 x A100 GPUs on Triton with TensorRT.
   You need to free up 1 GPU by moving fraud retraining to off-peak (2AM-4AM).
3. DEPLOY CANARY — set up shadow deployment routing 5% of traffic to v2
   for 1 week, while v1 stays live.

KEY FACTS YOU KNOW:
- Production: 8 x A100 (v1 uses 1, fraud uses 4, search uses 3)
- Staging: 4 x A100, currently idle — available for {{other_name}}'s training
- Blue-green rollback ready: can switch back to v1 in under 60 seconds
- Timeline: training Mon-Tue, eval Wed, integration Thu-Fri, canary Week 2

THINGS YOU NEED FROM {{other_name}}:
- "How much GPU memory does v2 need at inference?"
- "Is the model exported to Triton format yet?"
- "Is the feature pipeline ready for shadow deployment?"

THINGS THAT NEED EXTERNAL APPROVAL:
- Kafka consumer group for {{other_name}}'s wishlist fix: needs 24-hour platform
  team approval
- Extra GPU cost for v2 serving: ~$2,800/month, needs management sign-off
"""

SCENARIO = ScenarioConfig(
    id="ecommerce",
    title="E-commerce ML Deployment",
    description="Roll out a new recommendation model (RecDeep v2) with a Data Scientist and IT Infrastructure Lead.",
    guide_overview=(
        "Sarah and Mike are rolling out RecDeep v2, a next-generation recommendation "
        "model replacing the legacy RecLite v1. Sarah handles the ML side (training, "
        "evaluation, data pipelines) while Mike manages infrastructure (GPU clusters, "
        "Kafka provisioning, canary deployment). The model is 40x larger than v1 and "
        "promises +18% CTR and +23% revenue lift — but they need to coordinate GPU "
        "allocation, fix a 6-hour wishlist pipeline lag, and run a shadow deployment "
        "before going live."
    ),
    suggested_questions=[
        "Can you both introduce yourselves?",
        "What's the current status of the model training?",
        "How are the model metrics compared to v1?",
        "Sarah, is the data pipeline ready?",
        "Mike, can you check the GPU cluster availability?",
        "Can you deploy a canary with 5% traffic?",
        "Mike, can you provision the Kafka consumer group?",
        "What's the deployment status right now?",
        "Sarah, can you start the offline evaluation?",
        "What are the blockers before we go live?",
    ],
    agents=[
        AgentConfig(
            id="data_scientist",
            name="Sarah",
            role="Data Scientist",
            domain_keywords=[
                "ML", "models", "training", "data pipelines", "data quality",
                "evaluation", "MLOps", "feature engineering", "wishlist",
                "browsing data", "RecDeep", "RecLite", "NDCG", "CTR",
            ],
            voice_en="Magpie-Multilingual.EN-US.Aria",
            color="#76B900",
            situation=DS_SITUATION,
            intro_example=(
                "Hi, I'm {name}. I lead the ML and data science side here, "
                "focused on recommendation systems. Great to meet you."
            ),
            personality_quips=[
                'Classic {other_name} — already thinking about rollback before we even deploy.',
                "I'll believe the cluster is ready when I see it, {other_name}.",
                "Don't worry, I'll keep the model small enough for your precious GPUs.",
                "I know, I know, another training run...",
            ],
            fast_tools=[check_training_status, get_model_metrics, check_data_pipeline],
            slow_launchers=[start_training_job, start_offline_eval, start_model_export],
            launcher_to_executor={
                "start_training_job": "run_training_job",
                "start_offline_eval": "run_offline_eval",
                "start_model_export": "export_model_to_triton",
            },
            slow_tools={
                "run_training_job": run_training_job_async,
                "run_offline_eval": run_offline_eval_async,
                "export_model_to_triton": export_model_to_triton_async,
            },
        ),
        AgentConfig(
            id="it",
            name="Mike",
            role="IT Infrastructure Lead",
            domain_keywords=[
                "infrastructure", "GPUs", "clusters", "deployment",
                "Kafka provisioning", "security", "timelines", "budget",
                "Triton", "TensorRT", "canary", "rollback",
            ],
            voice_en="Magpie-Multilingual.EN-US.Jason",
            color="#00B4D8",
            situation=IT_SITUATION,
            intro_example=(
                "Hey, I'm {name}. I handle the infrastructure and deployment "
                "side — making sure everything runs smoothly in production. Nice to meet you."
            ),
            personality_quips=[
                "Sure, {other_name}, just 340 million parameters. No big deal for my cluster.",
                "Data scientists and their 'just one more training run'... I've heard that before.",
                "I'll have the GPUs ready. Try not to set them on fire this time.",
                "Yeah, I'm the person who worries about everything breaking at 3 AM.",
            ],
            fast_tools=[check_gpu_utilization, check_cluster_availability, check_deployment_status],
            slow_launchers=[start_kafka_provisioning, start_canary_deployment, start_gpu_reallocation],
            launcher_to_executor={
                "start_kafka_provisioning": "provision_kafka_consumer",
                "start_canary_deployment": "deploy_canary",
                "start_gpu_reallocation": "reallocate_gpus",
            },
            slow_tools={
                "provision_kafka_consumer": provision_kafka_consumer_async,
                "provision_kafka_consumer_resume": provision_kafka_consumer_resume_async,
                "deploy_canary": deploy_canary_async,
                "reallocate_gpus": reallocate_gpus_async,
            },
        ),
    ],
)

register_scenario(SCENARIO)
