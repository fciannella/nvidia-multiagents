"""Healthcare AI Diagnostics scenario.

Two agents — a Clinical AI Lead and a Hospital CTO — collaborate
to deploy ChestAI v3 (chest X-ray diagnostic AI) to a hospital network.
"""

import asyncio
import time
from typing import Any

from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from loguru import logger

from . import AgentConfig, ScenarioConfig, register_scenario

# ---------------------------------------------------------------------------
# Clinical AI Lead tools
# ---------------------------------------------------------------------------

@tool
def check_model_accuracy() -> dict:
    """Check the current accuracy metrics of ChestAI v3 across all pathologies.
    Returns sensitivity, specificity, and AUC per pathology category.
    """
    return {
        "model": "ChestAI v3",
        "overall_sensitivity": 0.942,
        "overall_specificity": 0.967,
        "overall_auc": 0.981,
        "per_pathology": {
            "pneumonia": {"sensitivity": 0.96, "specificity": 0.97, "auc": 0.99},
            "pneumothorax": {"sensitivity": 0.94, "specificity": 0.98, "auc": 0.98},
            "atelectasis": {"sensitivity": 0.91, "specificity": 0.95, "auc": 0.97},
            "cardiomegaly": {"sensitivity": 0.95, "specificity": 0.97, "auc": 0.98},
            "pleural_effusion": {"sensitivity": 0.93, "specificity": 0.96, "auc": 0.97},
            "fracture": {"sensitivity": 0.92, "specificity": 0.96, "auc": 0.97},
        },
        "false_positive_rate": "3.1% (target <5%)",
        "processing_time": "1.2s per image",
        "baseline_v2": {"sensitivity": 0.91, "pathologies": 8},
        "note": "v3 detects 14 pathologies (6 new) vs v2's 8 — transformer-based architecture",
    }


@tool
def check_validation_status() -> dict:
    """Check the progress of the 5000-case clinical validation study.
    Returns cases reviewed, pass rate, and remaining timeline.
    """
    return {
        "study": "ChestAI v3 Clinical Validation",
        "dataset_size": 5000,
        "cases_completed": 3200,
        "progress": "64%",
        "pass_rate": "94.8%",
        "partner_hospitals": {
            "Metro General": {"cases": 1200, "status": "complete"},
            "St. Luke's Medical": {"cases": 1100, "status": "complete"},
            "University Hospital": {"cases": 900, "status": "in progress"},
        },
        "remaining_cases": 1800,
        "estimated_completion": "2 weeks",
        "radiologist_agreement": "96.2% concordance with board-certified radiologists",
    }


@tool
def check_fda_status() -> dict:
    """Check the FDA 510(k) submission status for ChestAI v3.
    Returns submission milestones and current clearance status.
    """
    return {
        "submission_type": "510(k)",
        "predicate_device": "ChestAI v2 (K213456)",
        "status": "under review",
        "submitted": "2025-11-15",
        "milestones": {
            "pre_submission_meeting": "completed (2025-08-20)",
            "submission_filed": "completed (2025-11-15)",
            "acceptance_review": "completed — accepted (2025-12-01)",
            "substantive_review": "in progress",
        },
        "expected_clearance": "Q1 2026",
        "reviewer_questions": "1 pending — additional data on pleural effusion subgroup requested",
        "note": "Cannot deploy clinically until 510(k) clearance; research-use-only mode available",
    }


@tool
def start_validation_suite() -> dict:
    """Start the full clinical validation suite on the remaining 1800 cases.
    This is a LONG-RUNNING task (~30 minutes, runs AI inference + radiologist comparison).
    """
    return {
        "status": "submitted",
        "message": "Validation suite started — processing 1800 remaining cases with radiologist concordance check.",
    }


@tool
def start_bias_audit() -> dict:
    """Start a demographic bias audit across age, sex, and ethnicity cohorts.
    This is a LONG-RUNNING task (~20 minutes).
    """
    return {
        "status": "submitted",
        "message": "Bias audit started — evaluating model performance across demographic subgroups.",
    }


async def run_validation_suite_async(
    store: BaseStore, session_id: str, task_id: str
) -> dict:
    phases = [
        ("Loading 1800 remaining cases from partner hospitals...", 0.10),
        ("Running ChestAI v3 inference on batch 1/4 (450 cases)...", 0.25),
        ("Running ChestAI v3 inference on batch 2/4 (450 cases)...", 0.40),
        ("Running ChestAI v3 inference on batch 3/4 (450 cases)...", 0.55),
        ("Running ChestAI v3 inference on batch 4/4 (450 cases)...", 0.70),
        ("Comparing predictions against radiologist ground truth...", 0.85),
        ("Generating validation report and concordance statistics...", 0.95),
    ]
    for msg, progress in phases:
        await asyncio.sleep(2)
        logger.info(f"[CLINICAL-TOOL] run_validation_suite: {msg}")
        await store.aput(
            namespace=("tasks", session_id),
            key=task_id,
            value={
                "agent": "clinical_ai",
                "tool": "run_validation_suite",
                "status": "running",
                "progress": round(progress, 2),
                "message": msg,
                "updated_at": time.time(),
            },
        )
    await asyncio.sleep(3)
    result = {
        "status": "completed",
        "total_cases": 5000,
        "overall_sensitivity": 0.943,
        "overall_specificity": 0.968,
        "radiologist_concordance": "96.5%",
        "false_positive_rate": "2.9%",
        "false_negative_rate": "5.7%",
        "worst_pathology": "atelectasis (sensitivity 0.91)",
        "recommendation": "PASS — meets clinical deployment thresholds",
    }
    await store.aput(
        namespace=("tasks", session_id),
        key=task_id,
        value={
            "agent": "clinical_ai",
            "tool": "run_validation_suite",
            "status": "completed",
            "progress": 1.0,
            "message": "Validation complete — 96.5% concordance, all pathologies above threshold",
            "result": result,
            "updated_at": time.time(),
        },
    )
    return result


async def run_bias_audit_async(
    store: BaseStore, session_id: str, task_id: str
) -> dict:
    phases = [
        ("Stratifying dataset by age groups (18-40, 40-65, 65+)...", 0.15),
        ("Evaluating sensitivity/specificity by sex (M/F)...", 0.35),
        ("Evaluating sensitivity/specificity by ethnicity cohorts...", 0.55),
        ("Running statistical significance tests on subgroup differences...", 0.75),
        ("Generating demographic parity report...", 0.90),
    ]
    for msg, progress in phases:
        await asyncio.sleep(3)
        logger.info(f"[CLINICAL-TOOL] run_bias_audit: {msg}")
        await store.aput(
            namespace=("tasks", session_id),
            key=task_id,
            value={
                "agent": "clinical_ai",
                "tool": "run_bias_audit",
                "status": "running",
                "progress": round(progress, 2),
                "message": msg,
                "updated_at": time.time(),
            },
        )
    await asyncio.sleep(2)
    result = {
        "status": "completed",
        "age_groups": {
            "18-40": {"sensitivity": 0.95, "specificity": 0.97},
            "40-65": {"sensitivity": 0.94, "specificity": 0.97},
            "65+": {"sensitivity": 0.93, "specificity": 0.96},
        },
        "sex": {
            "male": {"sensitivity": 0.94, "specificity": 0.97},
            "female": {"sensitivity": 0.94, "specificity": 0.96},
        },
        "max_subgroup_variance": "1.8% (within 3% tolerance)",
        "statistical_significance": "No significant disparities detected (p > 0.05)",
        "recommendation": "PASS — no demographic bias concerns",
    }
    await store.aput(
        namespace=("tasks", session_id),
        key=task_id,
        value={
            "agent": "clinical_ai",
            "tool": "run_bias_audit",
            "status": "completed",
            "progress": 1.0,
            "message": "Bias audit complete — no significant demographic disparities found",
            "result": result,
            "updated_at": time.time(),
        },
    )
    return result


# ---------------------------------------------------------------------------
# Hospital CTO tools
# ---------------------------------------------------------------------------

@tool
def check_infrastructure_status() -> dict:
    """Check GPU, network, and storage health for the hospital AI infrastructure.
    Returns current resource status and availability.
    """
    return {
        "gpu_nodes": {
            "node_1": {"model": "A100 80GB", "status": "healthy", "utilization": "62%", "allocated_to": "ChestAI v2 (production)"},
            "node_2": {"model": "A100 80GB", "status": "healthy", "utilization": "15%", "allocated_to": "pathology AI (staging)"},
        },
        "network": {
            "internal_latency": "8ms",
            "pacs_link": "10 Gbps fiber",
            "ehr_link": "1 Gbps dedicated",
            "internet_isolated": True,
        },
        "storage": {
            "medical_imaging_nas": {"capacity": "120 TB", "used": "78 TB", "status": "healthy"},
            "model_registry": {"capacity": "2 TB", "used": "340 GB", "status": "healthy"},
        },
        "hipaa_compliance": "all nodes air-gapped, no cloud egress",
    }


@tool
def check_fhir_integration() -> dict:
    """Check the status of FHIR/HL7 integration with the hospital EHR system.
    Returns endpoint health, message throughput, and compatibility status.
    """
    return {
        "ehr_system": "Epic",
        "fhir_version": "R4",
        "endpoints": {
            "DiagnosticReport": {"status": "configured", "tested": True},
            "ImagingStudy": {"status": "configured", "tested": True},
            "Observation": {"status": "pending configuration", "tested": False},
        },
        "hl7_v2_feed": {
            "oru_messages": "operational — 400/day avg",
            "orm_messages": "operational",
        },
        "chestai_v3_integration": {
            "status": "not yet configured",
            "blocker": "Need to map v3's 14-pathology output to FHIR DiagnosticReport codes",
        },
    }


@tool
def check_pacs_status() -> dict:
    """Check the PACS integration status and current study routing queue.
    Returns routing rules, queue depth, and auto-forward configuration.
    """
    return {
        "pacs_vendor": "Horos / OsiriX",
        "studies_today": 387,
        "avg_daily": 400,
        "queue_depth": 12,
        "auto_forward_to_ai": {
            "chest_xray": {"enabled": True, "destination": "ChestAI v2 node"},
            "chest_ct": {"enabled": False, "note": "v3 supports X-ray only"},
        },
        "v3_routing": {
            "status": "not configured",
            "plan": "Update DICOM routing rules to forward to v3 inference node",
            "estimated_effort": "2-3 hours",
        },
        "dicom_conformance": "validated for v2 — needs re-validation for v3",
    }


@tool
def start_security_audit() -> dict:
    """Start a HIPAA security audit of the ChestAI v3 deployment environment.
    This is a LONG-RUNNING task (~15 minutes).
    """
    return {
        "status": "submitted",
        "message": "HIPAA security audit started — scanning network isolation, encryption, and access controls.",
    }


@tool
def start_pacs_integration() -> dict:
    """Start configuring PACS routing to forward chest X-rays to ChestAI v3 node.
    This is a LONG-RUNNING task (~10 minutes).
    """
    return {
        "status": "submitted",
        "message": "PACS integration started — updating DICOM routing rules and validating conformance.",
    }


async def run_security_audit_async(
    store: BaseStore, session_id: str, task_id: str
) -> dict:
    phases = [
        ("Scanning network isolation — verifying air-gap from internet...", 0.15),
        ("Checking encryption at rest (AES-256) on imaging storage...", 0.30),
        ("Validating TLS 1.3 on all internal FHIR/HL7 endpoints...", 0.45),
        ("Auditing role-based access controls and audit logs...", 0.60),
        ("Verifying PHI de-identification in model training pipeline...", 0.75),
        ("Running automated HIPAA compliance checklist (164 controls)...", 0.90),
    ]
    for msg, progress in phases:
        await asyncio.sleep(2)
        logger.info(f"[CTO-TOOL] run_security_audit: {msg}")
        await store.aput(
            namespace=("tasks", session_id),
            key=task_id,
            value={
                "agent": "hospital_cto",
                "tool": "run_security_audit",
                "status": "running",
                "progress": round(progress, 2),
                "message": msg,
                "updated_at": time.time(),
            },
        )
    await asyncio.sleep(3)
    result = {
        "status": "completed",
        "controls_checked": 164,
        "controls_passed": 161,
        "controls_failed": 0,
        "controls_warning": 3,
        "warnings": [
            "Audit log retention set to 1 year (recommend 3 years)",
            "Model registry lacks secondary backup location",
            "Service account password rotation overdue by 12 days",
        ],
        "hipaa_compliant": True,
        "recommendation": "PASS with 3 minor warnings — safe to proceed with deployment",
    }
    await store.aput(
        namespace=("tasks", session_id),
        key=task_id,
        value={
            "agent": "hospital_cto",
            "tool": "run_security_audit",
            "status": "completed",
            "progress": 1.0,
            "message": "Security audit complete — HIPAA compliant, 3 minor warnings",
            "result": result,
            "updated_at": time.time(),
        },
    )
    return result


async def run_pacs_integration_async(
    store: BaseStore, session_id: str, task_id: str
) -> dict:
    phases = [
        ("Backing up current DICOM routing configuration...", 0.15),
        ("Adding ChestAI v3 inference node as DICOM destination...", 0.30),
        ("Configuring modality worklist filter for chest X-ray studies...", 0.50),
        ("Running DICOM conformance validation on v3 endpoint...", 0.70),
        ("Sending 10 test studies through new routing path...", 0.85),
        ("Verifying results returned to PACS and EHR...", 0.95),
    ]
    for msg, progress in phases:
        await asyncio.sleep(2)
        logger.info(f"[CTO-TOOL] run_pacs_integration: {msg}")
        await store.aput(
            namespace=("tasks", session_id),
            key=task_id,
            value={
                "agent": "hospital_cto",
                "tool": "run_pacs_integration",
                "status": "running",
                "progress": round(progress, 2),
                "message": msg,
                "updated_at": time.time(),
            },
        )
    await asyncio.sleep(2)
    result = {
        "status": "completed",
        "dicom_routing": "chest X-ray → ChestAI v3 node (active)",
        "test_studies_sent": 10,
        "test_studies_passed": 10,
        "avg_round_trip": "1.8s (inference 1.2s + network 0.6s)",
        "ehr_integration": "results auto-posted to Epic DiagnosticReport",
        "recommendation": "PASS — ready for parallel run alongside v2",
    }
    await store.aput(
        namespace=("tasks", session_id),
        key=task_id,
        value={
            "agent": "hospital_cto",
            "tool": "run_pacs_integration",
            "status": "completed",
            "progress": 1.0,
            "message": "PACS integration complete — v3 routing active, 10/10 test studies passed",
            "result": result,
            "updated_at": time.time(),
        },
    )
    return result


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

CLINICAL_AI_SITUATION = """\
SCENARIO — ChestAI v3 HOSPITAL DEPLOYMENT:

You lead the clinical AI side of deploying ChestAI v3, a transformer-based
chest X-ray diagnostic AI that detects 14 pathologies. You're upgrading from
v2 (8 pathologies, 91% sensitivity) to v3 (14 pathologies, 94.2% sensitivity).

YOUR 3 ACTION ITEMS:
1. COMPLETE CLINICAL VALIDATION — finish the 5000-case validation study.
   3200 cases done across 3 partner hospitals, 1800 remaining. Need to hit
   >93% sensitivity and >95% radiologist concordance.
2. GET FDA UPDATE — 510(k) submission is under substantive review. One pending
   question from the reviewer about pleural effusion subgroup data. Need to
   prepare and submit the response.
3. TRAIN RADIOLOGY STAFF — coordinate with {{other_name}} on the AI-assisted
   workflow rollout. Radiologists need training on the new 14-pathology output
   and confidence scoring before go-live.

KEY FACTS YOU KNOW:
- v3 detects 6 new conditions (pleural effusion, cardiomegaly, atelectasis,
  fracture, lung opacity, consolidation) on top of v2's original 8
- False positive rate is 3.1% (target <5%), processing time 1.2s per image
- Validated at 3 partner hospitals (Metro General, St. Luke's, University Hospital)
- Model uses Vision Transformer architecture, ~120M parameters
- Cannot deploy clinically until FDA 510(k) clearance — research-use-only mode available

THINGS YOU NEED FROM {{other_name}}:
- "Is the on-prem GPU node ready for v3 inference?"
- "How's the PACS integration — can we auto-route chest X-rays to v3?"
- "What's the plan for FHIR mapping of the 14-pathology output?"
"""

HOSPITAL_CTO_SITUATION = """\
SCENARIO — ChestAI v3 HOSPITAL DEPLOYMENT:

You are responsible for deploying ChestAI v3 into the hospital's
on-premise infrastructure. Everything must be HIPAA-compliant with
zero cloud egress — all inference runs on-prem.

YOUR 3 ACTION ITEMS:
1. SET UP FHIR/HL7 INTEGRATION — map ChestAI v3's 14-pathology output
   to FHIR DiagnosticReport resources in Epic EHR. The Observation endpoint
   still needs configuration.
2. DEPLOY ON-PREM GPU NODE — v3 needs one A100 for inference. Node 2 is
   at 15% utilization and available. Need to run HIPAA security audit before
   deploying any new model.
3. CONFIGURE PACS INTEGRATION — update DICOM routing rules to auto-forward
   chest X-rays to the v3 inference node. Currently routing to v2.

KEY FACTS YOU KNOW:
- Hospital runs Epic EHR with FHIR R4 and HL7 v2 feeds
- 2 x A100 80GB on-prem: node 1 runs v2 (62% util), node 2 is mostly idle (15%)
- Current PACS handles ~400 studies/day, 10 Gbps fiber link
- Need 99.9% uptime SLA — any AI downtime means fallback to manual reads
- Network latency budget 200ms end-to-end (currently 8ms internal)
- All systems air-gapped from internet — no cloud, no external API calls

THINGS YOU NEED FROM {{other_name}}:
- "What's the validation status? Can we start integration testing?"
- "Is the FDA submission on track? We can't go clinical without clearance."
- "How much GPU memory does v3 need? Will it fit on a single A100?"

THINGS THAT NEED EXTERNAL COORDINATION:
- HIPAA security audit must pass before any new model is deployed
- Radiology department needs 2-week notice before workflow changes
- Epic FHIR configuration changes require a change management ticket
"""

SCENARIO = ScenarioConfig(
    id="healthcare",
    title="Healthcare AI Diagnostics",
    description="Deploy ChestAI v3 (chest X-ray diagnostic AI) to a hospital network — a Clinical AI Lead and Hospital CTO collaborate.",
    guide_overview=(
        "Priya and Marcus are deploying ChestAI v3, a transformer-based chest X-ray "
        "diagnostic AI that detects 14 pathologies (up from v2's 8). Priya leads the "
        "clinical AI side (model accuracy, clinical validation, FDA submission) while "
        "Marcus handles the hospital infrastructure (HIPAA compliance, PACS integration, "
        "GPU nodes, EHR integration). The model achieves 94.2% sensitivity but needs "
        "3,200 more validation cases, FDA 510(k) clearance, and HIPAA-compliant "
        "on-prem deployment with zero cloud egress."
    ),
    suggested_questions=[
        "Can you both introduce yourselves?",
        "Priya, what's the model accuracy looking like?",
        "How's the clinical validation progressing?",
        "What's the FDA submission status?",
        "Marcus, what's the infrastructure status?",
        "Marcus, can you run a HIPAA security audit?",
        "How's the PACS integration looking?",
        "Marcus, what's the FHIR integration status?",
        "Priya, can you start the bias audit?",
        "What are the blockers before we can deploy?",
    ],
    agents=[
        AgentConfig(
            id="clinical_ai",
            name="Priya",
            role="Clinical AI Lead",
            domain_keywords=[
                "medical imaging", "chest X-ray", "diagnostic AI",
                "clinical validation", "sensitivity", "specificity",
                "FDA", "HIPAA", "radiology", "model accuracy",
                "false positive", "false negative",
            ],
            voice_en="Magpie-Multilingual.EN-US.Aria",
            color="#76B900",
            situation=CLINICAL_AI_SITUATION,
            intro_example=(
                "Hi, I'm {name}. I lead the clinical AI side — developing "
                "and validating our diagnostic models. Nice to meet you."
            ),
            personality_quips=[
                "{other_name} will want three firewalls between the model and the internet — even though we're already air-gapped.",
                "I promise v3 won't melt your GPU, {other_name}. Probably.",
                "Classic {other_name} — running a security audit before I've even finished my coffee.",
                "I know, I know — another model update. But this one actually detects cardiomegaly.",
            ],
            fast_tools=[check_model_accuracy, check_validation_status, check_fda_status],
            slow_launchers=[start_validation_suite, start_bias_audit],
            launcher_to_executor={
                "start_validation_suite": "run_validation_suite",
                "start_bias_audit": "run_bias_audit",
            },
            slow_tools={
                "run_validation_suite": run_validation_suite_async,
                "run_bias_audit": run_bias_audit_async,
            },
        ),
        AgentConfig(
            id="hospital_cto",
            name="Marcus",
            role="Hospital CTO",
            domain_keywords=[
                "infrastructure", "HL7", "FHIR", "PACS", "network",
                "HIPAA", "security", "deployment", "uptime", "latency",
                "EHR integration", "disaster recovery",
            ],
            voice_en="Magpie-Multilingual.EN-US.Jason",
            color="#00B4D8",
            situation=HOSPITAL_CTO_SITUATION,
            intro_example=(
                "Hey, I'm {name}. I run the technology side at the hospital "
                "— infrastructure, security, making sure everything stays up. "
                "Good to meet you."
            ),
            personality_quips=[
                "{other_name} wants to deploy a new model every week — I want uptime.",
                "Sure, {other_name}, just push a 120-million-parameter model to my air-gapped node. What could go wrong?",
                "I'll have the PACS routing ready. Try not to change the output schema again.",
                "Another FDA question? Great, I'll just keep the infrastructure on standby...",
            ],
            fast_tools=[check_infrastructure_status, check_fhir_integration, check_pacs_status],
            slow_launchers=[start_security_audit, start_pacs_integration],
            launcher_to_executor={
                "start_security_audit": "run_security_audit",
                "start_pacs_integration": "run_pacs_integration",
            },
            slow_tools={
                "run_security_audit": run_security_audit_async,
                "run_pacs_integration": run_pacs_integration_async,
            },
        ),
    ],
)

register_scenario(SCENARIO)
