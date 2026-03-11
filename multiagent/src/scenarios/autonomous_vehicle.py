"""Autonomous Vehicle Safety Review scenario.

Two agents — a Perception Engineer and a Safety & Compliance Lead — collaborate
on a pre-release safety review for AV perception stack v4.2.
"""

import asyncio
import time
from typing import Any

from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from loguru import logger

from . import AgentConfig, ScenarioConfig, register_scenario

# ---------------------------------------------------------------------------
# Perception Engineer tools
# ---------------------------------------------------------------------------


@tool
def check_detection_metrics() -> dict:
    """Check the latest 3D object detection metrics for TransformerDet v4.2.
    Returns mAP, latency, and FPS compared to PointPillars v3 baseline.
    """
    return {
        "model": "TransformerDet v4.2",
        "baseline_model": "PointPillars v3",
        "parameters": "85M",
        "platform": "DRIVE Orin",
        "mAP": {"v4.2": 81.7, "v3": 72.4, "lift": "+12.8%"},
        "latency_ms": {"v4.2": 48, "v3": 35, "budget": 50},
        "fps": {"v4.2": 20.8, "v3": 28.6},
        "note": "Latency within budget but only 2ms headroom — monitor closely",
    }


@tool
def check_sensor_health() -> dict:
    """Check health status of all perception sensors (LiDAR, cameras, radar).
    Returns sensor status, calibration state, and sync drift.
    """
    return {
        "lidar": {
            "model": "Velodyne VLP-128",
            "status": "healthy",
            "point_density": "~230k pts/frame",
            "calibration": "valid (last calibrated 3 days ago)",
        },
        "cameras": {
            "count": 8,
            "status": "healthy",
            "resolution": "1920x1200",
            "sync_drift_ms": 12,
            "required_drift_ms": 5,
            "issue": "Camera-LiDAR sync drift at 12ms — needs fix before release",
        },
        "radar": {
            "count": 5,
            "status": "healthy",
            "range": "250m",
        },
        "overall": "degraded — camera-LiDAR sync drift exceeds 5ms threshold",
    }


@tool
def check_edge_case_coverage() -> dict:
    """Check detection pass rates for critical edge-case scenarios.
    Returns pass rates for rain, fog, nighttime, and other challenging conditions.
    """
    return {
        "total_scenarios_tested": 4200,
        "overall_pass_rate": "78.6%",
        "conditions": {
            "rain_heavy": {"pass_rate": "82.1%", "status": "pass", "threshold": "80%"},
            "fog_dense": {"pass_rate": "68.0%", "status": "FAIL", "threshold": "75%"},
            "nighttime_pedestrians": {"pass_rate": "74.2%", "status": "FAIL", "threshold": "75%"},
            "construction_zones": {"pass_rate": "88.5%", "status": "pass", "threshold": "80%"},
            "glare_direct_sun": {"pass_rate": "79.8%", "status": "pass", "threshold": "75%"},
        },
        "blocking_failures": ["fog_dense", "nighttime_pedestrians"],
        "note": "2 edge-case categories below threshold — must resolve before release",
    }


@tool
def start_perception_benchmark() -> dict:
    """Start a full benchmark run of TransformerDet v4.2 on the 10k-scenario test suite.
    This is a LONG-RUNNING task (~8 hours on DRIVE Orin cluster).
    """
    return {
        "status": "submitted",
        "message": "Benchmark submitted — 10k scenarios on DRIVE Orin cluster, estimated 8 hours.",
    }


@tool
def start_sensor_calibration() -> dict:
    """Start an automated sensor calibration pass to fix camera-LiDAR sync drift.
    This is a LONG-RUNNING task (~20 minutes).
    """
    return {
        "status": "submitted",
        "message": "Sensor calibration started — targeting camera-LiDAR sync drift reduction to <5ms.",
    }


async def run_perception_benchmark_async(
    store: BaseStore, session_id: str, task_id: str
) -> dict:
    phases = [
        ("Loading 10k scenario test suite...", 0.05),
        ("Running daytime urban scenarios (1/5)...", 0.20),
        ("Running highway scenarios (2/5)...", 0.40),
        ("Running adverse weather scenarios (3/5)...", 0.55),
        ("Running nighttime scenarios (4/5)...", 0.70),
        ("Running construction & edge cases (5/5)...", 0.85),
        ("Aggregating metrics and generating report...", 0.95),
    ]
    for msg, progress in phases:
        await asyncio.sleep(3)
        logger.info(f"[PE-TOOL] run_perception_benchmark: {msg}")
        await store.aput(
            namespace=("tasks", session_id),
            key=task_id,
            value={
                "agent": "perception_engineer",
                "tool": "run_perception_benchmark",
                "status": "running",
                "progress": round(progress, 2),
                "message": msg,
                "updated_at": time.time(),
            },
        )
    await asyncio.sleep(2)
    result = {
        "status": "completed",
        "scenarios_run": 10000,
        "overall_mAP": 81.7,
        "latency_p50_ms": 45,
        "latency_p99_ms": 48,
        "fog_detection_rate": "68.0%",
        "night_pedestrian_rate": "74.2%",
        "recommendation": "CONDITIONAL PASS — fog and nighttime pedestrian rates below threshold",
    }
    await store.aput(
        namespace=("tasks", session_id),
        key=task_id,
        value={
            "agent": "perception_engineer",
            "tool": "run_perception_benchmark",
            "status": "completed",
            "progress": 1.0,
            "message": "Benchmark complete — 10k scenarios evaluated, 2 edge-case categories still below threshold",
            "result": result,
            "updated_at": time.time(),
        },
    )
    return result


async def run_sensor_calibration_async(
    store: BaseStore, session_id: str, task_id: str
) -> dict:
    phases = [
        ("Capturing calibration targets from all 8 cameras...", 0.15),
        ("Computing LiDAR-to-camera extrinsic transforms...", 0.35),
        ("Optimizing temporal alignment parameters...", 0.55),
        ("Applying corrected sync offsets...", 0.75),
        ("Validating sync drift on live sensor feed...", 0.90),
    ]
    for msg, progress in phases:
        await asyncio.sleep(2)
        logger.info(f"[PE-TOOL] run_sensor_calibration: {msg}")
        await store.aput(
            namespace=("tasks", session_id),
            key=task_id,
            value={
                "agent": "perception_engineer",
                "tool": "run_sensor_calibration",
                "status": "running",
                "progress": round(progress, 2),
                "message": msg,
                "updated_at": time.time(),
            },
        )
    await asyncio.sleep(2)
    result = {
        "status": "completed",
        "sync_drift_before_ms": 12,
        "sync_drift_after_ms": 3.8,
        "threshold_ms": 5,
        "cameras_calibrated": 8,
        "recommendation": "PASS — sync drift reduced to 3.8ms, within 5ms threshold",
    }
    await store.aput(
        namespace=("tasks", session_id),
        key=task_id,
        value={
            "agent": "perception_engineer",
            "tool": "run_sensor_calibration",
            "status": "completed",
            "progress": 1.0,
            "message": "Calibration complete — camera-LiDAR sync drift reduced to 3.8ms",
            "result": result,
            "updated_at": time.time(),
        },
    )
    return result


# ---------------------------------------------------------------------------
# Safety & Compliance Lead tools
# ---------------------------------------------------------------------------


@tool
def check_safety_score() -> dict:
    """Check the current ASIL safety rating and SOTIF gap analysis status.
    Returns ASIL level, gap count, and outstanding risk items.
    """
    return {
        "target_asil": "ASIL-B",
        "v3_status": "certified ASIL-B",
        "v4_2_status": "pending re-certification",
        "sotif_gaps_identified": 7,
        "sotif_gaps_resolved": 4,
        "sotif_gaps_remaining": 3,
        "critical_gaps": [
            "Fog scenario detection rate below 75% threshold",
            "Nighttime pedestrian detection below 75% threshold",
            "Camera-LiDAR sync drift exceeds 5ms tolerance",
        ],
        "note": "Cannot certify until all 3 critical gaps are resolved",
    }


@tool
def check_compliance_status() -> dict:
    """Check the ISO 26262 compliance package status.
    Returns checklist items and completion percentage.
    """
    return {
        "standard": "ISO 26262",
        "target_level": "ASIL-B",
        "checklist": {
            "hardware_safety_requirements": {"status": "complete", "sign_off": True},
            "software_safety_requirements": {"status": "complete", "sign_off": True},
            "safety_analysis_fmea": {"status": "complete", "sign_off": True},
            "safety_validation_plan": {"status": "in_progress", "sign_off": False},
            "sotif_analysis_report": {"status": "in_progress", "sign_off": False},
            "field_test_report": {"status": "pending", "sign_off": False},
        },
        "completion": "50% (3/6 items signed off)",
        "blockers": ["SOTIF analysis depends on edge-case resolution", "Field test report needs 500 hours"],
    }


@tool
def check_field_test_results() -> dict:
    """Check the status of the field test campaign.
    Returns hours completed, fleet status, and scenario failure details.
    """
    return {
        "fleet_size": 12,
        "vehicles_active": 10,
        "vehicles_maintenance": 2,
        "hours_completed": 320,
        "hours_required": 500,
        "hours_remaining": 180,
        "estimated_completion": "~3 weeks at current pace",
        "critical_failures": {
            "count": 2,
            "details": [
                "Vehicle 07: false negative on cyclist in fog at 40m range",
                "Vehicle 11: 200ms latency spike during heavy rain transition",
            ],
        },
        "disengagement_rate": "0.8 per 1000 miles (target: < 1.0)",
    }


@tool
def start_sotif_analysis() -> dict:
    """Start a full SOTIF (Safety of the Intended Functionality) gap analysis for v4.2.
    This is a LONG-RUNNING task (~30 minutes).
    """
    return {
        "status": "submitted",
        "message": "SOTIF gap analysis started — evaluating all known and unknown hazard scenarios.",
    }


@tool
def start_field_test_campaign() -> dict:
    """Start a new batch of the field test campaign (dispatch vehicles for next 48-hour block).
    This is a LONG-RUNNING task (~15 minutes to coordinate fleet dispatch).
    """
    return {
        "status": "submitted",
        "message": "Field test campaign batch started — dispatching 10 vehicles for 48-hour block.",
    }


async def run_sotif_analysis_async(
    store: BaseStore, session_id: str, task_id: str
) -> dict:
    phases = [
        ("Cataloging known hazardous scenarios from v4.2 test results...", 0.15),
        ("Analyzing triggering conditions for perception failures...", 0.30),
        ("Evaluating unknown-unsafe scenarios via adversarial simulation...", 0.50),
        ("Computing residual risk for each hazard category...", 0.70),
        ("Cross-referencing with ISO 21448 requirements...", 0.85),
        ("Generating SOTIF gap report...", 0.95),
    ]
    for msg, progress in phases:
        await asyncio.sleep(3)
        logger.info(f"[SL-TOOL] run_sotif_analysis: {msg}")
        await store.aput(
            namespace=("tasks", session_id),
            key=task_id,
            value={
                "agent": "safety_lead",
                "tool": "run_sotif_analysis",
                "status": "running",
                "progress": round(progress, 2),
                "message": msg,
                "updated_at": time.time(),
            },
        )
    await asyncio.sleep(2)
    result = {
        "status": "completed",
        "total_hazards_analyzed": 42,
        "known_unsafe_resolved": 18,
        "known_unsafe_remaining": 3,
        "unknown_unsafe_identified": 2,
        "residual_risk_level": "medium",
        "blocking_items": [
            "Fog detection rate (68%) must reach 75% to close SOTIF gap #5",
            "Nighttime pedestrian rate (74.2%) must reach 75% to close SOTIF gap #6",
        ],
        "recommendation": "CONDITIONAL — resolve 2 blocking items, then re-run for final certification",
    }
    await store.aput(
        namespace=("tasks", session_id),
        key=task_id,
        value={
            "agent": "safety_lead",
            "tool": "run_sotif_analysis",
            "status": "completed",
            "progress": 1.0,
            "message": "SOTIF analysis complete — 2 blocking items remain before certification",
            "result": result,
            "updated_at": time.time(),
        },
    )
    return result


async def run_field_test_campaign_async(
    store: BaseStore, session_id: str, task_id: str
) -> dict:
    phases = [
        ("Pre-flight checks on 10 vehicles...", 0.10),
        ("Uploading v4.2 perception stack to fleet...", 0.25),
        ("Dispatching vehicles to designated test routes...", 0.40),
        ("Vehicles en route — monitoring telemetry...", 0.55),
        ("Collecting data from first batch of test runs...", 0.70),
        ("Analyzing disengagement events...", 0.85),
        ("Compiling field test batch report...", 0.95),
    ]
    for msg, progress in phases:
        await asyncio.sleep(2)
        logger.info(f"[SL-TOOL] run_field_test_campaign: {msg}")
        await store.aput(
            namespace=("tasks", session_id),
            key=task_id,
            value={
                "agent": "safety_lead",
                "tool": "run_field_test_campaign",
                "status": "running",
                "progress": round(progress, 2),
                "message": msg,
                "updated_at": time.time(),
            },
        )
    await asyncio.sleep(2)
    result = {
        "status": "completed",
        "batch_hours_added": 48,
        "total_hours_now": 368,
        "hours_remaining": 132,
        "vehicles_dispatched": 10,
        "new_disengagements": 3,
        "disengagement_rate": "0.7 per 1000 miles",
        "critical_events": "0 new critical events in this batch",
        "note": "On track — estimated 132 hours remaining (~2.5 weeks)",
    }
    await store.aput(
        namespace=("tasks", session_id),
        key=task_id,
        value={
            "agent": "safety_lead",
            "tool": "run_field_test_campaign",
            "status": "completed",
            "progress": 1.0,
            "message": "Field test batch complete — 48 hours added, total now 368/500",
            "result": result,
            "updated_at": time.time(),
        },
    )
    return result


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

PE_SITUATION = """\
SCENARIO — TransformerDet v4.2 SAFETY REVIEW:

You lead the perception stack upgrade from PointPillars v3 to TransformerDet v4.2,
a transformer-based 3D object detection model (85M params) running on DRIVE Orin.

YOUR 3 ACTION ITEMS:
1. RUN FINAL BENCHMARK — full 10k-scenario benchmark on the DRIVE Orin cluster.
   v4.2 shows mAP=81.7% (up from v3's 72.4%) but fog scenarios are still at 68%.
2. FIX CAMERA-LiDAR SYNC DRIFT — currently 12ms, needs to be under 5ms.
   Run calibration pass across all 8 cameras.
3. VALIDATE CORNER CASES — rain, fog, nighttime pedestrians all need to pass
   at ≥75% detection rate. Fog is at 68%, night pedestrians at 74.2%.

KEY FACTS YOU KNOW:
- v3 mAP=72.4%, v4.2 test mAP=81.7% — significant improvement
- v4.2 latency is 48ms (budget is 50ms) — tight but within spec
- 85M params, runs on DRIVE Orin at ~20.8 FPS
- Fog detection at 68% is the biggest blocker — needs targeted retraining
- {{other_name}} won't sign off until all edge cases pass and SOTIF gaps close

THINGS YOU NEED FROM {{other_name}}:
- "What's the latest SOTIF gap count — how many are we still blocking on?"
- "Are the field test hours on track? We need 500 for certification."
- "Can we get conditional approval to ship if fog improves to 73% but not 75%?"
"""

SL_SITUATION = """\
SCENARIO — TransformerDet v4.2 SAFETY REVIEW:

You are responsible for certifying the perception stack upgrade for ASIL-B
compliance, managing SOTIF analysis, and overseeing the field test campaign.

YOUR 3 ACTION ITEMS:
1. COMPLETE SOTIF GAP ANALYSIS — v4.2 introduces new failure modes that need
   evaluation against ISO 21448. Currently 3 of 7 gaps remain open.
2. RUN 500-HOUR FIELD TEST — fleet of 12 vehicles, currently at 320 hours.
   Need 180 more hours. 2 critical scenario failures still under investigation.
3. PREPARE ISO 26262 PACKAGE — compliance checklist is 50% complete.
   Blocked on SOTIF analysis and field test report.

KEY FACTS YOU KNOW:
- v3 passed ASIL-B certification — v4.2 needs full re-certification
- Field test fleet: 12 vehicles (10 active, 2 in maintenance)
- Current field hours: 320/500, ~3 weeks to complete at current pace
- 2 critical failures: false negative on cyclist in fog, latency spike in rain
- {{other_name}}'s fog detection rate of 68% is a hard blocker for SOTIF gap #5
- Camera-LiDAR sync drift of 12ms violates sensor fusion tolerance

THINGS YOU NEED FROM {{other_name}}:
- "What's the plan to get fog detection from 68% to 75%?"
- "When will the camera-LiDAR sync drift be fixed?"
- "Can you run the full 10k benchmark so I can update the SOTIF report?"

THINGS THAT NEED EXTERNAL APPROVAL:
- ASIL-B re-certification: requires all SOTIF gaps closed + 500 field test hours
- Conditional release: possible with management waiver, but only if all critical
  scenarios pass at ≥75%
"""

SCENARIO = ScenarioConfig(
    id="autonomous_vehicle",
    title="Autonomous Vehicle Safety Review",
    description="Pre-release safety review for AV perception stack v4.2 — a Perception Engineer and Safety Lead collaborate.",
    guide_overview=(
        "Alex and Jordan are doing a pre-release safety review for TransformerDet v4.2, "
        "a new 85M-parameter 3D object detection model for autonomous vehicles running on "
        "NVIDIA DRIVE Orin. Alex handles the perception stack (sensor fusion, detection "
        "accuracy, calibration) while Jordan manages safety certification (SOTIF analysis, "
        "ISO 26262 compliance, field testing). The model's mAP is up 12.8% but fog "
        "detection is only 68% (needs 75%), the camera-LiDAR sync drifts 12ms (needs <5ms), "
        "and they still need 180 more hours of field testing for certification."
    ),
    suggested_questions=[
        "Can you both introduce yourselves?",
        "What are the latest detection metrics for v4.2?",
        "Alex, how's the sensor health looking?",
        "What's the edge case coverage — especially fog?",
        "Jordan, what's the current safety score?",
        "How's the field testing going?",
        "Jordan, can you run the SOTIF analysis?",
        "Alex, can you start the full perception benchmark?",
        "What's the compliance status for ISO 26262?",
        "What are the blockers before we can certify?",
    ],
    agents=[
        AgentConfig(
            id="perception_engineer",
            name="Alex",
            role="Perception Engineer",
            domain_keywords=[
                "sensors", "LiDAR", "camera", "3D detection", "point cloud",
                "sensor fusion", "calibration", "PointPillars", "transformer",
                "latency", "FPS",
            ],
            voice_en="Magpie-Multilingual.EN-US.Jason",
            color="#76B900",
            situation=PE_SITUATION,
            intro_example=(
                "Hi, I'm {name}. I lead the perception stack — cameras, LiDAR, "
                "sensor fusion, the works. Good to meet you."
            ),
            personality_quips=[
                "Jordan will want twelve more edge-case tests before breakfast.",
                "I know, I know, {other_name} — 68% fog detection keeps you up at night.",
                "We're 2ms under the latency budget. Relax, {other_name}.",
                "If {other_name} had it their way, we'd test in a blizzard on Mars.",
            ],
            fast_tools=[check_detection_metrics, check_sensor_health, check_edge_case_coverage],
            slow_launchers=[start_perception_benchmark, start_sensor_calibration],
            launcher_to_executor={
                "start_perception_benchmark": "run_perception_benchmark",
                "start_sensor_calibration": "run_sensor_calibration",
            },
            slow_tools={
                "run_perception_benchmark": run_perception_benchmark_async,
                "run_sensor_calibration": run_sensor_calibration_async,
            },
        ),
        AgentConfig(
            id="safety_lead",
            name="Jordan",
            role="Safety & Compliance Lead",
            domain_keywords=[
                "safety", "SOTIF", "ISO 26262", "compliance", "edge cases",
                "regulation", "field test", "ODD", "ASIL", "validation",
                "risk assessment",
            ],
            voice_en="Magpie-Multilingual.EN-US.Aria",
            color="#00B4D8",
            situation=SL_SITUATION,
            intro_example=(
                "Hey, I'm {name}. I handle safety and compliance — making sure "
                "we don't ship anything that isn't rock solid."
            ),
            personality_quips=[
                "Sure, {other_name}, 81% mAP looks great — until someone's in a fog bank.",
                "{other_name} loves pushing detection accuracy while I chase edge cases.",
                "I'll sign off when the numbers say so, not when {other_name} says 'trust me'.",
                "Two milliseconds of headroom is not a safety margin, {other_name}.",
            ],
            fast_tools=[check_safety_score, check_compliance_status, check_field_test_results],
            slow_launchers=[start_sotif_analysis, start_field_test_campaign],
            launcher_to_executor={
                "start_sotif_analysis": "run_sotif_analysis",
                "start_field_test_campaign": "run_field_test_campaign",
            },
            slow_tools={
                "run_sotif_analysis": run_sotif_analysis_async,
                "run_field_test_campaign": run_field_test_campaign_async,
            },
        ),
    ],
)

register_scenario(SCENARIO)
