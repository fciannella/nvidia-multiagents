"""Game Studio AI Characters scenario.

Two agents — an AI Character Designer and a Technical Director — collaborate
to ship AI-driven NPCs for the open-world RPG 'Echoes of Aethermoor'.
"""

import asyncio
import time
from typing import Any

from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from loguru import logger

from . import AgentConfig, ScenarioConfig, register_scenario

# ---------------------------------------------------------------------------
# AI Character Designer tools
# ---------------------------------------------------------------------------

@tool
def check_dialogue_quality() -> dict:
    """Check the latest dialogue quality metrics for the neural dialogue system.
    Returns coherence, personality consistency, and lore accuracy scores.
    """
    return {
        "model": "NeuralDialogue v3 (400M distilled)",
        "test_conversations": 3000,
        "coherence": {"score": 0.87, "target": 0.92, "delta": "-5.4%"},
        "personality_consistency": {"score": 0.91, "target": 0.95, "delta": "-4.2%"},
        "lore_accuracy": {"score": 0.94, "target": 0.95, "delta": "-1.1%"},
        "avg_turn_length": "42 tokens",
        "note": "Coherence drops after 8+ turns — likely related to emotion drift bug",
    }


@tool
def check_npc_behavior_tests() -> dict:
    """Check the behavior tree test suite results for all NPC archetypes.
    Returns pass rates per archetype category and known failures.
    """
    return {
        "total_tests": 847,
        "passed": 793,
        "failed": 54,
        "pass_rate": "93.6%",
        "by_category": {
            "combat_npcs": {"passed": 210, "total": 215, "rate": "97.7%"},
            "merchant_npcs": {"passed": 178, "total": 182, "rate": "97.8%"},
            "quest_givers": {"passed": 156, "total": 175, "rate": "89.1%"},
            "ambient_villagers": {"passed": 249, "total": 275, "rate": "90.5%"},
        },
        "top_failures": [
            "Quest givers occasionally reveal future quest steps",
            "Ambient villagers repeat greetings within same conversation",
            "Emotion transitions too abrupt after combat encounters",
        ],
    }


@tool
def check_emotion_model() -> dict:
    """Check the emotion model state distribution and drift statistics.
    Returns current emotion state distribution across NPCs and drift metrics.
    """
    return {
        "emotion_states": 6,
        "distribution": {
            "neutral": "34%",
            "happy": "22%",
            "angry": "8%",
            "sad": "11%",
            "fearful": "6%",
            "surprised": "19%",
        },
        "drift_detected": True,
        "drift_details": {
            "affected_npcs": 7,
            "stuck_state": "angry",
            "avg_stuck_duration": "12+ conversation turns",
            "trigger": "Long conversations (>10 turns) with conflict topics",
            "workaround": "Emotion decay timer (not yet implemented)",
        },
        "personality_archetypes_active": 23,
        "npc_profiles_completed": "8 / 12 key NPCs",
    }


@tool
def start_dialogue_benchmark() -> dict:
    """Start a full dialogue quality benchmark across all 23 personality archetypes.
    This is a LONG-RUNNING task — evaluates coherence, consistency, and lore accuracy.
    """
    return {
        "status": "submitted",
        "message": "Dialogue benchmark started — testing all 23 archetypes across 3000 conversation logs.",
    }


@tool
def start_personality_audit() -> dict:
    """Start a personality profile audit for the 12 key NPCs.
    This is a LONG-RUNNING task — validates personality consistency and completeness.
    """
    return {
        "status": "submitted",
        "message": "Personality audit started — reviewing profiles for 12 key NPCs.",
    }


async def run_dialogue_benchmark_async(
    store: BaseStore, session_id: str, task_id: str
) -> dict:
    phases = [
        ("Loading 3000 conversation logs and archetype configs...", 0.10),
        ("Evaluating coherence scores across archetypes...", 0.30),
        ("Evaluating personality consistency per NPC...", 0.50),
        ("Cross-referencing dialogue against Aethermoor lore database...", 0.70),
        ("Testing long-conversation degradation (8+ turns)...", 0.85),
        ("Compiling benchmark report...", 0.95),
    ]
    for msg, progress in phases:
        await asyncio.sleep(2)
        logger.info(f"[CD-TOOL] run_dialogue_benchmark: {msg}")
        await store.aput(
            namespace=("tasks", session_id),
            key=task_id,
            value={
                "agent": "character_designer",
                "tool": "run_dialogue_benchmark",
                "status": "running",
                "progress": round(progress, 2),
                "message": msg,
                "updated_at": time.time(),
            },
        )
    await asyncio.sleep(2)
    result = {
        "status": "completed",
        "coherence": {"score": 0.89, "target": 0.92, "improved": True},
        "personality_consistency": {"score": 0.93, "target": 0.95},
        "lore_accuracy": {"score": 0.96, "target": 0.95, "passed": True},
        "long_conversation_degradation": "Coherence drops to 0.74 after 10 turns",
        "recommendation": "Implement emotion decay timer before final release",
    }
    await store.aput(
        namespace=("tasks", session_id),
        key=task_id,
        value={
            "agent": "character_designer",
            "tool": "run_dialogue_benchmark",
            "status": "completed",
            "progress": 1.0,
            "message": "Benchmark complete — coherence 0.89, lore accuracy passed, long-convo degradation confirmed",
            "result": result,
            "updated_at": time.time(),
        },
    )
    return result


async def run_personality_audit_async(
    store: BaseStore, session_id: str, task_id: str
) -> dict:
    phases = [
        ("Loading personality profiles for 12 key NPCs...", 0.15),
        ("Validating trait distributions and archetype alignment...", 0.35),
        ("Testing dialogue samples against personality baselines...", 0.55),
        ("Checking emotional range coverage per NPC...", 0.75),
        ("Flagging incomplete or inconsistent profiles...", 0.90),
    ]
    for msg, progress in phases:
        await asyncio.sleep(3)
        logger.info(f"[CD-TOOL] run_personality_audit: {msg}")
        await store.aput(
            namespace=("tasks", session_id),
            key=task_id,
            value={
                "agent": "character_designer",
                "tool": "run_personality_audit",
                "status": "running",
                "progress": round(progress, 2),
                "message": msg,
                "updated_at": time.time(),
            },
        )
    await asyncio.sleep(2)
    result = {
        "status": "completed",
        "profiles_reviewed": 12,
        "fully_complete": 8,
        "needs_work": 4,
        "incomplete_npcs": [
            {"name": "Elder Miravel", "issue": "Missing fear/surprise emotional responses"},
            {"name": "Captain Dray", "issue": "Combat personality bleeds into merchant interactions"},
            {"name": "The Weaver", "issue": "Lore references inconsistent with Act 3 rewrites"},
            {"name": "Puck", "issue": "Humor archetype lacks serious-mode fallback"},
        ],
        "recommendation": "Complete 4 remaining profiles before dialogue benchmark rerun",
    }
    await store.aput(
        namespace=("tasks", session_id),
        key=task_id,
        value={
            "agent": "character_designer",
            "tool": "run_personality_audit",
            "status": "completed",
            "progress": 1.0,
            "message": "Audit complete — 8/12 profiles pass, 4 NPCs need personality fixes",
            "result": result,
            "updated_at": time.time(),
        },
    )
    return result


# ---------------------------------------------------------------------------
# Technical Director tools
# ---------------------------------------------------------------------------

@tool
def check_frame_performance() -> dict:
    """Check current frame performance stats including FPS, frame time, and GPU budget.
    Returns per-system GPU budget breakdown and bottleneck analysis.
    """
    return {
        "target_fps": 60,
        "current_fps": 52,
        "frame_time_ms": 19.2,
        "frame_budget_ms": 16.67,
        "gpu_budget_breakdown": {
            "rendering": "8.1ms",
            "physics": "2.4ms",
            "ai_inference": "6.2ms",
            "audio": "1.1ms",
            "other": "1.4ms",
        },
        "ai_budget_target": "4.0ms",
        "ai_budget_over_by": "2.2ms",
        "target_hardware": "RTX 4070",
        "bottleneck": "AI NPC inference — 40 active NPCs at ~155µs each",
    }


@tool
def check_memory_usage() -> dict:
    """Check VRAM allocation across all game systems.
    Returns per-system VRAM usage and remaining budget.
    """
    return {
        "total_vram": "12 GB (RTX 4070)",
        "allocated": {
            "textures_lod": "4.8 GB",
            "geometry_buffers": "2.1 GB",
            "ai_models": "2.4 GB",
            "audio_streaming": "0.3 GB",
            "physics_sim": "0.6 GB",
            "frame_buffers": "1.2 GB",
        },
        "total_used": "11.4 GB",
        "remaining": "0.6 GB",
        "ai_vram_budget": "2.0 GB",
        "ai_vram_over_by": "0.4 GB",
        "note": "AI models over budget — LOD system for AI would reclaim ~0.6 GB",
    }


@tool
def check_lod_system() -> dict:
    """Check the Level-of-Detail system status for AI NPC inference.
    Returns LOD tier definitions, transition quality, and expected savings.
    """
    return {
        "lod_tiers": {
            "tier_0_close": {
                "range": "0-15m",
                "model": "Full 400M (FP16)",
                "latency": "3.2ms",
                "max_npcs": 5,
            },
            "tier_1_medium": {
                "range": "15-40m",
                "model": "200M distilled (INT8)",
                "latency": "1.1ms",
                "max_npcs": 15,
            },
            "tier_2_far": {
                "range": "40m+",
                "model": "Scripted fallback",
                "latency": "0.05ms",
                "max_npcs": 20,
            },
        },
        "implementation_status": "prototype — tier transitions cause visible dialogue stutters",
        "estimated_savings": "AI budget drops from 6.2ms to 3.8ms with LOD active",
        "transition_quality": "Needs work — players notice personality shift at tier boundaries",
    }


@tool
def start_gpu_profiling() -> dict:
    """Start a detailed GPU profiling session for NPC AI inference.
    This is a LONG-RUNNING task — profiles all 40 active NPCs across LOD tiers.
    """
    return {
        "status": "submitted",
        "message": "GPU profiling started — capturing per-NPC inference timing across all LOD tiers.",
    }


@tool
def start_batch_inference_test() -> dict:
    """Start a batch inference test for ambient NPC conversations.
    This is a LONG-RUNNING task — measures latency/throughput trade-offs.
    """
    return {
        "status": "submitted",
        "message": "Batch inference test started — measuring throughput vs latency for ambient NPCs.",
    }


async def run_gpu_profiling_async(
    store: BaseStore, session_id: str, task_id: str
) -> dict:
    phases = [
        ("Spawning 40 NPCs in test scene (Aethermoor Market)...", 0.10),
        ("Profiling Tier 0 (close-range) inference — 5 NPCs at full model...", 0.30),
        ("Profiling Tier 1 (medium-range) inference — 15 NPCs distilled...", 0.50),
        ("Profiling Tier 2 (far) scripted fallback — 20 NPCs...", 0.65),
        ("Measuring LOD transition overhead...", 0.80),
        ("Generating per-frame GPU timeline report...", 0.95),
    ]
    for msg, progress in phases:
        await asyncio.sleep(2)
        logger.info(f"[TD-TOOL] run_gpu_profiling: {msg}")
        await store.aput(
            namespace=("tasks", session_id),
            key=task_id,
            value={
                "agent": "tech_director",
                "tool": "run_gpu_profiling",
                "status": "running",
                "progress": round(progress, 2),
                "message": msg,
                "updated_at": time.time(),
            },
        )
    await asyncio.sleep(2)
    result = {
        "status": "completed",
        "total_ai_time_no_lod": "6.2ms",
        "total_ai_time_with_lod": "3.8ms",
        "savings": "2.4ms (38.7% reduction)",
        "tier_0_avg": "3.2ms for 5 NPCs",
        "tier_1_avg": "1.1ms for 15 NPCs (INT8 distilled)",
        "tier_2_avg": "0.05ms for 20 NPCs (scripted)",
        "lod_transition_cost": "0.4ms spike per transition",
        "recommendation": "LOD system brings AI within 4ms budget — fix transition stutters",
    }
    await store.aput(
        namespace=("tasks", session_id),
        key=task_id,
        value={
            "agent": "tech_director",
            "tool": "run_gpu_profiling",
            "status": "completed",
            "progress": 1.0,
            "message": "Profiling complete — LOD system saves 2.4ms, brings AI within budget",
            "result": result,
            "updated_at": time.time(),
        },
    )
    return result


async def run_batch_inference_test_async(
    store: BaseStore, session_id: str, task_id: str
) -> dict:
    phases = [
        ("Setting up batch inference pipeline for ambient NPCs...", 0.15),
        ("Running batch size sweep (1, 4, 8, 16 NPCs per batch)...", 0.35),
        ("Measuring per-batch latency and throughput...", 0.55),
        ("Testing 100ms latency tolerance for non-critical dialogue...", 0.75),
        ("Comparing batched vs individual inference cost...", 0.90),
    ]
    for msg, progress in phases:
        await asyncio.sleep(3)
        logger.info(f"[TD-TOOL] run_batch_inference_test: {msg}")
        await store.aput(
            namespace=("tasks", session_id),
            key=task_id,
            value={
                "agent": "tech_director",
                "tool": "run_batch_inference_test",
                "status": "running",
                "progress": round(progress, 2),
                "message": msg,
                "updated_at": time.time(),
            },
        )
    await asyncio.sleep(2)
    result = {
        "status": "completed",
        "batch_sizes_tested": [1, 4, 8, 16],
        "optimal_batch_size": 8,
        "individual_cost": "155µs per NPC",
        "batched_cost": "52µs per NPC (batch of 8)",
        "cost_reduction": "3.0x",
        "added_latency": "95ms for non-critical NPCs",
        "within_tolerance": True,
        "recommendation": "Use batch=8 for Tier 1+2 ambient NPCs — saves 1.5ms/frame",
    }
    await store.aput(
        namespace=("tasks", session_id),
        key=task_id,
        value={
            "agent": "tech_director",
            "tool": "run_batch_inference_test",
            "status": "completed",
            "progress": 1.0,
            "message": "Batch test complete — batch=8 optimal, 3x cost reduction, 95ms latency acceptable",
            "result": result,
            "updated_at": time.time(),
        },
    )
    return result


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

CD_SITUATION = """\
SCENARIO — ECHOES OF AETHERMOOR AI CHARACTER SYSTEM:

You lead the AI character design for "Echoes of Aethermoor", an open-world RPG.
You're replacing scripted NPC dialogue with a neural dialogue system powered by
emotion-driven behavior trees. Your goal is lifelike, lore-consistent NPCs.

YOUR 3 ACTION ITEMS:
1. COMPLETE NPC PROFILES — finish personality profiles for 12 key NPCs.
   8 are done, 4 still need work. Each profile defines traits, emotional range,
   and dialogue style for the neural dialogue model.
2. RUN DIALOGUE BENCHMARK — evaluate coherence, personality consistency, and
   lore accuracy across all 23 personality archetypes. Target: 92% coherence
   (currently at 87%). Need {{other_name}} to confirm inference budget allows full model.
3. FIX EMOTION DRIFT — NPCs get "stuck" in moods after long conversations
   (especially anger). Need an emotion decay timer. This affects {{other_name}}'s
   frame budget because decay checks add per-frame compute.

KEY FACTS YOU KNOW:
- Neural dialogue model is 2.1B params distilled to 400M for runtime
- Supports 6 emotion states, 23 personality archetypes
- Tested on 3000 player conversation logs, coherence score 87% (target 92%)
- Coherence drops to ~74% after 10+ turns — tied to emotion drift bug
- 12 key NPCs need personality profiles; 8 complete, 4 in progress

THINGS YOU NEED FROM {{other_name}}:
- "Can we keep the full 400M model for close-range NPCs?"
- "What's the per-NPC inference budget I need to stay within?"
- "Will batch inference add noticeable delay to player-facing dialogue?"
"""

TD_SITUATION = """\
SCENARIO — ECHOES OF AETHERMOOR AI CHARACTER SYSTEM:

You're the technical director responsible for shipping "Echoes of Aethermoor"
at 60 FPS on target hardware (RTX 4070). The AI character system is the biggest
performance risk — you need to make it fast without ruining {{other_name}}'s vision.

YOUR 3 ACTION ITEMS:
1. OPTIMIZE NPC INFERENCE — current AI budget is 6.2ms/frame, needs to be
   under 4ms. Implement LOD system: full model close-up, distilled model
   mid-range, scripted fallback for distant NPCs.
2. SET UP BATCH INFERENCE — batch ambient NPC conversations to reduce GPU
   cost by 3x. Adds 100ms latency for non-critical NPCs (acceptable).
   Need to coordinate with {{other_name}} on which NPCs are "critical" vs "ambient".
3. MANAGE VRAM BUDGET — AI models currently using 2.4 GB of the 2.0 GB budget.
   LOD system would reclaim ~0.6 GB. Need to finalize tier thresholds with {{other_name}}.

KEY FACTS YOU KNOW:
- Target: 60 FPS on RTX 4070 (16.67ms frame budget)
- AI inference currently 6.2ms/frame — over by 2.2ms
- 40 active NPCs max in any scene
- VRAM budget for AI: 2 GB (currently at 2.4 GB — 0.4 GB over)
- LOD system prototype works but tier transitions cause dialogue stutters
- Batch inference reduces per-NPC cost by 3x but adds 100ms latency

THINGS YOU NEED FROM {{other_name}}:
- "Which NPCs are critical (need full model) vs ambient (can batch)?"
- "Can we use the INT8 distilled model for mid-range without quality loss?"
- "Will the emotion decay timer add significant per-frame compute?"

CONSTRAINTS:
- Frame budget is non-negotiable — game MUST ship at 60 FPS
- LOD transition stutters need fixing before alpha — players notice personality shifts
- VRAM budget already tight — any new features need to fit within existing allocation
"""

SCENARIO = ScenarioConfig(
    id="game_studio",
    title="Game Studio AI Characters",
    description="Ship AI-driven NPCs for open-world RPG 'Echoes of Aethermoor' — an AI Character Designer and Technical Director collaborate.",
    guide_overview=(
        "Luna and Kai are shipping AI-driven NPCs for 'Echoes of Aethermoor', an "
        "open-world RPG. Luna designs NPC personalities, dialogue systems, and "
        "emotion models using a 400M-parameter neural dialogue system. Kai is the "
        "technical director making it all run at 60 FPS on an RTX 4070. The AI "
        "inference budget is over by 2.2ms, VRAM is over by 0.4 GB, and NPCs get "
        "stuck in angry moods after long conversations. They need to implement LOD "
        "tiers, batch inference, and fix the emotion drift bug before alpha."
    ),
    suggested_questions=[
        "Can you both introduce yourselves?",
        "Luna, how's the dialogue quality looking?",
        "What are the NPC behavior test results?",
        "Luna, what's going on with the emotion drift?",
        "Kai, what's the current frame performance?",
        "Kai, how's the VRAM budget looking?",
        "Can you check the LOD system status?",
        "Luna, can you run a dialogue benchmark?",
        "Kai, can you start GPU profiling?",
        "How do we get the AI budget under 4ms?",
    ],
    agents=[
        AgentConfig(
            id="character_designer",
            name="Luna",
            role="AI Character Designer",
            domain_keywords=[
                "NPC", "behavior tree", "dialogue", "emotion model",
                "personality", "character AI", "neural dialogue",
                "conversation", "quest", "narrative", "Aethermoor",
            ],
            voice_en="Magpie-Multilingual.EN-US.Aria",
            color="#76B900",
            situation=CD_SITUATION,
            intro_example=(
                "Hi, I'm {name}. I design the AI characters — their personalities, "
                "dialogue, how they feel and react. Great to meet you."
            ),
            personality_quips=[
                "Kai would ship NPCs as cardboard cutouts if it saved 2 FPS...",
                "{other_name} hears '2.1 billion parameters' and breaks out in a cold sweat.",
                "Sure, {other_name}, let's just make every NPC a static mesh with a speech bubble.",
                "I'll keep the emotion model under budget... probably.",
            ],
            fast_tools=[check_dialogue_quality, check_npc_behavior_tests, check_emotion_model],
            slow_launchers=[start_dialogue_benchmark, start_personality_audit],
            launcher_to_executor={
                "start_dialogue_benchmark": "run_dialogue_benchmark",
                "start_personality_audit": "run_personality_audit",
            },
            slow_tools={
                "run_dialogue_benchmark": run_dialogue_benchmark_async,
                "run_personality_audit": run_personality_audit_async,
            },
        ),
        AgentConfig(
            id="tech_director",
            name="Kai",
            role="Technical Director",
            domain_keywords=[
                "engine", "performance", "GPU", "memory", "FPS",
                "frame budget", "LOD", "optimization", "VRAM",
                "batch inference", "streaming", "Unreal",
            ],
            voice_en="Magpie-Multilingual.EN-US.Jason",
            color="#00B4D8",
            situation=TD_SITUATION,
            intro_example=(
                "Hey, I'm {name}. I'm the tech director — I make sure this game "
                "actually runs on real hardware. Nice to meet you."
            ),
            personality_quips=[
                "Luna thinks GPU memory grows on trees...",
                "{other_name}'s 400-million-parameter NPCs are my 4ms nightmare.",
                "I love how {other_name} says 'just distill it' like it's making a smoothie.",
                "Frame budget is 16 milliseconds. Not 16-ish. Not 17. Sixteen.",
            ],
            fast_tools=[check_frame_performance, check_memory_usage, check_lod_system],
            slow_launchers=[start_gpu_profiling, start_batch_inference_test],
            launcher_to_executor={
                "start_gpu_profiling": "run_gpu_profiling",
                "start_batch_inference_test": "run_batch_inference_test",
            },
            slow_tools={
                "run_gpu_profiling": run_gpu_profiling_async,
                "run_batch_inference_test": run_batch_inference_test_async,
            },
        ),
    ],
)

register_scenario(SCENARIO)
