"""Delegation detection — extract tool requests from agent responses.

Detects IMMEDIATE action intent only. The agent must use a present-tense
action phrase like "Let me check...", "I'm starting...", "Checking now..."
to trigger delegation. Future tense ("I'll kick off...", "I want to...")
and past tense ("I checked...") are NOT delegations.

Patterns are matched per-sentence (not across the full text) to avoid
greedy regex spanning unrelated clauses.
"""

import re
from loguru import logger

# Prefixes that indicate IMMEDIATE action (present tense, imperative)
_NOW_PREFIX = (
    r"(?:let me |i'm going to |i'll |checking |starting |running |"
    r"pulling up |looking at |let me just |i'm checking |i'm looking )"
)

_ACTION = r"(?:check|look at|confirm|verify|pull up|look up|see|inspect)"

DS_PATTERNS: list[tuple[str, str, dict]] = [
    (_NOW_PREFIX + _ACTION + r".{0,50}\btraining\b.{0,30}(?:status|progress|run|job)", "check_training_status", {}),
    (_NOW_PREFIX + _ACTION + r".{0,50}\b(?:metrics|evaluation results|eval|accuracy|click.?through|ctr)", "get_model_metrics", {}),
    (_NOW_PREFIX + _ACTION + r".{0,50}\b(?:data|feature|pipeline|wishlist|ingestion|lag)", "check_data_pipeline", {}),
    (_NOW_PREFIX + r"(?:start|kick off|run|launch|begin).{0,50}\btraining", "run_training_job", {}),
    (_NOW_PREFIX + r"(?:run|start|kick off|launch).{0,50}\b(?:evaluation|eval|offline eval)", "run_offline_eval", {}),
    (_NOW_PREFIX + r"(?:export|convert|serialize).{0,50}\b(?:model|triton|tensorrt|onnx)", "export_model_to_triton", {}),
]

IT_PATTERNS: list[tuple[str, str, dict]] = [
    (_NOW_PREFIX + _ACTION + r".{0,50}\bgpu.{0,30}(?:utilization|usage|availability|allocation|capacity)", "check_gpu_utilization", {}),
    (_NOW_PREFIX + _ACTION + r".{0,50}\bcluster.{0,30}(?:availability|status|schedule|ready|free|open|conflicts?)", "check_cluster_availability", {}),
    (_NOW_PREFIX + _ACTION + r".{0,50}\b(?:conflicts?|scheduled?\s*jobs?).{0,30}(?:window|slot|block|clear|available)", "check_cluster_availability", {}),
    (_NOW_PREFIX + _ACTION + r".{0,50}\b(?:staging|training)\s+(?:cluster|node|machine)", "check_cluster_availability", {}),
    (_NOW_PREFIX + r"(?:confirm|block|reserve).{0,50}\b(?:cluster|a100|gpu|window|slot)", "check_cluster_availability", {}),
    (_NOW_PREFIX + _ACTION + r".{0,50}\b(?:deployment|canary).{0,30}(?:status|health)", "check_deployment_status", {}),
    (_NOW_PREFIX + r"(?:provision|set up|create).{0,50}\bkafka.{0,30}(?:consumer|group)", "provision_kafka_consumer", {}),
    (_NOW_PREFIX + r"(?:deploy|start|launch|set up).{0,50}\b(?:canary|shadow)", "deploy_canary", {}),
    (_NOW_PREFIX + r"(?:reallocate|free up|move|shift).{0,50}\bgpu", "reallocate_gpus", {}),
]


def _split_sentences(text: str) -> list[str]:
    """Split text into rough sentences for per-sentence matching."""
    parts = re.split(r'[.!?]+', text)
    return [s.strip() for s in parts if s.strip()]


def detect_delegation(agent: str, text: str) -> dict | None:
    """Scan agent text for immediate tool-use intent.

    Only triggers on present-tense action phrases ("Let me check...",
    "I'm starting the training now"). Ignores future plans, past actions,
    and hypotheticals.

    Returns {"tool_name": ..., "tool_args": {...}} or None.
    """
    patterns = DS_PATTERNS if agent == "data_scientist" else IT_PATTERNS

    for sentence in _split_sentences(text):
        sentence_lower = sentence.lower()
        for pattern, tool_name, default_args in patterns:
            if re.search(pattern, sentence_lower):
                logger.info(
                    f"[DELEGATION] Detected {agent} → {tool_name} "
                    f"from sentence: {sentence[:100]!r}"
                )
                return {"tool_name": tool_name, "tool_args": default_args}

    return None
