"""Benchmark a model as a multi-agent router.

Tests structured-output routing accuracy and latency across a set of
representative utterances. Compares against expected routing decisions.
"""

import time
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import Literal

# ── Config ──────────────────────────────────────────────────────────────
BASE_URL = "http://192.168.7.163:8044/v1"
MODEL = "unsloth/Ministral-3-8B-Instruct-2512-FP8"
DS_NAME = "Sarah"
IT_NAME = "Mike"

DISABLE_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


class RoutingDecision(BaseModel):
    primary_agent: Literal["data_scientist", "it"] = Field(
        ..., description="The primary agent that should respond"
    )
    secondary_agent: Literal["data_scientist", "it", "none"] = Field(
        default="none", description="Secondary agent, or 'none'"
    )
    topic: str = Field(default="general", description="Detected topic of the utterance")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Routing confidence")


ROUTER_PROMPT = f"""Route messages to the correct agent.

AGENTS:
- {DS_NAME} (data_scientist): ML, models, training, data, evaluation, MLOps
- {IT_NAME} (it): Infrastructure, GPUs, deployment, security, timelines, budget

ROUTING RULES (follow strictly in order):
1. FOLLOW-UP DETECTION: If the user's message is a follow-up to the previous agent's response (e.g., "you" singular, continuing the same topic, reacting to what one agent just said), route ONLY to that same agent. secondary_agent = "none".
2. BY NAME: If the user addresses ONE agent by name ("{DS_NAME}", "{IT_NAME}"), route ONLY to that agent. secondary_agent = "none".
3. BOTH AGENTS: Set secondary_agent when ANY of these apply:
   - Plural address: "guys", "both", "everyone", "team", "you two", "all", "yourselves", "each of you"
   - Introduction requests: "introduce yourselves", "who are you", "tell me about yourselves" — ALWAYS set secondary_agent for introductions
   - The user explicitly asks BOTH agents to respond
4. SINGLE DOMAIN: If the question is about one domain only (ML or infra), route to that agent only. secondary_agent = "none".
5. DEFAULT: secondary_agent = "none". Only override this via rule 3.

CRITICAL: For introduction/greeting requests where the user does not name a specific agent, ALWAYS set secondary_agent so both agents can introduce themselves.

If unsure about primary, route to data_scientist.

TOPICS: introduction, personal, acknowledgment, clarification, training, model, data, evaluation, inference, infrastructure, deployment, security, timeline, budget, general

OUTPUT: primary_agent, secondary_agent ("none" if only one needed), topic, confidence (0-1)
"""


# ── Test cases: (utterance, expected_primary, expected_secondary, description) ──
TEST_CASES = [
    ("Good morning, please introduce yourselves.",
     "data_scientist", "it", "intro-both"),
    ("Hi guys, who are you?",
     "data_scientist", "it", "intro-plural"),
    ("Sarah, what's the status of the model training?",
     "data_scientist", "none", "sarah-by-name"),
    ("Mike, when can we start the deployment?",
     "it", "none", "mike-by-name"),
    ("What GPU resources do we need for serving the new model?",
     "it", "none", "infra-single"),
    ("How is the RecDeep model performing in offline evaluation?",
     "data_scientist", "none", "ds-single"),
    ("Team, give me a status update on the project.",
     "data_scientist", "it", "team-both"),
    ("What's the timeline for the full rollout?",
     "it", "none", "timeline-single"),
    ("Can you explain the difference between v1 and v2 architectures?",
     "data_scientist", "none", "model-arch"),
    ("Both of you, what are the current blockers?",
     "data_scientist", "it", "blockers-both"),
]


def run_benchmark():
    llm = ChatOpenAI(
        base_url=BASE_URL,
        model=MODEL,
        api_key="not-needed",
        temperature=0.0,
        max_tokens=256,
    )
    structured = llm.with_structured_output(RoutingDecision)

    print(f"\n{'='*72}")
    print(f"Router Benchmark: {MODEL}")
    print(f"Endpoint: {BASE_URL}")
    print(f"{'='*72}\n")

    latencies = []
    correct = 0
    total = len(TEST_CASES)

    for utterance, exp_primary, exp_secondary, label in TEST_CASES:
        messages = [
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=f'Context: (no prior messages)\nMessage: "{utterance}"\nRoute:'),
        ]

        t0 = time.perf_counter()
        try:
            result: RoutingDecision = structured.invoke(messages)
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000
            latencies.append(latency_ms)

            p = result.primary_agent
            s = result.secondary_agent
            if s not in ("data_scientist", "it"):
                s = "none"

            p_ok = p == exp_primary
            s_ok = s == exp_secondary
            both_ok = p_ok and s_ok
            if both_ok:
                correct += 1

            status = "OK" if both_ok else "MISS"
            print(f"[{status}] {label:20s} | {latency_ms:7.0f}ms | "
                  f"p={p:15s} s={s:15s} topic={result.topic:15s} conf={result.confidence:.2f}"
                  f"{'' if both_ok else f'  (expected p={exp_primary} s={exp_secondary})'}")
        except Exception as e:
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000
            latencies.append(latency_ms)
            print(f"[ERR ] {label:20s} | {latency_ms:7.0f}ms | {e}")

    print(f"\n{'─'*72}")
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.0f}%)")
    if latencies:
        avg = sum(latencies) / len(latencies)
        mn = min(latencies)
        mx = max(latencies)
        p50 = sorted(latencies)[len(latencies)//2]
        print(f"Latency:  avg={avg:.0f}ms  min={mn:.0f}ms  max={mx:.0f}ms  p50={p50:.0f}ms")
    print(f"{'─'*72}\n")


if __name__ == "__main__":
    run_benchmark()
