"""Orchestrator StateGraph: routes human messages and invokes agents.

Fully scenario-driven — the active scenario is passed via ``scenario_id``
in the graph input.  Routing decisions, agent prompts, tool binding, and
handoff logic all read from the scenario config at runtime.

After every agent turn, a lightweight handoff check runs (while TTS plays)
to decide if another agent should respond next.  The loop is bounded by
``max_turns`` (default 4) so conversations stay focused.

Agents generate plain text (NOT structured output) so that LangGraph can
stream tokens via stream_mode="messages" for low-latency TTS.
"""

import operator
import time
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from loguru import logger

from src.config import get_agent_model, get_router_model, get_scenario_id
from src.models import HandoffCheck, ProjectPlan, RoutingDecision
from src.prompts import get_agent_prompt, get_router_prompt
from src.scenarios import ScenarioConfig, get_scenario
from src.utils import format_context, get_agent_labels, to_langchain_messages

MAX_TOOL_ROUNDS = 3
MAX_CONVERSATION_TURNS = 4


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class InputState(TypedDict):
    human_input: str
    scenario_id: str


class State(TypedDict):
    human_input: str
    scenario_id: str
    messages: Annotated[list[dict], operator.add]
    project_plan: dict
    routing: dict
    turn_count: int
    max_turns: int
    next_agent: str
    queued_secondary: str
    tool_launches: list[dict]


def _get_scenario(state: State) -> ScenarioConfig:
    sid = state.get("scenario_id") or get_scenario_id()
    return get_scenario(sid)


# ---------------------------------------------------------------------------
# Agent invocation (plain text, NOT structured output)
# ---------------------------------------------------------------------------

def _invoke_agent(state: State, agent_id: str, hint: str | None = None) -> dict:
    """Call the specified agent, optionally with tool-calling (ReAct loop).

    Tool binding is controlled by the router's ``tools_allowed`` flag.
    """
    t0 = time.perf_counter()

    scenario = _get_scenario(state)
    agent_cfg = scenario.get_agent(agent_id)
    labels = get_agent_labels(scenario)

    messages = state.get("messages", [])
    plan_dict = state.get("project_plan") or ProjectPlan().model_dump()
    plan_text = ProjectPlan(**plan_dict).summary()

    lc_messages = to_langchain_messages(messages, perspective=agent_id, agent_labels=labels)
    if hint:
        lc_messages.append(HumanMessage(content=f"[Context: {hint}]"))

    model = get_agent_model()
    system_prompt = get_agent_prompt(scenario, agent_id, plan_text)
    tools = agent_cfg.all_tools
    launcher_map = agent_cfg.launcher_to_executor

    tools_allowed = state.get("routing", {}).get("tools_allowed", False)

    if tools_allowed and tools:
        tool_map = {t.name: t for t in tools}
        active_model = model.bind_tools(tools)
        logger.info(f"[REACT] {agent_id} tools=enabled ({len(tools)} tools)")
    else:
        tool_map = {}
        active_model = model
        logger.info(f"[REACT] {agent_id} tools=disabled")

    full_messages = [SystemMessage(content=system_prompt)] + lc_messages
    tool_launches: list[dict] = []

    for round_idx in range(MAX_TOOL_ROUNDS):
        t_invoke = time.perf_counter()
        result = active_model.invoke(full_messages)
        t_done = time.perf_counter()

        text = result.content if hasattr(result, "content") else ""
        tool_calls = getattr(result, "tool_calls", None) or []

        logger.info(
            f"[REACT] {agent_id} round={round_idx} "
            f"invoke={((t_done - t_invoke) * 1000):.0f}ms "
            f"tool_calls={[tc['name'] for tc in tool_calls]} "
            f"text_len={len(text)}"
        )

        if not tool_calls:
            break

        full_messages.append(result)

        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc.get("args", {})
            tool_fn = tool_map.get(tool_name)

            if tool_fn is None:
                logger.warning(f"[REACT] {agent_id} unknown tool: {tool_name}")
                full_messages.append(
                    ToolMessage(
                        content=f"Error: unknown tool '{tool_name}'",
                        tool_call_id=tc["id"],
                    )
                )
                continue

            t_tool = time.perf_counter()
            tool_result = tool_fn.invoke(tool_args)
            logger.info(
                f"[REACT] {agent_id} tool={tool_name} "
                f"exec={((time.perf_counter() - t_tool) * 1000):.0f}ms "
                f"result={str(tool_result)[:120]}"
            )

            full_messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tc["id"],
                )
            )

            if tool_name in launcher_map:
                tool_launches.append({
                    "agent": agent_id,
                    "launcher_tool": tool_name,
                    "executor_tool": launcher_map[tool_name],
                    "args": tool_args,
                })
    else:
        logger.warning(
            f"[REACT] {agent_id} hit max tool rounds ({MAX_TOOL_ROUNDS})"
        )

    text = result.content if hasattr(result, "content") else str(result)
    total_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        f"[TIMING] {agent_id} total={total_ms:.0f}ms "
        f"tokens≈{len(text.split())} "
        f"tools_called={len(tool_launches)} "
        f"text=[{text[:80]}...]"
    )

    return {
        "messages": [{"speaker": agent_id, "text": text}],
        "turn_count": state.get("turn_count", 0) + 1,
        "tool_launches": tool_launches,
    }


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def route(state: State) -> dict:
    """Classify the human message via the router LLM (structured output)."""
    t0 = time.perf_counter()

    scenario = _get_scenario(state)
    labels = get_agent_labels(scenario)
    agent_ids = scenario.agent_ids

    text = state["human_input"]
    messages = state.get("messages", [])

    context = format_context(messages, agent_labels=labels)
    lc_messages = [
        SystemMessage(content=get_router_prompt(scenario)),
        HumanMessage(content=f"Context: {context}\nMessage: \"{text}\"\nRoute:"),
    ]

    model = get_router_model()
    structured = model.with_structured_output(RoutingDecision)

    t_invoke = time.perf_counter()
    result: RoutingDecision = structured.invoke(lc_messages)
    t_done = time.perf_counter()

    routing = result.model_dump()

    primary = routing["primary_agent"]
    secondary = routing.get("secondary_agent", "none")

    if primary not in agent_ids:
        primary = agent_ids[0]
    if secondary not in agent_ids:
        secondary = ""
    if secondary == primary:
        secondary = ""

    routing["primary_agent"] = primary
    routing["secondary_agent"] = secondary

    topic = routing.get("topic", "general")
    tools_allowed = routing.get("tools_allowed", False)

    intro_keywords = {"intro", "greet", "hello", "who are you", "meet", "yoursel"}
    is_intro = any(kw in topic.lower() for kw in intro_keywords) or any(
        kw in text.lower() for kw in {"introduce", "who are you", "yourselves"}
    )
    if is_intro and secondary:
        max_turns = 2
        logger.info("[ROUTE] Introduction detected — capping at 2 turns")
    elif not secondary:
        max_turns = 2
        logger.info("[ROUTE] Single-agent routing — capping at 2 turns")
    else:
        max_turns = MAX_CONVERSATION_TURNS

    logger.info(
        f"[TIMING] router invoke={((t_done - t_invoke) * 1000):.0f}ms "
        f"total={((t_done - t0) * 1000):.0f}ms "
        f"primary={primary} secondary={secondary or 'none'} "
        f"topic={topic} tools={tools_allowed} "
        f"confidence={routing.get('confidence', 0):.2f}"
    )

    update: dict = {
        "messages": [{"speaker": "human", "text": text}],
        "routing": routing,
        "turn_count": 0,
        "max_turns": max_turns,
        "next_agent": primary,
        "queued_secondary": secondary,
    }

    if not state.get("project_plan"):
        update["project_plan"] = ProjectPlan().model_dump()

    return update


def invoke_agent(state: State) -> dict:
    """Invoke whichever agent is in ``next_agent``."""
    agent_id = state["next_agent"]
    turn_count = state.get("turn_count", 0)

    scenario = _get_scenario(state)
    labels = get_agent_labels(scenario)

    hint = None
    if turn_count > 0:
        messages = state.get("messages", [])
        last_speaker = ""
        for msg in reversed(messages):
            if msg.get("speaker") not in ("human",):
                last_speaker = msg.get("speaker", "")
                break

        if last_speaker:
            last_name = labels.get(last_speaker, last_speaker)
            max_t = state.get("max_turns", MAX_CONVERSATION_TURNS)

            is_last_turn = (turn_count + 1) >= max_t
            is_penultimate = (turn_count + 2) >= max_t and not is_last_turn

            base = (
                f"{last_name} just spoke and addressed you (see their message above). "
                "Answer their question or add your perspective directly. "
                "Do NOT repeat what they said. Keep it natural — 1-3 sentences."
            )

            if is_last_turn:
                hint = (
                    f"{base} "
                    f"You are the LAST speaker in this exchange. "
                    f"Do NOT ask {last_name} a question. "
                    f"Do NOT end with any question directed at a team member. "
                    f"Wrap up conclusively. If appropriate, address the human "
                    f"user directly with a brief next-step suggestion."
                )
            elif is_penultimate:
                hint = (
                    f"{base} "
                    f"The conversation is wrapping up after your response. "
                    f"Do NOT ask {last_name} a follow-up question — "
                    f"just give your answer and conclude."
                )
            else:
                hint = base

    logger.info(f"[TIMING] invoke_agent start agent={agent_id} turn={turn_count}")
    result = _invoke_agent(state, agent_id, hint=hint)

    queued = state.get("queued_secondary", "")
    if queued and queued == agent_id:
        result["queued_secondary"] = ""

    return result


def check_handoff(state: State) -> dict:
    """Decide whether another agent should speak next."""
    turn_count = state.get("turn_count", 0)
    max_turns = state.get("max_turns", MAX_CONVERSATION_TURNS)

    if turn_count >= max_turns:
        logger.info(f"[HANDOFF-CHECK] Max turns ({max_turns}) reached — stopping")
        return {"next_agent": ""}

    queued = state.get("queued_secondary", "")
    if queued:
        logger.info(f"[HANDOFF-CHECK] Router pre-queued secondary={queued}")
        return {"next_agent": queued}

    scenario = _get_scenario(state)
    labels = get_agent_labels(scenario)
    agent_ids = scenario.agent_ids

    messages = state.get("messages", [])
    last_speaker = ""
    last_text = ""
    for msg in reversed(messages):
        if msg.get("speaker") not in ("human",):
            last_speaker = msg.get("speaker", "")
            last_text = msg.get("text", "")
            break

    if not last_text:
        return {"next_agent": ""}

    other_agents = [aid for aid in agent_ids if aid != last_speaker]
    if not other_agents:
        return {"next_agent": ""}

    other_agent = other_agents[0]
    other_name = labels.get(other_agent, other_agent)
    speaker_name = labels.get(last_speaker, last_speaker)

    human_text = state.get("human_input", "")
    routing = state.get("routing", {})
    router_had_secondary = bool(routing.get("secondary_agent", ""))

    prompt = (
        f'The user said: "{human_text}"\n'
        f"The router decided ONLY {speaker_name} should respond"
        + (f" (no secondary agent was set).\n\n" if not router_had_secondary
           else f", with {other_name} as secondary.\n\n")
        + f"{speaker_name} just responded:\n"
        f'"{last_text[:500]}"\n\n'
        f"Should {other_name} ({other_agent}) speak next?\n\n"
        f"RULES (follow strictly):\n"
        f"- If the user addressed {speaker_name} specifically (by name, or "
        f"using 'you' as a follow-up), set should_speak=false — the user "
        f"wants to hear from {speaker_name}, not {other_name}.\n"
        f"- If {speaker_name}'s response is self-contained and doesn't ask "
        f"{other_name} a direct question, set should_speak=false.\n"
        f"- Set should_speak=true ONLY if {speaker_name} asked {other_name} "
        f"a direct question OR explicitly deferred to {other_name}.\n"
        f"- When in doubt, set should_speak=false.\n\n"
        f"Set tools_allowed=true only if {other_name} needs real-time data "
        f"to answer (e.g., check cluster status, GPU usage)."
    )

    model = get_router_model()
    structured = model.with_structured_output(HandoffCheck)

    t0 = time.perf_counter()
    result: HandoffCheck = structured.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Check:"),
    ])
    t_ms = (time.perf_counter() - t0) * 1000

    if result.should_speak:
        routing = dict(state.get("routing", {}))
        original_tools = routing.get("tools_allowed", False)
        if result.tools_allowed and not original_tools:
            logger.info(
                f"[HANDOFF-CHECK] Ignoring tools_allowed=true from handoff — "
                f"original route said false"
            )
        routing["tools_allowed"] = original_tools
        logger.info(
            f"[HANDOFF-CHECK] {speaker_name} → {other_name} "
            f"tools={original_tools} t={t_ms:.0f}ms"
        )
        return {"next_agent": other_agent, "routing": routing}

    logger.info(f"[HANDOFF-CHECK] No handoff needed t={t_ms:.0f}ms")
    return {"next_agent": ""}


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def should_continue(state: State) -> str:
    if state.get("next_agent", ""):
        return "invoke_agent"
    return END


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

builder = StateGraph(State, input=InputState)

builder.add_node("route", route)
builder.add_node("invoke_agent", invoke_agent)
builder.add_node("check_handoff", check_handoff)

builder.add_edge(START, "route")
builder.add_edge("route", "invoke_agent")
builder.add_edge("invoke_agent", "check_handoff")
builder.add_conditional_edges(
    "check_handoff",
    should_continue,
    ["invoke_agent", END],
)

graph = builder.compile()
