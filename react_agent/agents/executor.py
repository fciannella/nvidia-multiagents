"""Unified executor agent: ReAct loop with all business tools.

Handles both telco account workflows (with OTP authentication) and
data/ML workflows (analysis, training) in a single graph. The LLM
picks the right tools based on the task description.

Uses @task for LLM calls and tool execution so that interrupt/resume
cycles replay deterministically. The ask_user tool triggers interrupt()
for human-in-the-loop steps — the Pipecat service delivers the question
through the front agent and resumes with the user's answer.

State (checkpointed via `previous`): list of message dicts.
"""

import json
import uuid

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.func import entrypoint, task
from langgraph.types import interrupt
from loguru import logger

from config import get_executor_llm
from tools.telco_tools import (
    ALL_TOOLS as TELCO_TOOLS,
    TOOLS_BY_NAME as TELCO_TOOLS_BY_NAME,
)

# Tools that take session_id (auth + account ops; exclude ask_user and list_available_packages_tool)
TELCO_SESSION_TOOLS = {
    "check_auth_tool", "start_login_tool", "verify_login_tool",
    "get_current_package_tool", "get_data_balance_tool", "recommend_packages_tool",
    "get_roaming_info_tool", "close_contract_tool", "list_addons_tool",
    "purchase_roaming_pass_tool", "change_package_tool", "get_billing_summary_tool",
    "set_data_alerts_tool",
}
from tools.workflow_tools import (
    ALL_TOOLS as WORKFLOW_TOOLS,
    TOOLS_BY_NAME as WORKFLOW_TOOLS_BY_NAME,
    workflow_session_id,
)

ALL_TOOLS = TELCO_TOOLS + [t for t in WORKFLOW_TOOLS if t.name != "ask_user"]
TOOLS_BY_NAME = {**TELCO_TOOLS_BY_NAME, **WORKFLOW_TOOLS_BY_NAME}

SYSTEM_PROMPT = """\
You are a backend task execution agent. You use tools to accomplish tasks.
You are NOT the voice the user hears — a separate front-end agent handles
conversation. Your outputs will be rephrased by that agent before being
spoken aloud.

CRITICAL — TOOL USE IS MANDATORY:
- You MUST use the ask_user tool for EVERY question to the user. NEVER
  write a question as plain text. Always call ask_user(question="...").
- You MUST call the appropriate tools to perform actions. NEVER pretend
  you did something without actually calling the tool.
- The ONLY time you should return plain text is when you have a final
  answer or result to deliver after all tools have been called.

STYLE:
- Do NOT greet the user or make small talk.
- Be direct and functional in your ask_user questions.
- When returning results, provide clear factual information only.
- Plain text only, no markdown or formatting.

AUTHENTICATION (required before any telco account tool):
1. FIRST call check_auth_tool to see if the session is already verified.
   - If verified=true, use the msisdn from the response. SKIP steps 2-5.
   - If verified=false, proceed with steps 2-5.
2. Call ask_user to get the mobile phone number.
3. Call start_login_tool with the number and session_id.
4. Call ask_user to get the 6-digit OTP code.
5. Call verify_login_tool with the code. Retry up to 2 times if failed.
6. Only after verify_login_tool returns verified=true may you call account tools.
   Account tools enforce this: they return an error if the session is not verified.

TELCO TOOLS (account tools require authenticated session; session_id is injected):
- get_current_package_tool: current package, contract, addons
- get_data_balance_tool: data usage and remaining allowance
- list_available_packages_tool: all available packages
- recommend_packages_tool: personalized recommendations
- get_roaming_info_tool: roaming pricing and passes for a country
- purchase_roaming_pass_tool: buy a roaming pass
- change_package_tool: switch packages
- get_billing_summary_tool: billing and monthly fee
- set_data_alerts_tool: set usage alerts
- list_addons_tool: list active addons
- close_contract_tool: close contract (call with confirm=false first)

DATA / ML TOOLS (no authentication needed):
- quick_lookup: fast database search (~5s)
- long_analysis: detailed data analysis (~30s)
- train_model: ML model training (~60s)

OUTPUT RULES:
- Be concise and factual. State numbers clearly.
- When presenting options, use flowing sentences not lists.
- Plain text only — the front agent handles formatting for voice.
"""

_QUESTION_STARTERS = (
    "what is", "what's", "could you", "can you", "please provide",
    "may i have", "i need your", "do you have", "which country",
    "what country", "to proceed", "to continue", "in order to",
    "before i can", "i still need",
)

_REQUEST_PHRASES = (
    "i need you to", "i will need you to", "please confirm",
    "please provide", "could you confirm", "can you provide",
    "can you confirm", "could you provide", "i need the",
    "i'll need", "need you to confirm", "need to know",
    "specify the", "confirm the",
)


def _coerce_text_to_ask_user(response: dict, messages: list[dict]) -> dict:
    """If the LLM returned a question as plain text instead of calling
    ask_user, synthesise the tool call so interrupt() fires."""
    if response.get("tool_calls"):
        return response

    text = (response.get("content") or "").strip()
    if not text:
        return response

    lower = text.lower()
    is_question = (
        "?" in text
        or any(lower.startswith(q) for q in _QUESTION_STARTERS)
        or any(phrase in lower for phrase in _REQUEST_PHRASES)
    )
    if not is_question:
        return response

    tc_id = f"synthetic_{uuid.uuid4().hex[:8]}"
    logger.warning(f"[executor] Coercing text→ask_user: {text[:80]}")
    response["tool_calls"] = [
        {"id": tc_id, "name": "ask_user", "args": {"question": text}}
    ]
    response["content"] = ""
    messages[-1] = response
    return response


@task
async def call_llm(messages: list[dict]) -> dict:
    """Invoke the executor LLM with all tools bound."""
    llm = get_executor_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    lc_messages: list[BaseMessage] = []
    for m in messages:
        role = m["role"]
        content = m.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            tc = m.get("tool_calls")
            lc_messages.append(AIMessage(content=content, tool_calls=tc or []))
        elif role == "tool":
            lc_messages.append(ToolMessage(
                content=content,
                tool_call_id=m["tool_call_id"],
                name=m.get("name", ""),
            ))

    response = await llm_with_tools.ainvoke(lc_messages)

    return {
        "role": "assistant",
        "content": response.content or "",
        "tool_calls": [
            {"id": tc["id"], "name": tc["name"], "args": tc.get("args", {})}
            for tc in (response.tool_calls or [])
        ],
    }


@task
async def execute_tool(name: str, args: dict, tool_call_id: str) -> dict:
    """Execute a non-interrupt tool. Returns a serialisable ToolMessage dict."""
    tool_fn = TOOLS_BY_NAME[name]
    raw_result = await tool_fn.ainvoke(args)
    content = raw_result if isinstance(raw_result, str) else json.dumps(raw_result)
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": tool_call_id,
        "name": name,
    }


@entrypoint()
async def graph(
    inputs: dict, *, previous: list[dict] | None = None
) -> entrypoint.final[str, list[dict]]:
    """Unified executor: ReAct agent with telco + workflow tools.

    Inputs:
        message: str      — the task description (from front agent delegation)
        session_id: str   — used for OTP session tracking
    Previous:
        list of message dicts persisted across interrupt/resume cycles.
    """
    session_id = inputs.get("session_id", "default")
    workflow_session_id.set(session_id)
    messages: list[dict] = list(previous or [])

    if not messages or messages[0].get("role") != "system":
        system_content = (
            SYSTEM_PROMPT
            + f"\n\nSession context: session_id={session_id}. "
            "Pass this when calling start_login_tool and verify_login_tool."
        )
        messages.insert(0, {"role": "system", "content": system_content})

    user_msg = inputs.get("message", "")
    if user_msg:
        messages.append({"role": "user", "content": user_msg})
        logger.info(f"[executor] Task: {user_msg[:100]}")

    response = await call_llm(messages)
    messages.append(response)
    response = _coerce_text_to_ask_user(response, messages)

    max_iterations = 30
    iteration = 0

    while response.get("tool_calls") and iteration < max_iterations:
        iteration += 1

        for tc in response["tool_calls"]:
            tc_name = tc["name"]
            tc_args = tc.get("args", {})
            tc_id = tc["id"]

            if tc_name == "ask_user":
                question = tc_args.get("question", "Could you clarify?")
                logger.info(f"[executor] ask_user: {question}")
                answer = interrupt({"question": question})
                logger.info(f"[executor] User answered: {str(answer)[:100]}")
                messages.append({
                    "role": "tool",
                    "content": str(answer),
                    "tool_call_id": tc_id,
                    "name": "ask_user",
                })
            else:
                # All telco tools need session_id (auth + account operations)
                if tc_name in TELCO_SESSION_TOOLS:
                    tc_args.setdefault("session_id", session_id)
                logger.info(f"[executor] Tool: {tc_name}({tc_args})")
                tool_result = await execute_tool(tc_name, tc_args, tc_id)
                messages.append(tool_result)
                logger.info(f"[executor] Result: {tool_result['content'][:200]}")

        response = await call_llm(messages)
        messages.append(response)
        response = _coerce_text_to_ask_user(response, messages)

    final_text = response.get("content", "")
    logger.info(f"[executor] Final: {final_text[:100]}")

    return entrypoint.final(value=final_text, save=messages)
