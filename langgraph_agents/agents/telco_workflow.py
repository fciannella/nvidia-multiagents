"""Telco workflow agent: mobile operator assistant with OTP authentication.

ReAct loop that handles the full telco customer journey:
1. Greet and collect phone number (via ask_user interrupt)
2. Send OTP and verify identity (via ask_user interrupt)
3. Handle account tasks: packages, data, roaming, billing, addons

Uses the same @entrypoint/@task pattern as workflow.py, with interrupt()
for human-in-the-loop steps. The session_id is auto-injected into
login tools so the LLM doesn't need to manage it explicitly.
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

from config import get_telco_workflow_llm
from tools.telco_tools import ALL_TOOLS, TOOLS_BY_NAME

SYSTEM_PROMPT = """\
You are a backend task agent that manages mobile phone accounts. You are NOT \
the voice the user hears — a separate front-end agent handles conversation. \
Your outputs will be rephrased by that agent before being spoken aloud.

CRITICAL — TOOL USE IS MANDATORY:
- You MUST use the ask_user tool for EVERY question to the user. NEVER \
  write a question as plain text. Always call ask_user(question="...").
- You MUST call the appropriate tools (start_login_tool, verify_login_tool, \
  get_roaming_info_tool, etc.) to perform actions. NEVER pretend you did \
  something without actually calling the tool.
- The ONLY time you should return plain text (without a tool call) is when \
  you have a final answer or result to deliver after all tools have been called.

Because of this:
- Do NOT greet the user or make small talk.
- Do NOT say "Good morning" or "Happy to help" or any pleasantries.
- Be direct and functional in your ask_user questions. Just state what \
  you need: "What is your mobile phone number?" not "Hi there! Could \
  you please provide your number?"
- When returning results, provide clear factual information only.

AUTHENTICATION FLOW (mandatory before any account access):
1. Call ask_user to get the mobile phone number.
2. Once you have the number, call start_login_tool to send an OTP via SMS.
   Pass the session_id provided in the conversation context.
3. Call ask_user to get the 6-digit code.
4. Call verify_login_tool with the code. If verified=false, call ask_user \
   again (up to 2 retries).
5. Do NOT access any account tools until verify_login_tool returns verified=true.

AFTER AUTHENTICATION, you can help with:
- get_current_package_tool: current package, contract, addons
- get_data_balance_tool: data usage and remaining allowance
- list_available_packages_tool: all available packages
- recommend_packages_tool: personalized package recommendations
- get_roaming_info_tool: roaming pricing and passes for a country
- purchase_roaming_pass_tool: buy a roaming pass
- change_package_tool: switch to a different package
- get_billing_summary_tool: billing and monthly fee
- set_data_alerts_tool: set data usage alerts
- list_addons_tool: list active addons
- close_contract_tool: close contract (call with confirm=false first \
  to show early termination fee, then confirm=true only after user confirms)

OUTPUT RULES:
- Be concise and factual. State numbers clearly.
- Plain text only, no markdown or formatting.
- When presenting options, use flowing sentences not lists.
"""


@task
async def call_llm(messages: list[dict]) -> dict:
    """Invoke the LLM with telco tools bound."""
    llm = get_telco_workflow_llm()
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
            tool_calls = m.get("tool_calls")
            lc_messages.append(AIMessage(content=content, tool_calls=tool_calls or []))
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
    """Execute a telco tool. Returns a serializable ToolMessage dict."""
    tool_fn = TOOLS_BY_NAME[name]
    raw_result = await tool_fn.ainvoke(args)
    content = raw_result if isinstance(raw_result, str) else json.dumps(raw_result)
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": tool_call_id,
        "name": name,
    }


_QUESTION_STARTERS = (
    "what is", "what's", "could you", "can you", "please provide",
    "may i have", "i need your", "do you have", "which country",
    "what country",
)


def _coerce_text_to_ask_user(response: dict, messages: list[dict]) -> dict:
    """If the LLM returned a question as plain text instead of calling
    ask_user, synthesize the tool call so interrupt() fires properly."""
    if response.get("tool_calls"):
        return response

    text = (response.get("content") or "").strip()
    if not text:
        return response

    lower = text.lower()
    is_question = text.endswith("?") or any(lower.startswith(q) for q in _QUESTION_STARTERS)
    if not is_question:
        return response

    tc_id = f"synthetic_{uuid.uuid4().hex[:8]}"
    logger.warning(
        f"[telco] LLM returned question as text — synthesizing ask_user: {text[:80]}"
    )
    response["tool_calls"] = [
        {"id": tc_id, "name": "ask_user", "args": {"question": text}}
    ]
    response["content"] = ""
    messages[-1] = response
    return response


@entrypoint()
async def graph(inputs: dict, *, previous: list[dict] | None = None) -> entrypoint.final[str, list[dict]]:
    """Telco ReAct workflow agent.

    Inputs:
        message: str - the user's request or greeting
        session_id: str - used for OTP session tracking
    Previous:
        list of message dicts (conversation across interrupt/resume cycles)
    """
    session_id = inputs.get("session_id", "default")
    messages: list[dict] = list(previous or [])

    if not messages or messages[0].get("role") != "system":
        system_content = (
            SYSTEM_PROMPT
            + f"\n\nSession context: session_id={session_id}. "
            "Pass this session_id when calling start_login_tool and verify_login_tool."
        )
        messages.insert(0, {"role": "system", "content": system_content})

    user_msg = inputs.get("message", "")
    if user_msg:
        messages.append({"role": "user", "content": user_msg})
        logger.info(f"[telco] User: {user_msg[:100]}")

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
                logger.info(f"[telco] ask_user: {question}")
                answer = interrupt({"question": question})
                logger.info(f"[telco] User answered: {str(answer)[:100]}")
                messages.append({
                    "role": "tool",
                    "content": str(answer),
                    "tool_call_id": tc_id,
                    "name": "ask_user",
                })
            else:
                if tc_name in ("start_login_tool", "verify_login_tool"):
                    tc_args.setdefault("session_id", session_id)
                logger.info(f"[telco] Calling tool: {tc_name}({tc_args})")
                tool_result = await execute_tool(tc_name, tc_args, tc_id)
                messages.append(tool_result)
                logger.info(f"[telco] Tool result: {tool_result['content'][:200]}")

        response = await call_llm(messages)
        messages.append(response)
        response = _coerce_text_to_ask_user(response, messages)

    final_text = response.get("content", "")
    logger.info(f"[telco] Final: {final_text[:100]}")

    return entrypoint.final(value=final_text, save=messages)
