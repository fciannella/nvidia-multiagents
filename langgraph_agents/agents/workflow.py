"""Workflow agent: ReAct loop with tool calling and human-in-the-loop.

Runs on its own thread. The LLM has data tools bound via bind_tools()
and can call `ask_user` to pause execution (via interrupt()) and wait
for the user's answer. On resume, the interrupt returns the user's text
and the ReAct loop continues.

Key mechanics:
- All LLM calls and tool executions are wrapped in @task for
  deterministic replay on resume after interrupt().
- `ask_user` calls interrupt() directly (not via @task) since it's
  the suspension point itself.
- Messages are accumulated in the loop and saved via entrypoint.final
  when the workflow completes.
"""

import json

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

from config import get_workflow_llm
from tools.workflow_tools import ALL_TOOLS, TOOLS_BY_NAME, workflow_session_id

SYSTEM_PROMPT = """\
You are a task execution agent. You MUST use your tools to accomplish tasks.
NEVER respond with plain text when a tool call is appropriate.

CRITICAL RULES:
- When the user asks you to run an analysis, IMMEDIATELY call the appropriate
  tool (long_analysis, quick_lookup, or train_model). Do NOT ask for
  clarification unless absolutely necessary — use reasonable defaults.
- If you genuinely need information you cannot infer, call the `ask_user` tool.
  NEVER ask questions in plain text — ALWAYS use the ask_user tool.
- When a tool returns results, produce a clear text summary in plain,
  flowing sentences suitable for text-to-speech. No markdown, no bullet
  points, no special formatting.
- Keep responses concise (3-6 sentences for results).

Tool usage:
- `long_analysis`: For financial analysis, data analysis, trend detection.
  Use dataset="Q3" or similar when the user specifies a quarter/period.
- `quick_lookup`: For quick data searches and lookups.
- `train_model`: For training ML models.
- `ask_user`: ONLY when you cannot proceed without user input.
"""


@task
async def call_llm(messages: list[dict]) -> dict:
    """Invoke the LLM with tools bound. Returns serializable dict."""
    llm = get_workflow_llm()
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

    result = {
        "role": "assistant",
        "content": response.content or "",
        "tool_calls": [
            {"id": tc["id"], "name": tc["name"], "args": tc.get("args", {})}
            for tc in (response.tool_calls or [])
        ],
    }
    return result


@task
async def execute_tool(name: str, args: dict, tool_call_id: str) -> dict:
    """Execute a non-interrupt tool. Returns a serializable ToolMessage dict."""
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
async def graph(inputs: dict, *, previous: list[dict] | None = None) -> entrypoint.final[str, list[dict]]:
    """ReAct workflow agent.

    Inputs:
        message: str - the user's request or task description
    Previous:
        list of message dicts (conversation history across interrupt/resume cycles)
    Returns:
        The final text answer (value) and accumulated messages (save).
    """
    session_id = inputs.get("session_id", "default")
    workflow_session_id.set(session_id)

    messages: list[dict] = list(previous or [])

    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    user_msg = inputs.get("message", "")
    if user_msg:
        messages.append({"role": "user", "content": user_msg})
        logger.info(f"[workflow] User: {user_msg[:100]}")

    response = await call_llm(messages)
    messages.append(response)

    max_iterations = 20
    iteration = 0

    while response.get("tool_calls") and iteration < max_iterations:
        iteration += 1

        for tc in response["tool_calls"]:
            tc_name = tc["name"]
            tc_args = tc.get("args", {})
            tc_id = tc["id"]

            if tc_name == "ask_user":
                question = tc_args.get("question", "Could you clarify?")
                logger.info(f"[workflow] ask_user: {question}")
                answer = interrupt({"question": question})
                logger.info(f"[workflow] User answered: {str(answer)[:100]}")
                messages.append({
                    "role": "tool",
                    "content": str(answer),
                    "tool_call_id": tc_id,
                    "name": "ask_user",
                })
            else:
                logger.info(f"[workflow] Calling tool: {tc_name}({tc_args})")
                tool_result = await execute_tool(tc_name, tc_args, tc_id)
                messages.append(tool_result)
                logger.info(f"[workflow] Tool result: {tool_result['content'][:100]}")

        response = await call_llm(messages)
        messages.append(response)

    final_text = response.get("content", "")
    logger.info(f"[workflow] Final answer: {final_text[:100]}")

    return entrypoint.final(value=final_text, save=messages)
