"""Generic agent graph with chitchat mode toggle.

When mode="normal" (default): full ReAct agent with tools.
When mode="chitchat": conversational-only, no tools, aware of main thread status
                      via the cross-thread LangGraph Store.

The client controls which mode is used by setting the `mode` field in the input.
"""

import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from loguru import logger

from tools import ALL_TOOLS

load_dotenv()

NORMAL_SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to data analysis, system monitoring, \
and model training tools.

You speak naturally and conversationally — you're the user's colleague, not a robot.

When the user asks you to do something that requires a tool, use it. \
For quick questions or chitchat, just respond directly.

IMPORTANT: When you decide to call a tool, you MUST ALWAYS include a brief spoken \
message in your response alongside the tool call. For example: \
"Sure, let me run that analysis for you — this might take a little while." \
NEVER call a tool with empty text content. The user is listening via voice and \
needs to hear that something is happening.

Keep responses concise (2-4 sentences) unless the user asks for detail. \
When delivering tool results, summarize the key findings conversationally — \
don't dump raw JSON.
"""

CHITCHAT_SYSTEM_PROMPT = """\
You are a friendly AI assistant. Right now your main processing thread is busy \
working on a task for the user.

You are on a SECONDARY conversation thread. Your ONLY job is to keep the user \
company with light conversation while the work completes.

STRICT RULES:
- Do NOT call any tools. You have no tools available.
- Do NOT offer to start new tasks or analyses.
- If the user asks you to do something, politely explain you're currently \
  busy with the ongoing task and will handle it once that's done.
- You CAN and SHOULD tell the user about the status of the ongoing task \
  when they ask. Use the live progress information below.
- Keep responses short and friendly (1-3 sentences).
- Be personable — small talk, humor, and casual conversation are encouraged.

{status_context}
"""


class AgentState(dict):
    messages: Annotated[list[AnyMessage], add_messages]
    mode: str
    main_thread_status: str


def _get_llm(with_tools: bool = False) -> ChatOpenAI:
    base_url = os.getenv("AGENT_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("AGENT_LLM_MODEL", "google/gemini-3-flash-preview")
    api_key = os.getenv("OPENROUTER_API_KEY", "not-needed")

    llm = ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=0.7,
        max_tokens=512,
    )

    if with_tools:
        llm = llm.bind_tools(ALL_TOOLS)

    return llm


def _session_ns(config: RunnableConfig) -> tuple:
    sid = config.get("configurable", {}).get("session_id", "default")
    return ("session", sid)


def _build_store_context(store: BaseStore, config: RunnableConfig) -> str:
    """Read live task progress from the cross-thread store."""
    ns = _session_ns(config)
    try:
        item = store.get(ns, "task_progress")
    except Exception as e:
        logger.warning(f"[graph] Store read failed: {e}")
        return ""

    if not item:
        return ""

    p = item.value
    tool_name = p.get("tool", "unknown task")
    status = p.get("status", "unknown")

    if status == "running":
        lines = [f"LIVE TASK PROGRESS (from shared memory):"]
        lines.append(f"  Tool: {tool_name}")
        if p.get("phase"):
            lines.append(f"  Current phase: {p['phase']}")
        if p.get("progress"):
            lines.append(f"  Progress: {p['progress']}")
        if p.get("epoch"):
            lines.append(f"  Epoch: {p['epoch']}")
        if p.get("loss"):
            lines.append(f"  Current loss: {p['loss']}")
        if p.get("step"):
            lines.append(f"  Step: {p['step']}")
        if p.get("dataset"):
            lines.append(f"  Dataset: {p['dataset']}")
        if p.get("model_name"):
            lines.append(f"  Model: {p['model_name']}")
        if p.get("focus"):
            lines.append(f"  Focus: {p['focus']}")
        return "\n".join(lines)

    if status == "complete":
        summary = p.get("result_summary", "done")
        return f"TASK JUST COMPLETED: {tool_name}\nResult: {summary}"

    if status == "starting":
        return f"TASK STARTING: {tool_name} (just initiated, no progress yet)"

    return ""


def agent_node(state: dict, config: RunnableConfig, *, store: BaseStore) -> dict:
    mode = state.get("mode", "normal")
    messages = state.get("messages", [])

    if mode == "chitchat":
        store_context = _build_store_context(store, config)
        if not store_context:
            fallback = state.get("main_thread_status", "")
            if fallback:
                store_context = f"CURRENT TASK STATUS: {fallback}"
        system = CHITCHAT_SYSTEM_PROMPT.format(status_context=store_context)
        llm = _get_llm(with_tools=False)
    else:
        system = NORMAL_SYSTEM_PROMPT
        llm = _get_llm(with_tools=True)

    full_messages = [SystemMessage(content=system)] + messages
    response = llm.invoke(full_messages)

    if mode != "chitchat" and isinstance(response, AIMessage) and response.tool_calls:
        ns = _session_ns(config)
        tool_names = [tc["name"] for tc in response.tool_calls]
        try:
            store.put(
                ns,
                "task_progress",
                {
                    "tool": ", ".join(tool_names),
                    "status": "starting",
                    "phase": "Tool call initiated by agent",
                    "progress": "0%",
                },
            )
            logger.info(f"[graph] Wrote tool-start to store: {tool_names}")
        except Exception as e:
            logger.warning(f"[graph] Store write failed: {e}")

    return {"messages": [response]}


def should_continue(state: dict) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


tool_node = ToolNode(ALL_TOOLS)

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

graph = builder.compile()
