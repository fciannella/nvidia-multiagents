"""Router agent: classifies user intent and extracts tool parameters.

Stateless one-shot classifier. Receives the current user message plus
recent conversation history (provided by the client). Returns a
RouterDecision dict indicating whether a tool should be launched.

Uses with_structured_output (tool-calling) for guaranteed schema-compliant
responses — no manual JSON parsing or keyword fallbacks needed.
"""

from typing import Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.func import entrypoint
from loguru import logger
from pydantic import BaseModel, Field

from config import get_router_llm
from tools.mock_tools import get_available_tools

_TOOLS = get_available_tools()
TOOL_LIST = "\n".join(
    f"- {t['name']}: {t['description']} (~{t['duration_secs']}s)"
    for t in _TOOLS
)
_TOOL_NAMES = [t["name"] for t in _TOOLS]


class RouterDecision(BaseModel):
    """Intent classification result for a user message."""

    action: Literal["launch_tool", "none"] = Field(
        description=(
            "'launch_tool' if the user explicitly requests starting a NEW task, "
            "'none' otherwise"
        )
    )
    tool: Optional[str] = Field(
        None,
        description=(
            f"Tool name to launch. Valid values: {', '.join(_TOOL_NAMES)}. "
            "Only set when action is 'launch_tool'."
        ),
    )
    params: Optional[dict] = Field(
        default_factory=dict,
        description="Parameters for the tool, extracted from the user's request",
    )
    context_summary: Optional[str] = Field(
        None,
        description="Brief summary of what the user wants done",
    )


SYSTEM_PROMPT = f"""\
You are an intent classifier. Classify the user's message.

Available tools:
{TOOL_LIST}

Set action to "launch_tool" ONLY when the user explicitly asks to START a new task:
- "run the analysis", "look up revenue", "train a model", "search for X"

Set action to "none" for everything else:
- Asking about STATUS of a running job ("what's the status?", "how far along?")
- Asking for jokes, stories, fun, or casual conversation
- Asking what you can do, who you are, greetings
- Commenting on previous results ("that's interesting", "tell me more")
- Any question that is NOT requesting a NEW task to begin

When in doubt, set action to "none".
"""


@entrypoint()
async def graph(inputs: dict) -> dict:
    """Classify user intent. Returns structured action dict."""
    llm = get_router_llm()
    structured_llm = llm.with_structured_output(RouterDecision)

    message = inputs["message"]
    recent_history = inputs.get("recent_history", [])

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for turn in recent_history[-6:]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))

    messages.append(HumanMessage(content=message))

    try:
        decision: RouterDecision = await structured_llm.ainvoke(messages)
    except Exception as e:
        logger.warning(f"[router] Structured output failed: {e}")
        return {"action": "none"}

    result = decision.model_dump()

    if result["action"] == "launch_tool":
        if result.get("tool") not in _TOOL_NAMES:
            logger.warning(
                f"[router] Unknown tool '{result.get('tool')}', defaulting to none"
            )
            return {"action": "none"}
        logger.info(
            f"[router] Decision: launch {result['tool']} — "
            f"{result.get('context_summary', '')}"
        )
    else:
        logger.debug("[router] Decision: no action")

    return result
