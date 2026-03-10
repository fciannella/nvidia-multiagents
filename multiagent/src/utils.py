"""Message conversion and formatting utilities."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

AGENT_LABELS = {
    "data_scientist": "Sarah",
    "it": "Mike",
}


def to_langchain_messages(
    messages: list[dict],
    perspective: str = "",
) -> list[BaseMessage]:
    """Convert graph-state message dicts to LangChain message objects.

    When *perspective* is set (e.g. "data_scientist"), only that agent's
    messages become AIMessage.  The OTHER agent's messages become
    HumanMessage with a "[Name said]:" prefix so the LLM clearly knows
    they came from a different speaker and won't repeat them as its own.
    """
    lc: list[BaseMessage] = []
    for msg in messages:
        speaker = msg.get("speaker", "human")
        text = msg.get("text", "")
        if speaker == "human":
            lc.append(HumanMessage(content=text, name="human"))
        elif perspective and speaker != perspective:
            label = AGENT_LABELS.get(speaker, speaker)
            lc.append(HumanMessage(
                content=f"[{label} said]: {text}",
                name=speaker,
            ))
        else:
            lc.append(AIMessage(content=text, name=speaker))
    return lc


def format_context(messages: list[dict], last_n: int = 4) -> str:
    """Format the most recent messages as context for the router.

    Uses human-readable names so the router can associate follow-ups
    with the correct agent.
    """
    if not messages:
        return "(Start of conversation)"
    lines = []
    for msg in messages[-last_n:]:
        speaker = msg.get("speaker", "unknown")
        label = AGENT_LABELS.get(speaker, speaker)
        text = msg.get("text", "")
        if len(text) > 200:
            text = text[:200] + "..."
        lines.append(f"[{label}]: {text}")
    return "\n".join(lines)
