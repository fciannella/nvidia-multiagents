"""Message conversion and formatting utilities.

Agent labels are loaded from the active scenario config instead of being
hardcoded, so new scenarios work without code changes here.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.scenarios import ScenarioConfig


def get_agent_labels(scenario: ScenarioConfig) -> dict[str, str]:
    """Return {agent_id: display_name} from the scenario config."""
    return scenario.agent_names


def to_langchain_messages(
    messages: list[dict],
    perspective: str = "",
    agent_labels: dict[str, str] | None = None,
) -> list[BaseMessage]:
    """Convert graph-state message dicts to LangChain message objects.

    When *perspective* is set (e.g. "data_scientist"), only that agent's
    messages become AIMessage.  The OTHER agent's messages become
    HumanMessage with a "[Name said]:" prefix so the LLM clearly knows
    they came from a different speaker.
    """
    labels = agent_labels or {}
    lc: list[BaseMessage] = []
    for msg in messages:
        speaker = msg.get("speaker", "human")
        text = msg.get("text", "")
        if speaker == "human":
            lc.append(HumanMessage(content=text, name="human"))
        elif perspective and speaker != perspective:
            label = labels.get(speaker, speaker)
            lc.append(HumanMessage(
                content=f"[{label} said]: {text}",
                name=speaker,
            ))
        else:
            lc.append(AIMessage(content=text, name=speaker))
    return lc


def format_context(
    messages: list[dict],
    last_n: int = 4,
    agent_labels: dict[str, str] | None = None,
) -> str:
    """Format the most recent messages as context for the router."""
    labels = agent_labels or {}
    if not messages:
        return "(Start of conversation)"
    lines = []
    for msg in messages[-last_n:]:
        speaker = msg.get("speaker", "unknown")
        label = labels.get(speaker, speaker)
        text = msg.get("text", "")
        if len(text) > 200:
            text = text[:200] + "..."
        lines.append(f"[{label}]: {text}")
    return "\n".join(lines)
