"""Scenario configuration system for multi-agent voice pipelines.

Each scenario defines a set of agents with their prompts, tools, voices,
and domain context.  The orchestrator reads the active scenario at runtime
to determine routing, prompt generation, and tool binding.

Usage:
    from src.scenarios import get_scenario, list_scenarios

    scenario = get_scenario("ecommerce")
    for agent in scenario.agents:
        print(agent.id, agent.name)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

_REGISTRY: dict[str, "ScenarioConfig"] = {}


@dataclass
class AgentConfig:
    """Configuration for a single agent within a scenario."""

    id: str
    name: str
    role: str
    domain_keywords: list[str]
    voice_en: str
    color: str
    situation: str
    intro_example: str
    personality_quips: list[str] = field(default_factory=list)
    fast_tools: list = field(default_factory=list)
    slow_launchers: list = field(default_factory=list)
    launcher_to_executor: dict[str, str] = field(default_factory=dict)
    slow_tools: dict[str, Any] = field(default_factory=dict)
    delegation_patterns: list[tuple[str, str, dict]] = field(default_factory=list)

    @property
    def all_tools(self) -> list:
        return self.fast_tools + self.slow_launchers


@dataclass
class ScenarioConfig:
    """Full configuration for a multi-agent scenario."""

    id: str
    title: str
    description: str
    agents: list[AgentConfig]
    guide_overview: str = ""
    suggested_questions: list[str] = field(default_factory=list)

    def get_agent(self, agent_id: str) -> AgentConfig:
        for a in self.agents:
            if a.id == agent_id:
                return a
        raise ValueError(
            f"Unknown agent '{agent_id}' in scenario '{self.id}'. "
            f"Available: {self.agent_ids}"
        )

    @property
    def agent_ids(self) -> list[str]:
        return [a.id for a in self.agents]

    @property
    def agent_names(self) -> dict[str, str]:
        return {a.id: a.name for a in self.agents}

    @property
    def agent_colors(self) -> dict[str, str]:
        return {a.id: a.color for a in self.agents}

    @property
    def agent_voices(self) -> dict[str, str]:
        return {a.id: a.voice_en for a in self.agents}


def register_scenario(scenario: ScenarioConfig):
    _REGISTRY[scenario.id] = scenario


def get_scenario(scenario_id: str) -> ScenarioConfig:
    if scenario_id not in _REGISTRY:
        _load_all()
    if scenario_id not in _REGISTRY:
        raise ValueError(
            f"Unknown scenario: {scenario_id}. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[scenario_id]


def list_scenarios() -> list[ScenarioConfig]:
    if not _REGISTRY:
        _load_all()
    return list(_REGISTRY.values())


def _load_all():
    """Import all scenario modules so they self-register."""
    from . import ecommerce as _  # noqa: F401
    from . import autonomous_vehicle as _  # noqa: F401
    from . import healthcare as _  # noqa: F401
    from . import game_studio as _  # noqa: F401
