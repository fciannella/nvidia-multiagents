"""Pydantic models for the multi-agent orchestrator.

Only the RouterDecision uses structured output (fast, small).
Agents stream plain text — no structured output — for lowest TTFA.
"""

from typing import Literal

from pydantic import BaseModel, Field


class RoutingDecision(BaseModel):
    """Router output: who should speak and about what."""

    primary_agent: Literal["data_scientist", "it"] = Field(
        ..., description="The primary agent that should respond"
    )
    secondary_agent: Literal["data_scientist", "it", "none"] = Field(
        default="none", description="Secondary agent, or 'none'"
    )
    topic: str = Field(
        default="general", description="Detected topic of the utterance"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Routing confidence"
    )
    tools_allowed: bool = Field(
        default=False,
        description=(
            "True ONLY when the human explicitly asked to perform an action "
            "or check real-time data (e.g. 'check status', 'run training', "
            "'deploy'). False for explanations, introductions, opinions."
        ),
    )


class HandoffCheck(BaseModel):
    """Quick re-route check: should the other agent speak after the primary?"""

    should_speak: bool = Field(
        ..., description="True if the other agent should respond to what was said"
    )
    tools_allowed: bool = Field(
        default=False,
        description=(
            "True only if the other agent needs to check real-time data "
            "or perform an action to answer properly"
        ),
    )


class ProjectPlan(BaseModel):
    """Lightweight shared project plan that both agents can see."""

    training_notes: list[str] = Field(default_factory=list)
    infra_notes: list[str] = Field(default_factory=list)
    deployment_notes: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    project_name: str | None = None
    target_completion: str | None = None

    def summary(self) -> str:
        if not any([
            self.training_notes, self.infra_notes,
            self.deployment_notes, self.blockers, self.project_name,
        ]):
            return "No plan details yet."
        parts: list[str] = []
        if self.project_name:
            parts.append(f"Project: {self.project_name}")
        if self.target_completion:
            parts.append(f"Target: {self.target_completion}")
        if self.training_notes:
            parts.append("Training: " + "; ".join(self.training_notes[-3:]))
        if self.infra_notes:
            parts.append("Infra: " + "; ".join(self.infra_notes[-3:]))
        if self.deployment_notes:
            parts.append("Deployment: " + "; ".join(self.deployment_notes[-3:]))
        if self.blockers:
            parts.append("Blockers: " + "; ".join(self.blockers[-3:]))
        return "\n".join(parts)
