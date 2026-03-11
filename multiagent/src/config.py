"""LLM configuration for the multi-agent orchestrator.

All agents use the same LLM endpoint (configurable per-role).
Environment variables allow overriding per-agent.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DISABLE_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


def _build_llm(
    base_url_key: str,
    model_key: str,
    api_key_key: str = "",
    temperature: float = 0.7,
    max_tokens: int = 512,
    default_base_url: str = "http://localhost:18008/v1",
    default_model: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
) -> ChatOpenAI:
    base_url = os.getenv(base_url_key, default_base_url)
    model = os.getenv(model_key, default_model)
    api_key = os.getenv(api_key_key, "not-needed") if api_key_key else "not-needed"
    is_local = "localhost" in base_url or "192.168" in base_url
    supports_thinking = is_local and not any(
        k in model.lower() for k in ("mistral", "ministral")
    )

    return ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **({"extra_body": DISABLE_THINKING} if supports_thinking else {}),
    )


def get_router_model() -> ChatOpenAI:
    base_url = os.getenv("MA_ROUTER_BASE_URL", "")
    is_openrouter = "openrouter" in base_url
    return _build_llm(
        base_url_key="MA_ROUTER_BASE_URL",
        model_key="MA_ROUTER_MODEL",
        api_key_key="OPENROUTER_API_KEY" if is_openrouter else "",
        temperature=0.0,
        max_tokens=256,
    )


def get_agent_model() -> ChatOpenAI:
    """Get the LLM for any agent's responses. All agents share the same model."""
    base_url = os.getenv(
        "MA_AGENT_BASE_URL",
        os.getenv("MA_DS_BASE_URL", ""),
    )
    is_openrouter = "openrouter" in base_url
    return _build_llm(
        base_url_key="MA_AGENT_BASE_URL" if os.getenv("MA_AGENT_BASE_URL") else "MA_DS_BASE_URL",
        model_key="MA_AGENT_MODEL" if os.getenv("MA_AGENT_MODEL") else "MA_DS_MODEL",
        api_key_key="OPENROUTER_API_KEY" if is_openrouter else "",
        temperature=0.7,
        max_tokens=512,
    )


def get_scenario_id() -> str:
    return os.getenv("MA_SCENARIO", "ecommerce")
