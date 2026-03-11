"""Shared LLM configuration for the react agent system.

Two roles:
  - front_agent: fast model for low-latency conversation + delegation decisions
  - executor: capable model for tool-calling workflows

Each reads its own endpoint/model from env vars with fallback to the
common WORKER_LLM_* variables, so you can run both on the same model
during development.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DISABLE_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
_SKIP_THINKING_MODELS = ("mistral", "ministral")


def _thinking_extra(model: str) -> dict:
    if any(k in model.lower() for k in _SKIP_THINKING_MODELS):
        return {}
    return {"extra_body": DISABLE_THINKING}


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_FAST_MODEL = "x-ai/grok-4.1-fast"
OPENROUTER_DEFAULT_CAPABLE_MODEL = "google/gemini-3-flash-preview"


def _build_llm(
    base_url_key: str,
    model_key: str,
    api_key_key: str = "",
    temperature: float = 0.7,
    max_tokens: int = 512,
    fallback_base_url_key: str = "WORKER_LLM_BASE_URL",
    fallback_model_key: str = "WORKER_LLM_MODEL",
    default_model: str = "",
) -> ChatOpenAI:
    api_key = (
        os.environ.get(api_key_key)
        or os.environ.get("OPENROUTER_API_KEY", "not-needed")
    ) if api_key_key else os.environ.get("OPENROUTER_API_KEY", "not-needed")

    base_url = (
        os.environ.get(base_url_key)
        or os.environ.get(fallback_base_url_key)
        or OPENROUTER_BASE_URL
    )
    model = (
        os.environ.get(model_key)
        or os.environ.get(fallback_model_key)
        or default_model
        or OPENROUTER_DEFAULT_FAST_MODEL
    )

    is_local = "localhost" in base_url or "127.0.0.1" in base_url or "192.168" in base_url
    extra = _thinking_extra(model) if is_local else {}

    return ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **extra,
    )


def get_front_agent_llm() -> ChatOpenAI:
    """Fast conversational model — optimised for low time-to-first-token."""
    return _build_llm(
        base_url_key="FRONT_AGENT_LLM_BASE_URL",
        model_key="FRONT_AGENT_LLM_MODEL",
        api_key_key="FRONT_AGENT_LLM_API_KEY",
        temperature=0.7,
        max_tokens=512,
        fallback_base_url_key="CHITCHAT_LLM_BASE_URL",
        fallback_model_key="CHITCHAT_LLM_MODEL",
        default_model=OPENROUTER_DEFAULT_FAST_MODEL,
    )


def get_executor_llm() -> ChatOpenAI:
    """Capable model for multi-step tool calling workflows."""
    return _build_llm(
        base_url_key="EXECUTOR_LLM_BASE_URL",
        model_key="EXECUTOR_LLM_MODEL",
        api_key_key="EXECUTOR_LLM_API_KEY",
        temperature=0.3,
        max_tokens=1024,
        fallback_base_url_key="TELCO_LLM_BASE_URL",
        fallback_model_key="TELCO_LLM_MODEL",
        default_model=OPENROUTER_DEFAULT_CAPABLE_MODEL,
    )
