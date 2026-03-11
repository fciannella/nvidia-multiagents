"""Shared configuration: one ChatOpenAI client per agent role.

Each role (chitchat, router, worker) reads its own LLM endpoint and model
from environment variables so they can be swapped independently.
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


def get_chitchat_llm() -> ChatOpenAI:
    model = os.environ["CHITCHAT_LLM_MODEL"]
    return ChatOpenAI(
        base_url=os.environ["CHITCHAT_LLM_BASE_URL"],
        model=model,
        api_key="not-needed",
        temperature=0.7,
        max_tokens=512,
        **_thinking_extra(model),
    )


def get_router_llm() -> ChatOpenAI:
    model = os.environ["ROUTER_LLM_MODEL"]
    return ChatOpenAI(
        base_url=os.environ["ROUTER_LLM_BASE_URL"],
        model=model,
        api_key="not-needed",
        temperature=0.0,
        max_tokens=256,
        **_thinking_extra(model),
    )


def get_worker_llm() -> ChatOpenAI:
    model = os.environ["WORKER_LLM_MODEL"]
    return ChatOpenAI(
        base_url=os.environ["WORKER_LLM_BASE_URL"],
        model=model,
        api_key="not-needed",
        temperature=0.3,
        max_tokens=1024,
        **_thinking_extra(model),
    )


def _openrouter_llm(
    base_url_key: str,
    model_key: str,
    api_key_key: str = "",
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> ChatOpenAI:
    """Shared factory for workflow LLMs that may run on OpenRouter or locally."""
    api_key = os.environ.get(
        api_key_key,
        os.environ.get("OPENROUTER_API_KEY", "not-needed"),
    ) if api_key_key else os.environ.get("OPENROUTER_API_KEY", "not-needed")
    base_url = os.environ.get(base_url_key, os.environ["WORKER_LLM_BASE_URL"])
    model = os.environ.get(model_key, os.environ["WORKER_LLM_MODEL"])
    is_local = "localhost" in base_url or "192.168" in base_url
    extra = _thinking_extra(model) if is_local else {}
    return ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **extra,
    )


def get_workflow_llm() -> ChatOpenAI:
    return _openrouter_llm(
        base_url_key="WORKFLOW_LLM_BASE_URL",
        model_key="WORKFLOW_LLM_MODEL",
        api_key_key="WORKFLOW_LLM_API_KEY",
    )


def get_telco_workflow_llm() -> ChatOpenAI:
    return _openrouter_llm(
        base_url_key="TELCO_LLM_BASE_URL",
        model_key="TELCO_LLM_MODEL",
        api_key_key="TELCO_LLM_API_KEY",
    )
