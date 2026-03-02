"""Shared configuration: one ChatOpenAI client per agent role.

Each role (chitchat, router, worker) reads its own LLM endpoint and model
from environment variables so they can be swapped independently.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DISABLE_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


def get_chitchat_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=os.environ["CHITCHAT_LLM_BASE_URL"],
        model=os.environ["CHITCHAT_LLM_MODEL"],
        api_key="not-needed",
        temperature=0.7,
        max_tokens=512,
        extra_body=DISABLE_THINKING,
    )


def get_router_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=os.environ["ROUTER_LLM_BASE_URL"],
        model=os.environ["ROUTER_LLM_MODEL"],
        api_key="not-needed",
        temperature=0.0,
        max_tokens=256,
        extra_body=DISABLE_THINKING,
    )


def get_worker_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=os.environ["WORKER_LLM_BASE_URL"],
        model=os.environ["WORKER_LLM_MODEL"],
        api_key="not-needed",
        temperature=0.3,
        max_tokens=1024,
        extra_body=DISABLE_THINKING,
    )
