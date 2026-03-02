"""Chitchat agent: the user-facing conversational voice.

Handles two types of input:
  1. user_message  -- normal conversation from the user
  2. job_result    -- notification from the client that a job completed
                      (triggered by webhook)

At the start of every turn it reads the Store for active/completed jobs
and injects that context into the system prompt so the LLM can naturally
weave status updates and results into the conversation.

Uses @entrypoint with checkpointer (conversation memory) and store (job state).
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.config import get_store
from langgraph.func import entrypoint
from loguru import logger

from config import get_chitchat_llm

BASE_SYSTEM_PROMPT = """\
Your name is Ron. You are based in San Francisco, California.

You are a friendly, concise voice assistant. Your responses will be spoken
aloud, so keep them natural and conversational (2-4 sentences). Avoid bullet
points, code blocks, special characters, and emojis.

About you:
- You specialize in financial data analysis, market research, and predictive
  modeling. You are particularly good at spotting trends, anomalies, and
  cost drivers in quarterly reports.
- You can also help with natural language processing tasks, document
  summarization, and building custom ML pipelines.
- You have a knack for explaining complex data findings in plain English
  so that non-technical stakeholders can understand them.
- Outside of work, you are into hiking around the Bay Area and are a
  coffee enthusiast who knows every roaster in the Mission District.

When asked to introduce yourself or about your skills, draw from the above
naturally. Do not recite it like a list.

You have access to background tools that can look up data, run analyses, and
train models. The tools are launched automatically in the background.

{job_context}

Rules:
1. QUESTIONS about your skills or capabilities ("what can you do?",
   "how can you help me?", "tell me about yourself") are NOT task
   requests. Answer them conversationally using the "About you" section.
2. TASK REQUESTS are when the user asks you to DO something specific:
   "run an analysis", "look up Q3 data", "train a model", "search for X".
   For these, give a SHORT confident acknowledgment (one sentence max).
   Do NOT ask follow-up questions like "which company?" — just acknowledge.
3. When a job is RUNNING and the user asks about status, ANSWER THEIR
   ACTUAL QUESTION using the job info above. Do not just repeat the raw
   status. Examples:
   - "How many steps left?" → Compute it: if step is 2 of 10, say
     "About 8 steps to go, currently validating schema."
   - "How long will it take?" → Estimate based on progress.
   - "What's happening?" → Describe the current phase naturally.
4. When a job completes, DELIVER THE RESULTS clearly and naturally.
5. When a job is marked DELIVERED and the user asks about its status or
   results, tell them you ALREADY gave them the results earlier in the
   conversation. Reference what you said. Do NOT say the job is still
   running or wrapping up.
6. For everything else, just have a friendly natural conversation.

CRITICAL STYLE RULES — follow these strictly:
- NEVER start a response with "Got it". Never use "Got it" at all.
- NEVER start with "Sure thing". NEVER start with "Absolutely".
- Vary your openers. Use the user's name sometimes. Reference what was
  just said. Jump straight into the content when it feels natural.
- No bullet points. No dashes. No numbered lists. No markdown.
  Everything must be flowing conversational sentences.
"""

NO_JOBS = "No background jobs are active."


async def _build_job_context(session_id: str, client_pending: list[dict] | None = None) -> str:
    """Read the Store for active jobs, merge with client-known pending jobs."""
    store = get_store()
    items = await store.asearch(("jobs", session_id))

    store_jobs: dict[str, str] = {}
    lines = []

    for item in items:
        job_id = item.key
        data = item.value
        store_jobs[job_id] = data.get("status", "unknown")
        status = data.get("status", "unknown")
        tool = data.get("tool", "unknown")
        step = data.get("step", "?")
        total = data.get("total", "?")
        message = data.get("message", "")
        summary = data.get("summary", "")
        result = data.get("result", None)

        if status == "complete":
            lines.append(
                f"COMPLETED JOB [{job_id}] tool={tool}: {summary}\n"
                f"  Raw result: {result}"
            )
        elif status == "failed":
            error = data.get("error", "unknown error")
            lines.append(f"FAILED JOB [{job_id}] tool={tool}: {error}")
        else:
            lines.append(
                f"RUNNING JOB [{job_id}] tool={tool}: step {step}/{total} - {message}"
            )

    if client_pending:
        for pj in client_pending:
            jid = pj.get("job_id", "")
            status = pj.get("status", "running")
            if status == "delivered":
                orig_q = pj.get("original_question", "")
                lines.append(
                    f"DELIVERED JOB [{jid}] tool={pj.get('tool', '?')}: "
                    f"results already communicated to user "
                    f"(original request: \"{orig_q}\")"
                )
            elif jid and jid not in store_jobs:
                lines.append(
                    f"RUNNING JOB [{jid}] tool={pj.get('tool', '?')}: "
                    f"recently launched, executing now"
                )

    if not lines:
        return NO_JOBS

    return "Background jobs:\n" + "\n".join(lines)


@entrypoint(path="chitchat")
async def graph(inputs: dict, *, previous: list | None) -> entrypoint.final[str, list]:
    """Conversational agent with job-aware context.

    State (checkpointed via `previous`): list of message dicts.
    Returns the assistant's text response and saves updated history.
    """
    llm = get_chitchat_llm()
    history = previous or []

    session_id = inputs.get("session_id", "default")
    input_type = inputs.get("type", "user_message")
    client_pending = inputs.get("pending_jobs", None)

    job_context = await _build_job_context(session_id, client_pending)

    if input_type == "job_result":
        job_id = inputs.get("job_id", "unknown")
        original_question = inputs.get("original_question", "")
        human_content = (
            f"[SYSTEM NOTIFICATION] Job {job_id} has completed. "
            f"The user originally asked: \"{original_question}\". "
            f"Look at the conversation history to see what you and the user "
            f"were just discussing. Start with a brief, natural bridge from "
            f"that topic into the results. Do NOT start with 'Got it' or any "
            f"generic opener. Reference the actual conversation instead. "
            f"Then deliver the results in plain conversational English. "
            f"Be specific — name the exact analysis (Q2, Q3, etc.) the user "
            f"requested. No bullet points, no dashes, no lists — just "
            f"flowing sentences."
        )
    else:
        human_content = inputs.get("message", inputs.get("messages", ""))
        if isinstance(human_content, list):
            human_content = human_content[-1].get("content", "") if human_content else ""

    history.append({"role": "user", "content": human_content})

    system_prompt = BASE_SYSTEM_PROMPT.format(job_context=job_context)
    messages = [SystemMessage(content=system_prompt)]
    for turn in history[-20:]:
        role = turn["role"]
        content = turn["content"]
        if role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))

    response = await llm.ainvoke(messages)
    assistant_text = response.content

    history.append({"role": "assistant", "content": assistant_text})

    logger.info(f"[chitchat] session={session_id} type={input_type} response='{assistant_text[:80]}...'")

    return entrypoint.final(value=assistant_text, save=history)
