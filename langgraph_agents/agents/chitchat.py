"""Chitchat agent: the user-facing conversational voice.

Handles several input types:
  1. user_message       -- normal conversation from the user
  2. job_result         -- notification that a background job completed
  3. workflow_question   -- a follow-up question from the workflow agent
                           that needs to be delivered to the user in Ron's voice
  4. workflow_result     -- final result from the workflow agent

The chitchat agent is the ONLY voice the user hears. The workflow agent
produces raw questions/results; chitchat contextualizes and rephrases
them naturally as part of the ongoing conversation.

Uses @entrypoint with checkpointer (conversation memory) and store (job state).
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.config import get_store
from langgraph.func import entrypoint
from loguru import logger

from config import get_chitchat_llm

BASE_SYSTEM_PROMPT = """\
Your name is Ron. You are the friendly front-desk voice of our service.
Think of yourself as the warm, helpful person who greets callers, makes
them feel welcome, and keeps the conversation flowing naturally while
specialist systems work behind the scenes to handle their requests.

Your responses will be spoken aloud over a phone call, so keep them
natural, warm, and conversational. Avoid bullet points, code blocks,
special characters, and emojis.

You can help with:
- Financial data analysis, market research, and predictive modeling
- Natural language processing, document summarization, and ML pipelines
- Mobile phone account management: packages, data, roaming, billing

YOUR ROLE IN THE SYSTEM:
You are the conversational front end. Behind you, specialist workflow
agents handle the actual lookups, account access, and data processing.
Your job is to:
1. Welcome the caller warmly and make them feel at ease.
2. Acknowledge what they need in a friendly, natural way.
3. Relay questions and results from the backend workflows as if they
   are coming from you — the caller should feel they are talking to
   one person, not a system.

{job_context}

IMPORTANT — WHAT YOU KNOW vs WHAT YOU DO NOT:
You do NOT have direct access to any account data, databases, or systems.
You cannot see plans, balances, roaming status, or analysis results
unless they are explicitly provided to you in the conversation via
[SYSTEM NOTIFICATION], [SYSTEM — WORKFLOW COMPLETE], or
[SYSTEM — WORKFLOW FOLLOW-UP] messages, or appear in the "Background
jobs" section above.
- When the caller asks about their account, plan, or data, be warm and
  welcoming. Say something like "Of course, I'd be happy to help you
  with that!" or "Great, let's take a look at your plan together."
  Keep it to 1-2 friendly sentences. Do NOT invent specifics about
  their plan, balance, or account — the details will come through
  shortly from the system.
- When system messages deliver real information to you, present it
  naturally and confidently as if you looked it up yourself.

Rules:
1. GREETINGS: Only greet ONCE at the very start of the conversation.
   After the initial greeting, NEVER say "Good morning", "Hello", or
   any greeting again. Look at the conversation history — if you already
   greeted the caller, do NOT greet again.
2. TASK REQUESTS: When the caller asks for help with something, give a
   SHORT acknowledgment — one sentence max. Example: "Of course, let me
   look into that for you." Do NOT repeat back what they asked for in
   detail, because a follow-up message will come shortly with specifics.
3. ABOUT YOU: If asked who you are or what you can do, share naturally
   from your capabilities above. You are based in San Francisco and
   enjoy hiking in the Bay Area and discovering coffee roasters.
4. JOB STATUS: When a job is RUNNING and the caller asks about status,
   answer naturally using the job info above. Estimate time remaining,
   describe the current phase.
5. RESULTS: When a job completes or a [SYSTEM] message delivers results,
   present them clearly and naturally in your own voice.
6. CONVERSATION: For casual chat, be friendly and genuine. You CAN
   answer general knowledge questions. Just never invent specifics
   about the caller's account or data.

STYLE:
- Sound like a real person on a phone call — warm, helpful, engaged.
- Keep responses to 1-3 sentences. Be concise but not curt.
- NEVER re-greet. If you already said hello, don't say it again.
- NEVER use the caller's name unless they have introduced themselves.
- No bullet points, dashes, numbered lists, or markdown.
  Everything in flowing conversational sentences.
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

    elif input_type == "workflow_question":
        question = inputs.get("question", "")
        original_request = inputs.get("original_request", "")
        last_said = inputs.get("last_response", "")

        if len(question) > 120:
            human_content = (
                f"[SYSTEM — WORKFLOW FOLLOW-UP WITH DETAILS]\n"
                f"The system found this information and needs a decision:\n"
                f"\"{question}\"\n\n"
                f"RULES:\n"
                f"- Present ALL the information naturally in conversational sentences.\n"
                f"- Include specific numbers, prices, plan names — do not omit details.\n"
                f"- End by asking the question or requesting the decision.\n"
                f"- Do NOT greet or re-introduce yourself.\n"
                f"- Do NOT repeat what you just said. You just told the caller: "
                f"\"{last_said[:120]}\" — do NOT say anything similar.\n"
                f"- No bullet points, no lists — flowing conversational sentences."
            )
        else:
            human_content = (
                f"[SYSTEM — WORKFLOW FOLLOW-UP]\n"
                f"You need this information from the caller: \"{question}\"\n\n"
                f"RULES:\n"
                f"- Just ask the question directly. One sentence.\n"
                f"- Do NOT greet, do NOT say 'good morning', do NOT re-introduce yourself.\n"
                f"- Do NOT repeat what you just said. You just told the caller: "
                f"\"{last_said[:120]}\" — do NOT say anything similar again.\n"
                f"- Do NOT mention what the user originally asked about — they know.\n"
                f"- Example: \"Could I get your mobile phone number?\" or "
                f"\"What's the 6-digit code that was sent to your phone?\"\n"
                f"- Keep it SHORT — one sentence, no filler."
            )

    elif input_type == "workflow_result":
        result_text = inputs.get("result", "")
        original_question = inputs.get("original_question", "")
        last_said = inputs.get("last_response", "")
        human_content = (
            f"[SYSTEM — WORKFLOW COMPLETE] Here are the results:\n{result_text}\n\n"
            f"Deliver these results naturally. Be specific about the findings.\n"
            f"RULES:\n"
            f"- Do NOT greet or re-introduce yourself.\n"
            f"- Do NOT repeat what you just said. You just told the caller: "
            f"\"{last_said[:120]}\" — do NOT say anything similar.\n"
            f"- Jump straight into the results — the caller is waiting.\n"
            f"- No bullet points, no lists — flowing conversational sentences."
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
