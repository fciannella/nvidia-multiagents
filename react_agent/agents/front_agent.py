"""Front agent: conversational ReAct agent with task delegation.

Custom @entrypoint that handles four input scenarios:
  1. user_message     — normal conversation, may trigger delegate_task
  2. executor_update  — system-initiated delivery of executor questions/results
  3. answer_forwarded — user answered an executor question, ack only
  4. any of the above with starter_played context for seamless continuation

The LLM has one tool (delegate_task) that writes a delegation request to the
LangGraph Store. The Pipecat-side service reads the Store after each run and
starts the executor if a new delegation is found.

Tool binding is conditional: skipped when the agent just needs to relay
executor updates or acknowledge a forwarded answer (saves an LLM round-trip).

State (checkpointed via `previous`): list of message dicts.
"""

import json
import uuid

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.config import get_store
from langgraph.func import entrypoint
from loguru import logger

from config import get_front_agent_llm

# ---------------------------------------------------------------------------
# Tool definition (schema only — execution is inline in the ReAct loop
# so we can access get_store() directly in the @entrypoint context)
# ---------------------------------------------------------------------------


@tool
def delegate_task(description: str) -> str:
    """Delegate a task to the backend executor system.

    Use this when the user asks for something that requires backend processing:
    - Mobile phone account access (plans, data, roaming, billing) — e.g. "What
      do I have?", "Will roaming work in France?", "Check my plan"
    - Data analysis, financial research, trend detection
    - Machine learning model training or evaluation
    - Any operation that needs tool execution

    Always use this to look up the user's account when they ask what they have
    or whether something will work. Never say you don't have access — delegate.
    Provide a clear, specific description of what the user wants done.
    The task will run in the background and you will receive updates.
    """
    return ""


DELEGATE_TOOLS = [delegate_task]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

BASE_PERSONALITY = """\
Your name is Ron. You are the friendly front-desk voice of our service.
Think of yourself as the warm, helpful person who greets callers, makes
them feel welcome, and keeps the conversation flowing naturally while
specialist systems work behind the scenes to handle their requests.

Your responses will be spoken aloud over a phone call, so keep them
natural, warm, and conversational. Avoid bullet points, code blocks,
special characters, and emojis.

You can help with:
- Mobile phone account management: packages, data, roaming, billing
- Financial data analysis, market research, and predictive modeling
- Machine learning model training and evaluation

HOW DELEGATION WORKS:
When the user asks for something that requires backend processing, use the
delegate_task tool. Provide a clear description. The task runs in the
background. You will receive updates about its progress, questions it
needs answered, and its final results. Present everything naturally in
your own voice — the caller should feel they are talking to one person.

NEVER say "I don't have access to your account" or "I can't see your
account details." You do not have direct access — but you have delegate_task.
When the user asks what they have (roaming, plan, billing, add-ons, etc.),
or whether something will work for them, you MUST delegate so the backend
can look it up (and authenticate if needed). Say "Let me check that for you"
and delegate. Never refuse by claiming you lack access.

When the user is vague ("I need some", "help with my plan", "something for
my trip"), do NOT ask them to specify plan name or phone number first.
Delegate with a broad description (e.g. "Check current plan and what
add-ons or options the user might need") so the backend can authenticate
and then show them their plan; the executor will ask for phone/OTP as needed.

RULES:
1. GREETINGS — When the user greets you or makes small talk, respond
   warmly and ask how you can help. Do NOT ask for phone numbers,
   account details, or delegate anything. Just be friendly. Only greet
   ONCE at the very start; after that, never re-greet. If a filler
   message already greeted the user (check the FILLER ALREADY SPOKEN
   annotation), skip greeting entirely and go to "How can I help?"
2. CAPABILITY QUESTIONS — When the user asks what you can do, what's
   available, or whether you support something ("Can you do X?",
   "What services do you offer?", "Do you do machine learning?"),
   answer from the "You can help with" list above. Describe your
   capabilities in warm conversational sentences. Do NOT delegate or
   start any task — the user is asking for information, not action.
   Only delegate when they explicitly ask you to DO something.
3. DELEGATION — When the user asks about their account (roaming, plan,
   billing, "what do I have", "will X work for me"), ALWAYS delegate.
   Your delegation response must be ONLY a brief acknowledgment — nothing
   else. Good examples: "Let me check that for you.", "One moment.",
   "Checking on that now." Bad examples: "Let me check that for you.
   What is your phone number?" or "One moment while I look into your
   account. Could you tell me which plan you're on?"
   NEVER ask ANY question (phone number, plan name, OTP, preferences,
   country, etc.) in the same turn as delegating. The backend will ask
   the user for anything it needs. Your job is ONLY to acknowledge.
4. RESULTS — When executor results are provided, present them clearly
   and naturally in flowing conversational sentences. Be specific with
   numbers and details.
5. QUESTIONS — When the executor needs information from the user, ask
   in your own warm, natural way. One or two sentences max.
   Do NOT repeat or confirm the user's last message back to them.
   Do NOT use filler. At most a few words of lead-in, then the question.
6. STATUS — If the user asks about a running task, use the active tasks
   info in your context to give a natural status update.
7. CONVERSATION — For casual chat, be friendly and genuine. You CAN
   answer general knowledge questions. When a task is waiting for input,
   you can still respond to small talk, then naturally circle back to
   the question when ready.
8. NEVER invent account data, analysis results, system information,
   time estimates, or percentages. Only state what the executor or
   progress data explicitly provides. If you don't have a number,
   don't make one up.
9. Keep responses to 1-3 sentences. Concise but warm.
10. No bullet points, dashes, numbered lists, or markdown.
    Everything in flowing conversational sentences.
11. NEVER use the caller's name unless they introduced themselves.
12. NO ECHO — Never repeat the user's last message back to them as a
    confirmation (e.g. they said "France for a week" → do NOT say "for
    your week in France" or "I'll set up a plan for France"). Move the
    conversation forward; they already know what they said.
13. NO FILLER — Skip phrases like "I'll take care of that.", "Great—",
    "Sure thing—" when the next line is a question. One brief lead-in
    at most, then the question or the answer.

ABOUT YOU:
You work at a telecom company helping customers with their mobile accounts."""


def _build_system_prompt(
    *,
    active_tasks: list[dict],
    executor_updates: list[dict],
    answer_forwarded: bool,
) -> str:
    sections = [BASE_PERSONALITY]

    if active_tasks:
        lines = []
        for t in active_tasks:
            tid = t.get("task_id", "?")
            desc = t.get("description", "unknown")
            status = t.get("status", "unknown")
            progress = t.get("progress", "")
            line = f"  [{tid}] {desc} — {status}"
            if progress:
                line += f" ({progress})"
            lines.append(line)
        sections.append(
            "ACTIVE BACKGROUND TASKS:\n" + "\n".join(lines) + "\n"
            "If the user asks about progress, use ONLY the data shown above. "
            "Report the step number and what it's doing (e.g. 'We're on step "
            "7 of 12, currently running the regression.'). "
            "NEVER estimate completion time or duration — you don't know how "
            "long each step takes. Just report the step and phase."
        )
    else:
        sections.append(
            "NO ACTIVE BACKGROUND TASKS.\n"
            "There are no tasks currently running or queued. If the user asks "
            "about status of a previous task, say honestly that it has "
            "completed or that there is nothing currently running. "
            "NEVER say a task is 'queued', 'processing', or 'in progress' "
            "unless you see it listed above."
        )

    for update in executor_updates:
        utype = update.get("type", "")
        tid = update.get("task_id", "?")
        if utype == "result":
            sections.append(
                f"[EXECUTOR RESULT — task {tid}]\n"
                f"{update.get('result', '')}\n\n"
                "Deliver these results naturally. Be specific about numbers "
                "and details. Jump straight in — the caller is waiting."
            )
        elif utype == "question":
            question = update.get("question", "")
            sections.append(
                f"[EXECUTOR NEEDS INPUT — task {tid}]\n"
                f"The backend needs the user to answer: {question}\n\n"
                "Ask this question naturally. One or two sentences. "
                "No filler. No echo of the user's words."
            )
            if len(question) > 120:
                sections.append(
                    "Present the information and end with the question. "
                    "No echo of the user's last message; no filler."
                )
            else:
                sections.append(
                    "No echo of the user's last message; no filler. "
                    "If they respond with chitchat later, acknowledge and ask again."
                )
        elif utype == "progress":
            sections.append(
                f"[EXECUTOR PROGRESS — task {tid}]\n"
                f"Status: {update.get('message', 'working...')}"
            )
        elif utype == "error":
            sections.append(
                f"[EXECUTOR ERROR — task {tid}]\n"
                f"Error: {update.get('error', 'unknown')}\n\n"
                "Let the user know something went wrong. Apologize briefly."
            )

    if answer_forwarded:
        sections.append(
            "NOTE — ANSWER FORWARDED:\n"
            "The user just gave information the backend was waiting for. "
            "It has been passed along and the system is continuing. Give a "
            "brief, warm acknowledgment so they know you got it (e.g. 'Got it, "
            "checking that now…' or 'Thanks, one moment…'). One sentence. Then "
            "you can add a short follow-up if natural (e.g. 'I'll have that for "
            "you in a second.'). Keep it conversational."
        )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# LLM invocation (no @task — front agent has no interrupts)
# ---------------------------------------------------------------------------


async def _call_llm(messages: list[dict], *, bind_tools: bool = False) -> dict:
    llm = get_front_agent_llm()
    if bind_tools:
        llm = llm.bind_tools(DELEGATE_TOOLS)

    lc_messages: list[BaseMessage] = []
    for m in messages:
        role = m["role"]
        content = m.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            tc = m.get("tool_calls")
            lc_messages.append(AIMessage(content=content, tool_calls=tc or []))
        elif role == "tool":
            lc_messages.append(ToolMessage(
                content=content,
                tool_call_id=m["tool_call_id"],
                name=m.get("name", ""),
            ))

    response = await llm.ainvoke(lc_messages)

    return {
        "role": "assistant",
        "content": response.content or "",
        "tool_calls": [
            {"id": tc["id"], "name": tc["name"], "args": tc.get("args", {})}
            for tc in (response.tool_calls or [])
        ],
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


@entrypoint(path="front_agent")
async def graph(
    inputs: dict, *, previous: list[dict] | None = None
) -> entrypoint.final[str, list[dict]]:
    """Front agent: conversational ReAct with one delegation tool.

    Inputs:
        message: str            — the user's utterance (or empty for system-initiated)
        session_id: str         — session identifier
        type: str               — "user_message" (default) or "executor_update"
        starter_played: str     — text of the starter audio already playing
        active_tasks: list      — currently running executor tasks
        executor_updates: list  — pending results/questions from executor
        answer_forwarded: bool  — whether the user's answer was forwarded to executor
    """
    session_id = inputs.get("session_id", "default")
    message = inputs.get("message", "")
    input_type = inputs.get("type", "user_message")
    starter_played = inputs.get("starter_played", "")
    active_tasks = inputs.get("active_tasks", [])
    executor_updates = inputs.get("executor_updates", [])
    answer_forwarded = inputs.get("answer_forwarded", False)

    messages = list(previous or [])

    # Keep full context when delivering executor questions so we can merge:
    # if we already asked for phone/OTP in the previous turn, don't ask again.

    # --- Build dynamic system prompt (rebuilt every invocation) ---
    system_prompt = _build_system_prompt(
        active_tasks=active_tasks,
        executor_updates=executor_updates,
        answer_forwarded=answer_forwarded,
    )
    if messages and messages[0].get("role") == "system":
        messages[0] = {"role": "system", "content": system_prompt}
    else:
        messages.insert(0, {"role": "system", "content": system_prompt})

    # --- Add user message ---
    # When a starter/filler was spoken aloud, inject it as an assistant
    # message in the history so the model sees it as its own words, then
    # annotate the user turn to prevent overlap.
    if message:
        if starter_played:
            messages.append({"role": "assistant", "content": starter_played})
            user_content = (
                f'[CALLER SAID: "{message}"]\n\n'
                f"You ALREADY responded with: \"{starter_played}\"\n"
                f"The caller heard that. It is DONE. Do not repeat it.\n\n"
                f"Now continue naturally. Say ONLY what comes NEXT.\n"
                f"Do NOT re-greet, re-acknowledge, re-answer 'how are you', "
                f"or echo any sentiment from your previous line.\n"
                f"If the caller just greeted you, say 'How can I help you today?' "
                f"and nothing else."
            )
        else:
            user_content = message
        messages.append({"role": "user", "content": user_content})
        logger.info(f"[front] User ({input_type}): {message[:100]}")
    elif executor_updates:
        # System-initiated delivery: no user message, but we have executor question/result.
        update_types = [u.get("type") for u in executor_updates]
        if "question" in update_types:
            content = (
                "[System: The backend needs an answer from the user (see instructions). "
                "Ask the question now in your own warm, natural voice. One or two sentences.]"
            )
        else:
            content = (
                "[System: The instructions above contain a result from the backend. "
                "Deliver it to the user in your own warm, natural voice. "
                "One or two short sentences. No robotic phrasing.]"
            )
        messages.append({"role": "user", "content": content})
        logger.info(f"[front] Executor update delivery ({len(executor_updates)} update(s)) types={update_types}")
    else:
        logger.warning("[front] No message and no executor updates")
        return entrypoint.final(value="", save=messages)

    # --- Decide whether to bind tools ---
    # Skip tools when we're just relaying executor info or acknowledging
    should_bind = (
        input_type == "user_message"
        and not executor_updates
        and not answer_forwarded
    )

    response = await _call_llm(messages, bind_tools=should_bind)
    messages.append(response)

    # --- ReAct loop (max 2 iterations: delegate + respond) ---
    iteration = 0
    while response.get("tool_calls") and iteration < 2:
        iteration += 1

        for tc in response["tool_calls"]:
            if tc["name"] == "delegate_task":
                store = get_store()
                task_id = f"task-{uuid.uuid4().hex[:8]}"
                desc = tc["args"].get("description", "")
                await store.aput(
                    ("delegations", session_id),
                    task_id,
                    {
                        "description": desc,
                        "status": "requested",
                        "task_id": task_id,
                    },
                )
                logger.info(f"[front] Delegated: {task_id} — {desc[:80]}")
                tool_result = {
                    "role": "tool",
                    "content": json.dumps({
                        "task_id": task_id,
                        "status": "queued",
                        "message": "Task queued for the backend executor.",
                    }),
                    "tool_call_id": tc["id"],
                    "name": "delegate_task",
                }
            else:
                logger.warning(f"[front] Unknown tool: {tc['name']}")
                tool_result = {
                    "role": "tool",
                    "content": json.dumps({"error": f"Unknown tool: {tc['name']}"}),
                    "tool_call_id": tc["id"],
                    "name": tc["name"],
                }
            messages.append(tool_result)

        response = await _call_llm(messages, bind_tools=False)
        messages.append(response)

    final_text = response.get("content", "")

    # --- Trim history to prevent unbounded growth ---
    system_msg = messages[0] if messages and messages[0]["role"] == "system" else None
    non_system = [m for m in messages if m["role"] != "system"]
    if len(non_system) > 40:
        messages = ([system_msg] if system_msg else []) + non_system[-40:]

    logger.info(f"[front] Response ({input_type}): {final_text[:100]}")
    return entrypoint.final(value=final_text, save=messages)
