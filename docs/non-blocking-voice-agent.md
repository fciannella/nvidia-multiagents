# Non-Blocking Voice Agent: Dual-Thread Architecture

How to make a LangGraph agent non-blocking so users can chat while long-running
tools execute in the background.

---

## Overview

A standard voice agent blocks while waiting for tool results. If a tool takes
60 seconds (e.g. model training), the user stares at silence for a full minute.

The **dual-thread architecture** solves this by running two LangGraph threads
in parallel:

| Thread | Role | Has Tools | When Active |
|---|---|---|---|
| **Primary** | Full agent — handles real work | Yes | Always (default) |
| **Secondary** | Chitchat companion — keeps the user company | No | Only when primary is busy |

The two threads share state through the **LangGraph Store** (cross-thread
key-value memory), so the chitchat agent can report live progress on whatever
the primary agent is doing.

```
User speaks
    │
    ├── Primary idle? ──► Route to Primary Thread ──► Stream response
    │
    └── Primary busy? ──► Route to Secondary Thread ──► Stream chitchat
                              │
                              └── Reads live progress from Store
                                  (written by Primary's tools)
```

When the primary finishes, its result is **merged** into the conversation
through the secondary thread, which delivers it naturally ("Great news — the
training just finished with 91.8% accuracy!"), then the secondary is torn down.

---

## Part 1: Server-Side — Adapting Your LangGraph Agent

You need three changes to your existing LangGraph agent:

1. Add a `mode` field to your state (to switch between normal and chitchat)
2. Make your agent node dual-mode (chitchat reads from the Store)
3. Make your tools write progress to the Store

### 1.1 Add `mode` to your agent state

Your existing state probably looks like this:

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

Add two fields:

```python
class AgentState(dict):
    messages: Annotated[list[AnyMessage], add_messages]
    mode: str                  # "normal" or "chitchat"
    main_thread_status: str    # fallback status text from the client
```

The client controls `mode` by including it in the input payload. Your graph
doesn't need to route on it — the agent node handles it internally.

### 1.2 Make your agent node dual-mode

The core idea: one agent node, two personalities. In `normal` mode, it behaves
exactly as before. In `chitchat` mode, it has no tools and reads live progress
from the LangGraph Store.

**Before** (your existing agent node):

```python
def agent_node(state: dict) -> dict:
    messages = state["messages"]
    system = SystemMessage(content=YOUR_SYSTEM_PROMPT)
    llm = ChatOpenAI(...).bind_tools(YOUR_TOOLS)
    response = llm.invoke([system] + messages)
    return {"messages": [response]}
```

**After** (dual-mode with Store access):

```python
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore


def agent_node(state: dict, config: RunnableConfig, *, store: BaseStore) -> dict:
    mode = state.get("mode", "normal")
    messages = state.get("messages", [])

    if mode == "chitchat":
        # Read live progress from the cross-thread Store
        store_context = _build_store_context(store, config)
        system = CHITCHAT_SYSTEM_PROMPT.format(status_context=store_context)
        llm = ChatOpenAI(...)  # no tools bound
    else:
        system = NORMAL_SYSTEM_PROMPT
        llm = ChatOpenAI(...).bind_tools(YOUR_TOOLS)

    response = llm.invoke([SystemMessage(content=system)] + messages)

    # When the agent decides to call a tool, write "starting" to the Store
    if mode != "chitchat" and isinstance(response, AIMessage) and response.tool_calls:
        ns = _session_ns(config)
        tool_names = [tc["name"] for tc in response.tool_calls]
        store.put(ns, "task_progress", {
            "tool": ", ".join(tool_names),
            "status": "starting",
            "phase": "Tool call initiated",
            "progress": "0%",
        })

    return {"messages": [response]}
```

Key additions:
- **`config: RunnableConfig`** — injected by LangGraph, carries the `session_id`
  from the client
- **`store: BaseStore`** — injected by LangGraph, the cross-thread key-value store
- **`_session_ns(config)`** — extracts the store namespace from the session ID
- **`_build_store_context(store, config)`** — reads progress and formats it for
  the chitchat prompt

Here is the helper that extracts the session namespace:

```python
def _session_ns(config: RunnableConfig) -> tuple:
    sid = config.get("configurable", {}).get("session_id", "default")
    return ("session", sid)
```

And the helper that reads progress from the Store and formats it as context for
the chitchat system prompt:

```python
def _build_store_context(store: BaseStore, config: RunnableConfig) -> str:
    ns = _session_ns(config)
    item = store.get(ns, "task_progress")

    if not item:
        return ""

    p = item.value
    tool_name = p.get("tool", "unknown task")
    status = p.get("status", "unknown")

    if status == "running":
        lines = ["LIVE TASK PROGRESS (from shared memory):"]
        lines.append(f"  Tool: {tool_name}")
        if p.get("phase"):
            lines.append(f"  Current phase: {p['phase']}")
        if p.get("progress"):
            lines.append(f"  Progress: {p['progress']}")
        if p.get("epoch"):
            lines.append(f"  Epoch: {p['epoch']}")
        if p.get("loss"):
            lines.append(f"  Current loss: {p['loss']}")
        # ... add any other fields your tools write
        return "\n".join(lines)

    if status == "complete":
        return f"TASK JUST COMPLETED: {tool_name}\nResult: {p.get('result_summary', 'done')}"

    if status == "starting":
        return f"TASK STARTING: {tool_name} (just initiated, no progress yet)"

    return ""
```

### 1.3 Chitchat system prompt

The chitchat agent needs strict guardrails — it must not call tools or start
new work. The `{status_context}` placeholder is filled with the live progress
from the Store:

```python
CHITCHAT_SYSTEM_PROMPT = """\
You are a friendly AI assistant. Right now your main processing thread is busy \
working on a task for the user.

You are on a SECONDARY conversation thread. Your ONLY job is to keep the user \
company with light conversation while the work completes.

STRICT RULES:
- Do NOT call any tools. You have no tools available.
- Do NOT offer to start new tasks or analyses.
- If the user asks you to do something, politely explain you're currently \
  busy with the ongoing task and will handle it once that's done.
- You CAN and SHOULD tell the user about the status of the ongoing task \
  when they ask. Use the live progress information below.
- Keep responses short and friendly (1-3 sentences).

{status_context}
"""
```

### 1.4 Make your tools write progress to the Store

This is where the magic happens. Long-running tools use LangGraph's
`InjectedStore` to write live progress that the chitchat agent can read.

**Before** (a standard tool):

```python
@tool
async def train_model(model_name: str, dataset: str) -> str:
    """Train an ML model. Takes 40-60 seconds."""
    for epoch in range(1, 11):
        await asyncio.sleep(5)
        logger.info(f"Epoch {epoch}/10")
    return json.dumps({"accuracy": 0.92, "status": "complete"})
```

**After** (tool with Store writes):

```python
from typing import Annotated
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore


@tool
async def train_model(
    model_name: str,
    dataset: str,
    *,
    config: RunnableConfig,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """Train an ML model. Takes 40-60 seconds."""

    ns = _session_ns(config)  # ("session", "<uuid>")

    # Write initial status
    await store.aput(ns, "task_progress", {
        "tool": "train_model",
        "status": "running",
        "model_name": model_name,
        "dataset": dataset,
        "phase": "Initializing",
        "epoch": "0/10",
        "progress": "0%",
    })

    for epoch in range(1, 11):
        await asyncio.sleep(5)
        loss = round(2.5 - (epoch * 0.2), 3)

        # Update progress after each epoch
        await store.aput(ns, "task_progress", {
            "tool": "train_model",
            "status": "running",
            "model_name": model_name,
            "dataset": dataset,
            "phase": "Training",
            "epoch": f"{epoch}/10",
            "loss": str(loss),
            "progress": f"{epoch * 10}%",
        })

    # Write completion
    await store.aput(ns, "task_progress", {
        "tool": "train_model",
        "status": "complete",
        "result_summary": "Training complete — accuracy 92%, loss 0.34",
    })

    return json.dumps({"accuracy": 0.92, "status": "complete"})
```

Key points about the tool signature:

- **`config: RunnableConfig`** — keyword-only (after `*`), automatically
  injected by LangGraph, hidden from the LLM's tool schema
- **`store: Annotated[BaseStore, InjectedStore()]`** — keyword-only, injected
  by the `ToolNode`, hidden from the LLM's tool schema
- The LLM only sees `model_name` and `dataset` as parameters
- Use `await store.aput(...)` (async) inside async tools; use `store.put(...)`
  (sync) in sync graph nodes

### 1.5 Normal-mode system prompt

The primary agent should always include a spoken message when calling a tool,
so the user hears something before the long silence:

```python
NORMAL_SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to tools.

IMPORTANT: When you decide to call a tool, you MUST ALWAYS include a brief spoken
message in your response alongside the tool call. For example:
"Sure, let me run that analysis for you — this might take a little while."
NEVER call a tool with empty text content. The user is listening via voice and
needs to hear that something is happening.
"""
```

### 1.6 Graph structure (unchanged)

The graph itself stays the same — the standard ReAct loop. No new nodes or
edges needed. The mode switching and Store access all happen inside the existing
`agent_node` and `ToolNode`:

```python
tool_node = ToolNode(YOUR_TOOLS)  # ToolNode auto-injects store into tools

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

graph = builder.compile()
```

### 1.7 LangGraph server configuration

Run the dev server with multiple workers so the primary and secondary threads
can execute concurrently:

```bash
langgraph dev --n-jobs-per-worker 10 --no-browser
```

The default of 1 worker will block — the secondary thread will wait for the
primary to finish, defeating the purpose.

---

## Part 2: Client-Side — The Voice Frontend

The client is a Pipecat `FrameProcessor` called `GenericVoiceFrontend` that
sits in the audio pipeline between STT and TTS. It manages two LangGraph
threads and routes user utterances to the right one.

### 2.1 Architecture

```
Mic → VAD → STT → TranscriptionCapture → UserTurnProcessor → GenericVoiceFrontend → TTS → Speaker
                                                                  │
                                                        ┌────────┴────────┐
                                                        │                 │
                                                   Main Thread      Secondary Thread
                                                  (LangGraph)        (LangGraph)
                                                        │                 │
                                                        └────────┬────────┘
                                                                 │
                                                          LangGraph Store
                                                       (shared via session_id)
```

### 2.2 Core state

```python
class GenericVoiceFrontend(FrameProcessor):
    def __init__(self, *, server_url, graph_name, ui_queue, tts, **kwargs):
        # LangGraph SDK client
        self._client = get_client(url=server_url)
        self._graph_name = graph_name

        # Thread IDs
        self._main_thread_id = None       # created once at init
        self._secondary_thread_id = None  # created on demand, torn down after merge

        # State flags
        self._main_busy = False           # True when main is running a long tool
        self._is_streaming = False        # True when actively pushing frames
        self._cancelled = False           # True when user interrupted
        self._merge_in_progress = False   # True during result merge
        self._active_stream_thread = None # which thread can push to pipeline

        # Buffers
        self._main_result_buffer = None   # holds completed main response
        self._pending_utterance = None    # buffered speech during merge

        # Cross-thread store key
        self._session_id = str(uuid.uuid4())
```

### 2.3 The routing decision

When the user finishes speaking, `process_frame` receives a `TranscriptionFrame`
and routes it:

```
TranscriptionFrame arrives
    │
    ├── merge_in_progress? ──► Buffer utterance, don't interrupt
    │
    ├── main_busy == False ──► Send to Main Thread
    │       │
    │       ├── Response within 8 seconds? ──► Stream to TTS, done
    │       │
    │       └── Timeout (8s)? ──► Switch to dual-thread:
    │               • Mark main_busy = True
    │               • Let main continue in background
    │               • Future utterances go to Secondary
    │
    └── main_busy == True ──► Send to Secondary Thread ──► Stream chitchat
```

### 2.4 Passing the session_id

Every call to `client.runs.stream()` includes the `session_id` in the config
so both threads (and all tools) share the same Store namespace:

```python
async for chunk in self._client.runs.stream(
    thread_id,
    assistant_id=self._graph_name,
    input=payload,
    config={"configurable": {"session_id": self._session_id}},
    stream_mode="messages",
    multitask_strategy="interrupt",
):
    ...
```

### 2.5 The 8-second timeout

The timeout is the key heuristic that triggers dual-thread mode. When a user
utterance is sent to the primary thread:

1. A streaming task is created: `asyncio.create_task(self._stream_to_tts(...))`
2. We wait up to 8 seconds: `await asyncio.wait_for(asyncio.shield(task), timeout=8.0)`
3. If the response comes in time, we stream it normally (single-thread mode)
4. If it times out, we:
   - Mark `main_busy = True`
   - Close the partial response frame
   - Set `_active_stream_thread = None` (stops main from pushing more frames)
   - Let the main task continue in the background
   - Push a "Working on task..." status to the UI
   - Create a background watcher: `self._wait_for_main_response(task)`

The shielded task continues executing even after the timeout — the main thread
keeps running its tools, writing progress to the Store. It just no longer pushes
audio/text to the pipeline.

### 2.6 The merge

When the background watcher sees the main thread complete, it triggers a
**merge** — delivering the result through the secondary thread so the transition
feels natural:

```python
async def _do_merge(self):
    result_text = self._main_result_buffer

    # Send the result to the secondary thread as a system message
    relay_msg = (
        f"[SYSTEM] The background task has just completed. "
        f"Here is the result:\n\n"
        f'"{result_text}"\n\n'
        f"Deliver this result to the user naturally and conversationally."
    )

    self._active_stream_thread = self._secondary_thread_id
    response = await self._stream_to_tts(
        self._secondary_thread_id,
        {"messages": [{"role": "user", "content": relay_msg}], "mode": "chitchat"},
    )

    # Clean up — back to single-thread mode
    await self._teardown_secondary()
    self._main_busy = False
```

The secondary thread rephrases the raw result into natural speech, e.g.:
> "Perfect timing — the training just finished! Your model hit 91.8% accuracy
> over 10 epochs, which looks really solid."

### 2.7 Merge protection

If the user speaks during an active merge, we **buffer** the utterance instead
of interrupting. This prevents the merge result from being lost:

```python
# In process_frame, TranscriptionFrame handler:
if self._merge_in_progress:
    self._pending_utterance = text
    return  # don't cancel, don't push InterruptionFrame

# In _do_merge finally block:
pending = self._pending_utterance
self._pending_utterance = None
if pending:
    self._current_task = asyncio.create_task(
        self._handle_user_utterance(pending)
    )
```

### 2.8 Active stream gating

Only the "active" thread pushes frames and SSE events to the pipeline. This
prevents the background main stream from leaking partial responses:

```python
is_active = lambda: thread_id == self._active_stream_thread

# Inside the streaming loop:
if content and len(content) > prev_len:
    delta = content[prev_len:]
    if is_active():               # only push if this is the active stream
        await self.push_frame(TextFrame(text=delta))
    prev_len = len(content)
    full_response = content       # always capture, even if not active
```

The `full_response` is always captured regardless of `is_active()`. This ensures
the main thread's result is available for the merge even though its frames were
gated.

---

## Part 3: Summary of Files

| File | Role |
|---|---|
| `generic_agent/graph.py` | LangGraph graph — agent node with dual-mode + Store reads/writes |
| `generic_agent/tools.py` | Tools — long-running tools write live progress via `InjectedStore` |
| `generic_voice_frontend.py` | Pipecat FrameProcessor — manages threads, routing, merge, buffering |
| `pipecat_voice_bot_generic.py` | Pipeline assembly + web server + HTML/JS frontend |

---

## Part 4: Data Flow Diagram

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         LangGraph Server                               │
  │                                                                        │
  │  ┌──────────────┐                    ┌──────────────┐                  │
  │  │ Main Thread   │                    │ Sec. Thread   │                 │
  │  │ mode="normal" │                    │ mode="chitchat"│                │
  │  │               │                    │               │                 │
  │  │ agent_node ◄──┤                    │ agent_node ◄──┤                 │
  │  │   │           │                    │   │           │                 │
  │  │   ▼           │                    │   ▼           │                 │
  │  │ ToolNode      │                    │ (no tools)    │                 │
  │  │   │           │                    │               │                 │
  │  │   ▼           │                    │ Reads from ───┤──┐              │
  │  │ train_model   │                    └──────────────┘  │              │
  │  │   │           │                                      │              │
  │  │   │ Writes to ┤──┐                                   │              │
  │  └───┼───────────┘  │    ┌─────────────────────┐        │              │
  │      │              └───►│   LangGraph Store    │◄───────┘              │
  │      │                   │ ns=("session", sid)  │                       │
  │      │                   │ key="task_progress"  │                       │
  │      │                   │                      │                       │
  │      │                   │ {tool, status, phase, │                      │
  │      │                   │  epoch, loss, ...}    │                      │
  │      │                   └─────────────────────┘                       │
  └──────┼─────────────────────────────────────────────────────────────────┘
         │
         │  config.configurable.session_id = "<uuid>"
         │  (passed by client on every runs.stream() call)
         │
  ┌──────┼─────────────────────────────────────────────────────────────────┐
  │      │              GenericVoiceFrontend (Pipecat)                      │
  │      │                                                                  │
  │  Generates session_id ──► passes to both threads via config             │
  │  Routes user speech  ──► main (if idle) or secondary (if busy)         │
  │  Monitors main bg    ──► triggers merge when done                       │
  │  Buffers during merge ──► prevents interruption loss                    │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## Part 5: Checklist — Adding This to Your Agent

### Server-side (your LangGraph graph)

- [ ] Add `mode` and `main_thread_status` fields to your `AgentState`
- [ ] Add `config: RunnableConfig` and `store: BaseStore` parameters to your
      agent node function
- [ ] Add chitchat-mode branch: no tools, reads from Store, uses chitchat prompt
- [ ] Add `store.put(...)` in agent node when tool calls are detected (writes
      "starting" status)
- [ ] Add `InjectedStore` + `RunnableConfig` to long-running tools (keyword-only
      parameters after `*`)
- [ ] Add `await store.aput(...)` calls at meaningful progress points in each
      long-running tool
- [ ] Write a chitchat system prompt with `{status_context}` placeholder
- [ ] Add spoken-message instruction to normal system prompt (so the agent
      announces tool calls)
- [ ] Run with `--n-jobs-per-worker 10` (or higher) for concurrent execution

### Client-side (your voice frontend)

- [ ] Generate a stable `session_id` per user session
- [ ] Pass `config={"configurable": {"session_id": session_id}}` in every
      `client.runs.stream()` call
- [ ] Implement the 8-second timeout heuristic to detect long-running tasks
- [ ] Create the secondary thread on demand, tear it down after merge
- [ ] Gate stream output with `_active_stream_thread` so background streams
      don't leak frames
- [ ] Implement merge: relay main result through secondary, then return to
      single-thread mode
- [ ] Buffer user utterances during merge (don't interrupt result delivery)
- [ ] Send `mode` field in the input payload: `"normal"` for primary,
      `"chitchat"` for secondary
- [ ] (Optional) Send UI events for agent badge, status badge, streaming tokens

---

## Appendix: Store Key Schema

All tools write to the same key (`"task_progress"`) under the session namespace.
Each write overwrites the previous value, so the Store always reflects the
latest state.

**Namespace:** `("session", "<session-uuid>")`

**Key:** `"task_progress"`

**Value schema:**

| Field | Type | Description |
|---|---|---|
| `tool` | string | Name of the running tool |
| `status` | string | One of: `"starting"`, `"running"`, `"complete"` |
| `phase` | string | Current phase (e.g. "Computing statistics", "Training") |
| `progress` | string | Percentage (e.g. "60%") |
| `epoch` | string | For training: "6/10" |
| `loss` | string | For training: current loss value |
| `step` | string | For multi-phase: "4/7" |
| `dataset` | string | Dataset being processed |
| `model_name` | string | Model being trained |
| `focus` | string | Analysis focus area |
| `result_summary` | string | Brief summary (only when status="complete") |

Not all fields are present in every write — tools include only the fields
relevant to their current state.
