# Interruption Handling (Barge-In)

This document explains how user interruptions (barge-in) work across the
two Pipecat voice pipelines in this project: the **direct LLM pipeline**
(`pipecat_voice_bot_qwen3.py`) and the **LangGraph agent pipeline**
(`pipecat_voice_bot_langgraph.py`). Both share the same TTS service
(`qwen3_tts.py`) and the same core Pipecat interruption machinery; they
differ in what sits between STT and TTS.

---

## Pipeline Architecture

Both pipelines follow the same high-level chain:

```
Mic → Transport.input → STT → [UserAggregator] → LLM / Agent → TTS → Transport.output
```

The **UserAggregator** is backed by a `SileroVADAnalyzer` which
continuously classifies incoming audio as speech or silence.

---

## 1. Trigger: How an Interruption Starts

### Voice Activity Detection (VAD)

Pipecat's `SileroVADAnalyzer` runs on every incoming audio frame. When it
detects the user has started speaking, it fires a
`UserStartedSpeakingFrame`. Pipecat's aggregation layer then evaluates a
**user turn start strategy** to decide whether this constitutes a real
interruption or background noise.

### Turn Start Strategies

| Strategy | When it fires | Used by |
|---|---|---|
| **VAD only** (default) | VAD detects speech onset | LangGraph pipeline |
| **HybridUserTurnStartStrategy** | VAD when bot is silent; requires N transcribed words when bot is speaking | Available, imported in Qwen3 pipeline |

The **Hybrid strategy** (`hybrid_turn_strategy.py`) prevents false
barge-ins caused by speaker echo bleeding back into the mic:

- **Bot silent**: VAD alone triggers the turn start (instant).
- **Bot speaking**: The strategy waits until the STT transcription
  contains at least `min_words` (default 2) real words before triggering.
  This avoids echo-triggered interruptions while still allowing genuine
  barge-in.

```python
# hybrid_turn_strategy.py (simplified)
if isinstance(frame, VADUserStartedSpeakingFrame):
    if not self._bot_speaking:
        await self.trigger_user_turn_started()

elif isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
    min_words = self._min_words if self._bot_speaking else 1
    if len(frame.text.split()) >= min_words:
        await self.trigger_user_turn_started()
```

### The Interruption Frame

When Pipecat decides a user turn has started while audio is playing, the
**PipelineTask** generates a `StartInterruptionFrame` (which Pipecat
dispatches internally) and an `InterruptionFrame` that flows through the
pipeline. This is the signal that all downstream processors must stop
what they are doing.

---

## 2. TTS Layer: Qwen3TTSService

The TTS service (`qwen3_tts.py`) uses a **chained sentence streaming**
approach. Understanding this is critical to understanding how
interruptions cut off audio.

### Normal Flow

1. `LLMFullResponseStartFrame` → reset state, clear buffers.
2. `TextFrame` tokens arrive → buffer text, detect sentence boundaries
   (`.`, `!`, `?`, or `,` after 40+ chars).
3. Each complete sentence is dispatched as a chained `asyncio.Task` that
   calls the TTS HTTP API. Tasks are chained so sentences play in order.
4. `LLMFullResponseEndFrame` → flush any remaining buffer, wait for all
   TTS tasks to finish, then push `TTSStartedFrame` / `TTSStoppedFrame`.

### Interruption Flow

When an `InterruptionFrame` arrives:

```python
# qwen3_tts.py
async def process_frame(self, frame, direction):
    if isinstance(frame, InterruptionFrame):
        self._interrupted = True                      # (a)
        if self._tts_chain and not self._tts_chain.done():
            self._tts_chain.cancel()                  # (b)
        self._buffer = ""                             # (c)
        await self.push_frame(frame, direction)       # (d)
```

**(a)** Sets `_interrupted = True`. Every TTS HTTP streaming loop checks
this flag on every chunk and exits early if set:

```python
async for chunk in resp.aiter_bytes():
    if self._interrupted:
        return  # stop reading audio from TTS server
```

**(b)** Cancels the entire TTS task chain. Because sentences are chained
(`_chained_tts` awaits the previous task), cancelling the current chain
task also prevents all queued-but-not-started sentences from running.

**(c)** Clears the text buffer so no partially accumulated text gets
synthesized.

**(d)** Pushes the `InterruptionFrame` downstream so the transport stops
sending audio to the WebRTC peer.

The net effect: within one event-loop tick of receiving the
`InterruptionFrame`, all in-flight TTS HTTP streams are abandoned, queued
sentences are discarded, and the audio output goes silent.

### Voice Capture at Enqueue Time

To prevent voice "bleeding" in the multi-agent pipeline (where different
agents use different voices), `_enqueue_sentence` captures `self._voice`
at enqueue time rather than at synthesis time:

```python
def _enqueue_sentence(self, text):
    captured_voice = self._voice
    self._tts_chain = asyncio.create_task(
        self._chained_tts(self._tts_chain, text, label, voice=captured_voice)
    )
```

This is relevant to interruptions because a voice switch mid-stream
(e.g., switching from agent A to agent B after an interruption) must not
affect sentences already queued for agent A.

---

## 3. LangGraph Agent Service

The `LangGraphAgentService` (`langgraph_agent_service.py`) replaces the
standard LLM in the LangGraph pipeline. It manages a three-state machine:

| State | Meaning |
|---|---|
| `IDLE` | Normal chitchat + router |
| `WORKFLOW_ACTIVE` | Background workflow running |
| `AWAITING_INPUT` | Workflow paused at `interrupt()`, waiting for user |

### How Interruptions Affect Agent State

```python
# langgraph_agent_service.py
async def process_frame(self, frame, direction):
    ...
    elif isinstance(frame, StartInterruptionFrame):
        self._cancel_current()
        await self.push_frame(frame, direction)
```

`_cancel_current()` does:

```python
def _cancel_current(self):
    self._cancelled = True
    for task in (self._llm_task, self._router_task):
        if task and not task.done():
            task.cancel()
    self._llm_task = None
    self._router_task = None
    self._generating = False
```

This cancels:
- The **chitchat streaming task** (which is reading tokens from LangGraph
  and pushing `TextFrame`s).
- The **router classification task** (which is deciding whether to
  launch a workflow).

It does **not** cancel `_workflow_task` — a running workflow keeps
executing in the background even through an interruption. This is by
design: workflows represent long-running operations (API calls, tool
executions) that should complete even if the user interrupts the
conversational response.

### Cancellation Flag Propagation

The `_cancelled` flag is checked at multiple points during chitchat
streaming:

```python
async def _stream_chitchat(self, message, ...):
    ...
    async for chunk in self._client.runs.stream(...):
        if self._cancelled:
            interrupted = True
            break  # stop reading LLM tokens immediately
        ...

    if not interrupted:
        await self.push_frame(LLMFullResponseEndFrame())
    return full_response
```

When cancelled:
- The LLM stream is abandoned (no more `TextFrame`s pushed).
- `LLMFullResponseEndFrame` is **not** pushed, so the TTS service
  does not attempt to flush/wait.
- The `InterruptionFrame` arriving at the TTS handles cleanup there.

### User Speaking Tracking

The service also tracks `_user_speaking`:

```python
elif isinstance(frame, UserStartedSpeakingFrame):
    self._user_speaking = True
    await self.push_frame(frame, direction)

elif isinstance(frame, UserStoppedSpeakingFrame):
    self._user_speaking = False
    await self.push_frame(frame, direction)
```

This is used to **defer** workflow result/question delivery. If a
workflow completes while the user is speaking, the result is stored in
`_pending_result` or `_pending_question` and delivered on the next user
turn instead of immediately. This prevents the bot from talking over the
user with workflow results.

---

## 4. Direct LLM Pipeline (Qwen3 Bot)

In the simpler `pipecat_voice_bot_qwen3.py` pipeline, interruptions are
handled entirely by Pipecat's built-in mechanisms:

1. VAD detects user speech → `PipelineTask` fires
   `StartInterruptionFrame`.
2. The `InterruptionFrame` propagates through the pipeline.
3. Pipecat's `OpenAILLMService` stops generating tokens.
4. `Qwen3TTSService` cancels the TTS chain and stops audio (same as
   described above).
5. The transport stops sending audio packets over WebRTC.

No custom interruption logic is needed because Pipecat's frame
processors already respond correctly to `InterruptionFrame` out of the
box.

---

## 5. Transport Layer: WebRTC Audio Cutoff

The `SmallWebRTCTransport` handles the final step. When the
`InterruptionFrame` reaches `transport.output()`:

1. The output transport stops pulling audio frames from the pipeline.
2. The WebRTC audio track effectively goes silent.
3. The browser's `<audio>` element stops producing sound.

On the input side, the transport continues capturing the user's mic
audio and feeding it to the STT, so the user's barge-in speech is
transcribed normally.

---

## 6. End-to-End Interruption Timeline

Here is the typical sequence for a barge-in event:

```
t=0ms    User starts speaking (mic audio arrives at VAD)
t≈50ms   SileroVAD detects voice activity
t≈50ms   PipelineTask fires InterruptionFrame
t≈50ms   LangGraphAgentService._cancel_current():
           - Sets _cancelled = True
           - Cancels _llm_task and _router_task
t≈50ms   InterruptionFrame reaches Qwen3TTSService:
           - Sets _interrupted = True
           - Cancels _tts_chain
           - Clears text buffer
t≈50ms   InterruptionFrame reaches transport.output():
           - Audio output stops
t≈50ms   User hears silence (within 1-2 audio frames ≈ 20-40ms)
t≈200ms  STT begins producing interim transcriptions of user speech
t≈1-3s   STT produces final transcription
t≈1-3s   UserStoppedSpeaking triggers new LLM/Agent processing
```

The actual perceived latency from user speech onset to bot silence is
typically **under 100ms**, dominated by the audio buffer size (one 20ms
WebRTC audio frame) and VAD detection latency.

---

## 7. Edge Cases and Recovery

### Interruption During Workflow Question Delivery

If the user interrupts while the bot is delivering a workflow question
(e.g., "What is your phone number?"):

- `_question_delivered` remains `False`.
- On the user's next turn, if the agent is still in
  `WORKFLOW_ACTIVE` with a `_pending_question`, the question is
  re-delivered.

### Interruption During Workflow Result Delivery

If the user interrupts during a workflow result:

- The result is not lost — it was already stored.
- The `_pending_result` mechanism ensures it will be delivered on the
  next opportunity.

### Double Interruption / Rapid Barge-In

If the user speaks, is silent briefly, then speaks again:

- Each speech onset may generate a new `InterruptionFrame`.
- `_cancel_current()` is idempotent — calling it multiple times is safe.
- The TTS `_interrupted` flag is reset on the next
  `LLMFullResponseStartFrame`, so a new generation starts clean.

### Fragment Filtering

Very short utterances (under 3 characters or single words under 12
characters) are filtered out to avoid triggering full LLM processing on
coughs, "um", or STT artifacts:

```python
if self._state != AgentState.AWAITING_INPUT:
    if len(stripped) < 3 or (word_count <= 2 and len(stripped) < 12):
        logger.warning(f"[LangGraph] Ignoring fragment: '{stripped}'")
        return
```

This filter is disabled in `AWAITING_INPUT` state so that short answers
like "yes", "no", or a 6-digit OTP code are not dropped.

---

## 8. Design Principles

1. **Non-blocking `process_frame`**: The frame handler returns
   immediately. All long-running work (LLM streaming, router, workflow)
   runs in `asyncio.Task`s so that system frames like
   `InterruptionFrame` can flow through without delay.

2. **Cooperative cancellation**: Tasks check `_cancelled` /
   `_interrupted` flags in their inner loops rather than relying solely
   on `asyncio.CancelledError`. This allows graceful cleanup.

3. **Workflow persistence**: Workflows are never cancelled by
   interruptions. They represent real tool executions (API calls,
   database lookups) that should complete regardless of conversational
   interruptions.

4. **Deferred delivery**: Results and questions from workflows are
   buffered when the user is speaking, preventing the bot from talking
   over the user.

5. **Voice isolation**: TTS voice selection is captured at sentence
   enqueue time, not synthesis time, preventing voice switches from
   corrupting in-flight audio.
