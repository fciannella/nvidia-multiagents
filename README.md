# Magpie TTS Spark — Voice AI Pipelines

Three voice pipelines for real-time human-AI collaboration, built on [Pipecat](https://github.com/pipecat-ai/pipecat), [LangGraph](https://langchain-ai.github.io/langgraph/), and NVIDIA Riva NIM TTS.

## Pipelines

| # | Pipeline | Entry Point | Port | TTS | Agents |
|---|----------|-------------|------|-----|--------|
| 1 | **Multi-Agent (Magpie)** | `voice_pipeline_multiagent_magpie.py` | 7871 | Riva NIM (Magpie) | 4 scenarios, 2 agents each |
| 2 | **Multi-Agent (Qwen3)** | `voice_pipeline_multiagent.py` | 7870 | Qwen3-TTS | E-commerce only (Sarah + Mike) |
| 3 | **Single Agent (ReAct)** | `pipecat_voice_bot_react.py` | 7863 | Qwen3-TTS | Ron (single agent) |

---

## 1. Multi-Agent Voice Pipeline — Magpie TTS (recommended)

Two AI agents collaborate via voice on industry-specific scenarios. Includes a scenario picker, per-agent voice selection, and a built-in guide panel.

**Scenarios:**
- **E-commerce ML Deployment** — Sarah (Data Scientist) + Mike (IT Infrastructure Lead)
- **Autonomous Vehicle Safety Review** — Alex (Perception Engineer) + Jordan (Safety & Compliance Lead)
- **Healthcare AI Diagnostics** — Priya (Clinical AI Lead) + Marcus (Hospital CTO)
- **Game Studio AI Characters** — Luna (AI Character Designer) + Kai (Technical Director)

### Prerequisites

- Python 3.12+ with the project virtualenv (`.venv`)
- Riva NIM TTS server (Magpie) accessible at `192.168.7.203:9000` (or set `MAGPIE_TTS_SERVER`)
- AssemblyAI API key

### Environment variables

Create a `.env` file in the project root:

```bash
ASSEMBLYAI_API_KEY=your-assemblyai-key
```

Create `multiagent/.env` (see `multiagent/.env.example`):

```bash
# Router LLM (structured output for routing decisions)
MA_ROUTER_BASE_URL=https://openrouter.ai/api/v1
MA_ROUTER_MODEL=google/gemini-3-flash-preview
OPENROUTER_API_KEY=sk-or-v1-your-key

# Agent LLM (conversational responses)
MA_AGENT_BASE_URL=https://openrouter.ai/api/v1
MA_AGENT_MODEL=google/gemini-3-flash-preview
```

### Start

**Terminal 1 — LangGraph server:**

```bash
cd multiagent
source ../.venv/bin/activate
langgraph dev --n-jobs-per-worker 10 --no-browser
```

**Terminal 2 — Voice pipeline:**

```bash
source .venv/bin/activate
python voice_pipeline_multiagent_magpie.py
```

**Open** http://localhost:7871 in your browser. Select a scenario, choose a language, and click Start.

### Options

```bash
python voice_pipeline_multiagent_magpie.py --host 0.0.0.0 --port 7871
python voice_pipeline_multiagent_magpie.py -v     # debug logging
python voice_pipeline_multiagent_magpie.py -vv    # trace logging
```

### Override defaults

| Variable | Default | Description |
|----------|---------|-------------|
| `MAGPIE_TTS_SERVER` | `192.168.7.203:9000` | Riva NIM TTS endpoint |
| `LANGGRAPH_URL` | `http://localhost:2024` | LangGraph server URL |
| `ASSEMBLYAI_LANGUAGE` | `multi` | STT language (`multi` = auto-detect) |
| `MA_ROUTER_BASE_URL` | (in .env) | Router LLM endpoint |
| `MA_AGENT_BASE_URL` | (in .env) | Agent LLM endpoint |

---

## 2. Multi-Agent Voice Pipeline — Qwen3 TTS

Same multi-agent orchestrator as Pipeline 1, but uses Qwen3-TTS instead of Magpie. Fixed to the e-commerce scenario (Sarah + Mike).

### Prerequisites

- Qwen3-TTS server accessible (configure via env vars)
- AssemblyAI API key
- LangGraph server (same `multiagent/` backend)

### Start

**Terminal 1 — LangGraph server:**

```bash
cd multiagent
source ../.venv/bin/activate
langgraph dev --n-jobs-per-worker 10 --no-browser
```

**Terminal 2 — Voice pipeline:**

```bash
source .venv/bin/activate
python voice_pipeline_multiagent.py
```

**Open** http://localhost:7870

---

## 3. Single-Agent ReAct Voice Bot

A single conversational agent (Ron) with a front agent + background executor architecture. Handles telco and data/ML workflows with human-in-the-loop tool execution.

### Prerequisites

- Qwen3-TTS server accessible
- AssemblyAI API key

### Start

**Terminal 1 — LangGraph server:**

```bash
cd react_agent
source ../.venv/bin/activate
langgraph dev --n-jobs-per-worker 10 --no-browser
```

**Terminal 2 — Voice pipeline:**

```bash
source .venv/bin/activate
python pipecat_voice_bot_react.py
```

**Open** http://localhost:7863

---

## Docker Compose (all three pipelines)

Run all three pipelines in isolated containers with a single command. Each container bundles its own LangGraph server internally.

### Setup

```bash
cp .env.example .env
# Edit .env with your API keys and server addresses
```

### Run all pipelines

```bash
docker compose up --build
```

### Run a single pipeline

```bash
docker compose up --build magpie    # Pipeline 1 → http://localhost:7871
docker compose up --build qwen3     # Pipeline 2 → http://localhost:7870
docker compose up --build react     # Pipeline 3 → http://localhost:7863
```

### Ports

| Service | Host Port | Container Port |
|---------|-----------|----------------|
| `magpie` | 7871 | 7860 |
| `qwen3` | 7870 | 7860 |
| `react` | 7863 | 7860 |

### Configuration

All environment variables are read from `.env` (root) and `multiagent/.env`. See `.env.example` for the full list. You can override any variable directly in `docker-compose.yml` under the `environment` key for each service.

### Logs

```bash
docker compose logs -f magpie   # follow one service
docker compose logs -f          # follow all services
```

### Stop

```bash
docker compose down
```

---

## Project Structure

```
.
├── voice_pipeline_multiagent_magpie.py   # Pipeline 1 entry point
├── voice_pipeline_multiagent.py          # Pipeline 2 entry point
├── pipecat_voice_bot_react.py            # Pipeline 3 entry point
│
├── assemblyai_multilingual_stt.py        # Shared: AssemblyAI STT processor
├── riva_nim_realtime_tts.py              # Shared: Riva NIM WebSocket TTS processor
├── qwen3_tts.py                          # Shared: Qwen3-TTS processor
├── react_agent_service.py                # Pipeline 3: ReAct agent Pipecat processor
│
├── multiagent/                           # Backend for Pipelines 1 & 2
│   ├── langgraph.json                    #   Graph registry (orchestrator + executor)
│   ├── .env                              #   LLM configuration
│   ├── src/
│   │   ├── graph.py                      #   Orchestrator StateGraph
│   │   ├── executor.py                   #   Background tool executor
│   │   ├── config.py                     #   LLM config (router/agent models)
│   │   ├── models.py                     #   Pydantic models
│   │   ├── prompts.py                    #   Dynamic prompt templates
│   │   ├── utils.py                      #   Message conversion utilities
│   │   └── scenarios/                    #   Scenario definitions
│   │       ├── __init__.py               #     Registry + ScenarioConfig/AgentConfig
│   │       ├── ecommerce.py              #     Sarah + Mike (ML deployment)
│   │       ├── autonomous_vehicle.py     #     Alex + Jordan (AV safety)
│   │       ├── healthcare.py             #     Priya + Marcus (ChestAI)
│   │       └── game_studio.py            #     Luna + Kai (NPC AI)
│   └── voice/
│       ├── langgraph_processor.py        #   Pipecat ↔ LangGraph bridge
│       ├── voice_switch_processor.py     #   Per-agent voice switching
│       └── frames.py                     #   VoiceSwitchFrame definition
│
├── react_agent/                          # Backend for Pipeline 3
│   ├── agents/
│   │   ├── front_agent.py                #   Conversational ReAct agent
│   │   └── executor.py                   #   Background tool executor
│   └── config.py                         #   LLM config
│
├── Dockerfile                            # Shared container image
├── docker-compose.yml                    # Three-service compose file
├── docker/
│   ├── entrypoint-magpie.sh              #   Container entrypoint (Pipeline 1)
│   ├── entrypoint-qwen3.sh               #   Container entrypoint (Pipeline 2)
│   └── entrypoint-react.sh               #   Container entrypoint (Pipeline 3)
│
├── scripts/                              # Standalone utilities
├── pipelines/                            # Archived older pipeline entry points
├── test-pipelines/                       # Archived test pipeline files
└── archived/                             # Fully archived code
```
