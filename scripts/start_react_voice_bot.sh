#!/usr/bin/env bash
# Single-command runner: start LangGraph dev server, then Pipecat voice bot.
# Usage: ./scripts/start_react_voice_bot.sh [pipecat args...]
# Example: ./scripts/start_react_voice_bot.sh --port 7863

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LANGGRAPH_PORT="${LANGGRAPH_PORT:-2024}"
PIPECAT_PORT="${PIPECAT_PORT:-7863}"

cleanup() {
  if [[ -n "$LG_PID" ]] && kill -0 "$LG_PID" 2>/dev/null; then
    echo "[start_react_voice_bot] Stopping LangGraph (PID $LG_PID)..."
    kill "$LG_PID" 2>/dev/null || true
    wait "$LG_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# Start LangGraph from react_agent directory
cd "$ROOT/react_agent"
echo "[start_react_voice_bot] Starting LangGraph on port $LANGGRAPH_PORT..."
langgraph dev --no-browser --port "$LANGGRAPH_PORT" --n-jobs-per-worker 10 &
LG_PID=$!
cd "$ROOT"

# Wait for LangGraph to be ready
echo "[start_react_voice_bot] Waiting for LangGraph..."
for i in {1..60}; do
  if curl -s -o /dev/null "http://127.0.0.1:$LANGGRAPH_PORT/ok" 2>/dev/null || \
     curl -s -o /dev/null "http://127.0.0.1:$LANGGRAPH_PORT/" 2>/dev/null; then
    echo "[start_react_voice_bot] LangGraph ready."
    break
  fi
  if ! kill -0 "$LG_PID" 2>/dev/null; then
    echo "[start_react_voice_bot] LangGraph process exited." >&2
    exit 1
  fi
  sleep 1
done
if ! curl -s -o /dev/null "http://127.0.0.1:$LANGGRAPH_PORT/" 2>/dev/null; then
  echo "[start_react_voice_bot] Timeout waiting for LangGraph." >&2
  exit 1
fi

# Run Pipecat (forward args; default port 7863 if not specified)
echo "[start_react_voice_bot] Starting Pipecat..."
if [[ $# -eq 0 ]]; then
  exec python pipecat_voice_bot_react.py --port "$PIPECAT_PORT"
else
  exec python pipecat_voice_bot_react.py "$@"
fi
