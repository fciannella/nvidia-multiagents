#!/usr/bin/env bash
set -euo pipefail

# langgraph.json references ../env — create it from the container's
# environment so langgraph dev doesn't fail on a missing file.
# The real env vars are injected by docker-compose env_file.
touch /app/.env

echo "=== Starting LangGraph server (generic_agent) ==="
cd /app/generic_agent
langgraph dev --no-browser --n-jobs-per-worker 10 --host 0.0.0.0 --port 2024 &
LG_PID=$!

echo "Waiting for LangGraph on port 2024..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:2024/ok > /dev/null 2>&1; then
        echo "LangGraph is ready."
        break
    fi
    sleep 1
done

echo "=== Starting Generic Voice Pipeline (Dual-Thread) ==="
cd /app
exec python pipecat_voice_bot_generic.py --host 0.0.0.0 --port 7860
