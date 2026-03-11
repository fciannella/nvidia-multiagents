#!/usr/bin/env bash
set -euo pipefail

echo "=== Starting LangGraph server (multiagent) ==="
cd /app/multiagent
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

echo "=== Starting Multi-Agent Voice Pipeline (Magpie TTS) ==="
cd /app
exec python voice_pipeline_multiagent_magpie.py --host 0.0.0.0 --port 7860
