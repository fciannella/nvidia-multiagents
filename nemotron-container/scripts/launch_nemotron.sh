#!/bin/bash
set -e

DTYPE="${DTYPE:-FP8}"

if [ "$DTYPE" = "FP8" ]; then
    KV_CACHE_DTYPE="fp8"
    export VLLM_USE_FLASHINFER_MOE_FP8=1
    export VLLM_FLASHINFER_MOE_BACKEND=throughput
else
    KV_CACHE_DTYPE="auto"
fi

exec vllm serve "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-${DTYPE}" \
    --trust-remote-code \
    --async-scheduling \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --gpu-memory-utilization 0.63 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser-plugin nano_v3_reasoning_parser.py \
    --reasoning-parser nano_v3 \
    --tensor-parallel-size 1
