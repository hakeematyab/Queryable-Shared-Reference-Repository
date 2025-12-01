#!/bin/bash

# ============================================
# vLLM Server Launch Script
# ============================================
# Usage: bash launch_vllm_server.sh
# Stop:  Ctrl+C

# Bypass proxy for localhost
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1


# Set cache directories FIRST
export HF_HOME=/scratch/hakeem.at/Queryable-Shared-Reference-Repository/notebooks/pretrained_models
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME

# Model and server config
MODEL="Qwen/Qwen2.5-72B-Instruct-AWQ"
PORT=8000

echo "============================================"
echo "Launching vLLM Server"
echo "============================================"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "HF_HOME: $HF_HOME"
echo "============================================"
echo ""
echo "Once you see 'Application startup complete', the server is ready."
echo "Connect from notebook using: http://localhost:$PORT/v1"
echo ""
echo "Press Ctrl+C to stop the server"
echo "============================================"

source /scratch/hakeem.at/Queryable-Shared-Reference-Repository/.venv/bin/activate

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --host 0.0.0.0 \
    --port $PORT \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768  \
    --max-num-batched-tokens 16384 \
    --enable-prefix-caching \
    --enable-chunked-prefill
    --max-num-seqs 128 \
    --enforce-eager \
    --trust-remote-code
    --disable-log-requests