#!/usr/bin/env bash
# Serve a local GGUF with llama-server and run it over the held-out split.
# usage: scripts/eval_local.sh <gguf path> <run id> [split]
set -euo pipefail
cd "$(dirname "$0")/.."
model=$1; run_id=$2; split=${3:-held_out}
pkill -f 'llama-server -m' || true
sleep 2
llama-server -m "$model" --port 8090 -c 65536 --jinja -ngl 99 --alias student \
  --chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}' > "runs_llama_${run_id}.log" 2>&1 &
server=$!
until curl -s localhost:8090/health 2>/dev/null | grep -q '"ok"'; do sleep 3; done
uv run apprentice baseline --run-id "$run_id" --split "$split" --effort none --free --api-key local \
  --base-url http://localhost:8090/v1 --model student --max-tokens 4096 --workers 1 --budget-usd 1
kill "$server"
