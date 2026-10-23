#!/usr/bin/env bash
# After the 12B training job completes: export untuned and fine-tuned 12B as GGUF.
set -euo pipefail
cd "$(dirname "$0")/.."
JOB=dhairyapandya/6adbbd5a046eb82b1ecd6bc1
until hf jobs inspect "$JOB" 2>&1 | grep -qE "'stage': '(COMPLETED|ERROR|CANCELED|CANCELLED)'"; do sleep 60; done
hf jobs inspect "$JOB" 2>&1 | grep -q "'stage': 'COMPLETED'" || { echo "training job did not complete"; exit 1; }
hf jobs logs "$JOB" 2>&1 | grep -q "pushed adapter" || { echo "adapter was not pushed"; exit 1; }
hf repos create dhairyapandya/apprentice-gemma4-12b-gguf --private --exist-ok
hf jobs uv run --flavor cpu-xl --timeout 4h --secrets HF_TOKEN -d \
  -e BASE=google/gemma-4-12B-it -e NAME=gemma-4-12B-it \
  -e ADAPTER=dhairyapandya/apprentice-gemma4-12b-lora \
  -e OUT_REPO=dhairyapandya/apprentice-gemma4-12b-gguf \
  training/export_gguf.py
