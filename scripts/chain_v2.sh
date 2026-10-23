#!/usr/bin/env bash
# After Beam resampling finishes: export v2 fine-tuning data and launch 12B training.
set -euo pipefail
cd "$(dirname "$0")/.."
until [ "$(grep -c 'this run spent' runs_beam_resample.log 2>/dev/null || echo 0)" -ge 2 ]; do sleep 60; done
uv run apprentice export-sft beam-train-low beam-train-medium beam-train-low-s2 beam-train-low-s3 --out data/sft_v2.jsonl
hf upload dhairyapandya/apprentice-sft-private data/sft_v2.jsonl sft_v2.jsonl --repo-type dataset
hf repos create dhairyapandya/apprentice-gemma4-12b-lora --private --exist-ok || true
hf jobs uv run --flavor a100-large --timeout 6h --secrets HF_TOKEN -d \
  -e SFT_HUB_DATA=dhairyapandya/apprentice-sft-private:sft_v2.jsonl \
  -e SFT_PUSH_REPO=dhairyapandya/apprentice-gemma4-12b-lora \
  -e SFT_MODEL=unsloth/gemma-4-12b-it -e SFT_MAX_LEN=16384 -e SFT_EPOCHS=2 \
  training/train_sft.py
