#!/usr/bin/env bash
# Fine-tune the student on a free Colab T4 and bring the LoRA adapter back.
#
#   training/colab_train.sh data/sft.jsonl runs/student-v1 [KEY=VALUE ...]
#
# Extra KEY=VALUE pairs become environment variables for train_sft.py
# (e.g. SFT_MAX_STEPS=2 for a smoke test, SFT_THINKING=0 to drop the teacher's thinking).
set -euo pipefail

DATA=${1:?path to sft.jsonl}
OUT=${2:?local output directory}
shift 2
SESSION=apprentice-train-$(date +%s)
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"

ENV_ARGS=()
for kv in "$@"; do ENV_ARGS+=(--env "$kv"); done

cleanup() { colab stop -s "$SESSION" >/dev/null 2>&1 || true; }
trap cleanup EXIT

colab new -s "$SESSION" --gpu T4
colab install -s "$SESSION" -r "$HERE/requirements.txt"
colab upload -s "$SESSION" "$DATA" /content/sft.jsonl
colab exec -s "$SESSION" --timeout 43200 \
  --env SFT_DATA=/content/sft.jsonl --env SFT_OUT=/content/out "${ENV_ARGS[@]+"${ENV_ARGS[@]}"}" \
  -f "$HERE/train_sft.py" 2>&1 | tee "$OUT/train_stdout.log"
colab download -s "$SESSION" /content/out/adapter.zip "$OUT/adapter.zip"
colab download -s "$SESSION" /content/out/train_log.json "$OUT/train_log.json"
(cd "$OUT" && rm -rf adapter && mkdir adapter && unzip -q adapter.zip -d adapter)
echo "adapter saved to $OUT/adapter"
