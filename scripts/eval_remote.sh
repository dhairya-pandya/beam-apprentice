#!/usr/bin/env bash
# Wait for a llama-server exposed on HF Jobs, then evaluate it on a split from this machine.
# usage: scripts/eval_remote.sh <job id> <run id> [split] [workers]
# The endpoint is authorised with the token stored for JOB_ACCOUNT (default: the job namespace's login).
set -euo pipefail
cd "$(dirname "$0")/.."
job=$1; run_id=$2; split=${3:-held_out}; workers=${4:-2}
url="https://${job##*/}--8080.hf.jobs"
export APPRENTICE_API_KEY=$(python3 - "${JOB_ACCOUNT:-oauth-${job%%/*}}" <<'PY'
import configparser, os, sys
c = configparser.ConfigParser(); c.read(os.path.expanduser("~/.cache/huggingface/stored_tokens"))
print(c[sys.argv[1]]["hf_token"])
PY
)
until curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $APPRENTICE_API_KEY" "$url/health" | grep -q 200; do
  hf jobs inspect "$job" 2>&1 | grep -qE "'stage': '(ERROR|COMPLETED)'" && { echo "server job ended"; exit 1; }
  sleep 15
done
echo "server healthy at $url"
uv run apprentice baseline --run-id "$run_id" --split "$split" --effort none --free \
  --base-url "$url/v1" --model student --max-tokens 4096 --workers "$workers" --budget-usd 1
