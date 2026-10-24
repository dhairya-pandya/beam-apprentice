#!/usr/bin/env bash
# Launch llama-server on an HF Jobs GPU, serving a GGUF from a private repo, with the port exposed.
# usage: scripts/serve_gguf_job.sh <repo> <file> [flavor]
# Compute runs in the active `hf auth` account; the source repo is read with the token
# stored for SRC_ACCOUNT (default oauth-dhairyapandya), passed as a job secret.
set -euo pipefail
repo=$1; file=$2; flavor=${3:-l4x1}
secrets=$(mktemp); chmod 600 "$secrets"; trap 'rm -f "$secrets"' EXIT
python3 - "$secrets" "${SRC_ACCOUNT:-oauth-dhairyapandya}" <<'PY'
import configparser, os, sys
c = configparser.ConfigParser(); c.read(os.path.expanduser("~/.cache/huggingface/stored_tokens"))
open(sys.argv[1], "w").write("SRC_HF_TOKEN=" + c[sys.argv[2]]["hf_token"] + "\n")
PY
hf jobs run --flavor "$flavor" --timeout 8h --expose 8080 --secrets-file "$secrets" -d \
  ghcr.io/ggml-org/llama.cpp:server-cuda /bin/sh -c \
  "curl -sSfL --retry 5 -H \"Authorization: Bearer \$SRC_HF_TOKEN\" -o /tmp/model.gguf \
     https://huggingface.co/$repo/resolve/main/$file && \
   exec /app/llama-server -m /tmp/model.gguf \
   --host 0.0.0.0 --port 8080 --ctx-size 65536 --jinja --n-gpu-layers 99 --alias student \
   --chat-template-kwargs '{\"enable_thinking\":true,\"preserve_thinking\":true}'"
