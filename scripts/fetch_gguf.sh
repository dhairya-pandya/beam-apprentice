#!/usr/bin/env bash
# Resumable, retrying download of a file from a (private) Hub model repo.
# usage: scripts/fetch_gguf.sh <repo> <filename> <outdir>
set -u
repo=$1; name=$2; out=$3/$2
url="https://huggingface.co/$repo/resolve/main/$name"
auth="Authorization: Bearer $(cat ~/.cache/huggingface/token)"
want=$(curl -sIL --http1.1 -H "$auth" "$url" | grep -i '^x-linked-size:\|^content-length:' | tail -1 | tr -dc '0-9')
for try in $(seq 1 50); do
  have=$(stat -f %z "$out" 2>/dev/null || echo 0)
  [ -n "$want" ] && [ "$have" = "$want" ] && { echo "done $name $have"; exit 0; }
  curl -sS -L --http1.1 -C - -H "$auth" -o "$out" "$url" || sleep 5
done
echo "gave up on $name"; exit 1
