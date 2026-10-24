#!/usr/bin/env bash
# Finish the base E4B evaluation (no teacher needed), then collect on-policy labels with Beam.
# The dagger command waits whenever Beam's provider is down.
set -u
cd "$(dirname "$0")/.."
uv run apprentice baseline --run-id e4b-base-heldout-v13 --split held_out --effort none --free --api-key local \
  --base-url http://localhost:8090/v1 --model student --max-tokens 4096 --workers 1 --budget-usd 1
uv run apprentice dagger --run-id dagger-e4b-r1 --student-url http://localhost:8090/v1 --split train \
  --limit 25 --sample-seed 1 --beta 0.3 --workers 1 --budget-usd 2.50 --lifetime-cap-usd 40
uv run apprentice export-dagger dagger-e4b-r1 --out data/dagger_e4b_r1.jsonl
