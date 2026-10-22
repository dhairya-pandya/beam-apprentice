"""Episode traces on disk: one JSON file per episode plus an append-only results table."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from apprentice.episode import Episode
from apprentice.verify import VerifyResult


def write_trace(ep: Episode, result: VerifyResult, out_dir: Path, task_repo: str = "") -> Path:
    out_dir = Path(out_dir)
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    path = out_dir / "traces" / f"{ep.instance_id}.{ep.effort}.json"
    verify = asdict(result)
    verify["log"] = verify["log"][-5000:]
    path.write_text(json.dumps({"episode": asdict(ep), "verify": verify, "repo": task_repo}, indent=1))
    row = {
        "instance_id": ep.instance_id,
        "repo": task_repo,
        "effort": ep.effort,
        "resolved": result.resolved,
        "applied": result.applied,
        "stop_reason": ep.stop_reason,
        "turns": ep.turns,
        "tool_calls": ep.tool_calls,
        "starved_turns": ep.starved_turns,
        "cost_usd": round(ep.cost_usd, 6),
        "prompt_tokens": ep.usage_totals.get("prompt_tokens", 0),
        "completion_tokens": ep.usage_totals.get("completion_tokens", 0),
        "cached_tokens": ep.usage_totals.get("cached_tokens", 0),
        "wall_s": round(ep.wall_s, 1),
        "error": ep.error,
    }
    with open(out_dir / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    return path


def read_results(out_dir: Path) -> list[dict]:
    path = Path(out_dir) / "results.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
