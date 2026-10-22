"""Run a model over a set of tasks: episode, verify, record. Stops cleanly at the spend cap."""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Callable

from apprentice.episode import Episode
from apprentice.tasks import Task
from apprentice.traces import read_results, write_trace
from apprentice.verify import VerifyResult


def run_baseline(
    tasks: list[Task],
    episode_fn: Callable[[Task], Episode],
    verify_fn: Callable[[Task, str], VerifyResult],
    out_dir: Path,
    effort: str,
    resume: bool = True,
    log: Callable[[str], None] = print,
) -> list[dict]:
    done = {(r["instance_id"], r["effort"]) for r in read_results(out_dir)} if resume else set()
    todo = [t for t in tasks if (t.instance_id, effort) not in done]
    log(f"{len(tasks) - len(todo)} done already, {len(todo)} to run at effort={effort}")
    for i, task in enumerate(todo, 1):
        ep = episode_fn(task)
        result = (
            verify_fn(task, ep.patch)
            if ep.patch.strip()
            else VerifyResult(False, False, 0, 0, log="no patch")
        )
        write_trace(ep, result, out_dir, task.repo)
        log(
            f"[{i}/{len(todo)}] {task.instance_id}: {'RESOLVED' if result.resolved else 'failed'} "
            f"stop={ep.stop_reason} turns={ep.turns} ${ep.cost_usd:.4f}"
        )
        if ep.stop_reason == "budget":
            log(f"spend cap reached, stopping: {ep.error}")
            break
    return read_results(out_dir)


def summarize(rows: list[dict]) -> str:
    if not rows:
        return "no results"
    lines = []
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["effort"], r["repo"])].append(r)
        groups[(r["effort"], "ALL")].append(r)
    lines.append(f"{'effort':8} {'repo':22} {'n':>4} {'resolved':>9} {'rate':>6} {'$/ep':>8} {'turns':>6}")
    for (effort, repo), rs in sorted(groups.items()):
        n = len(rs)
        ok = sum(r["resolved"] for r in rs)
        lines.append(
            f"{effort:8} {repo:22} {n:4d} {ok:9d} {ok / n:6.1%} "
            f"{sum(r['cost_usd'] for r in rs) / n:8.4f} {sum(r['turns'] for r in rs) / n:6.1f}"
        )
    stops = collections.Counter(r["stop_reason"] for r in rows)
    lines.append("stop reasons: " + ", ".join(f"{k}={v}" for k, v in stops.most_common()))
    lines.append(f"total spend: ${sum(r['cost_usd'] for r in rows):.4f}")
    return "\n".join(lines)
