"""SWE-style task records: loading, train/held-out split, and test-file discovery."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    instance_id: str
    repo: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str = ""


def load_tasks(path: Path) -> list[Task]:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            tasks.append(
                Task(
                    instance_id=row["instance_id"],
                    repo=row["repo"],
                    base_commit=row["base_commit"],
                    patch=row.get("patch", ""),
                    test_patch=row.get("test_patch", ""),
                    problem_statement=row.get("problem_statement", ""),
                    hints_text=row.get("hints_text") or "",
                )
            )
    return tasks


def split_tasks(
    tasks: list[Task], held_out_frac: float = 0.22, seed: int = 0
) -> tuple[list[Task], list[Task]]:
    """Split per repo so every repo with 2+ tasks lands on both sides. Deterministic for a seed."""
    by_repo: dict[str, list[Task]] = defaultdict(list)
    for t in tasks:
        by_repo[t.repo].append(t)

    rng = random.Random(seed)
    held_ids: set[str] = set()
    for repo in sorted(by_repo):
        group = sorted(by_repo[repo], key=lambda t: t.instance_id)
        rng.shuffle(group)
        if len(group) < 2:
            continue
        n_held = min(len(group) - 1, max(1, round(len(group) * held_out_frac)))
        held_ids.update(t.instance_id for t in group[:n_held])

    train = [t for t in tasks if t.instance_id not in held_ids]
    held = [t for t in tasks if t.instance_id in held_ids]
    return train, held


_DIFF_TARGET = re.compile(r"^\+\+\+ (?:b/)?(.+?)\s*$", re.MULTILINE)


def target_test_files(task: Task) -> list[str]:
    """Paths the hidden test patch writes to, in order, without duplicates."""
    seen: list[str] = []
    for path in _DIFF_TARGET.findall(task.test_patch):
        if path != "/dev/null" and path not in seen:
            seen.append(path)
    return seen
