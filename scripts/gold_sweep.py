"""Build each task's test reference from its known-good fix and keep only usable tasks.

A task is usable when at least one hidden test fails without the fix and passes with it,
and the fix itself verifies. Results go to runs/references.jsonl and runs/excluded.txt.

Usage: uv run python scripts/gold_sweep.py [--data-root external/swe-tasks] [--workers 3]
"""

from __future__ import annotations

import argparse
import collections
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from apprentice.sandbox import image_exists, snapshot_path
from apprentice.tasks import load_tasks
from apprentice.verify import build_reference, load_references, save_references, verify


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="external/swe-tasks")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    data = Path(args.data_root)
    out = Path(args.out)
    refs_path = out / "references.jsonl"
    done = load_references(refs_path)
    log_path = out / "gold_sweep.jsonl"
    tasks = [
        t for t in load_tasks(data / "tasks.jsonl")
        if t.instance_id not in done and snapshot_path(t, data).exists() and image_exists(t.repo)
    ]
    print(f"{len(done)} already done, {len(tasks)} to sweep", flush=True)

    def one(task):
        try:
            ref = build_reference(task, data)
            ok = ref.usable and verify(task, task.patch, data, ref).resolved
            return task, ref, ok, ""
        except Exception as e:
            return task, None, False, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(args.workers) as pool:
        for fut in as_completed([pool.submit(one, t) for t in tasks]):
            task, ref, ok, err = fut.result()
            if ref is not None:
                done[task.instance_id] = ref
                save_references(list(done.values()), refs_path)
            row = {"instance_id": task.instance_id, "repo": task.repo, "usable": ok, "error": err,
                   "f2p": len(ref.fail_to_pass) if ref else 0, "p2p": len(ref.pass_to_pass) if ref else 0}
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

    rows = {}
    for line in open(log_path):
        r = json.loads(line)
        rows[r["instance_id"]] = r
    excluded = sorted(i for i, r in rows.items() if not r["usable"])
    (out / "excluded.txt").write_text("\n".join(excluded) + ("\n" if excluded else ""))
    per_repo = collections.defaultdict(lambda: [0, 0])
    for r in rows.values():
        per_repo[r["repo"]][0] += int(r["usable"])
        per_repo[r["repo"]][1] += 1
    for repo, (u, n) in sorted(per_repo.items()):
        print(f"{repo}: {u}/{n} usable")


if __name__ == "__main__":
    main()
