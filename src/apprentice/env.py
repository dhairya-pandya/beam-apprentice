"""Environment entry point in the `load_environment()` style used by RL environment hubs.

    env = load_environment(split="held_out")
    ep = env.rollout(client, env.dataset[0]["instance_id"], effort="low")
    reward = env.reward(ep)        # 1.0 if the hidden tests pass, else 0.0
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

from apprentice.beam_client import ChatClient
from apprentice.episode import Episode, run_episode
from apprentice.tasks import Task, load_tasks, split_tasks
from apprentice.tools import Limits
from apprentice.verify import TaskReference, VerifyResult, load_references, verify


class ApprenticeEnv:
    def __init__(self, tasks: list[Task], refs: dict[str, TaskReference], data_root: Path,
                 limits: Limits = Limits(), verify_fn=verify):
        self.tasks = {t.instance_id: t for t in tasks}
        self.refs = refs
        self.data_root = data_root
        self.limits = limits
        self._verify = verify_fn
        self.dataset = [
            {"instance_id": t.instance_id, "repo": t.repo, "problem_statement": t.problem_statement}
            for t in tasks
        ]
        self.last_result: VerifyResult | None = None

    def rollout(self, client: ChatClient, instance_id: str, effort: str = "low", max_tokens: int = 8192) -> Episode:
        return run_episode(self.tasks[instance_id], client, effort, self.limits, self.data_root, max_tokens)

    def score(self, episode: Episode) -> VerifyResult:
        task = self.tasks[episode.instance_id]
        if not episode.patch.strip():
            self.last_result = VerifyResult(False, False, 0, 0, log="no patch")
        else:
            self.last_result = self._verify(task, episode.patch, self.data_root, self.refs[task.instance_id])
        return self.last_result

    def reward(self, episode: Episode) -> float:
        return 1.0 if self.score(episode).resolved else 0.0

    def info(self) -> dict:
        return {"n_tasks": len(self.tasks), "limits": asdict(self.limits)}


def load_environment(
    data_root: str | None = None,
    split: str = "held_out",
    references: str = "runs/references.jsonl",
    seed: int = 0,
    limits: Limits = Limits(),
    **kw,
) -> ApprenticeEnv:
    root = Path(data_root or os.environ.get("APPRENTICE_DATA", "external/swe-tasks"))
    refs = load_references(Path(references))
    tasks = [t for t in load_tasks(root / "tasks.jsonl") if t.instance_id in refs and refs[t.instance_id].usable]
    train, held = split_tasks(tasks, seed=seed)
    chosen = {"train": train, "held_out": held, "all": tasks}[split]
    return ApprenticeEnv(chosen, refs, root, limits, **kw)
