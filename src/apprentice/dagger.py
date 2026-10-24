"""On-policy distillation (DAgger-style): the student drives, the teacher labels every state it reaches.

At each turn the teacher sees the same history the student saw, minus the student's private
reasoning, and proposes its own next action. That (state, teacher action) pair is recorded as a
training example. With probability beta the teacher's action is executed instead of the
student's, which steers some episodes into states closer to a good solution.

Training on these pairs teaches the student what to do from the situations it actually gets
itself into, including after its own mistakes, which plain imitation of teacher sessions never
shows it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from apprentice.beam_client import ChatClient, Completion, assistant_message
from apprentice.tools import SCHEMAS


def teacher_view(messages: list[dict]) -> list[dict]:
    """History as the teacher should see it: the student's reasoning removed, actions kept."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant":
            m["reasoning_content"] = ""
        out.append(m)
    return out


class DaggerRecorder:
    """turn_hook for run_episode: query the teacher at every state and record its action."""

    def __init__(self, teacher: ChatClient, teacher_effort: str = "low", beta: float = 0.0,
                 max_tokens: int = 8192, rng: random.Random | None = None):
        self.teacher = teacher
        self.teacher_effort = teacher_effort
        self.beta = beta
        self.max_tokens = max_tokens
        self.rng = rng or random.Random(0)
        self.samples: list[dict] = []
        self.teacher_errors = 0
        self.teacher_cost = 0.0

    def __call__(self, history: list[dict], student: Completion, task) -> Completion:
        try:
            t = self.teacher.complete(teacher_view(history), SCHEMAS, self.teacher_effort, self.max_tokens)
        except Exception:
            self.teacher_errors += 1
            return student
        self.teacher_cost += t.cost_usd
        if not t.tool_calls and not t.content.strip():
            return student  # an empty teacher reply is not a label
        use_teacher = self.rng.random() < self.beta
        self.samples.append({
            "instance_id": task.instance_id,
            "repo": task.repo,
            "step": sum(1 for m in history if m["role"] == "assistant"),
            "history": [dict(m) for m in history],
            "teacher": assistant_message(t),
            "student": assistant_message(student),
            "executed": "teacher" if use_teacher else "student",
        })
        return t if use_teacher else student

    def write(self, path: Path, resolved: bool) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            for s in self.samples:
                f.write(json.dumps({**s, "episode_resolved": resolved, "tools": SCHEMAS}) + "\n")
        return len(self.samples)
