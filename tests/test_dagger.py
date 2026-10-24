import json
import random

from apprentice.beam_client import Completion, ToolCall
from apprentice.dagger import DaggerRecorder, teacher_view
from apprentice.episode import run_episode
from apprentice.tasks import Task
from apprentice.tools import Limits

TASK = Task("t1", "org/pkg", "abc", "", "", "add() subtracts instead of adding")


def tc(name, i, **args):
    return ToolCall(f"{name}{i}", name, json.dumps(args))


def comp(calls=(), reasoning="r", content=""):
    return Completion(content, reasoning, list(calls), "tool_calls" if calls else "stop",
                      {"prompt_tokens": 10, "completion_tokens": 5}, cost_usd=0.001)


class Scripted:
    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def complete(self, messages, tools, effort, max_tokens):
        self.seen.append([dict(m) for m in messages])
        return self.script.pop(0)


FIX = dict(filepath="pkg/core.py", old_string="a - b", new_string="a + b")


def run(student, teacher, local_sandbox, beta, seed=0):
    rec = DaggerRecorder(teacher, teacher_effort="low", beta=beta, rng=random.Random(seed))
    ep = run_episode(TASK, student, "none", Limits(), data_root=None,
                     sandbox_factory=lambda t, d: local_sandbox, turn_hook=rec)
    return ep, rec


def test_student_acts_teacher_labels_every_state(local_sandbox):
    student = Scripted([comp([tc("read_file", 1, filepath="pkg/core.py")]), comp([tc("submit_patch", 2)])])
    teacher = Scripted([comp([tc("edit_file", 1, **FIX)], reasoning="fix the sign"),
                        comp([tc("submit_patch", 2)])])
    ep, rec = run(student, teacher, local_sandbox, beta=0.0)
    assert ep.stop_reason == "submitted"
    assert [s["executed"] for s in rec.samples] == ["student", "student"]
    assert rec.samples[0]["teacher"]["tool_calls"][0]["function"]["name"] == "edit_file"
    assert rec.samples[0]["teacher"]["reasoning_content"] == "fix the sign"
    # the state is the history before the step: system + user prompt for the first sample
    assert [m["role"] for m in rec.samples[0]["history"]] == ["system", "user"]
    assert len(rec.samples[1]["history"]) == 4  # + student's read_file call and its result


def test_beta_one_executes_teacher_actions(local_sandbox):
    student = Scripted([comp([tc("read_file", 1, filepath="pkg/core.py")]), comp([tc("read_file", 2, filepath="x")])])
    teacher = Scripted([comp([tc("edit_file", 1, **FIX)]), comp([tc("submit_patch", 2)])])
    ep, rec = run(student, teacher, local_sandbox, beta=1.0)
    assert [s["executed"] for s in rec.samples] == ["teacher", "teacher"]
    assert "+    return a + b" in ep.patch
    assert ep.stop_reason == "submitted"


def test_teacher_sees_history_without_student_reasoning(local_sandbox):
    student = Scripted([comp([tc("get_status", 1)], reasoning="student secret thoughts"), comp([tc("submit_patch", 2)])])
    teacher = Scripted([comp([tc("get_status", 1)]), comp([tc("submit_patch", 2)])])
    run(student, teacher, local_sandbox, beta=0.0)
    second_request = teacher.seen[1]
    assistant = [m for m in second_request if m["role"] == "assistant"]
    assert assistant and all(m.get("reasoning_content", "") == "" for m in assistant)


def test_teacher_view_strips_reasoning_but_keeps_tool_calls():
    msgs = [{"role": "assistant", "content": "", "reasoning_content": "x",
             "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}]
    view = teacher_view(msgs)
    assert view[0]["reasoning_content"] == "" and view[0]["tool_calls"] == msgs[0]["tool_calls"]
    assert msgs[0]["reasoning_content"] == "x"


def test_teacher_failure_falls_back_to_student(local_sandbox):
    class Broken:
        def complete(self, *a):
            raise RuntimeError("teacher down")
    student = Scripted([comp([tc("submit_patch", 1)])])
    ep, rec = run(student, Broken(), local_sandbox, beta=1.0)
    assert ep.stop_reason == "submitted"
    assert rec.samples == [] and rec.teacher_errors == 1
