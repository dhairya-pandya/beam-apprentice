import json

import pytest

from apprentice.beam_client import Completion, ToolCall
from apprentice.episode import run_episode
from apprentice.ledger import BudgetExceeded
from apprentice.prompt import NUDGE_CONTINUE, NUDGE_LENGTH
from apprentice.tasks import Task
from apprentice.tools import Limits

TASK = Task("t1", "org/pkg", "abc", "", "", "add() subtracts instead of adding")


def tc(name, i, **args):
    return ToolCall(f"c{i}", name, json.dumps(args))


def comp(content="", calls=(), finish="stop", reasoning="r"):
    return Completion(content, reasoning, list(calls), finish, {"prompt_tokens": 10, "completion_tokens": 5},
                      cost_usd=0.001)


class ScriptedClient:
    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def complete(self, messages, tools, effort, max_tokens):
        self.seen.append([dict(m) for m in messages])
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def run(client, sandbox, limits=Limits()):
    return run_episode(TASK, client, "low", limits, data_root=None, sandbox_factory=lambda t, d: sandbox)


def test_episode_stops_on_submit(local_sandbox):
    client = ScriptedClient([
        comp(calls=[tc("edit_file", 1, filepath="pkg/core.py", old_string="a - b", new_string="a + b")]),
        comp(calls=[tc("submit_patch", 2)]),
        comp(content="should never be requested"),
    ])
    ep = run(client, local_sandbox)
    assert ep.stop_reason == "submitted"
    assert "+    return a + b" in ep.patch
    assert len(client.script) == 1
    assert ep.cost_usd == pytest.approx(0.002)


def test_nudge_after_text_only_turn_then_stop_after_3(local_sandbox):
    client = ScriptedClient([comp(content="thinking out loud") for _ in range(4)])
    ep = run(client, local_sandbox, Limits(nudges=3))
    assert ep.stop_reason == "nudges"
    nudges = [m["content"] for m in ep.messages if m["role"] == "user"][1:]
    assert nudges == [NUDGE_CONTINUE] * 3


def test_length_nudge(local_sandbox):
    client = ScriptedClient([comp(finish="length"), comp(calls=[tc("submit_patch", 1)])])
    ep = run(client, local_sandbox)
    assert NUDGE_LENGTH in [m["content"] for m in ep.messages if m["role"] == "user"]


def test_turn_cap(local_sandbox):
    client = ScriptedClient([comp(calls=[tc("run_command", i, command="true")]) for i in range(10)])
    ep = run(client, local_sandbox, Limits(turns=3))
    assert ep.stop_reason == "turns"


def test_tool_call_cap(local_sandbox):
    client = ScriptedClient([comp(calls=[tc("run_command", i, command="true")]) for i in range(10)])
    ep = run(client, local_sandbox, Limits(tool_calls=2))
    assert ep.stop_reason == "tool_calls"


def test_budget_exceeded_mid_episode_sets_stop_reason(local_sandbox):
    client = ScriptedClient([
        comp(calls=[tc("edit_file", 1, filepath="pkg/core.py", old_string="a - b", new_string="a + b")]),
        BudgetExceeded("cap"),
    ])
    ep = run(client, local_sandbox)
    assert ep.stop_reason == "budget"
    assert "a + b" in ep.patch  # unsubmitted work is still recovered
    assert ep.messages[-1]["role"] == "tool"


def test_reasoning_preserved_in_history(local_sandbox):
    client = ScriptedClient([
        comp(calls=[tc("get_status", 1)], reasoning="  first thought\n"),
        comp(calls=[tc("submit_patch", 2)], reasoning="second"),
    ])
    run(client, local_sandbox)
    second_request = client.seen[1]
    assistant = [m for m in second_request if m["role"] == "assistant"]
    assert assistant[0]["reasoning_content"] == "  first thought\n"


def test_unexpected_error_is_recorded(local_sandbox):
    client = ScriptedClient([RuntimeError("provider exploded")])
    ep = run(client, local_sandbox)
    assert ep.stop_reason == "error"
    assert "provider exploded" in ep.error
