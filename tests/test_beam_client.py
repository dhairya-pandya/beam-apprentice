from types import SimpleNamespace as NS

import pytest

from apprentice.beam_client import BeamClient, Completion, ToolCall, assistant_message
from apprentice.ledger import BudgetExceeded, CostLedger


def _resp(content="", reasoning="", finish="stop", tool_calls=None, prompt=100, completion=50):
    msg = NS(content=content, reasoning_content=reasoning, tool_calls=tool_calls)
    usage = NS(model_dump=lambda: {"prompt_tokens": prompt, "completion_tokens": completion})
    return NS(choices=[NS(message=msg, finish_reason=finish)], usage=usage)


class FakeOpenAI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        return self.responses.pop(0)


def _client(fake, tmp_path, cap=10.0):
    return BeamClient(fake, ledger=CostLedger(tmp_path / "l.jsonl", cap_usd=cap), model="m")


def test_retries_on_length_with_empty_content(tmp_path):
    fake = FakeOpenAI([_resp(finish="length", reasoning="thinking..."), _resp(content="done")])
    out = _client(fake, tmp_path).complete([{"role": "user", "content": "hi"}], [], "low", 4096)
    assert out.content == "done"
    assert len(fake.calls) == 2
    assert fake.calls[0]["max_completion_tokens"] == 4096
    assert fake.calls[1]["max_completion_tokens"] == 8192


def test_no_retry_when_tool_call_present(tmp_path):
    tc = NS(id="c1", function=NS(name="read_file", arguments='{"filepath": "a.py"}'))
    fake = FakeOpenAI([_resp(finish="length", tool_calls=[tc])])
    out = _client(fake, tmp_path).complete([], [], "low", 4096)
    assert len(fake.calls) == 1
    assert out.tool_calls == [ToolCall("c1", "read_file", '{"filepath": "a.py"}')]


def test_retry_capped_at_32768(tmp_path):
    fake = FakeOpenAI([_resp(finish="length") for _ in range(10)])
    out = _client(fake, tmp_path).complete([], [], "high", 16384)
    assert [c["max_completion_tokens"] for c in fake.calls] == [16384, 32768]
    assert out.finish_reason == "length"
    assert out.content == ""
    assert out.starved is True


def test_assistant_message_preserves_reasoning():
    c = Completion(
        content="ok", reasoning="  exact\nreasoning  ", tool_calls=[ToolCall("c1", "get_status", "{}")],
        finish_reason="tool_calls", usage={},
    )
    msg = assistant_message(c)
    assert msg["role"] == "assistant"
    assert msg["reasoning_content"] == "  exact\nreasoning  "
    assert msg["tool_calls"][0]["function"] == {"name": "get_status", "arguments": "{}"}


def test_effort_and_tools_are_sent(tmp_path):
    fake = FakeOpenAI([_resp(content="x")])
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    _client(fake, tmp_path).complete([], tools, "medium", 1000)
    assert fake.calls[0]["reasoning_effort"] == "medium"
    assert fake.calls[0]["tools"] == tools


def test_ledger_checked_before_send(tmp_path):
    led = CostLedger(tmp_path / "l.jsonl", cap_usd=0.0)
    fake = FakeOpenAI([_resp(content="x")])
    with pytest.raises(BudgetExceeded):
        BeamClient(fake, ledger=led, model="m").complete([], [], "low", 100)
    assert fake.calls == []


def test_usage_is_recorded(tmp_path):
    fake = FakeOpenAI([_resp(content="x", prompt=1_000_000, completion=0)])
    client = _client(fake, tmp_path)
    client.complete([], [], "low", 100)
    assert client.ledger.spent() == pytest.approx(0.40)
