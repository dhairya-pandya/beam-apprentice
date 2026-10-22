"""Reasoning-safe chat client for Beam (any OpenAI-compatible endpoint works).

Two Beam habits break naive agent loops:
  * thinking spends the completion budget, so a reply can come back empty with
    finish_reason == "length"; we retry once with double the budget (capped);
  * follow-up turns must carry each earlier assistant turn's reasoning verbatim.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Protocol

from apprentice.ledger import CostLedger

BEAM_MODEL = "reflection/Beam-1.0-501B-A23B"
HF_ROUTER = "https://router.huggingface.co/v1"
MAX_COMPLETION_CAP = 32768
TRANSIENT_RETRIES = 4


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class Completion:
    content: str
    reasoning: str
    tool_calls: list[ToolCall]
    finish_reason: str
    usage: dict
    cost_usd: float = 0.0
    starved: bool = False
    attempts: int = 1
    raw_extra: dict = field(default_factory=dict)


class ChatClient(Protocol):
    def complete(
        self, messages: list[dict], tools: list[dict], effort: str, max_tokens: int
    ) -> Completion: ...


def _field(obj, *names):
    for n in names:
        v = getattr(obj, n, None)
        if v is None and isinstance(getattr(obj, "model_extra", None), dict):
            v = obj.model_extra.get(n)
        if v is not None:
            return v
    return None


def _usage_dict(usage) -> dict:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return dict(usage)


def assistant_message(c: Completion) -> dict:
    msg: dict = {"role": "assistant", "content": c.content or ""}
    msg["reasoning_content"] = c.reasoning or ""
    if c.tool_calls:
        msg["tool_calls"] = [
            {"id": t.id, "type": "function", "function": {"name": t.name, "arguments": t.arguments}}
            for t in c.tool_calls
        ]
    return msg


class BeamClient:
    def __init__(self, openai_client, ledger: CostLedger, model: str = BEAM_MODEL, meta: dict | None = None):
        self.client = openai_client
        self.ledger = ledger
        self.model = model
        self.meta = meta or {}

    @classmethod
    def from_env(cls, ledger: CostLedger, base_url: str = HF_ROUTER, model: str = BEAM_MODEL, **kw):
        from openai import OpenAI

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is not set")
        return cls(OpenAI(base_url=base_url, api_key=token, timeout=600), ledger, model, **kw)

    def _send(self, messages, tools, effort, max_tokens):
        kwargs = dict(
            model=self.model,
            messages=messages,
            reasoning_effort=effort,
            max_completion_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
        for attempt in range(TRANSIENT_RETRIES + 1):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as e:
                transient = getattr(e, "status_code", None) in (429, 500, 502, 503, 504) or type(
                    e
                ).__name__ in ("APIConnectionError", "APITimeoutError")
                if not transient or attempt == TRANSIENT_RETRIES:
                    raise
                time.sleep(min(60, 2 * 2**attempt))

    def complete(self, messages: list[dict], tools: list[dict], effort: str, max_tokens: int) -> Completion:
        budget = max_tokens
        attempts = 0
        total_cost = 0.0
        while True:
            self.ledger.check()
            attempts += 1
            resp = self._send(messages, tools, effort, budget)
            usage = _usage_dict(getattr(resp, "usage", None))
            total_cost += self.ledger.record(
                usage, {"model": self.model, "effort": effort, "max_tokens": budget, **self.meta}
            )
            choice = resp.choices[0]
            msg = choice.message
            calls = [
                ToolCall(t.id, t.function.name, t.function.arguments or "{}")
                for t in (_field(msg, "tool_calls") or [])
            ]
            content = _field(msg, "content") or ""
            reasoning = _field(msg, "reasoning_content", "reasoning") or ""
            finish = choice.finish_reason or ""
            starved = finish == "length" and not content.strip() and not calls
            if starved and budget < MAX_COMPLETION_CAP:
                budget = min(budget * 2, MAX_COMPLETION_CAP)
                continue
            return Completion(content, reasoning, calls, finish, usage, total_cost, starved, attempts)
