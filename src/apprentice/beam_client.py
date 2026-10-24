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
OUTAGE_RETRIES = 10  # about five minutes of 30-second waits


def provider_live(model: str = BEAM_MODEL) -> bool:
    """Whether the Hugging Face router currently lists a live provider for the model."""
    import json
    import urllib.request

    url = f"https://huggingface.co/api/models/{model}?expand[]=inferenceProviderMapping"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            mapping = json.load(r).get("inferenceProviderMapping") or {}
    except Exception:
        return False
    return any(v.get("status") == "live" for v in mapping.values())


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


def hf_token() -> str:
    """HF_TOKEN from the environment, else the token saved by `hf auth login`."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        path = os.path.join(home, "token")
        if os.path.exists(path):
            token = open(path).read().strip()
    if not token:
        raise RuntimeError("no Hugging Face token: set HF_TOKEN or run `hf auth login`")
    return token


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
    def __init__(self, openai_client, ledger: CostLedger, model: str = BEAM_MODEL, meta: dict | None = None,
                 extra_body: dict | None = None):
        self.client = openai_client
        self.ledger = ledger
        self.model = model
        self.meta = meta or {}
        self.extra_body = extra_body

    @classmethod
    def from_env(cls, ledger: CostLedger, base_url: str = HF_ROUTER, model: str = BEAM_MODEL,
                 api_key: str | None = None, **kw):
        from openai import OpenAI

        token = api_key or os.environ.get("APPRENTICE_API_KEY") or hf_token()
        return cls(OpenAI(base_url=base_url, api_key=token, timeout=600), ledger, model, **kw)

    def _send(self, messages, tools, effort, max_tokens):
        kwargs = dict(model=self.model, messages=messages, max_completion_tokens=max_tokens)
        if effort != "none":  # "none" = model without a reasoning dial (e.g. a local student)
            kwargs["reasoning_effort"] = effort
        if tools:
            kwargs["tools"] = tools
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        for attempt in range(TRANSIENT_RETRIES + 1):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as e:
                outage = "not supported by any provider" in str(e)  # provider briefly dropped the model
                transient = outage or getattr(e, "status_code", None) in (429, 500, 502, 503, 504) or type(
                    e
                ).__name__ in ("APIConnectionError", "APITimeoutError")
                if not transient or attempt == (OUTAGE_RETRIES if outage else TRANSIENT_RETRIES):
                    raise
                time.sleep(30 if outage else min(60, 2 * 2**attempt))

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
