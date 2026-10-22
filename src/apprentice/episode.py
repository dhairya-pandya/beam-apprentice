"""One agent episode: prompt, tool loop, nudges, caps, and the resulting patch."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from apprentice.beam_client import ChatClient, assistant_message
from apprentice.ledger import BudgetExceeded
from apprentice.patches import strip_protected
from apprentice.prompt import SYSTEM, build_prompt, nudge_for
from apprentice.sandbox import Sandbox
from apprentice.tasks import Task
from apprentice.tools import Limits, Toolset

StopReason = Literal["submitted", "turns", "tool_calls", "nudges", "budget", "error"]


@dataclass
class Episode:
    instance_id: str
    effort: str
    messages: list[dict]
    patch: str
    stop_reason: StopReason
    usage_totals: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    wall_s: float = 0.0
    turns: int = 0
    tool_calls: int = 0
    starved_turns: int = 0
    error: str = ""


def _add_usage(totals: dict, usage: dict) -> None:
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        totals[k] = totals.get(k, 0) + (usage.get(k) or 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    totals["cached_tokens"] = totals.get("cached_tokens", 0) + cached
    totals["reasoning_tokens"] = totals.get("reasoning_tokens", 0) + reasoning


def run_episode(
    task: Task,
    client: ChatClient,
    effort: str,
    limits: Limits,
    data_root: Path | None,
    max_tokens: int = 8192,
    sandbox_factory: Callable = Sandbox,
) -> Episode:
    started = time.time()
    with sandbox_factory(task, data_root) as sb:
        tools = Toolset(sb, limits)
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(task, tools.workspace_listing(), limits)},
        ]
        totals: dict = {}
        cost = 0.0
        starved = 0
        nudges = 0
        stop: StopReason = "turns"
        error = ""

        for turn in range(limits.turns):
            tools.turns_used = turn + 1
            try:
                c = client.complete(messages, tools.schemas(), effort, max_tokens)
            except BudgetExceeded as e:
                stop, error = "budget", str(e)
                break
            except Exception as e:
                stop, error = "error", f"{type(e).__name__}: {e}"
                break
            messages.append(assistant_message(c))
            _add_usage(totals, c.usage)
            cost += c.cost_usd
            starved += int(c.starved)

            if c.tool_calls:
                nudges = 0
                capped = False
                for call in c.tool_calls:
                    if call.name not in ("submit_patch", "get_status") and tools.tool_calls_used >= limits.tool_calls:
                        result = '{"status": "error", "error": "tool call budget exhausted"}'
                        capped = True
                    else:
                        result = tools.call(call.name, call.arguments)
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                if tools.submitted:
                    stop = "submitted"
                    break
                if capped:
                    stop = "tool_calls"
                    break
                continue

            nudges += 1
            if nudges > limits.nudges:
                stop = "nudges"
                break
            messages.append({"role": "user", "content": nudge_for(c.finish_reason, c.content)})

        patch = tools.patch if tools.submitted else strip_protected(sb.diff())

    return Episode(
        instance_id=task.instance_id,
        effort=effort,
        messages=messages,
        patch=patch,
        stop_reason=stop,
        usage_totals=totals,
        cost_usd=cost,
        wall_s=time.time() - started,
        turns=tools.turns_used,
        tool_calls=tools.tool_calls_used,
        starved_turns=starved,
        error=error,
    )
