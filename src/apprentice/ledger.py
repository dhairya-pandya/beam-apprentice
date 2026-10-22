"""Persistent spend ledger. Every model call is priced and appended; calls stop at the cap."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Prices:
    input_per_m: float = 0.40
    output_per_m: float = 1.20
    cached_input_per_m: float = 0.04

    def cost(self, usage: dict) -> float:
        prompt = usage.get("prompt_tokens") or 0
        completion = usage.get("completion_tokens") or 0
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        cached = min(cached, prompt)
        return (
            (prompt - cached) * self.input_per_m
            + cached * self.cached_input_per_m
            + completion * self.output_per_m
        ) / 1_000_000


class CostLedger:
    """`cap_usd` bounds lifetime spend in the file; `session_cap_usd` bounds this instance."""

    def __init__(
        self,
        path: Path,
        cap_usd: float = 25.0,
        prices: Prices = Prices(),
        session_cap_usd: float | None = None,
    ):
        self.path = Path(path)
        self.cap_usd = cap_usd
        self.session_cap_usd = session_cap_usd
        self.prices = prices
        self._lock = threading.Lock()
        self._session = 0.0
        self._lifetime = 0.0
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._lifetime += json.loads(line)["cost_usd"]

    def spent(self) -> float:
        return self._lifetime

    def session_spent(self) -> float:
        return self._session

    def check(self) -> None:
        if self._lifetime >= self.cap_usd:
            raise BudgetExceeded(f"lifetime cap ${self.cap_usd:.2f} reached (${self._lifetime:.4f})")
        if self.session_cap_usd is not None and self._session >= self.session_cap_usd:
            raise BudgetExceeded(f"session cap ${self.session_cap_usd:.2f} reached (${self._session:.4f})")

    def record(self, usage: dict, meta: dict) -> float:
        cost = self.prices.cost(usage)
        with self._lock:
            self._session += cost
            self._lifetime += cost
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps({"ts": time.time(), "cost_usd": cost, "usage": usage, **meta}) + "\n")
        return cost
