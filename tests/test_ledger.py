import pytest

from apprentice.ledger import BudgetExceeded, CostLedger, Prices


def test_cost_math(tmp_path):
    led = CostLedger(tmp_path / "l.jsonl", cap_usd=10, prices=Prices())
    cost = led.record({"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}, {})
    assert cost == pytest.approx(1.60)
    assert led.spent() == pytest.approx(1.60)


def test_cached_tokens_billed_at_cached_rate(tmp_path):
    led = CostLedger(tmp_path / "l.jsonl", cap_usd=10, prices=Prices())
    usage = {
        "prompt_tokens": 1_000_000,
        "completion_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 500_000},
    }
    assert led.record(usage, {}) == pytest.approx(0.20 + 0.02)


def test_check_raises_at_cap(tmp_path):
    led = CostLedger(tmp_path / "l.jsonl", cap_usd=0.50, prices=Prices())
    led.check()
    led.record({"prompt_tokens": 0, "completion_tokens": 500_000}, {})
    with pytest.raises(BudgetExceeded):
        led.check()


def test_ledger_persists_across_instances(tmp_path):
    path = tmp_path / "l.jsonl"
    CostLedger(path, cap_usd=10).record({"prompt_tokens": 1_000_000, "completion_tokens": 0}, {"run": "a"})
    assert CostLedger(path, cap_usd=10).spent() == pytest.approx(0.40)


def test_session_cap_separate_from_lifetime(tmp_path):
    path = tmp_path / "l.jsonl"
    CostLedger(path, cap_usd=10).record({"prompt_tokens": 0, "completion_tokens": 1_000_000}, {})
    led = CostLedger(path, cap_usd=10, session_cap_usd=0.5)
    led.check()
    led.record({"prompt_tokens": 0, "completion_tokens": 500_000}, {})
    with pytest.raises(BudgetExceeded):
        led.check()
