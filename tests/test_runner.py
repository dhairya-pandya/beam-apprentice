from apprentice.episode import Episode
from apprentice.runner import run_baseline, summarize
from apprentice.tasks import Task
from apprentice.traces import read_results
from apprentice.verify import VerifyResult

TASKS = [Task(f"t{i}", "org/a" if i < 2 else "org/b", "c", "", "", "bug") for i in range(3)]


def ep(task, stop="submitted", patch="diff --git a/x b/x\n", cost=0.01):
    return Episode(task.instance_id, "low", [{"role": "user", "content": "hi"}], patch, stop,
                   {"prompt_tokens": 10}, cost, 1.0, turns=3, tool_calls=2)


def test_baseline_writes_trace_and_result_rows(tmp_path):
    rows = run_baseline(TASKS, ep, lambda t, p: VerifyResult(t.instance_id != "t1", True, 1, 0),
                        tmp_path, "low", log=lambda s: None)
    assert [r["resolved"] for r in rows] == [True, False, True]
    assert (tmp_path / "traces" / "t0.low.json").exists()


def test_resume_skips_done_tasks(tmp_path):
    calls = []
    def episode(t):
        calls.append(t.instance_id)
        return ep(t)
    verify = lambda t, p: VerifyResult(True, True, 1, 0)
    run_baseline(TASKS[:2], episode, verify, tmp_path, "low", log=lambda s: None)
    run_baseline(TASKS, episode, verify, tmp_path, "low", log=lambda s: None)
    assert calls == ["t0", "t1", "t2"]


def test_budget_stop_is_clean(tmp_path):
    def episode(t):
        return ep(t, stop="budget" if t.instance_id == "t1" else "submitted")
    msgs = []
    rows = run_baseline(TASKS, episode, lambda t, p: VerifyResult(False, True, 0, 1), tmp_path, "low",
                        log=msgs.append)
    assert [r["instance_id"] for r in rows] == ["t0", "t1"]
    assert any("spend cap" in m for m in msgs)


def test_empty_patch_is_not_verified(tmp_path):
    called = []
    run_baseline(TASKS[:1], lambda t: ep(t, patch=""), lambda t, p: called.append(t) or VerifyResult(True, True, 1, 0),
                 tmp_path, "low", log=lambda s: None)
    assert called == []
    assert read_results(tmp_path)[0]["resolved"] is False


def test_summarize_counts(tmp_path):
    rows = run_baseline(TASKS, ep, lambda t, p: VerifyResult(t.instance_id == "t0", True, 1, 0),
                        tmp_path, "low", log=lambda s: None)
    text = summarize(rows)
    assert "ALL" in text and "org/a" in text
    assert "33.3%" in text
    assert "stop reasons: submitted=3" in text
