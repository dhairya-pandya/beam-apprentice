import json

from apprentice.sft import export_sft


def _trace(d, iid, resolved, effort="low", tool_out="ok"):
    (d / "traces").mkdir(parents=True, exist_ok=True)
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "", "reasoning_content": "look first",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": tool_out},
    ]
    trace = {"repo": "org/a", "episode": {"instance_id": iid, "effort": effort, "turns": 2, "messages": msgs},
             "verify": {"resolved": resolved}}
    (d / "traces" / f"{iid}.{effort}.json").write_text(json.dumps(trace))


def test_only_resolved_training_episodes_are_exported(tmp_path):
    run = tmp_path / "run"
    _trace(run, "a", True)
    _trace(run, "b", False)
    _trace(run, "held", True)
    counts = export_sft([run], tmp_path / "sft.jsonl", held_out_ids={"held"})
    rows = [json.loads(l) for l in (tmp_path / "sft.jsonl").read_text().splitlines()]
    assert [r["instance_id"] for r in rows] == ["a"]
    assert counts == {"examples": 1, "tasks": 1, "unresolved": 1, "held_out": 1, "duplicate": 0}


def test_reasoning_and_tools_are_kept(tmp_path):
    run = tmp_path / "run"
    _trace(run, "a", True)
    export_sft([run], tmp_path / "sft.jsonl", held_out_ids=set())
    row = json.loads((tmp_path / "sft.jsonl").read_text())
    assert row["messages"][2]["reasoning_content"] == "look first"
    assert {t["function"]["name"] for t in row["tools"]} >= {"read_file", "submit_patch"}


def test_long_tool_output_is_clipped(tmp_path):
    run = tmp_path / "run"
    _trace(run, "a", True, tool_out="x" * 10000)
    export_sft([run], tmp_path / "sft.jsonl", held_out_ids=set(), max_tool_chars=300)
    tool = json.loads((tmp_path / "sft.jsonl").read_text())["messages"][3]["content"]
    assert len(tool) < 400 and "trimmed" in tool


def test_one_per_task(tmp_path):
    run = tmp_path / "run"
    _trace(run, "a", True, effort="low")
    _trace(run, "a", True, effort="medium")
    counts = export_sft([run], tmp_path / "sft.jsonl", held_out_ids=set(), one_per_task=True)
    assert counts["examples"] == 1 and counts["duplicate"] == 1
