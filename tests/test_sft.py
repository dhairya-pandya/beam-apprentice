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


def _label(iid, history, tool_name="edit_file"):
    return {"instance_id": iid, "repo": "org/a", "step": 1, "history": history, "executed": "student",
            "teacher": {"role": "assistant", "content": "", "reasoning_content": "t",
                        "tool_calls": [{"id": "x", "type": "function", "function": {"name": tool_name, "arguments": "{}"}}]},
            "student": {"role": "assistant", "content": ""}, "tools": []}


def test_export_dagger_keeps_states_ending_on_tool_or_prompt(tmp_path):
    from apprentice.sft import export_dagger
    base = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    after_tool = base + [{"role": "assistant", "content": "", "tool_calls": []}, {"role": "tool", "tool_call_id": "c", "content": "out"}]
    after_nudge = after_tool + [{"role": "user", "content": "keep going"}]
    labels = tmp_path / "dagger.jsonl"
    rows = [_label("a", base), _label("a", after_tool), _label("a", after_nudge), _label("held", base)]
    labels.write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = export_dagger([labels], tmp_path / "out.jsonl", held_out_ids={"held"})
    out = [json.loads(l) for l in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert counts == {"examples": 2, "skipped_nudge_state": 1, "held_out": 1}
    assert all(r["train_last_step"] and r["messages"][-1]["reasoning_content"] == "t" for r in out)
    assert out[1]["messages"][:-1] == after_tool
