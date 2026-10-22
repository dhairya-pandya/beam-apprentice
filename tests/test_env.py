import json
from pathlib import Path

from apprentice.env import load_environment
from apprentice.episode import Episode
from apprentice.verify import TaskReference, VerifyResult, save_references

FIXTURE = Path(__file__).parent / "fixtures" / "tasks_small.jsonl"


def _setup(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "tasks.jsonl").write_text(FIXTURE.read_text())
    refs = [TaskReference(i, ["t::a"], []) for i in ("alpha_1", "alpha_2", "beta_1")]
    refs.append(TaskReference("beta_2", [], ["t::b"]))  # unusable: nothing flips
    save_references(refs, tmp_path / "refs.jsonl")


def test_load_environment_lists_split(tmp_path):
    _setup(tmp_path)
    env = load_environment(str(tmp_path / "data"), split="all", references=str(tmp_path / "refs.jsonl"))
    assert sorted(d["instance_id"] for d in env.dataset) == ["alpha_1", "alpha_2", "beta_1"]
    held = load_environment(str(tmp_path / "data"), split="held_out", references=str(tmp_path / "refs.jsonl"))
    train = load_environment(str(tmp_path / "data"), split="train", references=str(tmp_path / "refs.jsonl"))
    assert {d["instance_id"] for d in held.dataset}.isdisjoint({d["instance_id"] for d in train.dataset})


def test_reward_from_verify_result(tmp_path):
    _setup(tmp_path)
    env = load_environment(
        str(tmp_path / "data"), split="all", references=str(tmp_path / "refs.jsonl"),
        verify_fn=lambda task, patch, root, ref: VerifyResult(patch == "good", True, 1, 0),
    )
    mk = lambda patch: Episode("alpha_1", "low", [], patch, "submitted")
    assert env.reward(mk("good")) == 1.0
    assert env.reward(mk("bad")) == 0.0
    assert env.reward(mk("")) == 0.0
