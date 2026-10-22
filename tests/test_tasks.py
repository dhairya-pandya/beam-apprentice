from pathlib import Path

from apprentice.tasks import Task, load_tasks, split_tasks, target_test_files

FIXTURE = Path(__file__).parent / "fixtures" / "tasks_small.jsonl"


def test_load_tasks_reads_all_fields():
    tasks = load_tasks(FIXTURE)
    assert len(tasks) == 4
    t = tasks[1]
    assert isinstance(t, Task)
    assert t.instance_id == "alpha_2"
    assert t.repo == "org/alpha"
    assert t.base_commit == "aaa2"
    assert t.hints_text == "look at util"
    assert t.problem_statement == "y should be 3"
    assert "+y = 3" in t.patch
    assert "tests/test_util.py" in t.test_patch


def test_split_is_deterministic_and_disjoint():
    tasks = load_tasks(FIXTURE)
    train_a, held_a = split_tasks(tasks, held_out_frac=0.5, seed=7)
    train_b, held_b = split_tasks(tasks, held_out_frac=0.5, seed=7)
    assert [t.instance_id for t in train_a] == [t.instance_id for t in train_b]
    assert [t.instance_id for t in held_a] == [t.instance_id for t in held_b]
    train_ids = {t.instance_id for t in train_a}
    held_ids = {t.instance_id for t in held_a}
    assert not train_ids & held_ids
    assert train_ids | held_ids == {t.instance_id for t in tasks}


def test_split_keeps_each_repo_in_held_out():
    tasks = load_tasks(FIXTURE)
    train, held = split_tasks(tasks, held_out_frac=0.1, seed=0)
    assert {t.repo for t in held} == {"org/alpha", "org/beta"}
    assert {t.repo for t in train} == {"org/alpha", "org/beta"}


def test_test_files_parses_diff_headers():
    tasks = {t.instance_id: t for t in load_tasks(FIXTURE)}
    assert target_test_files(tasks["alpha_1"]) == ["tests/test_core.py"]
    assert target_test_files(tasks["alpha_2"]) == ["tests/test_util.py"]
    assert target_test_files(tasks["beta_1"]) == ["tests/unit/test_a.py", "tests/unit/test_b.py"]
    assert target_test_files(tasks["beta_2"]) == []
