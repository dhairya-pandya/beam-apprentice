"""End-to-end verification against real snapshots. Run with: pytest -m docker"""

import os
import shutil
from pathlib import Path

import pytest

from apprentice.sandbox import image_exists, snapshot_path
from apprentice.tasks import load_tasks
from apprentice.verify import build_reference, verify

DATA = Path(os.environ.get("APPRENTICE_DATA", "external/swe-tasks"))
TASK_IDS = os.environ.get("APPRENTICE_TEST_TASKS", "requests_6592,rich_3061,fastapi_14077").split(",")

pytestmark = pytest.mark.docker


def _tasks():
    if not shutil.which("docker") or not (DATA / "tasks.jsonl").exists():
        return []
    by_id = {t.instance_id: t for t in load_tasks(DATA / "tasks.jsonl")}
    return [
        by_id[i] for i in TASK_IDS
        if i in by_id and snapshot_path(by_id[i], DATA).exists() and image_exists(by_id[i].repo)
    ]


@pytest.fixture(scope="module", params=_tasks() or [None], ids=lambda t: t.instance_id if t else "none")
def task_and_ref(request):
    if request.param is None:
        pytest.skip("docker, task data, or sandbox image missing")
    return request.param, build_reference(request.param, DATA)


def test_reference_has_fail_to_pass(task_and_ref):
    _, ref = task_and_ref
    assert ref.fail_to_pass


def test_gold_patch_resolves(task_and_ref):
    task, ref = task_and_ref
    assert verify(task, task.patch, DATA, ref).resolved is True


def test_empty_patch_fails(task_and_ref):
    task, ref = task_and_ref
    assert verify(task, "", DATA, ref).resolved is False


def test_test_edits_are_ignored(task_and_ref):
    task, ref = task_and_ref
    assert verify(task, task.test_patch, DATA, ref).resolved is False
