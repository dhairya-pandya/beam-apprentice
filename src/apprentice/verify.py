"""Hidden-test verification in a fresh sandbox.

A task's reference is computed once from its known-good fix: tests that fail without the fix
and pass with it (fail_to_pass), plus tests that pass either way (pass_to_pass). A candidate
patch resolves the task when every reference test passes. This tolerates test files that
contain tests which can never pass offline, while still requiring the bug to be fixed.
"""

from __future__ import annotations

import json
import shlex
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

from apprentice.patches import is_protected, strip_protected
from apprentice.sandbox import Sandbox
from apprentice.tasks import Task, target_test_files

PYTEST_TIMEOUT = 300
JUNIT = "/tmp/_apprentice_junit.xml"


@dataclass
class TestRun:
    applied: bool
    outcomes: dict[str, str]  # test id -> passed | failed | error | skipped
    exit_code: int
    log: str


@dataclass
class TaskReference:
    instance_id: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]

    @property
    def usable(self) -> bool:
        return bool(self.fail_to_pass)


@dataclass
class VerifyResult:
    resolved: bool
    applied: bool
    tests_passed: int
    tests_failed: int
    missing: list[str] = field(default_factory=list)
    log: str = ""


def _changed_paths(patch: str) -> list[str]:
    paths = []
    for line in patch.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            p = line[4:].strip()
            if p != "/dev/null":
                p = p[2:] if p[:2] in ("a/", "b/") else p
                if p not in paths:
                    paths.append(p)
    return paths


def _apply(sb: Sandbox, patch: str, path: str = "/tmp/candidate.patch") -> bool:
    sb.write(path, patch if patch.endswith("\n") else patch + "\n")
    for cmd in (
        f"git apply --whitespace=nowarn {path}",
        f"git apply --whitespace=nowarn -3 {path}",
        f"git apply --whitespace=nowarn --ignore-space-change --ignore-whitespace {path}",
        f"git apply --whitespace=nowarn --recount {path}",
        f"patch -p1 --batch --forward --dry-run < {path} && patch -p1 --batch --forward < {path}",
    ):
        if sb.exec(cmd, timeout=120).exit_code == 0:
            return True
    return False


def parse_junit(xml_text: str) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    root = ET.fromstring(xml_text)
    for case in root.iter("testcase"):
        tid = f"{case.get('classname', '')}::{case.get('name', '')}"
        status = "passed"
        for child in case:
            if child.tag in ("failure", "error", "skipped"):
                status = {"failure": "failed"}.get(child.tag, child.tag)
                break
        outcomes[tid] = status
    return outcomes


def run_tests(task: Task, patch: str, data_root: Path) -> TestRun:
    """Fresh sandbox: apply patch (minus protected files), reset tests, add hidden tests, run them."""
    with Sandbox(task, data_root) as sb:
        clean = strip_protected(patch)
        applied = True
        if clean.strip():
            applied = _apply(sb, clean)
            if not applied:
                return TestRun(False, {}, -1, "failed to apply patch")

        targets = target_test_files(task)
        reset = sorted(set(targets) | {p for p in _changed_paths(patch) if is_protected(p)})
        if reset:
            quoted = " ".join(shlex.quote(p) for p in reset)
            sb.exec(f"git checkout HEAD -- {quoted} 2>/dev/null; git clean -fq -- {quoted}")
        if task.test_patch.strip() and not _apply(sb, task.test_patch, "/tmp/hidden_tests.patch"):
            return TestRun(applied, {}, -1, "failed to apply hidden tests")

        py_targets = [t for t in targets if t.endswith(".py")]
        if not py_targets:
            return TestRun(applied, {}, -1, "no python test targets")
        quoted = " ".join(shlex.quote(t) for t in py_targets)
        res = sb.exec(
            f"python -m pytest {quoted} -q -p no:cacheprovider -o addopts='' "
            f"--junitxml={JUNIT} --timeout={PYTEST_TIMEOUT}",
            timeout=PYTEST_TIMEOUT + 60,
        )
        xml = sb.exec(f"cat {JUNIT}")
        outcomes = parse_junit(xml.output) if xml.exit_code == 0 and xml.output.strip() else {}
        return TestRun(applied, outcomes, res.exit_code, res.output[-20000:])


def build_reference(task: Task, data_root: Path) -> TaskReference:
    gold = run_tests(task, task.patch, data_root)
    empty = run_tests(task, "", data_root)
    passed_gold = {t for t, s in gold.outcomes.items() if s == "passed"}
    passed_empty = {t for t, s in empty.outcomes.items() if s == "passed"}
    return TaskReference(
        task.instance_id,
        fail_to_pass=sorted(passed_gold - passed_empty),
        pass_to_pass=sorted(passed_gold & passed_empty),
    )


def verify(task: Task, patch: str, data_root: Path, ref: TaskReference) -> VerifyResult:
    if not strip_protected(patch).strip():
        return VerifyResult(False, False, 0, 0, log="empty patch after stripping protected files")
    run = run_tests(task, patch, data_root)
    required = ref.fail_to_pass + ref.pass_to_pass
    missing = [t for t in required if run.outcomes.get(t) != "passed"]
    passed = sum(1 for s in run.outcomes.values() if s == "passed")
    failed = sum(1 for s in run.outcomes.values() if s in ("failed", "error"))
    return VerifyResult(
        resolved=run.applied and ref.usable and not missing,
        applied=run.applied,
        tests_passed=passed,
        tests_failed=failed,
        missing=missing,
        log=run.log,
    )


def save_references(refs: list[TaskReference], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in refs:
            f.write(json.dumps(asdict(r)) + "\n")


def load_references(path: Path) -> dict[str, TaskReference]:
    if not Path(path).exists():
        return {}
    out = {}
    for line in open(path):
        d = json.loads(line)
        out[d["instance_id"]] = TaskReference(**d)
    return out
