"""Patch hygiene: an agent may not pass by editing tests or test-runner config."""

from __future__ import annotations

import fnmatch
import re
from pathlib import PurePosixPath

_PROTECTED_NAMES = {
    "conftest.py", "pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg",
    ".pytest.ini", "sitecustomize.py", "usercustomize.py",
}
_TEST_DIRS = {"tests", "test", "testing"}


def is_protected(path: str) -> bool:
    p = PurePosixPath(path)
    name = p.name
    if name in _PROTECTED_NAMES or name.endswith(".pth"):
        return True
    if name.endswith(".py"):
        if fnmatch.fnmatch(name, "test_*.py") or fnmatch.fnmatch(name, "*_test.py"):
            return True
        if _TEST_DIRS & set(p.parts[:-1]):
            return True
    return False


_FILE_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)


def _split_files(patch: str) -> list[str]:
    starts = [m.start() for m in _FILE_HEADER.finditer(patch)]
    if not starts:
        return [patch] if patch else []
    chunks = []
    if starts[0] > 0:
        chunks.append(patch[: starts[0]])
    for i, s in enumerate(starts):
        chunks.append(patch[s : starts[i + 1] if i + 1 < len(starts) else len(patch)])
    return chunks


def _chunk_paths(chunk: str) -> list[str]:
    m = _FILE_HEADER.match(chunk)
    if m:
        return [m.group(1), m.group(2)]
    return re.findall(r"^(?:\+\+\+|---) (?:[ab]/)?(\S+)", chunk, re.MULTILINE)


def strip_protected(patch: str) -> str:
    """Drop every per-file section of a git diff that touches a protected path."""
    kept = []
    for chunk in _split_files(patch):
        paths = [p for p in _chunk_paths(chunk) if p != "/dev/null"]
        if paths and any(is_protected(p) for p in paths):
            continue
        kept.append(chunk)
    return "".join(kept)
