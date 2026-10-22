"""A local stand-in for the Docker sandbox: a temp git repo driven by subprocess."""

import subprocess

import pytest

from apprentice.sandbox import ExecResult


class LocalSandbox:
    def __init__(self, root):
        self.root = root
        (root / "pkg").mkdir()
        (root / "pkg" / "core.py").write_text("def add(a, b):\n    return a - b\n")
        (root / "tests").mkdir()
        (root / "tests" / "test_core.py").write_text("from pkg.core import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
        for cmd in ("git init -q -b main", "git add -A", "git -c user.email=a@b -c user.name=a commit -qm baseline"):
            subprocess.run(cmd, shell=True, cwd=root, check=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def exec(self, cmd, timeout=300):
        try:
            r = subprocess.run(["bash", "-c", cmd], cwd=self.root, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return ExecResult(124, "", timed_out=True)
        return ExecResult(r.returncode, (r.stdout + r.stderr).decode(errors="replace"))

    def read(self, path):
        p = self.root / path
        if not p.is_file():
            raise FileNotFoundError(path)
        return p.read_text()

    def write(self, path, content):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def diff(self):
        return self.exec("git add -N . && git diff --binary HEAD").output


@pytest.fixture
def local_sandbox(tmp_path):
    return LocalSandbox(tmp_path)
