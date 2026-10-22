"""One throwaway Docker container per task, built from a repo snapshot, with no network."""

from __future__ import annotations

import base64
import shlex
import subprocess
import tarfile
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from apprentice.tasks import Task

GIT_EXCLUDE = "\n".join(
    ["__pycache__/", "*.pyc", ".pytest_cache/", "*.egg-info/", "build/", "dist/", ".coverage"]
)
DOCKERFILE = Path(__file__).resolve().parents[2] / "docker" / "Dockerfile"


@dataclass
class ExecResult:
    exit_code: int
    output: str
    timed_out: bool = False


def image_name(repo: str) -> str:
    return "apprentice-sandbox:" + repo.lower().replace("/", "-")


def snapshot_path(task: Task, data_root: Path) -> Path:
    return Path(data_root) / "snapshots" / f"{task.instance_id}.tgz"


def _docker(*args: str, input: bytes | None = None, timeout: float | None = None):
    return subprocess.run(["docker", *args], input=input, capture_output=True, timeout=timeout)


def build_image(repo: str, seed_snapshot: Path, quiet: bool = True) -> str:
    """Build the per-repo image, installing deps from one snapshot of that repo (needs network)."""
    tag = image_name(repo)
    with tempfile.TemporaryDirectory() as ctx:
        seed = Path(ctx) / "seed"
        seed.mkdir()
        with tarfile.open(seed_snapshot) as tar:
            tar.extractall(seed, filter="data")
        (Path(ctx) / "Dockerfile").write_text(DOCKERFILE.read_text())
        pins = DOCKERFILE.parent / "pins" / (repo.lower().replace("/", "-") + ".txt")
        (Path(ctx) / "pins.txt").write_text(pins.read_text() if pins.exists() else "")
        cmd = ["docker", "build", "-t", tag, ctx] + (["-q"] if quiet else [])
        subprocess.run(cmd, check=True, capture_output=quiet)
    return tag


def image_exists(repo: str) -> bool:
    return _docker("image", "inspect", image_name(repo)).returncode == 0


class Sandbox:
    """Context manager: fresh container, snapshot at /workspace, baseline commit."""

    def __init__(self, task: Task, data_root: Path, memory: str = "4g", cpus: str = "2"):
        self.task = task
        self.data_root = Path(data_root)
        self.memory = memory
        self.cpus = cpus
        self.container: str | None = None

    def __enter__(self) -> "Sandbox":
        snap = snapshot_path(self.task, self.data_root)
        if not snap.exists():
            raise FileNotFoundError(f"missing snapshot {snap}")
        name = f"apprentice-{self.task.instance_id}-{uuid.uuid4().hex[:8]}".lower()
        r = _docker(
            "run", "-d", "--rm", "--name", name, "--network", "none",
            "--memory", self.memory, "--cpus", self.cpus,
            image_name(self.task.repo), "sleep", "infinity",
        )
        if r.returncode != 0:
            raise RuntimeError(f"docker run failed: {r.stderr.decode()[-500:]}")
        self.container = name
        try:
            self._setup(snap)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def _setup(self, snap: Path) -> None:
        r = _docker("cp", str(snap), f"{self.container}:/tmp/snapshot.tgz")
        if r.returncode != 0:
            raise RuntimeError(f"docker cp failed: {r.stderr.decode()[-500:]}")
        steps = [
            "mkdir -p /workspace && tar xzf /tmp/snapshot.tgz -C /workspace && rm /tmp/snapshot.tgz",
            "find /workspace -xtype l -delete",
            f"printf %s {shlex.quote(GIT_EXCLUDE)} >> /workspace/.git/info/exclude",
            "pip install -q --no-index --no-deps --no-build-isolation -e /workspace >/dev/null 2>&1 || true",
            "git add -A && git commit -q --allow-empty -m baseline",
        ]
        for step in steps:
            res = self.exec(step, timeout=600)
            if res.exit_code != 0:
                raise RuntimeError(f"setup step failed: {step}\n{res.output[-1000:]}")

    def __exit__(self, *exc) -> None:
        if self.container:
            _docker("rm", "-f", self.container)
            self.container = None

    def exec(self, cmd: str, timeout: int = 300) -> ExecResult:
        """Run a bash command in /workspace; on timeout the command's process group is killed."""
        assert self.container, "sandbox not started"
        wrapped = f"timeout -k 5 {int(timeout)} bash -c {shlex.quote(cmd)}"
        try:
            r = _docker("exec", "-w", "/workspace", self.container, "bash", "-c", wrapped,
                        timeout=timeout + 30)
        except subprocess.TimeoutExpired:
            return ExecResult(124, "", timed_out=True)
        out = (r.stdout + r.stderr).decode("utf-8", errors="replace")
        return ExecResult(r.returncode, out, timed_out=r.returncode in (124, 137))

    def read(self, path: str) -> str:
        r = self.exec(f"cat -- {shlex.quote(path)}")
        if r.exit_code != 0:
            raise FileNotFoundError(path)
        return r.output

    def write(self, path: str, content: str) -> None:
        b64 = base64.b64encode(content.encode()).decode()
        target = shlex.quote(path)
        r = _docker(
            "exec", "-i", "-w", "/workspace", self.container, "bash", "-c",
            f"mkdir -p \"$(dirname {target})\" && base64 -d > {target}",
            input=b64.encode(),
        )
        if r.returncode != 0:
            raise OSError(r.stderr.decode()[-500:])

    def diff(self) -> str:
        r = self.exec("git add -N . && git diff --binary HEAD")
        return r.output if r.exit_code == 0 else ""
