# How the sandbox works

Each task runs in its own Docker container, started from a per-repository image and thrown away afterwards. Containers run with `--network none`, 4 GB of memory and 2 CPUs.

## Images

There is one image per repository (`apprentice-sandbox:<owner>-<repo>`), built natively for the host, so Apple Silicon works without emulation. The build is the only step that touches the network. It installs the repo's runtime and test dependencies from one snapshot of that repo, adds the pins in `docker/pins/<owner>-<repo>.txt` (for example, `werkzeug<3` for requests, because httpbin breaks on newer werkzeug), and then deletes the seed checkout. `pytest-timeout` is pinned to 2.1.0 because some repos pin an old pytest.

```bash
uv run python -c "from pathlib import Path; from apprentice.sandbox import build_image; \
  build_image('psf/requests', Path('external/swe-tasks/snapshots/requests_6592.tgz'))"
```

## Per-task setup

1. The repo snapshot is copied in and extracted to `/workspace`.
2. Dependencies are re-resolved **offline** against the local `wheels/` folder, mounted read-only at `/wheels`. Commits from different months need different versions of their dependencies (fastapi's Starlette range, for example), and one image cannot hold all of them. If that fails, the package is installed with `--no-deps` on top of the image's environment.
3. Build and test artifacts are added to `.git/info/exclude`, and a `baseline` commit is made. The agent's patch is `git diff` against that commit.

## Verification

Verification never trusts the agent's container. It starts a fresh one, applies the patch minus any hunks touching tests or runner config, resets every hidden-test file, applies the hidden tests, and runs pytest on those files with JUnit output.

Each task's reference comes from its known-good fix:

- **fail_to_pass**: tests that fail without the fix and pass with it
- **pass_to_pass**: tests that pass either way

A patch resolves the task when every reference test passes. Tests that fail even with the known-good fix, usually because they need the network or a different platform, are left out instead of failing every attempt. A task with no fail_to_pass tests is unusable and is excluded (`scripts/gold_sweep.py` writes `runs/excluded.txt`).

Deprecation warnings are ignored during verification. Some repos turn every warning into an error, and newer dependency versions emit deprecation warnings that have nothing to do with the task. The cost is that tasks which exist only to silence a deprecation warning become unusable, and the gold sweep drops them.

## Timing (M-series Mac, Docker 29.5)

| step | time |
|---|---|
| image build (requests) | ~80 s |
| one verification run | ~10 s (requests) to ~40 s (fastapi) |

## Cost per episode (measured)

First live run at `effort=low` on 2 fastapi tasks: 22 to 25 turns, about $0.02 per episode. In one 22-turn episode, about 145k of the 164k input tokens were served from the provider's prompt cache. That brings the input price down from $0.40 to $0.04 per million tokens, which is why long multi-turn sessions stay cheap. At this rate, $25 covers on the order of a thousand episodes.
