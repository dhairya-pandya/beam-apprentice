"""apprentice CLI: build sandbox images, run a model over tasks, summarize results."""

from __future__ import annotations

import os
from pathlib import Path

import typer

from apprentice.beam_client import BEAM_MODEL, HF_ROUTER, BeamClient
from apprentice.episode import run_episode
from apprentice.ledger import CostLedger, Prices
from apprentice.runner import run_baseline, summarize
from apprentice.sandbox import build_image, image_exists, snapshot_path
from apprentice.tasks import load_tasks, split_tasks
from apprentice.tools import Limits
from apprentice.traces import read_results
from apprentice.verify import load_references, verify

app = typer.Typer(no_args_is_help=True, add_completion=False)
DEFAULT_DATA = os.environ.get("APPRENTICE_DATA", "external/swe-tasks")


def _usable_tasks(data_root: Path, runs: Path, split: str, seed: int):
    refs = load_references(runs / "references.jsonl")
    tasks = [t for t in load_tasks(data_root / "tasks.jsonl") if t.instance_id in refs and refs[t.instance_id].usable]
    train, held = split_tasks(tasks, seed=seed)
    chosen = {"train": train, "held_out": held, "all": tasks}[split]
    return chosen, refs


@app.command("build-images")
def build_images(data_root: Path = typer.Option(DEFAULT_DATA), force: bool = False):
    """Build one sandbox image per repository (needs network)."""
    seen = {}
    for t in load_tasks(data_root / "tasks.jsonl"):
        if snapshot_path(t, data_root).exists():
            seen.setdefault(t.repo, snapshot_path(t, data_root))
    for repo, snap in sorted(seen.items()):
        if image_exists(repo) and not force:
            typer.echo(f"{repo}: exists")
            continue
        typer.echo(f"{repo}: building from {snap.name}")
        build_image(repo, snap)


@app.command()
def baseline(
    data_root: Path = typer.Option(DEFAULT_DATA),
    runs: Path = typer.Option(Path("runs")),
    run_id: str = typer.Option(..., help="results go to runs/<run_id>/"),
    split: str = typer.Option("train", help="train | held_out | all"),
    effort: str = typer.Option("low", help="low | medium | high | xhigh | max, or none for models without one"),
    limit: int = typer.Option(0, help="max tasks this invocation (0 = all)"),
    budget_usd: float = typer.Option(2.0, help="spend cap for this invocation"),
    lifetime_cap_usd: float = typer.Option(25.0, help="cap across all runs in the ledger"),
    max_tokens: int = typer.Option(8192),
    turns: int = typer.Option(40),
    tool_calls: int = typer.Option(60),
    seed: int = typer.Option(0),
    model: str = typer.Option(BEAM_MODEL),
    base_url: str = typer.Option(HF_ROUTER),
    api_key: str = typer.Option("", help="for local servers; default uses your Hugging Face login"),
    free: bool = typer.Option(False, help="price calls at $0 (local model)"),
    workers: int = typer.Option(1, help="episodes to run in parallel"),
):
    """Run a model through the gym on a task split, verify each patch, and record traces."""
    tasks, refs = _usable_tasks(data_root, runs, split, seed)
    tasks = [t for t in tasks if snapshot_path(t, data_root).exists()]
    if limit:
        tasks = tasks[:limit]
    prices = Prices(0, 0, 0) if free else Prices()
    ledger = CostLedger(runs / "ledger.jsonl", cap_usd=lifetime_cap_usd, session_cap_usd=budget_usd, prices=prices)
    typer.echo(f"lifetime spend so far ${ledger.spent():.4f} / ${lifetime_cap_usd:.2f}; this run capped at ${budget_usd:.2f}")
    client = BeamClient.from_env(ledger, base_url=base_url, model=model, api_key=api_key or None, meta={"run": run_id})
    limits = Limits(turns=turns, tool_calls=tool_calls)
    out_dir = runs / run_id
    run_baseline(
        tasks,
        episode_fn=lambda t: run_episode(t, client, effort, limits, data_root, max_tokens=max_tokens),
        verify_fn=lambda t, p: verify(t, p, data_root, refs[t.instance_id]),
        out_dir=out_dir,
        effort=effort,
        log=typer.echo,
        workers=workers,
    )
    typer.echo(summarize(read_results(out_dir)))
    typer.echo(f"this run spent ${ledger.session_spent():.4f}; lifetime ${ledger.spent():.4f}")


@app.command("export-sft")
def export_sft_cmd(
    run_ids: list[str] = typer.Argument(..., help="runs to draw verified episodes from"),
    out: Path = typer.Option(Path("data/sft.jsonl")),
    data_root: Path = typer.Option(DEFAULT_DATA),
    runs: Path = typer.Option(Path("runs")),
    seed: int = typer.Option(0),
    max_tool_chars: int = typer.Option(3000),
    one_per_task: bool = typer.Option(False),
):
    """Export resolved training-split episodes as fine-tuning data (held-out tasks are never included)."""
    from apprentice.sft import export_sft

    held, _ = _usable_tasks(data_root, runs, "held_out", seed)
    counts = export_sft([runs / r for r in run_ids], out, {t.instance_id for t in held}, max_tool_chars, one_per_task)
    typer.echo(f"{out}: {counts}")


@app.command("summarize")
def summarize_cmd(run_id: str, runs: Path = typer.Option(Path("runs"))):
    """Print resolve rate, cost and stop reasons for a run."""
    typer.echo(summarize(read_results(runs / run_id)))


if __name__ == "__main__":
    app()
