"""Checkpoint save and load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_checkpoint_path(config: dict[str, Any], checkpoint: str | Path | None) -> Path:
    """Resolve an explicit checkpoint or the default checkpoint under output_dir."""

    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        return checkpoint_path

    output_dir = Path(str(config["output_dir"]))
    for checkpoint_name in ("best_checkpoint.pt", "best_checkpoint"):
        default_checkpoint = output_dir / checkpoint_name
        if default_checkpoint.exists():
            return default_checkpoint
    latest_run_checkpoint = _latest_run_checkpoint(output_dir)
    if latest_run_checkpoint is not None:
        return latest_run_checkpoint

    raise FileNotFoundError(
        f"No checkpoint found under {output_dir}; run training or pass --checkpoint."
    )


def _latest_run_checkpoint(output_dir: Path) -> Path | None:
    runs_dir = output_dir / "runs"
    if not runs_dir.is_dir():
        return None
    run_dirs = sorted(
        (path for path in runs_dir.iterdir() if path.is_dir()),
        reverse=True,
    )
    for run_dir in run_dirs:
        for checkpoint_name in ("best_checkpoint.pt", "best_checkpoint"):
            checkpoint_path = run_dir / checkpoint_name
            if checkpoint_path.exists():
                return checkpoint_path
    return None
