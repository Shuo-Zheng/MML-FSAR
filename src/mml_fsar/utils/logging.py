"""Logging helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def create_timestamped_run_dir(
    output_dir: str | Path,
    timestamp: str | None = None,
) -> Path:
    """Create a unique timestamped run directory under ``output_dir/runs``."""

    base_dir = Path(output_dir) / "runs"
    base_dir.mkdir(parents=True, exist_ok=True)
    run_name = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / run_name
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"{run_name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir()
    return run_dir


class ExperimentLogger:
    """Small file-and-terminal logger for experiment runs."""

    def __init__(self, output_dir: str | Path, filename: str = "log.txt") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / filename
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.config_path = self.output_dir / "resolved_config.yaml"
        self._handle = self.log_path.open("w", encoding="utf-8", buffering=1)
        self.metrics_path.write_text("", encoding="utf-8")

    def print(self, message: str, print_to_terminal: bool = True) -> None:
        if print_to_terminal:
            print(message, flush=True)
        self._handle.write(message)
        self._handle.write("\n")

    def write_config(self, config: dict[str, Any]) -> None:
        with self.config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                _to_builtin(config),
                handle,
                allow_unicode=False,
                sort_keys=False,
            )

    def log_train(self, episode: int, loss: float, accuracy: float, lr: float) -> None:
        self.log_event(
            {
                "type": "train",
                "episode": episode,
                "loss": loss,
                "accuracy": accuracy,
                "lr": lr,
            }
        )

    def log_evaluation(
        self,
        split: str,
        episode: int,
        metrics: dict[str, Any],
    ) -> None:
        event = {"type": split, "episode": episode}
        event.update(metrics)
        self.log_event(event)

    def log_event(self, event: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            json.dump(_to_builtin(event), handle, sort_keys=True)
            handle.write("\n")

    def write_summary(self, summary: dict[str, Any]) -> None:
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(_to_builtin(summary), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
