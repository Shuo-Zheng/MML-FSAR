"""Task dictionary loading and validation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REQUIRED_EPISODE_KEYS = {"ss", "sd", "qs", "qd", "d2c"}


def load_json_metadata(path: str | Path) -> Any:
    """Load a JSON metadata file from disk."""

    metadata_path = Path(path)
    if metadata_path.suffix.lower() != ".json":
        raise ValueError(f"Only JSON metadata files are supported: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_task_dict(path: str | Path) -> dict[Any, dict[str, Any]]:
    """Load and validate a JSON episodic task dictionary."""

    task_dict = load_json_metadata(path)
    if not isinstance(task_dict, dict):
        raise ValueError(f"Task dictionary {path} must contain a mapping.")
    for task_id, episode in task_dict.items():
        if not isinstance(episode, dict):
            raise ValueError(f"Episode {task_id!r} must contain a mapping.")
        missing = sorted(_REQUIRED_EPISODE_KEYS - set(episode))
        if missing:
            raise ValueError(f"Episode {task_id!r} missing required keys: {', '.join(missing)}")
        task_dict[task_id] = _normalize_episode(episode)
    return task_dict


def load_length_dict(path: str | Path) -> Any:
    """Load a video length dictionary.

    The current training code only needs this object for compatibility with the
    original loader, so the exact structure is intentionally not constrained.
    """

    length_dict = load_json_metadata(path)
    if isinstance(length_dict, dict):
        return {str(key): int(value) for key, value in length_dict.items()}
    return length_dict


def _normalize_episode(episode: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(episode)
    normalized["sd"] = [int(label) for label in episode["sd"]]
    normalized["qd"] = [int(label) for label in episode["qd"]]
    normalized["d2c"] = {
        int(digit_label): int(class_label)
        for digit_label, class_label in episode["d2c"].items()
    }
    return normalized
