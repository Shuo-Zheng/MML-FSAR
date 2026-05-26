"""Configuration loading helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_EXPERIMENT_REQUIRED_KEYS = {
    "experiment",
    "dataset",
    "output_dir",
    "episode",
    "optimization",
    "model",
}
_SUBSTITUTION_PATTERN = re.compile(r"\$\{([^}]+)\}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file {config_path} must contain a mapping.")
    return data


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load a self-contained experiment config."""

    config_path = Path(path).resolve()
    config = load_yaml(config_path)
    missing = sorted(_EXPERIMENT_REQUIRED_KEYS - set(config))
    if missing:
        raise ValueError(f"Missing required experiment config keys: {', '.join(missing)}")

    project_root = _project_root_for_config(config_path)
    return _resolve_substitutions(config, context={"project_root": str(project_root)})


def _project_root_for_config(config_path: Path) -> Path:
    for parent in config_path.parents:
        if parent.name == "configs":
            return parent.parent
    return config_path.parent


def _resolve_substitutions(
    data: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = dict(data)
    for _ in range(8):
        value_context = {**(context or {}), **resolved}
        next_resolved = {
            key: _resolve_value(value, value_context)
            for key, value in resolved.items()
        }
        if next_resolved == resolved:
            break
        resolved = next_resolved
    return resolved


def _resolve_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return _SUBSTITUTION_PATTERN.sub(
            lambda match: _substitution_value(match.group(1), context),
            value,
        )
    if isinstance(value, dict):
        return _resolve_substitutions(value, context={**context, **value})
    if isinstance(value, list):
        return [_resolve_value(item, context) for item in value]
    return value


def _substitution_value(name: str, context: dict[str, Any]) -> str:
    if name in context:
        return str(context[name])
    if name in os.environ:
        return os.environ[name]
    return "${" + name + "}"
