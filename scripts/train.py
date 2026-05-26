#!/usr/bin/env python3
"""Training entry point for MML-FSAR experiments."""

from __future__ import annotations

import argparse
import json

from mml_fsar.engine.trainer import launch_training
from mml_fsar.utils.config import load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MML-FSAR experiment.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML file.")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    result = launch_training(config, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
