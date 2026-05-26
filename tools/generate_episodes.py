#!/usr/bin/env python3
"""Generate few-shot episode JSON files from split metadata."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate few-shot episode JSON files from splits/*/*_list.json."
    )
    parser.add_argument("--split-json", type=Path, required=True, help="Input split list JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Output episode JSON file.")
    parser.add_argument("--way", type=int, default=5, help="Number of classes per episode.")
    parser.add_argument("--shot", type=int, default=1, help="Support samples per class.")
    parser.add_argument(
        "--num-query",
        "--n-query",
        dest="num_query",
        type=int,
        required=True,
        help="Total query samples per episode.",
    )
    parser.add_argument(
        "--num-episodes",
        "--num",
        dest="num_episodes",
        type=int,
        default=10000,
        help="Number of unique episodes to generate.",
    )
    parser.add_argument("--seed", type=int, required=True, help="Random seed.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum sampling attempts before failing. Defaults to num_episodes * 100.",
    )
    parser.add_argument("--print-example", action="store_true", help="Print the first episode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_to_videos = load_split_list(args.split_json)
    episodes = generate_episodes(
        class_to_videos=class_to_videos,
        way=args.way,
        shot=args.shot,
        num_query=args.num_query,
        num_episodes=args.num_episodes,
        seed=args.seed,
        max_attempts=args.max_attempts,
    )
    write_episode_json(episodes, args.output)
    if args.print_example:
        first_key = next(iter(episodes))
        print(json.dumps({first_key: episodes[first_key]}, indent=2, sort_keys=True))


def load_split_list(path: Path) -> dict[int, list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a class-to-video-list mapping.")

    class_to_videos: dict[int, list[str]] = {}
    for raw_class_id, raw_videos in raw.items():
        class_id = int(raw_class_id)
        if not isinstance(raw_videos, list) or not all(
            isinstance(video, str) for video in raw_videos
        ):
            raise ValueError(f"{path} class {raw_class_id!r} must map to a list of strings.")
        class_to_videos[class_id] = list(raw_videos)
    return class_to_videos


def generate_episodes(
    class_to_videos: dict[int, list[str]],
    way: int,
    shot: int,
    num_query: int,
    num_episodes: int,
    seed: int,
    max_attempts: int | None = None,
) -> dict[str, dict[str, Any]]:
    validate_generation_args(class_to_videos, way, shot, num_query, num_episodes)
    rng = random.Random(seed)
    classes = sorted(class_to_videos)
    sample_to_id, sample_to_class = build_sample_indices(class_to_videos)
    episodes: dict[str, dict[str, Any]] = {}
    attempts = 0
    attempt_limit = max_attempts if max_attempts is not None else max(num_episodes * 100, 1000)

    while len(episodes) < num_episodes and attempts < attempt_limit:
        attempts += 1
        selected_classes = rng.sample(classes, way)
        rng.shuffle(selected_classes)
        episode = sample_episode(
            class_to_videos=class_to_videos,
            selected_classes=selected_classes,
            shot=shot,
            num_query=num_query,
            sample_to_class=sample_to_class,
            rng=rng,
        )
        if episode is None:
            continue
        task_id = make_task_id(episode["ss"], episode["qs"], sample_to_id)
        if task_id not in episodes:
            episodes[task_id] = episode

    if len(episodes) != num_episodes:
        raise RuntimeError(
            f"Generated {len(episodes)} unique episodes after {attempts} attempts; "
            f"requested {num_episodes}."
        )
    return episodes


def validate_generation_args(
    class_to_videos: dict[int, list[str]],
    way: int,
    shot: int,
    num_query: int,
    num_episodes: int,
) -> None:
    if way <= 0 or shot <= 0 or num_query <= 0 or num_episodes <= 0:
        raise ValueError("way, shot, num_query, and num_episodes must be positive.")
    if len(class_to_videos) < way:
        raise ValueError(f"Need at least {way} classes, got {len(class_to_videos)}.")
    small_classes = [
        class_id for class_id, videos in class_to_videos.items() if len(videos) < shot
    ]
    if small_classes:
        preview = ", ".join(str(class_id) for class_id in small_classes[:5])
        raise ValueError(f"Classes with fewer than {shot} support samples: {preview}.")


def build_sample_indices(
    class_to_videos: dict[int, list[str]],
) -> tuple[dict[str, int], dict[str, int]]:
    sample_to_id: dict[str, int] = {}
    sample_to_class: dict[str, int] = {}
    for sample in (
        video
        for class_id in sorted(class_to_videos)
        for video in class_to_videos[class_id]
    ):
        if sample in sample_to_id:
            raise ValueError(f"Duplicate video path in split metadata: {sample}")
        sample_to_id[sample] = len(sample_to_id)
    for class_id in sorted(class_to_videos):
        for sample in class_to_videos[class_id]:
            sample_to_class[sample] = class_id
    return sample_to_id, sample_to_class


def sample_episode(
    class_to_videos: dict[int, list[str]],
    selected_classes: list[int],
    shot: int,
    num_query: int,
    sample_to_class: dict[str, int],
    rng: random.Random,
) -> dict[str, Any] | None:
    support_samples: list[str] = []
    support_labels: list[int] = []
    query_pool: list[str] = []
    digit_to_class: dict[int, int] = {}
    class_to_digit: dict[int, int] = {}

    for digit, class_id in enumerate(selected_classes):
        class_samples = class_to_videos[class_id]
        support = rng.sample(class_samples, shot)
        support_samples.extend(support)
        support_labels.extend([digit] * shot)
        query_pool.extend(class_samples)
        digit_to_class[digit] = class_id
        class_to_digit[class_id] = digit

    support_set = set(support_samples)
    query_pool = [sample for sample in query_pool if sample not in support_set]
    if len(query_pool) < num_query:
        return None

    query_samples = rng.sample(query_pool, num_query)
    query_labels = [class_to_digit[sample_to_class[sample]] for sample in query_samples]
    return {
        "ss": support_samples,
        "qs": query_samples,
        "sd": support_labels,
        "qd": query_labels,
        "d2c": digit_to_class,
    }


def make_task_id(
    support_samples: list[str],
    query_samples: list[str],
    sample_to_id: dict[str, int],
) -> str:
    sample_ids = sorted(sample_to_id[sample] for sample in support_samples + query_samples)
    return "".join(f"{sample_id:05d}" for sample_id in sample_ids)


def write_episode_json(episodes: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(episodes, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
