"""Episodic dataset implementation for few-shot action recognition."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from mml_fsar.data.task_dict import load_length_dict, load_task_dict

try:
    import torch
except Exception:  # pragma: no cover - depends on the local smoke-test env
    torch = None
else:
    if not hasattr(torch, "as_tensor"):
        torch = None


class EpisodicVideoDataset:
    """Few-shot video episode dataset for frame directories or video files."""

    def __init__(
        self,
        task_dict: str | Path,
        length_dict: str | Path,
        num_frames: int,
        video_root: str | Path | None = None,
        start_iters: int = 0,
        is_trans: bool = True,
        seed: int = 1,
    ) -> None:
        self.task_dict = load_task_dict(task_dict)
        self.length_dict = load_length_dict(length_dict)
        self.task_ids = list(self.task_dict.keys())
        self.num_frames = int(num_frames)
        self.video_root = Path(video_root) if video_root is not None else None
        self.start_iters = int(start_iters)
        self.is_trans = bool(is_trans)
        self._rng = random.Random(seed)

        first_task = self.task_dict[self.task_ids[0]]
        self.num_way = len(first_task["d2c"])
        self.num_shot = int(len(first_task["ss"]) / self.num_way)

    def __len__(self) -> int:
        return max(len(self.task_ids) - self.start_iters, 0)

    def __getitem__(self, index: int) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        task_id = self.task_ids[index + self.start_iters]
        episode = self.task_dict[task_id]
        support_labels = episode["sd"]
        query_labels = episode["qd"]
        d2c = episode["d2c"]

        real_support_labels = [self._class_index(d2c, label) for label in support_labels]
        real_query_labels = [self._class_index(d2c, label) for label in query_labels]

        return (
            self.uniform_load_video(episode["ss"]),
            _as_tensor(support_labels),
            self.uniform_load_video(episode["qs"]),
            _as_tensor(query_labels),
            d2c,
            _as_tensor(real_support_labels),
            _as_tensor(real_query_labels),
        )

    def uniform_load_video(self, sample_paths: Sequence[str | Path]) -> list[Any]:
        return [self._load_uniform_video(path) for path in sample_paths]

    def _load_uniform_video(self, sample_path: str | Path) -> Any:
        path = self._resolve_sample_path(sample_path)
        if path.is_dir():
            return self._load_uniform_frame_directory(path)
        if path.is_file():
            return self._load_uniform_video_file(sample_path, path)
        raise FileNotFoundError(f"Video sample path does not exist: {path}")

    def _load_uniform_frame_directory(self, path: Path) -> Any:
        frame_paths = sorted(item for item in path.iterdir() if item.is_file())
        if len(frame_paths) < self.num_frames:
            raise ValueError(
                f"Video at {path} has {len(frame_paths)} frames, "
                f"but {self.num_frames} frames were requested."
            )

        frame_gap = len(frame_paths) // self.num_frames
        start_frame = self._rng.choice(range(len(frame_paths) % self.num_frames + 1))
        selected = [
            frame_paths[index * frame_gap + start_frame]
            for index in range(self.num_frames)
        ]
        frames = [_read_rgb_frame(frame_path) for frame_path in selected]
        return _stack(frames)

    def _load_uniform_video_file(self, sample_path: str | Path, path: Path) -> Any:
        cv2 = _import_cv2()
        video_length = self._video_length(sample_path, path)
        if video_length < self.num_frames:
            raise ValueError(
                f"Video at {path} has {video_length} frames, "
                f"but {self.num_frames} frames were requested."
            )

        frame_gap = video_length // self.num_frames
        start_frame = self._rng.choice(range(video_length % self.num_frames + 1))
        selected = {
            index * frame_gap + start_frame
            for index in range(self.num_frames)
        }
        capture = cv2.VideoCapture(str(path))
        frames = []
        frame_index = 0
        try:
            while len(frames) < self.num_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index in selected:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(_as_tensor(frame))
                frame_index += 1
        finally:
            capture.release()

        if len(frames) != self.num_frames:
            raise ValueError(
                f"Could not read {self.num_frames} frames from video file: {path}"
            )
        return _stack(frames)

    def _resolve_sample_path(self, sample_path: str | Path) -> Path:
        path = Path(sample_path)
        if path.is_absolute() or self.video_root is None:
            return path
        return self.video_root / path

    def _video_length(self, sample_path: str | Path, resolved_path: Path) -> int:
        candidate_keys = (
            str(sample_path),
            Path(sample_path).as_posix(),
            str(resolved_path),
            resolved_path.as_posix(),
        )
        for key in candidate_keys:
            if key in self.length_dict:
                return int(self.length_dict[key])
        raise KeyError(f"Video length not found for sample: {sample_path}")

    def _class_index(self, d2c: Any, digit_label: int) -> int:
        return int(d2c[digit_label])


def _read_rgb_frame(path: Path) -> Any:
    frame = np.asarray(Image.open(path).convert("RGB"))
    return _as_tensor(frame)


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on local runtime
        raise ImportError("Reading video files requires opencv-python.") from exc
    return cv2


def _as_tensor(value: Any) -> Any:
    if torch is not None:
        return torch.as_tensor(value)
    return np.asarray(value)


def _stack(values: Sequence[Any]) -> Any:
    if torch is not None:
        return torch.stack(list(values))
    return np.stack(values)
