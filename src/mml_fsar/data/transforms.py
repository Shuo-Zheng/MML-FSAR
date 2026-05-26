"""Video transform utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - depends on the local smoke-test env
    torch = None
else:
    if not hasattr(torch, "nn"):
        torch = None


class ConvertTHWCtoTCHW:
    """Convert video tensors from ``(T, H, W, C)`` to ``(T, C, H, W)``."""

    def __call__(self, vid: Any) -> Any:
        if hasattr(vid, "permute"):
            return vid.permute(0, 3, 1, 2)
        return np.transpose(vid, (0, 3, 1, 2))


class NormalizeImage:
    """Normalize image/video data from ``[0, 255]`` to ``[0, 1]``."""

    def __call__(self, vid: Any) -> Any:
        return vid / 255.0


class ToTensorVideo:
    """Convert ``(T, H, W, C)`` uint8 video data to ``(C, T, H, W)`` floats."""

    def __call__(self, clip: Any) -> Any:
        return self.forward(clip)

    def forward(self, clip: Any) -> Any:
        if hasattr(clip, "float") and hasattr(clip, "permute"):
            return clip.float().permute(3, 0, 1, 2) / 255.0
        return np.transpose(np.asarray(clip, dtype=np.float32), (3, 0, 1, 2)) / 255.0


class VideoTransformTrain:
    """Training transform matching the original experiment code."""

    def __init__(self, resize: int = 256, crop_size: tuple[int, int] = (224, 224)) -> None:
        self.transforms = _build_video_transform(resize, crop_size, train=True)

    def __call__(self, x: Any) -> Any:
        return self.transforms(x)


class VideoTransformTest:
    """Evaluation transform matching the original experiment code."""

    def __init__(self, resize: int = 256, crop_size: tuple[int, int] = (224, 224)) -> None:
        self.transforms = _build_video_transform(resize, crop_size, train=False)

    def __call__(self, x: Any) -> Any:
        return self.transforms(x)


def _build_video_transform(resize: int, crop_size: tuple[int, int], train: bool) -> Any:
    if torch is None:
        raise ImportError("Video transforms require PyTorch and torchvision.")

    import torchvision.transforms as transforms
    from torchvision.transforms import Compose
    from torchvision.transforms._transforms_video import (
        CenterCropVideo,
        NormalizeVideo,
        RandomCropVideo,
    )

    crop = RandomCropVideo(crop_size) if train else CenterCropVideo(crop_size)
    return Compose(
        [
            ToTensorVideo(),
            transforms.Resize(resize),
            crop,
            NormalizeVideo(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        ]
    )
