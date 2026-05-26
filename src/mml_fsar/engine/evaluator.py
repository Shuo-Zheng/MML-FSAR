"""Evaluation loop for MML-FSAR experiments."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mml_fsar.data.episodic_dataset import EpisodicVideoDataset
from mml_fsar.data.transforms import VideoTransformTest
from mml_fsar.models import MMLFSAR, MMLFSARConfig
from mml_fsar.utils.checkpoint import resolve_checkpoint_path
from mml_fsar.utils.metrics import aggregate_accuracy


def evaluate(
    config: dict[str, Any],
    checkpoint: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Evaluate an MML-FSAR experiment."""

    model_config = MMLFSARConfig.from_experiment_config(config)
    checkpoint_path = str(checkpoint) if dry_run and checkpoint is not None else None
    if not dry_run:
        checkpoint_path = str(resolve_checkpoint_path(config, checkpoint))

    result = {
        "experiment": config["experiment"],
        "dataset": config["dataset"]["name"],
        "dry_run": dry_run,
        "checkpoint": checkpoint_path,
        "num_way": model_config.num_way,
        "num_frames": model_config.num_frames,
    }
    if dry_run:
        return result

    torch = _require_torch_runtime()
    device = _resolve_device(torch, config.get("device", "cuda"))
    model = MMLFSAR(model_config).to(device)
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint_data.get("model_state_dict", checkpoint_data)
    model.load_state_dict(model_state)

    test_loader = _build_episode_loader(torch, config, split="test")
    transform = _build_eval_transform(config)
    result.update(
        evaluate_model_on_loader(
            model=model,
            loader=test_loader,
            transform=transform,
            device=device,
            torch=torch,
            use_dagger=_use_dagger(config),
            split="test",
        )
    )
    return result


def evaluate_model_on_loader(
    model: Any,
    loader: Any,
    transform: Any,
    device: Any,
    torch: Any,
    use_dagger: bool = False,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate a model over an episodic loader."""

    previous_split = getattr(model, "evaluation_split", "test")
    if hasattr(model, "set_evaluation_split"):
        model.set_evaluation_split(split)
    model.eval()
    losses: list[float] = []
    video_accuracies: list[float] = []
    dagger_accuracies: list[float] = []

    try:
        with torch.no_grad():
            for episode in loader:
                (
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                    real_support_labels,
                    real_query_labels,
                ) = prepare_episode(episode, transform, device, torch)
                outputs = model(
                    support_images=support_images,
                    support_labels=support_labels,
                    query_images=query_images,
                    query_labels=query_labels,
                    real_support_labels=real_support_labels,
                    real_query_labels=real_query_labels,
                    return_loss=True,
                )
                if "loss" in outputs:
                    losses.append(float(outputs["loss"].detach().item()))
                video_accuracies.append(
                    aggregate_accuracy(outputs["video_matching_probs"], query_labels)
                )
                if use_dagger:
                    dagger_accuracies.append(
                        aggregate_accuracy(outputs["dagger_probs"], query_labels)
                    )
    finally:
        if hasattr(model, "set_evaluation_split"):
            model.set_evaluation_split(previous_split)

    return _finalize_evaluation_metrics(
        losses=losses,
        video_accuracies=video_accuracies,
        dagger_accuracies=dagger_accuracies,
        use_dagger=use_dagger,
    )


def _finalize_evaluation_metrics(
    losses: list[float],
    video_accuracies: list[float],
    dagger_accuracies: list[float],
    use_dagger: bool,
) -> dict[str, Any]:
    if not video_accuracies:
        raise ValueError("Evaluation loader did not yield any episodes.")
    if use_dagger and not dagger_accuracies:
        raise ValueError("Dagger evaluation did not yield any episodes.")

    metrics = {
        "loss": _mean(losses),
        "video_accuracy": _percentage(video_accuracies),
        "accuracy": _percentage(video_accuracies),
        "confidence": _confidence(video_accuracies),
        "primary_metric": "video_accuracy",
        "num_episodes": len(video_accuracies),
    }
    if use_dagger:
        metrics["dagger_accuracy"] = _percentage(dagger_accuracies)
        metrics["accuracy"] = metrics["dagger_accuracy"]
        metrics["confidence"] = _confidence(dagger_accuracies)
        metrics["primary_metric"] = "dagger_accuracy"
    return metrics


def prepare_episode(
    episode: tuple[Any, Any, Any, Any, Any, Any, Any],
    transform: Any,
    device: Any,
    torch: Any,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Move one episode to the runtime device and apply video transforms."""

    (
        support_images,
        support_labels,
        query_images,
        query_labels,
        _digit_to_class,
        real_support_labels,
        real_query_labels,
    ) = episode
    return (
        _prepare_videos(support_images, transform, device, torch),
        _prepare_labels(support_labels, device),
        _prepare_videos(query_images, transform, device, torch),
        _prepare_labels(query_labels, device),
        _prepare_labels(real_support_labels, device),
        _prepare_labels(real_query_labels, device),
    )


def _prepare_videos(videos: Any, transform: Any, device: Any, torch: Any) -> Any:
    prepared = []
    for video in videos:
        if hasattr(video, "to"):
            video = video.to(device, non_blocking=True)
        prepared.append(transform(video))
    return torch.stack(prepared)


def _prepare_labels(labels: Any, device: Any) -> Any:
    if hasattr(labels, "to"):
        return labels.to(device, non_blocking=True).long()
    raise TypeError("Episode labels must be torch tensors during runtime.")


def _build_episode_loader(torch: Any, config: dict[str, Any], split: str) -> Any:
    dataset_settings = config["dataset"]
    dataset = EpisodicVideoDataset(
        task_dict=dataset_settings[f"{split}_task_dict"],
        length_dict=dataset_settings["length_dict"],
        num_frames=int(config["episode"]["num_frames"]),
        video_root=dataset_settings.get("video_root"),
        seed=int(config.get("seed", 0)),
    )
    DataLoader = torch.utils.data.DataLoader
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=int(config.get("num_workers", 0)),
        shuffle=False,
        pin_memory=_uses_cuda(config.get("device", "cuda")),
    )


def _build_eval_transform(config: dict[str, Any]) -> Any:
    image_size = int(config["episode"].get("image_size", 224))
    resize = int(config["episode"].get("resize", 256))
    return VideoTransformTest(resize=resize, crop_size=(image_size, image_size))


def _require_torch_runtime() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError("Full MML-FSAR evaluation requires PyTorch.") from exc
    if not hasattr(torch, "utils") or MMLFSAR is None:
        raise ImportError("Full MML-FSAR evaluation requires PyTorch.")
    return torch


def _resolve_device(torch: Any, requested_device: Any) -> Any:
    requested = str(requested_device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _uses_cuda(requested_device: Any) -> bool:
    return str(requested_device).startswith("cuda")


def _use_dagger(config: dict[str, Any]) -> bool:
    return bool(config.get("inference", {}).get("use_dagger", False))


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _percentage(values: list[float]) -> float:
    return _mean(values) * 100.0


def _confidence(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(196.0 * np.std(values) / math.sqrt(len(values)))
