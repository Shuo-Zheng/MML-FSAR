"""Training loop for MML-FSAR experiments."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from mml_fsar.data.episodic_dataset import EpisodicVideoDataset
from mml_fsar.data.transforms import VideoTransformTrain, VideoTransformTest
from mml_fsar.engine.evaluator import (
    _use_dagger,
    evaluate_model_on_loader,
    prepare_episode,
)
from mml_fsar.models import MMLFSAR, MMLFSARConfig
from mml_fsar.utils.logging import ExperimentLogger, create_timestamped_run_dir
from mml_fsar.utils.metrics import aggregate_accuracy
from mml_fsar.utils.seed import set_seed


def launch_training(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    """Launch training on one or more GPUs according to the experiment config."""

    if dry_run or not _should_launch_distributed(config):
        return train(config, dry_run=dry_run)

    torch = _require_torch_runtime()
    settings = _distributed_settings(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed MML-FSAR training requires CUDA.")
    if max(settings["gpu_ids"]) >= torch.cuda.device_count():
        raise RuntimeError(
            "Configured GPU id exceeds the visible CUDA device count: "
            f"{settings['gpu_ids']}"
        )

    torch.multiprocessing.spawn(
        _distributed_worker,
        args=(copy.deepcopy(config), settings),
        nprocs=settings["world_size"],
        join=True,
    )
    return _load_training_summary(config, settings)


def train(
    config: dict[str, Any],
    dry_run: bool = False,
    distributed_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train an MML-FSAR experiment.

    The dry-run path verifies configuration and construction boundaries without
    loading private datasets or model weights.
    """

    set_seed(int(config.get("seed", 0)))
    model_config = MMLFSARConfig.from_experiment_config(config)
    distributed = distributed_context or _distributed_settings(config)
    is_primary_process = _is_primary_process(distributed)
    configured_output_dir = Path(str(config["output_dir"]))
    output_dir = configured_output_dir
    if not dry_run and is_primary_process:
        output_dir = create_timestamped_run_dir(configured_output_dir)
    result = {
        "experiment": config["experiment"],
        "dataset": config["dataset"]["name"],
        "dry_run": dry_run,
        "num_way": model_config.num_way,
        "num_frames": model_config.num_frames,
        "output_dir": str(output_dir),
        "distributed": _public_distributed_settings(distributed),
    }
    if dry_run:
        return result

    torch = _require_torch_runtime()
    device = _resolve_device(torch, _requested_device(config, distributed))
    logger = None
    if is_primary_process:
        logger = ExperimentLogger(output_dir)
        logged_config = copy.deepcopy(config)
        logged_config["output_dir"] = str(output_dir)
        logger.write_config(logged_config)

    model = MMLFSAR(model_config).to(device)
    if distributed.get("enabled"):
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[int(distributed["device_id"])],
            output_device=int(distributed["device_id"]),
            find_unused_parameters=True,
        )
    optimizer = _build_optimizer(torch, model, config)
    train_loader = _build_episode_loader(
        torch,
        config,
        split="train",
        distributed_context=distributed,
    )
    valid_loader = (
        _build_episode_loader(torch, config, split="valid")
        if is_primary_process
        else None
    )
    test_loader = (
        _build_episode_loader(torch, config, split="test")
        if is_primary_process
        else None
    )
    train_transform = _build_train_transform(config)
    eval_transform = _build_eval_transform(config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[_warmup_decay_lambda(len(train_loader.dataset))],
    )

    tasks_per_batch = int(config["optimization"].get("tasks_per_batch", 1))
    total_iters = int(config["optimization"].get("total_iters", 0))
    print_freq = int(config["optimization"].get("print_freq", 0))
    save_freq = int(config["optimization"].get("save_freq", 0))
    train_log_freq = int(config.get("logging", {}).get("train_freq", print_freq))
    best_accuracy = 0.0
    best_checkpoint = output_dir / "best_checkpoint.pt"
    latest_validation: dict[str, Any] | None = None
    latest_testing: dict[str, Any] | None = None
    train_losses: list[float] = []
    train_accuracies: list[float] = []
    logged_train_losses: list[float] = []
    logged_train_accuracies: list[float] = []

    optimizer.zero_grad()
    iteration = 0
    for episode_index, episode in enumerate(train_loader, start=1):
        global_episode_index = episode_index * int(distributed.get("world_size", 1))
        if total_iters > 0 and global_episode_index > total_iters:
            break
        iteration = global_episode_index
        model.train()
        (
            support_images,
            support_labels,
            query_images,
            query_labels,
            real_support_labels,
            real_query_labels,
        ) = prepare_episode(episode, train_transform, device, torch)
        outputs = model(
            support_images=support_images,
            support_labels=support_labels,
            query_images=query_images,
            query_labels=query_labels,
            real_support_labels=real_support_labels,
            real_query_labels=real_query_labels,
            return_loss=True,
        )
        accumulation_steps = _gradient_accumulation_steps(
            tasks_per_batch,
            int(distributed.get("world_size", 1)),
        )
        loss = outputs["loss"] / accumulation_steps
        loss.backward()
        episode_loss = float(outputs["loss"].detach().item())
        episode_accuracy = aggregate_accuracy(
            outputs["video_matching_probs"],
            query_labels,
        )
        train_losses.append(episode_loss)
        train_accuracies.append(episode_accuracy)
        logged_train_losses.append(episode_loss)
        logged_train_accuracies.append(episode_accuracy)

        if iteration % tasks_per_batch == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(iteration)

        if print_freq > 0 and iteration % print_freq == 0:
            if logger is not None:
                logger.print(
                    "episode={episode} loss={loss:.6f} acc={acc:.4f} lr={lr:.8f}".format(
                        episode=iteration,
                        loss=_mean(train_losses),
                        acc=_mean(train_accuracies),
                        lr=scheduler.get_last_lr()[0],
                    )
                )
            train_losses = []
            train_accuracies = []

        if train_log_freq > 0 and iteration % train_log_freq == 0:
            if logger is not None:
                logger.log_train(
                    episode=iteration,
                    loss=_mean(logged_train_losses),
                    accuracy=_mean(logged_train_accuracies),
                    lr=scheduler.get_last_lr()[0],
                )
            logged_train_losses = []
            logged_train_accuracies = []

        if save_freq > 0 and iteration % save_freq == 0:
            if not is_primary_process:
                _distributed_barrier(torch, distributed)
                continue
            assert logger is not None
            assert valid_loader is not None
            assert test_loader is not None
            evaluation_model = _unwrap_distributed_model(model)
            validation = evaluate_model_on_loader(
                model=evaluation_model,
                loader=valid_loader,
                transform=eval_transform,
                device=device,
                torch=torch,
                use_dagger=_use_dagger(config),
                split="valid",
            )
            testing = evaluate_model_on_loader(
                model=evaluation_model,
                loader=test_loader,
                transform=eval_transform,
                device=device,
                torch=torch,
                use_dagger=_use_dagger(config),
                split="test",
            )
            latest_validation = validation
            latest_testing = testing
            logger.log_evaluation(split="valid", episode=iteration, metrics=validation)
            logger.log_evaluation(split="test", episode=iteration, metrics=testing)
            if validation["accuracy"] >= best_accuracy:
                best_accuracy = float(validation["accuracy"])
                _save_checkpoint(
                    torch,
                    best_checkpoint,
                    _unwrap_distributed_model(model),
                    optimizer,
                    scheduler,
                    iteration,
                    metrics={
                        "best_accuracy": best_accuracy,
                        "validation": validation,
                        "test": testing,
                    },
                )
            _distributed_barrier(torch, distributed)

    if iteration == 0:
        raise ValueError("Training loader did not yield any episodes.")
    if iteration % tasks_per_batch != 0:
        optimizer.step()
        optimizer.zero_grad()
    if logger is not None and logged_train_losses:
        logger.log_train(
            episode=iteration,
            loss=_mean(logged_train_losses),
            accuracy=_mean(logged_train_accuracies),
            lr=scheduler.get_last_lr()[0],
        )

    final_metrics: dict[str, Any] = {
        "best_accuracy": best_accuracy,
    }
    if latest_validation is not None:
        final_metrics["last_validation"] = latest_validation
    if latest_testing is not None:
        final_metrics["last_test"] = latest_testing
    if is_primary_process and best_accuracy == 0.0:
        _save_checkpoint(
            torch,
            best_checkpoint,
            _unwrap_distributed_model(model),
            optimizer,
            scheduler,
            iteration,
            metrics=final_metrics,
        )

    result.update(
        {
            "dry_run": False,
            "iterations": iteration,
            "best_accuracy": best_accuracy,
            "best_checkpoint": str(best_checkpoint),
        }
    )
    if logger is not None:
        logger.write_summary(
            {
                **result,
                "last_validation": latest_validation,
                "last_test": latest_testing,
                "primary_metric": (
                    "dagger_accuracy" if _use_dagger(config) else "video_accuracy"
                ),
            }
        )
        logger.close()
    return result


def _build_optimizer(torch: Any, model: Any, config: dict[str, Any]) -> Any:
    optimization = config["optimization"]
    parameters = filter(lambda parameter: parameter.requires_grad, model.parameters())
    learning_rate = float(optimization["learning_rate"])
    optimizer_name = str(optimization.get("optimizer", "adam")).lower()
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _build_episode_loader(
    torch: Any,
    config: dict[str, Any],
    split: str,
    distributed_context: dict[str, Any] | None = None,
) -> Any:
    dataset_settings = config["dataset"]
    dataset = EpisodicVideoDataset(
        task_dict=dataset_settings[f"{split}_task_dict"],
        length_dict=dataset_settings["length_dict"],
        num_frames=int(config["episode"]["num_frames"]),
        video_root=dataset_settings.get("video_root"),
        seed=int(config.get("seed", 0)),
    )
    sampler = None
    if distributed_context and distributed_context.get("enabled") and split == "train":
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=int(distributed_context["world_size"]),
            rank=int(distributed_context["rank"]),
            shuffle=False,
            drop_last=False,
        )
    DataLoader = torch.utils.data.DataLoader
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=int(config.get("num_workers", 0)),
        sampler=sampler,
        shuffle=False,
        pin_memory=_uses_cuda(config.get("device", "cuda")),
    )


def _build_train_transform(config: dict[str, Any]) -> Any:
    image_size = int(config["episode"].get("image_size", 224))
    resize = int(config["episode"].get("resize", 256))
    return VideoTransformTrain(resize=resize, crop_size=(image_size, image_size))


def _build_eval_transform(config: dict[str, Any]) -> Any:
    image_size = int(config["episode"].get("image_size", 224))
    resize = int(config["episode"].get("resize", 256))
    return VideoTransformTest(resize=resize, crop_size=(image_size, image_size))


def _warmup_decay_lambda(num_tasks: int) -> Any:
    warmup_iters = 10_000

    def schedule(episode: int) -> float:
        if episode < warmup_iters:
            return float((episode // 16 * 16) / warmup_iters)
        decay_iters = max(num_tasks - warmup_iters, 1)
        return max(float((num_tasks - episode) // 16 * 16 / decay_iters), 0.0)

    return schedule


def _save_checkpoint(
    torch: Any,
    path: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    iteration: int,
    metrics: dict[str, Any] | None = None,
) -> None:
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics or {},
        },
        path,
    )


def _distributed_worker(
    rank: int,
    config: dict[str, Any],
    settings: dict[str, Any],
) -> None:
    torch = _require_torch_runtime()
    gpu_id = int(settings["gpu_ids"][rank])
    torch.cuda.set_device(gpu_id)
    torch.distributed.init_process_group(
        backend=str(settings["backend"]),
        init_method=f"tcp://{settings['master_addr']}:{settings['master_port']}",
        rank=rank,
        world_size=int(settings["world_size"]),
    )
    worker_config = copy.deepcopy(config)
    worker_config["device"] = f"cuda:{gpu_id}"
    context = {
        **settings,
        "enabled": True,
        "rank": rank,
        "local_rank": rank,
        "device_id": gpu_id,
    }
    try:
        train(worker_config, dry_run=False, distributed_context=context)
    finally:
        torch.distributed.destroy_process_group()


def _distributed_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("distributed", {})
    gpu_ids = [int(gpu_id) for gpu_id in settings.get("gpu_ids", [0])]
    if not gpu_ids:
        gpu_ids = [0]
    enabled = bool(settings.get("enabled", False)) and len(gpu_ids) > 1
    active_gpu_ids = gpu_ids if enabled else [gpu_ids[0]]
    return {
        "enabled": enabled,
        "gpu_ids": active_gpu_ids,
        "world_size": len(active_gpu_ids),
        "backend": str(settings.get("backend", "nccl")),
        "master_addr": str(settings.get("master_addr", "127.0.0.1")),
        "master_port": str(settings.get("master_port", "29500")),
    }


def _should_launch_distributed(config: dict[str, Any]) -> bool:
    settings = _distributed_settings(config)
    return bool(settings["enabled"] and settings["world_size"] > 1)


def _public_distributed_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(settings.get("enabled", False)),
        "world_size": int(settings.get("world_size", 1)),
        "gpu_ids": list(settings.get("gpu_ids", [0])),
    }


def _requested_device(config: dict[str, Any], distributed: dict[str, Any]) -> str:
    requested = str(config.get("device", "cuda"))
    if requested == "cuda" and distributed.get("gpu_ids"):
        return f"cuda:{int(distributed['gpu_ids'][0])}"
    return requested


def _is_primary_process(distributed: dict[str, Any]) -> bool:
    return int(distributed.get("rank", 0)) == 0


def _gradient_accumulation_steps(tasks_per_batch: int, world_size: int) -> int:
    return max(int(tasks_per_batch) // max(int(world_size), 1), 1)


def _unwrap_distributed_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def _distributed_barrier(torch: Any, distributed: dict[str, Any]) -> None:
    if distributed.get("enabled") and hasattr(torch, "distributed"):
        torch.distributed.barrier()


def _load_training_summary(config: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(str(config["output_dir"]))
    summary_path = _latest_summary_path(output_dir)
    if summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    model_config = MMLFSARConfig.from_experiment_config(config)
    return {
        "experiment": config["experiment"],
        "dataset": config["dataset"]["name"],
        "dry_run": False,
        "num_way": model_config.num_way,
        "num_frames": model_config.num_frames,
        "output_dir": str(output_dir),
        "distributed": _public_distributed_settings(settings),
        "best_checkpoint": str(output_dir / "best_checkpoint.pt"),
    }


def _latest_summary_path(output_dir: Path) -> Path:
    direct_summary = output_dir / "summary.json"
    if direct_summary.is_file():
        return direct_summary
    runs_dir = output_dir / "runs"
    if runs_dir.is_dir():
        run_dirs = sorted(
            (path for path in runs_dir.iterdir() if path.is_dir()),
            reverse=True,
        )
        for run_dir in run_dirs:
            summary_path = run_dir / "summary.json"
            if summary_path.is_file():
                return summary_path
    return direct_summary


def _require_torch_runtime() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError("Full MML-FSAR training requires PyTorch.") from exc
    if not hasattr(torch, "utils") or MMLFSAR is None:
        raise ImportError("Full MML-FSAR training requires PyTorch.")
    return torch


def _resolve_device(torch: Any, requested_device: Any) -> Any:
    requested = str(requested_device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _uses_cuda(requested_device: Any) -> bool:
    return str(requested_device).startswith("cuda")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
