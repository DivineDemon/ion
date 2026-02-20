"""
Generic training loop with logging, checkpointing, and result persistence.
Uses seed from config; saves best checkpoint by validation metric; writes metrics to JSON/CSV.
"""

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .seeds import set_seeds


def get_device() -> torch.device:
    """Prefer CUDA, then MPS (Apple Silicon), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def _has_inductive_loss(model: nn.Module) -> bool:
    """True if model.forward can return (output, inductive_loss)."""
    import inspect
    sig = inspect.signature(model.forward)
    return "return_inductive_loss" in sig.parameters


def _forward_with_inductive(
    model: nn.Module, x: torch.Tensor, return_inductive: bool = True
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Run model forward; return (output, inductive_loss or None).
    Handles both ION-style models (two return values) and baselines (single return).
    """
    if _has_inductive_loss(model):
        out = model(x, return_inductive_loss=return_inductive)
    else:
        out = model(x)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, None


def _compute_task_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    task_type: str,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """Task loss: CE for classification, MSE for regression."""
    if task_type == "classification":
        return F.cross_entropy(output, target)
    return F.mse_loss(output, target)


def _compute_accuracy(output: torch.Tensor, target: torch.Tensor, task_type: str) -> float:
    """Accuracy: % correct for classification; for regression return 0 (use MSE in metrics)."""
    if task_type == "classification":
        pred = output.argmax(dim=-1)
        return (pred == target).float().mean().item()
    return 0.0


def _compute_mse(output: torch.Tensor, target: torch.Tensor) -> float:
    """MSE for regression."""
    return F.mse_loss(output, target).item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict[str, Any],
) -> tuple[float, float, Optional[float]]:
    """
    One training epoch. Returns (mean_train_loss, mean_train_metric, mean_ind_loss or None).
    Metric is accuracy for classification, MSE for regression.
    """
    model.train()
    task_type = config.get("output_type", "classification")
    num_classes = config.get("num_classes")
    lambda_ind = config.get("lambda_ind", 0.5)

    total_loss = 0.0
    total_metric = 0.0
    total_ind = 0.0
    n_ind = 0
    n_batches = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        x, y = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()

        output, ind_loss = _forward_with_inductive(model, x, return_inductive=True)
        task_loss = _compute_task_loss(output, y, task_type, num_classes)
        loss = task_loss
        if ind_loss is not None and lambda_ind > 0:
            loss = task_loss + lambda_ind * ind_loss
            total_ind += ind_loss.item()
            n_ind += 1

        loss.backward()
        max_grad_norm = config.get("max_grad_norm")
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        if task_type == "classification":
            total_metric += _compute_accuracy(output, y, task_type)
        else:
            total_metric += _compute_mse(output, y)
        n_batches += 1

    mean_loss = total_loss / n_batches
    mean_metric = total_metric / n_batches
    mean_ind = (total_ind / n_ind) if n_ind else None
    return mean_loss, mean_metric, mean_ind


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
) -> tuple[float, float]:
    """
    Validation or test evaluation. Returns (mean_loss, mean_metric).
    Metric is accuracy for classification, MSE for regression.
    """
    model.eval()
    task_type = config.get("output_type", "classification")
    num_classes = config.get("num_classes")

    total_loss = 0.0
    total_metric = 0.0
    n_batches = 0

    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        output, _ = _forward_with_inductive(model, x, return_inductive=False)
        task_loss = _compute_task_loss(output, y, task_type, num_classes)
        total_loss += task_loss.item()
        if task_type == "classification":
            total_metric += _compute_accuracy(output, y, task_type)
        else:
            total_metric += _compute_mse(output, y)
        n_batches += 1

    mean_loss = total_loss / n_batches
    mean_metric = total_metric / n_batches
    return mean_loss, mean_metric


def _is_better(current: float, best: float, task_type: str) -> bool:
    """True if current is better than best (higher accuracy, lower MSE)."""
    if task_type == "classification":
        return current > best
    return current < best


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    *,
    result_dir: Optional[Path] = None,
    run_seed: Optional[int] = None,
) -> dict[str, Any]:
    """
    Full training run: seed, train loop, validation, best checkpoint, result file.

    Args:
        model: PyTorch model (ION or baseline).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Full config dict. Expected keys: seed (or run_seed), device, output_root,
                task, model (name), epochs, lr, output_type, num_classes (if classification),
                lambda_ind (for ION), optional scheduler.
        result_dir: Directory for checkpoints and result file (e.g. results/length_gen/gru).
                   If None, built from config: output_root / task / model_name.
        run_seed: Override seed for this run (e.g. 42, 123). If None, use config["seed"].

    Returns:
        result: Dict with keys like train_loss, val_loss, val_accuracy (or val_mse),
                best_epoch, final_metrics, and per-epoch history (train_losses, val_losses, val_metrics).
    """
    seed = run_seed if run_seed is not None else config.get("seed", 42)
    run_config = {**config, "seed": seed}
    set_seeds(run_config)

    device = torch.device(config["device"]) if "device" in config and config["device"] in ("cuda", "mps", "cpu") else get_device()
    model = model.to(device)

    epochs = config.get("epochs", 50)
    lr = config.get("lr", 1e-3)
    task_type = config.get("output_type", "classification")
    warmup_epochs = config.get("warmup_epochs", 0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if config.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))

    if result_dir is None:
        output_root = Path(config.get("output_root", "results"))
        task = config.get("task", "default")
        model_name = config.get("model", "model")
        result_dir = output_root / str(task) / str(model_name)
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = result_dir / f"run_{seed}.pt"
    result_path = result_dir / f"run_{seed}.json"

    best_val_metric = -float("inf") if task_type == "classification" else float("inf")
    best_epoch = 0
    history = {
        "train_losses": [],
        "train_metrics": [],
        "val_losses": [],
        "val_metrics": [],
        "ind_losses": [],
    }

    for epoch in range(epochs):
        if warmup_epochs and epoch < warmup_epochs:
            scale = (epoch + 1) / warmup_epochs
            for g in optimizer.param_groups:
                g["lr"] = lr * scale
        elif scheduler is not None and warmup_epochs and epoch == warmup_epochs:
            for g in optimizer.param_groups:
                g["lr"] = lr
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}...", flush=True)
        train_loss, train_metric, mean_ind = train_one_epoch(
            model, train_loader, optimizer, device, config
        )
        val_loss, val_metric = evaluate(model, val_loader, device, config)

        if scheduler is not None and (warmup_epochs == 0 or epoch >= warmup_epochs):
            scheduler.step()

        history["train_losses"].append(train_loss)
        history["train_metrics"].append(train_metric)
        history["val_losses"].append(val_loss)
        history["val_metrics"].append(val_metric)
        if mean_ind is not None:
            history["ind_losses"].append(mean_ind)

        if _is_better(val_metric, best_val_metric, task_type):
            best_val_metric = val_metric
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metric": val_metric,
                    "config": config,
                },
                checkpoint_path,
            )

    # Persist results as JSON (and optionally CSV)
    import json

    metric_name = "val_accuracy" if task_type == "classification" else "val_mse"
    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_metric": best_val_metric,
        "final_train_loss": history["train_losses"][-1],
        "final_val_loss": history["val_losses"][-1],
        "final_val_metric": history["val_metrics"][-1],
        metric_name: best_val_metric,
        "history": history,
        "config": {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool, list, type(None)))},
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    # Also write a one-line CSV summary for easy aggregation (append-friendly)
    csv_path = result_dir / "runs.csv"
    csv_header = "seed,best_epoch,best_val_metric,final_val_loss\n"
    csv_line = f"{seed},{best_epoch},{best_val_metric},{history['val_losses'][-1]}\n"
    if not csv_path.exists():
        csv_path.write_text(csv_header + csv_line)
    else:
        with open(csv_path, "a") as f:
            f.write(csv_line)

    return result
