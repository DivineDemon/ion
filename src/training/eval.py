"""
Evaluation: test metrics (accuracy, MSE) and per-length metrics for length-generalization.
Load from checkpoint or use in-memory model; output structure suitable for tables.
"""

from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .train import _forward_with_inductive, evaluate as _evaluate


def _compute_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    pred = output.argmax(dim=-1)
    return (pred == target).float().mean().item()


def _compute_mse(output: torch.Tensor, target: torch.Tensor) -> float:
    return F.mse_loss(output, target).item()


@torch.no_grad()
def evaluate_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, float]:
    """
    Run model on test loader; return dict with test_loss, test_accuracy (or test_mse).

    Args:
        model: Trained model.
        test_loader: Test DataLoader (batches of (x, y)).
        device: Device to run on.
        config: Must contain output_type ("classification" or "regression") and
                 num_classes if classification.

    Returns:
        {"test_loss": float, "test_accuracy": float} or {"test_loss": float, "test_mse": float}.
    """
    model.eval()
    task_type = config.get("output_type", "classification")
    num_classes = config.get("num_classes")

    total_loss = 0.0
    total_metric = 0.0
    n_batches = 0

    for batch in test_loader:
        x, y = batch[0].to(device), batch[1].to(device)
        output, _ = _forward_with_inductive(model, x, return_inductive=False)
        loss = (
            F.cross_entropy(output, y)
            if task_type == "classification"
            else F.mse_loss(output, y)
        )
        total_loss += loss.item()
        if task_type == "classification":
            total_metric += _compute_accuracy(output, y)
        else:
            total_metric += _compute_mse(output, y)
        n_batches += 1

    n = max(1, n_batches)
    result: dict[str, float] = {
        "test_loss": total_loss / n,
    }
    if task_type == "classification":
        result["test_accuracy"] = total_metric / n
    else:
        result["test_mse"] = total_metric / n
    return result


@torch.no_grad()
def evaluate_length_gen(
    model: nn.Module,
    test_loaders_by_length: dict[int, DataLoader],
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate per test length for length-generalization; aggregate for table output.

    Args:
        model: Trained sequence model.
        test_loaders_by_length: Map length -> DataLoader (e.g. {50: loader50, 100: loader100}).
        device: Device.
        config: Same as evaluate_test (output_type, num_classes).

    Returns:
        {
            "per_length": { length: {"test_accuracy": float} or {"test_mse": float}, ... },
            "lengths": [50, 100, 200],
            "mean_accuracy": float or "mean_mse": float (mean over lengths),
            "overall_accuracy" or "overall_mse": float (metric over all samples),
        }
    """
    model.eval()
    task_type = config.get("output_type", "classification")

    per_length: dict[int, dict[str, float]] = {}
    all_metrics: list[float] = []
    total_samples = 0

    for length in sorted(test_loaders_by_length.keys()):
        loader = test_loaders_by_length[length]
        metrics = evaluate_test(model, loader, device, config)
        per_length[length] = metrics
        if task_type == "classification":
            acc = metrics["test_accuracy"]
            all_metrics.append(acc)
            total_samples += 1
        else:
            mse = metrics["test_mse"]
            all_metrics.append(mse)
            total_samples += 1

    lengths = sorted(test_loaders_by_length.keys())
    result: dict[str, Any] = {
        "per_length": per_length,
        "lengths": lengths,
    }
    if task_type == "classification":
        result["mean_accuracy"] = sum(all_metrics) / max(1, len(all_metrics))
        # Overall: weighted by number of samples per length (if we had counts we could do that;
        # here we treat each length equally as "mean over lengths")
        result["overall_accuracy"] = result["mean_accuracy"]
    else:
        result["mean_mse"] = sum(all_metrics) / max(1, len(all_metrics))
        result["overall_mse"] = result["mean_mse"]
    return result


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """
    Load model state from checkpoint file. Returns checkpoint dict (epoch, config, etc.).
    """
    path = Path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    return ckpt if isinstance(ckpt, dict) else {}


def evaluate_from_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    test_loader: DataLoader,
    config: Optional[dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> dict[str, float]:
    """
    Load checkpoint into model and run evaluate_test. Config can be taken from
    checkpoint if not provided (checkpoint may contain "config").
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(checkpoint_path, model, device)
    if config is None or not config:
        config = ckpt.get("config", {})
    model.to(device)
    return evaluate_test(model, test_loader, device, config)


def evaluate_length_gen_from_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    test_loaders_by_length: dict[int, DataLoader],
    config: Optional[dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Load checkpoint and run evaluate_length_gen."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(checkpoint_path, model, device)
    if config is None or not config:
        config = ckpt.get("config", {})
    model.to(device)
    return evaluate_length_gen(model, test_loaders_by_length, device, config)
