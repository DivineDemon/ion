"""
Invariant drift: compute |P(h_{t+1}) - F(P(h_t))| for ION models.
Used for figures: drift vs step (recurrent) or vs layer (universal).
Returns mean/std over steps and samples.
"""

from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _drift_recurrent(
    model: nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """
    Per-step invariant drift for IONRecurrent.
    drift_t = mean over batch of ||P(h_{t+1}) - F(P(h_t), x_t)|| (L2 norm per sample, then mean).

    Returns:
        per_step_drift: (T-1,) tensor, mean drift at each step over batch.
        mean_drift: scalar, mean over all steps.
        std_drift: scalar, std over all steps.
    """
    with torch.no_grad():
        gru_out, _ = model.phi(x)  # (B, T, H)
    B, T, _ = x.shape
    if T < 2:
        return torch.tensor([], device=x.device), 0.0, 0.0

    per_step = []
    for t in range(T - 1):
        h_t = gru_out[:, t, :]      # (B, H)
        h_next = gru_out[:, t + 1, :]
        x_t = x[:, t, :]            # (B, D)
        p_t = model.P(h_t)
        p_next = model.P(h_next)
        p_pred = model.F(torch.cat([p_t, x_t], dim=-1))
        # L2 norm per sample: (B,), then mean over batch
        diff = (p_next - p_pred)
        norm_per_sample = diff.norm(dim=1)  # (B,)
        per_step.append(norm_per_sample.mean().item())
    per_step_t = torch.tensor(per_step, device=x.device, dtype=x.dtype)
    mean_d = per_step_t.mean().item()
    std_d = per_step_t.std().item() if len(per_step_t) > 1 else 0.0
    return per_step_t, mean_d, std_d


def _drift_universal(
    model: nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """
    Per-layer invariant drift for IONUniversal.
    drift_l = mean over batch of ||P(h^{(l+1)}) - F(P(h^{(l)}))||.

    Returns:
        per_layer_drift: (L-1,) tensor for layers 1..L-1.
        mean_drift, std_drift.
    """
    h_list = model._get_layer_outputs(x)  # h_list[0]=input, [1..L]=hidden
    if len(h_list) < 3:
        return torch.tensor([], device=x.device), 0.0, 0.0

    per_layer = []
    for l in range(1, len(h_list) - 1):
        h_l = h_list[l]
        h_next = h_list[l + 1]
        p_l = model.P(h_l)
        p_next = model.P(h_next)
        p_pred = model.F(p_l)
        diff = (p_next - p_pred)
        norm_per_sample = diff.norm(dim=1)
        per_layer.append(norm_per_sample.mean().item())
    per_layer_t = torch.tensor(per_layer, device=x.device, dtype=x.dtype)
    mean_d = per_layer_t.mean().item()
    std_d = per_layer_t.std().item() if len(per_layer_t) > 1 else 0.0
    return per_layer_t, mean_d, std_d


def _is_recurrent_ion(model: nn.Module) -> bool:
    return hasattr(model, "phi") and hasattr(model, "P") and hasattr(model, "F") and not hasattr(
        model, "_get_layer_outputs"
    )


def _is_universal_ion(model: nn.Module) -> bool:
    return hasattr(model, "_get_layer_outputs") and hasattr(model, "P") and hasattr(model, "F")


def compute_drift(
    model: nn.Module,
    x: torch.Tensor,
) -> dict[str, Any]:
    """
    Compute invariant drift for one batch. Dispatches to recurrent or universal ION.

    Args:
        model: IONRecurrent or IONUniversal (trained).
        x: One batch: (B, T, D) for recurrent, (B, ...) for universal.

    Returns:
        {
            "per_index": tensor of per-step or per-layer drift (mean over batch),
            "mean_drift": float,
            "std_drift": float,
            "type": "recurrent" | "universal",
        }
    """
    model.eval()
    if _is_recurrent_ion(model):
        per_index, mean_d, std_d = _drift_recurrent(model, x)
        return {
            "per_index": per_index,
            "mean_drift": mean_d,
            "std_drift": std_d,
            "type": "recurrent",
        }
    if _is_universal_ion(model):
        per_index, mean_d, std_d = _drift_universal(model, x)
        return {
            "per_index": per_index,
            "mean_drift": mean_d,
            "std_drift": std_d,
            "type": "universal",
        }
    raise TypeError("Model is not IONRecurrent or IONUniversal")


def compute_drift_over_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> dict[str, Any]:
    """
    Compute mean and std of drift over a dataset (or first max_batches batches).
    For recurrent: aggregates over all steps and all samples; returns also
    per-step mean across batches (so we can plot drift vs step for a few lengths).

    Returns:
        {
            "mean_drift": float (mean over all steps and samples),
            "std_drift": float (std over the per-batch mean drifts),
            "per_index_means": list of tensors (one per batch) or single aggregated
                per-step mean; for recurrent we can average per-step across batches
                only when all batches have same T (length). So we return
                "per_batch_results": list of {"mean_drift", "std_drift", "per_index"},
                and "mean_drift" / "std_drift" are over batches.
        }
    """
    model.eval()
    all_mean_drifts = []
    per_batch_results = []

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = batch[0].to(device)
        out = compute_drift(model, x)
        all_mean_drifts.append(out["mean_drift"])
        per_batch_results.append(
            {
                "mean_drift": out["mean_drift"],
                "std_drift": out["std_drift"],
                "per_index": out["per_index"].cpu() if out["per_index"].numel() > 0 else None,
            }
        )

    import numpy as np
    arr = np.array(all_mean_drifts)
    mean_over_batches = float(arr.mean())
    std_over_batches = float(arr.std()) if len(arr) > 1 else 0.0

    return {
        "mean_drift": mean_over_batches,
        "std_drift": std_over_batches,
        "per_batch_results": per_batch_results,
        "n_batches": len(all_mean_drifts),
    }
