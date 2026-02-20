"""
Parameter matching utilities for ION vs baselines.

Use count_parameters(model) to get total trainable params. When comparing ION to
a baseline, set ION h_dim/p_dim so total params are within a few percent of the
baseline for fair comparison.

Formula: sum(p.numel() for p in model.parameters() if p.requires_grad)
"""

from typing import Any, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn

from .baselines import count_parameters
from .ion_recurrent import IONRecurrent
from .ion_universal import IONUniversal

__all__ = ["count_parameters", "suggest_ion_recurrent_dims", "suggest_ion_universal_dims"]


def suggest_ion_recurrent_dims(
    target_params: int,
    input_dim: int,
    num_classes_or_1: int,
    num_layers: int = 1,
    output_type: Literal["classification", "regression"] = "regression",
    tolerance: float = 0.05,
    h_dim_range: Tuple[int, int] = (16, 256),
    p_dim_range: Tuple[int, int] = (4, 64),
) -> Optional[Dict[str, int]]:
    """
    Suggest (hidden_dim, p_dim) for IONRecurrent to match target param count.

    Performs a simple grid search over h_dim and p_dim. Returns the first config
    within tolerance of target, or None if no match found.

    Args:
        target_params: Desired total parameter count (e.g. from GRU baseline).
        input_dim: Input dimension.
        num_classes_or_1: num_classes for classification, or 1 for regression.
        num_layers: Number of GRU layers.
        output_type: 'classification' or 'regression'.
        tolerance: Accept configs within this fraction of target (e.g. 0.05 = 5%).
        h_dim_range: (min, max) hidden_dim to try.
        p_dim_range: (min, max) p_dim to try.

    Returns:
        Dict with 'hidden_dim', 'p_dim', 'param_count' or None.
    """
    best: Optional[Dict[str, Any]] = None
    best_diff = float("inf")

    for h_dim in range(h_dim_range[0], h_dim_range[1] + 1, 8):
        for p_dim in range(p_dim_range[0], min(p_dim_range[1], h_dim) + 1, 4):
            try:
                model = IONRecurrent(
                    input_dim=input_dim,
                    hidden_dim=h_dim,
                    p_dim=p_dim,
                    num_layers=num_layers,
                    output_type=output_type,
                    num_classes=num_classes_or_1 if output_type == "classification" else None,
                )
                n = count_parameters(model)
                diff = abs(n - target_params) / max(1, target_params)
                if diff <= tolerance:
                    return {"hidden_dim": h_dim, "p_dim": p_dim, "param_count": n}
                if diff < best_diff:
                    best_diff = diff
                    best = {"hidden_dim": h_dim, "p_dim": p_dim, "param_count": n}
            except Exception:
                continue
    return best


def suggest_ion_universal_dims(
    target_params: int,
    input_size: int,
    output_size: int,
    num_layers: int = 4,
    tolerance: float = 0.05,
    hidden_dim_range: Tuple[int, int] = (64, 512),
    p_dim_range: Tuple[int, int] = (4, 64),
) -> Optional[Dict[str, int]]:
    """
    Suggest (hidden_dim, p_dim) for IONUniversal to match target param count.

    Args:
        target_params: Desired total parameter count (e.g. from MLP baseline).
        input_size: Flattened input dimension.
        output_size: Output dimension.
        num_layers: Number of hidden layers.
        tolerance: Accept configs within this fraction of target.
        hidden_dim_range: (min, max) hidden_dim to try.
        p_dim_range: (min, max) p_dim to try.

    Returns:
        Dict with 'hidden_dim', 'p_dim', 'param_count' or None.
    """
    best: Optional[Dict[str, Any]] = None
    best_diff = float("inf")

    step_h = max(8, (hidden_dim_range[1] - hidden_dim_range[0]) // 10)
    step_p = max(2, (p_dim_range[1] - p_dim_range[0]) // 8)

    for h_dim in range(hidden_dim_range[0], hidden_dim_range[1] + 1, step_h):
        for p_dim in range(p_dim_range[0], min(p_dim_range[1], h_dim) + 1, step_p):
            try:
                model = IONUniversal(
                    input_size=input_size,
                    hidden_dim=h_dim,
                    num_layers=num_layers,
                    output_size=output_size,
                    p_dim=p_dim,
                )
                n = count_parameters(model)
                diff = abs(n - target_params) / max(1, target_params)
                if diff <= tolerance:
                    return {"hidden_dim": h_dim, "p_dim": p_dim, "param_count": n}
                if diff < best_diff:
                    best_diff = diff
                    best = {"hidden_dim": h_dim, "p_dim": p_dim, "param_count": n}
            except Exception:
                continue
    return best
