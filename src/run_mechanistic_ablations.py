#!/usr/bin/env python3
"""
Mechanistic ablations: full ION vs P-only (F frozen), F-only (P frozen random), random P (fixed random P).

Runs on cumsum with 3-5 seeds per variant; writes results to results/mechanistic_ablations/.
Used to show that both P and F are necessary (Table: Mechanistic ablations).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
import yaml

from src.data.length_gen import TASK_CUMSUM, get_length_gen_loaders
from src.models import (
    GRUBaseline,
    IONRecurrent,
    count_parameters,
    suggest_ion_recurrent_dims,
)
from src.training.eval import evaluate_length_gen
from src.training.train import run_training


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(*paths: Path) -> dict:
    merged = {}
    for p in paths:
        if p.exists():
            with open(p) as f:
                merged = _deep_merge(merged, yaml.safe_load(f) or {})
    return merged


def get_seeds(config: dict, n: int = 5) -> list[int]:
    if config.get("seeds"):
        return list(config["seeds"])[:n]
    base = config.get("seed", 42)
    return [base + i * 1000 for i in range(n)]


# ---------------------------------------------------------------------------
# Identity F: for P-only ablation, F(p, x) = p
# ---------------------------------------------------------------------------

class IdentityF(nn.Module):
    """F that returns the first p_dim elements of input (p part of concat(p, x))."""

    def __init__(self, p_dim: int, input_dim: int):
        super().__init__()
        self.p_dim = p_dim

    def forward(self, px: torch.Tensor) -> torch.Tensor:
        return px[:, : self.p_dim].clone()


# ---------------------------------------------------------------------------
# Build ION with optional mechanistic variant
# ---------------------------------------------------------------------------

def build_ion_variant(
    config: dict,
    variant: str,
    device: torch.device,
) -> IONRecurrent:
    """Build ION for cumsum; variant in full_ion, P_only, F_only, random_P."""
    input_dim = config.get("input_dim", 1)
    num_layers = config.get("num_layers", 2)
    hidden_dim = config.get("hidden_dim", 64)
    p_dim = config.get("p_dim", 16)
    dropout = config.get("dropout", 0.0)

    gru = GRUBaseline(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_type="regression",
        num_classes=None,
        dropout=dropout,
    )
    target_params = count_parameters(gru)
    suggested = suggest_ion_recurrent_dims(
        target_params=target_params,
        input_dim=input_dim,
        num_classes_or_1=1,
        num_layers=num_layers,
        output_type="regression",
        tolerance=0.10,
    )
    if suggested:
        hidden_dim = suggested["hidden_dim"]
        p_dim = suggested["p_dim"]

    model = IONRecurrent(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        p_dim=p_dim,
        num_layers=num_layers,
        output_type="regression",
        num_classes=None,
        dropout=dropout,
    )

    if variant == "full_ion":
        pass
    elif variant == "P_only":
        # Freeze F and replace with identity so L_ind = ||P(h_{t+1}) - p_t||^2
        model.F = IdentityF(p_dim, input_dim)
        for p in model.F.parameters():
            p.requires_grad = False
    elif variant in ("F_only", "random_P"):
        # Freeze P: reinit as random linear and freeze
        torch.manual_seed(42)
        model.P = nn.Sequential(
            nn.Linear(hidden_dim, p_dim),
            nn.Tanh(),
        )
        for p in model.P.parameters():
            p.requires_grad = False
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return model


def get_loaders(config: dict, seed: int):
    return get_length_gen_loaders(
        task=TASK_CUMSUM,
        train_min_len=config["train_min_len"],
        train_max_len=config["train_max_len"],
        test_lengths=config["test_lengths"],
        batch_size=config.get("batch_size", 32),
        train_samples=config.get("train_samples", 10_000),
        val_samples=config.get("val_samples", 1_000),
        test_samples_per_length=config.get("test_samples_per_length", 500),
        seed=seed,
        num_workers=config.get("num_workers", 0),
        vocab_size=config.get("vocab_size", 2),
    )


# ---------------------------------------------------------------------------
# Run one variant over seeds
# ---------------------------------------------------------------------------

VARIANTS = ["full_ion", "P_only", "F_only", "random_P"]


def main():
    parser = argparse.ArgumentParser(description="Mechanistic ablations (cumsum)")
    parser.add_argument("--results-dir", type=Path, default=_ROOT / "results" / "mechanistic_ablations")
    parser.add_argument("--config-dir", type=Path, default=_ROOT / "configs")
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--variants", type=str, nargs="+", default=VARIANTS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_dir = args.config_dir / "length_gen"
    base = load_config(args.config_dir / "base.yaml")
    protocol = load_config(cfg_dir / "protocol.yaml")
    task = load_config(cfg_dir / "cumsum.yaml", cfg_dir / "cumsum_ion.yaml")
    config = _deep_merge(_deep_merge(base, protocol), task)
    config["device"] = str(device)
    config["task"] = TASK_CUMSUM
    config["output_type"] = "regression"
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.seeds:
        config["seeds"] = [int(s) for s in args.seeds.split(",")]

    seeds = get_seeds(config)
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        variant_dir = results_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        metrics = []
        for run_seed in seeds:
            torch.manual_seed(run_seed)
            model = build_ion_variant(config, variant, device)
            train_loader, val_loader, test_loaders = get_loaders(config, run_seed)
            run_config = {
                **config,
                "lambda_ind": config.get("lambda_ind", 0.5),
                "output_type": "regression",
            }
            run_training(
                model,
                train_loader,
                val_loader,
                run_config,
                result_dir=variant_dir,
                run_seed=run_seed,
            )
            ckpt_path = variant_dir / f"run_{run_seed}.pt"
            model = build_ion_variant(config, variant, device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)
            model.to(device)
            model.eval()
            lg_result = evaluate_length_gen(model, test_loaders, device, config)
            mean_mse = lg_result.get("mean_mse") or sum(
                lg_result["per_length"].get(l, {}).get("test_mse", 0)
                for l in config["test_lengths"]
            ) / max(1, len(config["test_lengths"]))
            metrics.append(mean_mse)
            out = {
                "variant": variant,
                "seed": run_seed,
                "metric_value": mean_mse,
                "per_length": lg_result.get("per_length", {}),
            }
            (variant_dir / f"run_{run_seed}.json").write_text(json.dumps(out, indent=2))
        summary = {
            "variant": variant,
            "seeds": seeds,
            "mean_mse": sum(metrics) / len(metrics),
            "std_mse": (sum((m - sum(metrics) / len(metrics)) ** 2 for m in metrics) / max(1, len(metrics) - 1)) ** 0.5 if len(metrics) > 1 else 0.0,
            "n": len(metrics),
        }
        (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"{variant}: mean_mse={summary['mean_mse']:.2f} std={summary['std_mse']:.2f} n={summary['n']}")

    print(f"Done. Results in {results_dir}")


if __name__ == "__main__":
    main()
