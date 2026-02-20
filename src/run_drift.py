#!/usr/bin/env python3
"""
Generate invariant drift data for fig_drift.

Loads a trained ION model (MNIST or length-gen) and computes per-layer (universal)
or per-step (recurrent) drift. Saves to results/drift/drift.json for plot_drift.

Usage:
  python -m src.run_drift
  python -m src.run_drift --checkpoint results/mnist/ion/run_42.pt
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
import yaml

from src.analysis.invariant_drift import compute_drift_over_dataset
from src.data.mnist_loader import get_mnist_loaders
from src.models import (
    IONUniversal,
    MLPBaseline,
    count_parameters,
    suggest_ion_universal_dims,
)
from src.training.eval import load_checkpoint


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(*config_paths: Path) -> dict:
    merged: dict = {}
    for path in config_paths:
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, data)
    return merged


def build_ion(config: dict) -> IONUniversal:
    num_layers = config["num_layers"]
    hidden_dim = config["hidden_dim"]
    p_dim = config.get("p_dim", 32)
    mlp = MLPBaseline(
        input_size=config["input_size"],
        hidden_dim=config["hidden_dim"],
        num_layers=num_layers,
        output_size=config["output_size"],
        dropout=config.get("dropout", 0.0),
    )
    target_params = count_parameters(mlp)
    suggested = suggest_ion_universal_dims(
        target_params=target_params,
        input_size=config["input_size"],
        output_size=config["output_size"],
        num_layers=num_layers,
        tolerance=0.10,
    )
    if suggested:
        hidden_dim = suggested["hidden_dim"]
        p_dim = suggested["p_dim"]
    return IONUniversal(
        input_size=config["input_size"],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_size=config["output_size"],
        p_dim=p_dim,
        dropout=config.get("dropout", 0.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute invariant drift for fig_drift")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to .pt checkpoint (default: results/mnist/ion/run_42.pt)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=_ROOT / "configs",
        help="Config directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "results" / "drift" / "drift.json",
        help="Output path for drift.json",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=50,
        help="Max batches for drift computation (default: 50)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_dir = args.config_dir
    base_path = config_dir / "base.yaml"
    mnist_path = config_dir / "mnist.yaml"

    if args.checkpoint is None:
        ckpt_path = _ROOT / "results" / "mnist" / "ion" / "run_42.pt"
    else:
        ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        print(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run MNIST first: python -m src.run_mnist --epochs 5 --seeds 42"
        )
        sys.exit(1)

    if not base_path.exists() or not mnist_path.exists():
        print(f"Config not found. Need {base_path} and {mnist_path}")
        sys.exit(1)

    config = load_config(base_path, mnist_path)
    config["model"] = "ion"
    model = build_ion(config)
    load_checkpoint(ckpt_path, model, device)
    model.to(device)

    _, _, test_loader = get_mnist_loaders(
        data_root=config.get("data_root", "data/mnist"),
        batch_size=64,
        num_workers=0,
        seed=42,
        download=config.get("download", True),
    )

    print("Computing invariant drift (ION universal)...")
    result = compute_drift_over_dataset(
        model, test_loader, device, max_batches=args.max_batches
    )

    # Aggregate per_index across batches (universal: same layers per batch)
    per_batch = result.get("per_batch_results", [])
    per_index_list = [r["per_index"] for r in per_batch if r["per_index"] is not None]
    if not per_index_list:
        print("No per-index drift computed")
        sys.exit(1)

    # Stack and mean over batches; convert to list for JSON
    stacked = torch.stack([p.float() for p in per_index_list])
    mean_per_layer = stacked.mean(dim=0).cpu().numpy().tolist()

    drift_data = {
        "type": "universal",
        "per_index": mean_per_layer,
        "mean_drift": result["mean_drift"],
        "std_drift": result["std_drift"],
        "n_batches": result["n_batches"],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(drift_data, f, indent=2)

    print(f"Saved drift data to {out_path}")
    print(f"  Mean drift: {result['mean_drift']:.6f} ± {result['std_drift']:.6f}")


if __name__ == "__main__":
    main()
