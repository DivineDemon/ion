#!/usr/bin/env python3
"""
LRA (Long Range Arena) experiments: ListOps classification.

Trains Transformer baseline or ION-Transformer on LRA ListOps; reports val/test accuracy.
Requires LRA data at data/lra/lra_release/listops-1000/ (see scripts/fetch_lra_listops.sh).

Usage:
  python -m src.run_lra --task listops --model transformer
  python -m src.run_lra --task listops --model ion --epochs 10 --seeds 42,123
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml
import torch

from src.data.lra import get_lra_listops_loaders, get_lra_image_loaders
from src.models import (
    TransformerALiBi,
    TransformerBaseline,
    IONTransformer,
    count_parameters,
)
from src.training.eval import evaluate_test, load_checkpoint
from src.training.train import run_training
from src.analysis.stats import mean_std


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


def get_seeds(config: dict) -> list[int]:
    if "seeds" in config and config["seeds"]:
        return list(config["seeds"])
    n = config.get("runs_per_experiment", 5)
    base = config.get("seed", 42)
    return [base + i * 1000 for i in range(n)]


def build_transformer(config: dict) -> TransformerBaseline:
    return TransformerBaseline(
        input_dim=config["input_dim"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        max_len=config["max_len"],
        output_type=config.get("output_type", "classification"),
        num_classes=config.get("num_classes", 10),
        dropout=config.get("dropout", 0.1),
    )


def build_transformer_alibi(config: dict) -> TransformerALiBi:
    return TransformerALiBi(
        input_dim=config["input_dim"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        max_len=config["max_len"],
        output_type=config.get("output_type", "classification"),
        num_classes=config.get("num_classes", 10),
        dropout=config.get("dropout", 0.1),
    )


def build_ion_transformer(config: dict) -> IONTransformer:
    return IONTransformer(
        input_dim=config["input_dim"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        p_dim=config.get("p_dim", 32),
        max_len=config["max_len"],
        output_type=config.get("output_type", "classification"),
        num_classes=config.get("num_classes", 10),
        dropout=config.get("dropout", 0.1),
    )


def build_model(config: dict, model_name: str) -> torch.nn.Module:
    if model_name == "transformer":
        return build_transformer(config)
    if model_name == "transformer_alibi":
        return build_transformer_alibi(config)
    if model_name == "ion":
        return build_ion_transformer(config)
    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LRA experiments (ListOps)")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=_ROOT / "configs",
        help="Config directory",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_ROOT / "results" / "lra",
        help="Results root",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="listops",
        choices=["listops", "image"],
        help="LRA task: listops or image (CIFAR-10 as sequence)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ion",
        choices=["transformer", "transformer_alibi", "ion"],
        help="Model: transformer, transformer_alibi (ALiBi), or ion (ION-Transformer)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds (e.g. 42,123,456)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Device to use (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    from src.training import get_device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Using device: {device}")
    base_path = args.config_dir / "base.yaml"
    task = args.task
    lra_path = args.config_dir / "lra" / ("listops.yaml" if task == "listops" else "image.yaml")
    if not base_path.exists():
        raise FileNotFoundError(f"Need {base_path}")
    if not lra_path.exists():
        raise FileNotFoundError(f"Need {lra_path}")
    config = load_config(base_path, lra_path)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.seeds is not None:
        config["seeds"] = [int(s) for s in args.seeds.split(",")]

    if task == "listops":
        train_loader, val_loader, test_loader, input_dim = get_lra_listops_loaders(
            data_root=config.get("data_root", "data/lra"),
            batch_size=config.get("batch_size", 32),
            max_length=config.get("max_length", 512),
            num_workers=config.get("num_workers", 0),
            seed=config.get("seed", 42),
        )
    else:
        train_loader, val_loader, test_loader, input_dim = get_lra_image_loaders(
            data_root=config.get("data_root", "data/cifar10"),
            batch_size=config.get("batch_size", 32),
            max_length=config.get("max_length", 1024),
            num_workers=config.get("num_workers", 0),
            seed=config.get("seed", 42),
        )
    config["input_dim"] = input_dim
    config["model"] = args.model
    config["task"] = args.task
    config["device"] = str(device)  # so run_training uses selected device
    config.setdefault("output_type", "classification")
    config.setdefault("num_classes", 10)
    if args.model == "ion":
        config.setdefault("lambda_ind", 0.5)

    seeds = get_seeds(config)
    result_dir = args.results_dir / args.task / args.model
    result_dir.mkdir(parents=True, exist_ok=True)

    test_accuracies: list[float] = []
    for run_seed in seeds:
        model = build_model(config, args.model)
        print(f"Run seed {run_seed} -> {result_dir}")
        run_training(
            model,
            train_loader,
            val_loader,
            config,
            result_dir=result_dir,
            run_seed=run_seed,
        )
        ckpt_path = result_dir / f"run_{run_seed}.pt"
        model = build_model(config, args.model)
        load_checkpoint(ckpt_path, model, device)
        model.to(device)
        test_result = evaluate_test(model, test_loader, device, config)
        test_accuracies.append(test_result["test_accuracy"])

    mean_acc, std_acc = mean_std(test_accuracies)
    summary = {
        "task": args.task,
        "model": args.model,
        "seeds": seeds,
        "test_accuracy_mean": mean_acc,
        "test_accuracy_std": std_acc,
        "test_accuracies": test_accuracies,
    }
    summary_path = result_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Test accuracy {mean_acc:.4f} ± {std_acc:.4f}. Summary: {summary_path}")


if __name__ == "__main__":
    main()
