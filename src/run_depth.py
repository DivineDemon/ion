#!/usr/bin/env python3
"""
Depth-stability experiments: baseline MLP vs universal ION at depths 4, 8, 16, 32.

Uses MNIST; same width (hidden_dim) across depths; 5 seeds per (depth, model).
Outputs accuracy vs depth and optional convergence (epochs to reach target accuracy)
to results/depth/.

Usage:
  python -m src.run_depth
  python -m src.run_depth --depths 4 8 --models mlp ion --epochs 10
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

from src.data.mnist_loader import get_mnist_loaders
from src.data.cifar_loader import get_cifar_loaders
from src.models import (
    MLPBaseline,
    IONUniversal,
    count_parameters,
    suggest_ion_universal_dims,
)
from src.training.eval import evaluate_test, load_checkpoint
from src.training.train import run_training
from src.analysis.stats import mean_std


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------


def build_mlp(config: dict, depth: int) -> MLPBaseline:
    return MLPBaseline(
        input_size=config["input_size"],
        hidden_dim=config["hidden_dim"],
        num_layers=depth,
        output_size=config["output_size"],
        dropout=config.get("dropout", 0.0),
    )


def build_ion(config: dict, depth: int, param_match_to_mlp: bool = True) -> IONUniversal:
    hidden_dim = config["hidden_dim"]
    p_dim = config.get("p_dim", 32)
    if param_match_to_mlp:
        mlp = build_mlp(config, depth)
        target_params = count_parameters(mlp)
        suggested = suggest_ion_universal_dims(
            target_params=target_params,
            input_size=config["input_size"],
            output_size=config["output_size"],
            num_layers=depth,
            tolerance=0.10,
        )
        if suggested:
            hidden_dim = suggested["hidden_dim"]
            p_dim = suggested["p_dim"]
    return IONUniversal(
        input_size=config["input_size"],
        hidden_dim=hidden_dim,
        num_layers=depth,
        output_size=config["output_size"],
        p_dim=p_dim,
        dropout=config.get("dropout", 0.0),
    )


def build_model(config: dict, model_name: str, depth: int) -> torch.nn.Module:
    if model_name == "mlp":
        return build_mlp(config, depth)
    if model_name == "ion":
        return build_ion(config, depth)
    raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def get_loaders(config: dict, seed: int):
    dataset = config.get("dataset", "mnist")
    common = dict(
        batch_size=config.get("batch_size", 64),
        num_workers=config.get("num_workers", 0),
        seed=seed,
        download=config.get("download", True),
    )
    if dataset == "cifar":
        return get_cifar_loaders(
            data_root=config.get("data_root", "data/cifar10"),
            train_size=config.get("train_size", 45_000),
            val_size=config.get("val_size", 5_000),
            **common,
        )
    return get_mnist_loaders(
        data_root=config.get("data_root", "data/mnist"),
        train_size=config.get("train_size", 50_000),
        val_size=config.get("val_size", 10_000),
        **common,
    )


# ---------------------------------------------------------------------------
# Run one (depth, model) over all seeds; aggregate accuracy and optional convergence
# ---------------------------------------------------------------------------


def _epochs_to_reach_accuracy(history: dict, target: float, task_type: str) -> int | None:
    """First epoch (1-based) at which val_metric >= target, or None."""
    if task_type != "classification":
        return None
    vals = history.get("val_metrics", [])
    for epoch_1based, v in enumerate(vals, start=1):
        if v >= target:
            return epoch_1based
    return None


def run_one_depth_model(
    depth: int,
    model_name: str,
    config: dict,
    result_dir: Path,
    device: torch.device,
) -> dict:
    """
    Train and evaluate one (depth, model) for all seeds.
    Returns summary: list of test accuracies, list of epochs_to_95 (optional), mean±std.
    """
    seeds = get_seeds(config)
    train_loader, val_loader, test_loader = get_loaders(config, seeds[0])
    lambda_ind = 0.0
    if model_name == "ion":
        lambda_ind = config.get("lambda_ind", 0.5)
        depth_thresh = config.get("depth_threshold_for_lambda", 16)
        if depth >= depth_thresh:
            lambda_ind = config.get("lambda_ind_deep", 0.15)
    run_config = {
        **config,
        "output_type": config.get("output_type", "classification"),
        "num_classes": config.get("num_classes", 10),
        "lambda_ind": lambda_ind,
    }

    test_accuracies: list[float] = []
    epochs_to_target: list[int | None] = []
    target_acc = config.get("target_accuracy_for_convergence")

    for run_seed in seeds:
        model = build_model(config, model_name, depth)
        run_training(
            model,
            train_loader,
            val_loader,
            run_config,
            result_dir=result_dir,
            run_seed=run_seed,
        )

        ckpt_path = result_dir / f"run_{run_seed}.pt"
        model = build_model(config, model_name, depth)
        load_checkpoint(ckpt_path, model, device)
        model.to(device)
        test_result = evaluate_test(model, test_loader, device, run_config)
        test_acc = test_result["test_accuracy"]
        test_accuracies.append(test_acc)

        result_path = result_dir / f"run_{run_seed}.json"
        if result_path.exists():
            with open(result_path) as f:
                data = json.load(f)
            data["test_accuracy"] = test_acc
            if target_acc is not None:
                if "history" in data:
                    ep = _epochs_to_reach_accuracy(
                        data["history"], target_acc, run_config["output_type"]
                    )
                    epochs_to_target.append(ep)
                    data["epochs_to_target_accuracy"] = ep
                else:
                    epochs_to_target.append(None)
            with open(result_path, "w") as f:
                json.dump(data, f, indent=2)

    mean_acc, std_acc = mean_std(test_accuracies)
    summary = {
        "depth": depth,
        "model": model_name,
        "seeds": seeds,
        "test_accuracy_mean": mean_acc,
        "test_accuracy_std": std_acc,
        "test_accuracies": test_accuracies,
    }
    if target_acc is not None and any(e is not None for e in epochs_to_target):
        valid = [e for e in epochs_to_target if e is not None]
        if valid:
            mean_ep, std_ep = mean_std(valid)
            summary["epochs_to_target_mean"] = mean_ep
            summary["epochs_to_target_std"] = std_ep
            summary["epochs_to_target_values"] = epochs_to_target
    return summary


# ---------------------------------------------------------------------------
# Aggregate tables: accuracy vs depth, optional convergence
# ---------------------------------------------------------------------------


def write_aggregate_tables(results_root: Path, depth_summaries: list[dict]) -> None:
    """Write accuracy_vs_depth.csv and optional convergence CSV + summary.json."""
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    depths = sorted({s["depth"] for s in depth_summaries})
    models = sorted({s["model"] for s in depth_summaries})

    # accuracy_vs_depth.csv: rows = (depth, model), columns = mean, std, n
    n_runs = lambda s: len(s.get("test_accuracies", s.get("seeds", [])))
    acc_path = results_root / "accuracy_vs_depth.csv"
    with open(acc_path, "w") as f:
        f.write("depth,model,mean_accuracy,std_accuracy,n\n")
        for s in depth_summaries:
            n = n_runs(s)
            f.write(
                f"{s['depth']},{s['model']},{s['test_accuracy_mean']:.6f},{s['test_accuracy_std']:.6f},{n}\n"
            )

    # Wide table: rows = depth, columns = mlp_mean±std, ion_mean±std
    wide_path = results_root / "accuracy_vs_depth_table.csv"
    with open(wide_path, "w") as f:
        header = "depth," + ",".join(f"{m}_mean,{m}_std" for m in models)
        f.write(header + "\n")
        for d in depths:
            row = [str(d)]
            for m in models:
                s = next((x for x in depth_summaries if x["depth"] == d and x["model"] == m), None)
                if s is not None:
                    row.append(f"{s['test_accuracy_mean']:.4f}")
                    row.append(f"{s['test_accuracy_std']:.4f}")
                else:
                    row.append("")
                    row.append("")
            f.write(",".join(row) + "\n")

    # Optional: convergence (epochs to target accuracy)
    if any("epochs_to_target_mean" in s for s in depth_summaries):
        conv_path = results_root / "convergence_vs_depth.csv"
        with open(conv_path, "w") as f:
            f.write("depth,model,epochs_to_target_mean,epochs_to_target_std\n")
            for s in depth_summaries:
                if "epochs_to_target_mean" in s:
                    f.write(
                        f"{s['depth']},{s['model']},"
                        f"{s['epochs_to_target_mean']:.2f},{s['epochs_to_target_std']:.2f}\n"
                    )

    summary_path = results_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "depths": depths,
                "models": models,
                "summaries": depth_summaries,
            },
            f,
            indent=2,
        )


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


CONFIG_DIR = _ROOT / "configs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Depth-stability experiments (MNIST or CIFAR-10)"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=CONFIG_DIR,
        help="Config directory",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_ROOT / "results" / "depth",
        help="Results root (default: results/depth)",
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=None,
        help="Override depths (e.g. 4 8 16 32)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        choices=["mlp", "ion"],
        help="Override models",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Override seeds as comma-separated (e.g. 42,123,456)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["mnist", "cifar"],
        help="Dataset: mnist (default) or cifar for CIFAR-10 depth experiment",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Device to use (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_path = args.config_dir / "base.yaml"
    depth_path = args.config_dir / "depth" / "config.yaml"
    if not depth_path.exists():
        depth_path = args.config_dir / "depth.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Need {base_path}")
    config = load_config(base_path, depth_path)

    # Dataset-specific config: mnist uses mnist.yaml; cifar uses depth/cifar.yaml
    dataset = args.dataset or config.get("dataset", "mnist")
    config["dataset"] = dataset
    if dataset == "cifar":
        cifar_path = args.config_dir / "depth" / "cifar.yaml"
        if cifar_path.exists():
            config = _deep_merge(config, yaml.safe_load(cifar_path.open()) or {})
    else:
        mnist_path = args.config_dir / "mnist.yaml"
        if mnist_path.exists():
            config = _deep_merge(config, yaml.safe_load(mnist_path.open()) or {})

    if args.depths is not None:
        config["depths"] = args.depths
    if args.models is not None:
        config["models"] = args.models
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.seeds is not None:
        config["seeds"] = [int(s) for s in args.seeds.split(",")]

    config["device"] = str(device)

    depths = config["depths"]
    models = config["models"]
    results_dir = Path(args.results_dir)
    if dataset == "cifar":
        results_dir = results_dir / "cifar"
    depth_summaries: list[dict] = []

    for depth in depths:
        for model_name in models:
            result_dir = results_dir / model_name / f"depth_{depth}"
            result_dir.mkdir(parents=True, exist_ok=True)
            print(f"Running depth={depth} model={model_name} -> {result_dir}")
            summary = run_one_depth_model(
                depth, model_name, config, result_dir, device
            )
            depth_summaries.append(summary)

    write_aggregate_tables(results_dir, depth_summaries)
    print(f"Done. Results and accuracy_vs_depth tables in {results_dir}")


if __name__ == "__main__":
    main()
