#!/usr/bin/env python3
"""
MNIST experiments: MLP baseline vs MLP + universal ION.

Uses 5 seeds per model; parameter-matched ION; optional robustness test
(Gaussian noise on test set). Saves test accuracy, loss curves (in run JSON history),
and optional robustness to results/mnist/.

Usage:
  python -m src.run_mnist
  python -m src.run_mnist --no-robustness --epochs 10
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
from src.models import (
    MLPBaseline,
    IONUniversal,
    count_parameters,
    suggest_ion_universal_dims,
)
from src.training.eval import evaluate_test, load_checkpoint
from src.training.train import run_training, _forward_with_inductive
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


def build_mlp(config: dict) -> MLPBaseline:
    return MLPBaseline(
        input_size=config["input_size"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        output_size=config["output_size"],
        dropout=config.get("dropout", 0.0),
    )


def build_ion(config: dict, param_match_to_mlp: bool = True) -> IONUniversal:
    num_layers = config["num_layers"]
    hidden_dim = config["hidden_dim"]
    p_dim = config.get("p_dim", 32)
    if param_match_to_mlp:
        mlp = build_mlp(config)
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


def build_model(config: dict, model_name: str) -> torch.nn.Module:
    if model_name == "mlp":
        return build_mlp(config)
    if model_name == "ion":
        return build_ion(config)
    raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def get_loaders(config: dict, seed: int):
    return get_mnist_loaders(
        data_root=config.get("data_root", "data/mnist"),
        batch_size=config.get("batch_size", 64),
        train_size=config.get("train_size", 50_000),
        val_size=config.get("val_size", 10_000),
        num_workers=config.get("num_workers", 0),
        seed=seed,
        download=config.get("download", True),
    )


# ---------------------------------------------------------------------------
# Robustness: accuracy under Gaussian noise
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_robustness_accuracy(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    noise_std: float = 0.1,
) -> float:
    """Test accuracy with Gaussian noise added to inputs (scale [0,1])."""
    model.eval()
    total_correct = 0
    total_samples = 0
    for batch in test_loader:
        x, y = batch[0].to(device), batch[1].to(device)
        x_noisy = x + noise_std * torch.randn_like(x, device=x.device)
        x_noisy = x_noisy.clamp(0.0, 1.0)
        logits, _ = _forward_with_inductive(model, x_noisy, return_inductive=False)
        pred = logits.argmax(dim=-1)
        total_correct += (pred == y).sum().item()
        total_samples += y.size(0)
    return total_correct / max(1, total_samples)


# ---------------------------------------------------------------------------
# Run one model over all seeds; collect test accuracy, curves, robustness
# ---------------------------------------------------------------------------


def run_one_model(
    model_name: str,
    config: dict,
    result_dir: Path,
    device: torch.device,
    run_robustness: bool = True,
    noise_std: float = 0.1,
) -> dict:
    """
    Train and evaluate one model (mlp or ion) for all seeds.
    Saves per-run JSON (with test_accuracy, history, optional robustness) and returns summary.
    """
    seeds = get_seeds(config)
    train_loader, val_loader, test_loader = get_loaders(config, seeds[0])

    run_config = {
        **config,
        "output_type": config.get("output_type", "classification"),
        "num_classes": config.get("num_classes", 10),
        "lr": config.get("lr", 1e-3),
        "lambda_ind": config.get("lambda_ind", 0.5) if model_name == "ion" else 0.0,
    }

    test_accuracies: list[float] = []
    robustness_accuracies: list[float] = [] if run_robustness else []

    for run_seed in seeds:
        model = build_model(config, model_name)
        run_training(
            model,
            train_loader,
            val_loader,
            run_config,
            result_dir=result_dir,
            run_seed=run_seed,
        )

        ckpt_path = result_dir / f"run_{run_seed}.pt"
        model = build_model(config, model_name)
        load_checkpoint(ckpt_path, model, device)
        model.to(device)

        test_result = evaluate_test(model, test_loader, device, run_config)
        test_acc = test_result["test_accuracy"]
        test_accuracies.append(test_acc)

        if run_robustness:
            robust_acc = evaluate_robustness_accuracy(
                model, test_loader, device, noise_std=noise_std
            )
            robustness_accuracies.append(robust_acc)

        # Update run JSON with test_accuracy and optional robustness
        result_path = result_dir / f"run_{run_seed}.json"
        if result_path.exists():
            with open(result_path) as f:
                data = json.load(f)
            data["test_accuracy"] = test_acc
            data["test_loss"] = test_result.get("test_loss")
            if run_robustness:
                data["robustness_accuracy"] = robust_acc
                data["robustness_noise_std"] = noise_std
            with open(result_path, "w") as f:
                json.dump(data, f, indent=2)

    mean_acc, std_acc = mean_std(test_accuracies)
    summary = {
        "model": model_name,
        "seeds": seeds,
        "test_accuracy_mean": mean_acc,
        "test_accuracy_std": std_acc,
        "test_accuracies": test_accuracies,
    }
    if run_robustness and robustness_accuracies:
        mean_rob, std_rob = mean_std(robustness_accuracies)
        summary["robustness_accuracy_mean"] = mean_rob
        summary["robustness_accuracy_std"] = std_rob
        summary["robustness_accuracies"] = robustness_accuracies
        summary["robustness_noise_std"] = noise_std
    return summary


# ---------------------------------------------------------------------------
# Save aggregate artifacts to results/mnist/
# ---------------------------------------------------------------------------


def write_mnist_artifacts(
    results_root: Path,
    summaries: list[dict],
    run_robustness: bool,
) -> None:
    """
    Save test accuracy summary, loss curve data (reference to run JSONs), and optional robustness.
    """
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    # 1) Test accuracy summary (mean ± std per model)
    accuracy_path = results_root / "test_accuracy_summary.json"
    with open(accuracy_path, "w") as f:
        json.dump(
            {
                "models": [s["model"] for s in summaries],
                "summaries": summaries,
            },
            f,
            indent=2,
        )

    # 2) CSV table: model, test_accuracy_mean, test_accuracy_std, n [, robustness ...]
    n_seeds = lambda s: len(s.get("seeds", s.get("test_accuracies", [])))
    csv_path = results_root / "mnist_results.csv"
    with open(csv_path, "w") as f:
        if run_robustness and any("robustness_accuracy_mean" in s for s in summaries):
            f.write("model,test_accuracy_mean,test_accuracy_std,n,robustness_accuracy_mean,robustness_accuracy_std\n")
            for s in summaries:
                n = n_seeds(s)
                rob_mean = s.get("robustness_accuracy_mean")
                rob_std = s.get("robustness_accuracy_std")
                rmean = f"{rob_mean:.6f}" if rob_mean is not None else ""
                rstd = f"{rob_std:.6f}" if rob_std is not None else ""
                f.write(
                    f"{s['model']},{s['test_accuracy_mean']:.6f},{s['test_accuracy_std']:.6f},{n},{rmean},{rstd}\n"
                )
        else:
            f.write("model,test_accuracy_mean,test_accuracy_std,n\n")
            for s in summaries:
                n = n_seeds(s)
                f.write(f"{s['model']},{s['test_accuracy_mean']:.6f},{s['test_accuracy_std']:.6f},{n}\n")

    # 3) Loss curves: each run_<seed>.json already has "history" (train_losses, val_losses, val_metrics).
    #    Write a README pointing to run JSONs for curve data.
    readme_path = results_root / "README.md"
    readme_lines = [
        "# MNIST experiment results",
        "",
        "## Contents",
        "- `mlp/run_<seed>.json`: per-seed results for MLP (history = train/val loss curves).",
        "- `ion/run_<seed>.json`: per-seed results for ION (history = train/val loss curves).",
        "- `test_accuracy_summary.json`: mean ± std test accuracy per model.",
        "- `mnist_results.csv`: table of test accuracy (and optional robustness).",
        "",
        "## Loss curves",
        "Each `run_<seed>.json` contains a `history` object with `train_losses`, `val_losses`, `train_metrics`, `val_metrics` (one value per epoch). Use these to plot train/val loss and accuracy curves.",
    ]
    readme_path.write_text("\n".join(readme_lines) + "\n")


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


CONFIG_DIR = _ROOT / "configs"


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST: MLP vs MLP+ION")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=CONFIG_DIR,
        help="Config directory",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_ROOT / "results" / "mnist",
        help="Results root (default: results/mnist)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        choices=["mlp", "ion"],
        help="Models to run (default: mlp ion)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs",
    )
    parser.add_argument(
        "--no-robustness",
        action="store_true",
        help="Skip robustness test (Gaussian noise on test set)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.1,
        help="Gaussian noise std for robustness test (default: 0.1)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Override seeds as comma-separated (e.g. 42,123,456)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_path = args.config_dir / "base.yaml"
    mnist_path = args.config_dir / "mnist.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Need {base_path}")
    config = load_config(base_path, mnist_path)
    config["device"] = str(device)

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.seeds is not None:
        config["seeds"] = [int(s) for s in args.seeds.split(",")]
    if args.models is not None:
        models = args.models
    else:
        models = ["mlp", "ion"]

    config.setdefault("lr", 1e-3)
    config.setdefault("lambda_ind", 0.5)

    results_dir = Path(args.results_dir)
    run_robustness = not args.no_robustness
    summaries: list[dict] = []

    for model_name in models:
        result_dir = results_dir / model_name
        result_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running model={model_name} -> {result_dir}")
        summary = run_one_model(
            model_name,
            config,
            result_dir,
            device,
            run_robustness=run_robustness,
            noise_std=args.noise_std,
        )
        summaries.append(summary)

    write_mnist_artifacts(results_dir, summaries, run_robustness)
    print(f"Done. Results in {results_dir}")


if __name__ == "__main__":
    main()
