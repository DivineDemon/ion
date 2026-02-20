#!/usr/bin/env python3
"""
Phase A9: Ablations for λ (inductive loss weight) and p_dim (invariant size).

Runs ION with varying λ and p_dim on one length-gen task (cumsum) and on MNIST;
saves per-seed results to results/ablations/ with clear naming.

Usage:
  python -m src.run_ablations                    # run both lambda and p_dim on cumsum + MNIST
  python -m src.run_ablations --sweep lambda    # lambda only
  python -m src.run_ablations --sweep p_dim     # p_dim only
  python -m src.run_ablations --task cumsum     # one task only
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

from src.data.length_gen import TASK_CUMSUM, get_length_gen_loaders
from src.data.mnist_loader import get_mnist_loaders
from src.models import (
    IONRecurrent,
    IONUniversal,
    count_parameters,
    suggest_ion_recurrent_dims,
    suggest_ion_universal_dims,
)
from src.training.eval import evaluate_length_gen, evaluate_test, load_checkpoint
from src.training.train import run_training, _forward_with_inductive


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


def get_seeds(config: dict, n: int | None = None) -> list[int]:
    if "seeds" in config and config["seeds"]:
        return list(config["seeds"])[: (n or len(config["seeds"]))]
    runs = n or config.get("runs_per_experiment", 5)
    base = config.get("seed", 42)
    return [base + i * 1000 for i in range(runs)]


# ---------------------------------------------------------------------------
# Length-gen (cumsum): build model and run one (sweep_value, seed)
# ---------------------------------------------------------------------------

LAMBDA_VALUES = [0.1, 0.3, 0.5, 0.7, 1.0]
P_DIM_VALUES = [4, 8, 16, 32]
ABLATION_SEEDS_COUNT = 5


def build_length_gen_ion(config: dict, param_match: bool = True) -> IONRecurrent:
    """Build ION recurrent for length-gen from config; optionally param-match to GRU."""
    from src.models import GRUBaseline

    input_dim = config.get("input_dim", 1)
    output_type = config.get("output_type", "regression")
    num_classes = config.get("num_classes", 2)
    num_layers = config.get("num_layers", 2)
    dropout = config.get("dropout", 0.0)
    hidden_dim = config.get("hidden_dim", 64)
    p_dim = config.get("p_dim", 16)

    if param_match:
        gru = GRUBaseline(
            input_dim=input_dim,
            hidden_dim=config.get("hidden_dim", 64),
            num_layers=num_layers,
            output_type=output_type,
            num_classes=num_classes if output_type == "classification" else None,
            dropout=dropout,
        )
        target_params = count_parameters(gru)
        num_co = num_classes if output_type == "classification" else 1
        suggested = suggest_ion_recurrent_dims(
            target_params=target_params,
            input_dim=input_dim,
            num_classes_or_1=num_co,
            num_layers=num_layers,
            output_type=output_type,
            tolerance=0.10,
        )
        if suggested:
            hidden_dim = suggested["hidden_dim"]
            p_dim = suggested["p_dim"]

    return IONRecurrent(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        p_dim=p_dim,
        num_layers=num_layers,
        output_type=output_type,
        num_classes=num_classes if output_type == "classification" else None,
        dropout=dropout,
    )


def build_length_gen_loaders(config: dict, seed: int):
    return get_length_gen_loaders(
        task=config["task"],
        train_min_len=config["train_min_len"],
        train_max_len=config["train_max_len"],
        test_lengths=config["test_lengths"],
        batch_size=config.get("batch_size", 32),
        train_samples=config.get("train_samples", 10_000),
        val_samples=config.get("val_samples", 1_000),
        test_samples_per_length=config.get("test_samples_per_length", 500),
        seed=seed,
        num_workers=config.get("num_workers", 0),
        **({"vocab_size": config["vocab_size"]} if "vocab_size" in config else {}),
    )


def run_one_length_gen_ablation(
    config: dict,
    sweep_type: str,
    sweep_value: float | int,
    seed: int,
    result_dir: Path,
    device: torch.device,
) -> dict:
    """Train and evaluate ION on cumsum for one (sweep_value, seed). Returns metrics dict."""
    param_match = sweep_type == "lambda"
    model = build_length_gen_ion(config, param_match=param_match)
    train_loader, val_loader, test_loaders = build_length_gen_loaders(config, seed)

    run_training(
        model,
        train_loader,
        val_loader,
        config,
        result_dir=result_dir,
        run_seed=seed,
    )

    ckpt_path = result_dir / f"run_{seed}.pt"
    model = build_length_gen_ion(config, param_match=param_match)
    load_checkpoint(ckpt_path, model, device)
    model.to(device)
    model.eval()

    lg_result = evaluate_length_gen(model, test_loaders, device, config)
    task_type = config.get("output_type", "regression")
    metric_key = "mean_accuracy" if task_type == "classification" else "mean_mse"
    metric_value = lg_result.get("mean_accuracy") or lg_result.get("mean_mse")

    return {
        "sweep": sweep_type,
        "sweep_value": sweep_value,
        "task": "cumsum",
        "seed": seed,
        "metric": metric_key,
        "metric_value": metric_value,
        "per_length": lg_result.get("per_length"),
        "lengths": lg_result.get("lengths"),
    }


# ---------------------------------------------------------------------------
# MNIST: build ION and run one (sweep_value, seed)
# ---------------------------------------------------------------------------


def build_mnist_ion(config: dict, param_match: bool = True) -> IONUniversal:
    """Build ION universal for MNIST; optionally param-match to MLP."""
    from src.models import MLPBaseline

    num_layers = config["num_layers"]
    hidden_dim = config["hidden_dim"]
    p_dim = config.get("p_dim", 32)

    if param_match:
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


def get_mnist_loaders_from_config(config: dict, seed: int):
    return get_mnist_loaders(
        data_root=config.get("data_root", "data/mnist"),
        batch_size=config.get("batch_size", 64),
        train_size=config.get("train_size", 50_000),
        val_size=config.get("val_size", 10_000),
        num_workers=config.get("num_workers", 0),
        seed=seed,
        download=config.get("download", True),
    )


def run_one_mnist_ablation(
    config: dict,
    sweep_type: str,
    sweep_value: float | int,
    seed: int,
    result_dir: Path,
    device: torch.device,
) -> dict:
    """Train and evaluate ION on MNIST for one (sweep_value, seed). Returns metrics dict."""
    param_match = sweep_type == "lambda"
    model = build_mnist_ion(config, param_match=param_match)
    train_loader, val_loader, test_loader = get_mnist_loaders_from_config(config, seed)

    run_config = {
        **config,
        "output_type": config.get("output_type", "classification"),
        "num_classes": config.get("num_classes", 10),
        "lr": config.get("lr", 1e-3),
        "lambda_ind": config.get("lambda_ind", 0.5),
    }

    run_training(
        model,
        train_loader,
        val_loader,
        run_config,
        result_dir=result_dir,
        run_seed=seed,
    )

    ckpt_path = result_dir / f"run_{seed}.pt"
    model = build_mnist_ion(config, param_match=param_match)
    load_checkpoint(ckpt_path, model, device)
    model.to(device)

    test_result = evaluate_test(model, test_loader, device, run_config)
    test_acc = test_result["test_accuracy"]

    return {
        "sweep": sweep_type,
        "sweep_value": sweep_value,
        "task": "mnist",
        "seed": seed,
        "metric": "test_accuracy",
        "metric_value": test_acc,
        "test_loss": test_result.get("test_loss"),
    }


# ---------------------------------------------------------------------------
# Main: run sweeps and save to results/ablations/
# ---------------------------------------------------------------------------


def run_ablations(
    config_dir: Path,
    results_dir: Path,
    sweep: str,
    task: str | None,
    seeds: list[int],
    device: torch.device,
    epochs_override: int | None = None,
) -> None:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    base_path = config_dir / "base.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Need {base_path}")

    run_cumsum = task is None or task == "cumsum"
    run_mnist = task is None or task == "mnist"

    # Length-gen config (cumsum + ION)
    protocol_path = config_dir / "length_gen" / "protocol.yaml"
    task_cfg = config_dir / "length_gen" / "cumsum.yaml"
    ion_cfg = config_dir / "length_gen" / "cumsum_ion.yaml"
    if run_cumsum and protocol_path.exists() and task_cfg.exists() and ion_cfg.exists():
        lg_config = load_config(base_path, protocol_path, task_cfg, ion_cfg)
        lg_config["model"] = "ion"
        lg_config["device"] = str(device)
    else:
        lg_config = None

    # MNIST config
    mnist_path = config_dir / "mnist.yaml"
    if run_mnist and mnist_path.exists():
        mnist_config = load_config(base_path, mnist_path)
        mnist_config["model"] = "ion"
        mnist_config["device"] = str(device)
        mnist_config.setdefault("p_dim", 32)
        if epochs_override is not None:
            mnist_config["epochs"] = epochs_override
    else:
        mnist_config = None

    if sweep == "lambda":
        values = LAMBDA_VALUES
    else:
        values = P_DIM_VALUES

    for value in values:
        for seed in seeds:
            if run_cumsum and lg_config is not None:
                cfg = dict(lg_config)
                if sweep == "lambda":
                    cfg["lambda_ind"] = value
                else:
                    cfg["p_dim"] = value
                subdir = results_dir / f"{sweep}_{value}_cumsum"
                subdir.mkdir(parents=True, exist_ok=True)
                print(f"  cumsum {sweep}={value} seed={seed}")
                result = run_one_length_gen_ablation(
                    cfg, sweep, value, seed, subdir, device
                )
                out_path = results_dir / f"{sweep}_{value}_cumsum_seed_{seed}.json"
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)

            if run_mnist and mnist_config is not None:
                cfg = dict(mnist_config)
                if sweep == "lambda":
                    cfg["lambda_ind"] = value
                else:
                    cfg["p_dim"] = value
                subdir = results_dir / f"{sweep}_{value}_mnist"
                subdir.mkdir(parents=True, exist_ok=True)
                print(f"  mnist {sweep}={value} seed={seed}")
                result = run_one_mnist_ablation(
                    cfg, sweep, value, seed, subdir, device
                )
                out_path = results_dir / f"{sweep}_{value}_mnist_seed_{seed}.json"
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablations: lambda and p_dim on cumsum + MNIST")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=_ROOT / "configs",
        help="Config directory",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_ROOT / "results" / "ablations",
        help="Results directory (default: results/ablations)",
    )
    parser.add_argument(
        "--sweep",
        choices=["lambda", "p_dim", "both"],
        default="both",
        help="Sweep to run: lambda, p_dim, or both",
    )
    parser.add_argument(
        "--task",
        choices=["cumsum", "mnist"],
        default=None,
        help="Task to run (default: both cumsum and mnist)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds (default: 5 from config)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=ABLATION_SEEDS_COUNT,
        help=f"Number of seeds if not using config seeds (default: {ABLATION_SEEDS_COUNT})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs for MNIST ablations (for quick runs)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = args.config_dir / "base.yaml"
    base_config = load_config(base_path) if base_path.exists() else {}
    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = get_seeds(base_config, n=args.n_seeds)

    sweeps = ["lambda", "p_dim"] if args.sweep == "both" else [args.sweep]
    for sw in sweeps:
        print(f"Running sweep: {sw}")
        run_ablations(
            args.config_dir,
            args.results_dir,
            sweep=sw,
            task=args.task,
            seeds=seeds,
            device=device,
            epochs_override=args.epochs,
        )

    print(f"Done. Results in {args.results_dir}")


if __name__ == "__main__":
    main()
