#!/usr/bin/env python3
"""
Length-generalization experiments: ION vs GRU vs LSTM on cumsum, parity, Dyck-1, Dyck-2.

Loads config (base + protocol + task + model); runs training and per-length evaluation
for each seed; writes raw per-seed metrics and accuracy-vs-length tables to results/length_gen/.

Usage:
  python -m src.run_length_gen --task cumsum --model ion
  python -m src.run_length_gen --task parity --model gru
  python -m src.run_length_gen --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml
import torch

from src.data.length_gen import (
    TASK_CUMSUM,
    TASK_DYCK1,
    TASK_DYCK2,
    TASK_LAST_TOKEN,
    TASK_PARITY,
    get_length_gen_loaders,
)
from src.models import (
    GRUBaseline,
    IONRecurrent,
    LSTMBaseline,
    count_parameters,
    suggest_ion_recurrent_dims,
)
from src.training.eval import evaluate_length_gen
from src.training.train import run_training


# ---------------------------------------------------------------------------
# Config loading and merging
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Lists and scalars are replaced."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(*config_paths: Path) -> dict:
    """Load and merge multiple YAML config files in order."""
    merged: dict = {}
    for path in config_paths:
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, data)
    return merged


def get_seeds(config: dict) -> list[int]:
    """Return list of seeds for multi-run experiment (5 per protocol)."""
    if "seeds" in config and config["seeds"]:
        return list(config["seeds"])
    n = config.get("runs_per_experiment", 5)
    base = config.get("seed", 42)
    return [base + i * 1000 for i in range(n)]


# ---------------------------------------------------------------------------
# Model and data building
# ---------------------------------------------------------------------------


def build_model(config: dict, param_match_to_gru: bool = True):
    """Build sequence model from config. For ION, optionally match GRU param count."""
    model_name = config.get("model", "gru")
    input_dim = config.get("input_dim", 1)
    output_type = config.get("output_type", "classification")
    num_classes = config.get("num_classes", 2)
    num_layers = config.get("num_layers", 2)
    dropout = config.get("dropout", 0.0)

    if model_name == "gru":
        return GRUBaseline(
            input_dim=input_dim,
            hidden_dim=config.get("hidden_dim", 64),
            num_layers=num_layers,
            output_type=output_type,
            num_classes=num_classes if output_type == "classification" else None,
            dropout=dropout,
        )
    if model_name == "lstm":
        return LSTMBaseline(
            input_dim=input_dim,
            hidden_dim=config.get("hidden_dim", 64),
            num_layers=num_layers,
            output_type=output_type,
            num_classes=num_classes if output_type == "classification" else None,
            dropout=dropout,
        )
    if model_name == "ion":
        hidden_dim = config.get("hidden_dim", 64)
        p_dim = config.get("p_dim", 16)
        if param_match_to_gru:
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
    raise ValueError(f"Unknown model: {model_name}")


def build_loaders(config: dict, seed: int):
    """Build train, val, and test loaders for length-gen from config."""
    task = config["task"]
    train_min = config["train_min_len"]
    train_max = config["train_max_len"]
    test_lengths = config["test_lengths"]
    batch_size = config.get("batch_size", 32)
    train_samples = config.get("train_samples", 10_000)
    val_samples = config.get("val_samples", 1_000)
    test_per_length = config.get("test_samples_per_length", 500)
    num_workers = config.get("num_workers", 0)

    task_kwargs = {}
    if task == TASK_CUMSUM:
        task_kwargs["vocab_size"] = config.get("vocab_size", 2)
    if task in (TASK_DYCK1, TASK_DYCK2):
        task_kwargs["valid_fraction"] = config.get("valid_fraction", 0.5)

    return get_length_gen_loaders(
        task=task,
        train_min_len=train_min,
        train_max_len=train_max,
        test_lengths=test_lengths,
        batch_size=batch_size,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples_per_length=test_per_length,
        seed=seed,
        num_workers=num_workers,
        **task_kwargs,
    )


# ---------------------------------------------------------------------------
# Run one (task, model) over all seeds; write artifacts
# ---------------------------------------------------------------------------


def run_one_task_model(
    task: str,
    model_name: str,
    config: dict,
    result_dir: Path,
    device: torch.device,
) -> None:
    """Train and evaluate one (task, model) for all seeds; write per-seed and table."""
    seeds = get_seeds(config)
    task_type = config.get("output_type", "classification")
    metric_key = "test_accuracy" if task_type == "classification" else "test_mse"

    all_per_length_by_seed: list[dict] = []

    for run_seed in seeds:
        # Fresh model per seed
        model = build_model(config)
        train_loader, val_loader, test_loaders = build_loaders(config, run_seed)

        result_dir.mkdir(parents=True, exist_ok=True)

        # Train (writes run_{seed}.json and run_{seed}.pt)
        run_training(
            model,
            train_loader,
            val_loader,
            config,
            result_dir=result_dir,
            run_seed=run_seed,
        )

        # Load best checkpoint and evaluate per length
        ckpt_path = result_dir / f"run_{run_seed}.pt"
        model = build_model(config)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)
        model.to(device)
        model.eval()

        length_gen_result = evaluate_length_gen(
            model, test_loaders, device, config
        )

        # Raw per-seed length-gen metrics
        run_length_gen = {
            "seed": run_seed,
            "per_length": length_gen_result["per_length"],
            "lengths": length_gen_result["lengths"],
            "mean_accuracy" if task_type == "classification" else "mean_mse": (
                length_gen_result.get("mean_accuracy") or length_gen_result.get("mean_mse")
            ),
        }
        length_gen_path = result_dir / f"run_{run_seed}_length_gen.json"
        with open(length_gen_path, "w") as f:
            json.dump(run_length_gen, f, indent=2)

        all_per_length_by_seed.append(length_gen_result["per_length"])

    # Accuracy-vs-length table: mean ± std over seeds per (model, length)
    lengths = config["test_lengths"]
    from src.analysis.stats import mean_std

    rows = []
    for length in lengths:
        values = [
            pl[length].get(metric_key)
            for pl in all_per_length_by_seed
        ]
        values = [v for v in values if v is not None]
        if not values:
            continue
        mean, std = mean_std(values)
        rows.append({"length": length, "mean": mean, "std": std})

    table_path = result_dir / "accuracy_vs_length.csv"
    with open(table_path, "w") as f:
        f.write("length,mean,std\n")
        for r in rows:
            f.write(f"{r['length']},{r['mean']:.6f},{r['std']:.6f}\n")

    # Also write a summary JSON for this (task, model)
    summary = {
        "task": task,
        "model": model_name,
        "seeds": seeds,
        "test_lengths": lengths,
        "accuracy_vs_length": {r["length"]: {"mean": r["mean"], "std": r["std"]} for r in rows},
    }
    summary_path = result_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def aggregate_tables(results_root: Path) -> None:
    """Write combined accuracy-vs-length table per task (all models) in results/length_gen/."""
    results_root = Path(results_root)
    for task_dir in results_root.iterdir():
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        # Collect summary.json per model
        model_summaries = {}
        for model_dir in task_dir.iterdir():
            if not model_dir.is_dir():
                continue
            summary_path = model_dir / "summary.json"
            if not summary_path.exists():
                continue
            with open(summary_path) as f:
                model_summaries[model_dir.name] = json.load(f)

        if not model_summaries:
            continue

        lengths = sorted(
            next(iter(model_summaries.values())).get("test_lengths", [])
        )
        table_path = task_dir / "accuracy_vs_length_table.csv"
        with open(table_path, "w") as f:
            header = "model," + ",".join(str(l) for l in lengths)
            f.write(header + "\n")
            for model_name, summary in sorted(model_summaries.items()):
                avl = summary.get("accuracy_vs_length", {})
                n_seeds = len(summary.get("seeds", []))
                cells = []
                for L in lengths:
                    c = avl.get(L, avl.get(str(L), {}))
                    m, s = c.get("mean", 0), c.get("std", 0)
                    if n_seeds <= 1:
                        cells.append(f"{m:.4f} (n=1)")
                    else:
                        cells.append(f"{m:.4f}±{s:.4f}")
                f.write(model_name + "," + ",".join(cells) + "\n")


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

LENGTH_GEN_TASKS = [TASK_CUMSUM, TASK_PARITY, TASK_DYCK1, TASK_DYCK2, TASK_LAST_TOKEN]
LENGTH_GEN_MODELS = ["ion", "gru", "lstm"]
CONFIG_DIR = _ROOT / "configs"
LENGTH_GEN_CONFIG_DIR = CONFIG_DIR / "length_gen"


def main() -> None:
    parser = argparse.ArgumentParser(description="Length-generalization experiments")
    parser.add_argument("--task", choices=LENGTH_GEN_TASKS, help="Task to run")
    parser.add_argument("--model", choices=LENGTH_GEN_MODELS, help="Model to run")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all task x model combinations",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=CONFIG_DIR,
        help="Config directory (default: repo configs/)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_ROOT / "results" / "length_gen",
        help="Results root (default: results/length_gen)",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only aggregate existing runs into tables",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs (default from config)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Override seeds as comma-separated (e.g. 42,123)",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    base_path = args.config_dir / "base.yaml"
    protocol_path = args.config_dir / "length_gen" / "protocol.yaml"
    if not protocol_path.exists():
        protocol_path = args.config_dir / "length_gen" / "protocol.yaml"
    if not base_path.exists() or not protocol_path.exists():
        raise FileNotFoundError(
            f"Need base.yaml and length_gen/protocol.yaml under {args.config_dir}"
        )

    if args.aggregate_only:
        aggregate_tables(args.results_dir)
        print("Aggregated tables written under", args.results_dir)
        return

    if args.all:
        pairs = [(t, m) for t in LENGTH_GEN_TASKS for m in LENGTH_GEN_MODELS]
    elif args.task and args.model:
        pairs = [(args.task, args.model)]
    else:
        parser.error("Specify --task and --model, or --all")

    for task, model_name in pairs:
        # Config: base + protocol + task config + task_model config
        task_cfg = args.config_dir / "length_gen" / f"{task}.yaml"
        model_cfg = args.config_dir / "length_gen" / f"{task}_{model_name}.yaml"
        if not task_cfg.exists():
            print(f"Skipping {task}: no {task_cfg}")
            continue
        if not model_cfg.exists():
            print(f"Skipping {task}/{model_name}: no {model_cfg}")
            continue

        config = load_config(base_path, protocol_path, task_cfg, model_cfg)
        if args.epochs is not None:
            config["epochs"] = args.epochs
        if args.seeds is not None:
            config["seeds"] = [int(s) for s in args.seeds.split(",")]
        result_dir = args.results_dir / task / model_name

        print(f"Running {task} / {model_name} -> {result_dir}")
        run_one_task_model(task, model_name, config, result_dir, device)

    # Aggregate tables (per-task table with all models)
    aggregate_tables(args.results_dir)
    print("Done. Results and accuracy_vs_length tables under", args.results_dir)


if __name__ == "__main__":
    main()
