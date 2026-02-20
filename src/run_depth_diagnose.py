#!/usr/bin/env python3
"""
Diagnostic run for depth-16 ION collapse: 2–3 seeds, per-epoch train/val loss and L_ind.
Writes results to results/depth/ion/depth_16/ and a summary to results/depth_diagnose/.
Run: python -m src.run_depth_diagnose
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse run_depth's config and run logic
from src.run_depth import (
    load_config,
    run_one_depth_model,
    get_seeds,
    _deep_merge,
)
import yaml

CONFIG_DIR = _ROOT / "configs"
DIAGNOSE_SEEDS = [42, 123, 456]
DEPTH_DIAGNOSE = 16
MODEL_DIAGNOSE = "ion"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Depth-16 ION diagnostic (train/val loss + L_ind)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for a quick run")
    args = parser.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_path = CONFIG_DIR / "base.yaml"
    mnist_path = CONFIG_DIR / "mnist.yaml"
    depth_path = CONFIG_DIR / "depth" / "config.yaml"
    if not depth_path.exists():
        depth_path = CONFIG_DIR / "depth.yaml"
    config = load_config(base_path, depth_path)
    if mnist_path.exists():
        config = _deep_merge(config, yaml.safe_load(mnist_path.open()) or {})
    config["depths"] = [DEPTH_DIAGNOSE]
    config["models"] = [MODEL_DIAGNOSE]
    config["seeds"] = DIAGNOSE_SEEDS
    if args.epochs is not None:
        config["epochs"] = args.epochs

    results_dir = _ROOT / "results" / "depth"
    result_dir = results_dir / MODEL_DIAGNOSE / f"depth_{DEPTH_DIAGNOSE}"
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running depth={DEPTH_DIAGNOSE} {MODEL_DIAGNOSE} (seeds {DIAGNOSE_SEEDS}) -> {result_dir}")

    summary = run_one_depth_model(
        DEPTH_DIAGNOSE,
        MODEL_DIAGNOSE,
        config,
        result_dir,
        device,
    )

    # Load per-run histories for diagnosis
    seeds = get_seeds(config)
    diagnosis = {
        "depth": DEPTH_DIAGNOSE,
        "model": MODEL_DIAGNOSE,
        "seeds": seeds,
        "summary": summary,
        "runs": [],
    }
    for run_seed in seeds:
        p = result_dir / f"run_{run_seed}.json"
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        hist = data.get("history", {})
        run_info = {
            "seed": run_seed,
            "best_val_metric": data.get("best_val_metric"),
            "test_accuracy": data.get("test_accuracy"),
            "final_train_loss": hist.get("train_losses", [])[-1] if hist.get("train_losses") else None,
            "final_val_loss": hist.get("val_losses", [])[-1] if hist.get("val_losses") else None,
            "final_ind_loss": hist.get("ind_losses", [])[-1] if hist.get("ind_losses") else None,
            "epochs": len(hist.get("train_losses", [])),
            "train_losses": hist.get("train_losses"),
            "val_losses": hist.get("val_losses"),
            "val_metrics": hist.get("val_metrics"),
            "ind_losses": hist.get("ind_losses"),
        }
        diagnosis["runs"].append(run_info)

    out_dir = _ROOT / "results" / "depth_diagnose"
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnosis_path = out_dir / "diagnosis.json"
    with open(diagnosis_path, "w") as f:
        json.dump(diagnosis, f, indent=2)
    print(f"Diagnosis written to {diagnosis_path}")

    # Print short summary
    print("\n--- Depth-16 ION diagnosis summary ---")
    print(f"Test accuracy mean ± std: {summary.get('test_accuracy_mean', 0):.4f} ± {summary.get('test_accuracy_std', 0):.4f}")
    for r in diagnosis["runs"]:
        bv = r.get("best_val_metric")
        bv_str = f"{bv:.4f}" if bv is not None else "N/A"
        li = r.get("final_ind_loss")
        li_str = f"{li:.6f}" if li is not None else "N/A"
        print(f"  Seed {r['seed']}: best_val={bv_str}, test_acc={r.get('test_accuracy')}, final L_ind={li_str}")
    if diagnosis["runs"] and diagnosis["runs"][0].get("val_metrics"):
        print("  Val accuracy by epoch (seed 42):", [f"{v:.3f}" for v in diagnosis["runs"][0]["val_metrics"][:5]], "...")
    print("--------------------------------------")


if __name__ == "__main__":
    main()
