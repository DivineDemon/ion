"""
Paper figures: length generalization, depth, MNIST, ablations, invariant drift.

All figures are saved to paper/figures/ (PDF and optionally PNG).
Reads from results/ directory structure produced by run_* scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------


def _results_root() -> Path:
    return Path(__file__).resolve().parents[2] / "results"


def _figures_dir() -> Path:
    """Default output directory for figures: paper/figures/ (single place for all figures)."""
    d = Path(__file__).resolve().parents[2] / "paper" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 1) Length generalization: accuracy (or task metric) vs sequence length
# ---------------------------------------------------------------------------


def load_length_gen_table(results_root: Path, task: str) -> Optional[pd.DataFrame]:
    """
    Load combined accuracy-vs-length table for a task.
    CSV format: model,50,100,200 with cells like "0.49±0.00" or "319.79±0.00" (MSE).
    Returns DataFrame with index=model, columns=lengths (int), values=mean (float).
    Also need std for error bars: we parse mean±std from each cell.
    """
    table_path = results_root / "length_gen" / task / "accuracy_vs_length_table.csv"
    if not table_path.exists():
        return None
    try:
        df = pd.read_csv(table_path, index_col=0, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(table_path, index_col=0, encoding="latin-1")
    # Columns may be '50', '100', '200' (str) or model names
    return df


def _parse_mean_std(cell: str) -> tuple[float, float]:
    """Parse '0.49±0.00' or '319.79±0.00' -> (mean, std)."""
    if pd.isna(cell) or not isinstance(cell, str):
        return float("nan"), 0.0
    if "±" in cell:
        parts = cell.split("±")
        try:
            mean = float(parts[0].strip())
            std = float(parts[1].strip()) if len(parts) > 1 else 0.0
            return mean, std
        except ValueError:
            return float("nan"), 0.0
    try:
        return float(cell), 0.0
    except (ValueError, TypeError):
        return float("nan"), 0.0


def plot_length_gen(
    results_root: Optional[Path] = None,
    tasks: Optional[list[str]] = None,
    output_dir: Optional[Path] = None,
    fig_format: str = "pdf",
) -> list[Path]:
    """
    Plot accuracy (or task metric) vs sequence length, one curve per model, error bars or shaded std.
    One figure per task. Saves to paper/figures/fig_length_gen_<task>.<format>.
    """
    root = results_root or _results_root()
    out = output_dir or _figures_dir()
    saved: list[Path] = []

    lg_dir = root / "length_gen"
    if not lg_dir.exists():
        return saved

    if tasks is None:
        tasks = [d.name for d in lg_dir.iterdir() if d.is_dir()]

    for task in tasks:
        df = load_length_gen_table(root, task)
        if df is None or df.empty:
            continue
        lengths = [int(c) for c in df.columns if str(c).isdigit()]
        if not lengths:
            continue
        lengths = sorted(lengths)
        models = df.index.tolist()

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(models), 1)))
        for i, model in enumerate(models):
            means = []
            stds = []
            for L in lengths:
                cell = df.loc[model, str(L)] if str(L) in df.columns else None
                if cell is None:
                    cell = df.loc[model].iloc[0] if len(df.columns) > 0 else "0±0"
                m, s = _parse_mean_std(str(cell))
                means.append(m)
                stds.append(s)
            means = np.array(means)
            stds = np.array(stds)
            ax.plot(lengths, means, "o-", label=model, color=colors[i % len(colors)])
            if np.any(stds > 0):
                ax.fill_between(
                    lengths,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.2,
                    color=colors[i % len(colors)],
                )
            else:
                ax.errorbar(lengths, means, yerr=stds, fmt="none", color=colors[i % len(colors)])

        ax.set_xlabel("Sequence length")
        # Cumsum uses MSE; parity uses accuracy
        ylabel = "Test MSE" if task == "cumsum" else "Test accuracy"
        ax.set_ylabel(ylabel)
        ax.set_title(f"Length generalization: {task}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out / f"fig_length_gen_{task}.{fig_format}"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved


# ---------------------------------------------------------------------------
# 2) Depth: accuracy vs depth (baseline vs ION)
# ---------------------------------------------------------------------------


def plot_depth(
    results_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    fig_format: str = "pdf",
) -> Optional[Path]:
    """
    Plot test accuracy vs depth for MLP vs ION. Reads from results/depth/.
    """
    root = results_root or _results_root()
    out = output_dir or _figures_dir()
    depth_dir = root / "depth"
    if not depth_dir.exists():
        return None

    # Prefer accuracy_vs_depth.csv: depth,model,mean_accuracy,std_accuracy
    csv_path = depth_dir / "accuracy_vs_depth.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        depths = sorted(df["depth"].unique())
        models = df["model"].unique().tolist()
    else:
        summary_path = depth_dir / "summary.json"
        if not summary_path.exists():
            return None
        with open(summary_path) as f:
            data = json.load(f)
        summaries = data.get("summaries", [])
        if not summaries:
            return None
        depths = sorted({s["depth"] for s in summaries})
        models = sorted({s["model"] for s in summaries})
        df = pd.DataFrame(summaries)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"mlp": "C0", "ion": "C1"}
    for model in models:
        if "depth" in df.columns and "model" in df.columns:
            sub = df[df["model"] == model].sort_values("depth")
            x = sub["depth"].values
            mean_col = "mean_accuracy" if "mean_accuracy" in sub.columns else "test_accuracy_mean"
            std_col = "std_accuracy" if "std_accuracy" in sub.columns else "test_accuracy_std"
            if mean_col not in sub.columns and "test_accuracy_mean" in df.columns:
                mean_col = "test_accuracy_mean"
                std_col = "test_accuracy_std"
            y = sub[mean_col].values
            yerr = sub[std_col].values if std_col in sub.columns else np.zeros_like(y)
        else:
            sub = [s for s in data.get("summaries", []) if s["model"] == model]
            sub = sorted(sub, key=lambda s: s["depth"])
            x = [s["depth"] for s in sub]
            y = [s.get("test_accuracy_mean", s.get("test_accuracy", 0)) for s in sub]
            yerr = [s.get("test_accuracy_std", 0) for s in sub]
        ax.errorbar(
            x, y, yerr=yerr, marker="o", label=model, color=colors.get(model, "C2")
        )
    ax.set_xlabel("Depth")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Depth stability (MNIST)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out / f"fig_depth.{fig_format}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3) MNIST: train/val loss or accuracy over epochs (baseline vs ION)
# ---------------------------------------------------------------------------


def plot_mnist_curves(
    results_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    fig_format: str = "pdf",
) -> list[Path]:
    """
    Plot train/val loss and val accuracy over epochs for MLP and ION.
    Reads run_<seed>.json from results/mnist/<model>/; uses first available seed.
    """
    root = results_root or _results_root()
    out = output_dir or _figures_dir()
    mnist_dir = root / "mnist"
    saved: list[Path] = []
    if not mnist_dir.exists():
        return saved

    model_dirs = [d for d in mnist_dir.iterdir() if d.is_dir() and d.name in ("mlp", "ion")]
    if not model_dirs:
        return saved

    # Collect history from first run per model
    histories = {}
    for md in model_dirs:
        run_files = list(md.glob("run_*.json"))
        if not run_files:
            continue
        with open(run_files[0]) as f:
            data = json.load(f)
        hist = data.get("history", {})
        if hist:
            histories[md.name] = hist

    if not histories:
        return saved

    # Loss curves
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, hist in histories.items():
        train_losses = hist.get("train_losses", [])
        val_losses = hist.get("val_losses", [])
        epochs = range(1, len(train_losses) + 1)
        if val_losses:
            ax.plot(epochs, val_losses, "o-", label=f"{model} (val)", markersize=3)
        if train_losses:
            ax.plot(epochs, train_losses, "--", alpha=0.7, label=f"{model} (train)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("MNIST: train/val loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out / f"fig_mnist_loss.{fig_format}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # Accuracy curves (val_metrics if classification)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for model, hist in histories.items():
        val_metrics = hist.get("val_metrics", [])
        if val_metrics:
            epochs = range(1, len(val_metrics) + 1)
            ax2.plot(epochs, val_metrics, "o-", label=model, markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val accuracy")
    ax2.set_title("MNIST: validation accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    path2 = out / f"fig_mnist_accuracy.{fig_format}"
    fig2.savefig(path2, bbox_inches="tight")
    plt.close(fig2)
    saved.append(path2)
    return saved


# ---------------------------------------------------------------------------
# 4) Ablations: metric vs λ and metric vs p_dim
# ---------------------------------------------------------------------------


def load_ablation_metrics(
    results_dir: Path,
    sweep: str,
    task: str,
) -> tuple[list[float], list[float], list[float]]:
    """
    Load sweep values and metric values from results/ablations.
    sweep in ('lambda', 'p_dim'); task in ('cumsum', 'mnist').
    Returns (sweep_values, means, stds) per sweep value (aggregated over seeds).
    """
    # Files: lambda_0.5_cumsum_seed_42.json, lambda_0.5_mnist_seed_42.json
    prefix = f"{sweep}_"
    suffix = f"_{task}_seed_"
    pattern = f"{prefix}*_{task}_seed_*.json"
    files = list(results_dir.glob(pattern))
    by_value: dict[float, list[float]] = {}
    for f in files:
        # lambda_0.5_cumsum_seed_42.json -> value 0.5
        name = f.stem
        if suffix not in name or not name.startswith(prefix):
            continue
        mid = name[len(prefix) : name.index(suffix)]
        try:
            value = float(mid)
        except ValueError:
            continue
        with open(f) as fp:
            data = json.load(fp)
        metric_value = data.get("metric_value")
        if metric_value is None:
            continue
        by_value.setdefault(value, []).append(metric_value)
    if not by_value:
        return [], [], []
    values = sorted(by_value.keys())
    means = [np.mean(by_value[v]) for v in values]
    stds = [np.std(by_value[v], ddof=1) if len(by_value[v]) > 1 else 0.0 for v in values]
    return values, means, stds


def plot_ablations(
    results_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    fig_format: str = "pdf",
) -> list[Path]:
    """
    Plot task metric vs λ and vs p_dim for cumsum and MNIST. Saves to paper/figures/.
    """
    root = results_root or _results_root()
    out = output_dir or _figures_dir()
    ablat_dir = root / "ablations"
    saved: list[Path] = []
    if not ablat_dir.exists():
        return saved

    for sweep in ("lambda", "p_dim"):
        for task in ("cumsum", "mnist"):
            values, means, stds = load_ablation_metrics(ablat_dir, sweep, task)
            if not values:
                continue
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.errorbar(values, means, yerr=stds, marker="o", capsize=4)
            ax.set_xlabel(sweep.replace("_", " ") + (" (λ)" if sweep == "lambda" else ""))
            if task == "cumsum":
                ax.set_ylabel("Test MSE (lower is better)")
            else:
                ax.set_ylabel("Test accuracy")
            ax.set_title(f"Ablation: {sweep} on {task}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = out / f"fig_ablation_{sweep}_{task}.{fig_format}"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
    return saved


# ---------------------------------------------------------------------------
# 5) Invariant drift: drift vs step (recurrent) or vs layer (universal)
# ---------------------------------------------------------------------------


def plot_drift(
    drift_data: Optional[dict[str, Any]] = None,
    drift_path: Optional[Path] = None,
    results_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    fig_format: str = "pdf",
) -> list[Path]:
    """
    Plot invariant drift vs step (recurrent) or vs layer (universal).

    drift_data: optional dict with keys:
      - "type": "recurrent" | "universal"
      - "per_index": list of drift values (or dict mapping length/depth -> list)
      - For multiple lengths: "by_length" -> {50: [drift per step], 100: [...]}
    drift_path: path to JSON file with the above structure (e.g. results/drift/drift.json).
    If neither provided, looks for results/drift/drift.json under results_root.
    """
    root = results_root or _results_root()
    out = output_dir or _figures_dir()
    saved: list[Path] = []

    if drift_data is None:
        if drift_path is None:
            drift_path = root / "drift" / "drift.json"
        if drift_path is not None and drift_path.exists():
            with open(drift_path) as f:
                drift_data = json.load(f)
    if drift_data is None:
        return saved

    drift_type = drift_data.get("type", "recurrent")
    per_index = drift_data.get("per_index")
    by_length = drift_data.get("by_length", {})

    if by_length:
        fig, ax = plt.subplots(figsize=(6, 4))
        for length, steps in sorted(by_length.items()):
            if isinstance(steps, list):
                indices = range(1, len(steps) + 1)
                ax.plot(indices, steps, "o-", label=f"length={length}", markersize=2)
        ax.set_xlabel("Step" if drift_type == "recurrent" else "Layer")
        ax.set_ylabel("Invariant drift")
        ax.set_title("ION: invariant drift vs " + ("step" if drift_type == "recurrent" else "layer"))
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out / f"fig_drift.{fig_format}"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    elif per_index is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        steps = per_index if isinstance(per_index, list) else list(per_index)
        indices = range(1, len(steps) + 1)
        ax.plot(indices, steps, "o-", markersize=2)
        ax.set_xlabel("Step" if drift_type == "recurrent" else "Layer")
        ax.set_ylabel("Invariant drift")
        ax.set_title("ION: invariant drift vs " + ("step" if drift_type == "recurrent" else "layer"))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out / f"fig_drift.{fig_format}"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved


# ---------------------------------------------------------------------------
# Run all plot generators
# ---------------------------------------------------------------------------


def generate_all_plots(
    results_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    fig_format: str = "pdf",
) -> list[Path]:
    """
    Generate all paper figures from results/ directory.
    Returns list of saved figure paths.
    """
    root = results_root or _results_root()
    out = output_dir or _figures_dir()
    all_saved: list[Path] = []

    # 1) Length generalization (one fig per task)
    all_saved.extend(plot_length_gen(root, output_dir=out, fig_format=fig_format))

    # 2) Depth
    p = plot_depth(root, output_dir=out, fig_format=fig_format)
    if p is not None:
        all_saved.append(p)

    # 3) MNIST curves
    all_saved.extend(plot_mnist_curves(root, output_dir=out, fig_format=fig_format))

    # 4) Ablations
    all_saved.extend(plot_ablations(root, output_dir=out, fig_format=fig_format))

    # 5) Drift (if data available)
    all_saved.extend(plot_drift(results_root=root, output_dir=out, fig_format=fig_format))

    return all_saved


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate paper figures from results/")
    parser.add_argument("--results-dir", type=Path, default=None, help="Results root")
    parser.add_argument("--output-dir", type=Path, default=None, help="Figures output directory")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"], help="Figure format")
    args = parser.parse_args()
    paths = generate_all_plots(args.results_dir, args.output_dir, args.format)
    for p in paths:
        print("Saved:", p)
