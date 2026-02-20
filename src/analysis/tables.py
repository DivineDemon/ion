"""
Generate LaTeX and CSV tables from experiment results.

Reads from results/ directory; writes to results/tables/ and paper/tables/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.analysis.stats import compare_two_methods


def _results_root() -> Path:
    return Path(__file__).resolve().parents[2] / "results"


def _paper_root() -> Path:
    return Path(__file__).resolve().parents[2] / "paper"


def _ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def _length_gen_significance_note(task_dir: Path, task: str, lengths: list) -> str:
    """
    Compute ION vs best baseline at longest length; return a caption note with p-value if we have
    per-seed data and >=2 seeds per group. Task 'cumsum' -> test_mse (lower better); else test_accuracy (higher better).
    """
    if not lengths:
        return ""
    longest = max(int(l) for l in lengths) if lengths else None
    if longest is None:
        return ""
    metric_key = "test_mse" if task == "cumsum" else "test_accuracy"
    higher_better = task != "cumsum"

    per_model_metrics: dict[str, list[float]] = {}
    for model_dir in task_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        values = []
        for p in model_dir.glob("run_*_length_gen.json"):
            try:
                with open(p) as f:
                    data = json.load(f)
                pl = data.get("per_length") or {}
                # keys may be int or str
                rec = pl.get(longest) or pl.get(str(longest))
                if rec is not None and metric_key in rec:
                    values.append(float(rec[metric_key]))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        if values:
            per_model_metrics[model_name] = values

    ion_vals = per_model_metrics.get("ion")
    baselines = {m: v for m, v in per_model_metrics.items() if m != "ion" and len(v) >= 2}
    if not ion_vals or len(ion_vals) < 2 or not baselines:
        return ""

    # Best baseline by mean (higher better for accuracy, lower for MSE)
    def mean_fn(vals):
        return sum(vals) / len(vals)

    best_name = max(
        baselines.keys(),
        key=lambda m: mean_fn(baselines[m]) if higher_better else -mean_fn(baselines[m]),
    )
    best_vals = baselines[best_name]
    cmp = compare_two_methods(ion_vals, best_vals, paired=False, alpha=0.05)
    p_val = cmp.get("p_value")
    sig = cmp.get("significant", False)
    d = cmp.get("cohens_d")
    if p_val is None or (isinstance(p_val, float) and (p_val != p_val)):  # nan
        return ""
    p_str = f"{p_val:.3f}" if p_val >= 0.001 else f"{p_val:.2e}"
    sig_str = "significant" if sig else "not significant"
    effect = f", $d={d:.2f}$" if d is not None and isinstance(d, (int, float)) else ""
    return f" ION vs {best_name} at length {longest}: $p={p_str}$ ({sig_str} at $\\alpha=0.05${effect})."


# ---------------------------------------------------------------------------
# 1) Length-gen: rows = models, columns = test lengths, cells = accuracy mean ± std
# ---------------------------------------------------------------------------


def table_length_gen(
    results_root: Optional[Path] = None,
    out_csv_dir: Optional[Path] = None,
    out_tex_dir: Optional[Path] = None,
) -> list[Path]:
    """
    Generate LaTeX and CSV tables for length generalization (all tasks).
    Reads from results/length_gen/<task>/accuracy_vs_length_table.csv.
    """
    root = results_root or _results_root()
    csv_dir = out_csv_dir or root / "tables"
    tex_dir = out_tex_dir or _paper_root() / "tables"
    _ensure_dirs(csv_dir, tex_dir)
    saved: list[Path] = []

    lg_dir = root / "length_gen"
    if not lg_dir.exists():
        return saved

    for task_dir in lg_dir.iterdir():
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        table_path = task_dir / "accuracy_vs_length_table.csv"
        if not table_path.exists():
            continue
        lines = table_path.read_text().strip().split("\n")
        if len(lines) < 2:
            continue
        header = lines[0]
        rows = lines[1:]

        # CSV: already in good shape; copy or rewrite with consistent formatting
        csv_path = csv_dir / f"length_gen_{task}.csv"
        csv_path.write_text(table_path.read_text())
        saved.append(csv_path)

        # Significance note (ION vs best baseline at longest length)
        lengths_cols = [c.strip() for c in header.split(",")[1:]]
        try:
            length_ints = [int(x) for x in lengths_cols]
        except ValueError:
            length_ints = []
        sig_note = _length_gen_significance_note(task_dir, task, length_ints)

        # LaTeX
        parts = header.split(",")
        col_names = [c.strip() for c in parts]
        num_cols = len(col_names)
        caption = "Length generalization: " + task + " (test metric by length)." + sig_note
        tex_lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{" + caption + "}",
            "\\label{tab:length_gen_" + task + "}",
            "\\begin{tabular}{l" + "c" * (num_cols - 1) + "}",
            "\\toprule",
            "Model & " + " & ".join(col_names[1:]) + " \\\\",
            "\\midrule",
        ]
        for row in rows:
            cells = [c.strip() for c in row.split(",")]
            if len(cells) < num_cols:
                continue
            model = cells[0].replace("_", "\\_")
            # Cells may be "0.49±0.00" -> 0.49 $\\pm$ 0.00 for LaTeX; or "0.49 (n=1)"
            tex_cells = []
            for c in cells[1:]:
                c = c.strip()
                if "±" in c:
                    a, b = c.split("±", 1)
                    tex_cells.append(f"{a.strip()} $\\pm$ {b.strip()}")
                elif "(n=1)" in c:
                    tex_cells.append(c)
                else:
                    tex_cells.append(c)
            tex_lines.append(model + " & " + " & ".join(tex_cells) + " \\\\")
        tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        tex_path = tex_dir / f"length_gen_{task}.tex"
        tex_path.write_text("\n".join(tex_lines) + "\n")
        saved.append(tex_path)
    return saved


# ---------------------------------------------------------------------------
# 2) Depth: rows = depth, columns = baseline vs ION
# ---------------------------------------------------------------------------


def table_depth(
    results_root: Optional[Path] = None,
    out_csv_dir: Optional[Path] = None,
    out_tex_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate LaTeX and CSV tables for depth experiment."""
    root = results_root or _results_root()
    csv_dir = out_csv_dir or root / "tables"
    tex_dir = out_tex_dir or _paper_root() / "tables"
    _ensure_dirs(csv_dir, tex_dir)
    saved: list[Path] = []

    depth_dir = root / "depth"
    if not depth_dir.exists():
        return saved

    # Prefer accuracy_vs_depth.csv (long format: depth, model, mean_accuracy, std_accuracy)
    csv_src = depth_dir / "accuracy_vs_depth.csv"
    summary_path = depth_dir / "summary.json"
    if csv_src.exists():
        content = csv_src.read_text()
        csv_out = csv_dir / "depth_accuracy.csv"
        csv_out.write_text(content)
        saved.append(csv_out)
        lines = content.strip().split("\n")
        if len(lines) >= 2:
            header = lines[0]
            rows = lines[1:]
            # Pivot for LaTeX: depth | mlp (mean ± std) | ion (mean ± std)
            depth_vals = set(r.split(",")[0] for r in rows)
            try:
                depths = sorted(depth_vals, key=int)
            except ValueError:
                depths = sorted(depth_vals)
            models = sorted(set(r.split(",")[1] for r in rows))
            data = {}
            for r in rows:
                p = r.split(",")
                if len(p) >= 4:
                    d, m, mean, std = p[0], p[1], p[2].strip(), p[3].strip()
                    try:
                        n = int(p[4].strip()) if len(p) >= 5 else 2
                    except (ValueError, IndexError):
                        n = 2
                    data[(d, m)] = (mean, std, n)
            tex_lines = [
                "\\begin{table}[t]",
                "\\centering",
                "\\caption{Depth stability: test accuracy (MNIST).}",
                "\\label{tab:depth}",
                "\\begin{tabular}{l" + "c" * len(models) + "}",
                "\\toprule",
                "Depth & " + " & ".join(m.upper() for m in models) + " \\\\",
                "\\midrule",
            ]
            for d in depths:
                cells = [d]
                for m in models:
                    entry = data.get((d, m), ("--", "--", 2))
                    if len(entry) == 3:
                        mean, std, n = entry
                        if n <= 1:
                            cells.append(f"{mean} (n=1)")
                        else:
                            cells.append(f"{mean} $\\pm$ {std}")
                    else:
                        mean, std = entry[0], entry[1]
                        cells.append(f"{mean} $\\pm$ {std}")
                tex_lines.append(" & ".join(cells) + " \\\\")
            tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
            tex_path = tex_dir / "depth.tex"
            tex_path.write_text("\n".join(tex_lines) + "\n")
            saved.append(tex_path)
    elif summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
        summaries = data.get("summaries", [])
        if not summaries:
            return saved
        # CSV
        csv_out = csv_dir / "depth_accuracy.csv"
        csv_lines = ["depth,model,mean_accuracy,std_accuracy"]
        for s in summaries:
            d = s.get("depth", "")
            m = s.get("model", "")
            mean = s.get("test_accuracy_mean", s.get("test_accuracy", 0))
            std = s.get("test_accuracy_std", 0)
            csv_lines.append(f"{d},{m},{mean:.4f},{std:.4f}")
        csv_out.write_text("\n".join(csv_lines) + "\n")
        saved.append(csv_out)
        # LaTeX: sort depths numerically
        depth_set = {s["depth"] for s in summaries}
        try:
            depths = sorted(depth_set, key=int)
        except (ValueError, TypeError):
            depths = sorted(depth_set)
        models = sorted({s["model"] for s in summaries})
        by_key = {}
        for s in summaries:
            n = len(s.get("test_accuracies", s.get("seeds", [])))
            by_key[(s["depth"], s["model"])] = (
                s.get("test_accuracy_mean", s.get("test_accuracy", 0)),
                s.get("test_accuracy_std", 0),
                n,
            )
        tex_lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Depth stability: test accuracy (MNIST).}",
            "\\label{tab:depth}",
            "\\begin{tabular}{l" + "c" * len(models) + "}",
            "\\toprule",
            "Depth & " + " & ".join(m.upper() for m in models) + " \\\\",
            "\\midrule",
        ]
        for d in depths:
            cells = [str(d)]
            for m in models:
                entry = by_key.get((d, m), (0, 0, 2))
                mean = entry[0]
                std = entry[1] if len(entry) > 1 else 0
                n = entry[2] if len(entry) > 2 else 2
                if isinstance(mean, (int, float)):
                    if n <= 1:
                        cells.append(f"{mean:.4f} (n=1)")
                    else:
                        cells.append(f"{mean:.4f} $\\pm$ {std:.4f}")
                else:
                    cells.append("--")
            tex_lines.append(" & ".join(cells) + " \\\\")
        tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        tex_path = tex_dir / "depth.tex"
        tex_path.write_text("\n".join(tex_lines) + "\n")
        saved.append(tex_path)
    return saved


def table_depth_cifar(
    results_root: Optional[Path] = None,
    out_csv_dir: Optional[Path] = None,
    out_tex_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate LaTeX and CSV tables for depth experiment on CIFAR-10 (reads results/depth/cifar/)."""
    root = results_root or _results_root()
    csv_dir = out_csv_dir or root / "tables"
    tex_dir = out_tex_dir or _paper_root() / "tables"
    _ensure_dirs(csv_dir, tex_dir)
    saved: list[Path] = []

    depth_dir = root / "depth" / "cifar"
    if not depth_dir.exists():
        return saved

    csv_src = depth_dir / "accuracy_vs_depth.csv"
    summary_path = depth_dir / "summary.json"
    if csv_src.exists():
        content = csv_src.read_text()
        csv_out = csv_dir / "depth_cifar_accuracy.csv"
        csv_out.write_text(content)
        saved.append(csv_out)
        lines = content.strip().split("\n")
        if len(lines) >= 2:
            header = lines[0]
            rows = lines[1:]
            depth_vals = set(r.split(",")[0] for r in rows)
            try:
                depths = sorted(depth_vals, key=int)
            except ValueError:
                depths = sorted(depth_vals)
            models = sorted(set(r.split(",")[1] for r in rows))
            data = {}
            for r in rows:
                p = r.split(",")
                if len(p) >= 4:
                    d, m, mean, std = p[0], p[1], p[2].strip(), p[3].strip()
                    try:
                        n = int(p[4].strip()) if len(p) >= 5 else 2
                    except (ValueError, IndexError):
                        n = 2
                    data[(d, m)] = (mean, std, n)
            tex_lines = [
                "\\begin{table}[t]",
                "\\centering",
                "\\caption{Depth stability: test accuracy (CIFAR-10).}",
                "\\label{tab:depth_cifar}",
                "\\begin{tabular}{l" + "c" * len(models) + "}",
                "\\toprule",
                "Depth & " + " & ".join(m.upper() for m in models) + " \\\\",
                "\\midrule",
            ]
            for d in depths:
                cells = [d]
                for m in models:
                    entry = data.get((d, m), ("--", "--", 2))
                    if len(entry) == 3:
                        mean, std, n = entry
                        if n <= 1:
                            cells.append(f"{mean} (n=1)")
                        else:
                            cells.append(f"{mean} $\\pm$ {std}")
                    else:
                        mean, std = entry[0], entry[1]
                        cells.append(f"{mean} $\\pm$ {std}")
                tex_lines.append(" & ".join(cells) + " \\\\")
            tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
            tex_path = tex_dir / "depth_cifar.tex"
            tex_path.write_text("\n".join(tex_lines) + "\n")
            saved.append(tex_path)
    elif summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
        summaries = data.get("summaries", [])
        if not summaries:
            return saved
        csv_out = csv_dir / "depth_cifar_accuracy.csv"
        csv_lines = ["depth,model,mean_accuracy,std_accuracy"]
        for s in summaries:
            d = s.get("depth", "")
            m = s.get("model", "")
            mean = s.get("test_accuracy_mean", s.get("test_accuracy", 0))
            std = s.get("test_accuracy_std", 0)
            csv_lines.append(f"{d},{m},{mean:.4f},{std:.4f}")
        csv_out.write_text("\n".join(csv_lines) + "\n")
        saved.append(csv_out)
        depth_set = {s["depth"] for s in summaries}
        try:
            depths = sorted(depth_set, key=int)
        except (ValueError, TypeError):
            depths = sorted(depth_set)
        models = sorted({s["model"] for s in summaries})
        by_key = {}
        for s in summaries:
            n = len(s.get("test_accuracies", s.get("seeds", [])))
            by_key[(s["depth"], s["model"])] = (
                s.get("test_accuracy_mean", s.get("test_accuracy", 0)),
                s.get("test_accuracy_std", 0),
                n,
            )
        tex_lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Depth stability: test accuracy (CIFAR-10).}",
            "\\label{tab:depth_cifar}",
            "\\begin{tabular}{l" + "c" * len(models) + "}",
            "\\toprule",
            "Depth & " + " & ".join(m.upper() for m in models) + " \\\\",
            "\\midrule",
        ]
        for d in depths:
            cells = [str(d)]
            for m in models:
                entry = by_key.get((d, m), (0, 0, 2))
                mean = entry[0]
                std = entry[1] if len(entry) > 1 else 0
                n = entry[2] if len(entry) > 2 else 2
                if isinstance(mean, (int, float)):
                    if n <= 1:
                        cells.append(f"{mean:.4f} (n=1)")
                    else:
                        cells.append(f"{mean:.4f} $\\pm$ {std:.4f}")
                else:
                    cells.append("--")
            tex_lines.append(" & ".join(cells) + " \\\\")
        tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        tex_path = tex_dir / "depth_cifar.tex"
        tex_path.write_text("\n".join(tex_lines) + "\n")
        saved.append(tex_path)
    return saved


# ---------------------------------------------------------------------------
# 3) MNIST: baseline vs ION test accuracy
# ---------------------------------------------------------------------------


def table_mnist(
    results_root: Optional[Path] = None,
    out_csv_dir: Optional[Path] = None,
    out_tex_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate LaTeX and CSV tables for MNIST experiment."""
    root = results_root or _results_root()
    csv_dir = out_csv_dir or root / "tables"
    tex_dir = out_tex_dir or _paper_root() / "tables"
    _ensure_dirs(csv_dir, tex_dir)
    saved: list[Path] = []

    mnist_dir = root / "mnist"
    if not mnist_dir.exists():
        return saved

    csv_src = mnist_dir / "mnist_results.csv"
    if not csv_src.exists():
        summary_path = mnist_dir / "test_accuracy_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
            summaries = data.get("summaries", [])
            csv_out = csv_dir / "mnist_results.csv"
            csv_lines = ["model,test_accuracy_mean,test_accuracy_std"]
            for s in summaries:
                m = s.get("model", "")
                mean = s.get("test_accuracy_mean", 0)
                std = s.get("test_accuracy_std", 0)
                csv_lines.append(f"{m},{mean:.6f},{std:.6f}")
            csv_out.write_text("\n".join(csv_lines) + "\n")
            saved.append(csv_out)
            # LaTeX
            tex_lines = [
                "\\begin{table}[t]",
                "\\centering",
                "\\caption{MNIST: test accuracy.}",
                "\\label{tab:mnist}",
                "\\begin{tabular}{lc}",
                "\\toprule",
                "Model & Test accuracy \\\\",
                "\\midrule",
            ]
            for s in summaries:
                model = s.get("model", "").replace("_", "\\_")
                mean = s.get("test_accuracy_mean", 0)
                std = s.get("test_accuracy_std", 0)
                n = len(s.get("seeds", []))
                if n <= 1:
                    tex_lines.append(f"{model} & ${mean:.4f}$ (n=1) \\\\")
                else:
                    tex_lines.append(f"{model} & ${mean:.4f} \\pm {std:.4f}$ \\\\")
            tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
            tex_path = tex_dir / "mnist.tex"
            tex_path.write_text("\n".join(tex_lines) + "\n")
            saved.append(tex_path)
        return saved

    content = csv_src.read_text()
    csv_out = csv_dir / "mnist_results.csv"
    csv_out.write_text(content)
    saved.append(csv_out)
    lines = content.strip().split("\n")
    if len(lines) >= 2:
        header = lines[0].split(",")
        rows = lines[1:]
        idx_model, idx_mean, idx_std = 0, 1, 2
        idx_n = 3 if len(header) > 3 and header[3].strip() == "n" else None
        tex_lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{MNIST: test accuracy.}",
            "\\label{tab:mnist}",
            "\\begin{tabular}{lc}",
            "\\toprule",
            "Model & Test accuracy \\\\",
            "\\midrule",
        ]
        for r in rows:
            cells = [c.strip() for c in r.split(",")]
            if len(cells) <= idx_std:
                continue
            model = cells[idx_model].replace("_", "\\_")
            mean = cells[idx_mean]
            std = cells[idx_std] if len(cells) > idx_std else "0"
            try:
                n = int(cells[idx_n]) if idx_n is not None and len(cells) > idx_n else 2
            except (ValueError, TypeError, IndexError):
                n = 2
            if n <= 1:
                tex_lines.append(f"{model} & ${mean}$ (n=1) \\\\")
            else:
                tex_lines.append(f"{model} & ${mean} \\pm {std}$ \\\\")
        tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        tex_path = tex_dir / "mnist.tex"
        tex_path.write_text("\n".join(tex_lines) + "\n")
        saved.append(tex_path)
    return saved


# ---------------------------------------------------------------------------
# 4) Ablations: λ and p_dim
# ---------------------------------------------------------------------------


def _aggregate_ablations(results_dir: Path, sweep: str, task: str) -> list[tuple[float, float, float, int]]:
    """Return list of (sweep_value, mean_metric, std_metric, n_runs)."""
    prefix = f"{sweep}_"
    suffix = f"_{task}_seed_"
    by_value: dict[float, list[float]] = {}
    for f in results_dir.glob(f"{prefix}*_{task}_seed_*.json"):
        name = f.stem
        if suffix not in name:
            continue
        mid = name[len(prefix) : name.index(suffix)]
        try:
            value = float(mid)
        except ValueError:
            continue
        with open(f) as fp:
            data = json.load(fp)
        mval = data.get("metric_value")
        if mval is None:
            continue
        by_value.setdefault(value, []).append(mval)
    if not by_value:
        return []
    import numpy as np
    out = []
    for v in sorted(by_value.keys()):
        arr = by_value[v]
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        out.append((v, mean, std, len(arr)))
    return out


def table_ablations(
    results_root: Optional[Path] = None,
    out_csv_dir: Optional[Path] = None,
    out_tex_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate LaTeX and CSV tables for ablation studies (lambda, p_dim)."""
    root = results_root or _results_root()
    csv_dir = out_csv_dir or root / "tables"
    tex_dir = out_tex_dir or _paper_root() / "tables"
    _ensure_dirs(csv_dir, tex_dir)
    saved: list[Path] = []

    ablat_dir = root / "ablations"
    if not ablat_dir.exists():
        return saved

    for sweep in ("lambda", "p_dim"):
        for task in ("cumsum", "mnist"):
            rows = _aggregate_ablations(ablat_dir, sweep, task)
            if not rows:
                continue
            # CSV
            metric_name = "test_mse" if task == "cumsum" else "test_accuracy"
            csv_path = csv_dir / f"ablation_{sweep}_{task}.csv"
            csv_lines = [f"{sweep},{metric_name}_mean,{metric_name}_std,n"]
            for v, mean, std, n in rows:
                csv_lines.append(f"{v},{mean:.6f},{std:.6f},{n}")
            csv_path.write_text("\n".join(csv_lines) + "\n")
            saved.append(csv_path)
            # LaTeX: when n=1 show mean and (n=1) instead of misleading 0 std
            col = sweep.replace("_", " ").title() + (" ($\\lambda$)" if sweep == "lambda" else "")
            tex_lines = [
                "\\begin{table}[t]",
                "\\centering",
                "\\caption{Ablation: " + sweep + " on " + task + ".}",
                "\\label{tab:ablation_" + sweep + "_" + task + "}",
                "\\begin{tabular}{lcc}",
                "\\toprule",
                col + " & Mean & Std \\\\",
                "\\midrule",
            ]
            for v, mean, std, n in rows:
                if n <= 1:
                    tex_lines.append(f"{v} & {mean:.4f} & (n=1) \\\\")
                else:
                    tex_lines.append(f"{v} & {mean:.4f} & {std:.4f} \\\\")
            tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
            tex_path = tex_dir / f"ablation_{sweep}_{task}.tex"
            tex_path.write_text("\n".join(tex_lines) + "\n")
            saved.append(tex_path)
    return saved


# ---------------------------------------------------------------------------
# 5) Mechanistic ablations: full ION, P-only, F-only, random P
# ---------------------------------------------------------------------------


def table_mechanistic_ablations(
    results_root: Optional[Path] = None,
    out_csv_dir: Optional[Path] = None,
    out_tex_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate LaTeX/CSV table for mechanistic ablations (cumsum)."""
    root = results_root or _results_root()
    csv_dir = out_csv_dir or root / "tables"
    tex_dir = out_tex_dir or _paper_root() / "tables"
    _ensure_dirs(csv_dir, tex_dir)
    saved: list[Path] = []

    mech_dir = root / "mechanistic_ablations"
    if not mech_dir.exists():
        return saved

    rows: list[tuple[str, float, float, int]] = []
    for variant_dir in sorted(mech_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        summary_path = variant_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            s = json.load(f)
        mean_mse = s.get("mean_mse", 0)
        std_mse = s.get("std_mse", 0)
        n = s.get("n", len(s.get("seeds", [])))
        label = variant_dir.name.replace("_", "\\_")
        rows.append((label, mean_mse, std_mse, n))

    if not rows:
        return saved

    csv_path = csv_dir / "mechanistic_ablations.csv"
    with open(csv_path, "w") as f:
        f.write("variant,mean_mse,std_mse,n\n")
        for label, mean_mse, std_mse, n in rows:
            f.write(f"{label.replace(chr(92)+'_', '_')},{mean_mse:.4f},{std_mse:.4f},{n}\n")
    saved.append(csv_path)

    tex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Mechanistic ablations on cumulative sum (test MSE). Full ION vs P-only (F frozen), F-only (P frozen random), random P. Full ION and P-only are competitive; F-only and random P are worse. The learned invariant $P$ is the main driver; $F$ is task-dependent (identity suffices for cumsum).}",
        "\\label{tab:mechanistic_ablations}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Variant & Mean MSE & Std \\\\",
        "\\midrule",
    ]
    for label, mean_mse, std_mse, n in rows:
        if n <= 1:
            tex_lines.append(f"{label} & {mean_mse:.2f} & (n=1) \\\\")
        else:
            tex_lines.append(f"{label} & {mean_mse:.2f} & {std_mse:.2f} \\\\")
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    tex_path = tex_dir / "mechanistic_ablations.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")
    saved.append(tex_path)
    return saved


# ---------------------------------------------------------------------------
# 6) LRA ListOps: Transformer vs ION-Transformer test accuracy
# ---------------------------------------------------------------------------


def table_lra(
    results_root: Optional[Path] = None,
    out_csv_dir: Optional[Path] = None,
    out_tex_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate LaTeX and CSV table for LRA ListOps (Transformer vs ION-Transformer)."""
    root = results_root or _results_root()
    csv_dir = out_csv_dir or root / "tables"
    tex_dir = out_tex_dir or _paper_root() / "tables"
    _ensure_dirs(csv_dir, tex_dir)
    saved: list[Path] = []

    lra_dir = root / "lra" / "listops"
    if not lra_dir.exists():
        return saved

    _model_display = {"transformer": "Transformer", "transformer_alibi": "Transformer (ALiBi)", "ion": "ION-Transformer"}
    rows: list[tuple[str, float, float, int]] = []
    for model_name in ("transformer", "transformer_alibi", "ion"):
        summary_path = lra_dir / model_name / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            s = json.load(f)
        mean_acc = s.get("test_accuracy_mean", 0.0)
        std_acc = s.get("test_accuracy_std", 0.0)
        n = len(s.get("seeds", []))
        display_name = _model_display.get(model_name, model_name.replace("_", " "))
        rows.append((display_name, mean_acc, std_acc, n))

    if not rows:
        return saved

    csv_path = csv_dir / "lra_listops.csv"
    with open(csv_path, "w") as f:
        f.write("model,test_accuracy_mean,test_accuracy_std,n\n")
        for display_name, mean_acc, std_acc, n in rows:
            f.write(f"{display_name},{mean_acc:.6f},{std_acc:.6f},{n}\n")
    saved.append(csv_path)

    tex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{LRA ListOps: test accuracy (mean $\\pm$ std over seeds).}",
        "\\label{tab:lra_listops}",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Model & Test accuracy \\\\",
        "\\midrule",
    ]
    for display_name, mean_acc, std_acc, n in rows:
        if n <= 1:
            tex_lines.append(f"{display_name} & ${mean_acc:.4f}$ (n=1) \\\\")
        else:
            tex_lines.append(f"{display_name} & ${mean_acc:.4f} \\pm {std_acc:.4f}$ \\\\")
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    tex_path = tex_dir / "lra_listops.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")
    saved.append(tex_path)
    return saved


def table_lra_image(
    results_root: Optional[Path] = None,
    out_csv_dir: Optional[Path] = None,
    out_tex_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate LaTeX and CSV table for LRA Image (CIFAR-10 as sequence): Transformer vs ION-Transformer."""
    root = results_root or _results_root()
    csv_dir = out_csv_dir or root / "tables"
    tex_dir = out_tex_dir or _paper_root() / "tables"
    _ensure_dirs(csv_dir, tex_dir)
    saved: list[Path] = []

    lra_dir = root / "lra" / "image"
    if not lra_dir.exists():
        return saved

    rows: list[tuple[str, float, float, int]] = []
    for model_name in ("transformer", "ion"):
        summary_path = lra_dir / model_name / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            s = json.load(f)
        mean_acc = s.get("test_accuracy_mean", 0.0)
        std_acc = s.get("test_accuracy_std", 0.0)
        n = len(s.get("seeds", []))
        display_name = "Transformer" if model_name == "transformer" else "ION-Transformer"
        rows.append((display_name, mean_acc, std_acc, n))

    if not rows:
        return saved

    csv_path = csv_dir / "lra_image.csv"
    with open(csv_path, "w") as f:
        f.write("model,test_accuracy_mean,test_accuracy_std,n\n")
        for display_name, mean_acc, std_acc, n in rows:
            f.write(f"{display_name},{mean_acc:.6f},{std_acc:.6f},{n}\n")
    saved.append(csv_path)

    tex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{LRA Image (CIFAR-10 as sequence): test accuracy (mean $\\pm$ std over seeds).}",
        "\\label{tab:lra_image}",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Model & Test accuracy \\\\",
        "\\midrule",
    ]
    for display_name, mean_acc, std_acc, n in rows:
        if n <= 1:
            tex_lines.append(f"{display_name} & ${mean_acc:.4f}$ (n=1) \\\\")
        else:
            tex_lines.append(f"{display_name} & ${mean_acc:.4f} \\pm {std_acc:.4f}$ \\\\")
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    tex_path = tex_dir / "lra_image.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")
    saved.append(tex_path)
    return saved


# ---------------------------------------------------------------------------
# Run all table generators
# ---------------------------------------------------------------------------


def generate_all_tables(
    results_root: Optional[Path] = None,
    out_csv_dir: Optional[Path] = None,
    out_tex_dir: Optional[Path] = None,
) -> list[Path]:
    """
    Generate all LaTeX and CSV tables from results/ directory.
    Writes to results/tables/ (CSV) and paper/tables/ (LaTeX).
    """
    root = results_root or _results_root()
    csv_dir = out_csv_dir or root / "tables"
    tex_dir = out_tex_dir or _paper_root() / "tables"
    _ensure_dirs(csv_dir, tex_dir)
    all_saved: list[Path] = []

    all_saved.extend(table_length_gen(root, csv_dir, tex_dir))
    all_saved.extend(table_depth(root, csv_dir, tex_dir))
    all_saved.extend(table_mnist(root, csv_dir, tex_dir))
    all_saved.extend(table_ablations(root, csv_dir, tex_dir))
    all_saved.extend(table_mechanistic_ablations(root, csv_dir, tex_dir))
    all_saved.extend(table_lra(root, csv_dir, tex_dir))
    all_saved.extend(table_lra_image(root, csv_dir, tex_dir))
    all_saved.extend(table_depth_cifar(root, csv_dir, tex_dir))

    return all_saved


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate paper tables from results/")
    parser.add_argument("--results-dir", type=Path, default=None, help="Results root")
    parser.add_argument("--csv-dir", type=Path, default=None, help="CSV output directory")
    parser.add_argument("--tex-dir", type=Path, default=None, help="LaTeX output directory")
    args = parser.parse_args()
    paths = generate_all_tables(args.results_dir, args.csv_dir, args.tex_dir)
    for p in paths:
        print("Saved:", p)
