#!/usr/bin/env python3
"""
Create placeholder table stubs and figure PDFs so the paper compiles on Overleaf
before any experiments have been run.

Run from repo root or from paper/figures/:
  python paper/figures/create_placeholders.py
  python create_placeholders.py  # when run from paper/figures/

Writes:
  - paper/tables/*.tex  (minimal stub content so \\input{...} does not fail)
  - paper/figures/fig_*.pdf (blank placeholder PDFs so \\includegraphics does not fail)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve paper/ directory: script may be run from repo root or from paper/figures/
SCRIPT = Path(__file__).resolve()
FIGURES_DIR = SCRIPT.parent
if FIGURES_DIR.name == "figures":
    PAPER_DIR = FIGURES_DIR.parent
else:
    PAPER_DIR = FIGURES_DIR
TABLES_DIR = PAPER_DIR / "tables"


# Tables referenced in paper/sections/experiments.tex (without .tex in \\input{...})
TABLE_NAMES = [
    "length_gen_cumsum",
    "length_gen_parity",
    "length_gen_dyck2",
    "length_gen_last_token",
    "depth",
    "mnist",
    "ablation_lambda_cumsum",
    "ablation_lambda_mnist",
    "ablation_p_dim_cumsum",
    "ablation_p_dim_mnist",
    "mechanistic_ablations",
]

# Figures referenced in experiments.tex (\\includegraphics{fig_...} uses .pdf by default)
FIGURE_NAMES = [
    "fig_length_gen_cumsum",
    "fig_length_gen_parity",
    "fig_length_gen_dyck2",
    "fig_depth",
    "fig_mnist_loss",
    "fig_mnist_accuracy",
    "fig_ablation_lambda_cumsum",
    "fig_ablation_lambda_mnist",
    "fig_ablation_p_dim_cumsum",
    "fig_ablation_p_dim_mnist",
    "fig_drift",
]


def _stub_table_tex(name: str) -> str:
    """Minimal LaTeX so the table compiles (caption + placeholder row)."""
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Placeholder: run experiments and \\texttt{python -m src.run\\_figures\\_tables} to generate.}\n"
        "\\begin{tabular}{ll}\n"
        "\\hline\n"
        "Model & Metric \\\\\n"
        "\\hline\n"
        "--- & --- \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\label{tab:" + name + "}\n"
        "\\end{table}\n"
    )


def _write_placeholder_pdf(path: Path) -> None:
    """Write a minimal valid PDF (blank page) so \\includegraphics does not fail."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, _ = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor("white")
        plt.axis("off")
        plt.text(0.5, 0.5, "Figure to be generated\n(run experiments and run_figures_tables)", ha="center", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(path, format="pdf", bbox_inches="tight")
        plt.close()
    except Exception:
        # Fallback: write minimal PDF bytes (single blank page)
        minimal_pdf = (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF\n"
        )
        path.write_bytes(minimal_pdf)


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for name in TABLE_NAMES:
        tex_path = TABLES_DIR / f"{name}.tex"
        if not tex_path.exists():
            tex_path.write_text(_stub_table_tex(name), encoding="utf-8")
            print("  table:", tex_path)

    for name in FIGURE_NAMES:
        pdf_path = FIGURES_DIR / f"{name}.pdf"
        if not pdf_path.exists():
            _write_placeholder_pdf(pdf_path)
            print("  figure:", pdf_path)

    print("Done. Placeholder tables and figures written. Compile the paper with pdflatex.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
