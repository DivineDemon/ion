#!/usr/bin/env python3
"""Create minimal placeholder PDFs so the paper compiles when figures are not yet generated."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = Path(__file__).resolve().parent

# All figure basenames referenced in sections/experiments.tex (no extension)
NAMES = [
    "fig_length_gen_cumsum",
    "fig_length_gen_parity",
    "fig_depth",
    "fig_mnist_loss",
    "fig_mnist_accuracy",
    "fig_ablation_lambda_cumsum",
    "fig_ablation_lambda_mnist",
    "fig_drift",
]


def main() -> None:
    for name in NAMES:
        path = FIGURES_DIR / f"{name}.pdf"
        if path.exists():
            continue
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set_facecolor("#f8f8f8")
        ax.text(0.5, 0.5, f"Placeholder: {name}", ha="center", va="center", fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Created {path}")
    print("Done.")


if __name__ == "__main__":
    main()
