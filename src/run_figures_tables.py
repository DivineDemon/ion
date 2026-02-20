#!/usr/bin/env python3
"""
Phase A10: Regenerate every figure and table from results/ in one command.

Calls src.analysis.plots.generate_all_plots and src.analysis.tables.generate_all_tables.
Figures are saved to paper/figures/; tables to results/tables/ (CSV) and paper/tables/ (LaTeX).

Usage:
  python -m src.run_figures_tables
  python -m src.run_figures_tables --results-dir /path/to/results --format png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analysis.plots import generate_all_plots
from src.analysis.tables import generate_all_tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate all paper figures and tables from results/"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_ROOT / "results",
        help="Results root directory (default: results/)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Figures output directory (default: paper/figures)",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png"],
        help="Figure format (default: pdf)",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Only generate tables, skip figures",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Only generate figures, skip tables",
    )
    args = parser.parse_args()

    results_root = args.results_dir.resolve()
    if not results_root.exists():
        print(f"Results directory does not exist: {results_root}")
        sys.exit(1)

    figures_dir = args.figures_dir or (_ROOT / "paper" / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not args.tables_only:
        paths = generate_all_plots(
            results_root=results_root,
            output_dir=figures_dir,
            fig_format=args.format,
        )
        print(f"Generated {len(paths)} figure(s) in {figures_dir}")
        for p in paths:
            print("  ", p)

    if not args.figures_only:
        table_paths = generate_all_tables(results_root=results_root)
        print(f"Generated {len(table_paths)} table(s)")
        for p in table_paths:
            print("  ", p)

    print("Done.")


if __name__ == "__main__":
    main()
