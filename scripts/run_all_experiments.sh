#!/usr/bin/env bash
# Full experiment suite: length-gen (all task x model), MNIST, depth, ablations,
# mechanistic ablations, drift; then aggregate and generate figures/tables.
# Uses 5 seeds (42, 123, 456, 789, 1024) unless overridden.
# Run from repo root.

set -e
SEEDS="${SEEDS:-42,123,456,789,1024}"
QUICK="${QUICK:-0}"
EPOCHS=""
if [ "$QUICK" = "1" ]; then
  EPOCHS="--epochs 2"
  echo "QUICK=1: using --epochs 2 for faster smoke test"
fi

echo "Seeds: $SEEDS"
echo "=== 1. Length generalization (all tasks x models) ==="
python3 -m src.run_length_gen --all --seeds "$SEEDS" $EPOCHS

echo "=== 2. MNIST (mlp, ion) ==="
python3 -m src.run_mnist --seeds "$SEEDS" $EPOCHS

echo "=== 3. Depth stability (depths 4,8,16,32; mlp, ion) ==="
python3 -m src.run_depth --seeds "$SEEDS" $EPOCHS

echo "=== 4. Ablations (lambda, p_dim on cumsum + mnist) ==="
python3 -m src.run_ablations --sweep both --seeds "$SEEDS" $EPOCHS

echo "=== 5. Mechanistic ablations (cumsum) ==="
python3 -m src.run_mechanistic_ablations --seeds "$SEEDS" $EPOCHS

echo "=== 6. Invariant drift (requires MNIST ION) ==="
python3 -m src.run_drift || true

echo "=== 7. Aggregate length-gen tables ==="
python3 -m src.run_length_gen --aggregate-only

echo "=== 8. Generate figures and tables ==="
python3 -m src.run_figures_tables

echo "Done. Results in results/; figures in paper/figures/; tables in paper/tables/."
