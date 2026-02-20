# MNIST experiment results

## Contents
- `mlp/run_<seed>.json`: per-seed results for MLP (history = train/val loss curves).
- `ion/run_<seed>.json`: per-seed results for ION (history = train/val loss curves).
- `test_accuracy_summary.json`: mean ± std test accuracy per model.
- `mnist_results.csv`: table of test accuracy (and optional robustness).

## Loss curves
Each `run_<seed>.json` contains a `history` object with `train_losses`, `val_losses`, `train_metrics`, `val_metrics` (one value per epoch). Use these to plot train/val loss and accuracy curves.
