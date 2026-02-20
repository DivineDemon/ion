# Ablation results (Phase A9)

Ablations for **λ** (inductive loss weight) and **p_dim** (invariant size) on one length-gen task (cumsum) and MNIST.

## Naming

- **Lambda sweep:** `lambda_<value>_<task>_seed_<seed>.json` (e.g. `lambda_0.1_cumsum_seed_42.json`)
- **p_dim sweep:** `p_dim_<value>_<task>_seed_<seed>.json` (e.g. `p_dim_8_mnist_seed_42.json`)

Checkpoints and run artifacts for each (sweep, value, task) are in subdirs: `lambda_0.1_cumsum/`, `lambda_0.1_mnist/`, etc.

## Values

- **λ:** 0.1, 0.3, 0.5, 0.7, 1.0
- **p_dim:** 4, 8, 16, 32

## How to reproduce

```bash
python -m src.run_ablations                    # both sweeps, both tasks, 5 seeds
python -m src.run_ablations --sweep lambda     # lambda only
python -m src.run_ablations --sweep p_dim     # p_dim only
python -m src.run_ablations --task cumsum     # cumsum only
python -m src.run_ablations --n-seeds 3       # use 3 seeds
```

Or: `./scripts/run_ablations.sh`
