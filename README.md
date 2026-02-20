# PMI-NN (ION)

This repository accompanies a research paper on a **novel approach for training neural networks using the Principle of Mathematical Induction (PMI)**. **ION** (Inductive invariant for Ordered Networks) enforces *inductive consistency*: the model learns an invariant \(P\) and a rule \(F\) so that along sequences or depth, \(P(h_{t+1}) \approx F(P(h_t), x_t)\). This supports **length generalization** (train on short sequences, evaluate on long ones) and **depth stability** (consistent behavior across many layers).

**Summary:** ION adds an auxiliary loss that ties consecutive hidden states via a learned invariant and update rule. We compare ION to GRU, LSTM, and MLP baselines on synthetic length-gen tasks (cumsum, parity, Dyck), MNIST, depth experiments, and ablations. Results show improved length generalization and stable depth when the inductive regularizer is used; we also report a failure case (last-token prediction) where ION does not help.

---

## Requirements

- **Python:** 3.9 or higher  
- **OS:** Windows, macOS, or Linux  
- **Hardware:** CPU is enough for small runs; **GPU (CUDA) recommended** for full experiments. The default config uses `device: cuda` when available.  
- **GPU (e.g. NVIDIA RTX 3080):** For GPU use, install PyTorch with CUDA support, e.g.  
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`  
  (or `cu121`). Then verify:  
  `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"`  
- **Optional:** Bash (for `scripts/run_all_experiments.sh` and `scripts/fetch_lra_listops.sh`; on Windows you can use WSL or run the Python commands below instead)

---

## Setup (after cloning)

### 1. Create and activate a virtual environment (recommended)

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

This installs PyTorch, torchvision, numpy, matplotlib, seaborn, pandas, pyyaml, and tqdm.

### 3. (Optional) LRA ListOps data

Required only for LRA experiments (`python -m src.run_lra --task listops ...`). Either:

- **Bash:** From repo root run  
  `./scripts/fetch_lra_listops.sh`  
  (downloads or generates data under `data/lra/lra_release/listops-1000/`).

- **Manual:** See `data/README.md` for download links and directory layout.

MNIST and CIFAR-10 are downloaded automatically on first use. Length-generalization tasks use synthetic data generated at runtime (no extra files).

---

## How to run

All commands are run from the **repository root**.

### Verify installation

```bash
python -m src.run_unit_check
```

Checks: forward pass shapes, inductive loss, parameter-count helpers.

### Main entry points

| What you want              | Command |
|----------------------------|--------|
| Length-gen (one task/model)| `python -m src.run_length_gen --task cumsum --model ion` |
| Length-gen (all tasks×models) | `python -m src.run_length_gen --all` |
| MNIST                     | `python -m src.run_mnist --seeds 42,123,456,789,1024` |
| Depth (CIFAR)             | `python -m src.run_depth --dataset cifar` |
| Ablations                 | `python -m src.run_ablations --sweep both --seeds 42,123,456,789,1024` |
| Mechanistic ablations     | `python -m src.run_mechanistic_ablations --seeds 42,123,456,789,1024` |
| Drift (needs MNIST first) | `python -m src.run_drift` |
| LRA ListOps               | `python -m src.run_lra --task listops --model transformer` or `--model ion` |

**Quick smoke test** (few epochs, one seed):

```bash
python -m src.run_length_gen --task cumsum --model ion --epochs 2 --seeds 42
python -m src.run_mnist --epochs 5 --seeds 42
```

### Full experiment suite

Runs length-gen, LRA ListOps, depth (MNIST and CIFAR-10), MNIST, ablations, mechanistic ablations, drift, then aggregates and generates figures (several hours; GPU recommended):

```bash
./scripts/run_all_experiments.sh
```

Quick run with fewer epochs:

```bash
QUICK=1 ./scripts/run_all_experiments.sh
```

On Windows without Bash, run the same `python -m src.run_*` commands as in the script (see `scripts/run_all_experiments.sh` for the order).

---

## Where results go

- **Length-gen:** `results/length_gen/<task>/<model>/` (e.g. `accuracy_vs_length.csv`, `summary.json`).  
- **MNIST:** `results/mnist/`  
- **Depth:** `results/depth/` (or `results/depth/cifar/` for CIFAR)  
- **LRA:** `results/lra/listops/<model>/`  
- **Ablations / mechanistic / drift:** `results/ablations/`, `results/mechanistic_ablations/`, `results/drift/`  

Figures and LaTeX tables for the paper are generated into `paper/figures/` and `paper/tables/` by:

```bash
python -m src.run_figures_tables
```

(Requires existing results in `results/`.)

---

## Parameter matching (ION vs baselines)

For fair comparison, match ION parameter count to baselines using `count_parameters(model)`:

```python
from src.models import count_parameters, IONRecurrent, GRUBaseline, suggest_ion_recurrent_dims

gru = GRUBaseline(input_dim=1, hidden_dim=64, num_layers=2, output_type="regression")
target = count_parameters(gru)

cfg = suggest_ion_recurrent_dims(target, input_dim=1, num_classes_or_1=1, num_layers=2, output_type="regression")
if cfg:
    ion = IONRecurrent(..., hidden_dim=cfg["hidden_dim"], p_dim=cfg["p_dim"])
```

Prefer matched counts within a few percent; otherwise report both.

---

## Reproducing the paper

1. **Environment:** Python 3.9+; `pip install -r requirements.txt`. Main deps: PyTorch, numpy, pyyaml, matplotlib, pandas.  
2. **Seeds:** Default seeds are in `configs/base.yaml` (`42, 123, 456, 789, 1024`); override with `--seeds` or `SEEDS=...`.  
3. **Full run:** From repo root: `./scripts/run_all_experiments.sh` (or the equivalent `python -m src.run_*` sequence on Windows).  
4. **Figures/tables only:** `python -m src.run_figures_tables`. If figures are missing, run `python paper/figures/create_placeholders.py` so the paper compiles.  
5. **Build the paper:** Open `paper/` in Overleaf (or any LaTeX editor) and compile `main.tex`.

---

## Data and config reference

- **Data layout and downloads:** `data/README.md`  
- **Base config (device, seeds, output root):** `configs/base.yaml`  
- **Length-gen configs:** `configs/length_gen/` (cumsum, parity, dyck1, dyck2)
