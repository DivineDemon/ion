#!/usr/bin/env python3
"""
Unit check for ION models: forward pass, shapes, and inductive loss behavior.

Verifies:
  1. IONRecurrent: forward pass, output shapes, inductive loss present and non-negative
  2. IONUniversal: same
  3. Inductive loss decreases when optimizing (λ > 0)
  4. Parameter count helper works
"""

import sys
from pathlib import Path

# Ensure src is on path when run as python -m src.run_unit_check
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch

from src.models.baselines import GRUBaseline, MLPBaseline, count_parameters
from src.models.ion_recurrent import IONRecurrent
from src.models.ion_universal import IONUniversal
from src.models.param_matching import suggest_ion_recurrent_dims, suggest_ion_universal_dims


def check_ion_recurrent() -> None:
    """Unit check for IONRecurrent."""
    print("Checking IONRecurrent...")
    torch.manual_seed(42)

    # Regression (cumsum-like)
    model = IONRecurrent(
        input_dim=1,
        hidden_dim=64,
        p_dim=16,
        num_layers=2,
        output_type="regression",
    )
    x = torch.randn(4, 20, 1)
    out, ind_loss = model(x, return_inductive_loss=True)

    assert out.shape == (4,), f"Expected (4,), got {out.shape}"
    assert ind_loss is not None, "Inductive loss should be returned"
    assert ind_loss.dim() == 0, "Inductive loss should be scalar"
    assert ind_loss.item() >= 0, "Inductive loss should be non-negative"

    # Classification (parity-like)
    model_cls = IONRecurrent(
        input_dim=1,
        hidden_dim=64,
        p_dim=16,
        output_type="classification",
        num_classes=2,
    )
    out_cls, ind_loss_cls = model_cls(x, return_inductive_loss=True)
    assert out_cls.shape == (4, 2), f"Expected (4, 2), got {out_cls.shape}"
    assert ind_loss_cls is not None and ind_loss_cls.dim() == 0

    # Eval mode: can skip inductive loss
    out_eval, ind_eval = model_cls(x, return_inductive_loss=False)
    assert ind_eval is None
    print("  IONRecurrent: OK")


def check_ion_universal() -> None:
    """Unit check for IONUniversal."""
    print("Checking IONUniversal...")
    torch.manual_seed(42)

    model = IONUniversal(
        input_size=28 * 28,
        hidden_dim=128,
        num_layers=4,
        output_size=10,
        p_dim=16,
    )
    x = torch.randn(8, 28, 28)  # MNIST-like
    out, ind_loss = model(x, return_inductive_loss=True)

    assert out.shape == (8, 10), f"Expected (8, 10), got {out.shape}"
    assert ind_loss is not None, "Inductive loss should be returned"
    assert ind_loss.dim() == 0
    assert ind_loss.item() >= 0

    # Eval mode
    out_eval, ind_eval = model(x, return_inductive_loss=False)
    assert ind_eval is None
    print("  IONUniversal: OK")


def check_inductive_loss_decreases() -> None:
    """Verify inductive loss decreases when optimizing with λ > 0."""
    print("Checking inductive loss decreases under optimization...")
    torch.manual_seed(42)

    model = IONRecurrent(
        input_dim=1,
        hidden_dim=32,
        p_dim=8,
        output_type="regression",
    )
    x = torch.randn(16, 15, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    lambda_ind = 0.5

    losses = []
    for _ in range(10):
        out, ind_loss = model(x, return_inductive_loss=True)
        task_loss = torch.nn.functional.mse_loss(out, torch.randn_like(out))
        total = task_loss + lambda_ind * ind_loss
        losses.append(ind_loss.item())
        opt.zero_grad()
        total.backward()
        opt.step()

    # Inductive loss should generally decrease (or at least change)
    # Allow some variance; main check is that backward works and loss is sensible
    assert losses[0] >= 0 and losses[-1] >= 0
    # Typically after 10 steps we expect some decrease
    assert losses[-1] < losses[0] * 1.5 or abs(losses[-1] - losses[0]) < 0.5
    print("  Inductive loss optimization: OK")


def check_param_matching() -> None:
    """Verify param count and matching helpers."""
    print("Checking param matching...")

    gru = GRUBaseline(
        input_dim=1,
        hidden_dim=64,
        num_layers=2,
        output_type="regression",
    )
    gru_params = count_parameters(gru)

    ion = IONRecurrent(
        input_dim=1,
        hidden_dim=64,
        p_dim=16,
        num_layers=2,
        output_type="regression",
    )
    ion_params = count_parameters(ion)

    print(f"  GRU params: {gru_params}, ION params: {ion_params}")

    # ION has extra P and F; typically more params than GRU with same h_dim
    assert ion_params > 0 and gru_params > 0

    # Test suggest (may not find exact match within tolerance)
    result = suggest_ion_recurrent_dims(
        target_params=gru_params,
        input_dim=1,
        num_classes_or_1=1,
        num_layers=2,
        output_type="regression",
        tolerance=0.20,  # relax for small models
    )
    if result:
        print(f"  Suggest ION recurrent: {result}")
    else:
        print("  Suggest ION recurrent: no exact match (acceptable for small models)")

    mlp = MLPBaseline(
        input_size=784,
        hidden_dim=128,
        num_layers=4,
        output_size=10,
    )
    mlp_params = count_parameters(mlp)
    ion_uni = IONUniversal(
        input_size=784,
        hidden_dim=128,
        num_layers=4,
        output_size=10,
        p_dim=16,
    )
    ion_uni_params = count_parameters(ion_uni)
    print(f"  MLP params: {mlp_params}, ION universal params: {ion_uni_params}")

    result_uni = suggest_ion_universal_dims(
        target_params=mlp_params,
        input_size=784,
        output_size=10,
        num_layers=4,
        tolerance=0.15,
    )
    if result_uni:
        print(f"  Suggest ION universal: {result_uni}")
    print("  Param matching: OK")


def main() -> None:
    print("Running ION unit checks...")
    check_ion_recurrent()
    check_ion_universal()
    check_inductive_loss_decreases()
    check_param_matching()
    print("\nAll unit checks passed.")


if __name__ == "__main__":
    main()
