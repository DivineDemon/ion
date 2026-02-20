"""
Universal (layer-depth) ION for feedforward networks (MLP, etc.).

Formulation from draft:
  Index = layer depth (not time).
  p^{(l)} = P(h^{(l)})       (invariant at layer l)
  p̂^{(l+1)} = F(p^{(l)})    (predicted next invariant)
  L_ind = Σ_l ||P(h^{(l+1)}) - F(p^{(l)})||^2

Integrates into MLP: same layers as baseline MLP plus shared P and F per layer.
Configurable: p_dim, λ (inductive weight).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F_fn


class IONUniversal(nn.Module):
    """
    Universal ION: MLP with inductive consistency across layers.

    At each layer l: h^{(l+1)} = φ(h^{(l)}), and we enforce
    P(h^{(l+1)}) ≈ F(P(h^{(l)})).

    Same interface as MLPBaseline for task output; forward returns
    (output, inductive_loss) for training.
    """

    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        num_layers: int,
        output_size: int,
        p_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_size = output_size
        self.p_dim = p_dim

        # Build hidden layers (same as MLP baseline)
        layers = []
        in_d = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_d = hidden_dim
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(in_d, output_size)

        # P: invariant projection h -> p (shared across layers)
        self.P = nn.Sequential(
            nn.Linear(hidden_dim, p_dim),
            nn.Tanh(),
        )

        # F: inductive rule p^{(l)} -> p^{(l+1)} (no input x at layer level)
        self.F = nn.Sequential(
            nn.Linear(p_dim, p_dim),
            nn.Tanh(),
        )

    def _get_layer_outputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run forward through layers and return h^{(0)}, h^{(1)}, ..., h^{(L)}.
        h^{(0)} = input (flattened), h^{(1)} = after first linear+ReLU, etc.
        """
        if x.dim() > 2:
            x = x.flatten(1)
        h_list = [x]
        h = x
        for i, m in enumerate(self.layers):
            h = m(h)
            if isinstance(m, nn.ReLU):
                h_list.append(h)
        return h_list

    def forward(
        self, x: torch.Tensor, return_inductive_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, C, H, W) for images or (B, input_size) already flat
            return_inductive_loss: if True, compute and return inductive loss

        Returns:
            output: (B, output_size) logits
            inductive_loss: scalar, or None if return_inductive_loss=False
        """
        h_list = self._get_layer_outputs(x)
        # h_list[0] = input, h_list[1] = after first ReLU, ..., h_list[L] = after last ReLU
        # For inductive loss we need pairs (h^{(l)}, h^{(l+1)}) where both are hidden_dim.
        # Input might be input_size != hidden_dim. So we only apply P/F to layers
        # that have hidden_dim. That's h_list[1], h_list[2], ..., h_list[L].

        # Inductive loss: pairs of consecutive hidden layers (skip input layer)
        # h_list[0] = input (input_size), h_list[1..L] = hidden (hidden_dim)
        inductive_loss = None
        if return_inductive_loss and len(h_list) >= 3:
            losses = []
            for l in range(1, len(h_list) - 1):
                h_l = h_list[l]
                h_next = h_list[l + 1]
                p_l = self.P(h_l)
                p_next = self.P(h_next)
                p_pred = self.F(p_l)
                losses.append(F_fn.mse_loss(p_next, p_pred, reduction="mean"))
            if losses:
                inductive_loss = torch.stack(losses).mean()

        # Task output
        h_final = h_list[-1]
        output = self.head(h_final)
        return output, inductive_loss
