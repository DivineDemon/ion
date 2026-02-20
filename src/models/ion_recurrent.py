"""
Recurrent ION (Inductive Operator Network) for sequence tasks.

Formulation from draft:
  h_{n+1} = φ_θ(h_n, x_n)   (e.g. GRU cell)
  p_n = P_ψ(h_n)            (invariant projection)
  p_{n+1} ≈ F_ω(p_n, x_n)   (inductive rule)

Loss: L = L_task + λ * L_ind
  L_ind = Σ_t ||P(h_{t+1}) - F(P(h_t), x_t)||^2

Configurable: h_dim, p_dim, λ (inductive weight).
"""

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class IONRecurrent(nn.Module):
    """
    Recurrent ION: learns inductive invariant P and rule F alongside GRU transition φ.

    Input: (B, T, D) sequences.
    Output: task output (logits or scalar) from last hidden; plus inductive loss for training.

    Same interface as RecurrentBaseline for task output; forward returns (output, inductive_loss)
    for training (use inductive_loss in total loss).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        p_dim: int,
        num_layers: int = 1,
        output_type: Literal["classification", "regression"] = "classification",
        num_classes: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.p_dim = p_dim
        self.num_layers = num_layers
        self.output_type = output_type
        self.num_classes = num_classes

        # φ: inductive state transition (GRU)
        self.phi = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # P: invariant projection h -> p
        self.P = nn.Sequential(
            nn.Linear(hidden_dim, p_dim),
            nn.Tanh(),
        )

        # F: inductive rule (p, x) -> p_next
        self.F = nn.Sequential(
            nn.Linear(p_dim + input_dim, p_dim),
            nn.Tanh(),
        )

        # Task head
        if output_type == "classification":
            if num_classes is None:
                raise ValueError("num_classes required for classification")
            self.head = nn.Linear(hidden_dim, num_classes)
        else:
            self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self, x: torch.Tensor, return_inductive_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            return_inductive_loss: if True, compute and return inductive loss

        Returns:
            output: (B, num_classes) or (B,) for regression
            inductive_loss: scalar, or None if return_inductive_loss=False
        """
        B, T, D = x.shape
        device = x.device

        # Run GRU to get all hidden states
        # gru_out: (B, T, hidden_dim), h_n: (num_layers, B, hidden_dim)
        gru_out, _ = self.phi(x)
        # gru_out[:, t, :] = h_t after processing x_0..x_t

        # Build inductive loss: for each t in 0..T-2 (or 0..T-1 with proper indexing)
        # At step t: h_t = gru_out[:, t], h_{t+1} = gru_out[:, t+1]
        # p_t = P(h_t), p_{t+1} = P(h_{t+1}), p_pred = F(p_t, x_t)
        # loss_t = ||p_{t+1} - p_pred||^2
        inductive_loss = None
        if return_inductive_loss and T >= 2:
            losses = []
            for t in range(T - 1):
                h_t = gru_out[:, t, :]  # (B, hidden_dim)
                h_next = gru_out[:, t + 1, :]  # (B, hidden_dim)
                x_t = x[:, t, :]  # (B, input_dim)

                p_t = self.P(h_t)  # (B, p_dim)
                p_next = self.P(h_next)  # (B, p_dim)
                p_pred = self.F(torch.cat([p_t, x_t], dim=-1))  # (B, p_dim)

                losses.append(F.mse_loss(p_next, p_pred, reduction="mean"))
            inductive_loss = torch.stack(losses).mean()

        # Task output from last hidden state
        last_h = gru_out[:, -1, :]  # (B, hidden_dim)
        logits_or_scalar = self.head(last_h)
        if self.output_type == "regression":
            output = logits_or_scalar.squeeze(-1)  # (B,)
        else:
            output = logits_or_scalar  # (B, num_classes)

        return output, inductive_loss
