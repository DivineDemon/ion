"""
ION-Transformer: encoder-only Transformer with inductive auxiliary loss over time.

Same formulation as IONRecurrent but applied to the Transformer encoder output:
  h_t = encoder(x)[:, t]   (hidden state at position t)
  p_t = P(h_t),  p_{t+1} ≈ F(p_t, x_t)
  L_ind = Σ_t ||P(h_{t+1}) - F(P(h_t), x_t)||^2

Use for Long Range Arena (LRA) and other long-sequence classification tasks.
Interface: (B, T, D) in, (output, inductive_loss) out; same training hook as IONRecurrent.
"""

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class IONTransformer(nn.Module):
    """
    Transformer encoder with inductive consistency over time steps.

    Input: (B, T, input_dim). Output: task logits from last position, plus L_ind.
    P and F are applied to the encoder output sequence; L_ind ties consecutive
    positions via the learned invariant and rule.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        p_dim: int,
        max_len: int = 2048,
        output_type: Literal["classification", "regression"] = "classification",
        num_classes: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_len = max_len
        self.output_type = output_type
        self.num_classes = num_classes

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.P = nn.Sequential(
            nn.Linear(d_model, p_dim),
            nn.Tanh(),
        )
        self.F = nn.Sequential(
            nn.Linear(p_dim + input_dim, p_dim),
            nn.Tanh(),
        )

        if output_type == "classification":
            if num_classes is None:
                raise ValueError("num_classes required for classification")
            self.head = nn.Linear(d_model, num_classes)
        else:
            self.head = nn.Linear(d_model, 1)

    def forward(
        self, x: torch.Tensor, return_inductive_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            return_inductive_loss: if True, compute and return L_ind over time

        Returns:
            output: (B, num_classes) or (B,) for regression
            inductive_loss: scalar, or None if return_inductive_loss=False
        """
        B, T, D_in = x.shape
        x_proj = self.input_proj(x)
        x_proj = x_proj + self.pos_embed[:, :T, :]
        enc = self.encoder(x_proj)

        inductive_loss = None
        if return_inductive_loss and T >= 2:
            losses = []
            for t in range(T - 1):
                h_t = enc[:, t, :]
                h_next = enc[:, t + 1, :]
                x_t = x[:, t, :]
                p_t = self.P(h_t)
                p_next = self.P(h_next)
                p_pred = self.F(torch.cat([p_t, x_t], dim=-1))
                losses.append(F.mse_loss(p_next, p_pred, reduction="mean"))
            inductive_loss = torch.stack(losses).mean()

        last_h = enc[:, -1, :]
        out = self.head(last_h)
        if self.output_type == "regression":
            out = out.squeeze(-1)
        return out, inductive_loss
