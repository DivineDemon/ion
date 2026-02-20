"""
Baseline models for sequence and feedforward settings.
Same interface: (B, T, D) in for recurrent; task output (logits or scalars) out.
No inductive loss; training uses only task loss (CE or MSE).

Parameter matching (ION vs baselines):
  Use count_parameters(model) to get total trainable params. When comparing ION to
  a baseline, either (1) set ION h_dim/p_dim so total params are within a few
  percent of the baseline, or (2) report both and document the mismatch. Prefer
  matched counts for fair comparison. Formula: sum(p.numel() for p in model.parameters() if p.requires_grad).
"""

from typing import Literal, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Parameter count (for matching ION vs baselines)
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """
    Total number of trainable parameters.
    Use this to match ION and baseline sizes when comparing fairly.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Recurrent baselines: GRU and LSTM
# ---------------------------------------------------------------------------


class RecurrentBaseline(nn.Module):
    """
    Base for GRU/LSTM: input (B, T, D), output from last hidden state.
    Supports classification (logits) or regression (scalar per sequence).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_type: Literal["classification", "regression"],
        num_classes: Optional[int] = None,
        dropout: float = 0.0,
        cell: Literal["gru", "lstm"] = "gru",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_type = output_type
        self.num_classes = num_classes
        self.cell = cell

        if cell == "gru":
            self.rnn = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        else:
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )

        if output_type == "classification":
            if num_classes is None:
                raise ValueError("num_classes required for classification")
            self.head = nn.Linear(hidden_dim, num_classes)
        else:
            self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            Logits (B, num_classes) for classification, or (B,) for regression.
        """
        # x: (B, T, D)
        out, _ = self.rnn(x)  # out: (B, T, hidden_dim)
        last_h = out[:, -1, :]  # (B, hidden_dim)
        logits_or_scalar = self.head(last_h)  # (B, num_classes) or (B, 1)
        if self.output_type == "regression":
            return logits_or_scalar.squeeze(-1)  # (B,)
        return logits_or_scalar  # (B, num_classes)


class GRUBaseline(RecurrentBaseline):
    """GRU baseline: (B, T, D) in, task output out. Same interface as LSTM."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        output_type: Literal["classification", "regression"] = "classification",
        num_classes: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_type=output_type,
            num_classes=num_classes,
            dropout=dropout,
            cell="gru",
        )


class LSTMBaseline(RecurrentBaseline):
    """LSTM baseline: (B, T, D) in, task output out. Same interface as GRU."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        output_type: Literal["classification", "regression"] = "classification",
        num_classes: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_type=output_type,
            num_classes=num_classes,
            dropout=dropout,
            cell="lstm",
        )


# ---------------------------------------------------------------------------
# Transformer baseline (optional, for sequence tasks)
# ---------------------------------------------------------------------------


class TransformerBaseline(nn.Module):
    """
    Small encoder-only Transformer for sequence tasks.
    Input (B, T, D), output from last position (same as GRU/LSTM).
    Fixed d_model, nhead, num_layers, max_len; use count_parameters() to match ION.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        max_len: int = 512,
        output_type: Literal["classification", "regression"] = "classification",
        num_classes: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
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

        if output_type == "classification":
            if num_classes is None:
                raise ValueError("num_classes required for classification")
            self.head = nn.Linear(d_model, num_classes)
        else:
            self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            Logits (B, num_classes) or (B,) for regression.
        """
        B, T, _ = x.shape
        x = self.input_proj(x)  # (B, T, d_model)
        x = x + self.pos_embed[:, :T, :]
        # TransformerEncoder expects (B, T, d_model); attention mask not applied for simplicity
        x = self.encoder(x)  # (B, T, d_model)
        last_h = x[:, -1, :]  # (B, d_model)
        out = self.head(last_h)
        if self.output_type == "regression":
            return out.squeeze(-1)
        return out


# ---------------------------------------------------------------------------
# Transformer with ALiBi (Attention with Linear Biases) for length extrapolation
# ---------------------------------------------------------------------------


def _alibi_mask(seq_len: int, n_heads: int, device: torch.device) -> torch.Tensor:
    """ALiBi: bias matrix (seq_len, seq_len) added to attention scores. Uses mean slope across heads."""
    # Per-head slopes: 2^(-8/n), 2^(-16/n), ... (Press et al.)
    slopes = torch.tensor([2 ** (-8 * (i + 1) / n_heads) for i in range(n_heads)], device=device, dtype=torch.float32)
    slope = slopes.mean().item()
    # bias[i,j] = -slope * |i - j|
    i = torch.arange(seq_len, device=device, dtype=torch.float32)
    j = torch.arange(seq_len, device=device, dtype=torch.float32)
    dist = (i.unsqueeze(1) - j.unsqueeze(0)).abs()
    mask = -slope * dist
    return mask  # (T, T)


class TransformerALiBi(nn.Module):
    """
    Encoder-only Transformer with ALiBi (no learned positional embedding).
    Same interface as TransformerBaseline; uses attention bias for length extrapolation.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        max_len: int = 512,
        output_type: Literal["classification", "regression"] = "classification",
        num_classes: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.max_len = max_len
        self.output_type = output_type
        self.num_classes = num_classes

        self.input_proj = nn.Linear(input_dim, d_model)
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

        if output_type == "classification":
            if num_classes is None:
                raise ValueError("num_classes required for classification")
            self.head = nn.Linear(d_model, num_classes)
        else:
            self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x = self.input_proj(x)
        alibi = _alibi_mask(T, self.nhead, x.device)
        x = self.encoder(x, mask=alibi)
        last_h = x[:, -1, :]
        out = self.head(last_h)
        if self.output_type == "regression":
            return out.squeeze(-1)
        return out


# ---------------------------------------------------------------------------
# MLP baseline (for MNIST / depth experiments)
# ---------------------------------------------------------------------------


class MLPBaseline(nn.Module):
    """
    Feedforward MLP: flatten -> linear -> ReLU -> ... -> output.
    Configurable width (hidden_dim) and depth (num_layers). Use count_parameters() for matching.
    """

    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.0,
    ):
        """
        Args:
            input_size: Flattened input dimension (e.g. 28*28 for MNIST).
            hidden_dim: Width of each hidden layer.
            num_layers: Number of hidden layers (each: linear -> ReLU).
            output_size: Output dimension (e.g. 10 for MNIST classes, or 1 for regression).
            dropout: Dropout after hidden layers (0 to disable).
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_size = output_size

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) for images or (B, input_size) already flat.
        Returns:
            (B, output_size) logits.
        """
        if x.dim() > 2:
            x = x.flatten(1)
        x = self.layers(x)
        return self.head(x)
