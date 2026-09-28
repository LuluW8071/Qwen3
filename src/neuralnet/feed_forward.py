import torch

from torch import nn
from torch.nn import functional as F


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Learns gating signal for each intermediate feature.
        # SiLU is applied to this branch before gating the value branch.
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)

        # Projects intermediate features back to model dimension.
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

        # Projects input into intermediate/value features.
        # These features are selectively scaled by the gate branch.
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU Feed-Forward Network.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
               Input is expected to be RMS-normalized by the surrounding
               Transformer block.

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """

        # Gate branch:
        #   d_model -> d_ff
        #   SiLU provides smooth, learnable nonlinear gating.

        # Element-wise gating:
        # The gate selectively suppresses, preserves, or amplifies
        # intermediate features from the value branch.
        activated_x = F.silu(self.gate_proj(x)) * self.up_proj(x)

        # Apply dropout, then project d_ff -> d_model.
        return self.down_proj(self.dropout(activated_x))
