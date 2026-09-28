import torch
from torch import nn

from src.configs.model_config import ModelConfig
from src.neuralnet.attention import GroupedQueryAttention
from src.neuralnet.feed_forward import SwiGLUFeedForward

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):  # Pass the entire config object
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = SwiGLUFeedForward(
            config.d_model,
            config.d_ff,
            config.dropout
        )
        self.norm1 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x
