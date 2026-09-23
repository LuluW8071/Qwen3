import torch

from torch import nn
from torch.nn import functional as F

from src.configs.logging import logger
from src.configs.model_config import ModelConfig
from src.neuralnet.rotatory_pos_embed import RoPE


def repeat_kv_heads(
    hidden_states: torch.Tensor,
    n_rep: int,
) -> torch.Tensor:
    """
    Equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seq_len, head_dim) to
    (batch, num_attn_heads, seq_len, head_dim)
    """

    # Extract 4 dims from input tensor
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape

    # Early return if no repeatition is required
    if n_rep == 1:
        return hidden_states

    # Add a new dimension at index 2 (after num_key_value_heads) and expand
    # Shape Transformation:
    # (batch, num_key_value_heads, seq_len, head_dim)
    # -> (batch, num_key_value_heads, 1, seq_len, head_dim)
    # -> (batch, num_key_value_heads, n_rep, seq_len, head_dim)
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)

    # Flatten the num_key_value_heads and n_rep dimensions together
    # Final shape: (batch, num_key_value_heads * n_rep, seq_len, head_dim)
    # This repeats each key/value head n_rep times
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)



class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_groups = config.n_kv_groups
        self.d_k = config.d_k

        # Separate linear layers for Q, K, V
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_k, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_k, bias=config.attention_bias)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

        # QK-Normalization layers
        if config.rms_norm:
            self.q_norm = nn.RMSNorm(self.d_k, eps=config.rms_norm_eps)
            self.k_norm = nn.RMSNorm(self.d_k, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.LayerNorm(self.d_k, eps=config.rms_norm_eps)
            self.k_norm = nn.LayerNorm(self.d_k, eps=config.rms_norm_eps)

        self.rotary = RoPE(self.d_k, config.max_seq_len)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Qwen3 Grouped Query Attention
        ---

        Args:
            x (torch.Tensor): Input tensor (batch, seq_len, d_model)
            Here `x` input is normalized by RMS Norm

        Returns:
            torch.Tensor: Attention output (batch, seq_len, d_model)
        """
        logger.debug("x: {}\nx.shape: {}", x, x.shape)

        batch_size, seq_len = x.size(0), x.size(1)
        logger.debug("batch_size, seq_len: {}, {}", batch_size, seq_len)
        logger.debug("self.n_heads, self.d_k: {}, {}", self.n_heads, self.d_k)

        # 1. Project Q, K, V separately
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        logger.debug("Projected QKV\nq: {}\nk: {}\nv: {}", q, k, v)
        logger.debug("q.shape, k.shape, v.shape: {}, {}, {}", q.shape, k.shape, v.shape)

        # 2. Reshape into heads
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        logger.debug("Reshaped to attention heads\nq: {}\nk: {}\nv: {}", q, k, v)
        logger.debug("q.shape, k.shape, v.shape: {}, {}, {}", q.shape, k.shape, v.shape)

        # 3. Apply QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        logger.debug("q.shape, k.shape: {}, {}", q.shape, k.shape)

        # 4. Apply RoPE
        # Transpose to (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k) for rotary
        q = self.rotary(q.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        k = self.rotary(k.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        logger.debug("q.shape, k.shape: {}, {}", q.shape, k.shape)

        # Transpose for attention: (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
        Q = q.transpose(1, 2)
        K = k.transpose(1, 2)
        V = v.transpose(1, 2)
        logger.debug("Q.shape, K.shape, V.shape: {}, {}, {}", Q.shape, K.shape, V.shape)

        # 5. Repeat K and V heads for GQA
        K = repeat_kv_heads(K, self.n_kv_groups)
        V = repeat_kv_heads(V, self.n_kv_groups)
        logger.debug("K.shape, V.shape: {}, {}", K.shape, V.shape)

        # 6. Scaled Dot-Product Attention
        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        logger.debug("attn_output.shape: {}", attn_output.shape)

        # 7. Reshape and final projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        logger.debug("attn_output.shape: {}", attn_output.shape)

        return self.w_o(attn_output)
