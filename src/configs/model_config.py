from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int    # Embedding dimension
    n_heads: int    # Number of attention heads
    n_layers: int   # Number of transformer blocks
    d_ff: int       # Feed-forward dimension

    n_kv_heads: int
    sliding_window: int
    attention_bias: bool
    rms_norm_eps: float
    rms_norm: bool

    max_seq_len: int
    num_documents: int
    max_tokens: int

    batch_size: int
    max_steps: int
    gradient_accumulation_steps: int
    muon_lr: float

    eval_every: int
    eval_steps: int

    weight_decay: float
    dropout: float
    grad_clip: float

    use_amp: bool
    device: str
    vocab_size: int | None

    def __post_init__(self):
        self.rms_norm_eps = float(self.rms_norm_eps)

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

    @property
    def d_k(self) -> int:
        return self.d_model // self.n_heads # head_dim

    @property
    def n_kv_groups(self) -> int:
        return self.n_heads // self.n_kv_heads

# import yaml

# with open("configs/model.yaml") as f:
#     config = ModelConfig(**yaml.safe_load(f))
