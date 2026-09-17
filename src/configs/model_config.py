from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int

    n_kv_heads: int
    max_seq_len: int

    batch_size: int
    max_steps: int
    grad_accumulation_steps: int

    muon_lr: float
    weight_decay: float
    dropout: float
    grad_clip: float

    eval_every: int
    eval_steps: int

    use_amp: bool
    device: str
    vocab_size: int | None

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_kv_groups(self) -> int:
        return self.n_heads // self.n_kv_heads

# import yaml

# with open("configs/model.yaml") as f:
#     config = ModelConfig(**yaml.safe_load(f))