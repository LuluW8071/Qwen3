from pathlib import Path

import torch
import yaml

from src.configs.model_config import ModelConfig
from src.neuralnet.attention import GroupedQueryAttention


def test_grouped_query_attention_example():
    config_path = Path(__file__).parents[1] / "src/configs/example.yaml"
    with config_path.open() as config_file:
        config = ModelConfig(**yaml.safe_load(config_file))

    attention = GroupedQueryAttention(config)
    x = torch.randn(1, 4, config.d_model)

    output = attention(x)

    assert output.shape == x.shape
