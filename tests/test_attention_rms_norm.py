import copy
import time
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from src.configs.logging import logger
from src.configs.model_config import ModelConfig
from src.neuralnet.attention import GroupedQueryAttention


def _load_config() -> ModelConfig:
    config_path = Path(__file__).parents[1] / "src/configs/example.yaml"
    with config_path.open() as config_file:
        return ModelConfig(**yaml.safe_load(config_file))


def _build_attention(config: ModelConfig, rms_norm: bool) -> GroupedQueryAttention:
    variant_config = copy.copy(config)
    variant_config.rms_norm = rms_norm
    attention = GroupedQueryAttention(variant_config)
    attention.eval()
    return attention


def _latency_ms(
    attention: GroupedQueryAttention,
    x: torch.Tensor,
    device: torch.device,
) -> float:
    for _ in range(10):
        attention(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(50):
        attention(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000 / 50


def test_rms_norm_quality_and_latency_drift():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = replace(_load_config(), d_model=32)
    x = torch.randn(2, 16, config.d_model, device=device)

    rms_attention = _build_attention(config, rms_norm=True).to(device)
    no_rms_attention = _build_attention(config, rms_norm=False).to(device)
    shared_weights = {
        name: value
        for name, value in rms_attention.state_dict().items()
        if not name.startswith(("q_norm", "k_norm"))
    }
    no_rms_attention.load_state_dict(shared_weights, strict=False)

    logger.disable("src.neuralnet.attention")
    try:
        with torch.inference_mode():
            rms_output = rms_attention(x)
            no_rms_output = no_rms_attention(x)

        difference = rms_output - no_rms_output
        rms_error = difference.square().mean().sqrt().item()
        mean_absolute_error = difference.abs().mean().item()
        relative_l2_error = difference.norm().div(rms_output.norm().clamp_min(1e-12)).item()
        cosine_similarity = torch.nn.functional.cosine_similarity(
            rms_output.flatten(), no_rms_output.flatten(), dim=0
        ).item()

        with torch.inference_mode():
            rms_latency_ms = _latency_ms(rms_attention, x, device)
            no_rms_latency_ms = _latency_ms(no_rms_attention, x, device)
    finally:
        logger.enable("src.neuralnet.attention")

    latency_delta_percent = (
        (rms_latency_ms - no_rms_latency_ms) / no_rms_latency_ms * 100
    )
    logger.info(
        "rms_norm quality drift | device={} rmse={} mae={} relative_l2={} cosine_similarity={}",
        device,
        rms_error,
        mean_absolute_error,
        relative_l2_error,
        cosine_similarity,
    )
    logger.info(
        "rms_norm latency | device={} true_ms={} false_ms={} delta_percent={}",
        device,
        rms_latency_ms,
        no_rms_latency_ms,
        latency_delta_percent,
    )

    assert torch.isfinite(rms_output).all()
    assert torch.isfinite(no_rms_output).all()
    assert rms_output.shape == no_rms_output.shape == x.shape
