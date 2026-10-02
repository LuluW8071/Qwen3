from dataclasses import fields
from pathlib import Path

import pytest
import torch
import yaml

import main
from src.configs.model_config import ModelConfig
from src.data import dataloader
from src.data.dataset import TextTokenDataset
from src.optim.hybridoptim import HybridOptimizer
from src.optim.muon import Muon
from src.trainer import QwenTrainer


class FakeTokenizer:
    """Character-level stand-in for the HF tokenizer cached by the datamodule.

    `eos_token_id` defaults to None so random sampling in tests does not stop
    early; pass an id explicitly when testing eos handling.
    """

    def __init__(self, vocab_size: int = 40, eos_token_id: int | None = None):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id

    def encode(self, text, add_special_tokens=False):
        return [(ord(char) % (self.vocab_size - 1)) + 1 for char in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr((int(i) % 26) + 97) for i in ids)


def make_config(**overrides) -> ModelConfig:
    values = {
        "d_model": 8,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 16,
        "n_kv_heads": 1,
        "sliding_window": 16,
        "attention_bias": False,
        "rms_norm_eps": 1e-6,
        "rms_norm": True,
        "max_seq_len": 4,
        "num_documents": 2,
        "max_tokens": 32,
        "batch_size": 2,
        "max_steps": 2,
        "gradient_accumulation_steps": 1,
        "muon_lr": 0.01,
        "eval_every": 1,
        "eval_steps": 1,
        "weight_decay": 0.1,
        "dropout": 0.0,
        "grad_clip": 1.0,
        "use_amp": False,
        "device": "cpu",
        "vocab_size": 16,
    }
    values.update(overrides)
    return ModelConfig(**values)


def make_model(config: ModelConfig) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Embedding(config.vocab_size, config.d_model),
        torch.nn.Linear(config.d_model, config.vocab_size),
    )


def test_text_token_dataset_shifts_targets_by_one():
    dataset = TextTokenDataset([1, 2, 3, 4, 5], seq_len=3)

    assert len(dataset) == 2
    inputs, targets = dataset[1]
    assert inputs.tolist() == [2, 3, 4]
    assert targets.tolist() == [3, 4, 5]


def test_model_config_coerces_rms_norm_eps_string():
    config = make_config(rms_norm_eps="1e-6")

    assert config.rms_norm_eps == 1e-6
    assert isinstance(config.rms_norm_eps, float)


def test_model_config_derives_head_dims():
    config = make_config(d_model=64, n_heads=8, n_kv_heads=4)

    assert config.d_k == 8
    assert config.n_kv_groups == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"d_model": 10, "n_heads": 4, "n_kv_heads": 2},
        {"d_model": 8, "n_heads": 3, "n_kv_heads": 1},
    ],
)
def test_model_config_rejects_incompatible_head_counts(overrides):
    with pytest.raises(ValueError):
        make_config(**overrides)


def test_data_module_splits_cached_tokens_deterministically(monkeypatch):
    config = make_config(max_seq_len=4)
    monkeypatch.setattr(
        dataloader,
        "load_and_cache_data",
        lambda config, cache_dir: ([], FakeTokenizer(), list(range(24))),
    )

    first = dataloader.QwenDataModule(config)
    second = dataloader.QwenDataModule(config)
    first.setup("fit")
    second.setup("fit")

    assert len(first.train_dataset) == len(second.train_dataset)
    assert len(first.val_dataset) == len(second.val_dataset)
    assert first.train_dataset.indices == second.train_dataset.indices


def test_data_module_exposes_tokenizer_for_generation(monkeypatch):
    tokenizer = FakeTokenizer()
    config = make_config(max_seq_len=4)
    monkeypatch.setattr(
        dataloader,
        "load_and_cache_data",
        lambda config, cache_dir: ([], tokenizer, list(range(24))),
    )

    data = dataloader.QwenDataModule(config)
    data.setup("fit")

    assert data.tokenizer is tokenizer


def test_data_module_rejects_tiny_validation_split():
    config = make_config(max_seq_len=4)

    with pytest.raises(ValueError):
        dataloader.QwenDataModule(config, validation_fraction=1.5)


def test_trainer_builds_muon_and_adamw_optimizers():
    config = make_config()
    trainer = QwenTrainer(make_model(config), config)

    settings = trainer.configure_optimizers()
    optimizer = settings["optimizer"]

    assert isinstance(optimizer, HybridOptimizer)
    muon, adam = optimizer.optimizers
    assert isinstance(muon, Muon)
    assert isinstance(adam, torch.optim.AdamW)
    assert settings["lr_scheduler"]["interval"] == "step"


def test_trainer_adamw_lr_is_a_tenth_of_muon_lr():
    config = make_config()
    trainer = QwenTrainer(make_model(config), config, muon_lr=0.05, weight_decay=0.2)

    muon, adam = trainer._build_optimizer().optimizers

    assert muon.param_groups[0]["lr"] == 0.05
    assert adam.param_groups[0]["lr"] == 0.05 * 0.1
    assert adam.param_groups[0]["weight_decay"] == 0.2


def test_trainer_uses_adamw_without_8bit_optimizer():
    config = make_config(device="cuda")
    trainer = QwenTrainer(make_model(config), config, use_8bit_optimizer=False)

    optimizers = trainer.configure_optimizers()["optimizer"].optimizers

    assert optimizers[1].__class__.__name__ == "AdamW"


def test_trainer_sends_only_hidden_weights_to_muon():
    config = make_config()
    trainer = QwenTrainer(make_model(config), config)

    muon, adam = trainer.configure_optimizers()["optimizer"].optimizers
    muon_ids = {id(p) for group in muon.param_groups for p in group["params"]}
    adam_ids = {id(p) for group in adam.param_groups for p in group["params"]}

    assert muon_ids and adam_ids
    assert not muon_ids & adam_ids


def test_trainer_falls_back_to_config_vocab_size():
    trainer = QwenTrainer(make_model(make_config()), make_config())

    assert trainer.vocab_size == 16


def test_trainer_requires_a_vocab_size():
    model = make_model(make_config())

    with pytest.raises(ValueError):
        QwenTrainer(model, make_config(vocab_size=None))


def test_arg_parser_only_exposes_launch_flags():
    """Hyperparameters stay in the yaml; the CLI keeps per-launch switches only."""
    dests = {action.dest for action in main.build_parser()._actions}
    yaml_fields = {field.name for field in fields(ModelConfig)}

    assert dests & yaml_fields == {"device"}
    assert {"config", "gpus", "dist_backend", "precision", "checkpoint_dir",
            "resume_checkpoint", "cache_dir", "num_workers",
            "inference_every"} <= dests


@pytest.mark.parametrize(
    "config_name", ["example.yaml", "param.yaml"]
)
def test_shipped_configs_define_every_field(config_name):
    path = Path(__file__).parents[1] / "src/configs" / config_name

    with path.open() as file:
        raw = yaml.safe_load(file)

    assert set(raw) == {field.name for field in fields(ModelConfig)}
    ModelConfig(**raw)


def test_cli_overrides_device_only():
    config = make_config(device="cuda")
    args = main.parse_args(["--device", "cpu"])

    main.apply_overrides(config, args)

    assert config.device == "cpu"
    assert config.max_steps == 2 and config.muon_lr == 0.01


def test_cli_leaves_config_untouched_when_flags_absent():
    config = make_config()
    before = vars(config).copy()

    main.apply_overrides(config, main.parse_args([]))

    assert vars(config) == before


def test_device_resolution_rejects_gpu_without_cuda(monkeypatch):
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError):
        main.resolve_device(make_config(device="cuda"), main.parse_args([]))


def test_device_resolution_normalizes_to_cpu():
    config = make_config(device="cpu")

    assert main.resolve_device(config, main.parse_args([])) == "cpu"
    assert config.device == "cpu"


def test_device_resolution_rejects_multi_gpu_on_cpu():
    args = main.parse_args(["--gpus", "2"])

    with pytest.raises(ValueError):
        main.resolve_device(make_config(device="cpu"), args)


def test_precision_defaults_to_bf16_with_amp_on_cuda(monkeypatch):
    config = make_config(device="cuda", use_amp=True)
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(main.torch.cuda, "device_count", lambda: 8)

    precision, use_8bit = main.resolve_precision(config, main.parse_args([]))

    assert precision == "bf16-mixed"
    assert use_8bit is False


def test_precision_falls_back_to_fp32_without_amp(monkeypatch):
    config = make_config(device="cuda", use_amp=False)
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: True)

    precision, _ = main.resolve_precision(config, main.parse_args([]))

    assert precision == "32-true"


def test_precision_rejects_fp16_on_cpu():
    config = make_config(device="cpu")
    args = main.parse_args(["--precision", "fp16-mixed"])

    with pytest.raises(ValueError):
        main.resolve_precision(config, args)


def test_precision_rejects_8bit_on_cpu():
    config = make_config(device="cpu")
    args = main.parse_args(["--precision", "8bit"])

    with pytest.raises(ValueError):
        main.resolve_precision(config, args)


def test_checkpoint_filename_renders_step_and_metric():
    template = "qwen3-step{step:06d}-val_loss{val_loss:.4f}"

    assert template.format(step=1, val_loss=0.5) == "qwen3-step000001-val_loss0.5000"