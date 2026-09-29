import torch
from torch import nn

from src.configs.model_config import ModelConfig
from src.data import dataloader
from src.data.dataset import TextTokenDataset
from src.trainer import QwenTrainer


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


def test_data_module_splits_cached_tokens_deterministically(monkeypatch):
    config = make_config(max_seq_len=4)
    monkeypatch.setattr(
        dataloader,
        "load_and_cache_data",
        lambda config, cache_dir: ([], None, list(range(24))),
    )

    first = dataloader.QwenDataModule(config)
    second = dataloader.QwenDataModule(config)
    first.setup("fit")
    second.setup("fit")

    assert len(first.train_dataset) == len(second.train_dataset)
    assert len(first.val_dataset) == len(second.val_dataset)
    assert first.train_dataset.indices == second.train_dataset.indices


def test_trainer_builds_muon_and_8bit_optimizers():
    config = make_config()
    model = nn.Sequential(nn.Embedding(config.vocab_size, config.d_model), nn.Linear(8, 16))
    trainer = QwenTrainer(model, config)

    optimizers, _ = trainer.configure_optimizers()

    assert len(optimizers) == 2
    assert optimizers[0].__class__.__name__ == "Muon"
    expected_optimizer = (
        "Adam8bit"
        if config.device == "cuda" and torch.cuda.is_available()
        else "AdamW"
    )
    assert optimizers[1].__class__.__name__ == expected_optimizer


def test_trainer_uses_adamw_without_8bit_optimizer():
    config = make_config(device="cuda")
    model = nn.Sequential(nn.Embedding(config.vocab_size, config.d_model), nn.Linear(8, 16))
    trainer = QwenTrainer(model, config, use_8bit_optimizer=False)

    optimizers, _ = trainer.configure_optimizers()

    assert optimizers[1].__class__.__name__ == "AdamW"
