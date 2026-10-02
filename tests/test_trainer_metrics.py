import lightning.pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.configs.model_config import ModelConfig
from src.trainer import (
    IGNORE_INDEX,
    GenerationConfig,
    QwenTrainer,
    generate,
    perplexity,
    score_generation,
)
from tests.test_data_training import FakeTokenizer, make_config


def make_model(config: ModelConfig) -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Embedding(config.vocab_size, config.d_model),
        torch.nn.Linear(config.d_model, config.vocab_size),
    )


def make_tokenizer(config: ModelConfig) -> FakeTokenizer:
    return FakeTokenizer(vocab_size=config.vocab_size)


class AlwaysEosModel(torch.nn.Module):
    """Model stub whose argmax is always `eos_token_id`."""

    def __init__(self, vocab_size: int, eos_token_id: int):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(vocab_size))
        self.eos_token_id = eos_token_id

    def forward(self, x):
        logits = self.bias.expand(*x.shape, -1).clone()
        logits[..., self.eos_token_id] = 10.0
        return logits


def fit_module(module: QwenTrainer, max_steps: int = 4, val_check_interval: int = 2):
    data = TensorDataset(
        torch.randint(1, 16, (8, 4)), torch.randint(1, 16, (8, 4))
    )
    loader = DataLoader(data, batch_size=2)
    trainer = pl.Trainer(
        max_steps=max_steps,
        max_epochs=-1,
        accelerator="cpu",
        devices=1,
        val_check_interval=val_check_interval,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    return trainer


def test_perplexity_matches_exp_of_mean_cross_entropy():
    config = make_config()
    logits = torch.randn(2, 4, config.vocab_size)
    targets = torch.randint(0, config.vocab_size, (2, 4))

    expected = torch.exp(
        torch.nn.functional.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
    )

    assert perplexity(logits, targets, ignore_index=IGNORE_INDEX) == pytest.approx(expected.item(), rel=1e-5)


def test_perplexity_ignores_masked_targets():
    config = make_config()
    logits = torch.randn(1, 6, config.vocab_size)
    targets = torch.randint(0, config.vocab_size, (1, 6))
    targets[0, 3:] = IGNORE_INDEX

    full = perplexity(logits, targets, ignore_index=IGNORE_INDEX)
    partial = perplexity(logits[:, :3], targets[:, :3], ignore_index=IGNORE_INDEX)

    assert full == pytest.approx(partial.item(), rel=1e-5)


def test_training_and_validation_log_perplexity():
    config = make_config(max_steps=2)
    trainer = QwenTrainer(make_model(config), config, max_steps=2)
    lightning = fit_module(trainer, max_steps=2)

    assert "train_perplexity_step" in lightning.logged_metrics
    assert "val_perplexity" in lightning.logged_metrics
    assert "val_loss" in lightning.logged_metrics


def test_generate_returns_prompt_plus_new_tokens():
    config = make_config(max_seq_len=32)
    tokenizer = make_tokenizer(config)

    ids, prompt_len = generate(
        make_model(config), tokenizer, "abc", max_new_tokens=5, max_seq_len=32
    )

    assert ids.shape == (1, 8)
    assert prompt_len == 3
    assert ids[0, :prompt_len].tolist() == tokenizer.encode("abc")


def test_generate_stops_at_eos():
    config = make_config(max_seq_len=32)
    eos_id = 7
    tokenizer = FakeTokenizer(vocab_size=config.vocab_size, eos_token_id=eos_id)

    ids, _ = generate(
        AlwaysEosModel(config.vocab_size, eos_id),
        tokenizer,
        "abc",
        max_new_tokens=20,
        greedy=True,
        max_seq_len=32,
    )

    assert ids.shape == (1, 4)
    assert ids[0, -1].item() == eos_id


def test_generate_never_exceeds_max_seq_len():
    config = make_config(max_seq_len=4)

    ids, _ = generate(
        make_model(config),
        make_tokenizer(config),
        "abcdefgh",
        max_new_tokens=50,
        max_seq_len=4,
    )

    assert ids.shape[1] <= 4


def test_generate_truncates_oversized_prompts():
    config = make_config(max_seq_len=4)
    tokenizer = make_tokenizer(config)

    ids, prompt_len = generate(
        make_model(config), tokenizer, "abcdefgh", max_new_tokens=50, max_seq_len=4
    )

    assert prompt_len == 3
    assert ids[0, :prompt_len].tolist() == tokenizer.encode("abcdefgh")[-3:]


def test_greedy_generation_is_deterministic():
    config = make_config(max_seq_len=32)
    model = make_model(config)
    tokenizer = make_tokenizer(config)

    first, _ = generate(model, tokenizer, "abc", max_new_tokens=6, greedy=True)
    second, _ = generate(model, tokenizer, "abc", max_new_tokens=6, greedy=True)

    assert torch.equal(first, second)


def test_top_k_restricts_the_sampled_support():
    config = make_config(max_seq_len=64)
    model = make_model(config)
    tokenizer = make_tokenizer(config)
    prompt = tokenizer.encode("abc")
    allowed = torch.topk(model(torch.tensor([prompt]))[0, -1], 1).indices.tolist()

    torch.manual_seed(0)
    sampled, prompt_len = generate(
        model, tokenizer, "abc", max_new_tokens=1, top_k=1, top_p=1.0, max_seq_len=64
    )

    assert sampled[0, prompt_len:].tolist() == allowed


def test_score_generation_reports_tail_only_metrics():
    config = make_config()
    model = make_model(config)
    input_ids = torch.tensor([[5, 6, 7, 8, 9]], dtype=torch.long)
    prompt_len = 3

    metrics = score_generation(model, input_ids, prompt_len)

    with torch.no_grad():
        logits = model(input_ids).float()
        step_logits = logits[:, prompt_len - 1:-1].reshape(-1, config.vocab_size)
        targets = input_ids[:, prompt_len:].reshape(-1)
        expected_loss = torch.nn.functional.cross_entropy(step_logits, targets)
        expected_acc = (step_logits.argmax(dim=-1) == targets).float().mean()

    assert metrics["loss"] == pytest.approx(expected_loss.item(), rel=1e-5)
    assert metrics["accuracy"] == pytest.approx(expected_acc.item(), rel=1e-5)
    assert metrics["perplexity"] == pytest.approx(expected_loss.exp().item(), rel=1e-4)


def test_score_generation_handles_empty_continuation():
    config = make_config()
    metrics = score_generation(make_model(config), torch.tensor([[5, 6]]), 2)

    assert metrics["loss"] != metrics["loss"]  # NaN


def test_validation_logs_generation_metrics_every_inference_interval():
    config = make_config(max_steps=4)
    module = QwenTrainer(
        make_model(config),
        config,
        max_steps=4,
        tokenizer=make_tokenizer(config),
        generation_config=GenerationConfig(prompts=("ab", "cd"), max_new_tokens=4),
        inference_every=2,
    )

    lightning = fit_module(module, max_steps=4, val_check_interval=2)

    assert "val_generation_perplexity" in lightning.logged_metrics
    assert "val_generation_accuracy" in lightning.logged_metrics
    assert "val_generation_loss" in lightning.logged_metrics


def test_generation_metrics_skipped_on_off_interval_steps():
    config = make_config(max_steps=2)
    module = QwenTrainer(
        make_model(config),
        config,
        max_steps=2,
        tokenizer=make_tokenizer(config),
        generation_config=GenerationConfig(prompts=("ab",), max_new_tokens=4),
        inference_every=1000,
    )

    lightning = fit_module(module, max_steps=2, val_check_interval=1)

    assert "val_generation_perplexity" not in lightning.logged_metrics


def test_generation_metrics_require_a_tokenizer():
    config = make_config(max_steps=2)
    module = QwenTrainer(make_model(config), config, max_steps=2, inference_every=1)

    lightning = fit_module(module, max_steps=2, val_check_interval=1)

    assert "val_generation_perplexity" not in lightning.logged_metrics


def test_run_inference_restores_training_mode():
    config = make_config()
    module = QwenTrainer(
        make_model(config),
        config,
        tokenizer=make_tokenizer(config),
        generation_config=GenerationConfig(prompts=("ab", "cd"), max_new_tokens=3),
    )
    module.train()

    results = module._run_inference(step=5)

    assert module.training
    assert len(results) == 2
    for result in results:
        assert {"prompt", "text", "loss", "perplexity", "accuracy"} <= set(result)