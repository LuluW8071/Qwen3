from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl
from torchmetrics.functional.text import perplexity

from src.configs.logging import logger
from src.configs.model_config import ModelConfig
from src.optim.muon import Muon
from src.optim.hybridoptim import HybridOptimizer

IGNORE_INDEX = -100

DEFAULT_INFERENCE_PROMPTS = (
    "Once upon a time, in a small village surrounded by mountains",
    "In a distant galaxy, far beyond the reach of human civilization",
    "In the beginning, there was nothing but darkness and silence",
    "Artificial intelligence is transforming the way people work, learn, and communicate",
    "The future of technology depends on how we understand and use it",
)


@dataclass
class GenerationConfig:
    """Sampling knobs for the validation-time inference engine."""

    prompts: tuple[str, ...] = DEFAULT_INFERENCE_PROMPTS
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    greedy: bool = False


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    greedy: bool = False,
    device: torch.device | None = None,
    max_seq_len: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Sample a continuation for `prompt`.

    Returns the full [1, T] token ids plus how many leading tokens are prompt,
    so callers can score only the sampled part. Prompts longer than `max_seq_len`
    are left-truncated to leave room for at least one new token.
    """
    if device is None:
        device = next(model.parameters()).device

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False) or [
        tokenizer.eos_token_id
    ]
    generated = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    eos_id = tokenizer.eos_token_id

    budget = max_new_tokens
    if max_seq_len is not None:
        generated = generated[:, -(max_seq_len - 1):]
        budget = min(budget, max_seq_len - generated.size(1))
    prompt_len = generated.size(1)

    for _ in range(budget):
        logits = model(generated)[0, -1, :].float()
        logits = logits / max(temperature, 1e-5)

        if top_k and top_k > 0:
            k = min(top_k, logits.size(-1))
            threshold = torch.topk(logits, k).values[-1]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            # Drop a token once the mass before it already exceeds top_p, so the
            # token that crosses the threshold always survives.
            drop = torch.cumsum(sorted_probs, dim=-1) - sorted_probs > top_p
            sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
            logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)

        if greedy:
            next_token = logits.argmax(dim=-1)
        else:
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)

        generated = torch.cat([generated, next_token.view(1, 1)], dim=1)
        if eos_id is not None and int(next_token) == eos_id:
            break

    return generated, prompt_len


@torch.no_grad()
def score_generation(
    model: nn.Module, input_ids: torch.Tensor, prompt_len: int
) -> dict[str, float]:
    """Teacher-force `input_ids`, scoring only the generated tail (>= prompt_len)."""
    logits = model(input_ids).float()
    tail_targets = input_ids[:, prompt_len:]
    if tail_targets.numel() == 0:
        return {"loss": float("nan"), "perplexity": float("nan"), "accuracy": float("nan")}

    step_logits = logits[:, prompt_len - 1:-1]
    loss = F.cross_entropy(step_logits.reshape(-1, logits.size(-1)), tail_targets.reshape(-1))
    accuracy = (step_logits.argmax(dim=-1) == tail_targets).float().mean()
    return {
        "loss": loss.item(),
        "perplexity": float(perplexity(step_logits, tail_targets, ignore_index=IGNORE_INDEX)),
        "accuracy": accuracy.item(),
    }


class QwenTrainer(pl.LightningModule):
    """Train/Validates the Qwen3 LLM with Muon + AdamW hybrid optimizer."""

    def __init__(
        self, 
        model: nn.Module, 
        config: ModelConfig, 
        muon_lr: float = 1e-2,
        weight_decay: float = 0.1,
        batch_size: int = 8,
        num_gpus: int = 1,
        max_steps: int = 10000,
        vocab_size: int = None,
        tokenizer=None,
        generation_config: GenerationConfig | None = None,
        inference_every: int = 100,
        use_8bit_optimizer: bool | None = None
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.muon_lr = muon_lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.vocab_size = vocab_size or config.vocab_size
        self.tokenizer = tokenizer
        self.generation_config = generation_config or GenerationConfig()
        self.inference_every = inference_every
        self.use_8bit_optimizer = use_8bit_optimizer
        if self.vocab_size is None:
            raise ValueError("vocab_size must be set on the config or passed in")
        if not self.generation_config.prompts:
            logger.warning("No inference prompts configured; generation logging is off")

        self.save_hyperparameters(ignore=["model", "config", "tokenizer"])

        # Only sync metrics across processes when training on multiple GPUs.
        self.sync_dist = num_gpus > 1

    def forward(self, x):
        return self.model(x)

    def _build_optimizer(self) -> HybridOptimizer:
        muon_params, adamw_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 2 and "token_embedding" not in name and "norm" not in name:
                muon_params.append(p)
            else:
                adamw_params.append(p)

        logger.info("Muon params: {}", sum(p.numel() for p in muon_params))
        logger.info("AdamW params: {}", sum(p.numel() for p in adamw_params))

        muon = Muon(muon_params, lr=self.muon_lr, momentum=0.95)

        adam_kwargs = {
            "lr": self.muon_lr * 0.1,
            "betas": (0.9, 0.975),
            "eps": 1e-8,
            "weight_decay": self.weight_decay,
        }
        if self.use_8bit_optimizer:
            if not torch.cuda.is_available():
                raise RuntimeError("8-bit optimizer requires CUDA")
            import bitsandbytes as bnb

            adam = bnb.optim.Adam8bit(adamw_params, **adam_kwargs)
        else:
            adam = optim.AdamW(adamw_params, **adam_kwargs)

        return HybridOptimizer(muon, adam)

    def configure_optimizers(self):
        optimizer = self._build_optimizer()
        max_steps = self.max_steps
        warmup = max(1, max_steps // 20)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / warmup
            progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size), y.reshape(-1))
        ppl = perplexity(logits.detach().float(), y, ignore_index=IGNORE_INDEX)
        acc = (logits.argmax(dim=-1) == y).float().mean()

        is_train = stage == "train"
        self.log(f"{stage}_loss", loss, prog_bar=is_train, on_step=is_train, on_epoch=True,
                 sync_dist=True, batch_size=x.size(0))
        self.log(f"{stage}_perplexity", ppl, prog_bar=is_train, on_step=is_train, on_epoch=True,
                 sync_dist=True, batch_size=x.size(0))
        self.log(f"{stage}_accuracy", acc, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=x.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def _run_inference(self, step: int) -> list[dict[str, float | str]]:
        """Sample from every prompt and score the generated text under the model."""
        cfg = self.generation_config
        was_training = self.training
        self.eval()
        results: list[dict[str, float | str]] = []
        try:
            for prompt in cfg.prompts:
                input_ids, prompt_len = generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    top_p=cfg.top_p,
                    greedy=cfg.greedy,
                    device=self.device,
                    max_seq_len=self.config.max_seq_len,
                )
                metrics = score_generation(self.model, input_ids, prompt_len)
                text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                logger.info(
                    "step {} | acc={:.4f} loss={:.4f} ppl={:.2f} | prompt={!r} -> {!r}",
                    step,
                    metrics["accuracy"],
                    metrics["loss"],
                    metrics["perplexity"],
                    prompt,
                    text,
                )
                self._log_generation_to_comet(step, prompt, text, metrics)
                results.append({"prompt": prompt, "text": text, **metrics})
        finally:
            if was_training:
                self.train()
        return results

    def _log_generation_to_comet(
        self, step: int, prompt: str, text: str, metrics: dict[str, float]
    ) -> None:
        """Send generated text to Comet when the active Lightning logger supports it."""
        trainer = getattr(self, "_trainer", None)
        if trainer is None or not trainer.is_global_zero:
            return

        experiment = getattr(getattr(trainer, "logger", None), "experiment", None)
        log_text = getattr(experiment, "log_text", None)
        if log_text is None:
            return

        log_text(
            text=text,
            step=step,
            metadata={"prompt": prompt, **metrics},
        )

    def on_validation_epoch_end(self) -> None:
        """Log sample generations (with acc/ppl) every `inference_every` steps."""
        if self.tokenizer is None or self.inference_every <= 0:
            return
        if not self.generation_config.prompts:
            return
        if self.trainer.sanity_checking:
            return
        if self.global_step % self.inference_every != 0:
            return

        results = self._run_inference(self.global_step)
        count = len(results)

        def mean(key: str) -> float:
            return sum(float(r[key]) for r in results) / count

        self.log("val_generation_perplexity", mean("perplexity"), prog_bar=True,
                 sync_dist=self.sync_dist, batch_size=count)
        self.log("val_generation_accuracy", mean("accuracy"), sync_dist=self.sync_dist,
                 batch_size=count)
        self.log("val_generation_loss", mean("loss"), sync_dist=self.sync_dist,
                 batch_size=count)
