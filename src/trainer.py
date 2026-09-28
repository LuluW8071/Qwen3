from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl

from src.configs.logging import logger
from src.configs.model_config import ModelConfig
from src.optim.muon import Muon


class QwenTrainer(pl.LightningModule):
    """Train/Validates the Qwen3 LLM with Muon Optimizer Loss Fn"""

    def __init__(self, model: nn.Module, config: ModelConfig, num_gpus: int = 1):
        super().__init__()

        self.model = model
        self.config = config
        self.automatic_optimization = False
        self._accumulated_batches = 0

        self.save_hyperparameters(ignore=["model", "config"])
        self.sync_dist = num_gpus > 1


    def forward(self, x):
        return self.model(x)

    def setup_optimizers(self, config: ModelConfig):
        """Setup Hybrid Muon Optimizer with AdamW Approach"""
        muon_params, adamW_params = [], []

        for name, param in self.named_parameters():
            if (
                param.ndim == 2 and
                'token_embedding' not in name and
                'norm' not in name and
                param.requires_grad
            ):
                muon_params.append(param)
            else:
                adamW_params.append(param)

        logger.info("Muon Params: {}", sum(p.numel() for p in muon_params))
        logger.info("AdamW parameters: {}", sum(p.numel() for p in adamW_params))

        muon_optimizer = Muon(
            muon_params,
            lr=config.muon_lr,
            momentum=0.95
        )
        if config.device == "cuda" and torch.cuda.is_available():
            try:
                import bitsandbytes as bnb
            except ImportError as exc:
                raise ImportError(
                    "8-bit optimizer requires `bitsandbytes`."
                ) from exc
            adamW_optimizer = bnb.optim.Adam8bit(
                adamW_params,
                lr=config.muon_lr * 0.1,
                weight_decay=config.weight_decay,
            )
        else:
            adamW_optimizer = optim.AdamW(
                adamW_params,
                lr=config.muon_lr * 0.1,
                weight_decay=config.weight_decay,
            )

        return [muon_optimizer, adamW_optimizer]

    def configure_optimizers(self):
        optimizers = self.setup_optimizers(self.config)
        warmup_steps = max(1, self.config.max_steps // 20)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = min(
                1.0,
                (step - warmup_steps) / max(1, self.config.max_steps - warmup_steps),
            )
            return 0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(torch.pi * progress))).item()

        schedulers = [
            {"scheduler": optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), "interval": "step"}
            for optimizer in optimizers
        ]
        return optimizers, schedulers

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), y.reshape(-1)
        )
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == y).float().mean()
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=stage == "train",
            on_step=stage == "train",
            on_epoch=True,
            sync_dist=self.sync_dist,
            batch_size=x.size(0),
        )
        self.log(
            f"{stage}_accuracy",
            accuracy,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=self.sync_dist,
            batch_size=x.size(0),
        )
        return loss


    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "train")
        optimizer_list = self.optimizers()
        optimizers = list(optimizer_list) if isinstance(optimizer_list, (list, tuple)) else [optimizer_list]
        loss = loss / self.config.gradient_accumulation_steps
        self.manual_backward(loss)
        self._accumulated_batches += 1

        if self._accumulated_batches >= self.config.gradient_accumulation_steps:
            for optimizer in optimizers:
                self.clip_gradients(
                    optimizer,
                    gradient_clip_val=self.config.grad_clip,
                    gradient_clip_algorithm="norm",
                )
                optimizer.step()
                optimizer.zero_grad()
            schedulers = self.lr_schedulers()
            schedulers = list(schedulers) if isinstance(schedulers, (list, tuple)) else [schedulers]
            for scheduler in schedulers:
                scheduler.step()
            self._accumulated_batches = 0
        return loss.detach()


    def validation_step(self, batch, batch_idx):
        if batch_idx >= self.config.eval_steps:
            return None
        return self._shared_step(batch, "val")
