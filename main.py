from __future__ import annotations

import argparse
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.plugins import BitsandbytesPrecision

from src.configs.model_config import ModelConfig
from src.data.dataloader import QwenDataModule
from src.neuralnet.qwen import Qwen3LLM
from src.trainer import QwenTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen3 from scratch")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/configs/example.yaml"),
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    args = parser.parse_args()

    with args.config.open() as file:
        config = ModelConfig(**yaml.safe_load(file))
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_seq_len is not None:
        config.max_seq_len = args.max_seq_len

    data = QwenDataModule(config, num_workers=0)
    data.setup("fit")
    model = Qwen3LLM(config)
    lightning_module = QwenTrainer(model, config)

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requests CUDA, but no CUDA device is available")
    use_cuda = config.device == "cuda"
    trainer = pl.Trainer(
        accelerator="gpu" if use_cuda else "cpu",
        devices=1,
        max_steps=config.max_steps,
        accumulate_grad_batches=1,
        limit_val_batches=config.eval_steps,
        check_val_every_n_epoch=None,
        val_check_interval=config.eval_every,
        plugins=BitsandbytesPrecision(
            mode="int8-training",
            dtype=torch.float16,
            ignore_modules={"lm_head"},
        ),
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(lightning_module, datamodule=data)


if __name__ == "__main__":
    main()
