from __future__ import annotations

import argparse
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint

from src.configs.model_config import ModelConfig
from src.data.dataloader import QwenDataModule
from src.neuralnet.qwen import Qwen3LLM
from src.trainer import QwenTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3 from scratch")

    # Configuration and device
    parser.add_argument("-c", "--config", default=Path("src/configs/example.yaml"), type=Path,
                        help="path to model configuration")
    parser.add_argument("-g", "--gpus", default=1, type=int,
                        help="number of GPUs to use")
    parser.add_argument("-db", "--dist_backend", default="deepspeed_stage_2", type=str,
                        help="distributed backend for multi-GPU training")
    parser.add_argument("-d", "--device", default=None, type=str,
                        help="training device: cpu or cuda")
    parser.add_argument("--precision", default="8bit", type=str,
                        help="training precision: 8bit or fp16")
    parser.add_argument("--checkpoint_dir", "--checkpoint-dir", default="checkpoints", type=Path,
                        help="directory for checkpoints")
    parser.add_argument("--resume_checkpoint", "--resume-checkpoint", default=None, type=Path,
                        help="checkpoint path to resume")

    # Training hyperparameters
    parser.add_argument("--max_steps", "--max-steps", default=None, type=int,
                        help="maximum number of optimizer steps")
    parser.add_argument("--batch_size", "--batch-size", default=None, type=int,
                        help="training batch size")
    parser.add_argument("--max_seq_len", "--max-seq-len", default=None, type=int,
                        help="maximum sequence length")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpus < 1:
        raise ValueError("--gpus must be at least 1")

    with args.config.open() as file:
        config = ModelConfig(**yaml.safe_load(file))
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_seq_len is not None:
        config.max_seq_len = args.max_seq_len
    if args.device is not None:
        config.device = args.device

    requested_device = str(config.device).lower()
    use_cuda = requested_device.startswith("cuda") or requested_device == "gpu"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("Config requests CUDA, but no CUDA device is available")
    if use_cuda and args.gpus > torch.cuda.device_count():
        raise RuntimeError(
            f"Requested {args.gpus} GPUs, but only {torch.cuda.device_count()} are available"
        )
    if not use_cuda and args.gpus != 1:
        raise ValueError("--gpus > 1 requires CUDA")
    if args.precision == "fp16" and not use_cuda:
        raise ValueError("--precision fp16 requires --device cuda")
    config.device = "cuda" if use_cuda else "cpu"

    data = QwenDataModule(config, num_workers=0)
    data.setup("fit")
    model = Qwen3LLM(config)
    lightning_module = QwenTrainer(
        model,
        config,
        num_gpus=args.gpus if use_cuda else 1,
        use_8bit_optimizer=args.precision == "8bit",
    )

    plugins = []
    trainer_precision = "32-true"
    if use_cuda and args.precision == "8bit":
        from lightning.pytorch.plugins import BitsandbytesPrecision

        plugins.append(
            BitsandbytesPrecision(
                mode="int8-training",
                dtype=torch.float16,
                ignore_modules={"lm_head"},
            )
        )
    elif args.precision == "fp16":
        trainer_precision = "16-mixed"
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="qwen3-step{step:06d}-val_loss{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        accelerator="gpu" if use_cuda else "cpu",
        devices=args.gpus if use_cuda else 1,
        strategy=args.dist_backend if use_cuda and args.gpus > 1 else "auto",
        max_epochs=-1,
        max_steps=config.max_steps,
        accumulate_grad_batches=1,
        limit_val_batches=config.eval_steps,
        check_val_every_n_epoch=None,
        val_check_interval=config.eval_every,
        plugins=plugins,
        precision=trainer_precision,
        callbacks=[checkpoint_callback],
        logger=False,
        enable_checkpointing=True,
    )
    trainer.fit(
        lightning_module,
        datamodule=data,
        ckpt_path=args.resume_checkpoint,
    )


if __name__ == "__main__":
    main()
