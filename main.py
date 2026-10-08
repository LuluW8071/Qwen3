from __future__ import annotations

import argparse
import os
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint

from src.configs.model_config import ModelConfig
from src.data.dataloader import QwenDataModule
from src.neuralnet.qwen import Qwen3LLM
from src.trainer import QwenTrainer

# Model and training hyperparameters live in the yaml config; the CLI only
# carries what changes per launch. `device` is the one exception, because it is
# hardware rather than experiment definition.
CONFIG_OVERRIDES = {"device": str}

PRECISION_TO_LIGHTNING = {
    "32-true": "32-true",
    "bf16-mixed": "bf16-mixed",
    "fp16-mixed": "16-mixed",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Qwen3 from scratch. Model and training hyperparameters "
                    "come from the yaml config; this CLI only overrides per-launch "
                    "settings.")

    # Configuration and device
    parser.add_argument("-c", "--config", default=Path("src/configs/param.yaml"), type=Path,
                        help="path to model configuration")
    parser.add_argument("-g", "--gpus", default=1, type=int,
                        help="number of GPUs to use")
    parser.add_argument("-db", "--dist_backend", default="deepspeed_stage_2", type=str,
                        help="distributed backend for multi-GPU training")
    parser.add_argument("-d", "--device", default=None, type=str,
                        help="training device: cpu or cuda; overrides the yaml value")
    parser.add_argument("--precision", default=None,
                        choices=["32-true", "bf16-mixed", "fp16-mixed", "8bit"],
                        help="train precision; defaults to bf16-mixed when use_amp is on and CUDA is used")
    parser.add_argument("--compile", action="store_true",
                        help="compile the model with torch.compile")
    parser.add_argument("--checkpoint_dir", "--checkpoint-dir", default="checkpoints", type=Path,
                        help="directory for checkpoints")
    parser.add_argument("--resume_checkpoint", "--resume-checkpoint", default=None, type=Path,
                        help="checkpoint path to resume")

    # Data plumbing
    parser.add_argument("--cache_dir", "--cache-dir", default="data_cache", type=Path,
                        help="directory holding the tokenized dataset cache")
    parser.add_argument("--num_workers", "--num-workers", default=0, type=int,
                        help="dataloader workers")

    # Inference
    parser.add_argument("--inference_every", "--inference-every", default=100, type=int,
                        help="sample and score generations during validation every N steps")

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def load_config(path: Path) -> ModelConfig:
    with path.open() as file:
        return ModelConfig(**yaml.safe_load(file))


def apply_overrides(config: ModelConfig, args: argparse.Namespace) -> ModelConfig:
    for name in CONFIG_OVERRIDES:
        value = getattr(args, name, None)
        if value is not None:
            setattr(config, name, value)
    return config


def resolve_device(config: ModelConfig, args: argparse.Namespace) -> str:
    if args.gpus < 1:
        raise ValueError("--gpus must be at least 1")

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

    config.device = "cuda" if use_cuda else "cpu"
    return config.device


def resolve_precision(config: ModelConfig, args: argparse.Namespace) -> tuple[str | None, bool]:
    """Return the Lightning precision string and whether 8-bit weights are requested."""
    use_cuda = config.device == "cuda"
    precision = args.precision
    if precision is None:
        precision = "bf16-mixed" if use_cuda and config.use_amp else "32-true"

    if precision == "8bit":
        if not use_cuda:
            raise ValueError("--precision 8bit requires --device cuda")
        # BitsandbytesPrecision is itself a Lightning precision plugin; passing
        # an explicit precision value alongside it raises a configuration error.
        return None, True
    if precision == "fp16-mixed" and not use_cuda:
        raise ValueError("--precision fp16-mixed requires --device cuda")
    if precision == "bf16-mixed" and not use_cuda:
        # bf16 autocast only exists on CUDA; fall back instead of crashing.
        return "32-true", False
    return PRECISION_TO_LIGHTNING[precision], False


def build_logger():
    """Enable Comet only when an API key is configured in the environment."""
    api_key = os.getenv("COMET_API_KEY")
    if not api_key:
        return False

    from lightning.pytorch.loggers import CometLogger

    return CometLogger(
        api_key=api_key,
        project_name=os.getenv("COMET_PROJECT_NAME", "qwen3"),
        workspace=os.getenv("COMET_WORKSPACE") or None,
        experiment_key=os.getenv("COMET_EXPERIMENT_KEY") or None,
    )


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    resolve_device(config, args)
    trainer_precision, use_8bit = resolve_precision(config, args)

    data = QwenDataModule(config, cache_dir=args.cache_dir, num_workers=args.num_workers)
    data.setup("fit")
    if config.vocab_size != data.tokenizer.vocab_size:
        raise ValueError(
            f"--vocab_size {config.vocab_size} does not match tokenizer vocab "
            f"{data.tokenizer.vocab_size}"
        )
    model = Qwen3LLM(config)
    if args.compile:
        model = torch.compile(model)
    lightning_module = QwenTrainer(
        model,
        config,
        muon_lr=config.muon_lr,
        weight_decay=config.weight_decay,
        batch_size=config.batch_size,
        num_gpus=args.gpus if config.device == "cuda" else 1,
        max_steps=config.max_steps,
        vocab_size=config.vocab_size,
        use_8bit_optimizer=use_8bit,
        tokenizer=data.tokenizer,
        inference_every=args.inference_every,
    )

    plugins = []
    if use_8bit:
        from lightning.pytorch.plugins import BitsandbytesPrecision

        plugins.append(
            BitsandbytesPrecision(
                mode="int8-training",
                dtype=torch.float16,
                ignore_modules={"lm_head"},
            )
        )
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
        accelerator="gpu" if config.device == "cuda" else "cpu",
        devices=args.gpus if config.device == "cuda" else 1,
        strategy=args.dist_backend if config.device == "cuda" and args.gpus > 1 else "auto",
        max_epochs=-1,
        max_steps=config.max_steps,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        gradient_clip_val=config.grad_clip,
        gradient_clip_algorithm="norm",
        limit_val_batches=config.eval_steps,
        check_val_every_n_epoch=None,
        val_check_interval=config.eval_every,
        plugins=plugins,
        precision=trainer_precision,
        callbacks=[checkpoint_callback],
        logger=build_logger(),
        enable_checkpointing=True,
    )
    trainer.fit(
        lightning_module,
        datamodule=data,
        ckpt_path=args.resume_checkpoint,
    )


if __name__ == "__main__":
    main()
