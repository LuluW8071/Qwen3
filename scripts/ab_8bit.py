"""Short A/B + resume checks for the --optimizer flag. Not a training entrypoint.

  python scripts/ab_8bit.py run    --optimizer muon8bit --steps 200 --out run.json
  python scripts/ab_8bit.py resume --optimizer muon8bit
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configs.model_config import ModelConfig  # noqa: E402
from src.data.dataloader import QwenDataModule  # noqa: E402
from src.neuralnet.qwen import Qwen3LLM  # noqa: E402
from src.trainer import QwenTrainer  # noqa: E402


class Recorder(pl.Callback):
    def __init__(self):
        self.rows = []

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        m = trainer.callback_metrics
        self.rows.append({k: float(m[k]) for k in
                          ("train_loss_step", "grad_norm", "lr_muon", "lr_adamw",
                           "tokens_per_sec", "peak_mem_mb") if k in m})


def build(args, steps):
    config = ModelConfig(**yaml.safe_load(open(args.config)))
    config.device, config.batch_size, config.max_steps = "cuda", args.batch_size, steps
    if args.no_dropout:
        config.dropout = 0.0
    data = QwenDataModule(config, cache_dir=args.cache_dir)
    data.setup("fit")
    module = QwenTrainer(Qwen3LLM(config), config, muon_lr=config.muon_lr,
                         weight_decay=config.weight_decay, batch_size=config.batch_size,
                         max_steps=steps, vocab_size=config.vocab_size,
                         optimizer=args.optimizer)
    return config, data, module


def fit(args, steps, ckpt_in=None, ckpt_out=None, recorder=None):
    pl.seed_everything(args.seed, workers=True)
    config, data, module = build(args, steps)
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_steps=steps, max_epochs=-1,
                         gradient_clip_val=config.grad_clip, precision="bf16-mixed",
                         limit_val_batches=0, num_sanity_val_steps=0, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         callbacks=[recorder] if recorder else [])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.fit(module, datamodule=data, ckpt_path=ckpt_in)
    if ckpt_out:
        trainer.save_checkpoint(ckpt_out)
    return trainer, module, time.time() - t0


def cmd_run(args):
    rec = Recorder()
    _, _, secs = fit(args, args.steps, recorder=rec)
    out = {"optimizer": args.optimizer, "steps": args.steps, "seconds": secs,
           "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20, "rows": rec.rows}
    Path(args.out).write_text(json.dumps(out))
    losses = [r["train_loss_step"] for r in rec.rows]
    print(f"{args.optimizer}: first={losses[0]:.4f} last={losses[-1]:.4f} "
          f"peak={out['peak_mem_mb']:.0f}MB finite={all(map(torch.isfinite, map(torch.tensor, losses)))}")


def manual_loop(args, module, opt, sched, batches, start, stop):
    """Plain loop mirroring Lightning's step: bf16 autocast, clip once, step both, sched."""
    losses = []
    module.train()
    for x, y in batches[start:stop]:
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = module(x.cuda())
        loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, module.vocab_size), y.cuda().reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
    return losses


def cmd_resume(args):
    """Train N steps, save, reload into fresh objects, train K more; compare with an
    uninterrupted N+K run on the same fixed batches (dropout off, same seed)."""
    args.no_dropout = True
    n, k = args.steps, args.extra
    config, data, _ = build(args, n + k)
    g = torch.Generator().manual_seed(123)
    ds = data.train_dataset
    idx = torch.randperm(len(ds), generator=g)[: (n + k) * args.batch_size].view(n + k, -1)
    batches = [tuple(torch.stack(t) for t in zip(*[ds[int(i)] for i in row])) for row in idx]

    def fresh():
        pl.seed_everything(args.seed, workers=True)
        _, _, m = build(args, n + k)
        m = m.cuda()
        cfg = m.configure_optimizers()
        return m, cfg["optimizer"], cfg["lr_scheduler"]["scheduler"]

    m, opt, sch = fresh()
    full = manual_loop(args, m, opt, sch, batches, 0, n + k)

    del m, opt, sch
    torch.cuda.empty_cache()
    m, opt, sch = fresh()
    full2 = manual_loop(args, m, opt, sch, batches, 0, n + k)
    print("run-to-run (two uninterrupted runs) max abs diff:",
          max(abs(a - b) for a, b in zip(full[n:], full2[n:])))
    del m, opt, sch
    torch.cuda.empty_cache()
    m, opt, sch = fresh()
    manual_loop(args, m, opt, sch, batches, 0, n)
    ckpt = {"model": m.state_dict(), "opt": opt.state_dict(), "sched": sch.state_dict()}
    buf = Path(tempfile.mkdtemp()) / "mid.pt"
    torch.save(ckpt, buf)
    del m, opt, sch, ckpt
    torch.cuda.empty_cache()

    m2, opt2, sch2 = fresh()
    ck = torch.load(buf, map_location="cuda", weights_only=False)
    m2.load_state_dict(ck["model"]); opt2.load_state_dict(ck["opt"]); sch2.load_state_dict(ck["sched"])
    resumed = manual_loop(args, m2, opt2, sch2, batches, n, n + k)

    print(f"optimizer={args.optimizer} resume at step {n}, next {k} steps")
    print("uninterrupted:", [f"{x:.6f}" for x in full[n:]])
    print("resumed      :", [f"{x:.6f}" for x in resumed])
    print("max abs diff :", max(abs(a - b) for a, b in zip(full[n:], resumed)))
    shutil.rmtree(buf.parent, ignore_errors=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "resume"])
    p.add_argument("--optimizer", default="adamw")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--extra", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default="src/configs/param.yaml")
    p.add_argument("--cache-dir", default="data_cache")
    p.add_argument("--no-dropout", action="store_true")
    p.add_argument("--out", default="run.json")
    args = p.parse_args()
    {"run": cmd_run, "resume": cmd_resume}[args.cmd](args)


if __name__ == "__main__":
    main()
