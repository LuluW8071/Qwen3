# PLAN: 8-bit optimizer state (Muon8bit + AdamW8bit)

Branch: `8bit-muon` (from `main` @ 4122e66). Written before any code change.

## 1. Audit findings

| Item | Finding |
|---|---|
| Framework | PyTorch Lightning 2.6.6 (`pl.Trainer`, `LightningModule`), argparse entrypoint `main.py` + yaml config `src/configs/param.yaml` |
| Model | Custom `Qwen3LLM` (`src/neuralnet/qwen.py`): d_model 512, 6 layers, 8 heads / 4 KV heads, d_ff 1536, vocab 49152 (SmolLM-135M tokenizer), seq 1024 |
| Params | 44,048,384 total |
| Current optimizer | `HybridOptimizer(Muon, torch.optim.AdamW)` (`src/optim/hybridoptim.py`). Custom 32-bit Muon (`src/optim/muon.py`, lr 0.01, mom 0.95, no wd) + AdamW (lr = muon_lr*0.1 = 1e-3, betas (0.9, 0.975), wd 0.1) |
| Scheduler | One `LambdaLR` over the facade: 5% linear warmup, cosine to 10% floor, per step |
| Precision | `bf16-mixed` when `use_amp` and CUDA, else `32-true`. fp32 master weights, bf16 autocast. No GradScaler under bf16 (Lightning only creates one for `16-mixed`) |
| Grad clip / accum | Lightning `gradient_clip_val=1.0` (norm), `accumulate_grad_batches=1`. Clips across all params of the optimizer facade, once, before step |
| Grad checkpointing | None |
| Distributed | Single GPU now. `--dist_backend` defaults to `deepspeed_stage_2`, used only when `--gpus > 1` |
| Checkpoints | Lightning `ModelCheckpoint` (top-3 val_loss + last). Optimizer/scheduler state saved via facade `state_dict` (`opt_0`, `opt_1`) |
| Data | Cosmopedia-v2 text, one concatenated token stream, sliding-window next-token samples, loss on every token. **Plain pretraining, not chat/agent data** (see section 8) |
| Env | Python 3.14.0, torch 2.14.0+cu126, nvcc 12.8, driver 560.35.05, RTX 4050 Laptop 6 GB (sm_89), triton 3.8.0, installed bitsandbytes 0.50.2 (released) |
| Released bnb | 0.50.2 has `AdamW8bit` but **no Muon** (confirmed by grep and `dir(bnb.optim)`) |
| Baseline tests | `pytest`: 47 passed |

### Blockers / flags
- **Tied embeddings**: `lm_head.weight is token_embedding.weight`. It goes to the AdamW group (it is the embedding), so there is no Muon conflict. `named_parameters()` lists it once.
- **DeepSpeed ZeRO-2 on multi-GPU**: ZeRO-2 shards optimizer state, which is incompatible with bnb optimizers handing back full matrices. The 8-bit path will refuse `--gpus > 1` with deepspeed; DDP would be fine (not tested, one GPU here). Single-GPU runs are unaffected.
- **fp16**: not used by default (bf16). `8bit` with `fp16-mixed` is not blocked, but bf16 is recommended.
- **LoRA**: not used.
- **The user removed 8-bit support in commit 4122e66** (earlier Lightning `BitsandbytesPrecision` int8-training + `Adam8bit`). This plan does not restore int8 weights, only optimizer state, via an opt-in flag.
- Existing custom Muon differs from the fork's: no weight decay, lr scaling `max(1, rows/cols)**0.5`, and `lerp_` mutates `p.grad` in place. The fork applies decoupled wd and `adjust_lr="rms_norm"`. These are **different update rules**, so lrs are not interchangeable (see 3).

## 2. Fork API (read from source, commit `c80d430`, `bitsandbytes/optim/muon.py`)

```
bnb.optim.Muon8bit(params, lr=1e-3, momentum=0.95, weight_decay=0.1, nesterov=True,
                   adjust_lr="rms_norm", orthogonalize_fn=None, ns_chunk_size=8,
                   min_8bit_size=4096, is_paged=False)
```
- Rejects params with ndim < 2 (`ValueError`), rejects FSDP1/ZeRO-3 layouts.
- Momentum buffer quantized to uint8, blocksize 256; tensors under `min_8bit_size` elements stay fp32. Every Muon matrix here is >= 131,072 elements, so all are quantized.
- `adjust_lr="rms_norm"` scales lr by `0.2*sqrt(max(fan_out, fan_in))` (AdamW-RMS matching), so Muon lr is on an AdamW-like scale.
- `Muon32bit` is the same class with `optim_bits=32`, which gives a same-update-rule fp32 baseline.
- Uses gram_newton_schulz if installed (not installed), else its own 5-step bf16 Newton-Schulz. Fused Triton kernels for the 8-bit momentum path are used when triton imports (triton 3.8.0 is present).

## 3. Param grouping rule

Rule (same as current trainer): Muon if `ndim == 2` and `"token_embedding" not in name` and `"norm" not in name`; everything else goes to AdamW. This lands as:

| Group | Tensors | numel | Names |
|---|---|---|---|
| Muon | 42 | 18,874,368 (42.8%) | per layer 0-5: `attention.{q_proj,k_proj,v_proj,w_o}.weight`, `feed_forward.{gate_proj,up_proj,down_proj}.weight` (7 per layer; 3,145,728 numel per layer) |
| AdamW | 38 | 25,174,016 (57.2%) | `token_embedding.weight` (25,165,824, tied lm_head) plus all 1-D: `q_norm/k_norm .weight/.bias`, `norm1/norm2.weight`, final `norm.weight` |

The startup summary printed by the code is the source of truth; RESULTS records the actual numbers.

## 4. Optimizer config (and where each value comes from)

| Optimizer | Param | Value | Source |
|---|---|---|---|
| Muon8bit | lr | 1e-3 (= AdamW lr) | fork default 1e-3; `rms_norm` adjust expects an AdamW-scale lr |
| | momentum | 0.95 | fork default, equals current trainer |
| | nesterov | True | fork default, equals current trainer |
| | weight_decay | 0.1 | fork default; equals the yaml `weight_decay` already used for AdamW |
| | adjust_lr | "rms_norm" | fork default |
| | ns_chunk_size / min_8bit_size | 8 / 4096 | fork defaults |
| AdamW8bit | lr | `muon_lr*0.1` = 1e-3 | current trainer |
| | betas / eps / wd | (0.9, 0.975) / 1e-8 / 0.1 | current trainer |
| | min_8bit_size | 4096 (bnb default) | bnb `AdamW8bit` default |

Muon lr is the main uncertainty. If it diverges or lags the baseline, Step 4.4 goes: lower Muon lr, then 32-bit AdamW for non-matrix params.

## 5. Scheduler

Two optimizers need one schedule shape (5% warmup, cosine to 10%). Lightning's automatic optimization does not support multiple optimizers/schedulers, so the existing `HybridOptimizer` facade stays: one `LambdaLR` multiplies each param-group `base_lr` by the same factor, which is exactly "one scheduler per optimizer with the same shape". Per-group lr is logged separately. **This deviates from the literal "two schedulers" wording on purpose**; the alternative is switching to manual optimization, which is a much bigger diff. Both inner optimizers' states are saved via `HybridOptimizer.state_dict`.

## 6. Install steps (fork)

```
git clone https://github.com/theblackcat102/bitsandbytes <dir>
git -C <dir> checkout c80d430a002edd3ef403a128ba3927e6c02a6417     # feature/muon, 2026-06-14
SKBUILD_CMAKE_ARGS="-DCOMPUTE_BACKEND=cuda;-DCOMPUTE_CAPABILITY=89" \
  PATH=/usr/local/cuda/bin:$PATH uv build --wheel --out-dir <wheeldir> <dir>
uv pip install --python .venv/bin/python --no-deps --force-reinstall <wheeldir>/bitsandbytes-*.whl
```
Needs cmake >= 3.22.1, gcc 11, CUDA toolkit 12.8 (all present). Compute capability 89 matches the RTX 4050; adjust for other GPUs. Rollback: `uv pip install --python .venv/bin/python bitsandbytes==0.50.2`. The wheel is kept so the install is repeatable. The package is not added to pyproject dependencies (it is not on PyPI); a README note documents the pinned commit.

## 7. Fallback

If the fork fails to build or run: `torch.optim.Muon` (32-bit, torch 2.14) for matrices + `bnb.optim.AdamW8bit` (released 0.50.2) for the rest. Selectable via `--optimizer muon32` for the A/B baseline anyway (fork's `Muon32bit`).

## 8. Step 5 (agent CLI data) - audit-only preview

The repo has **no chat template, no roles, no loss masking, no packing logic, no tool-call tokens**. `TextTokenDataset` is a sliding window over one concatenated stream of Cosmopedia text with `add_special_tokens=False`, loss on all tokens, `max_tokens` truncation of the stream, and the first `3000` characters of each document. None of the five checks has anything to test against. Building this is a new feature, not a bug fix; I will ask rather than guess (dataset, template, special tokens).

## 9. Memory estimate (optimizer state, fp32 weights, 44.05M params)

| | Before (current) | After (8bit) |
|---|---|---|
| Muon momentum (18.87M) | 4 B = 75.5 MB | ~1 B + absmax = ~19.0 MB |
| AdamW m+v (25.17M) | 8 B = 201.4 MB | 2 B + absmax = ~50.9 MB |
| Total opt state | ~277 MB | ~70 MB |

Params (176 MB) and grads (176 MB) are unchanged. Activation memory at batch 8 x seq 1024 will dominate peak on the 6 GB GPU, so peak will drop by roughly the ~207 MB saved, not by a large factor. RESULTS_8BIT.md reports measured `max_memory_allocated` only.

## 10. Risks and tests

| Risk | Test |
|---|---|
| Fork does not build/run on py3.14 / torch 2.14 | Build and `import` + one `step()` before touching trainer code; fall back per section 7 |
| Muon8bit loss diverges vs baseline | 200+ step A/B, same seed/data: adamw+custom Muon, muon32 (fork), 8bit |
| Facade + bnb `state_dict`/`load_state_dict` mismatch on resume | 50 steps -> save -> reload -> 10 steps vs uninterrupted run, compare losses |
| Grad clip not applied to both groups | Lightning clips on the facade's `param_groups`; unit test checks the grad norm after clipping |
| 1-D param reaches Muon | Param-group helper test (all Muon params ndim == 2, disjoint, union = all params) |
| torch.compile interaction (`--compile`) | Not in scope; existing flag unchanged, noted as untested with 8-bit |
| Existing behavior regresses | Default `--optimizer adamw` keeps today's code path; the 47 existing tests must still pass |
