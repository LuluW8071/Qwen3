from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional progress display

    def tqdm(iterable, **kwargs):
        return iterable


from src.configs.logging import logger
from src.configs.model_config import ModelConfig
from src.data.tokenizer import TextTokenizer

TOKENIZER_TRAIN_DOCS = 100_000


class TextTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Turn one contiguous token stream into next-token prediction samples."""

    def __init__(self, tokens: list[int], seq_len: int = 512):
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        # int32 array: a Python list of 100M+ ints costs ~5 GB per process and per worker
        # keep ndarrays (incl. memmaps) as-is so a mapped cache is never copied into RAM
        self.tokens = (
            tokens if isinstance(tokens, np.ndarray) else np.asarray(tokens, dtype=np.int32)
        )
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index
        window = self.tokens[start : start + self.seq_len + 1].astype(np.int64)
        return torch.from_numpy(window[:-1]), torch.from_numpy(window[1:])


def _resolve_vocab_size(config: ModelConfig, tokenizer) -> None:
    tokenizer_vocab_size = tokenizer.vocab_size
    if config.vocab_size is None:
        config.vocab_size = tokenizer_vocab_size
        return
    if config.vocab_size != tokenizer_vocab_size:
        raise ValueError(
            f"Configured vocab_size {config.vocab_size} does not match tokenizer "
            f"vocab_size {tokenizer_vocab_size}. Use a matching tokenizer or set "
            "vocab_size to null for automatic detection."
        )


def load_and_cache_data(
    config: ModelConfig,
    cache_dir: str | os.PathLike[str] = "data_cache",
) -> tuple[list[str], Any, list[int]]:
    """Load Cosmopedia text, tokenize it once, and reuse local cache afterward."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    requested_vocab_size = config.vocab_size or "auto"
    cache_file = cache_path / (
        f"tokenized_data_{config.num_documents}_{config.max_tokens}_"
        f"vocab{requested_vocab_size}_tokenizers.pkl"
    )

    tokens_file = cache_file.with_suffix(".tokens.npy")

    if cache_file.exists():
        logger.info("Loading cached data from {}", cache_file)
        with cache_file.open("rb") as file:
            cached_data = pickle.load(file)

        texts = cached_data["texts"]
        tokenizer = TextTokenizer.from_json(cached_data["tokenizer_json"])
        tokens = cached_data["tokens"]
        if tokens is None:  # new format: tokens live in a memory-mapped .npy beside the pickle
            tokens = np.load(tokens_file, mmap_mode="r")
        elif not isinstance(tokens, np.ndarray):  # old format: Python list, convert once
            tokens = np.asarray(tokens, dtype=np.int32)
        _resolve_vocab_size(config, tokenizer)
        logger.info("Loaded {} documents and {:,} tokens", len(texts), len(tokens))
        return texts, tokenizer, tokens

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Data preparation requires `datasets`.") from exc

    logger.info("Processing new data; cache will be written to {}", cache_file)
    base_tokenizer = TextTokenizer.from_pretrained()

    dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
        split="train",
        streaming=True,
    )
    texts: list[str] = []
    for index, item in enumerate(dataset):
        if index >= config.num_documents:
            break
        texts.append(item["text"][:3000])

    if config.vocab_size is not None and config.vocab_size != base_tokenizer.vocab_size:
        # BPE merges converge long before 500k docs; training on a prefix keeps this step fast.
        tokenizer = base_tokenizer.train_new(texts[:TOKENIZER_TRAIN_DOCS], config.vocab_size)
    else:
        tokenizer = base_tokenizer

    chunks: list[np.ndarray] = []
    batch_size = 256
    for start in tqdm(range(0, len(texts), batch_size), desc="Tokenizing"):
        for ids in tokenizer.encode_batch(texts[start : start + batch_size]):
            chunks.append(np.asarray(ids, dtype=np.int32))

    tokens = np.concatenate(chunks)[: config.max_tokens]
    if len(tokens) <= config.max_seq_len:
        raise ValueError(
            f"Need more than max_seq_len ({config.max_seq_len}) tokens; got {len(tokens)}."
        )

    _resolve_vocab_size(config, tokenizer)

    # Cached processed data; tokens go to a .npy so every rank/worker can mmap them
    np.save(tokens_file, tokens.astype(np.uint16 if tokenizer.vocab_size < 2**16 else np.int32))
    with cache_file.open("wb") as file:
        pickle.dump({"texts": texts, "tokenizer_json": tokenizer.to_json(), "tokens": None}, file)
    tokens = np.load(tokens_file, mmap_mode="r")
    logger.info("Cached {} tokens to {}", len(tokens), cache_file)
    return texts, tokenizer, tokens
