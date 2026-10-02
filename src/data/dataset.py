from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, List

import torch
from torch.utils.data import Dataset
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional progress display
    def tqdm(iterable, **kwargs):
        return iterable

from src.configs.logging import logger
from src.configs.model_config import ModelConfig


class TextTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Turn one contiguous token stream into next-token prediction samples."""

    def __init__(self, tokens: List[int], seq_len: int = 512):
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index
        x = torch.tensor(self.tokens[start:start + self.seq_len], dtype=torch.long)
        y = torch.tensor(
            self.tokens[start + 1:start + self.seq_len + 1], dtype=torch.long
        )
        return x, y


def load_and_cache_data(
    config: ModelConfig,
    cache_dir: str | os.PathLike[str] = "data_cache",
) -> tuple[list[str], Any, list[int]]:
    """Load Cosmopedia text, tokenize it once, and reuse local cache afterward."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / (
        f"tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"
    )

    if cache_file.exists():
        logger.info("Loading cached data from {}", cache_file)
        with cache_file.open("rb") as file:
            cached_data = pickle.load(file)

        texts = cached_data["texts"]
        tokenizer = cached_data["tokenizer"]
        tokens = cached_data["tokens"]
        config.vocab_size = tokenizer.vocab_size
        logger.info("Loaded {} documents and {:,} tokens", len(texts), len(tokens))
        return texts, tokenizer, tokens

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Data preparation requires `datasets` and `transformers`."
        ) from exc

    logger.info("Processing new data; cache will be written to {}", cache_file)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    all_tokens: list[int] = []
    for text in tqdm(texts, desc="Tokenizing"):
        all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))

    tokens = all_tokens[:config.max_tokens]
    if len(tokens) <= config.max_seq_len:
        raise ValueError(
            f"Need more than max_seq_len ({config.max_seq_len}) tokens; "
            f"got {len(tokens)}."
        )

    config.vocab_size = tokenizer.vocab_size

    # Cached processed data
    with cache_file.open("wb") as file:
        pickle.dump({"texts": texts, "tokenizer": tokenizer, "tokens": tokens}, file)
    logger.info("Cached {} tokens to {}", len(tokens), cache_file)
    return texts, tokenizer, tokens
