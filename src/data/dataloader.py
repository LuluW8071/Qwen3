from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl

from src.configs.model_config import ModelConfig
from src.data.dataset import TextTokenDataset, load_and_cache_data


class QwenDataModule(pl.LightningDataModule):
    """Prepare deterministic train and validation windows from cached tokens."""

    def __init__(
        self,
        config: ModelConfig,
        cache_dir: str = "data_cache",
        num_workers: int = 0,
        validation_fraction: float = 0.1,
    ):
        super().__init__()
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        self.config = config
        self.cache_dir = cache_dir
        self.num_workers = num_workers
        self.validation_fraction = validation_fraction
        self.train_dataset: Optional[TextTokenDataset] = None
        self.val_dataset: Optional[TextTokenDataset] = None
        self.tokenizer = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        _, tokenizer, tokens = load_and_cache_data(self.config, self.cache_dir)
        self.tokenizer = tokenizer
        dataset = TextTokenDataset(tokens, self.config.max_seq_len)
        if len(dataset) < 2:
            raise ValueError("Dataset needs at least two sequence samples")

        val_size = max(1, int(len(dataset) * self.validation_fraction))
        train_size = len(dataset) - val_size
        if train_size < 1:
            raise ValueError("Dataset needs at least one training sample")

        self.train_dataset, self.val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("validate")
        return self._loader(self.val_dataset, shuffle=False)
