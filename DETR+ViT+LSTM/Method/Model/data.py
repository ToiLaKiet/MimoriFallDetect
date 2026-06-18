from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader

# Allow importing when running from repo root.
this_dir = Path(__file__).resolve().parent
if str(this_dir) not in sys.path:
    sys.path.insert(0, str(this_dir))

from model import EmbeddingStandardScaler, VitPoseSequenceDataset, pad_collate_sequences  # noqa: E402


@dataclass(frozen=True)
class DataConfig:
    data_root: Path
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = True
    embedding_dim: int = 1280
    min_frames: int = 2
    scaler: EmbeddingStandardScaler | None = None


def make_dataset(
    *,
    data_root: str | Path,
    split: Literal["train", "val", "test"],
    embedding_dim: int = 1280,
    min_frames: int = 2,
    scaler: EmbeddingStandardScaler | None = None,
) -> VitPoseSequenceDataset:
    return VitPoseSequenceDataset(
        root_dir=Path(data_root),
        split=split,
        embedding_dim=embedding_dim,
        min_frames=min_frames,
        load_metadata=False,
        scaler=scaler,
    )


def make_loader(
    ds: torch.utils.data.Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=pad_collate_sequences,
        drop_last=False,
    )


def make_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    train_ds = make_dataset(
        data_root=cfg.data_root,
        split="train",
        embedding_dim=cfg.embedding_dim,
        min_frames=cfg.min_frames,
        scaler=cfg.scaler,
    )
    val_ds = make_dataset(
        data_root=cfg.data_root,
        split="val",
        embedding_dim=cfg.embedding_dim,
        min_frames=cfg.min_frames,
        scaler=cfg.scaler,
    )

    train_loader = make_loader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    test_dir = Path(cfg.data_root) / "test"
    test_loader: DataLoader | None = None
    if test_dir.is_dir():
        test_ds = make_dataset(
            data_root=cfg.data_root,
            split="test",
            embedding_dim=cfg.embedding_dim,
            min_frames=cfg.min_frames,
            scaler=cfg.scaler,
        )
        test_loader = make_loader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )

    return train_loader, val_loader, test_loader

