from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


LabelName = Literal["fall", "normal"]
LABEL_TO_ID: dict[LabelName, int] = {"fall": 1, "normal": 0}
ID_TO_LABEL: dict[int, LabelName] = {v: k for k, v in LABEL_TO_ID.items()}


def _sorted_npy_paths(sequence_dir: Path) -> list[Path]:
    npys = [p for p in sequence_dir.iterdir() if p.is_file() and p.suffix.lower() == ".npy"]
    return sorted(npys, key=lambda p: p.name) 


def _infer_label_from_path(sequence_dir: Path) -> int | None:
    """
    Infer label from any parent folder named 'fall' or 'normal'.
    Returns:
      - int label id when found
      - None when unlabeled (e.g. test split without class folders)
    """
    for parent in sequence_dir.parents:
        name = parent.name.lower()
        if name in LABEL_TO_ID:
            return LABEL_TO_ID[name]  # type: ignore[arg-type]
    return None


def _read_metadata(sequence_dir: Path) -> dict[str, Any] | None:
    path = sequence_dir / "metadata.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@dataclass(frozen=True)
class SequenceSample:
    sequence_dir: Path
    label: int | None
    length: int
    metadata: dict[str, Any] | None


@dataclass(frozen=True)
class EmbeddingStandardScaler:
    """Per-dimension standard scaler fit on train frames: z = (x - mean) / std."""

    mean: np.ndarray  # (D,)
    std: np.ndarray  # (D,)
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.std.ndim != 1:
            raise ValueError(f"mean/std must be 1-D, got {self.mean.shape}, {self.std.shape}")
        if self.mean.shape != self.std.shape:
            raise ValueError(f"mean/std shape mismatch: {self.mean.shape} vs {self.std.shape}")

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return (x - mean) / std.clamp_min(self.eps)

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "std": self.std, "eps": self.eps}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingStandardScaler:
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
            eps=float(data.get("eps", 1e-8)),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std, eps=np.float64(self.eps))

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingStandardScaler:
        with np.load(path) as data:
            return cls(
                mean=data["mean"].astype(np.float32),
                std=data["std"].astype(np.float32),
                eps=float(data["eps"]) if "eps" in data else 1e-8,
            )


def fit_embedding_scaler(
    dataset: VitPoseSequenceDataset,
    *,
    eps: float = 1e-8,
) -> EmbeddingStandardScaler:
    """Fit per-dimension mean/std on all frames in a dataset (intended for train split)."""
    d = dataset.embedding_dim
    n = 0
    sum_x = np.zeros(d, dtype=np.float64)
    sum_x2 = np.zeros(d, dtype=np.float64)

    for sample in dataset.samples:
        for p in _sorted_npy_paths(sample.sequence_dir):
            arr = np.load(p)
            arr = np.asarray(arr, dtype=np.float64)
            if arr.ndim != 1 or arr.shape[0] != d:
                raise ValueError(
                    f"Unexpected embedding shape in {p}: got {arr.shape}, expected ({d},)"
                )
            sum_x += arr
            sum_x2 += arr * arr
            n += 1

    if n == 0:
        raise ValueError("Cannot fit scaler: no frames found in dataset.")

    mean = sum_x / n
    var = sum_x2 / n - mean**2
    std = np.sqrt(np.maximum(var, eps))
    return EmbeddingStandardScaler(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        eps=eps,
    )


class VitPoseSequenceDataset(torch.utils.data.Dataset):
    """
    One sample == one sequence folder containing frame_*.npy (each is (1280,)).

    Supports layouts:
      - train/fall/seq001/frame*.npy
      - train/normal/seq002/frame*.npy
      - val/(fall|normal)/seq*/...
      - test/seq*/... (unlabeled) OR test/(fall|normal)/seq*/...
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["train", "val", "test"],
        *,
        embedding_dim: int = 1280,
        load_metadata: bool = True,
        min_frames: int = 1,
        scaler: EmbeddingStandardScaler | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.embedding_dim = int(embedding_dim)
        self.load_metadata = bool(load_metadata)
        self.min_frames = int(min_frames)
        self.scaler = scaler

        split_dir = self.root_dir / split
        if not split_dir.is_dir():
            raise ValueError(f"Split directory not found: {split_dir}")

        self.samples: list[SequenceSample] = self._index_sequences(split_dir)
        if len(self.samples) == 0:
            raise ValueError(f"No sequences found under: {split_dir}")

    def _index_sequences(self, split_dir: Path) -> list[SequenceSample]:
        samples: list[SequenceSample] = []

        # Common cases:
        # - split_dir has class subfolders (fall/normal)
        # - or split_dir has sequences directly (unlabeled test)
        candidate_sequence_dirs: list[Path] = []

        for child in sorted(split_dir.iterdir()):
            if not child.is_dir():
                continue
            if child.name.lower() in LABEL_TO_ID:
                for seq_dir in sorted(child.iterdir()):
                    if seq_dir.is_dir():
                        candidate_sequence_dirs.append(seq_dir)
            else:
                candidate_sequence_dirs.append(child)

        for seq_dir in candidate_sequence_dirs:
            frame_paths = _sorted_npy_paths(seq_dir)
            if len(frame_paths) < self.min_frames:
                continue
            label = _infer_label_from_path(seq_dir)
            metadata = _read_metadata(seq_dir) if self.load_metadata else None
            samples.append(
                SequenceSample(
                    sequence_dir=seq_dir,
                    label=label,
                    length=len(frame_paths),
                    metadata=metadata,
                )
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        frame_paths = _sorted_npy_paths(sample.sequence_dir)

        frames: list[torch.Tensor] = []
        for p in frame_paths:
            arr = np.load(p)
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim != 1 or arr.shape[0] != self.embedding_dim:
                raise ValueError(
                    f"Unexpected embedding shape in {p}: got {arr.shape}, expected ({self.embedding_dim},)"
                )
            frames.append(torch.from_numpy(arr))

        x = torch.stack(frames, dim=0)  # (T, D)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        out: dict[str, Any] = {
            "x": x,
            "length": x.shape[0],
            "sequence_dir": str(sample.sequence_dir),
        }
        if sample.label is not None:
            out["y"] = int(sample.label)
        if sample.metadata is not None:
            out["metadata"] = sample.metadata
        return out


def pad_collate_sequences(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate a list of items from VitPoseSequenceDataset into padded batch tensors.
    Collate là hàm để gom nhóm các tensor thành một batch.
    Ví dụ:
                        batch = [
                            {"x": torch.tensor([[1, 2, 3], [4, 5, 6]]), "length": 2},
                            {"x": torch.tensor([[7, 8, 9], [10, 11, 12]]), "length": 2},
                        ]
                        pad_collate_sequences(batch) sẽ trả về:
                        {"x": torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), "lengths": torch.tensor([2, 2])}
                        Vì:
                        - B là số lượng batch.
                        - T_max là số lượng frame tối đa trong batch.
                        - D là số lượng đặc trưng của mỗi frame.
                        - lengths là số lượng frame của mỗi sequence trong batch.
                        - sequence_dir là danh sách các đường dẫn đến các sequence trong batch.
    Returns:
      - x: (B, T_max, D) float32
      - lengths: (B,) long
      - y: (B,) long if labels exist in batch
      - sequence_dir: list[str]
    """
    xs = [item["x"] for item in batch]
    lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)

    # pad to max length
    d = xs[0].shape[-1]
    t_max = int(max(x.shape[0] for x in xs))
    x_pad = xs[0].new_zeros((len(xs), t_max, d))
    for i, x in enumerate(xs):
        t = x.shape[0]
        x_pad[i, :t] = x

    out: dict[str, Any] = {
        "x": x_pad,
        "lengths": lengths,
        "sequence_dir": [item["sequence_dir"] for item in batch],
    }

    if all(("y" in item) for item in batch):
        out["y"] = torch.tensor([int(item["y"]) for item in batch], dtype=torch.long)

    if any(("metadata" in item) for item in batch):
        out["metadata"] = [item.get("metadata") for item in batch]

    return out


class LSTMActivityClassifier(nn.Module):
    """
    Sequence classifier for (T, 768) ViTPose embeddings -> {normal, fall}.
    Uses packed sequences for variable-length batches.
    """

    def __init__(
        self,
        *,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
        num_classes: int = 2,
        pooling: Literal["last", "mean"] = "last",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.num_classes = int(num_classes)
        self.pooling: Literal["last", "mean"] = pooling

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )

        out_dim = self.hidden_dim * (2 if self.bidirectional else 1)

        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(out_dim, self.num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (B, T, D)
          lengths: (B,) lengths before padding
        Returns:
          logits: (B, num_classes)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x shape (B,T,D), got {tuple(x.shape)}")
        if lengths.ndim != 1:
            raise ValueError(f"Expected lengths shape (B,), got {tuple(lengths.shape)}")

        lengths_cpu = lengths.to(device="cpu")
        packed = pack_padded_sequence(
            x,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        ) # packed is a tuple of (data, lengths) where data is a tensor of shape (B, T, D) and lengths is a tensor of shape (B,).
        packed_out, (h_n, _c_n) = self.lstm(packed) # packed_out is a tuple of (output, hidden_state, cell_state) where output is a tensor of shape (B, T, H) and hidden_state is a tensor of shape (num_layers, B, H) and cell_state is a tensor of shape (num_layers, B, H).
        # h_n is the hidden state of the last layer.
        # _c_n is the cell state of the last layer.
        if self.pooling == "last":
            # h_n: (num_layers * num_directions, B, hidden_dim)
            if self.bidirectional:
                last_fwd = h_n[-2]
                last_bwd = h_n[-1]
                feat = torch.cat([last_fwd, last_bwd], dim=-1)
            else:
                feat = h_n[-1]
        elif self.pooling == "mean":
            out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, T_max, H*)
            b, t_max, feat_dim = out.shape
            mask = (
                torch.arange(t_max, device=lengths.device)
                .unsqueeze(0)
                .expand(b, t_max)
                < lengths.unsqueeze(1)
            )
            out = out * mask.unsqueeze(-1)
            feat = out.sum(dim=1) / lengths.clamp_min(1).unsqueeze(1).to(out.dtype)
            if feat.shape[-1] != feat_dim:
                raise RuntimeError("Unexpected pooled feature dim.")
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")

        return self.head(feat)


def predict_label(logits: torch.Tensor) -> list[LabelName]:
    pred = logits.argmax(dim=-1).detach().cpu().tolist()
    return [ID_TO_LABEL[int(i)] for i in pred]

