from __future__ import annotations

import sys
import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Allow running from repo root without installing as a package.
this_dir = Path(__file__).resolve().parent
if str(this_dir) not in sys.path:
    sys.path.insert(0, str(this_dir))

from data import DataConfig, make_dataset, make_dataloaders  # noqa: E402
from model import (  # noqa: E402
    EmbeddingStandardScaler,
    ID_TO_LABEL,
    LSTMActivityClassifier,
    VitPoseSequenceDataset,
    fit_embedding_scaler,
)


def resolve_device(device_name: str | None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    y_true = y_true.to(torch.long).view(-1)
    y_pred = y_pred.to(torch.long).view(-1)
    k = num_classes
    idx = y_true * k + y_pred
    cm = torch.bincount(idx, minlength=k * k).reshape(k, k)
    return cm


@dataclass
class EarlyStopping:
    patience: int = 10
    min_delta: float = 0.0
    mode: str = "min"  # "min" for loss, "max" for metric

    best: float | None = None
    bad_epochs: int = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            self.bad_epochs = 0
            return False

        improved = False
        if self.mode == "min":
            improved = value < (self.best - self.min_delta)
        elif self.mode == "max":
            improved = value > (self.best + self.min_delta)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        if improved:
            self.best = value
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        x = batch["x"].to(device)
        lengths = batch["lengths"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * y.shape[0]
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.shape[0])

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    criterion: nn.Module,
    num_classes: int = 2,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for batch in loader:
        x = batch["x"].to(device)
        lengths = batch["lengths"].to(device)
        y = batch["y"].to(device)

        logits = model(x, lengths)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * y.shape[0]
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.shape[0])
        cm += confusion_matrix(y, pred, num_classes=num_classes).cpu()

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "confusion_matrix": cm,
    }


def count_class_labels(dataset: VitPoseSequenceDataset, num_classes: int = 2) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for sample in dataset.samples:
        if sample.label is None:
            continue
        label = int(sample.label)
        if label < 0 or label >= num_classes:
            raise ValueError(f"Unexpected label id {label} in {sample.sequence_dir}")
        counts[label] += 1
    return counts


def compute_balanced_class_weights(counts: torch.Tensor) -> torch.Tensor:
    """Balanced weights: w_c = N / (K * n_c)."""
    n = int(counts.sum().item())
    k = int(counts.numel())
    if n == 0:
        raise ValueError("Cannot compute class weights: no labeled training samples.")

    weights = torch.zeros(k, dtype=torch.float32)
    for c in range(k):
        n_c = int(counts[c].item())
        if n_c == 0:
            raise ValueError(f"Cannot compute class weights: class {c} ({ID_TO_LABEL.get(c, '?')}) has 0 samples.")
        weights[c] = n / (k * n_c)
    return weights


def parse_class_weights_arg(value: str, num_classes: int = 2) -> torch.Tensor:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != num_classes:
        raise ValueError(f"--class-weights expects {num_classes} comma-separated values, got {len(parts)}.")
    return torch.tensor([float(p) for p in parts], dtype=torch.float32)


def build_criterion(
    train_dataset: VitPoseSequenceDataset,
    *,
    device: torch.device,
    class_weights_arg: str | None,
    num_classes: int = 2,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    counts = count_class_labels(train_dataset, num_classes=num_classes)
    if class_weights_arg is not None:
        weights = parse_class_weights_arg(class_weights_arg, num_classes=num_classes)
    else:
        weights = compute_balanced_class_weights(counts)

    for c in range(num_classes):
        name = ID_TO_LABEL.get(c, str(c))
        print(f"Class counts (train): {name}={int(counts[c].item())}")
    weight_parts = [f"{ID_TO_LABEL.get(c, str(c))}={weights[c].item():.4f}" for c in range(num_classes)]
    print(f"Class weights: {', '.join(weight_parts)}")

    return nn.CrossEntropyLoss(weight=weights.to(device)), counts, weights


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return ckpt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM fall/normal classifier on ViTPose embeddings.")
    p.add_argument("--data-root", type=Path, required=True, help="Folder containing train/ val/ test/ splits.")
    p.add_argument("--outdir", type=Path, default=Path("runs/lstm_vitpose"), help="Output directory for checkpoints.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default=None, help="cuda|mps|cpu. Auto-detect if omitted.")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs).")
    p.add_argument("--min-delta", type=float, default=0.01, help="Minimum improvement to reset patience.") 
    p.add_argument("--early-stop-metric", choices=("val_loss", "val_acc"), default="val_loss")
    p.add_argument(
        "--class-weights",
        type=str,
        default=None,
        help="Optional manual class weights as 'w_normal,w_fall' (label ids 0,1). "
        "If omitted, weights are computed from train split as N/(K*n_c).",
    )

    # Model params (as requested defaults)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--bidirectional", action="store_true", default=True)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--pooling", choices=("last", "mean"), default="last")
    p.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable per-dimension standard scaling of embeddings.",
    )
    p.add_argument(
        "--scaler-path",
        type=Path,
        default=None,
        help="Load precomputed scaler (.npz) instead of fitting on train split.",
    )
    return p.parse_args()


def resolve_scaler(
    args: argparse.Namespace,
    *,
    data_root: Path,
    embedding_dim: int,
    min_frames: int,
    outdir: Path,
) -> EmbeddingStandardScaler | None:
    if args.no_standardize:
        return None

    if args.scaler_path is not None:
        scaler = EmbeddingStandardScaler.load(args.scaler_path)
        print(f"Loaded embedding scaler from {args.scaler_path.resolve()}")
        return scaler

    train_ds = make_dataset(
        data_root=data_root,
        split="train",
        embedding_dim=embedding_dim,
        min_frames=min_frames,
    )
    scaler = fit_embedding_scaler(train_ds)
    scaler_path = outdir / "scaler.npz"
    scaler.save(scaler_path)
    n_frames = sum(sample.length for sample in train_ds.samples)
    print(
        f"Fitted embedding scaler on {n_frames} train frames "
        f"(dim={embedding_dim}), saved to {scaler_path.resolve()}"
    )
    return scaler


def scaler_checkpoint_payload(scaler: EmbeddingStandardScaler | None) -> dict[str, object]:
    if scaler is None:
        return {"scaler": None}
    return {"scaler": scaler.to_dict()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    scaler = resolve_scaler(
        args,
        data_root=args.data_root,
        embedding_dim=1280,
        min_frames=2,
        outdir=outdir,
    )
    data_cfg = DataConfig(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        scaler=scaler,
    )
    train_loader, val_loader, test_loader = make_dataloaders(data_cfg)
    print(f"Data config: {data_cfg}")

    model = LSTMActivityClassifier(
        input_dim=data_cfg.embedding_dim, # config this to 1280 or 768 depending on the model
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        bidirectional=bool(args.bidirectional),
        dropout=float(args.dropout),
        pooling=args.pooling,
    ).to(device)

    print(f"Model: {model}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    train_dataset = train_loader.dataset
    assert isinstance(train_dataset, VitPoseSequenceDataset)
    criterion, class_counts, class_weights = build_criterion(
        train_dataset,
        device=device,
        class_weights_arg=args.class_weights,
    )

    early = EarlyStopping(
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        mode="min" if args.early_stop_metric == "val_loss" else "max",
    )

    best_path = outdir / "best.pt"
    last_path = outdir / "last.pt"
    history_path = outdir / "history.jsonl"

    best_metric: float | None = None

    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {outdir.resolve()}")

    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()
        tr = train_one_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
        )
        va = evaluate(model, val_loader, device=device, criterion=criterion, num_classes=2)
        dt = time.time() - t0

        val_loss = float(va["loss"])
        val_acc = float(va["acc"])
        metric_value = val_loss if args.early_stop_metric == "val_loss" else val_acc

        cm = va["confusion_matrix"]
        assert isinstance(cm, torch.Tensor)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"time {dt:.1f}s"
        )
        print(f"Confusion matrix (rows=true, cols=pred):\n{cm.numpy()}")

        record = {
            "epoch": epoch,
            "train_loss": float(tr["loss"]),
            "train_acc": float(tr["acc"]),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "confusion_matrix": cm.tolist(),
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # Save last
        save_checkpoint(
            last_path,
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "confusion_matrix": cm,
                "class_counts": class_counts,
                "class_weights": class_weights,
                **scaler_checkpoint_payload(scaler),
            },
        )

        # Save best
        is_best = False
        if best_metric is None:
            is_best = True
        else:
            if args.early_stop_metric == "val_loss":
                is_best = metric_value < best_metric
            else:
                is_best = metric_value > best_metric

        if is_best:
            best_metric = metric_value
            save_checkpoint(
                best_path,
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "confusion_matrix": cm,
                    "class_counts": class_counts,
                    "class_weights": class_weights,
                    **scaler_checkpoint_payload(scaler),
                },
            )

        if early.step(metric_value):
            print(
                f"Early stopping at epoch {epoch} "
                f"(best {args.early_stop_metric}={early.best}, patience={early.patience})."
            )
            break

    if test_loader is None:
        print("No test/ split found — skipping test evaluation.")
        return

    if not best_path.is_file():
        print(f"No checkpoint at {best_path} — skipping test evaluation.")
        return

    print(f"\nEvaluating best checkpoint on test set: {best_path}")
    ckpt = load_checkpoint(best_path, model, device)
    te = evaluate(model, test_loader, device=device, criterion=criterion, num_classes=2)

    test_loss = float(te["loss"])
    test_acc = float(te["acc"])
    test_cm = te["confusion_matrix"]
    assert isinstance(test_cm, torch.Tensor)

    print(f"Test loss {test_loss:.4f} acc {test_acc:.4f}")
    print(f"Confusion matrix (rows=true, cols=pred):\n{test_cm.numpy()}")
    print(f"Best checkpoint epoch: {ckpt.get('epoch', '?')}")

    test_results_path = outdir / "test_results.json"
    test_results = {
        "checkpoint": str(best_path.resolve()),
        "best_epoch": ckpt.get("epoch"),
        "val_loss": float(ckpt.get("val_loss", float("nan"))),
        "val_acc": float(ckpt.get("val_acc", float("nan"))),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "confusion_matrix": test_cm.tolist(),
        "class_counts": class_counts.tolist(),
        "class_weights": class_weights.tolist(),
        "scaler": scaler.to_dict() if scaler is not None else None,
    }
    test_results_path.write_text(json.dumps(test_results, indent=2) + "\n", encoding="utf-8")
    print(f"Test results saved to {test_results_path.resolve()}")


if __name__ == "__main__":
    main()

