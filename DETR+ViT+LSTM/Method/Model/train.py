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

from data import DataConfig, make_dataloaders  # noqa: E402
from model import LSTMActivityClassifier  # noqa: E402


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


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


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
    p.add_argument("--min-delta", type=float, default=0.0, help="Minimum improvement to reset patience.")
    p.add_argument("--early-stop-metric", choices=("val_loss", "val_acc"), default="val_loss")

    # Model params (as requested defaults)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--bidirectional", action="store_true", default=True)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--pooling", choices=("last", "mean"), default="last")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_dataloaders(
        DataConfig(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
    )

    model = LSTMActivityClassifier(
        input_dim=1280,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        bidirectional=bool(args.bidirectional),
        dropout=float(args.dropout),
        pooling=args.pooling,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    criterion = nn.CrossEntropyLoss()

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
                },
            )

        if early.step(metric_value):
            print(
                f"Early stopping at epoch {epoch} "
                f"(best {args.early_stop_metric}={early.best}, patience={early.patience})."
            )
            break


if __name__ == "__main__":
    main()

