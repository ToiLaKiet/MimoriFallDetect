#!/usr/bin/env python3
"""Train frozen ViTPose-backbone embeddings + LSTM classifier."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from bbox_sequence_data import (
    BBoxSequenceDataBundle,
    BBoxSequenceItem,
    load_bbox_sequence_data,
    parse_image_size,
)
from common import SCRIPT_DIR, add_old_pipeline_to_path, choose_device
from model import DEFAULT_VIT_MODEL, FrozenVitPoseEmbeddingLSTMClassifier

add_old_pipeline_to_path()
from utils import evaluate_model, train_one_epoch  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_ids_for_report(data: BBoxSequenceDataBundle, num_classes: int) -> list[int]:
    label_ids = {item.label for item in data.sequences}
    label_ids.update(range(num_classes))
    return sorted(label_ids)


def split_class_counter(sequences: list[BBoxSequenceItem]) -> Counter:
    return Counter(item.label for item in sequences)


def print_split_class_distribution(data: BBoxSequenceDataBundle, num_classes: int) -> None:
    class_ids = class_ids_for_report(data, num_classes)
    split_counts = {
        "train": split_class_counter(data.train_sequences),
        "val": split_class_counter(data.val_sequences),
        "test": split_class_counter(data.test_sequences),
    }
    split_totals = {
        "train": len(data.train_sequences),
        "val": len(data.val_sequences),
        "test": len(data.test_sequences),
    }

    print("\nClass distribution by split:")
    print(f"{'class':>7} {'train':>8} {'val':>8} {'test':>8} {'total':>8}")
    for class_id in class_ids:
        train_count = split_counts["train"].get(class_id, 0)
        val_count = split_counts["val"].get(class_id, 0)
        test_count = split_counts["test"].get(class_id, 0)
        print(
            f"{class_id:>7} "
            f"{train_count:>8} "
            f"{val_count:>8} "
            f"{test_count:>8} "
            f"{train_count + val_count + test_count:>8}"
        )
    print(
        f"{'total':>7} "
        f"{split_totals['train']:>8} "
        f"{split_totals['val']:>8} "
        f"{split_totals['test']:>8} "
        f"{sum(split_totals.values()):>8}\n"
    )


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    data_loader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.to(device)
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for x, y in data_loader:
        x = x.to(device).float()
        y = y.to(device).long()
        logits = model(x)
        predicted = logits.argmax(dim=1)
        y_true.extend(int(value) for value in y.detach().cpu().tolist())
        y_pred.extend(int(value) for value in predicted.detach().cpu().tolist())
    return y_true, y_pred


def classification_report_rows(
    y_true: list[int],
    y_pred: list[int],
    class_ids: list[int],
) -> tuple[list[dict[str, float]], dict[str, float]]:
    rows = []
    total_support = len(y_true)
    total_correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)

    for class_id in class_ids:
        tp = sum(
            1
            for true, pred in zip(y_true, y_pred)
            if true == class_id and pred == class_id
        )
        fp = sum(
            1
            for true, pred in zip(y_true, y_pred)
            if true != class_id and pred == class_id
        )
        fn = sum(
            1
            for true, pred in zip(y_true, y_pred)
            if true == class_id and pred != class_id
        )
        support = sum(1 for true in y_true if true == class_id)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        rows.append(
            {
                "class": class_id,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    summary = {
        "accuracy": total_correct / max(total_support, 1),
        "macro_precision": sum(row["precision"] for row in rows) / max(len(rows), 1),
        "macro_recall": sum(row["recall"] for row in rows) / max(len(rows), 1),
        "macro_f1": sum(row["f1"] for row in rows) / max(len(rows), 1),
        "weighted_precision": sum(row["precision"] * row["support"] for row in rows)
        / max(total_support, 1),
        "weighted_recall": sum(row["recall"] * row["support"] for row in rows)
        / max(total_support, 1),
        "weighted_f1": sum(row["f1"] * row["support"] for row in rows)
        / max(total_support, 1),
        "support": total_support,
    }
    return rows, summary


def print_classification_report(
    split_name: str,
    y_true: list[int],
    y_pred: list[int],
    class_ids: list[int],
) -> None:
    rows, summary = classification_report_rows(y_true, y_pred, class_ids)
    print(f"\nClassification report [{split_name}]:")
    print(
        f"{'class':>12} "
        f"{'precision':>10} "
        f"{'recall':>10} "
        f"{'f1-score':>10} "
        f"{'support':>10}"
    )
    for row in rows:
        print(
            f"{int(row['class']):>12} "
            f"{row['precision']:>10.4f} "
            f"{row['recall']:>10.4f} "
            f"{row['f1']:>10.4f} "
            f"{int(row['support']):>10}"
        )
    print(
        f"{'accuracy':>12} {'':>10} {'':>10} "
        f"{summary['accuracy']:>10.4f} "
        f"{int(summary['support']):>10}"
    )
    print(
        f"{'macro avg':>12} "
        f"{summary['macro_precision']:>10.4f} "
        f"{summary['macro_recall']:>10.4f} "
        f"{summary['macro_f1']:>10.4f} "
        f"{int(summary['support']):>10}"
    )
    print(
        f"{'weighted avg':>12} "
        f"{summary['weighted_precision']:>10.4f} "
        f"{summary['weighted_recall']:>10.4f} "
        f"{summary['weighted_f1']:>10.4f} "
        f"{int(summary['support']):>10}"
    )


def print_all_classification_reports(
    model: nn.Module,
    data: BBoxSequenceDataBundle,
    device: torch.device,
    num_classes: int,
) -> None:
    class_ids = class_ids_for_report(data, num_classes)
    split_loaders = (
        ("train", data.train_eval_loader or data.train_loader),
        ("val", data.val_loader),
        ("test", data.test_loader),
    )
    for split_name, data_loader in split_loaders:
        if data_loader is None:
            continue
        y_true, y_pred = collect_predictions(model, data_loader, device)
        print_classification_report(split_name, y_true, y_pred, class_ids)


def checkpoint_payload(
    model: FrozenVitPoseEmbeddingLSTMClassifier,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "model_state_dict": model.trainable_state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "args": args_for_json(args),
    }


def save_checkpoint(
    model: FrozenVitPoseEmbeddingLSTMClassifier,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, args, epoch, metrics), output_path)


def load_checkpoint_file(path: Path, device: torch.device) -> object:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_trainable_state(checkpoint: object) -> dict[str, object]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(checkpoint)!r}")
    for key in ("model_state_dict", "model_state", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def load_trainable_checkpoint(
    model: FrozenVitPoseEmbeddingLSTMClassifier,
    path: Path,
    device: torch.device,
    strict: bool = True,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = load_checkpoint_file(path, device)
    state = extract_trainable_state(checkpoint)
    model.load_trainable_state_dict(state, strict=strict)
    print(f"Loaded trainable checkpoint: {path}")


def write_history(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def args_for_json(args: argparse.Namespace) -> dict[str, object]:
    data = {}
    for key, value in vars(args).items():
        data[key] = str(value) if isinstance(value, Path) else value
    return data


def write_metadata(
    args: argparse.Namespace,
    data: BBoxSequenceDataBundle,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "args": args_for_json(args),
        "total_sequences": len(data.sequences),
        "train_sequences": len(data.train_sequences),
        "val_sequences": len(data.val_sequences),
        "test_sequences": len(data.test_sequences),
        "class_counts": dict(Counter(item.label for item in data.sequences)),
        "split_class_counts": {
            "train": dict(Counter(item.label for item in data.train_sequences)),
            "val": dict(Counter(item.label for item in data.val_sequences)),
            "test": dict(Counter(item.label for item in data.test_sequences)),
        },
    }
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train_model(
    model: FrozenVitPoseEmbeddingLSTMClassifier,
    data: BBoxSequenceDataBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if not data.train_sequences or data.train_loader is None:
        raise RuntimeError("Train split is empty.")

    print_split_class_distribution(data, args.num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            train_loader=data.train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=args.grad_clip,
            show_progress=not args.no_progress,
            progress_desc=f"Epoch {epoch:03d}/{args.epochs:03d} train",
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
        }

        if data.val_loader is not None:
            val_metrics = evaluate_model(
                model=model,
                data_loader=data.val_loader,
                criterion=criterion,
                device=device,
                show_progress=not args.no_progress,
                progress_desc=f"Epoch {epoch:03d}/{args.epochs:03d} val",
            )
            row["val_loss"] = val_metrics["loss"]
            row["val_accuracy"] = val_metrics["accuracy"]
            if row["val_loss"] < best_val_loss:
                best_val_loss = row["val_loss"]
                save_checkpoint(model, args, epoch, row, args.checkpoint_path)

        history.append(row)
        message = (
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f}"
        )
        if "val_loss" in row:
            message += f" val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.4f}"
        print(message)

    save_checkpoint(model, args, args.epochs, history[-1], args.final_checkpoint_path)
    write_history(history, args.history_csv)
    write_metadata(args, data, args.metadata_json)

    report_checkpoint = args.checkpoint_path if args.checkpoint_path.is_file() else args.final_checkpoint_path
    load_trainable_checkpoint(model, report_checkpoint, device)
    if data.test_loader is not None:
        test_metrics = evaluate_model(model, data.test_loader, criterion=criterion, device=device)
        print(
            "Test metrics: "
            f"loss={test_metrics['loss']:.4f} accuracy={test_metrics['accuracy']:.4f}"
        )
    print(f"Classification reports checkpoint: {report_checkpoint}")
    print_all_classification_reports(model, data, device, args.num_classes)
    print(f"Best checkpoint: {args.checkpoint_path}")
    print(f"Final checkpoint: {args.final_checkpoint_path}")
    print(f"Training history: {args.history_csv}")
    print(f"Training metadata: {args.metadata_json}")


@torch.no_grad()
def run_inference(
    model: FrozenVitPoseEmbeddingLSTMClassifier,
    data: BBoxSequenceDataBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    checkpoint_path = args.inference_checkpoint or args.checkpoint_path
    if not checkpoint_path.is_file():
        checkpoint_path = args.final_checkpoint_path
    load_trainable_checkpoint(model, checkpoint_path, device)
    model.to(device)
    model.eval()

    rows = []
    sequence_index = 0
    for x, y in data.inference_loader:
        x = x.to(device).float()
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = probabilities.max(dim=1)
        for batch_index in range(predicted.size(0)):
            item = data.sequences[sequence_index]
            rows.append(
                {
                    "sequence_index": sequence_index,
                    "group_key": item.group_key,
                    "label": int(y[batch_index].item()),
                    "prediction": int(predicted[batch_index].detach().cpu().item()),
                    "confidence": float(confidence[batch_index].detach().cpu().item()),
                    "crop_paths": "|".join(str(path) for path in item.crop_paths),
                }
            )
            sequence_index += 1

    args.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Wrote predictions: {args.predictions_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/infer DETR crop -> frozen ViT embedding -> LSTM classifier."
    )
    parser.add_argument("--mode", choices=("train", "infer"), default="train")
    parser.add_argument(
        "--sequence-data",
        type=Path,
        default=SCRIPT_DIR / "bbox_sequence_data.json",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument(
        "--image-size",
        type=parse_image_size,
        default=parse_image_size("192x256"),
        help="Crop resize as WIDTHxHEIGHT before frozen ViT. Default: 192x256.",
    )
    parser.add_argument("--vit-model", default=DEFAULT_VIT_MODEL)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--num-classes", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "detr_vit_lstm_best.pt",
    )
    parser.add_argument(
        "--final-checkpoint-path",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "detr_vit_lstm_final.pt",
    )
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-non-strict", action="store_true")
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "detr_vit_lstm_history.csv",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "detr_vit_lstm_metadata.json",
    )
    parser.add_argument("--inference-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "detr_vit_lstm_predictions.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    args.sequence_data = args.sequence_data.resolve()
    args.checkpoint_path = args.checkpoint_path.resolve()
    args.final_checkpoint_path = args.final_checkpoint_path.resolve()
    args.history_csv = args.history_csv.resolve()
    args.metadata_json = args.metadata_json.resolve()
    args.predictions_csv = args.predictions_csv.resolve()
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = args.resume_checkpoint.resolve()
    if args.inference_checkpoint is not None:
        args.inference_checkpoint = args.inference_checkpoint.resolve()

    device = choose_device(args.device)
    data = load_bbox_sequence_data(
        sequence_data_path=args.sequence_data,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if not data.sequences:
        raise RuntimeError("No bbox crop sequences are available.")
    args.sequence_length = len(data.sequences[0].crop_paths)
    if any(len(item.crop_paths) != args.sequence_length for item in data.sequences):
        raise RuntimeError("Sequence data contains mixed sequence lengths.")

    print(f"Using device: {device}")
    print(f"Using sequence data from {args.sequence_data}")
    print(
        f"Built {len(data.sequences)} sequences: "
        f"train={len(data.train_sequences)} "
        f"val={len(data.val_sequences)} "
        f"test={len(data.test_sequences)}"
    )
    print(f"Class counts: {dict(Counter(item.label for item in data.sequences))}")

    model = FrozenVitPoseEmbeddingLSTMClassifier(
        vit_model_name=args.vit_model,
        allow_download=args.allow_download,
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    )
    if args.resume_checkpoint is not None:
        load_trainable_checkpoint(
            model,
            args.resume_checkpoint,
            device,
            strict=not args.resume_non_strict,
        )

    if args.mode == "train":
        train_model(model, data, args, device)
    else:
        run_inference(model, data, args, device)


if __name__ == "__main__":
    main()
