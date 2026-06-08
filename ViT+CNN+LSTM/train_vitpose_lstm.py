#!/usr/bin/env python3
"""Train or run inference with the CNN+LSTM classifier from prepared sequences."""

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


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import SkeletonImageLSTMClassifier  # noqa: E402
from sequence_data import (  # noqa: E402
    SequenceDataBundle,
    SequenceItem,
    load_sequence_data,
    parse_image_size,
)
from utils import evaluate_model, train_model  # noqa: E402


def seed_everything(seed: int) -> None:
    """Fix random seeds for Python, NumPy, and Torch so split/training is repeatable."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda/mps into a usable torch.device with safe fallback messages."""

    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        requested = "cpu"
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        print("MPS is not available; falling back to CPU.")
        requested = "cpu"

    return torch.device(requested)


def build_model(args: argparse.Namespace) -> SkeletonImageLSTMClassifier:
    """Create the CNN+LSTM classifier using model hyperparameters from CLI args."""
    
    return SkeletonImageLSTMClassifier(
        sequence_length=args.sequence_length,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    )


def load_state_dict_file(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load a model state dict while preferring PyTorch's safer weights-only mode."""

    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def write_history(history: list[dict[str, float]], output_path: Path) -> None:
    """Write per-epoch train/validation metrics to CSV for later plotting/debugging."""

    if not history:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def args_for_json(args: argparse.Namespace) -> dict[str, object]:
    """Convert argparse values into JSON-safe metadata, including Path values."""

    data = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            data[key] = str(value)
        else:
            data[key] = value
    return data


def save_metadata(
    args: argparse.Namespace,
    sequences: list[SequenceItem],
    train_sequences: list[SequenceItem],
    val_sequences: list[SequenceItem],
    test_sequences: list[SequenceItem],
    output_path: Path,
) -> None:
    """Save training metadata: arguments, split sizes, and class distribution."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "args": args_for_json(args),
        "total_sequences": len(sequences),
        "train_sequences": len(train_sequences),
        "val_sequences": len(val_sequences),
        "test_sequences": len(test_sequences),
        "class_counts": dict(Counter(item.label for item in sequences)),
    }
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def report_sequence_data(data: SequenceDataBundle) -> None:
    """Print a short sanity report for the loaded serialized sequence dataset."""

    print(
        f"Loaded {data.trial_count} Trial/Camera groups, {data.total_inputs} frames, "
        f"matched {data.matched_frames} skeleton frames, "
        f"missing labels for {len(data.missing_labels)} frames, "
        f"missing skeleton files for {len(data.missing_skeletons)} frames, "
        f"invalid timestamps for {len(data.invalid_timestamps)} manifest rows."
    )

    if data.missing_labels:
        preview = ", ".join(path.name for path in data.missing_labels[:5])
        print(f"First missing-label images: {preview}")
    if data.missing_skeletons:
        preview = ", ".join(str(path) for path in data.missing_skeletons[:5])
        print(f"First missing skeleton files: {preview}")
    if data.invalid_timestamps:
        preview = ", ".join(data.invalid_timestamps[:5])
        print(f"First invalid timestamp rows: {preview}")


def run_train(
    model: SkeletonImageLSTMClassifier,
    data: SequenceDataBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Train the model, save checkpoints/history/metadata, then evaluate test split."""

    if not data.train_sequences:
        raise RuntimeError("Train split is empty; reduce val-split.")
    if data.train_loader is None:
        raise RuntimeError("Train loader is empty; check sequence_data.py dataset creation.")

    history = train_model(
        model=model,
        train_loader=data.train_loader,
        val_loader=data.val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        grad_clip=args.grad_clip,
        checkpoint_path=args.checkpoint_path if data.val_loader is not None else None,
        show_progress=not args.no_progress,
    )

    args.final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.final_checkpoint_path)
    write_history(history, args.history_csv)
    save_metadata(
        args=args,
        sequences=data.sequences,
        train_sequences=data.train_sequences,
        val_sequences=data.val_sequences,
        test_sequences=data.test_sequences,
        output_path=args.metadata_json,
    )

    if data.val_loader is not None:
        print(f"Best validation checkpoint: {args.checkpoint_path}")
    if data.test_loader is not None:
        if data.val_loader is not None and args.checkpoint_path.is_file():
            model.load_state_dict(load_state_dict_file(args.checkpoint_path, device))
        test_metrics = evaluate_model(model, data.test_loader, device=device)
        print(
            "Test metrics: "
            f"loss={test_metrics['loss']:.4f} "
            f"accuracy={test_metrics['accuracy']:.4f}"
        )
    print(f"Final checkpoint: {args.final_checkpoint_path}")
    print(f"Training history: {args.history_csv}")
    print(f"Training metadata: {args.metadata_json}")


@torch.no_grad()
def run_inference(
    model: SkeletonImageLSTMClassifier,
    data: SequenceDataBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Run inference over every serialized sequence and write predictions to CSV."""

    checkpoint_path = args.inference_checkpoint or args.checkpoint_path
    if not checkpoint_path.is_file():
        checkpoint_path = args.final_checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Inference checkpoint not found: {args.inference_checkpoint or args.checkpoint_path}"
        )

    model.load_state_dict(load_state_dict_file(checkpoint_path, device))
    model.to(device)
    model.eval()

    rows = []
    sequence_index = 0
    for x, y in data.inference_loader:
        x = x.to(device).float()
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = probabilities.max(dim=1)

        batch_size = predicted.size(0)
        for batch_index in range(batch_size):
            item = data.sequences[sequence_index]
            rows.append(
                {
                    "sequence_index": sequence_index,
                    "group_key": item.group_key,
                    "label": int(y[batch_index].item()),
                    "prediction": int(predicted[batch_index].detach().cpu().item()),
                    "confidence": float(confidence[batch_index].detach().cpu().item()),
                    "skeleton_paths": "|".join(str(path) for path in item.skeleton_paths),
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
    """Parse training/inference CLI args; sequence construction lives in sequence_data.py."""

    parser = argparse.ArgumentParser(
        description="Train or run inference from a serialized skeleton sequence dataset."
    )
    parser.add_argument("--mode", choices=("train", "infer"), default="train")
    parser.add_argument(
        "--sequence-data",
        type=Path,
        default=SCRIPT_DIR / "sequence_data.json",
        help="JSON file created by sequence_data.py.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Training/inference device. Default: auto.",
    )
    parser.add_argument(
        "--image-size",
        type=parse_image_size,
        default=parse_image_size("224"),
        help="Resize skeleton images as WIDTHxHEIGHT, WIDTH,HEIGHT, or one square integer.",
    )
    parser.add_argument("--num-classes", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-batch tqdm progress bars during training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "vitpose_lstm_best.pt",
        help="Best validation checkpoint path.",
    )
    parser.add_argument(
        "--final-checkpoint-path",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "vitpose_lstm_final.pt",
        help="Final checkpoint path.",
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "vitpose_lstm_history.csv",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "vitpose_lstm_metadata.json",
    )
    parser.add_argument(
        "--inference-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint for --mode infer. Defaults to checkpoint-path, then final-checkpoint-path.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "vitpose_lstm_predictions.csv",
        help="Output CSV for --mode infer.",
    )
    return parser.parse_args()


def main() -> None:
    """Load sequence_data.json, validate it, build the model, then train or infer."""

    args = parse_args()
    seed_everything(args.seed)

    args.sequence_data = args.sequence_data.resolve()
    args.checkpoint_path = args.checkpoint_path.resolve()
    args.final_checkpoint_path = args.final_checkpoint_path.resolve()
    args.history_csv = args.history_csv.resolve()
    args.metadata_json = args.metadata_json.resolve()
    args.predictions_csv = args.predictions_csv.resolve()
    if args.inference_checkpoint is not None:
        args.inference_checkpoint = args.inference_checkpoint.resolve()

    device = choose_device(args.device)
    data = load_sequence_data(
        sequence_data_path=args.sequence_data,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    report_sequence_data(data)

    if not data.sequences:
        raise RuntimeError(
            "No skeleton sequences are available; run extract_vitpose_skeletons.py first "
            "or reduce sequence length."
        )
    args.sequence_length = len(data.sequences[0].skeleton_paths)
    if any(len(item.skeleton_paths) != args.sequence_length for item in data.sequences):
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

    model = build_model(args)
    if args.mode == "train":
        run_train(model, data, args, device)
    else:
        run_inference(model, data, args, device)


if __name__ == "__main__":
    main()
