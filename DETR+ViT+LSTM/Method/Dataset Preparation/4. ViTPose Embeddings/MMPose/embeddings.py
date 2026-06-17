from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mmpose_vitpose_estimator import MMPoseEmbeddingSource, MMPoseVitPoseEstimator


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
CLASS_NAMES = ("fall", "normal")


def iter_sequence_image_paths(sequence_dir: Path) -> list[Path]:
    image_paths = [
        path
        for path in sequence_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths, key=lambda path: path.name)


def iter_dataset_images(dataset_dir: Path) -> list[tuple[str, Path]]:
    """Yield (split_name, image_path) for every frame in the dataset."""

    frames: list[tuple[str, Path]] = []
    for split_name in SPLITS:
        split_dir = dataset_dir / split_name
        if not split_dir.is_dir():
            continue

        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue

            for sequence_dir in sorted(class_dir.iterdir()):
                if not sequence_dir.is_dir():
                    continue
                for image_path in iter_sequence_image_paths(sequence_dir):
                    frames.append((split_name, image_path))

    return frames


def embedding_output_path(
    image_path: Path,
    dataset_dir: Path,
    output_dir: Path,
) -> Path:
    relative_path = image_path.relative_to(dataset_dir)
    return output_dir / relative_path.with_suffix(".npy")


def resolve_device(device_name: str | None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_dataset_embeddings(
    dataset_dir: Path,
    output_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device | None = None,
    embedding_source: MMPoseEmbeddingSource = "pre_head_gap",
    skip_existing: bool = False,
    limit: int = 0,
) -> dict[str, int]:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    if not dataset_dir.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")

    resolved_device = device or resolve_device(None)
    estimator = MMPoseVitPoseEstimator(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=resolved_device,
    )

    frames = iter_dataset_images(dataset_dir)
    if limit > 0:
        frames = frames[:limit]

    stats: Counter = Counter()
    stats["frames_total"] = len(frames)

    for split_name, image_path in frames:
        output_path = embedding_output_path(image_path, dataset_dir, output_dir)
        if skip_existing and output_path.is_file():
            stats["frames_skipped"] += 1
            continue

        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
                embedding = estimator.extract_embedding(
                    image=rgb_image,
                    source=embedding_source,
                )
        except Exception:
            stats["frames_failed"] += 1
            print(f"Failed to extract embedding: {image_path}")
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embedding)
        stats["frames_written"] += 1
        stats[f"{split_name}_written"] += 1

    return dict(stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ViTPose pre-head embeddings (MMPose extract_feat) from a sequence dataset. "
            "Each image frame_000.jpg is saved as frame_000.npy with the same "
            "relative folder structure."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Root folder containing train/, val/, and test/ splits.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination root mirroring the dataset layout with .npy files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="MMPose config (.py) for ViTPose top-down model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="MMPose checkpoint (.pth) for the config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cuda, mps, or cpu. Auto-detect when omitted.",
    )
    parser.add_argument(
        "--embedding-source",
        choices=(
            "pre_head_gap",
            "pre_head_flatten",
        ),
        default="pre_head_gap",
        help="Which pre-head feature to pool into an embedding.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip frames whose .npy output already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N frames (0 means no limit).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    stats = extract_dataset_embeddings(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
        embedding_source=args.embedding_source,
        skip_existing=args.skip_existing,
        limit=args.limit,
    )

    print(f"Dataset: {args.dataset_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Embedding source: {args.embedding_source}")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
