from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from yolo_inference import DEFAULT_YOLO_MODEL, detect_largest_person_bbox, load_yolo_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_timestamp_images(image_dir: Path, limit: int = 0):
    """Yield image files under Subject/Activity/Trial/Camera folders."""
    count = 0
    for image_path in sorted(image_dir.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_parts = image_path.relative_to(image_dir).parts
        if len(relative_parts) != 5:
            continue

        yield image_path
        count += 1
        if limit > 0 and count >= limit:
            return


def clamp_bbox_to_image(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    """Clamp xyxy bbox to image bounds and return integer crop coordinates."""

    x1, y1, x2, y2 = bbox
    left = max(0, min(image_width, int(round(x1))))
    top = max(0, min(image_height, int(round(y1))))
    right = max(0, min(image_width, int(round(x2))))
    bottom = max(0, min(image_height, int(round(y2))))

    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def crop_person_image(
    image_path: Path,
    output_path: Path,
    model,
    conf: float,
    iou: float,
    imgsz: int | None,
    device: str | int | None,
    overwrite: bool,
) -> str:
    """Detect largest person in one image, crop it, and save without resizing."""

    if output_path.exists() and not overwrite:
        return "exists"

    bbox = detect_largest_person_bbox(
        image=image_path,
        model=model,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
    )
    if bbox is None:
        return "missed"

    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
        crop_box = clamp_bbox_to_image(bbox, image.width, image.height)
        if crop_box is None:
            return "missed"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.crop(crop_box).save(output_path)
    return "cropped"


def crop_dataset(
    image_dir: Path,
    output_dir: Path,
    model_path: str | Path = DEFAULT_YOLO_MODEL,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int | None = None,
    device: str | int | None = None,
    overwrite: bool = False,
    limit: int = 0,
    log_every: int = 100,
) -> dict[str, int]:
    """Crop person images into output_dir while preserving input folder structure."""

    image_dir = image_dir.resolve()
    output_dir = output_dir.resolve()
    image_paths = list(iter_timestamp_images(image_dir, limit=limit))
    if not image_paths:
        return {"seen": 0, "cropped": 0, "existing": 0, "missed": 0}

    model = load_yolo_model(model_path)

    stats = {
        "seen": 0,
        "cropped": 0,
        "existing": 0,
        "missed": 0,
    }

    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(image_dir)
        output_path = output_dir / relative_path

        status = crop_person_image(
            image_path=image_path,
            output_path=output_path,
            model=model,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            overwrite=overwrite,
        )

        stats["seen"] += 1
        stats[status] += 1

        if index == len(image_paths) or index % log_every == 0:
            print(
                "YOLO crop "
                f"{index}/{len(image_paths)} | "
                f"cropped={stats['cropped']} "
                f"existing={stats['existing']} "
                f"missed={stats['missed']}"
            )

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect person bboxes with YOLO and crop images while preserving "
            "Subject/Activity/Trial/Camera/Timestamp.png structure."
        )
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_YOLO_MODEL))
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image_dir.is_dir():
        raise ValueError(f"Image directory does not exist: {args.image_dir}")
    args.log_every = max(1, args.log_every)

    stats = crop_dataset(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        overwrite=args.overwrite,
        limit=args.limit,
        log_every=args.log_every,
    )
    print(f"Done. Output directory: {args.output_dir.resolve()}")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
