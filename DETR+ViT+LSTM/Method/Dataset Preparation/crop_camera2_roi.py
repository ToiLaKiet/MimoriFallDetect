from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


CAMERA2_ROI = (3, 4, 407, 479)  # x1, y1, x2, y2
DEFAULT_IMAGE_NAME = "2018-07-10T12_21_21.947585.png"


def crop_camera2_roi(image_path: Path, output_path: Path | None = None) -> Path:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if output_path is None:
        output_path = image_path.with_name(f"{image_path.stem}_camera2_roi{image_path.suffix}")

    x1, y1, x2, y2 = CAMERA2_ROI
    with Image.open(image_path) as image:
        cropped = image.crop((x1, y1, x2, y2))
        cropped.save(output_path)

    return output_path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_image_path = script_dir / DEFAULT_IMAGE_NAME

    parser = argparse.ArgumentParser(
        description="Crop Camera2 ROI from an image in Dataset Preparation."
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=default_image_path,
        help=f"Input image path. Default: {default_image_path}",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output cropped image path. Default: <input_stem>_camera2_roi<suffix>",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = crop_camera2_roi(args.image_path, args.output_path)
    print(f"Saved cropped image to: {output_path}")


if __name__ == "__main__":
    main()
