from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

from PIL import Image, ImageEnhance

AugmentMethod = Literal["brightness", "rotate"]

AUGMENT_RATIO = 0.5
BRIGHTNESS_RANGE = (0.5, 0.8)
ROTATE_RANGE = (-20.0, 20.0)


def pick_augment_method(rng: random.Random) -> AugmentMethod:
    return rng.choice(("brightness", "rotate"))


def pick_augment_params(method: AugmentMethod, rng: random.Random) -> dict[str, float]:
    if method == "brightness":
        return {"factor": rng.uniform(*BRIGHTNESS_RANGE)}
    return {"angle": rng.uniform(*ROTATE_RANGE)}


def transform_image(
    image: Image.Image,
    method: AugmentMethod,
    params: dict[str, float],
) -> Image.Image:
    if method == "brightness":
        return ImageEnhance.Brightness(image).enhance(params["factor"])
    return image.rotate(params["angle"], resample=Image.BICUBIC, expand=False)


def write_augmented_sequence(
    source_seq_dir: Path,
    augment_seq_name: str,
    rng: random.Random,
) -> AugmentMethod:
    metadata_path = source_seq_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    method = pick_augment_method(rng)
    params = pick_augment_params(method, rng)

    dest_dir = source_seq_dir.parent / augment_seq_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    frames_meta: list[dict[str, object]] = []
    for frame in metadata["frames"]:
        frame_index = int(frame["frame_index"])
        timestamp = str(frame["timestamp"])
        source_image = source_seq_dir / str(frame["frame_file"])

        with Image.open(source_image) as image:
            augmented = transform_image(image.convert("RGB"), method, params)
            frame_file = f"{timestamp}.jpg"
            augmented.save(dest_dir / frame_file, quality=95)

        frames_meta.append(
            {
                "frame_index": frame_index,
                "frame_file": frame_file,
                "timestamp": timestamp,
                "source_rel_path": frame.get("source_rel_path", ""),
                "source_abs_path": frame.get("source_abs_path", ""),
            }
        )

    augment_metadata = {
        "Subject": metadata.get("Subject", ""),
        "Activity": metadata.get("Activity", ""),
        "Trial": metadata.get("Trial", ""),
        "Camera": metadata.get("Camera", ""),
        "fall_alert": metadata.get("fall_alert", 0),
        "sequence_name": augment_seq_name,
        "augmented_from": metadata.get("sequence_name", source_seq_dir.name),
        "augment_method": method,
        "augment_params": params,
        "frame_count": len(frames_meta),
        "frames": frames_meta,
    }
    with (dest_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(augment_metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return method
