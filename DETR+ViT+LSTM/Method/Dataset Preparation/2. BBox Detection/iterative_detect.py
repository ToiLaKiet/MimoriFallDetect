from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from yolo_inference import DEFAULT_YOLO_MODEL, detect_largest_person_bbox, load_yolo_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DetectorFn = Callable[..., tuple[float, float, float, float] | None]
ImageSizeArg = int | tuple[int, int]
CameraRoi = tuple[int, int, int, int]
DEFAULT_CAMERA_ROIS: dict[str, CameraRoi] = {
    "Camera1": (6, 115, 595, 479), # x1, y1, x2, y2 for cropping to the area where the person is expected to be in Camera1 views. Adjust as needed based on actual camera setup and field of view.
    "Camera2": (142, 1, 485, 479), # x1, y1, x2, y2 for cropping to the area where the person is expected to be in Camera2 views. Adjust as needed based on actual camera setup and field of view.
}
DEFAULT_CAMERA_IMGSZ: dict[str, ImageSizeArg] = {
    "Camera1": (589, 364),
    "Camera2": (343, 478),
}


def parse_imgsz(value: str | None) -> ImageSizeArg | None:
    """Parse Ultralytics imgsz from 640, 640x480, or 640x480x3."""

    if value is None:
        return None

    text = str(value).strip().lower()
    if text in {"", "none", "auto"}:
        return None

    normalized = text.replace("×", "x").replace(",", "x")
    if "x" not in normalized:
        try:
            imgsz = int(normalized)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid --imgsz value: {value!r}. Use 640, 640x480, or 640x480x3."
            ) from exc
        if imgsz <= 0:
            raise argparse.ArgumentTypeError("--imgsz must be positive.")
        return imgsz

    parts = [part.strip() for part in normalized.split("x") if part.strip()]
    if len(parts) not in {2, 3}:
        raise argparse.ArgumentTypeError(
            f"Invalid --imgsz value: {value!r}. Use 640, 640x480, or 640x480x3."
        )

    try:
        dims = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --imgsz value: {value!r}. Dimensions must be integers."
        ) from exc

    if any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("--imgsz dimensions must be positive.")
    if len(dims) == 3 and dims[2] not in {1, 3, 4}:
        raise argparse.ArgumentTypeError(
            "--imgsz channel dimension must be 1, 3, or 4 when provided."
        )

    return dims[0], dims[1]


def parse_camera_imgsz(value: str) -> tuple[str, ImageSizeArg]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Use CAMERA=IMGSZ, for example Camera1=589x364."
        )

    camera_name, imgsz_text = value.split("=", 1)
    camera_name = camera_name.strip()
    if not camera_name:
        raise argparse.ArgumentTypeError("Camera name cannot be empty.")

    imgsz = parse_imgsz(imgsz_text.strip())
    if imgsz is None:
        raise argparse.ArgumentTypeError("Camera imgsz cannot be none/auto.")
    return camera_name, imgsz


def load_python_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Python module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_detector_backend(
    detector: str,
    model_path: str | Path | None,
) -> tuple[Any, DetectorFn, str | Path]:
    """Load one detector backend and return model plus bbox inference function."""

    if detector == "yolo":
        resolved_model_path = model_path or DEFAULT_YOLO_MODEL
        return (
            load_yolo_model(resolved_model_path),
            detect_largest_person_bbox,
            resolved_model_path,
        )

    if detector == "rtdetr-x":
        backend_path = Path(__file__).with_name("rtdetr-x_inference.py")
        backend = load_python_file("rtdetr_x_inference", backend_path)
        resolved_model_path = model_path or backend.DEFAULT_RTDETR_MODEL
        return (
            backend.load_rtdetr_model(resolved_model_path),
            backend.detect_largest_person_bbox,
            resolved_model_path,
        )

    raise ValueError(f"Unsupported detector: {detector}")


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


def normalize_camera_name(camera_name: str) -> str:
    return "".join(char.lower() for char in camera_name if char.isalnum())


def roi_from_entry(entry: Any) -> CameraRoi:
    if isinstance(entry, dict):
        if "xyxy" in entry:
            values = entry["xyxy"]
        elif all(key in entry for key in ("x1", "y1", "x2", "y2")):
            values = [entry["x1"], entry["y1"], entry["x2"], entry["y2"]]
        elif all(key in entry for key in ("x", "y", "width", "height")):
            x1 = entry["x"]
            y1 = entry["y"]
            values = [x1, y1, x1 + entry["width"], y1 + entry["height"]]
        else:
            raise ValueError(f"Unsupported ROI entry: {entry!r}")
    else:
        values = entry

    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"ROI must be a 4-value xyxy list, got: {entry!r}")

    x1, y1, x2, y2 = (int(round(float(value))) for value in values)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid ROI coordinates: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def imgsz_from_entry(entry: Any) -> ImageSizeArg | None:
    if isinstance(entry, dict):
        if "imgsz" in entry:
            return imgsz_from_entry(entry["imgsz"])
        if all(key in entry for key in ("width", "height")):
            return int(entry["width"]), int(entry["height"])
        if "xyxy" in entry:
            x1, y1, x2, y2 = roi_from_entry(entry["xyxy"])
            return x2 - x1, y2 - y1
        if all(key in entry for key in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = roi_from_entry(entry)
            return x2 - x1, y2 - y1
        if all(key in entry for key in ("x", "y", "width", "height")):
            return int(entry["width"]), int(entry["height"])
        raise ValueError(f"Unsupported camera imgsz entry: {entry!r}")

    if isinstance(entry, int):
        return entry
    if isinstance(entry, str):
        return parse_imgsz(entry)
    if isinstance(entry, (list, tuple)):
        if len(entry) == 1:
            return int(entry[0])
        if len(entry) in {2, 3}:
            dims = [int(value) for value in entry]
            if any(dim <= 0 for dim in dims):
                raise ValueError(f"Invalid camera imgsz entry: {entry!r}")
            return dims[0], dims[1]

    raise ValueError(f"Unsupported camera imgsz entry: {entry!r}")


def load_camera_rois(config_path: Path | None) -> dict[str, CameraRoi]:
    if config_path is None:
        return dict(DEFAULT_CAMERA_ROIS)

    with config_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    roi_entries = data.get("rois", data) if isinstance(data, dict) else data
    if not isinstance(roi_entries, dict):
        raise ValueError("ROI config must be a JSON object.")

    return {
        str(camera_name): roi_from_entry(entry)
        for camera_name, entry in roi_entries.items()
    }


def load_camera_imgszs(
    config_path: Path | None,
    overrides: list[tuple[str, ImageSizeArg]] | None,
) -> dict[str, ImageSizeArg]:
    camera_imgszs = dict(DEFAULT_CAMERA_IMGSZ)

    if config_path is not None:
        with config_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        imgsz_entries = (
            data.get("imgsz", data.get("rois", data))
            if isinstance(data, dict)
            else data
        )
        if not isinstance(imgsz_entries, dict):
            raise ValueError("Camera imgsz config must be a JSON object.")

        camera_imgszs = {}
        for camera_name, entry in imgsz_entries.items():
            imgsz = imgsz_from_entry(entry)
            if imgsz is not None:
                camera_imgszs[str(camera_name)] = imgsz

    for camera_name, imgsz in overrides or []:
        camera_imgszs[camera_name] = imgsz

    return camera_imgszs


def camera_roi_for_image(
    image_path: Path,
    image_dir: Path,
    camera_rois: dict[str, CameraRoi] | None,
) -> CameraRoi | None:
    if not camera_rois:
        return None

    relative_parts = image_path.relative_to(image_dir).parts
    if len(relative_parts) < 4:
        return None

    camera_name = relative_parts[3]
    if camera_name in camera_rois:
        return camera_rois[camera_name]

    normalized_rois = {
        normalize_camera_name(name): roi
        for name, roi in camera_rois.items()
    }
    return normalized_rois.get(normalize_camera_name(camera_name))


def camera_imgsz_for_image(
    image_path: Path,
    image_dir: Path,
    camera_imgszs: dict[str, ImageSizeArg] | None,
) -> ImageSizeArg | None:
    if not camera_imgszs:
        return None

    relative_parts = image_path.relative_to(image_dir).parts
    if len(relative_parts) < 4:
        return None

    camera_name = relative_parts[3]
    if camera_name in camera_imgszs:
        return camera_imgszs[camera_name]

    normalized_imgszs = {
        normalize_camera_name(name): imgsz
        for name, imgsz in camera_imgszs.items()
    }
    return normalized_imgszs.get(normalize_camera_name(camera_name))


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


def offset_bbox(
    bbox: tuple[float, float, float, float] | None,
    x_offset: int,
    y_offset: int,
) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    return (
        x1 + x_offset,
        y1 + y_offset,
        x2 + x_offset,
        y2 + y_offset,
    )


def build_detector_input(
    image,
    roi: CameraRoi | None,
) -> tuple[Any, tuple[int, int]]:
    if roi is None:
        return image, (0, 0)

    crop_box = clamp_bbox_to_image(roi, image.width, image.height)
    if crop_box is None:
        return image, (0, 0)

    left, top, right, bottom = crop_box
    return image.crop((left, top, right, bottom)).copy(), (left, top)


def save_crop_from_bbox(
    image_path: Path,
    output_path: Path,
    bbox: tuple[float, float, float, float] | None,
    overwrite: bool,
) -> str:
    """Save one crop from an already selected bbox."""

    if output_path.exists() and not overwrite:
        return "existing"
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


def crop_person_image(
    image_path: Path,
    output_path: Path,
    model,
    detect_bbox: DetectorFn,
    conf: float,
    iou: float,
    imgsz: ImageSizeArg | None,
    device: str | int | None,
    overwrite: bool,
    roi: CameraRoi | None = None,
) -> str:
    """Detect largest person in one image, crop it, and save without resizing."""

    if output_path.exists() and not overwrite:
        return "existing"

    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
        detector_input, (x_offset, y_offset) = build_detector_input(image, roi)

        bbox = detect_bbox(
            image=detector_input,
            model=model,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
        )

    bbox = offset_bbox(bbox, x_offset, y_offset)
    return save_crop_from_bbox(
        image_path=image_path,
        output_path=output_path,
        bbox=bbox,
        overwrite=overwrite,
    )


def update_stats(stats: dict[str, int], status: str) -> None:
    stats["seen"] += 1
    if status not in stats:
        raise RuntimeError(f"Unknown crop status: {status}")
    stats[status] += 1


def print_progress(prefix: str, index: int, total: int, stats: dict[str, int]) -> None:
    print(
        f"{prefix} "
        f"{index}/{total} | "
        f"cropped={stats['cropped']} "
        f"existing={stats['existing']} "
        f"missed={stats['missed']}"
    )


def crop_dataset(
    image_dir: Path,
    output_dir: Path,
    detector: str = "yolo",
    model_path: str | Path | None = None,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: ImageSizeArg | None = None,
    device: str | int | None = None,
    overwrite: bool = False,
    camera_rois: dict[str, CameraRoi] | None = None,
    camera_imgszs: dict[str, ImageSizeArg] | None = None,
    limit: int = 0,
    log_every: int = 100,
) -> dict[str, int]:
    """Crop person images into output_dir while preserving input folder structure."""

    image_dir = image_dir.resolve()
    output_dir = output_dir.resolve()
    image_paths = list(iter_timestamp_images(image_dir, limit=limit))
    if not image_paths:
        return {"seen": 0, "cropped": 0, "existing": 0, "missed": 0}

    model, detect_bbox, resolved_model_path = load_detector_backend(
        detector=detector,
        model_path=model_path,
    )
    print(f"Detector: {detector}")
    print(f"Model: {resolved_model_path}")
    if camera_rois:
        print(f"Camera ROI: enabled ({len(camera_rois)} camera(s))")
    if camera_imgszs:
        print(f"Camera imgsz: enabled ({len(camera_imgszs)} camera(s))")

    stats = {
        "seen": 0,
        "cropped": 0,
        "existing": 0,
        "missed": 0,
    }

    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(image_dir)
        output_path = output_dir / relative_path
        roi = camera_roi_for_image(
            image_path=image_path,
            image_dir=image_dir,
            camera_rois=camera_rois,
        )
        image_imgsz = (
            camera_imgsz_for_image(
                image_path=image_path,
                image_dir=image_dir,
                camera_imgszs=camera_imgszs,
            )
            or imgsz
        )

        status = crop_person_image(
            image_path=image_path,
            output_path=output_path,
            model=model,
            detect_bbox=detect_bbox,
            conf=conf,
            iou=iou,
            imgsz=image_imgsz,
            device=device,
            overwrite=overwrite,
            roi=roi,
        )

        update_stats(stats, status)

        if index == len(image_paths) or index % log_every == 0:
            print_progress("BBox crop", index, len(image_paths), stats)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect person bboxes and crop images while preserving "
            "Subject/Activity/Trial/Camera/Timestamp.png structure."
        )
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--detector",
        choices=("yolo", "rtdetr-x"),
        default="yolo",
        help="Detection backend. Default: yolo.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Model .pt path or Ultralytics model name. Defaults to yolo26x.pt "
            "for YOLO and rtdetr-x.pt for RT-DETR-X."
        ),
    )
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--imgsz",
        type=parse_imgsz,
        default=None,
        help=(
            "Ultralytics inference image size. Accepts 640, 640x480, "
            "or 640x480x3; channel is ignored."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--camera-roi-config",
        type=Path,
        default=None,
        help=(
            "Optional camera ROI JSON. Supports the format generated by "
            "select_camera_rois.py. If omitted, built-in Camera1/Camera2 ROIs "
            "are used."
        ),
    )
    parser.add_argument(
        "--no-camera-roi",
        action="store_true",
        help="Disable ROI pre-crop before detector inference.",
    )
    parser.add_argument(
        "--camera-imgsz-config",
        type=Path,
        default=None,
        help=(
            "Optional camera imgsz JSON. If omitted, built-in Camera1/Camera2 "
            "imgsz values are used. ROI JSON is also accepted and converted to "
            "ROI width x height."
        ),
    )
    parser.add_argument(
        "--camera-imgsz",
        action="append",
        type=parse_camera_imgsz,
        default=None,
        help=(
            "Override one camera imgsz as CAMERA=IMGSZ, e.g. Camera1=589x364. "
            "Repeat for multiple cameras."
        ),
    )
    parser.add_argument(
        "--no-camera-imgsz",
        action="store_true",
        help="Disable camera-specific imgsz and use only --imgsz/default model size.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image_dir.is_dir():
        raise ValueError(f"Image directory does not exist: {args.image_dir}")
    args.log_every = max(1, args.log_every)
    camera_rois = (
        None
        if args.no_camera_roi
        else load_camera_rois(args.camera_roi_config)
    )
    camera_imgszs = (
        None
        if args.no_camera_imgsz
        else load_camera_imgszs(
            config_path=args.camera_imgsz_config or args.camera_roi_config,
            overrides=args.camera_imgsz,
        )
    )

    stats = crop_dataset(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        detector=args.detector,
        model_path=args.model_path,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        overwrite=args.overwrite,
        camera_rois=camera_rois,
        camera_imgszs=camera_imgszs,
        limit=args.limit,
        log_every=args.log_every,
    )
    print(f"Done. Output directory: {args.output_dir.resolve()}")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
