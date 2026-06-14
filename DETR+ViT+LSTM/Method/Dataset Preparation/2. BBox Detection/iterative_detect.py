from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from yolo_inference import DEFAULT_YOLO_MODEL, detect_largest_person_bbox, load_yolo_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DetectorFn = Callable[..., tuple[float, float, float, float] | None]
ImageSizeArg = int | tuple[int, int]
CameraRoi = tuple[int, int, int, int]
DEFAULT_CAMERA_ROIS: dict[str, CameraRoi] = {
    "Camera1": (0, 0, 640, 480),
    "Camera2": (3, 2, 591, 479),
}
DEFAULT_CAMERA_IMGSZ: dict[str, ImageSizeArg] = {
    "Camera1": (384, 608),  # height, width
    "Camera2": (480, 608),  # height, width
}


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
    return "".join(char.lower() for char in camera_name if char.isalnum()) # Hàm normalize_camera_name nhận một chuỗi camera_name và trả về một chuỗi mới đã được chuẩn hóa. Cụ thể, nó loại bỏ tất cả các ký tự không phải là chữ cái hoặc số và chuyển tất cả các ký tự còn lại thành chữ thường. Ví dụ, nếu camera_name là "Camera 1", thì normalize_camera_name sẽ trả về "camera1". Điều này giúp đảm bảo rằng việc so sánh tên camera sẽ không bị ảnh hưởng bởi sự khác biệt về định dạng hoặc kiểu chữ.


def camera_roi_for_image(
    image_path: Path,
    image_dir: Path,
) -> CameraRoi:
    relative_parts = image_path.relative_to(image_dir).parts
    if len(relative_parts) < 4:
        raise ValueError(
            f"Cannot infer camera folder from image path: {image_path}"
        )

    camera_name = relative_parts[3]
    if camera_name in DEFAULT_CAMERA_ROIS:
        return DEFAULT_CAMERA_ROIS[camera_name]

    normalized_rois = {
        normalize_camera_name(name): roi
        for name, roi in DEFAULT_CAMERA_ROIS.items()
    }
    roi = normalized_rois.get(normalize_camera_name(camera_name))
    if roi is None:
        raise ValueError(
            f"No hardcoded ROI for camera folder {camera_name!r} in {image_path}"
        )
    return roi


def camera_imgsz_for_image(
    image_path: Path,
    image_dir: Path,
) -> ImageSizeArg:
    relative_parts = image_path.relative_to(image_dir).parts # cụ thể relative_parts sẽ là một tuple chứa các phần của đường dẫn tương đối từ image_dir đến image_path. Ví dụ, nếu image_dir là "/data/images" và image_path là "/data/images/Subject1/ActivityA/Trial1/Camera1/20220101_120000.png", thì relative_parts sẽ là ("Subject1", "ActivityA", "Trial1", "Camera1", "20220101_120000.png").
    if len(relative_parts) < 4:
        raise ValueError(
            f"Cannot infer camera folder from image path: {image_path}"
        )

    camera_name = relative_parts[3]
    if camera_name in DEFAULT_CAMERA_IMGSZ:
        return DEFAULT_CAMERA_IMGSZ[camera_name]

    normalized_imgszs = {
        normalize_camera_name(name): imgsz
        for name, imgsz in DEFAULT_CAMERA_IMGSZ.items()
    }
    imgsz = normalized_imgszs.get(normalize_camera_name(camera_name))
    if imgsz is None:
        raise ValueError(
            f"No hardcoded imgsz for camera folder {camera_name!r} in {image_path}"
        )
    return imgsz


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

    model, detect_bbox, resolved_model_path = load_detector_backend(
        detector=detector,
        model_path=model_path,
    )
    print(f"Detector: {detector}")
    print(f"Model: {resolved_model_path}")
    print(f"Camera ROI: hardcoded ({len(DEFAULT_CAMERA_ROIS)} camera(s))")
    print(f"Camera imgsz: hardcoded ({len(DEFAULT_CAMERA_IMGSZ)} camera(s))")

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
        )
        image_imgsz = camera_imgsz_for_image(
            image_path=image_path,
            image_dir=image_dir,
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
        detector=args.detector,
        model_path=args.model_path,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        overwrite=args.overwrite,
        limit=args.limit,
        log_every=args.log_every,
    )
    print(f"Done. Output directory: {args.output_dir.resolve()}")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
