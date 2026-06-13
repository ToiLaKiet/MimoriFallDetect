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


def group_images_by_camera(image_dir: Path, limit: int = 0) -> list[tuple[Path, list[Path]]]:
    """Return sorted timestamp images grouped by Subject/Activity/Trial/Camera."""

    groups: dict[Path, list[Path]] = {}
    for image_path in iter_timestamp_images(image_dir, limit=limit):
        groups.setdefault(image_path.parent, []).append(image_path)

    return [
        (camera_dir, sorted(paths))
        for camera_dir, paths in sorted(groups.items())
    ]


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
) -> str:
    """Detect largest person in one image, crop it, and save without resizing."""

    if output_path.exists() and not overwrite:
        return "existing"

    bbox = detect_bbox(
        image=image_path,
        model=model,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
    )
    return save_crop_from_bbox(
        image_path=image_path,
        output_path=output_path,
        bbox=bbox,
        overwrite=overwrite,
    )


def select_tracked_person_bbox(
    result: Any,
    target_track_id: int | None,
) -> tuple[tuple[float, float, float, float] | None, int | None]:
    """Choose the current target track bbox, falling back to the largest tracked person."""
    # Nếu như mà không có định vị được box nào thì trả về None và target_track_id để qua bức ảnh kế tiếp
    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None or len(boxes_obj) == 0:
        return None, target_track_id 
    
    # Chuyển cái boxes tìm được đó về dạng numpy
    boxes = boxes_obj.xyxy.detach().cpu().numpy() 
    # Lấy track_ids là id của các bounding box định vị được.
    # Cơ chế sắp xếp boxes của Ultralytics Tracker sẽ đảm bảo rằng track_ids[i] sẽ là ID của boxes[i]. Tuy nhiên, không phải lúc nào tracker cũng sẽ gán ID cho tất cả các box, đặc biệt là trong khung hình đầu tiên khi tracker mới bắt đầu theo dõi. Do đó, chúng ta sử dụng getattr để lấy thuộc tính .id của boxes_obj nếu nó tồn tại, và nếu không tồn tại thì ids_tensor sẽ là None. Nếu ids_tensor không phải là None, chúng ta sẽ chuyển nó về numpy array để dễ dàng xử lý sau này.
    
    ids_tensor = getattr(boxes_obj, "id", None) # The .id attribute may not exist if the tracker did not assign IDs to the boxes, so we use getattr to avoid an AttributeError. If ids_tensor is not None, we convert it to a numpy array of integers for easier processing.
    # Nếu có tồn tại ids_tensor thì chuyển nó về numpy array, còn nếu không có thì track_ids sẽ là None. Sau đó, nếu target_track_id đã được cung cấp từ trước, chúng ta sẽ tìm trong track_ids xem có track ID nào trùng với target_track_id không. Nếu có, chúng ta sẽ chọn box tương ứng với track ID đó và trả về box cùng với target_track_id cũ cho lần theo dõi tiếp theo. Nếu không có track ID nào trùng, hoặc nếu target_track_id là None, thì chúng ta sẽ tính diện tích của tất cả các box và chọn box có diện tích lớn nhất làm box được theo dõi hiện tại. Chúng ta cũng sẽ cập nhật target_track_id thành track ID của box lớn nhất này (nếu track_ids không phải là None) để sử dụng cho lần theo dõi tiếp theo.
    
    track_ids = None
    if ids_tensor is not None:
        track_ids = ids_tensor.detach().cpu().numpy().astype(int)
    
    # Nếu target_track_id đã được cung cấp từ trước, thì đoạn code sẽ vào đây và tìm trong track_ids xem có track ID nào trùng với target_track_id không. Nếu có, nó sẽ chọn box tương ứng với track ID đó và trả về box cùng với target_track_id cũ cho lần theo dõi tiếp theo. Nếu không có track ID nào trùng, hoặc nếu target_track_id là None, thì đoạn code sẽ tính diện tích của tất cả các box và chọn box có diện tích lớn nhất làm box được theo dõi hiện tại. Nó cũng sẽ cập nhật target_track_id thành track ID của box lớn nhất này (nếu track_ids không phải là None) để sử dụng cho lần theo dõi tiếp theo.
    if target_track_id is not None and track_ids is not None:
        matches = [
            index
            for index, track_id in enumerate(track_ids)
            if track_id == target_track_id
        ]
        if matches:
            box = boxes[matches[0]] 
            return tuple(float(value) for value in box), target_track_id

    widths = [max(0.0, float(box[2] - box[0])) for box in boxes]
    heights = [max(0.0, float(box[3] - box[1])) for box in boxes]
    largest_index = max(range(len(boxes)), key=lambda index: widths[index] * heights[index])
    if track_ids is not None:
        target_track_id = int(track_ids[largest_index])

    return tuple(float(value) for value in boxes[largest_index]), target_track_id


def track_person_bbox(
    model,
    image_path: Path,
    tracker: str,
    persist: bool,
    target_track_id: int | None,
    conf: float,
    iou: float,
    imgsz: ImageSizeArg | None,
    device: str | int | None,
) -> tuple[tuple[float, float, float, float] | None, int | None]:
    """Run Ultralytics tracking on one frame and return the selected person bbox."""

    track_kwargs: dict[str, Any] = {
        "source": str(image_path),
        "tracker": tracker,
        "persist": persist,
        "classes": [0],
        "conf": conf,
        "iou": iou,
        "verbose": False,
    }
    if imgsz is not None:
        track_kwargs["imgsz"] = imgsz
    if device is not None:
        track_kwargs["device"] = device

    results = model.track(**track_kwargs)
    if not results:
        return None, target_track_id
    return select_tracked_person_bbox(results[0], target_track_id)


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
    track: bool = False,
    tracker: str = "bytetrack.yaml",
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
    if track:
        print(f"Tracking: enabled ({tracker})")

    stats = {
        "seen": 0,
        "cropped": 0,
        "existing": 0,
        "missed": 0,
    }

    if track:
        processed = 0
        camera_groups = group_images_by_camera(image_dir, limit=limit)
        total_images = sum(len(paths) for _, paths in camera_groups)

        for group, image_paths_in_camera in camera_groups:
            target_track_id = None
            for frame_index, image_path in enumerate(image_paths_in_camera):
                relative_path = image_path.relative_to(image_dir)
                output_path = output_dir / relative_path
                if target_track_id is not None:
                    print(f'Processing {group} with tracking {target_track_id}', end=' ')
                
                bbox, target_track_id = track_person_bbox(
                    model=model,
                    image_path=image_path,
                    tracker=tracker,
                    persist=frame_index > 0,
                    target_track_id=target_track_id,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    device=device,
                )
                status = save_crop_from_bbox(
                    image_path=image_path,
                    output_path=output_path,
                    bbox=bbox,
                    overwrite=overwrite,
                )

                processed += 1
                update_stats(stats, status)
                if processed == total_images or processed % log_every == 0:
                    print_progress("Track crop", processed, total_images, stats)

        return stats

    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(image_dir)
        output_path = output_dir / relative_path

        status = crop_person_image(
            image_path=image_path,
            output_path=output_path,
            model=model,
            detect_bbox=detect_bbox,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            overwrite=overwrite,
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
        "--track",
        action="store_true",
        help=(
            "Use Ultralytics tracking with ByteTrack. Tracker is reset for each "
            "Subject/Activity/Trial/Camera folder."
        ),
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker config. Default: bytetrack.yaml.",
    )
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
        imgsz=args.imgsz,
        device=args.device,
        overwrite=args.overwrite,
        track=args.track,
        tracker=args.tracker,
        limit=args.limit,
        log_every=args.log_every,
    )
    print(f"Done. Output directory: {args.output_dir.resolve()}")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
