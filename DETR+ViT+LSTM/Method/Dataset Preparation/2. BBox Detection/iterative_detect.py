from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from yolo_inference import DEFAULT_YOLO_MODEL, detect_largest_person_bbox, load_yolo_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DetectorFn = Callable[..., tuple[float, float, float, float] | None]
ImageSizeArg = int | tuple[int, int]
FeatureVector = np.ndarray
ReIDImageSizeArg = tuple[int, int]


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


def parse_reid_imgsz(value: str) -> ReIDImageSizeArg:
    """Parse Torchreid image size as height x width, e.g. 256x128."""

    text = str(value).strip().lower().replace("×", "x").replace(",", "x")
    parts = [part.strip() for part in text.split("x") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid --reid-imgsz value: {value!r}. Use height x width, e.g. 256x128."
        )

    try:
        height, width = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --reid-imgsz value: {value!r}. Dimensions must be integers."
        ) from exc

    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("--reid-imgsz dimensions must be positive.")
    return height, width


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


def group_images_by_subject_activity(
    image_dir: Path,
    limit: int = 0,
) -> list[tuple[Path, list[tuple[Path, list[Path]]]]]:
    """Return timestamp images grouped by Subject/Activity, then Trial/Camera."""

    groups: dict[Path, dict[Path, list[Path]]] = {}
    for image_path in iter_timestamp_images(image_dir, limit=limit):
        relative_parts = image_path.relative_to(image_dir).parts
        subject_activity_dir = image_dir / relative_parts[0] / relative_parts[1]
        camera_dir = image_path.parent
        groups.setdefault(subject_activity_dir, {}).setdefault(
            camera_dir,
            [],
        ).append(image_path)

    return [
        (
            subject_activity_dir,
            [
                (camera_dir, sorted(paths))
                for camera_dir, paths in sorted(camera_groups.items())
            ],
        )
        for subject_activity_dir, camera_groups in sorted(groups.items())
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


def bbox_area(bbox: tuple[float, float, float, float] | np.ndarray) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def resolve_reid_device(
    reid_device: str | None,
    detector_device: str | int | None,
) -> str:
    """Resolve a Torch device string for Torchreid."""

    if reid_device:
        device_text = str(reid_device)
    elif detector_device is not None:
        device_text = str(detector_device)
    else:
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device_text.isdigit():
        return f"cuda:{device_text}"
    return device_text


class ReIDFeatureExtractor:
    """Thin wrapper around Torchreid FeatureExtractor for person bbox crops."""

    def __init__(
        self,
        model_name: str,
        model_path: str | Path | None,
        device: str,
        image_size: ReIDImageSizeArg,
    ) -> None:
        try:
            from torchreid.utils import FeatureExtractor
        except ImportError as exc:
            raise ImportError(
                "Torchreid is required for OSNet ReID features. Install "
                "deep-person-reid/torchreid, then rerun with --track."
            ) from exc

        resolved_model_path = "" if model_path is None else str(model_path)
        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=resolved_model_path,
            image_size=image_size,
            device=device,
            verbose=False,
        )
        self.model_name = model_name
        self.model_path = resolved_model_path or "torchreid-pretrained"
        self.device = device
        self.image_size = image_size

    def extract(
        self,
        image_path: Path,
        bbox: tuple[float, float, float, float] | np.ndarray,
    ) -> FeatureVector | None:
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            crop_box = clamp_bbox_to_image(
                tuple(float(value) for value in bbox),
                image.width,
                image.height,
            )
            if crop_box is None:
                return None

            crop = image.crop(crop_box)

        crop_array = np.asarray(crop, dtype=np.uint8)
        feature_tensor = self.extractor(crop_array)
        feature = feature_tensor.detach().cpu().numpy()[0].astype(np.float32)
        norm = float(np.linalg.norm(feature))
        if norm <= 0.0:
            return None
        return feature / norm


def extract_bbox_feature(
    image_path: Path,
    bbox: tuple[float, float, float, float] | np.ndarray,
    reid_extractor: ReIDFeatureExtractor,
) -> FeatureVector | None:
    """Extract an OSNet/Torchreid embedding from one person bbox crop."""

    return reid_extractor.extract(image_path=image_path, bbox=bbox)


def feature_similarity(
    feature_a: FeatureVector | None,
    feature_b: FeatureVector | None,
) -> float:
    if feature_a is None or feature_b is None:
        return float("-inf")
    return float(np.dot(feature_a, feature_b))


def update_reference_feature(
    reference_feature: FeatureVector | None,
    selected_feature: FeatureVector | None,
) -> FeatureVector | None:
    if selected_feature is None:
        return reference_feature
    if reference_feature is None:
        return selected_feature

    updated_feature = 0.9 * reference_feature + 0.1 * selected_feature
    norm = float(np.linalg.norm(updated_feature))
    if norm <= 0.0:
        return reference_feature
    return updated_feature / norm


def select_initial_track_index(
    image_path: Path,
    boxes: np.ndarray,
    reference_feature: FeatureVector | None,
    reid_extractor: ReIDFeatureExtractor,
) -> tuple[int, FeatureVector | None]:
    """Choose a local track for a new Trial/Camera sequence."""

    if reference_feature is None:
        largest_index = max(range(len(boxes)), key=lambda index: bbox_area(boxes[index]))
        return largest_index, extract_bbox_feature(
            image_path,
            boxes[largest_index],
            reid_extractor,
        )

    best_index = None
    best_feature = None
    best_similarity = float("-inf")
    for index, box in enumerate(boxes):
        candidate_feature = extract_bbox_feature(image_path, box, reid_extractor)
        similarity = feature_similarity(reference_feature, candidate_feature)
        if similarity > best_similarity:
            best_index = index
            best_feature = candidate_feature
            best_similarity = similarity

    if best_index is not None:
        return best_index, best_feature

    largest_index = max(range(len(boxes)), key=lambda index: bbox_area(boxes[index]))
    return largest_index, extract_bbox_feature(
        image_path,
        boxes[largest_index],
        reid_extractor,
    )


def reset_model_trackers(model) -> None:
    """Reset Ultralytics tracker state before a new Trial/Camera sequence."""

    predictor = getattr(model, "predictor", None)
    if predictor is None:
        return

    trackers = getattr(predictor, "trackers", None) or []
    for tracker in trackers:
        reset = getattr(tracker, "reset", None)
        if callable(reset):
            reset()

    if hasattr(predictor, "vid_path"):
        predictor.vid_path = [None] * max(1, len(trackers))


def get_track_ids(boxes_obj: Any) -> list[int] | None:
    """Return real tracker IDs for boxes when Ultralytics provides them."""

    ids_tensor = getattr(boxes_obj, "id", None)
    if ids_tensor is None:
        return None

    track_ids = ids_tensor.detach().cpu().numpy().astype(int).tolist()
    if len(track_ids) != len(boxes_obj):
        return None
    return [int(track_id) for track_id in track_ids]


def select_tracked_person_bbox(
    result: Any,
    target_track_id: int | None,
    image_path: Path,
    reference_feature: FeatureVector | None,
    reid_extractor: ReIDFeatureExtractor,
) -> tuple[tuple[float, float, float, float] | None, int | None, FeatureVector | None]:
    """Choose the target track bbox for one Trial/Camera sequence."""

    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None or len(boxes_obj) == 0:
        return None, target_track_id, None

    boxes = boxes_obj.xyxy.detach().cpu().numpy()
    track_ids = get_track_ids(boxes_obj)
    if track_ids is None:
        return None, target_track_id, None

    if target_track_id is not None:
        # Giả định rằng có nhiều bounding box định vị. Nếu track_ids không có target_track_id, thì sẽ trả về None.
        matches = [
            index
            for index, track_id in enumerate(track_ids)
            if track_id == target_track_id
        ]
        if matches:
            box = boxes[matches[0]] 
            selected_feature = extract_bbox_feature(image_path, box, reid_extractor)
            return tuple(float(value) for value in box), target_track_id, selected_feature
        return None, target_track_id, None

    selected_index, selected_feature = select_initial_track_index(
        image_path=image_path,
        boxes=boxes,
        reference_feature=reference_feature,
        reid_extractor=reid_extractor,
    )
    
    target_track_id = int(track_ids[selected_index])

    return tuple(float(value) for value in boxes[selected_index]), target_track_id, selected_feature


def track_person_bbox(
    model,
    image_path: Path,
    tracker: str,
    persist: bool,
    target_track_id: int | None,
    reference_feature: FeatureVector | None,
    reid_extractor: ReIDFeatureExtractor,
    conf: float,
    iou: float,
    imgsz: ImageSizeArg | None,
    device: str | int | None,
) -> tuple[tuple[float, float, float, float] | None, int | None, FeatureVector | None]:
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
        return None, target_track_id, None
    return select_tracked_person_bbox(
        result=results[0],
        target_track_id=target_track_id,
        image_path=image_path,
        reference_feature=reference_feature,
        reid_extractor=reid_extractor,
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
    track: bool = False,
    tracker: str = "bytetrack.yaml",
    reid_model_name: str = "osnet_x1_0",
    reid_model_path: str | Path | None = None,
    reid_device: str | None = None,
    reid_imgsz: ReIDImageSizeArg = (256, 128),
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
        resolved_reid_device = resolve_reid_device(
            reid_device=reid_device,
            detector_device=device,
        )
        reid_extractor = ReIDFeatureExtractor(
            model_name=reid_model_name,
            model_path=reid_model_path,
            device=resolved_reid_device,
            image_size=reid_imgsz,
        )
        print(
            "ReID: "
            f"{reid_extractor.model_name} "
            f"({reid_extractor.model_path}, "
            f"device={reid_extractor.device}, "
            f"imgsz={reid_extractor.image_size[0]}x{reid_extractor.image_size[1]})"
        )
    else:
        reid_extractor = None

    stats = {
        "seen": 0,
        "cropped": 0,
        "existing": 0,
        "missed": 0,
    }

    if track:
        processed = 0
        subject_activity_groups = group_images_by_subject_activity(
            image_dir,
            limit=limit,
        )
        total_images = sum(
            len(paths)
            for _, camera_groups in subject_activity_groups
            for _, paths in camera_groups
        )

        for subject_activity_dir, camera_groups in subject_activity_groups:
            subject_reference_feature = None
            relative_subject_activity = subject_activity_dir.relative_to(image_dir)

            for _camera_dir, image_paths_in_camera in camera_groups:
                reset_model_trackers(model)
                target_track_id = None
                
                for image_path in image_paths_in_camera:
                    relative_path = image_path.relative_to(image_dir)
                    output_path = output_dir / relative_path

                    bbox, target_track_id, selected_feature = track_person_bbox(
                        model=model,
                        image_path=image_path,
                        tracker=tracker,
                        persist=True,
                        target_track_id=target_track_id,
                        reference_feature=subject_reference_feature,
                        reid_extractor=reid_extractor,
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                        device=device,
                    )
                    subject_reference_feature = update_reference_feature(
                        subject_reference_feature,
                        selected_feature,
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
                        print_progress(
                            f"Track crop {relative_subject_activity}",
                            processed,
                            total_images,
                            stats,
                        )

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
            "Use Ultralytics tracking with ByteTrack. Target reference is shared "
            "per Subject/Activity; tracker state is reset for each Trial/Camera."
        ),
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker config. Default: bytetrack.yaml.",
    )
    parser.add_argument(
        "--reid-model-name",
        default="osnet_x1_0",
        help="Torchreid model name for OSNet embeddings. Default: osnet_x1_0.",
    )
    parser.add_argument(
        "--reid-model-path",
        default=None,
        help=(
            "Optional Torchreid model weights path. If omitted, Torchreid uses "
            "its pretrained model loading behavior."
        ),
    )
    parser.add_argument(
        "--reid-device",
        default=None,
        help=(
            "Torchreid device, e.g. cuda, cuda:0, or cpu. Defaults to --device "
            "when provided, otherwise auto-detects CUDA."
        ),
    )
    parser.add_argument(
        "--reid-imgsz",
        type=parse_reid_imgsz,
        default=(256, 128),
        help="Torchreid input size as height x width. Default: 256x128.",
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
        reid_model_name=args.reid_model_name,
        reid_model_path=args.reid_model_path,
        reid_device=args.reid_device,
        reid_imgsz=args.reid_imgsz,
        limit=args.limit,
        log_every=args.log_every,
    )
    print(f"Done. Output directory: {args.output_dir.resolve()}")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
