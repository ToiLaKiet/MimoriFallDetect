from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from yolo_inference import (
    DEFAULT_YOLO_MODEL,
    detect_person_detections as detect_yolo_person_detections,
    load_yolo_model,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DetectorFn = Callable[..., np.ndarray]
ImageSizeArg = int | tuple[int, int]
CameraRoi = tuple[int, int, int, int]
DEFAULT_CAMERA_ROIS: dict[str, CameraRoi] = {
    "Camera1": (0, 0, 640, 480),
    "Camera2": (0, 0, 591, 479),
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
    """Load one detector backend and return model plus detection function."""

    if detector == "yolo":
        resolved_model_path = model_path or DEFAULT_YOLO_MODEL
        return (
            load_yolo_model(resolved_model_path),
            detect_yolo_person_detections,
            resolved_model_path,
        )

    if detector == "rtdetr-x":
        backend_path = Path(__file__).with_name("rtdetr-x_inference.py")
        backend = load_python_file("rtdetr_x_inference", backend_path)
        resolved_model_path = model_path or backend.DEFAULT_RTDETR_MODEL
        return (
            backend.load_rtdetr_model(resolved_model_path),
            backend.detect_person_detections,
            resolved_model_path,
        )

    raise ValueError(f"Unsupported detector: {detector}")


def natural_sort_key(value: str) -> tuple[tuple[int, int | str], ...]:
    # Split the string into digit and non-digit parts, converting digit parts to integers for natural sorting (e.g., "Camera10" > "Camera2").
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.lower())
        for part in re.split(r"(\d+)", value)
        if part
    )


def iter_timestamp_images(image_dir: Path, limit: int = 0):
    """Yield image files under Subject/Activity/Trial/Camera folders."""
    count = 0
    for image_path in sorted(
        image_dir.rglob("*"),
        key=lambda path: natural_sort_key(str(path.relative_to(image_dir))),
    ):
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


def iter_sequence_image_groups(image_dir: Path, limit: int = 0):
    """Yield images grouped by Subject/Activity/Trial/Camera folder."""

    groups: dict[Path, list[Path]] = {}
    for image_path in iter_timestamp_images(image_dir, limit=limit):
        relative_parts = image_path.relative_to(image_dir).parts
        sequence_path = Path(*relative_parts[:4]) # For eg : Subject1/Activity1/Trial1/Camera1. * is used to unpack the first 4 parts of the relative path and create a new Path object representing the sequence folder.
        groups.setdefault(sequence_path, []).append(image_path) 

    for sequence_path in sorted(
        groups,
        key=lambda path: natural_sort_key(str(path)),
    ):
        yield sequence_path, sorted(
            groups[sequence_path],
            key=lambda path: natural_sort_key(path.name),
        ) # yield the sequence path and the list of image paths sorted by filename. yeild is used to create a generator that can be iterated over, allowing for memory-efficient processing of large datasets.


def normalize_camera_name(camera_name: str) -> str:
    return "".join(char.lower() for char in camera_name if char.isalnum())


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
    relative_parts = image_path.relative_to(image_dir).parts
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


def ensure_detection_array(detections: np.ndarray) -> np.ndarray:
    detections = np.asarray(detections, dtype=np.float32)
    if detections.size == 0:
        return np.empty((0, 6), dtype=np.float32) 
    if detections.ndim == 1:
        detections = detections.reshape(1, -1) # this mean the detector returned a single detection as a 1D array, so we reshape it to (1, 6)
    if detections.shape[1] != 6:
        raise ValueError(
            "Detector must return rows as [x1, y1, x2, y2, confidence, class_id]."
        )
    return detections


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    xywh = np.empty_like(boxes, dtype=np.float32)
    xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
    xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
    xywh[:, 2] = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    xywh[:, 3] = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return xywh


class TrackerDetections:
    # update method : BYTETracker expects a list of detections in the format of [x_center, y_center, width, height, confidence, class_id], so we create a wrapper class that takes the xyxy detections from our detector and provides the necessary properties to convert them to the expected format for tracking. This allows us to use the same detection output for both cropping and tracking without modifying the original detection code.
    """Minimal Ultralytics Boxes-like wrapper used by BYTETracker."""

    def __init__(self, detections: np.ndarray):
        self.detections = ensure_detection_array(detections)

    def __len__(self) -> int:
        return len(self.detections)

    def __getitem__(self, index):
        return TrackerDetections(self.detections[index])

    @property
    def xywh(self) -> np.ndarray:
        return xyxy_to_xywh(self.detections[:, :4])

    @property
    def conf(self) -> np.ndarray:
        return self.detections[:, 4]

    @property
    def cls(self) -> np.ndarray:
        return self.detections[:, 5]


def create_bytetrack_tracker(
    max_missed: int,
    match_thresh: float,
    track_high_thresh: float = 0.25,
    track_low_thresh: float = 0.1,
    new_track_thresh: float = 0.25,
):
    try:
        from ultralytics.trackers.byte_tracker import BYTETracker
    except ModuleNotFoundError as exc:
        if exc.name in {"lap", "lapx"}:
            raise ImportError(
                "ByteTrack requires the Ultralytics tracking dependency 'lapx'. "
                "Install it in the runtime with: pip install lapx"
            ) from exc
        raise

    args = argparse.Namespace(
        tracker_type="bytetrack",
        track_high_thresh=track_high_thresh,
        track_low_thresh=track_low_thresh,
        new_track_thresh=new_track_thresh,
        track_buffer=max_missed,
        match_thresh=match_thresh,
        fuse_score=True,
    )
    return BYTETracker(args=args) # what if the frame rate is not 30? This is only used for motion prediction, so it may not be critical to get it exactly right. If the frame rate is known, it would be better to set it accordingly for improved tracking performance.


def largest_bbox_from_detections(
    detections: np.ndarray,
) -> tuple[float, float, float, float] | None:
    detections = ensure_detection_array(detections)
    if len(detections) == 0:
        return None

    boxes = detections[:, :4]
    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    largest_index = int(np.argmax(widths * heights))
    return tuple(float(value) for value in boxes[largest_index])


def track_area(track: np.ndarray) -> float:
    width = max(0.0, float(track[2] - track[0]))
    height = max(0.0, float(track[3] - track[1]))
    return width * height


def select_largest_track(tracks: np.ndarray) -> np.ndarray | None:
    if len(tracks) == 0:
        return None
    areas = np.asarray([track_area(track) for track in tracks], dtype=np.float32)
    return tracks[int(np.argmax(areas))] # shape: (x1, y1, x2, y2, track_id, confidence)


def select_track_by_id(tracks: np.ndarray, track_id: int) -> np.ndarray | None:
    if len(tracks) == 0:
        return None
    track_ids = tracks[:, 4].astype(np.int64)
    matches = tracks[track_ids == track_id]
    if len(matches) == 0:
        return None
    return matches[int(np.argmax(matches[:, 5]))]


def detect_detections_in_roi(
    image_path: Path,
    model,
    detect_detections: DetectorFn,
    conf: float,
    iou: float,
    imgsz: ImageSizeArg,
    device: str | int | None,
    roi: CameraRoi,
) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
        detector_input, offset = build_detector_input(image, roi)
        detections = detect_detections(
            image=detector_input,
            model=model,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
        )

    return ensure_detection_array(detections), offset


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
    detect_detections: DetectorFn,
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

        detections = detect_detections(
            image=detector_input,
            model=model,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
        )

    bbox = largest_bbox_from_detections(detections)
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


def print_progress(
    prefix: str,
    index: int,
    total: int,
    stats: dict[str, int],
    relative_path: Path,
) -> None:
    print(
        f"{prefix} "
        f"{index}/{total} | "
        f"cropped={stats['cropped']} "
        f"existing={stats['existing']} "
        f"missed={stats['missed']}"
        f" | {relative_path}"
    )


def crop_tracked_sequence(
    sequence_path: Path,
    image_paths: list[Path],
    image_dir: Path,
    output_dir: Path,
    model,
    detect_detections: DetectorFn,
    conf: float,
    iou: float,
    device: str | int | None,
    overwrite: bool,
    max_missed: int,
    track_match_thresh: float,
    stats: dict[str, int],
    processed_count: int,
    total_images: int,
    log_every: int,
) -> int:
    tracker = create_bytetrack_tracker(
        max_missed=max_missed,
        match_thresh=track_match_thresh,
    )
    target_track_id: int | None = None
    missed_count = 0
    target_lost_reported = False

    for sequence_index, image_path in enumerate(image_paths, start=1):
        # iterate through the images in the sequence, running detection and tracking to maintain a consistent target across frames. The first detected track in the first frame is selected as the target, and subsequent frames attempt to find that track ID. If the track is lost for too many frames, a warning is printed.
        processed_count += 1
        relative_path = image_path.relative_to(image_dir)
        output_path = output_dir / relative_path
        roi = camera_roi_for_image(image_path=image_path, image_dir=image_dir)
        image_imgsz = camera_imgsz_for_image(
            image_path=image_path,
            image_dir=image_dir,
        )
        detections, (x_offset, y_offset) = detect_detections_in_roi(
            image_path=image_path,
            model=model,
            detect_detections=detect_detections,
            conf=conf,
            iou=iou,
            imgsz=image_imgsz,
            device=device,
            roi=roi,
        )
        tracks = tracker.update(TrackerDetections(detections)) # ByteTrack return format : list of tracks, where each track is represented as a numpy array with the format [x1, y1, x2, y2, track_id, confidence]. The tracker.update() method takes the current frame's detections and updates the internal state of the tracker, returning the list of active tracks after processing the new detections
        bbox = None
        if target_track_id is None:
            target_track = select_largest_track(tracks)
            if target_track is None:
                if sequence_index == 1:
                    print(
                        "WARNING: ByteTrack could not initialize target on "
                        f"first frame: {relative_path}"
                    )
            else:
                target_track_id = int(target_track[4])
                bbox = tuple(float(value) for value in target_track[:4])
                if sequence_index > 1:
                    print(
                        "WARNING: ByteTrack target initialized after first "
                        f"frame in {sequence_path}: {relative_path}"
                    )
        else:
            target_track = select_track_by_id(tracks, target_track_id)
            if target_track is None:
                missed_count += 1
                if missed_count >= max_missed and not target_lost_reported:
                    print(
                        "WARNING: ByteTrack target lost for "
                        f"{missed_count} frame(s) in {sequence_path}. "
                        f"Last checked frame: {relative_path}"
                    )
                    target_lost_reported = True
            else:
                missed_count = 0
                target_lost_reported = False
                bbox = tuple(float(value) for value in target_track[:4])

        bbox = offset_bbox(bbox, x_offset, y_offset)
        status = save_crop_from_bbox(
            image_path=image_path,
            output_path=output_path,
            bbox=bbox,
            overwrite=overwrite,
        )
        update_stats(stats, status)

        if processed_count == total_images or processed_count % log_every == 0:
            print_progress(
                "BBox track",
                processed_count,
                total_images,
                stats,
                relative_path,
            )

    return processed_count


def crop_dataset(
    image_dir: Path,
    output_dir: Path,
    detector: str = "yolo",
    model_path: str | Path | None = None,
    conf: float = 0.1,
    iou: float = 0.7,
    device: str | int | None = None,
    overwrite: bool = False,
    limit: int = 0,
    log_every: int = 100,
    tracking: str = "bytetrack",
    max_missed: int = 10,
    track_match_thresh: float = 0.8,
) -> dict[str, int]:
    """Crop person images into output_dir while preserving input folder structure."""

    image_dir = image_dir.resolve()
    output_dir = output_dir.resolve()
    sequence_groups = list(iter_sequence_image_groups(image_dir, limit=limit))
    image_paths = [
        image_path
        for _, sequence_image_paths in sequence_groups
        for image_path in sequence_image_paths
    ]
    if not image_paths:
        return {"seen": 0, "cropped": 0, "existing": 0, "missed": 0}

    model, detect_detections, resolved_model_path = load_detector_backend(
        detector=detector,
        model_path=model_path,
    )
    print(f"Detector: {detector}")
    print(f"Model: {resolved_model_path}")
    print(f"Camera ROI: hardcoded ({len(DEFAULT_CAMERA_ROIS)} camera(s))")
    print(f"Camera imgsz: hardcoded ({len(DEFAULT_CAMERA_IMGSZ)} camera(s))")
    print(f"Tracking: {tracking}")
    if tracking == "bytetrack":
        print(f"ByteTrack max_missed: {max_missed}")
        print(f"ByteTrack match_thresh: {track_match_thresh}")

    stats = {
        "seen": 0,
        "cropped": 0,
        "existing": 0,
        "missed": 0,
        "sequences": len(sequence_groups),
    }

    if tracking == "bytetrack":
        processed_count = 0
        for sequence_path, sequence_image_paths in sequence_groups:
            processed_count = crop_tracked_sequence(
                sequence_path=sequence_path,
                image_paths=sequence_image_paths,
                image_dir=image_dir,
                output_dir=output_dir,
                model=model,
                detect_detections=detect_detections,
                conf=conf,
                iou=iou,
                device=device,
                overwrite=overwrite,
                max_missed=max_missed,
                track_match_thresh=track_match_thresh,
                stats=stats,
                processed_count=processed_count,
                total_images=len(image_paths),
                log_every=log_every,
            )
        return stats

    if tracking != "none":
        raise ValueError(f"Unsupported tracking mode: {tracking}")

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
            detect_detections=detect_detections,
            conf=conf,
            iou=iou,
            imgsz=image_imgsz,
            device=device,
            overwrite=overwrite,
            roi=roi,
        )

        update_stats(stats, status)

        if index == len(image_paths) or index % log_every == 0:
            print_progress("BBox crop", index, len(image_paths), stats, relative_path)

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
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help=(
            "Detector confidence threshold. Default 0.1 keeps low-score boxes "
            "available for ByteTrack association."
        ),
    )
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--tracking",
        choices=("bytetrack", "none"),
        default="bytetrack",
        help="Tracking mode. Default: bytetrack.",
    )
    parser.add_argument(
        "--max-missed",
        type=int,
        default=10,
        help="Maximum missed frames kept by ByteTrack before target is considered lost.",
    )
    parser.add_argument(
        "--track-match-thresh",
        type=float,
        default=0.8,
        help="ByteTrack association threshold. Higher is looser.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image_dir.is_dir():
        raise ValueError(f"Image directory does not exist: {args.image_dir}")
    args.log_every = max(1, args.log_every)
    args.max_missed = max(1, args.max_missed)

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
        tracking=args.tracking,
        max_missed=args.max_missed,
        track_match_thresh=args.track_match_thresh,
    )
    print(f"Done. Output directory: {args.output_dir.resolve()}")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
