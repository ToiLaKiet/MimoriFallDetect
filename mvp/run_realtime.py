#!/usr/bin/env python3
"""Demo runner for realtime ViTPose + skeleton sequence classification."""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from realtime_core import (
    BASE_DIR,
    DEFAULT_DETECTOR_MODEL,
    RTDetrPersonDetector,
    SkeletonSequenceClassifier,
    VitPoseRunner,
    draw_skeleton_on_black,
    draw_skeleton_overlay,
    fall_label_for_class,
    load_state_dict,
    prepare_classifier_tensor,
    put_status,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime ViTPose skeleton MVP")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=1280, help="Camera capture width")
    parser.add_argument("--height", type=int, default=720, help="Camera capture height")
    parser.add_argument(
        "--vitpose-model",
        default="usyd-community/vitpose-base-simple",
        help="Hugging Face model id or local ViTPose folder",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow transformers to download ViTPose if it is not cached",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BASE_DIR / "vitpose_lstm_best.pt",
        help="CNN+LSTM classifier checkpoint",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument(
        "--detector-model",
        default=DEFAULT_DETECTOR_MODEL,
        help="RT-DETR detector model id or local model folder",
    )
    parser.add_argument("--det-threshold", type=float, default=0.5)
    parser.add_argument("--person-label", default="person")
    parser.add_argument("--max-persons", type=int, default=1)
    parser.add_argument("--keypoint-threshold", type=float, default=0.25)
    parser.add_argument("--classifier-size", type=int, default=224)
    parser.add_argument("--classify-every", type=int, default=1)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames. 0 means run until q/interrupt.",
    )
    parser.add_argument("--headless", action="store_true", help="Do not open preview windows")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "out")
    parser.add_argument(
        "--save-latest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continuously write output-dir/latest_skeleton.jpg",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.classify_every < 1:
        raise ValueError("--classify-every must be >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    print(f"Using device: {device}")
    pose_runner = VitPoseRunner(args.vitpose_model, device, args.allow_download)
    detector = RTDetrPersonDetector(
        args.detector_model,
        device,
        args.allow_download,
        threshold=args.det_threshold,
        person_label=args.person_label,
        max_persons=args.max_persons,
    )

    state_dict = load_state_dict(args.checkpoint)
    num_classes = int(state_dict["classifier.1.bias"].numel())
    classifier = SkeletonSequenceClassifier(num_classes=num_classes)
    classifier.load_state_dict(state_dict, strict=True)
    classifier.to(device)
    classifier.eval()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    model_buffer: deque[np.ndarray] = deque(maxlen=10)
    last_action = ""
    last_log = time.monotonic()
    frame_count = 0
    fps = 0.0
    last_tick = time.monotonic()

    print("Running. Press q in the demo window to quit.")
    try:
        while True:
            ok, camera_frame = cap.read()
            if not ok:
                print("Camera frame read failed.")
                break

            boxes = detector.detect(camera_frame)
            poses = pose_runner.estimate(camera_frame, boxes)

            model_input_frame = draw_skeleton_on_black(
                camera_frame.shape,
                poses,
                args.keypoint_threshold,
            )
            model_buffer.append(model_input_frame.copy())

            frame_count += 1
            now = time.monotonic()
            delta = now - last_tick
            last_tick = now
            fps = (
                0.85 * fps + 0.15 * (1.0 / max(delta, 1e-6))
                if fps
                else 1.0 / max(delta, 1e-6)
            )

            if len(model_buffer) == model_buffer.maxlen and frame_count % args.classify_every == 0:
                tensor = prepare_classifier_tensor(model_buffer, args.classifier_size, device)
                with torch.inference_mode():
                    logits = classifier(tensor)
                    probs = torch.softmax(logits, dim=-1)[0]
                score, index = torch.max(probs, dim=0)
                class_index = int(index)
                last_action = (
                    f"{fall_label_for_class(class_index)} "
                    f"class={class_index} score={float(score):.2f}"
                )

            if args.save_latest:
                cv2.imwrite(str(args.output_dir / "latest_skeleton.jpg"), model_input_frame)

            if not args.headless:
                demo_frame = draw_skeleton_overlay(
                    camera_frame,
                    poses,
                    args.keypoint_threshold,
                )
                for x, y, w, h in boxes:
                    cv2.rectangle(
                        demo_frame,
                        (int(x), int(y)),
                        (int(x + w), int(y + h)),
                        (0, 255, 255),
                        2,
                    )
                put_status(demo_frame, fps, len(model_buffer), last_action)
                cv2.imshow("demo_camera_skeleton", demo_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if now - last_log >= 2.0:
                print(
                    f"frames={frame_count} fps={fps:.1f} buffer={len(model_buffer)}/10 "
                    f"action={last_action or 'warming_up'}"
                )
                last_log = now

            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
