#!/usr/bin/env python3
"""Demo runner for realtime ViTPose + skeleton sequence classification."""

from __future__ import annotations

import argparse
import threading
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
    parser.add_argument("--camera", type=int, default=2, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=640, help="Camera capture width")
    parser.add_argument("--height", type=int, default=480, help="Camera capture height")
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        help="Requested camera/display FPS. Camera backends may ignore this value.",
    )
    parser.add_argument(
        "--async-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run DETR/ViTPose/classifier in a worker so preview FPS stays smooth.",
    )
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


class AsyncInferenceWorker:
    def __init__(
        self,
        detector: RTDetrPersonDetector,
        pose_runner: VitPoseRunner,
        classifier: SkeletonSequenceClassifier,
        device: torch.device,
        args: argparse.Namespace,
    ) -> None:
        self.detector = detector
        self.pose_runner = pose_runner
        self.classifier = classifier
        self.device = device
        self.args = args

        self.model_buffer: deque[np.ndarray] = deque(maxlen=10)
        self.lock = threading.Lock()
        self.frame_ready = threading.Event()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="mvp-inference", daemon=True)

        self.pending_frame: np.ndarray | None = None
        self.boxes: list[list[float]] = []
        self.poses: list[dict] = []
        self.buffer_size = 0
        self.last_action = ""
        self.inference_fps = 0.0
        self.processed_count = 0

    def start(self) -> None:
        self.thread.start()

    def submit(self, frame: np.ndarray) -> None:
        with self.lock:
            self.pending_frame = frame.copy()
            self.frame_ready.set()

    def snapshot(self) -> tuple[list[list[float]], list[dict], int, str, float, int]:
        with self.lock:
            return (
                [box[:] for box in self.boxes],
                list(self.poses),
                self.buffer_size,
                self.last_action,
                self.inference_fps,
                self.processed_count,
            )

    def stop(self) -> None:
        self.stop_event.set()
        self.frame_ready.set()
        self.thread.join(timeout=3.0)

    def _take_latest_frame(self) -> np.ndarray | None:
        self.frame_ready.wait(timeout=0.1)
        if self.stop_event.is_set():
            return None
        with self.lock:
            frame = self.pending_frame
            self.pending_frame = None
            self.frame_ready.clear()
        return frame

    def _run(self) -> None:
        last_tick = time.monotonic()
        while not self.stop_event.is_set():
            frame = self._take_latest_frame()
            if frame is None:
                continue

            boxes = self.detector.detect(frame)
            poses = self.pose_runner.estimate(frame, boxes)
            model_input_frame = draw_skeleton_on_black(
                frame.shape,
                poses,
                self.args.keypoint_threshold,
            )
            self.model_buffer.append(model_input_frame.copy())

            processed_count = self.processed_count + 1
            now = time.monotonic()
            delta = now - last_tick
            last_tick = now
            inference_fps = (
                0.85 * self.inference_fps + 0.15 * (1.0 / max(delta, 1e-6))
                if self.inference_fps
                else 1.0 / max(delta, 1e-6)
            )
            last_action = self.last_action

            if (
                len(self.model_buffer) == self.model_buffer.maxlen
                and processed_count % self.args.classify_every == 0
            ):
                tensor = prepare_classifier_tensor(
                    self.model_buffer,
                    self.args.classifier_size,
                    self.device,
                )
                with torch.inference_mode():
                    logits = self.classifier(tensor)
                    probs = torch.softmax(logits, dim=-1)[0]
                score, index = torch.max(probs, dim=0)
                class_index = int(index)
                last_action = (
                    f"{fall_label_for_class(class_index)} "
                    f"class={class_index} score={float(score):.2f}"
                )

            if self.args.save_latest:
                cv2.imwrite(str(self.args.output_dir / "latest_skeleton.jpg"), model_input_frame)

            with self.lock:
                self.boxes = boxes
                self.poses = poses
                self.buffer_size = len(self.model_buffer)
                self.last_action = last_action
                self.inference_fps = inference_fps
                self.processed_count = processed_count


def update_fps(fps: float, last_tick: float) -> tuple[float, float]:
    now = time.monotonic()
    delta = now - last_tick
    fps = 0.85 * fps + 0.15 * (1.0 / max(delta, 1e-6)) if fps else 1.0 / max(delta, 1e-6)
    return fps, now


def maybe_limit_loop_rate(start_time: float, target_fps: float) -> None:
    if target_fps <= 0:
        return
    target_period = 1.0 / target_fps
    elapsed = time.monotonic() - start_time
    if elapsed < target_period:
        time.sleep(target_period - elapsed)


def draw_demo_frame(
    camera_frame: np.ndarray,
    boxes: list[list[float]],
    poses: list[dict],
    keypoint_threshold: float,
    fps: float,
    buffer_size: int,
    action_text: str,
) -> np.ndarray:
    demo_frame = draw_skeleton_overlay(
        camera_frame,
        poses,
        keypoint_threshold,
    )
    for x, y, w, h in boxes:
        cv2.rectangle(
            demo_frame,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            (0, 255, 255),
            2,
        )
    put_status(demo_frame, fps, buffer_size, action_text)
    return demo_frame


def run_async_loop(
    args: argparse.Namespace,
    cap: cv2.VideoCapture,
    detector: RTDetrPersonDetector,
    pose_runner: VitPoseRunner,
    classifier: SkeletonSequenceClassifier,
    device: torch.device,
) -> None:
    worker = AsyncInferenceWorker(detector, pose_runner, classifier, device, args)
    worker.start()

    display_fps = 0.0
    last_tick = time.monotonic()
    last_log = time.monotonic()
    frame_count = 0

    try:
        while True:
            loop_start = time.monotonic()
            ok, camera_frame = cap.read()
            if not ok:
                print("Camera frame read failed.")
                break

            worker.submit(camera_frame)
            frame_count += 1
            display_fps, last_tick = update_fps(display_fps, last_tick)

            boxes, poses, buffer_size, last_action, inference_fps, processed_count = (
                worker.snapshot()
            )
            status = f"infer={inference_fps:.1f}"
            if last_action:
                status += f" | {last_action}"
            elif processed_count == 0:
                status += " | warming_up"

            if not args.headless:
                demo_frame = draw_demo_frame(
                    camera_frame,
                    boxes,
                    poses,
                    args.keypoint_threshold,
                    display_fps,
                    buffer_size,
                    status,
                )
                cv2.imshow("demo_camera_skeleton", demo_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            now = time.monotonic()
            if now - last_log >= 2.0:
                print(
                    f"frames={frame_count} display_fps={display_fps:.1f} "
                    f"infer_fps={inference_fps:.1f} buffer={buffer_size}/10 "
                    f"processed={processed_count} action={last_action or 'warming_up'}"
                )
                last_log = now

            if args.max_frames and frame_count >= args.max_frames:
                break

            maybe_limit_loop_rate(loop_start, args.target_fps)
    finally:
        worker.stop()


def run_sync_loop(
    args: argparse.Namespace,
    cap: cv2.VideoCapture,
    detector: RTDetrPersonDetector,
    pose_runner: VitPoseRunner,
    classifier: SkeletonSequenceClassifier,
    device: torch.device,
) -> None:
    model_buffer: deque[np.ndarray] = deque(maxlen=10)
    last_action = ""
    last_log = time.monotonic()
    frame_count = 0
    fps = 0.0
    last_tick = time.monotonic()

    while True:
        loop_start = time.monotonic()
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
        model_buffer.append(model_input_frame.copy()) # automatically pop oldest if maxlen exceeded

        frame_count += 1 # Increment frame count after processing to reflect actual inference count in logs and status.
        fps, last_tick = update_fps(fps, last_tick)

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


        if not args.headless:
            demo_frame = draw_demo_frame(
                camera_frame,
                boxes,
                poses,
                args.keypoint_threshold,
                fps,
                len(model_buffer),
                last_action,
            )
            cv2.imshow("demo_camera_skeleton", demo_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        now = time.monotonic()
        if now - last_log >= 2.0:
            print(
                f"frames={frame_count} fps={fps:.1f} buffer={len(model_buffer)}/10 "
                f"action={last_action or 'warming_up'}"
            )
            last_log = now

        if args.max_frames and frame_count >= args.max_frames:
            break

        maybe_limit_loop_rate(loop_start, args.target_fps)


def main() -> int:
    args = parse_args()
    if args.classify_every < 1:
        raise ValueError("--classify-every must be >= 1")
    if args.target_fps < 0:
        raise ValueError("--target-fps must be >= 0")
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
    if args.target_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, args.target_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Running. Press q in the demo window to quit.")
    mode = "async" if args.async_inference else "sync"
    print(f"Runtime mode: {mode}; requested camera/display FPS: {args.target_fps:g}")
    try:
        if args.async_inference:
            run_async_loop(
                args,
                cap,
                detector,
                pose_runner,
                classifier,
                device,
            )
        else:
            run_sync_loop(
                args,
                cap,
                detector,
                pose_runner,
                classifier,
                device,
            )
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
