from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from PIL import Image

from pipeline import FallDetectionPipeline, FrameResult, build_fall_confirmation


class LiveSession:
    """Stateful sliding-window session for webcam / live camera demo."""

    def __init__(self, pipeline: FallDetectionPipeline) -> None:
        self.pipeline = pipeline
        self.reset()

    def reset(self) -> None:
        self._window: deque[np.ndarray] = deque(maxlen=self.pipeline.config.window_size)
        self._frames: list[FrameResult] = []
        self._frame_index = 0
        self._max_history = (
            self.pipeline.config.stability_frames + self.pipeline.config.window_size + 32
        )

    @property
    def active(self) -> bool:
        return self._frame_index > 0 or len(self._window) > 0

    @property
    def frames_total(self) -> int:
        return self._frame_index

    @property
    def buffer_size(self) -> int:
        return len(self._window)

    def process_image(self, image: Image.Image) -> dict[str, Any]:
        rgb_image = image.convert("RGB")
        name = f"live_{self._frame_index:06d}"
        frame = self.pipeline.process_one_frame(
            rgb_image,
            index=self._frame_index,
            name=name,
            window=self._window,
        )
        self._frames.append(frame)
        self._frame_index += 1

        if len(self._frames) > self._max_history:
            self._frames = self._frames[-self._max_history :]

        fall_confirmation = build_fall_confirmation(self._frames, self.pipeline.config)

        return {
            "frame": frame.to_dict(),
            "frames_total": self._frame_index,
            "window_size": self.pipeline.config.window_size,
            "fall_confirmation": fall_confirmation,
        }
