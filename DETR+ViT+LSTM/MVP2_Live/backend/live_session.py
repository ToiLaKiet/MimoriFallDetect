from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from PIL import Image

from alert_fsm import AlertStateMachine
from pipeline import FallDetectionPipeline


class LiveSession:
    """Stateful sliding-window session for MimamoriFall live camera."""

    def __init__(self, pipeline: FallDetectionPipeline) -> None:
        self.pipeline = pipeline
        self.reset()

    def reset(self) -> None:
        self._window: deque[np.ndarray] = deque(maxlen=self.pipeline.config.window_size)
        self._frame_index = 0
        self._fsm = AlertStateMachine(self.pipeline.config)

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
        self._frame_index += 1

        alert = self._fsm.update(frame, rgb_image=rgb_image)

        return {
            "frame": frame.to_dict(),
            "frames_total": self._frame_index,
            "window_size": self.pipeline.config.window_size,
            "alert": alert,
        }
