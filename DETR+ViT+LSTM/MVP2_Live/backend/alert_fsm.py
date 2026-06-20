from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from PIL import Image

from agent import trigger_agent
from config_loader import AppConfig
from frame_codec import encode_rgb_jpeg_base64
from pipeline import FrameResult, is_bbox_pair_stable


class AlertState(str, Enum):
    IDLE = "idle"
    FALLING = "falling"
    MONITORING = "monitoring"
    TRIGGERED = "triggered"
    COOLDOWN = "cooldown"


@dataclass
class AlertStateMachine:
    config: AppConfig
    state: AlertState = AlertState.IDLE
    prev_label: str | None = None
    monitoring_started_at: float | None = None
    monitoring_anchor_bbox: list[float] | None = None
    last_bbox: list[float] | None = None
    cooldown_until: float | None = None
    agent_result: dict[str, Any] | None = None
    last_reason: str | None = None

    def reset(self) -> None:
        self.state = AlertState.IDLE
        self.prev_label = None
        self.monitoring_started_at = None
        self.monitoring_anchor_bbox = None
        self.last_bbox = None
        self.cooldown_until = None
        self.agent_result = None
        self.last_reason = None

    def _now(self) -> float:
        return time.monotonic()

    def _in_cooldown(self) -> bool:
        if self.cooldown_until is None:
            return False
        return self._now() < self.cooldown_until

    def _exit_cooldown_if_done(self) -> None:
        if self.state == AlertState.COOLDOWN and not self._in_cooldown():
            self.state = AlertState.IDLE
            self.cooldown_until = None
            self.last_reason = "cooldown_finished"

    def _current_label(self, frame: FrameResult) -> str | None:
        if frame.prediction is None:
            return self.prev_label
        return str(frame.prediction.get("label"))

    def _bbox_stable_vs_anchor(self, bbox: list[float]) -> bool:
        if self.monitoring_anchor_bbox is None:
            return True
        return is_bbox_pair_stable(
            self.monitoring_anchor_bbox,
            bbox,
            min_iou=self.config.stability_min_mean_iou,
            max_center_shift=self.config.stability_max_center_shift,
        )

    def update(self, frame: FrameResult, *, rgb_image: Image.Image | None = None) -> dict[str, Any]:
        self._exit_cooldown_if_done()

        current_label = self._current_label(frame)
        bbox = frame.bbox_xyxy if frame.embedding_ok else None

        trigger_now = False

        if self.state == AlertState.COOLDOWN:
            self.prev_label = current_label
            if bbox is not None:
                self.last_bbox = bbox
            return self._build_response(current_label, trigger_now)

        if self.state == AlertState.TRIGGERED:
            self.state = AlertState.COOLDOWN
            self.cooldown_until = self._now() + self.config.alert_cooldown_seconds
            self.monitoring_started_at = None
            self.monitoring_anchor_bbox = None
            self.last_reason = "alert_sent_entering_cooldown"

        if self._in_cooldown():
            self.state = AlertState.COOLDOWN
            self.prev_label = current_label
            if bbox is not None:
                self.last_bbox = bbox
            return self._build_response(current_label, trigger_now)

        if current_label == "fall" and self.state in (AlertState.IDLE, AlertState.FALLING):
            self.state = AlertState.FALLING
            self.last_reason = "fall_detected"

        elif (
            self.state == AlertState.FALLING
            and self.prev_label == "fall"
            and current_label == "normal"
        ):
            self.state = AlertState.MONITORING
            self.monitoring_started_at = self._now()
            self.monitoring_anchor_bbox = list(bbox) if bbox is not None else self.last_bbox
            self.last_reason = "fall_to_normal_start_monitoring"

        elif self.state == AlertState.MONITORING:
            if current_label == "fall":
                self.state = AlertState.FALLING
                self.monitoring_started_at = None
                self.monitoring_anchor_bbox = None
                self.last_reason = "fall_again_reset_monitoring"
            elif bbox is not None and not self._bbox_stable_vs_anchor(bbox):
                self.state = AlertState.IDLE
                self.monitoring_started_at = None
                self.monitoring_anchor_bbox = None
                self.last_reason = "bbox_unstable_reset"
            elif self.monitoring_started_at is not None:
                elapsed = self._now() - self.monitoring_started_at
                if elapsed >= self.config.stability_seconds:
                    self.state = AlertState.TRIGGERED
                    context: dict[str, Any] = {
                        "frame_index": frame.index,
                        "frame_name": frame.name,
                        "bbox_xyxy": bbox,
                        "monitoring_elapsed_s": round(elapsed, 2),
                        "prev_label": self.prev_label,
                        "current_label": current_label,
                    }
                    if rgb_image is not None:
                        width, height = rgb_image.size
                        context["frame_image_base64"] = encode_rgb_jpeg_base64(rgb_image)
                        context["frame_image_width"] = width
                        context["frame_image_height"] = height
                    self.agent_result = trigger_agent(context)
                    trigger_now = True
                    self.last_reason = "stable_5s_trigger_agent"

        if bbox is not None:
            self.last_bbox = bbox

        if current_label is not None:
            self.prev_label = current_label

        return self._build_response(current_label, trigger_now)

    def _build_response(
        self,
        current_label: str | None,
        trigger_now: bool,
    ) -> dict[str, Any]:
        elapsed: float | None = None
        remaining: float | None = None
        bbox_stable: bool | None = None

        if self.state == AlertState.MONITORING and self.monitoring_started_at is not None:
            elapsed = max(0.0, self._now() - self.monitoring_started_at)
            remaining = max(0.0, self.config.stability_seconds - elapsed)
            if self.last_bbox is not None and self.monitoring_anchor_bbox is not None:
                bbox_stable = self._bbox_stable_vs_anchor(self.last_bbox)

        cooldown_remaining: float | None = None
        if self._in_cooldown() and self.cooldown_until is not None:
            cooldown_remaining = max(0.0, self.cooldown_until - self._now())

        return {
            "state": self.state.value,
            "prev_label": self.prev_label,
            "current_label": current_label,
            "monitoring_elapsed_s": round(elapsed, 2) if elapsed is not None else None,
            "monitoring_remaining_s": round(remaining, 2) if remaining is not None else None,
            "stability_seconds": self.config.stability_seconds,
            "bbox_stable": bbox_stable,
            "trigger_agent": trigger_now,
            "agent_result": self.agent_result,
            "reason": self.last_reason,
            "cooldown_remaining_s": round(cooldown_remaining, 2) if cooldown_remaining else None,
        }
