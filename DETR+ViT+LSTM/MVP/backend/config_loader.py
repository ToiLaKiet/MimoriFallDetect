from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from paths import BACKEND_DIR


@dataclass(frozen=True)
class AppConfig:
    checkpoint: Path
    rtdetr_model: str
    rtdetr_conf: float
    rtdetr_iou: float
    mmpose_config: Path
    mmpose_checkpoint: Path
    embedding_source: str
    window_size: int
    embedding_dim: int
    stability_frames: int
    stability_min_mean_iou: float
    stability_max_center_shift: float
    device: str | None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint.resolve()),
            "rtdetr_model": self.rtdetr_model,
            "rtdetr_conf": self.rtdetr_conf,
            "rtdetr_iou": self.rtdetr_iou,
            "mmpose_config": str(self.mmpose_config.resolve()),
            "mmpose_checkpoint": str(self.mmpose_checkpoint.resolve()),
            "embedding_source": self.embedding_source,
            "window_size": self.window_size,
            "embedding_dim": self.embedding_dim,
            "stability_frames": self.stability_frames,
            "stability_min_mean_iou": self.stability_min_mean_iou,
            "stability_max_center_shift": self.stability_max_center_shift,
            "device": self.device,
        }


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_config(config_path: str | Path | None = None) -> AppConfig:
    config_path = Path(config_path or BACKEND_DIR / "config.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    base_dir = config_path.parent
    rtdetr_raw = raw.get("rtdetr_model", raw.get("yolo_model", "yolo26x.pt"))
    rtdetr_path = _resolve_path(rtdetr_raw, base_dir)
    rtdetr_model = str(rtdetr_path) if rtdetr_path.is_file() else str(rtdetr_raw)

    mmpose_config = _resolve_path(raw["mmpose_config"], base_dir)
    mmpose_checkpoint = _resolve_path(raw["mmpose_checkpoint"], base_dir)

    return AppConfig(
        checkpoint=_resolve_path(raw.get("checkpoint", "../../runs5/best.pt"), base_dir),
        rtdetr_model=rtdetr_model,
        rtdetr_conf=float(raw.get("rtdetr_conf", raw.get("yolo_conf", 0.25))),
        rtdetr_iou=float(raw.get("rtdetr_iou", raw.get("yolo_iou", 0.7))),
        mmpose_config=mmpose_config,
        mmpose_checkpoint=mmpose_checkpoint,
        embedding_source=str(raw.get("embedding_source", "pre_head_gap")),
        window_size=int(raw.get("window_size", 10)),
        embedding_dim=int(raw.get("embedding_dim", 1280)),
        stability_frames=int(raw.get("stability_frames", 100)),
        stability_min_mean_iou=float(raw.get("stability_min_mean_iou", 0.65)),
        stability_max_center_shift=float(raw.get("stability_max_center_shift", 0.12)),
        device=raw.get("device"),
    )
