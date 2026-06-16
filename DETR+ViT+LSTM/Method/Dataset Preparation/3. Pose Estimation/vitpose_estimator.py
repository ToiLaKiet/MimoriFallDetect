from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def configure_runtime_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "vitpose-pose-estimate-cache"
    for child in ("matplotlib", "xdg"):
        (cache_root / child).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))


def full_image_bbox_xywh(width: int, height: int) -> list[float]:
    return [0.0, 0.0, float(width), float(height)]


class VitPoseEstimator:
    """Top-down ViTPose-H pipeline for pre-cropped person images."""

    def __init__(
        self,
        pose_model_name: str,
        device: torch.device,
        dataset_index: int,
        allow_download: bool,
    ) -> None:
        configure_runtime_cache()

        from transformers import AutoProcessor, VitPoseForPoseEstimation  # noqa: PLC0415

        self.device = device
        self.dataset_index = dataset_index
        self.use_dataset_index = "plus" in pose_model_name.lower()

        print(f"Loading ViTPose model: {pose_model_name}")
        processor_kwargs = {}
        model_kwargs = {}
        if not allow_download:
            processor_kwargs["local_files_only"] = True
            model_kwargs["local_files_only"] = True

        self.pose_processor = AutoProcessor.from_pretrained(
            pose_model_name,
            **processor_kwargs,
        )
        self.pose_model = (
            VitPoseForPoseEstimation.from_pretrained(pose_model_name, **model_kwargs)
            .to(device)
            .eval()
        )

    @torch.no_grad()
    def estimate_with_bbox(
        self,
        image: Image.Image,
        bbox_xywh: list[float],
    ) -> tuple[list[float], np.ndarray, np.ndarray]:
        pose_inputs = self.pose_processor(
            image,
            boxes=[[bbox_xywh]],
            return_tensors="pt",
        ).to(self.device)

        if self.use_dataset_index:
            batch_size = pose_inputs["pixel_values"].shape[0]
            pose_inputs["dataset_index"] = torch.full(
                (batch_size,),
                self.dataset_index,
                dtype=torch.long,
                device=self.device,
            )
            pose_outputs = self.pose_model(**pose_inputs)
        else:
            pose_outputs = self.pose_model(**pose_inputs)

        pose_results = self.pose_processor.post_process_pose_estimation(
            pose_outputs,
            boxes=[[bbox_xywh]],
        )[0]
        if not pose_results:
            return bbox_xywh, np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

        pose = pose_results[0]
        keypoints = pose["keypoints"].detach().cpu().numpy().astype(np.float32)
        scores = pose["scores"].detach().cpu().numpy().astype(np.float32)
        return bbox_xywh, keypoints, scores

    @torch.no_grad()
    def estimate(self, image: Image.Image) -> tuple[list[float] | None, np.ndarray, np.ndarray]:
        width, height = image.size
        return self.estimate_with_bbox(image, full_image_bbox_xywh(width, height))
