from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image


EmbeddingSource = Literal[
    "backbone_last_gap",
    "backbone_last_flatten",
    "backbone_hidden_gap",
]


def configure_runtime_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "vitpose-pose-estimate-cache"
    for child in ("matplotlib", "xdg"):
        (cache_root / child).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))


def full_image_bbox_xywh(width: int, height: int) -> list[float]:
    return [0.0, 0.0, float(width), float(height)]


def pool_tokens_gap(token_features: torch.Tensor) -> torch.Tensor:
    """Global average pool over token dimension: (B, T, C) -> (B, C)."""

    return token_features.mean(dim=1)


def pool_feature_map_gap(feature_map: torch.Tensor) -> torch.Tensor:
    """Global average pool over spatial dims: (B, C, H, W) -> (B, C)."""

    return feature_map.mean(dim=(-2, -1))


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
        self.dataset_index = dataset_index # dataset_index is the index of the dataset. :)? 1 is for train, 2 is for val, 3 is for test.
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

    def _build_pose_inputs(self, image: Image.Image, bbox_xywh: list[float]) -> dict[str, torch.Tensor]:
        pose_inputs = self.pose_processor(
            image,
            boxes=[[bbox_xywh]],
            return_tensors="pt",
        )
        pose_inputs = {key: value.to(self.device) for key, value in pose_inputs.items()}

        if self.use_dataset_index:
            batch_size = pose_inputs["pixel_values"].shape[0] # pose_inputs['pixel_values'] is a tensor of shape (B, C, H, W) where B is the batch size, C is the number of channels, H is the height, and W is the width.
            pose_inputs["dataset_index"] = torch.full(
                (batch_size,),
                self.dataset_index,
                dtype=torch.long,
                device=self.device,
            ) # torch.full creates a tensor of shape (B,) filled with the value self.dataset_index. dataset_index is the index of the dataset. 
        return pose_inputs

    def _backbone_outputs(self, pose_inputs: dict[str, torch.Tensor]):
        backbone_kwargs = {
            "output_hidden_states": True,
        }
        if self.use_dataset_index:
            backbone_kwargs["dataset_index"] = pose_inputs["dataset_index"]

        return self.pose_model.backbone.forward_with_filtered_kwargs(
            pose_inputs["pixel_values"],
            **backbone_kwargs,
        )

    def _feature_map_from_backbone(self, backbone_outputs) -> torch.Tensor:
        """Match HF ViTPose forward: last backbone stage -> (B, C, H, W)."""

        sequence_output = backbone_outputs.feature_maps[-1] # feature_maps is a list of tensors, each tensor is a feature map of the backbone. the last one is the output of the backbone.
        batch_size = sequence_output.shape[0]
        patch_height = (
            self.pose_model.config.backbone_config.image_size[0]
            // self.pose_model.config.backbone_config.patch_size[0] # patch_size is the size of the patch in the image, image_size is the size of the image, so this is the number of patches in the image. // is the integer division.
        )
        patch_width = (
            self.pose_model.config.backbone_config.image_size[1]
            // self.pose_model.config.backbone_config.patch_size[1]
        )
        return (
            sequence_output.permute(0, 2, 1) # to permute is to change the order of the dimensions of the tensor.
            .reshape(batch_size, -1, patch_height, patch_width)  # -1 is the reference to the last dimension.
            .contiguous() # to contiguous is to make the tensor contiguous in memory. contiguous means that the memory is in a contiguous block.
        )

    def _embedding_from_backbone(
        self,
        backbone_outputs,
        source: EmbeddingSource,
        layer_index: int,
    ) -> torch.Tensor:
        if source == "backbone_last_gap":
            feature_map = self._feature_map_from_backbone(backbone_outputs)
            return pool_feature_map_gap(feature_map)

        if source == "backbone_last_flatten":
            feature_map = self._feature_map_from_backbone(backbone_outputs)
            return feature_map.flatten(1)

        if source == "backbone_hidden_gap":
            if not backbone_outputs.hidden_states:
                raise ValueError("Backbone did not return hidden_states.")
            token_features = backbone_outputs.hidden_states[layer_index]
            return pool_tokens_gap(token_features)

        raise ValueError(f"Unsupported embedding source: {source}")

    @torch.no_grad()
    def extract_embedding(
        self,
        image: Image.Image,
        bbox_xywh: list[float] | None = None,
        source: EmbeddingSource = "backbone_last_gap",
        layer_index: int = -1,
    ) -> np.ndarray:
        """Extract a frame-level embedding from ViTPose backbone features.

        Recommended default: ``backbone_last_gap``.
        This pools the last backbone feature map with GAP — the same tensor
        that is fed into the pose decoder before heatmap prediction.
        """

        if bbox_xywh is None:
            width, height = image.size
            bbox_xywh = full_image_bbox_xywh(width, height)

        pose_inputs = self._build_pose_inputs(image, bbox_xywh)
        backbone_outputs = self._backbone_outputs(pose_inputs)
        embedding = self._embedding_from_backbone(
            backbone_outputs=backbone_outputs,
            source=source,
            layer_index=layer_index,
        )
        return embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def estimate_with_bbox(
        self,
        image: Image.Image,
        bbox_xywh: list[float],
    ) -> tuple[list[float], np.ndarray, np.ndarray]:
        pose_inputs = self._build_pose_inputs(image, bbox_xywh)

        if self.use_dataset_index:
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
    def estimate(
        self,
        image: Image.Image,
    ) -> tuple[list[float] | None, np.ndarray, np.ndarray]:
        width, height = image.size
        return self.estimate_with_bbox(image, full_image_bbox_xywh(width, height))

    @torch.no_grad()
    def estimate_with_embedding(
        self,
        image: Image.Image,
        bbox_xywh: list[float] | None = None,
        source: EmbeddingSource = "backbone_last_gap",
        layer_index: int = -1,
    ) -> tuple[list[float], np.ndarray, np.ndarray, np.ndarray]:
        if bbox_xywh is None:
            width, height = image.size
            bbox_xywh = full_image_bbox_xywh(width, height)

        bbox_xywh, keypoints, scores = self.estimate_with_bbox(image, bbox_xywh)
        embedding = self.extract_embedding(
            image=image,
            bbox_xywh=bbox_xywh,
            source=source,
            layer_index=layer_index,
        )
        return bbox_xywh, keypoints, scores, embedding
