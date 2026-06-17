from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


MMPoseEmbeddingSource = Literal[
    "pre_head_gap",
    "pre_head_flatten",
]


@dataclass(frozen=True)
class MMPosePreprocessSpec:
    image_size_hw: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def _to_rgb_uint8(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image array (H,W,3), got shape={rgb.shape}.")
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8, copy=False)
    return rgb


def _normalize_chw_0_255(
    chw_0_255: torch.Tensor,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=chw_0_255.dtype, device=chw_0_255.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=chw_0_255.dtype, device=chw_0_255.device).view(3, 1, 1)
    return (chw_0_255 - mean_t) / std_t


class MMPoseVitPoseEstimator:
    """Extract ViTPose features from MMPose right before the pose head.

    Notes:
    - This class intentionally does NOT implement pose estimation.
    - Embeddings are pooled from `model.extract_feat(inputs)` which is
      backbone (+ neck if any) output, before `head.forward(...)`.
    """

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: torch.device,
    ) -> None:
        self.device = device

        # Ensure mmpretrain modules are registered for ViTPose backbone configs
        # (e.g. type='mmpretrain.VisionTransformer').
        try:
            import mmpretrain  # noqa: F401  # pylint: disable=unused-import
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency `mmpretrain`. Install it to use ViTPose in MMPose."
            ) from exc

        try:
            from mmpose.apis import init_model  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Missing dependency `mmpose`. Install MMPose v1.x.") from exc

        self.model = init_model(
            str(config_path),
            str(checkpoint_path),
            device=str(device),
        )
        self.model.eval()

        self.preprocess = self._resolve_preprocess_spec()

    def _resolve_preprocess_spec(self) -> MMPosePreprocessSpec:
        # Prefer model cfg values when available (MMPose 1.x style).
        cfg = getattr(self.model, "cfg", None)
        if cfg is None:
            # Fallback to common defaults used in official configs.
            return MMPosePreprocessSpec(
                image_size_hw=(256, 192),
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
            )

        # image size is defined in ViTPose backbone config as img_size=(H, W)
        image_size_hw = None
        try:
            backbone = cfg.model.backbone
            img_size = getattr(backbone, "img_size", None) or backbone.get("img_size")
            if img_size is not None:
                image_size_hw = (int(img_size[0]), int(img_size[1]))
        except Exception:
            image_size_hw = None

        # mean/std are defined in PoseDataPreprocessor and are in 0..255 domain.
        mean = None
        std = None
        try:
            dp = cfg.model.data_preprocessor
            mean_v = getattr(dp, "mean", None) or dp.get("mean")
            std_v = getattr(dp, "std", None) or dp.get("std")
            if mean_v is not None:
                mean = (float(mean_v[0]), float(mean_v[1]), float(mean_v[2]))
            if std_v is not None:
                std = (float(std_v[0]), float(std_v[1]), float(std_v[2]))
        except Exception:
            mean, std = None, None

        return MMPosePreprocessSpec(
            image_size_hw=image_size_hw or (256, 192),
            mean=mean or (123.675, 116.28, 103.53),
            std=std or (58.395, 57.12, 57.375),
        )

    def _build_inputs(self, image: Image.Image) -> torch.Tensor:
        # Hàm này để chuyển đổi ảnh từ định dạng PIL sang tensor torch và chuẩn hóa.
        # Ngoài ra, nếu ảnh ko đúng kích thước, nó sẽ resize lại bằng bilinear interpolation.
        rgb = _to_rgb_uint8(image)
        chw = torch.from_numpy(rgb).permute(2, 0, 1).to(dtype=torch.float32)  # (3,H,W), 0..255
        chw = chw.to(self.device)

        target_h, target_w = self.preprocess.image_size_hw
        if (chw.shape[1], chw.shape[2]) != (target_h, target_w):
            chw = F.interpolate(
                chw.unsqueeze(0),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        chw = _normalize_chw_0_255(chw, mean=self.preprocess.mean, std=self.preprocess.std)
        return chw.unsqueeze(0)  # (1,3,H,W)

    @torch.no_grad()
    def extract_embedding(
        self,
        image: Image.Image,
        bbox_xywh: list[float] | None = None,
        source: MMPoseEmbeddingSource = "pre_head_gap",
    ) -> np.ndarray:
        """Extract an embedding pooled from features right before the pose head.

        Args:
            image: PIL image. Ideally a pre-cropped person image.
            bbox_xywh: Accepted for API parity; ignored in this estimator.
            source:
              - pre_head_gap: global-average-pool feature map -> (C,)
              - pre_head_flatten: flatten feature map -> (C*H*W,)
        """

        _ = bbox_xywh  # bbox not used; image is assumed already cropped.
        inputs = self._build_inputs(image)

        feats = self.model.extract_feat(inputs)
        feat = feats[-1] if isinstance(feats, (tuple, list)) else feats

        if feat.ndim == 4:  # (B,C,H,W)
            if source == "pre_head_gap":
                emb = feat.mean(dim=(-2, -1)) # Shape : (B,C)
            elif source == "pre_head_flatten":
                emb = feat.flatten(1) # Shape : (B,C*H*W)
            else:
                raise ValueError(f"Unsupported embedding source: {source}")
        elif feat.ndim == 3:  # (B,T,C) token features, T is the number of tokens.
            if source == "pre_head_gap":
                emb = feat.mean(dim=1) # Shape : (B,C)
            elif source == "pre_head_flatten":
                emb = feat.flatten(1) # Shape : (B,T*C) 
            else:
                raise ValueError(f"Unsupported embedding source: {source}")
        else:
            raise ValueError(f"Unexpected feature shape from extract_feat: {tuple(feat.shape)}")

        return emb.squeeze(0).detach().cpu().numpy().astype(np.float32)

