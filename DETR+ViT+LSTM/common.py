"""Shared helpers for the DETR + frozen ViT + LSTM pipeline."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
OLD_PIPELINE_DIR = REPO_DIR / "ViT+CNN+LSTM"


def add_old_pipeline_to_path() -> None:
    """Allow reusing manifest/data utilities from the older pipeline."""

    if str(OLD_PIPELINE_DIR) not in sys.path:
        sys.path.append(str(OLD_PIPELINE_DIR))


def configure_runtime_cache(cache_name: str = "detr-vit-lstm-cache") -> None:
    """Point runtime caches to writable directories before importing transformers."""

    cache_root = Path(tempfile.gettempdir()) / cache_name
    for child in ("matplotlib", "xdg", "xdg/fontconfig"):
        (cache_root / child).mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

    default_hf_home = Path.home() / ".cache" / "huggingface"
    if default_hf_home.exists():
        os.environ.setdefault("HF_HOME", str(default_hf_home))


def choose_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda/mps into a usable torch device."""

    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        requested = "cpu"
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        print("MPS is not available; falling back to CPU.")
        requested = "cpu"

    return torch.device(requested)


def resolve_hf_model_source(model_name_or_path: str, allow_download: bool) -> str:
    """Return a local snapshot path when a HF model is cached and downloads are off."""

    path = Path(model_name_or_path).expanduser()
    if path.exists():
        return str(path)
    if allow_download:
        return model_name_or_path

    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    repo_cache = cache_root / f"models--{model_name_or_path.replace('/', '--')}"
    refs_main = repo_cache / "refs" / "main"
    if refs_main.exists():
        snapshot = repo_cache / "snapshots" / refs_main.read_text().strip()
        if snapshot.exists():
            return str(snapshot)

    snapshots_root = repo_cache / "snapshots"
    if snapshots_root.exists():
        for snapshot in sorted(snapshots_root.iterdir()):
            if (snapshot / "config.json").exists():
                return str(snapshot)

    return model_name_or_path
