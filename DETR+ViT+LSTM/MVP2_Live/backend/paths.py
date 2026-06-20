from __future__ import annotations

import sys
from pathlib import Path

MVP2_BACKEND_DIR = Path(__file__).resolve().parent
MVP2_DIR = MVP2_BACKEND_DIR.parent
PROJECT_ROOT = MVP2_DIR.parent
MVP_BACKEND_DIR = PROJECT_ROOT / "MVP" / "backend"

MODEL_DIR = PROJECT_ROOT / "Method" / "Model"
BBOX_DIR = PROJECT_ROOT / "Method" / "Dataset Preparation" / "2. BBox Detection"
VITPOSE_DIR = PROJECT_ROOT / "Method" / "Dataset Preparation" / "4. ViTPose Embeddings"
MMPOSE_DIR = VITPOSE_DIR / "MMPose"

for path in (MODEL_DIR, BBOX_DIR, VITPOSE_DIR, MMPOSE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
