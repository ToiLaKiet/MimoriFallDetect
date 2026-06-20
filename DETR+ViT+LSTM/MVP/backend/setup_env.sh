#!/usr/bin/env bash
set -euo pipefail

# Notebook equivalent of:
#   pip install -U openmim
#   mim install mmengine
#   mim install "mmcv==2.1.0"
#   mim install "mmpretrain>=1.0.0"
#   mim install "mmpose>=1.3.0"
#
# IMPORTANT: use Python 3.10 — system Python 3.12 breaks `mim` (setuptools/ImpImporter).

PY="${PY:-$HOME/.pyenv/versions/3.10.16/bin/python}"
MIM="${MIM:-$PY -m mim}"

if [[ ! -x "$PY" ]]; then
  echo "Python 3.10.16 not found at $PY"
  echo "Install with: pyenv install 3.10.16"
  exit 1
fi

echo "Using: $("$PY" --version)"
echo "Python: $PY"
echo

"$PY" -m pip install -U pip "setuptools>=61,<70" wheel
"$PY" -m pip install -q -U openmim

# mim needs torch to pick the correct mmcv wheel index.
"$PY" -m pip install "torch==2.4.1" "torchvision==0.19.1"

"$PY" -m pip install -r requirements.txt

echo ">>> mim install mmengine"
$MIM install mmengine

echo ">>> mim install mmcv==2.1.0"
if ! $MIM install "mmcv==2.1.0"; then
  echo "WARN: mmcv==2.1.0 failed (common on macOS — no prebuilt wheel)."
  echo "      Falling back to mmcv-lite (enough for ViTPose feature extraction)."
  "$PY" -m pip install "mmcv-lite>=2.0.0"
fi

echo ">>> mim install mmpretrain>=1.0.0"
$MIM install "mmpretrain>=1.0.0"

echo ">>> mim install mmpose>=1.3.0"
if ! $MIM install "mmpose>=1.3.0"; then
  echo "WARN: mmpose install failed (often chumpy build on macOS)."
  echo "      Falling back to: pip install mmpose --no-deps"
  "$PY" -m pip install "mmpose>=1.3.0" --no-deps
fi

echo ">>> extra mmpose runtime deps (needed when installed with --no-deps)"
"$PY" -m pip install json-tricks munkres
"$PY" -m pip install xtcocotools --no-build-isolation || true

echo
echo "Verify imports:"
"$PY" - <<'PY'
from mmpose_vitpose_estimator import MMPoseVitPoseEstimator
import mmcv, mmengine, mmpose, mmpretrain

print("mmcv", mmcv.__version__)
print("mmengine", mmengine.__version__)
print("mmpose", mmpose.__version__)
print("mmpretrain ok")
print("MMPoseVitPoseEstimator ok")
PY

echo
echo "Done. Start backend:"
echo "  $PY app.py"
