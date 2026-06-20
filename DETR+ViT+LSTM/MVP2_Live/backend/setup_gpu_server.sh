#!/usr/bin/env bash
set -euo pipefail

# Setup Python 3.10.16 virtualenv and install MVP2_Live backend dependencies.
# Optimized for macOS Apple Silicon (M1/M2/M3)
#
# Usage:
#   ./setup.sh
#
# Override Python binary:
#   PY=/path/to/python3.10 ./setup.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"
PY="${PY:-$HOME/.pyenv/versions/3.10.16/bin/python}"

if [[ ! -f "$REQUIREMENTS" ]]; then
  echo "requirements.txt not found at $REQUIREMENTS"
  exit 1
fi

if [[ ! -x "$PY" ]]; then
  echo "Python 3.10.16 not found at $PY"
  echo "Install with: pyenv install 3.10.16"
  exit 1
fi

PY_VERSION="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
if [[ "$PY_VERSION" != "3.10.16" ]]; then
  echo "Expected Python 3.10.16, got $PY_VERSION ($PY)"
  exit 1
fi

echo "Using: $("$PY" --version)"
echo "Python: $PY"
echo "Venv:   $VENV_DIR"
echo

if [[ ! -d "$VENV_DIR" ]]; then
  echo ">>> Creating virtualenv"
  "$PY" -m venv "$VENV_DIR"
fi

VPY="$VENV_DIR/bin/python"
MIM="$VPY -m mim"

# Nâng cấp các công cụ đóng gói cơ bản
"$VPY" -m pip install -U pip "setuptools>=61,<70" wheel
"$VPY" -m pip install -q -U openmim==0.3.9

# 1. SỬA LỖI: Cài đặt đúng phiên bản PyTorch thực tế (Bản CPU/MPS ổn định cho Mac)
echo ">>> Cài đặt PyTorch và Torchvision ổn định cho macOS"
"$VPY" -m pip install "torch==2.1.2" "torchvision==0.16.2"

# Thiết lập các biến môi trường để hỗ trợ compile trên Mac ARM64 nếu cần
export MACOSX_DEPLOYMENT_TARGET=10.15

# 2. THAY ĐỔI THỨ TỰ: Cài đặt các gói OpenMMLab qua MIM TRƯỚC để tránh xung đột pip
echo ">>> mim install mmengine"
$MIM install mmengine==0.10.7

echo ">>> mim install mmcv==2.1.0"
if ! $MIM install "mmcv==2.1.0"; then
  echo "WARN: mmcv==2.1.0 biên dịch thất bại (Phổ biến trên macOS do thiếu Clang/Ninja)."
  echo "      Đang chuyển hướng cài đặt mmcv-lite..."
  "$VPY" -m pip install "mmcv-lite>=2.0.0"
fi

echo ">>> mim install mmpretrain==1.2.0"
$MIM install "mmpretrain==1.2.0"

echo ">>> mim install mmpose==1.3.2"
# Khắc phục lỗi cài đặt chumpy/mmpose trên macOS
if ! $MIM install "mmpose==1.3.2"; then
  echo "WARN: mmpose cài đặt lỗi (thường do lỗi build wheel của thư viện chumpy trên macOS)."
  echo "      Đang cài đặt cưỡng bức qua: pip install mmpose==1.3.2 --no-deps"
  "$VPY" -m pip install "mmpose==1.3.2" --no-deps
fi

echo ">>> Cài đặt bổ sung các thư viện runtime còn thiếu của mmpose"
"$VPY" -m pip install json-tricks munkres
# xtcocotools thường lỗi cô lập trên Mac, cần thêm thuộc tính --no-build-isolation
"$VPY" -m pip install xtcocotools --no-build-isolation || true

# 3. CÀI ĐẶT FILE REQUIREMENTS (Lúc này các thư viện lớn đã được xử lý xong)
echo ">>> pip install -r requirements.txt"
"$VPY" -m pip install -r "$REQUIREMENTS"

echo
echo "=== Kiểm tra import các thư viện ==="
"$VPY" - <<'PY'
try:
    import mmengine, mmpose, mmpretrain
    import flask, torch, ultralytics
    # Khối kiểm tra mmcv/mmcv-lite độc lập
    try:
        import mmcv
        print("mmcv", mmcv.__version__)
    except ImportError:
        import mmcv_lite
        print("mmcv-lite đã được nạp thay thế thành công")
        
    print("mmengine", mmengine.__version__)
    print("mmpose", mmpose.__version__)
    print("mmpretrain ok")
    print("torch", torch.__version__)
    print("flask ok")
    print("ultralytics ok")
    print("\n🎉 MÔI TRƯỜNG ĐÃ SẴN SÀNG CHẠY!")
except Exception as e:
    print(f"❌ Lỗi kiểm tra môi trường: {e}")
PY

echo
echo "Done. Kích hoạt môi trường và chạy dự án bằng lệnh:"
echo "  source $VENV_DIR/bin/activate"
echo "  python $SCRIPT_DIR/app.py"