#!/usr/bin/env bash
set -euo pipefail

# Setup Python 3.10.16 virtualenv for macOS (Intel & Apple Silicon M1/M2/M3)
# Tối ưu thứ tự cài đặt chặn đứng toàn bộ lỗi cô lập build wheel.
#
# Usage:
#   ./setup_macos.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"
PY="${PY:-$HOME/.pyenv/versions/3.10.16/bin/python}"

# 1. Kiểm tra Hệ điều hành
if [[ "$OSTYPE" != "darwin"* ]]; then
  echo "❌ Lỗi: Script này chỉ dành riêng cho hệ điều hành macOS!"
  exit 1
fi

if [[ ! -f "$REQUIREMENTS" ]]; then
  echo "❌ Không tìm thấy file requirements.txt tại: $REQUIREMENTS"
  exit 1
fi

if [[ ! -x "$PY" ]]; then
  echo "❌ Không tìm thấy Python 3.10.16 tại: $PY"
  echo "👉 Hãy cài đặt trước bằng lệnh: pyenv install 3.10.16"
  exit 1
fi

ARCH_NAME=$(uname -m)
echo "=== 🍎 Khởi chạy cấu hình Môi trường macOS ($ARCH_NAME) ==="
echo "Using: $("$PY" --version)"
echo "Python: $PY"
echo "Venv:   $VENV_DIR"
echo

# 2. Cấu hình biến môi trường Biên dịch cho Mac
export MACOSX_DEPLOYMENT_TARGET="10.15"
if [[ "$ARCH_NAME" == "arm64" ]]; then
  export CFLAGS="-O2 -Wall -arch arm64"
  export CXXFLAGS="-O2 -Wall -arch arm64"
  export LDFLAGS="-arch arm64"
fi

# 3. Khởi tạo môi trường ảo venv
if [[ ! -d "$VENV_DIR" ]]; then
  echo ">>> [1/5] Đang tạo môi trường ảo virtualenv..."
  "$PY" -m venv "$VENV_DIR"
fi

VPY="$VENV_DIR/bin/python"
MIM="$VENV_DIR/bin/mim"

# Nâng cấp công cụ đóng gói nền tảng trước
"$VPY" -m pip install -U pip "setuptools>=61,<70" wheel
"$VPY" -m pip install -q -U openmim==0.3.9

# 4. Cài đặt các thư viện nền móng và file requirements trước để có sẵn NumPy 1.26.4
echo ">>> [2/5] Cài đặt file requirements.txt (NumPy, OpenCV, Flask, Thư viện bổ trợ)..."
"$VPY" -m pip install -r "$REQUIREMENTS"

# 5. Fix môi trường build bằng cách cài đè chumpy/pycocotools không cô lập
echo ">>> [3/5] Cấu hình cố định môi trường build cho cấu trúc Mac..."
"$VPY" -m pip install chumpy==0.70 --no-build-isolation
"$VPY" -m pip install pycocotools==2.0.11 --no-build-isolation

# 6. Cài đặt hệ sinh thái OpenMMLab thông qua MIM (Lúc này đã có sẵn NumPy nền)
echo ">>> [4/5] Cài đặt cấu trúc OpenMMLab bằng MIM..."
$MIM install mmengine==0.10.7

echo ">>> Cài đặt mmcv==2.1.0..."
if ! $MIM install "mmcv==2.1.0"; then
  echo "⚠️ Biên dịch mmcv gốc thất bại. Tự động fallback sang mmcv-lite..."
  "$VPY" -m pip install "mmcv-lite>=2.0.0"
fi

echo ">>> Cài đặt mmpretrain..."
$MIM install "mmpretrain==1.2.0"

echo ">>> Cài đặt mmpose..."
if ! $MIM install "mmpose==1.3.2"; then
  echo "⚠️ Mmpose cài đặt gốc lỗi. Tiến hành cài bypass không check deps..."
  "$VPY" -m pip install "mmpose==1.3.2" --no-deps
fi

# Cài đặt bù các gói runtime bổ sung
"$VPY" -m pip install json-tricks munkres

echo
echo "=== 🔍 Tiến hành kiểm tra Imports tự động ==="
"$VPY" - <<'PY'
import sys
print(f"Python hoạt động: {sys.version}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"👉 Hỗ trợ tăng tốc GPU Mac (MPS): {torch.backends.mps.is_available()}")
    
    import mmengine, mmpose, mmpretrain, flask, ultralytics, pycocotools
    try:
        import mmcv
        print(f"✅ mmcv: {mmcv.__version__}")
    except ImportError:
        import mmcv_lite
        print("✅ mmcv-lite: Đã nạp thành công để thay thế")
        
    print(f"✅ mmengine: {mmengine.__version__}")
    print(f"✅ mmpose: {mmpose.__version__}")
    print(f"✅ pycocotools: tích hợp thành công")
    print("✅ mmpretrain, flask, ultralytics: OK")
    print("\n🎉 XUẤT SẮC: MÔI TRƯỜNG DỰ ÁN CỦA BẠN ĐÃ ĐƯỢC SETUP HOÀN TOÀN THÀNH CÔNG!")
except Exception as e:
    print(f"❌ Lỗi kiểm tra thư viện: {e}")
PY

echo
echo "Done! Kích hoạt môi trường ảo bằng lệnh:"
echo "  source $VENV_DIR/bin/activate"