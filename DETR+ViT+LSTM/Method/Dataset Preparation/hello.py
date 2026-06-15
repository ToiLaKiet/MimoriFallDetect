import cv2
import json
import sys
import os

if len(sys.argv) < 2:
    print("Cách dùng: python roi.py path/to/image.jpg")
    sys.exit(1)

image_path = sys.argv[1]

print("Đường dẫn ảnh đang đọc:", image_path)
print("File có tồn tại không?", os.path.exists(image_path))

image = cv2.imread(image_path)

if image is None:
    print("Không đọc được ảnh. Kiểm tra lại đường dẫn hoặc định dạng file.")
    sys.exit(1)

x, y, w, h = cv2.selectROI("Chon vung quan tam", image, showCrosshair=True)

cv2.destroyAllWindows()

bbox = {
    "x": int(x),
    "y": int(y),
    "width": int(w),
    "height": int(h),
    "x1": int(x),
    "y1": int(y),
    "x2": int(x + w),
    "y2": int(y + h)
}

print(json.dumps(bbox, indent=4, ensure_ascii=False))
