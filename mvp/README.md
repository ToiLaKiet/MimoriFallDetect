# ViTPose Realtime MVP

MVP này đọc camera máy, dùng RT-DETR để detect người, chạy ViTPose realtime, rồi tách thành 2 luồng ảnh:

- Demo: vẽ skeleton trực tiếp lên ảnh camera.
- Model input: vẽ skeleton lên ảnh nền đen, đẩy vào buffer 10 frame và chạy checkpoint `vitpose_lstm_best.pt`.

## Chạy

```bash
cd /Users/buithianhdao/job
python3 mvp/run_realtime.py
```

Phím `q` để thoát ở cửa sổ demo.

Mặc định demo chạy `--async-inference`: camera/preview nhắm tới 30 FPS, còn DETR + ViTPose + classifier chạy ở worker riêng và chỉ xử lý frame mới nhất. Status/log sẽ có:

- `FPS`: FPS của preview camera.
- `infer`: FPS inference thật của model.

Muốn so sánh với vòng lặp đồng bộ cũ:

```bash
python3 mvp/run_realtime.py --no-async-inference
```

Muốn đổi FPS camera/preview được yêu cầu:

```bash
python3 mvp/run_realtime.py --target-fps 30
```

Mặc định script dùng model detector `PekingU/rtdetr_r50vd_coco_o365` và ViTPose `usyd-community/vitpose-base-simple` đã cache sẵn. Nếu cần cho phép tải model khi máy chưa có cache:

```bash
python3 mvp/run_realtime.py --allow-download
```

## Output

- Cửa sổ `demo_camera_skeleton`: ảnh camera kèm skeleton overlay, bbox và status.
- File `mvp/out/latest_skeleton.jpg`: skeleton nền đen mới nhất, được ghi đè liên tục.
- Buffer model trong RAM: `deque(maxlen=10)`, frame mới vào thì frame cũ nhất tự bị đẩy ra.

## Tái sử dụng hàm

Core nằm ở `mvp/realtime_core.py`, demo runner chỉ import lại các hàm/class:

```python
from realtime_core import (
    VitPoseRunner,
    draw_skeleton_overlay,
    draw_skeleton_on_black,
    fall_label_for_class,
    prepare_classifier_tensor,
)
```

## Tuỳ chọn hữu ích

```bash
python3 mvp/run_realtime.py --device cpu
python3 mvp/run_realtime.py --camera 1
python3 mvp/run_realtime.py --target-fps 30
python3 mvp/run_realtime.py --no-async-inference
python3 mvp/run_realtime.py --det-threshold 0.6
python3 mvp/run_realtime.py --max-persons 2
python3 mvp/run_realtime.py --headless
python3 mvp/run_realtime.py --headless --max-frames 30
python3 mvp/run_realtime.py --no-save-latest
```

Nếu RT-DETR không tìm thấy người trong frame, code fallback về bbox full-frame để ViTPose vẫn có input.

Classifier được map nhị phân theo class index: `0-4` là `Fall`, còn `5+` là `No Fall`.
