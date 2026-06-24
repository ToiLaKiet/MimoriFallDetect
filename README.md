# MimamoriFall — Hệ thống phát hiện ngã

Dự án nghiên cứu và triển khai hệ thống **phát hiện ngã (fall detection)** từ video/camera, phục vụ giám sát người cao tuổi hoặc môi trường chăm sóc. Hệ thống kết hợp ba mô hình deep learning theo pipeline:

```
Camera / Video  →  RT-DETR-X (phát hiện người)  →  ViTPose (trích embedding tư thế)  →  LSTM (phân loại Fall / Normal)
```

Dữ liệu huấn luyện chủ yếu lấy từ bộ **HAR-UP** (Human Activity Recognition — University of Porto), được thu thập qua các script trong thư mục `crawler/`.

---

## Cấu trúc thư mục

```
job/
├── DETR+ViT+LSTM/     # Pipeline chính: RT-DETR + ViTPose + LSTM
├── ViT+CNN+LSTM/      # Phiên bản thử nghiệm sớm: ViTPose skeleton + CNN + LSTM
├── crawler/           # Thu thập & tải dataset HAR-UP
├── paper/             # Tài liệu nghiên cứu liên quan
├── plot.py            # Vẽ confusion matrix & tính metric đánh giá
├── requirements.txt   # Dependencies dùng chung (training, crawler)
└── note.txt           # Ghi chú nhanh về các script
```

---

## `DETR+ViT+LSTM/` — Pipeline chính

Đây là nhánh phát triển chính, từ chuẩn bị dữ liệu đến ứng dụng realtime.

### `Method/` — Quy trình nghiên cứu & huấn luyện

| Thư mục | Mục đích |
|---------|----------|
| `Dataset Preparation/0. Labeling Timestamps/` | Gán nhãn fall/normal theo timestamp từ file CSV |
| `Dataset Preparation/1. Manifest Creation/` | Tạo file manifest ánh xạ ảnh ↔ nhãn ↔ timestamp |
| `Dataset Preparation/2. BBox Detection/` | Phát hiện bounding box người bằng RT-DETR-X hoặc YOLO |
| `Dataset Preparation/3. Sequences Dataset/` | Ghép chuỗi frame, crop ảnh người, augment dữ liệu |
| `Dataset Preparation/4. ViTPose Embeddings/` | Trích vector embedding từ ViTPose cho từng frame |
| `imageonly_embedded_dataset/` | Dataset đã xử lý sẵn (train / val / test) |
| `Model/` | Định nghĩa mô hình LSTM, loader dữ liệu và script `train.py` |

### `runs1/` … `runs5/` — Kết quả huấn luyện

Mỗi thư mục lưu checkpoint của một lần chạy thử nghiệm:

- `best.pt` — checkpoint tốt nhất
- `last.pt` — checkpoint cuối cùng
- `scaler.npz` — tham số chuẩn hóa embedding
- `history.jsonl` — log quá trình train

Checkpoint đang dùng cho inference: `runs5/best.pt`.

### `MVP/` — Ứng dụng web demo (batch + live camera)

Ứng dụng web để thử nghiệm pipeline trên ảnh/video hoặc camera trực tiếp.

| Thành phần | Mô tả |
|------------|-------|
| `backend/` | API Flask: load model, xử lý ảnh theo batch hoặc từng frame live |
| `frontend/` | Giao diện React (Vite): upload ảnh, xem kết quả, demo camera |

Pipeline inference: RT-DETR → crop người → ViTPose embedding → buffer 10 frame → LSTM phân loại.

### `MVP2_Live/` — Hệ thống cảnh báo ngã realtime

Phiên bản nâng cấp của MVP, tập trung vào **cảnh báo thời gian thực**:

- Backend Flask (port 5002) + Frontend React (port 5174)
- FSM (finite state machine) xử lý logic cảnh báo: phát hiện fall → theo dõi ổn định bbox 5 giây → kích hoạt agent gửi thông báo
- Chi tiết cài đặt và API: xem [`DETR+ViT+LSTM/MVP2_Live/README.md`](DETR+ViT+LSTM/MVP2_Live/README.md)

---

## `ViT+CNN+LSTM/` — Phiên bản thử nghiệm sớm

Hướng tiếp cận ban đầu: trích skeleton từ ViTPose, vẽ lên nền đen, đưa vào **CNN + LSTM** để phân loại.

| File / thư mục | Mục đích |
|----------------|----------|
| `prepare_labels.py` | Gán nhãn từ timestamp trong CSV |
| `extract_vitpose_skeletons.py` | Trích xuất ảnh skeleton |
| `manifestcreation.ipynb` | Tạo manifest mapping nhãn ↔ ảnh |
| `sequence_data.py` | Load và chuẩn bị chuỗi frame cho training |
| `model.py` | Định nghĩa `SkeletonImageLSTMClassifier` (CNN + LSTM) |
| `train_vitpose_lstm.py` | Script huấn luyện và inference |
| `utils.py` | Hàm tiện ích train/evaluate |
| `mvp/` | Demo realtime qua OpenCV (camera → skeleton overlay → classifier) |

Xem hướng dẫn chạy MVP: [`ViT+CNN+LSTM/mvp/README.md`](ViT+CNN+LSTM/mvp/README.md).

---

## `crawler/` — Thu thập dataset HAR-UP

Script tự động crawl và tải dữ liệu từ trang HAR-UP:

| File | Mục đích |
|------|----------|
| `crawl_har_up.py` | Crawl link dataset từ website |
| `crawl_csv_har_up.py` | Crawl link file CSV |
| `download_har_up_datasets.py` | Tải file CSV theo danh sách trong `har_up_dataset_links.json` |
| `har_up_links.json` / `har_up_dataset_links.json` | Danh sách link đã crawl |

---

## `paper/`

Chứa tài liệu nghiên cứu liên quan (`upfall.pdf`).

---

## File ở thư mục gốc

| File | Mục đích |
|------|----------|
| `requirements.txt` | Dependencies Python dùng chung (PyTorch, transformers, ultralytics, selenium, …) |
| `plot.py` | Vẽ heatmap confusion matrix và tính Accuracy / Precision / Recall / F1 cho lớp Fall |
| `confusion_matrix.png` | Kết quả đánh giá đã xuất |
| `note.txt` | Ghi chú nhanh về vai trò các script trong `ViT+CNN+LSTM/` |

---

## Luồng làm việc tổng quát

```
1. Thu thập dữ liệu          crawler/
2. Gán nhãn & tạo manifest   Method/Dataset Preparation/  (hoặc ViT+CNN+LSTM/)
3. Phát hiện người (bbox)    RT-DETR-X
4. Trích embedding tư thế    ViTPose
5. Huấn luyện LSTM           Method/Model/train.py  →  runs*/
6. Triển khai inference      MVP/  hoặc  MVP2_Live/
```

---

## Yêu cầu hệ thống

- Python 3.10+
- PyTorch (CUDA / MPS / CPU)
- OpenMMLab stack cho ViTPose (xem `DETR+ViT+LSTM/MVP/backend/setup_env.sh`)
- Node.js 18+ (cho frontend MVP / MVP2_Live)
