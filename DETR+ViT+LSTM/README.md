# DETR + Frozen ViT + LSTM

Pipeline mới:

1. RT-DETR detect người trên frame gốc.
2. Chỉ giữ bbox người có diện tích lớn nhất, crop và ghi `bbox_manifest.csv`.
3. Tạo sequence 10 crop liên tục trong từng Trial/Camera, cùng logic `trial_key_from_path(image_path.parent, image_dir)` của pipeline cũ.
4. Frozen ViTPose backbone trích xuất embedding, không decode keypoints/heatmap.
5. Train LSTM + Linear classifier bằng CrossEntropyLoss. Softmax dùng khi predict/report.

## 1. Extract bbox crops

```bash
python3 'DETR+ViT+LSTM/extract_detr_bboxes.py' \
  --image-dir /path/to/images \
  --manifest-path 'ViT+CNN+LSTM/manifest.csv' \
  --label-col Label \
  --crop-dir 'DETR+ViT+LSTM/detr_crops' \
  --output-manifest 'DETR+ViT+LSTM/bbox_manifest.csv'
```

Frame không detect được người sẽ bị bỏ qua. Bbox được lưu theo crop người lớn nhất.

## 2. Build sequence data

```bash
python3 'DETR+ViT+LSTM/bbox_sequence_data.py' \
  --bbox-manifest 'DETR+ViT+LSTM/bbox_manifest.csv' \
  --image-dir /path/to/images \
  --sequence-length 10 \
  --stride 1 \
  --output 'DETR+ViT+LSTM/bbox_sequence_data.json'
```

Sequence builder đọc lại cột `frame`, normalize timestamp từ filename, resolve frame theo `--image-dir`, rồi group theo Trial/Camera giống hàm `build_manifest_sequence_groups` cũ. Nó không dùng `group_key` có sẵn trong manifest để tránh nối nhầm Trial.

## 3. Train frozen ViT + LSTM

```bash
python3 'DETR+ViT+LSTM/train_detr_vit_lstm.py' \
  --sequence-data 'DETR+ViT+LSTM/bbox_sequence_data.json' \
  --epochs 20
```

Default ViT encoder là ViTPose backbone `usyd-community/vitpose-base-simple`, đã cache sẵn trên máy này. Checkpoint chỉ lưu LSTM/classifier trainable state, không lưu lại toàn bộ frozen ViT.
