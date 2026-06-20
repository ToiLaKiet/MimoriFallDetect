# MimamoriFall — Live Fall Alert System

Real-time fall detection pipeline: **RT-DETR-X** → **ViTPose embedding** → **LSTM** with alert FSM (fall→normal + 5s bbox stability).

## Prerequisites

- Python 3.10+ with PyTorch, OpenMMLab stack (see [`../MVP/backend/setup_env.sh`](../MVP/backend/setup_env.sh))
- Node.js 18+
- Model files (shared with MVP, not duplicated):
  - `../../MVP/backend/rtdetr-x.pt`
  - `../../MVP/backend/vitpose/` (config + `.pth` checkpoint)
  - `../../runs5/best.pt` (LSTM checkpoint)

## Backend (port 5002)

```bash
cd DETR+ViT+LSTM/MVP2_Live/backend
pip install -r requirements.txt
python app.py
```

Health check: `http://127.0.0.1:5002/api/health`

## Frontend (port 5174)

```bash
cd DETR+ViT+LSTM/MVP2_Live/frontend
npm install
npm run dev
```

Open `http://localhost:5174`

## Alert logic

1. LSTM predicts **fall** → state `falling`
2. LSTM switches to **normal** → start 5-second monitoring timer
3. If bbox stays stable for 5 seconds → `trigger_agent()` (mail stub)
4. 30-second cooldown before next alert

Configure thresholds in [`backend/config.yaml`](backend/config.yaml).

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Service status |
| POST | `/api/live/start` | Start session |
| POST | `/api/live/stop` | Stop session |
| POST | `/api/live/frame` | Process one frame `{ "image": "data:image/jpeg;base64,..." }` |
