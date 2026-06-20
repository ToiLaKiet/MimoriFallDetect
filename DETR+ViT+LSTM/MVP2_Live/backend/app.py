from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from config_loader import load_config
from live_session import LiveSession
from pipeline import FallDetectionPipeline

app = Flask(__name__)
CORS(app)

_pipeline: FallDetectionPipeline | None = None
_live_session: LiveSession | None = None
_config_path = Path(
    os.environ.get("MIMAMORIFALL_CONFIG_PATH", Path(__file__).parent / "config.yaml")
)


def get_pipeline(*, load: bool = True) -> FallDetectionPipeline:
    global _pipeline
    if _pipeline is None:
        if not load:
            raise RuntimeError("Pipeline has not been loaded yet.")
        _pipeline = FallDetectionPipeline(load_config(_config_path))
    return _pipeline


@app.get("/api/health")
def health():
    cfg = load_config(_config_path)
    payload: dict = {
        "ok": True,
        "service": "MimamoriFall",
        "config": cfg.to_public_dict(),
        "models_loaded": _pipeline is not None,
    }
    if _pipeline is not None:
        payload.update(_pipeline.status)
    return jsonify(payload)


def _decode_base64_image(image_data: str) -> Image.Image:
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    raw = base64.b64decode(image_data)
    return Image.open(io.BytesIO(raw))


def _decode_image_payload() -> Image.Image:
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image")
        if not image_data:
            raise ValueError("Missing 'image' field (base64 data URL or raw base64).")
        return _decode_base64_image(str(image_data))

    upload = request.files.get("frame") or request.files.get("image")
    if upload is None or not upload.filename:
        raise ValueError("Upload a JPEG/PNG frame as 'frame' or send JSON { image: base64 }.")
    return Image.open(upload.stream)


@app.post("/api/live/start")
def live_start():
    global _live_session
    try:
        pipeline = get_pipeline(load=True)
        _live_session = LiveSession(pipeline)
        return jsonify(
            {
                "ok": True,
                "message": "MimamoriFall live session started.",
                "window_size": pipeline.config.window_size,
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/live/stop")
def live_stop():
    global _live_session
    frames_total = _live_session.frames_total if _live_session is not None else 0
    _live_session = None
    return jsonify({"ok": True, "message": "Live session stopped.", "frames_total": frames_total})


@app.get("/api/live/status")
def live_status():
    if _live_session is None:
        return jsonify({"ok": True, "active": False, "frames_total": 0})
    return jsonify(
        {
            "ok": True,
            "active": True,
            "frames_total": _live_session.frames_total,
            "buffer_size": _live_session.buffer_size,
            "window_size": _live_session.pipeline.config.window_size,
        }
    )


@app.post("/api/live/frame")
def live_frame():
    if _live_session is None:
        return jsonify(
            {"ok": False, "error": "Live session not started. POST /api/live/start first."}
        ), 400
    try:
        image = _decode_image_payload()
        result = _live_session.process_image(image)
        return jsonify({"ok": True, **result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/live/frames")
def live_frames():
    """Process multiple temporally-spaced frames in one request (less HTTP overhead)."""
    if _live_session is None:
        return jsonify(
            {"ok": False, "error": "Live session not started. POST /api/live/start first."}
        ), 400
    try:
        payload = request.get_json(silent=True) or {}
        images = payload.get("images")
        if not isinstance(images, list) or not images:
            raise ValueError("Send JSON { images: [base64, ...] } with at least one frame.")

        result: dict | None = None
        for image_data in images:
            image = _decode_base64_image(str(image_data))
            result = _live_session.process_image(image)

        assert result is not None
        return jsonify({"ok": True, **result, "batch_size": len(images)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5002"))
    app.run(host="0.0.0.0", port=port, debug=True)
