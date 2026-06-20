from __future__ import annotations

import base64
import io
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename

from config_loader import load_config
from live_session import LiveSession
from pipeline import FallDetectionPipeline, IMAGE_EXTENSIONS

app = Flask(__name__)
CORS(app)

_pipeline: FallDetectionPipeline | None = None
_live_session: LiveSession | None = None
_config_path = Path(os.environ.get("MVP_CONFIG_PATH", Path(__file__).parent / "config.yaml"))


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
    payload: dict = {"ok": True, "config": cfg.to_public_dict(), "models_loaded": _pipeline is not None}
    if _pipeline is not None:
        payload.update(_pipeline.status)
    return jsonify(payload)


@app.get("/api/config")
def config():
    cfg = load_config(_config_path)
    return jsonify(cfg.to_public_dict())


@app.post("/api/reload")
def reload_models():
    global _pipeline
    try:
        cfg = load_config(_config_path)
        if _pipeline is None:
            _pipeline = FallDetectionPipeline(cfg)
        else:
            _pipeline.reload(cfg)
        return jsonify({"ok": True, **_pipeline.status})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


def _save_uploaded_images(temp_dir: Path) -> list[Path]:
    saved: list[Path] = []

    if "folder_path" in request.form and request.form["folder_path"].strip():
        folder = Path(request.form["folder_path"].strip()).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(f"Folder not found: {folder}")
        saved = [
            path
            for path in sorted(folder.iterdir(), key=lambda item: item.name)
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not saved:
            raise ValueError(f"No supported images found in {folder}")
        return saved

    files = request.files.getlist("images")
    if not files:
        raise ValueError("Upload at least one image via 'images', or provide 'folder_path'.")

    for upload in files:
        if not upload or not upload.filename:
            continue
        filename = secure_filename(upload.filename)
        if not filename:
            continue
        suffix = Path(filename).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS and not filename.lower().endswith(".zip"):
            continue

        target = temp_dir / filename
        upload.save(target)
        if target.suffix.lower() == ".zip":
            extract_dir = temp_dir / f"{target.stem}_extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(target, "r") as archive:
                archive.extractall(extract_dir)
            for path in sorted(extract_dir.rglob("*"), key=lambda item: item.name):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    saved.append(path)
        else:
            saved.append(target)

    saved = sorted(saved, key=lambda path: path.name)
    if not saved:
        raise ValueError("No supported image files found in upload.")
    return saved


@app.post("/api/process")
def process():
    temp_dir = Path(tempfile.mkdtemp(prefix="mvp_frames_"))
    try:
        image_paths = _save_uploaded_images(temp_dir)
        pipeline = get_pipeline(load=True)
        result = pipeline.process_images(image_paths)
        return jsonify({"ok": True, **result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _decode_image_payload() -> Image.Image:
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image")
        if not image_data:
            raise ValueError("Missing 'image' field (base64 data URL or raw base64).")
        if isinstance(image_data, str) and "," in image_data:
            image_data = image_data.split(",", 1)[1]
        raw = base64.b64decode(image_data)
        return Image.open(io.BytesIO(raw))

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
                "message": "Live session started.",
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
        return jsonify({"ok": False, "error": "Live session not started. POST /api/live/start first."}), 400
    try:
        image = _decode_image_payload()
        result = _live_session.process_image(image)
        return jsonify({"ok": True, **result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)
