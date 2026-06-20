from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


def _public_context(context: dict[str, Any]) -> dict[str, Any]:
    public = {k: v for k, v in context.items() if k != "frame_image_base64"}
    if "frame_image_base64" in context:
        public["frame_image_included"] = True
        public["frame_image_base64_length"] = len(context["frame_image_base64"])
    return public


def _log_context(context: dict[str, Any]) -> dict[str, Any]:
    return _public_context(context)


def send_telegram_message(context: dict[str, Any]) -> bool:
    """Send alert payload (including cached frame) to n8n webhook."""
    url = "https://voanhkiet05.app.n8n.cloud/webhook-test/4ab931a2-afe2-422e-8e29-3403c22ee5d0"
    payload = {
        "System": "Telegram",
        "title": "🚨 FALL DETECTED",
        "short_message": "⚠️ Immediate attention required",
        "message": f"Fall alert: {context.get('frame_name', 'unknown')}",
        "frame_image_base64": context.get("frame_image_base64", ""),
        "frame_image_width": context.get("frame_image_width", 0),
        "frame_image_height": context.get("frame_image_height", 0),
        "context": context,
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers, timeout=30)
    logger.info("n8n webhook status=%s body=%s", response.status_code, response.text[:200])
    response.raise_for_status()
    return True


def trigger_agent(context: dict[str, Any]) -> dict[str, Any]:
    """Send fall alert notification via n8n webhook."""
    logger.warning("MimamoriFall alert triggered: %s", _log_context(context))

    sent = False
    error: str | None = None
    try:
        send_telegram_message(context)
        sent = True
    except requests.RequestException as exc:
        error = str(exc)
        logger.exception("n8n webhook failed")

    return {
        "sent": sent,
        "stub": not sent,
        "message": "Alert sent to n8n." if sent else error or "n8n webhook failed.",
        "context": _public_context(context),
    }
