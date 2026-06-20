from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def trigger_agent(context: dict[str, Any]) -> dict[str, Any]:
    """Send fall alert notification. Mail integration to be added later."""
    logger.warning("MimamoriFall alert triggered: %s", context)
    return {
        "sent": False,
        "stub": True,
        "message": "Agent stub — email not configured yet.",
        "context": context,
    }
