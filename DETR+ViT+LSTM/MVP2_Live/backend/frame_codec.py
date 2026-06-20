from __future__ import annotations

import base64
import io

from PIL import Image


def encode_rgb_jpeg_base64(image: Image.Image, *, quality: int = 75) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
