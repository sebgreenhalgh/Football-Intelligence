"""Bounded HTTP request and static-path helpers for the local reviewer."""

from __future__ import annotations

import json
from typing import Any

from football_intelligence.temporal_reviewer.contracts import MAX_JSON_BODY_BYTES, is_json_content_type


class UnsupportedMediaType(ValueError):
    pass


class RequestBodyTooLarge(ValueError):
    pass


def read_json_request(handler: Any) -> dict[str, Any]:
    if not is_json_content_type(handler.headers.get("Content-Type")):
        raise UnsupportedMediaType("Content-Type must be application/json with optional UTF-8 charset")
    raw_length = handler.headers.get("Content-Length")
    if raw_length is None:
        raise ValueError("Content-Length is required")
    try:
        length = int(raw_length)
    except ValueError as exc:
        raise ValueError("Content-Length must be an integer") from exc
    if length < 0:
        raise ValueError("Content-Length must not be negative")
    if length > MAX_JSON_BODY_BYTES:
        raise RequestBodyTooLarge(f"request body exceeds {MAX_JSON_BODY_BYTES} bytes")
    data = handler.rfile.read(length)
    if len(data) != length:
        raise ValueError("request body ended before Content-Length")
    value = json.loads(data or b"{}")
    if not isinstance(value, dict):
        raise ValueError("JSON request body must be an object")
    return value
