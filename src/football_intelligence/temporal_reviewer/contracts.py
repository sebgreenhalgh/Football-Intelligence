"""Strict R6.1 action, request, and filesystem-boundary contracts."""

from __future__ import annotations

from collections.abc import Mapping
import re
from pathlib import Path
from typing import Any
import uuid

MAX_JSON_BODY_BYTES = 2 * 1024 * 1024
R6_ACTION_SCHEMA = "football_intelligence.g7e_b_r6.browser_action.v1"
R6_1_REVIEW_REVISION = "G7E_B_R6_1_FINAL_BYTE_VISUAL_RUNTIME_CLOSURE_V1"

ACTION_ENVELOPE_FIELDS = frozenset(
    {
        "schema_version",
        "action_id",
        "idempotency_key",
        "review_revision",
        "contract_hash",
        "mode",
        "tranche_id",
        "burst_id",
        "expected_draft_revision",
        "expected_draft_sha256",
        "question_instance_key",
        "action_type",
        "payload",
        "client_timestamp",
    }
)
SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_action_uuid(value: Any, field: str) -> str:
    """Return one lowercase canonical UUID string or fail closed."""

    if not isinstance(value, str):
        raise ValueError(f"{field} must be a canonical UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{field} must be a canonical UUID string") from exc
    canonical = str(parsed)
    if value != canonical:
        raise ValueError(f"{field} must use lowercase canonical UUID encoding")
    return canonical


def validate_action_envelope(action: Mapping[str, Any], allowed_actions: set[str] | frozenset[str]) -> tuple[str, str]:
    """Validate fields that must be safe before any action path is resolved."""

    if not isinstance(action, Mapping):
        raise ValueError("action envelope must be a JSON object")
    missing = ACTION_ENVELOPE_FIELDS - set(action)
    if missing:
        raise ValueError(f"action envelope is missing fields: {sorted(missing)}")
    extras = set(action) - ACTION_ENVELOPE_FIELDS
    if extras:
        raise ValueError(f"action envelope contains unsupported fields: {sorted(extras)}")
    if action.get("schema_version") != R6_ACTION_SCHEMA:
        raise ValueError("action schema mismatch")
    action_id = canonical_action_uuid(action.get("action_id"), "action_id")
    idempotency_key = canonical_action_uuid(action.get("idempotency_key"), "idempotency_key")
    if idempotency_key != action_id:
        raise ValueError("idempotency_key must equal action_id")
    if action.get("mode") not in {"real", "practice"}:
        raise ValueError("action mode must be real or practice")
    for field in ("tranche_id", "burst_id"):
        value = action.get(field)
        if not isinstance(value, str) or not SAFE_CASE_ID.fullmatch(value):
            raise ValueError(f"{field} must be a bounded safe identifier")
    for field in ("contract_hash", "expected_draft_sha256"):
        value = action.get(field)
        if not isinstance(value, str) or not SHA256.fullmatch(value):
            raise ValueError(f"{field} must be a lowercase SHA-256")
    revision = action.get("expected_draft_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("expected_draft_revision must be a non-negative integer")
    question_key = action.get("question_instance_key")
    if not isinstance(question_key, str) or not 1 <= len(question_key) <= 512:
        raise ValueError("question_instance_key must be a bounded string")
    if any(ord(character) < 32 for character in question_key):
        raise ValueError("question_instance_key contains control characters")
    timestamp = action.get("client_timestamp")
    if not isinstance(timestamp, str) or not 1 <= len(timestamp) <= 64:
        raise ValueError("client_timestamp must be a bounded string")
    if action.get("action_type") not in allowed_actions:
        raise ValueError("unsupported action type")
    if not isinstance(action.get("payload"), Mapping):
        raise ValueError("action payload must be a JSON object")
    return action_id, idempotency_key


def contained_path(root: Path, *relative_parts: str | Path) -> Path:
    """Resolve a descendant and reject absolute, parent, or escaped paths."""

    resolved_root = root.resolve()
    candidate = resolved_root
    for raw in relative_parts:
        part = Path(raw)
        if part.is_absolute() or any(token in {"", ".", ".."} for token in part.parts):
            raise ValueError("path component is not a safe relative path")
        candidate = candidate / part
    candidate = candidate.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError("resolved path escapes its approved root")
    return candidate


def is_json_content_type(value: str | None) -> bool:
    if not value:
        return False
    media_type, *parameters = [part.strip() for part in value.split(";")]
    if media_type.lower() != "application/json":
        return False
    return all(parameter.lower() in {"charset=utf-8", 'charset="utf-8"'} for parameter in parameters)
