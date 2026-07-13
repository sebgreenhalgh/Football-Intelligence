from __future__ import annotations

from pathlib import Path
from typing import Any

from football_intelligence.review.schemas import safety_payload, utc_now
from football_intelligence.review_chassis.persistence import atomic_write_json


def confirm_smoke(
    *,
    stage_root: Path,
    passed: bool,
    failed: bool,
    reason: str | None = None,
    reviewer_session_id: str = "local-gif-smoke",
) -> dict[str, Any]:
    if passed == failed:
        raise ValueError("provide exactly one of --passed or --failed")
    path = stage_root.resolve() / "continuity_v5" / "smoke_test" / "smoke_test_confirmation.json"
    payload = {
        "schema_version": "football_intelligence.review_chassis.smoke_confirmation.v1",
        "created_at": utc_now(),
        "gif_browser_smoke_passed": passed,
        "gif_browser_smoke_failed": failed,
        "reason": reason,
        "reviewer_session_id": reviewer_session_id,
        **safety_payload(),
    }
    atomic_write_json(path, payload)
    return payload
