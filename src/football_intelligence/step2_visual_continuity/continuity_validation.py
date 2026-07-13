from __future__ import annotations

from collections import Counter
from typing import Any

FORBIDDEN_KEYS = {
    "identity_id",
    "persistent_player_id",
    "confirmed_player_id",
    "player_slot_id",
    "goalkeeper_slot_id",
}


def _find_forbidden(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_KEYS:
                found.add(key)
            found.update(_find_forbidden(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_find_forbidden(child))
    return found


def validate_continuity_payload(candidate_payload: dict[str, Any], *, max_degree: int = 3) -> dict[str, Any]:
    rows = candidate_payload.get("rows", [])
    source_degree = Counter(str(row.get("source_visible_person_base_id")) for row in rows)
    target_degree = Counter(str(row.get("target_visible_person_base_id")) for row in rows)
    forbidden = sorted(_find_forbidden(candidate_payload))
    non_identity_flags_ok = all(
        row.get("visual_continuity_is_real_identity") is False
        and row.get("visual_continuity_is_player_slot") is False
        and row.get("visual_continuity_is_metric") is False
        for row in rows
    )
    return {
        "artifact": "m5_4d_continuity_validation",
        "passed": not forbidden
        and non_identity_flags_ok
        and max(source_degree.values() or [0]) <= max_degree
        and max(target_degree.values() or [0]) <= max_degree,
        "forbidden_keys": forbidden,
        "non_identity_flags_ok": non_identity_flags_ok,
        "max_source_candidate_degree": max(source_degree.values() or [0]),
        "max_target_candidate_degree": max(target_degree.values() or [0]),
        "broad_all_pairs_graph_prohibited": True,
    }
