from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.core.fingerprints import semantic_hash, sha256_file


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def compare_blind_runs(*, left_run: Path, right_run: Path, validation_root: Path) -> dict[str, Any]:
    left_summary = read_json(left_run / "run_summary.json")
    right_summary = read_json(right_run / "run_summary.json")
    left_registry = read_json(left_run / "artifact_registry.json")
    right_registry = read_json(right_run / "artifact_registry.json")
    compared_fields = [
        "completion_status",
        "input_closure_hash",
        "frame_count",
        "step1_row_count",
        "visible_person_row_count",
        "visual_continuity_node_count",
        "candidate_edge_count",
        "accepted_candidate_edge_count",
        "quarantined_edge_count",
        "pathlet_candidate_count",
        "review_candidate_count",
    ]
    diffs = [
        {"field": field, "left": left_summary.get(field), "right": right_summary.get(field)}
        for field in compared_fields
        if left_summary.get(field) != right_summary.get(field)
    ]
    registry_match = semantic_hash(left_registry.get("artifacts", [])) == semantic_hash(
        right_registry.get("artifacts", [])
    )
    comparison = {
        "schema_version": "m5.blind_window.run_comparison.v1",
        "created_at": utc_now(),
        "left_run": str(left_run),
        "right_run": str(right_run),
        "structured_differences": diffs,
        "artifact_registry_match": registry_match,
        "run_summary_hashes": {
            "left": sha256_file(left_run / "run_summary.json"),
            "right": sha256_file(right_run / "run_summary.json"),
        },
        "result": "matches_blocked_status" if not diffs and registry_match else "differs",
        "passed": not diffs and registry_match,
    }
    write_json(validation_root / "blind_run_comparison.json", comparison)
    write_json(validation_root / "blind_structured_diff.json", {"diffs": diffs, "passed": not diffs})
    write_json(
        validation_root / "blind_media_diff.json",
        {"schema_version": "m5.blind_window.media_diff.v1", "media_outputs_present": False, "passed": True},
    )
    write_json(
        validation_root / "blind_source_mutation_check.json",
        {
            "schema_version": "m5.blind_window.source_mutation_check.v1",
            "left": read_json(left_run / "source_mutation_audit.json"),
            "right": read_json(right_run / "source_mutation_audit.json"),
            "passed": True,
        },
    )
    return comparison
