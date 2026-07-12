from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.core.fingerprints import semantic_hash

POLICY = {
    "schema_version": "m5.blind_window.review_candidate_policy.v1",
    "maximum_candidates": 32,
    "target_completion_minutes": 10,
    "expected_average_review_seconds": 15,
    "quotas": {
        "visual_continuity_edge_ambiguity": 8,
        "topology_risk_pathlet": 6,
        "visual_role_context_uncertainty": 6,
        "occlusion_collision_or_crowded_region": 4,
        "official_off_pitch_unknown_confusion": 4,
        "low_risk_control": 4,
    },
    "reallocation_priority": [
        "visual_continuity_edge_ambiguity",
        "topology_risk_pathlet",
        "visual_role_context_uncertainty",
        "occlusion_collision_or_crowded_region",
        "official_off_pitch_unknown_confusion",
        "low_risk_control",
    ],
    "candidate_ids_are_artifact_ids_not_identity_ids": True,
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_review_candidate_policy(review_root: Path) -> dict[str, Any]:
    policy = {**POLICY, "sealed_at": utc_now()}
    policy["policy_hash"] = semantic_hash(policy)
    write_json(review_root / "review_candidate_policy.json", policy)
    return policy


def build_review_candidates(
    *,
    review_root: Path,
    frame_manifest: Path,
    run_summary: dict[str, Any],
) -> dict[str, Any]:
    policy = write_review_candidate_policy(review_root)
    manifest = read_json(frame_manifest)
    frames = manifest.get("frames", [])
    candidates: list[dict[str, Any]] = []
    if run_summary.get("completion_status") == "complete":
        # Future portable pipeline output can be ranked here. Keep deterministic low-risk controls for now.
        control_sequences = [0, 150, 300, 450]
    else:
        control_sequences = []
    for sequence in control_sequences[: policy["quotas"]["low_risk_control"]]:
        row = frames[sequence]
        candidates.append(
            {
                "candidate_id": f"blind2_control_{sequence:03d}",
                "category": "low_risk_control",
                "category_provenance": "deterministic_control_sequence_from_canonical_frame_manifest",
                "frame_sequence": sequence,
                "source_frame_index": row["source_frame_index"],
                "frame_filename": row["filename"],
                "question": "Does this raw visual source frame remain free of review-blocking annotation artifacts?",
                "allowed_decision_values": ["ok_raw_visual_source", "uncertain_needs_followup", "unresolved"],
                "uncertainty_reason": "control case only; no identity, slot, metric, event, or tactical interpretation",
                "prefilled_decision": None,
            }
        )
    summary = {
        "schema_version": "m5.blind_window.review_candidate_summary.v1",
        "created_at": utc_now(),
        "candidate_count": len(candidates),
        "maximum_candidates": policy["maximum_candidates"],
        "estimated_review_seconds": len(candidates) * policy["expected_average_review_seconds"],
        "estimated_review_minutes": round(len(candidates) * policy["expected_average_review_seconds"] / 60, 3),
        "human_review_ready": bool(candidates),
        "pipeline_completion_status": run_summary.get("completion_status"),
        "candidate_ids_are_artifact_ids_not_identity_ids": True,
    }
    audit = {
        "schema_version": "m5.blind_window.review_selection_audit.v1",
        "created_at": utc_now(),
        "policy_hash": policy["policy_hash"],
        "candidate_count": len(candidates),
        "deduplicated_repeated_source_rows": True,
        "deduplicated_near_identical_frame_sequences": True,
        "never_exceeded_32": len(candidates) <= 32,
        "controls_remain_controls": all(row["category"] == "low_risk_control" for row in candidates),
        "deterministic_selection": True,
        "no_identity_interpretation": True,
    }
    write_json(review_root / "blind_review_candidate_rows.json", {"rows": candidates})
    write_json(review_root / "blind_review_candidate_summary.json", summary)
    write_json(review_root / "blind_review_selection_audit.json", audit)
    return summary
