from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image


def synthetic_f3_payload(row_count: int = 3) -> dict[str, Any]:
    rows = []
    for index in range(row_count):
        rows.append(
            {
                "visible_person_base_id": f"vpb_{index:03d}",
                "frame_id": f"frame_{index:03d}",
                "frame_sequence": index,
                "timestamp_seconds": float(index) / 10.0,
                "detection_id": f"det_{index:03d}",
                "bbox": {"x1": 10.0 + index, "y1": 20.0, "x2": 40.0 + index, "y2": 70.0},
                "footpoint": {"x": 25.0 + index, "y": 70.0},
                "crop_quality": "medium",
                "candidate_type": "player_candidate_source",
                "roi_status": "inside_or_unverified_visual_roi",
                "step1f3_final_visual_role_state": "team_1_outfield_visual_context",
                "step1f3_final_visual_role_group": "player_outfield_visual_context",
                "step1f3_role_team_context": "team_1_outfield_visual_context",
                "step1f3_warning_flags": [],
                "step1f3_review_required": False,
            }
        )
    return {"artifact": "synthetic_step1f3_rows", "row_count": row_count, "rows": rows}


def synthetic_g1_manifest(row_count: int = 3) -> dict[str, Any]:
    return {
        "artifact": "synthetic_step1g_manifest",
        "f3_row_count": row_count,
        "step1g1_freeze_candidate_created": True,
        "step1g1_safe_for_step2_visual_continuity_candidate": True,
    }


def synthetic_m3t_payloads() -> dict[str, Any]:
    pathlet = {
        "pathlet_id": "step2m3t_pathlet_000001",
        "member_visible_person_base_ids": ["vpb_000", "vpb_001", "vpb_002"],
        "member_frame_sequences": [0, 1, 2],
        "accepted_continuity_edge_ids": ["edge_001", "edge_002"],
        "min_frame_sequence": 0,
        "max_frame_sequence": 2,
        "frame_span": 2,
        "seconds_span": 0.2,
        "max_members_per_frame": 1,
        "max_in_degree": 1,
        "max_out_degree": 1,
        "branch_count": 0,
        "merge_count": 0,
    }
    edges = [
        {
            "continuity_edge_id": "edge_001",
            "pathlet_id": "step2m3t_pathlet_000001",
            "source_visible_person_base_id": "vpb_000",
            "target_visible_person_base_id": "vpb_001",
            "source_frame_sequence": 0,
            "target_frame_sequence": 1,
        },
        {
            "continuity_edge_id": "edge_002",
            "pathlet_id": "step2m3t_pathlet_000001",
            "source_visible_person_base_id": "vpb_001",
            "target_visible_person_base_id": "vpb_002",
            "source_frame_sequence": 1,
            "target_frame_sequence": 2,
        },
    ]
    decision = {
        "step2m3t_review_candidate_id": "candidate_001",
        "review_subject_type": "sparse_visual_continuity_pathlet",
        "pathlet_id": "step2m3t_pathlet_000001",
        "continuity_edge_id": "",
        "accepted_continuity_edge_ids": ["edge_001", "edge_002"],
        "human_review_decision": "accept_sparse_pathlet_for_visual_handoff",
        "current_review_version": "step2m3t_sparse_pathlets_review_v1",
        "review_decisions_collected_with_review_version": "step2m3t_sparse_pathlets_review_v1",
        "current_visual_evidence_version": "step2m3t_visual_evidence_v1_animation",
        "review_decisions_collected_with_visual_evidence_version": "step2m3t_visual_evidence_v1_animation",
        "review_decisions_visual_evidence_version_matches_current": True,
        "human_approved": False,
        "approve_any_identity_tracking": False,
        "approve_any_player_slot_use": False,
        "approve_any_goalkeeper_slot_use": False,
        "approve_any_metric_use": False,
        "approve_event_or_tactical_analysis": False,
        "approve_exact_22_or_exact_two_goalkeeper_forcing": False,
        "approve_official_referee_exclusion": False,
        "approve_bad_detection_deletion": False,
        "approve_production_promotion": False,
    }
    return {
        "handoff": {
            "future_handoff_ready_candidate": True,
            "sparse_pathlet_count": 1,
            "sparse_selected_edge_count": 2,
            "forbidden_keys_present": [],
        },
        "progress": {"reviewed_candidates": 40, "sparse_pathlet_review_completed": True},
        "validation": {"forbidden_keys_present": []},
        "review_candidates": [
            {"step2m3t_review_candidate_id": "candidate_001", "pathlet_id": "step2m3t_pathlet_000001"}
        ],
        "decisions": {"rows": [decision]},
        "pathlets": {"rows": [pathlet]},
        "edges": edges,
        "quarantined_edges": [],
    }


def write_synthetic_frames(root: Path, frame_count: int = 3) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(frame_count):
        path = root / f"frame_{index:03d}.jpg"
        image = Image.new("RGB", (120, 80), (30 + index, 40, 50))
        image.save(path)
        frames.append(
            {
                "frame_id": f"frame_{index:03d}",
                "frame_file": str(path),
                "frame_sequence": index,
                "timestamp_seconds": float(index) / 10.0,
                "width": 120,
                "height": 80,
            }
        )
    manifest = {"run_id": "synthetic_frames", "frames": frames, "summary": {"frame_count": frame_count}}
    (root / "frame_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
