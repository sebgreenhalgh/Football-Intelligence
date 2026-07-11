from __future__ import annotations

from pathlib import Path
from typing import Any

M5_2_STAGE_URI = "matches/128058/runs/step_m5/03_m4_repeatability"
M5_2_RUN_PARENT_URI = f"{M5_2_STAGE_URI}/runs"
M5_1_CANONICAL_BASELINE_URI = (
    "matches/128058/runs/step_m5/02_infrastructure_hardening/runs/" "m5_baseline_20260711T125508Z_325fa715"
)
M5_1_CONTROL_BASELINE_URI = (
    "matches/128058/runs/step_m5/02_infrastructure_hardening/runs/" "m5_baseline_20260711T125442Z_9ffc1b65"
)
PRESERVED_M4_ROOT_URI = "matches/128058/calibration/step2_visual_continuity/step2m4_sparse_handoff_package"
EXPECTED_HEADLINE_SEMANTIC_HASH = "dfccb51f80bb80663f6c45765095d3f5320b27ff1063b4597e30ec2aa64cf78e"
EXPECTED_STRUCTURED_CONTENT_HASH = "6b7db49e662a39eab7c860c4d0c36dc5617d80b7f8cd7a4a63ad2037e3ca3149"
EXPECTED_BASELINE_CONFIG_SET_HASH = "7e87ac30ce1ee514995985644d8925edde714a1c33aa1277bab21c371f5e5e9a"

M4_STRUCTURED_FILES = {
    "m4_pathlets": ("step2m4_sparse_handoff_pathlets.json", "rows", "m4_handoff_pathlet_id"),
    "m4_edges": ("step2m4_sparse_handoff_edges.jsonl.gz", "jsonl_gzip_rows", "continuity_edge_id"),
    "m4_summary": ("step2m4_sparse_handoff_summary.json", "document", "artifact"),
    "m4_validation_summary": ("step2m4_validation_summary.json", "document", "artifact"),
    "m4_safety_guardrail_audit": ("step2m4_safety_guardrail_audit.json", "document", "artifact"),
    "m4_handoff_manifest": ("step2m4_handoff_manifest.json", "document", "artifact"),
    "m4_freeze_candidate_manifest": ("step2m4_freeze_candidate_manifest.json", "document", "artifact"),
}

M4_REQUIRED_FILES = [
    "step2m4_sparse_handoff_pathlets.json",
    "step2m4_sparse_handoff_edges.jsonl.gz",
    "step2m4_sparse_handoff_summary.json",
    "step2m4_sparse_handoff_viewer.html",
    "step2m4_handoff_manifest.json",
    "step2m4_validation_summary.json",
    "step2m4_safety_guardrail_audit.json",
    "step2m4_issue_register.json",
    "step2m4_freeze_candidate_manifest.json",
]

M4_EVIDENCE_DIRS = [
    "step2m4_pathlet_overlay_frames",
    "step2m4_pathlet_overlay_strips",
    "step2m4_pathlet_overlay_gifs",
]

PROTECTED_ROOT_URIS = [
    "matches/128058/calibration/step2_visual_continuity/step2m1_visual_continuity_sandbox",
    "matches/128058/calibration/step2_visual_continuity/step2m2_match_local_adaptation",
    "matches/128058/calibration/step2_visual_continuity/step2m3_adaptation_safe_continuity_output",
    "matches/128058/calibration/step2_visual_continuity/step2m3r_topology_qa",
    "matches/128058/calibration/step2_visual_continuity/step2m3s_topology_safe_handoff_subset",
    "matches/128058/calibration/step2_visual_continuity/step2m3t_sparse_pathlets",
    PRESERVED_M4_ROOT_URI,
    "matches/128058/manual_review",
]

M3T_DECISION_ALLOWED_VALUES = {
    "accept_sparse_pathlet_for_visual_handoff",
    "reject_or_quarantine_sparse_pathlet",
    "unsure_needs_later_review",
}
M3T_REVIEW_VERSION = "step2m3t_sparse_pathlets_review_v1"
M3T_VISUAL_EVIDENCE_VERSION = "step2m3t_visual_evidence_v1_animation"


def stage_root(artifact_root: Path) -> Path:
    return artifact_root / M5_2_STAGE_URI


def run_root(artifact_root: Path, run_id: str) -> Path:
    return artifact_root / M5_2_RUN_PARENT_URI / run_id


def expected_counts() -> dict[str, Any]:
    return {
        "m4_handoff_pathlet_count": 795,
        "m4_handoff_edge_count": 7393,
        "overlay_asset_count": 857,
        "overlay_frame_count": 757,
        "overlay_strip_count": 50,
        "overlay_gif_count": 50,
        "source_m3t_reviewed_decisions_count": 40,
        "pathlets_over_cap": 0,
        "duplicate_frame_pathlets": 0,
        "branch_merge_pathlets": 0,
        "forbidden_keys_present": [],
    }
