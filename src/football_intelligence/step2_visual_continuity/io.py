# ruff: noqa: E501

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from football_intelligence.paths import CLIP_ID, MATCH_ID, MATCH_ROOT, SOCCERTRACK_ROOT, ensure_dir, require_file
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_ROWS_PATH,
    STEP1G1_FREEZE_CANDIDATE_MANIFEST_PATH,
)
from football_intelligence.step2_visual_continuity.schema import (
    DEFAULT_MAX_FRAME_GAP,
    PRODUCTION_READY,
    VISUAL_ONLY_WARNING,
    guardrail_stamp,
    safe_int,
    utc_iso,
)


STEP2_VISUAL_CONTINUITY_DIR = MATCH_ROOT / "calibration" / "step2_visual_continuity"
STEP2M1_OUTPUT_DIR = STEP2_VISUAL_CONTINUITY_DIR / "step2m1_visual_continuity_sandbox"

STEP2M1_NODE_ROWS_PATH = STEP2M1_OUTPUT_DIR / "step2m1_visual_continuity_node_rows.json"
STEP2M1_EDGE_CANDIDATE_ROWS_PATH = STEP2M1_OUTPUT_DIR / "step2m1_visual_continuity_edge_candidate_rows.json"
STEP2M1_EDGE_CANDIDATE_SUMMARY_PATH = STEP2M1_OUTPUT_DIR / "step2m1_visual_continuity_edge_candidate_summary.json"
STEP2M1_EDGE_CANDIDATE_SAMPLE_PATH = STEP2M1_OUTPUT_DIR / "step2m1_visual_continuity_edge_candidate_sample.json"
STEP2M1_EDGE_CANDIDATE_ROWS_JSONL_GZ_PATH = STEP2M1_OUTPUT_DIR / "step2m1_visual_continuity_edge_candidate_rows.jsonl.gz"
STEP2M1_GROUP_ROWS_SANDBOX_PATH = STEP2M1_OUTPUT_DIR / "step2m1_visual_continuity_group_rows_sandbox.json"
STEP2M1_REVIEW_CANDIDATE_ROWS_PATH = STEP2M1_OUTPUT_DIR / "step2m1_visual_continuity_review_candidate_rows.json"
STEP2M1_REVIEWED_DECISIONS_PATH = STEP2M1_OUTPUT_DIR / "step2m1_reviewed_visual_continuity_decisions.json"
STEP2M1_REVIEW_PROGRESS_SUMMARY_PATH = STEP2M1_OUTPUT_DIR / "step2m1_visual_continuity_review_progress_summary.json"
STEP2M1_REVIEW_DECISION_SUMMARY_PATH = STEP2M1_OUTPUT_DIR / "step2m1_visual_continuity_review_decision_summary.json"
STEP2M1_HUMAN_CORRECTED_EDGE_ROWS_PATH = STEP2M1_OUTPUT_DIR / "step2m1_human_corrected_visual_continuity_edge_rows.json"
STEP2M1_HUMAN_CORRECTED_EDGE_SUMMARY_PATH = STEP2M1_OUTPUT_DIR / "step2m1_human_corrected_visual_continuity_edge_summary.json"
STEP2M1_HUMAN_CORRECTED_EDGE_SAMPLE_PATH = STEP2M1_OUTPUT_DIR / "step2m1_human_corrected_visual_continuity_edge_sample.json"
STEP2M1_HUMAN_CORRECTED_EDGE_ROWS_JSONL_GZ_PATH = STEP2M1_OUTPUT_DIR / "step2m1_human_corrected_visual_continuity_edge_rows.jsonl.gz"
STEP2M1_CORRECTION_AUDIT_ROWS_PATH = STEP2M1_OUTPUT_DIR / "step2m1_visual_continuity_correction_audit_rows.json"
STEP2M1_TRAINING_EXAMPLES_PATH = STEP2M1_OUTPUT_DIR / "step2m1_visual_continuity_training_examples.jsonl"
STEP2M1_VALIDATION_SUMMARY_PATH = STEP2M1_OUTPUT_DIR / "step2m1_visual_continuity_validation_summary.json"
STEP2M1_ISSUE_REGISTER_PATH = STEP2M1_OUTPUT_DIR / "step2m1_visual_continuity_issue_register.json"
STEP2M1_SAFETY_GUARDRAIL_AUDIT_PATH = STEP2M1_OUTPUT_DIR / "step2m1_visual_continuity_safety_guardrail_audit.json"
STEP2M1_FREEZE_CANDIDATE_MANIFEST_PATH = STEP2M1_OUTPUT_DIR / "step2m1_visual_continuity_freeze_candidate_manifest.json"
STEP2M1_REVIEW_UI_HTML_PATH = STEP2M1_OUTPUT_DIR / "step2m1_visual_continuity_review_ui.html"
STEP2M1_REVIEW_CONTACT_SHEET_PATH = STEP2M1_OUTPUT_DIR / "step2m1_visual_continuity_review_contact_sheet.jpg"
STEP2M1_REVIEW_UI_MANIFEST_PATH = STEP2M1_OUTPUT_DIR / "step2m1_visual_continuity_review_ui_manifest.json"
STEP2M1_REVIEW_PACK_DIR = STEP2M1_OUTPUT_DIR / "review_pack"
STEP2M1_REVIEW_PACK_MANIFEST_PATH = STEP2M1_REVIEW_PACK_DIR / "step2m1_visual_continuity_review_pack_manifest.json"

STEP2M1_SOURCE_CONTEXT_IMAGES_DIR = STEP2M1_OUTPUT_DIR / "step2m1_source_context_images"
STEP2M1_TARGET_CONTEXT_IMAGES_DIR = STEP2M1_OUTPUT_DIR / "step2m1_target_context_images"
STEP2M1_SOURCE_CROP_IMAGES_DIR = STEP2M1_OUTPUT_DIR / "step2m1_source_crop_images"
STEP2M1_TARGET_CROP_IMAGES_DIR = STEP2M1_OUTPUT_DIR / "step2m1_target_crop_images"

STEP2M1R_ADAPTATION_SAFE_EDGE_ROWS_PATH = STEP2M1_OUTPUT_DIR / "step2m1r_adaptation_safe_visual_continuity_edge_rows.json"
STEP2M1R_ADAPTATION_SAFE_EDGE_SUMMARY_PATH = STEP2M1_OUTPUT_DIR / "step2m1r_adaptation_safe_visual_continuity_edge_summary.json"
STEP2M1R_ADAPTATION_SAFE_EDGE_SAMPLE_PATH = STEP2M1_OUTPUT_DIR / "step2m1r_adaptation_safe_visual_continuity_edge_sample.json"
STEP2M1R_ADAPTATION_SAFE_EDGE_ROWS_JSONL_GZ_PATH = STEP2M1_OUTPUT_DIR / "step2m1r_adaptation_safe_visual_continuity_edge_rows.jsonl.gz"
STEP2M1R_ADAPTATION_SAFE_GROUP_ROWS_PATH = STEP2M1_OUTPUT_DIR / "step2m1r_adaptation_safe_visual_continuity_group_rows.json"
STEP2M1R_ADAPTATION_SAFE_GROUP_SUMMARY_PATH = STEP2M1_OUTPUT_DIR / "step2m1r_adaptation_safe_visual_continuity_group_summary.json"
STEP2M1R_ADAPTATION_SAFE_GROUP_SAMPLE_PATH = STEP2M1_OUTPUT_DIR / "step2m1r_adaptation_safe_visual_continuity_group_sample.json"
STEP2M1R_GROUP_SPAN_REMEDIATION_SUMMARY_PATH = STEP2M1_OUTPUT_DIR / "step2m1r_group_span_remediation_summary.json"
STEP2M1R_TARGETED_REVIEW_CANDIDATE_ROWS_PATH = STEP2M1_OUTPUT_DIR / "step2m1r_targeted_review_candidate_rows.json"
STEP2M1R_TARGETED_REVIEW_CANDIDATE_SUMMARY_PATH = STEP2M1_OUTPUT_DIR / "step2m1r_targeted_review_candidate_summary.json"
STEP2M1R_TARGETED_REVIEW_CANDIDATE_SAMPLE_PATH = STEP2M1_OUTPUT_DIR / "step2m1r_targeted_review_candidate_sample.json"
STEP2M1R_REVIEW_UI_HTML_PATH = STEP2M1_OUTPUT_DIR / "step2m1r_review_ui.html"
STEP2M1R_REVIEW_CONTACT_SHEET_PATH = STEP2M1_OUTPUT_DIR / "step2m1r_review_contact_sheet.jpg"
STEP2M1R_REVIEWED_DECISIONS_PATH = STEP2M1_OUTPUT_DIR / "step2m1r_reviewed_visual_continuity_decisions.json"
STEP2M1R_REVIEW_PROGRESS_SUMMARY_PATH = STEP2M1_OUTPUT_DIR / "step2m1r_review_progress_summary.json"
STEP2M1R_REVIEW_DECISION_SUMMARY_PATH = STEP2M1_OUTPUT_DIR / "step2m1r_review_decision_summary.json"
STEP2M1R_ADAPTATION_SAFETY_MANIFEST_PATH = STEP2M1_OUTPUT_DIR / "step2m1r_adaptation_safety_manifest.json"
STEP2M1R_REVIEW_BURST_CLIPS_DIR = STEP2M1_OUTPUT_DIR / "step2m1r_review_burst_clips"
STEP2M1R_REVIEW_BURST_STRIPS_DIR = STEP2M1_OUTPUT_DIR / "step2m1r_review_burst_strips"
STEP2M1R_REVIEW_BURST_RAW_STRIPS_DIR = STEP2M1_OUTPUT_DIR / "step2m1r_review_burst_raw_strips"
STEP2M1R_REVIEW_BURST_COMPARISON_STRIPS_DIR = STEP2M1_OUTPUT_DIR / "step2m1r_review_burst_comparison_strips"
STEP2M1R_BURST_OVERLAY_DEBUG_ROWS_PATH = STEP2M1_OUTPUT_DIR / "step2m1r_burst_overlay_debug_rows.json"
STEP2M1R_BURST_OVERLAY_ALIGNMENT_SUMMARY_PATH = STEP2M1_OUTPUT_DIR / "step2m1r_burst_overlay_alignment_summary.json"
STEP2M1R_BURST_OVERLAY_QA_DIR = STEP2M1_OUTPUT_DIR / "step2m1r_burst_overlay_alignment_qa"
STEP2M1R_SOURCE_CONTEXT_IMAGES_DIR = STEP2M1_OUTPUT_DIR / "step2m1r_source_context_images"
STEP2M1R_TARGET_CONTEXT_IMAGES_DIR = STEP2M1_OUTPUT_DIR / "step2m1r_target_context_images"
STEP2M1R_SOURCE_CROP_IMAGES_DIR = STEP2M1_OUTPUT_DIR / "step2m1r_source_crop_images"
STEP2M1R_TARGET_CROP_IMAGES_DIR = STEP2M1_OUTPUT_DIR / "step2m1r_target_crop_images"

STEP2M2_OUTPUT_DIR = STEP2_VISUAL_CONTINUITY_DIR / "step2m2_match_local_adaptation"
STEP2M2_REVIEWED_DECISION_TRAINING_ROWS_PATH = STEP2M2_OUTPUT_DIR / "step2m2_reviewed_decision_training_rows.json"
STEP2M2_REVIEWED_DECISION_TRAINING_SUMMARY_PATH = STEP2M2_OUTPUT_DIR / "step2m2_reviewed_decision_training_summary.json"
STEP2M2_MATCH_LOCAL_ADAPTATION_PROFILE_PATH = STEP2M2_OUTPUT_DIR / "step2m2_match_local_adaptation_profile.json"
STEP2M2_ADAPTED_EDGE_CANDIDATES_JSONL_GZ_PATH = STEP2M2_OUTPUT_DIR / "step2m2_adapted_visual_continuity_edge_candidates.jsonl.gz"
STEP2M2_ADAPTED_EDGE_CANDIDATE_SUMMARY_PATH = STEP2M2_OUTPUT_DIR / "step2m2_adapted_visual_continuity_edge_candidate_summary.json"
STEP2M2_ADAPTED_EDGE_CANDIDATE_SAMPLE_PATH = STEP2M2_OUTPUT_DIR / "step2m2_adapted_visual_continuity_edge_candidate_sample.json"
STEP2M2_TARGETED_REVIEW_CANDIDATE_ROWS_PATH = STEP2M2_OUTPUT_DIR / "step2m2_targeted_review_candidate_rows.json"
STEP2M2_REVIEW_UI_HTML_PATH = STEP2M2_OUTPUT_DIR / "step2m2_review_ui.html"
STEP2M2_REVIEW_CONTACT_SHEET_PATH = STEP2M2_OUTPUT_DIR / "step2m2_review_contact_sheet.jpg"
STEP2M2_REVIEWED_DECISIONS_PATH = STEP2M2_OUTPUT_DIR / "step2m2_reviewed_visual_continuity_decisions.json"
STEP2M2_REVIEW_PROGRESS_SUMMARY_PATH = STEP2M2_OUTPUT_DIR / "step2m2_review_progress_summary.json"
STEP2M2_REVIEW_DECISION_SUMMARY_PATH = STEP2M2_OUTPUT_DIR / "step2m2_review_decision_summary.json"
STEP2M2_REVIEW_BURST_CLIPS_DIR = STEP2M2_OUTPUT_DIR / "step2m2_review_burst_clips"
STEP2M2_REVIEW_BURST_STRIPS_DIR = STEP2M2_OUTPUT_DIR / "step2m2_review_burst_strips"
STEP2M2_REVIEW_BURST_RAW_STRIPS_DIR = STEP2M2_OUTPUT_DIR / "step2m2_review_burst_raw_strips"
STEP2M2_REVIEW_BURST_COMPARISON_STRIPS_DIR = STEP2M2_OUTPUT_DIR / "step2m2_review_burst_comparison_strips"
STEP2M2_BURST_OVERLAY_DEBUG_ROWS_PATH = STEP2M2_OUTPUT_DIR / "step2m2_burst_overlay_debug_rows.json"
STEP2M2_BURST_OVERLAY_ALIGNMENT_SUMMARY_PATH = STEP2M2_OUTPUT_DIR / "step2m2_burst_overlay_alignment_summary.json"
STEP2M2_BURST_OVERLAY_QA_DIR = STEP2M2_OUTPUT_DIR / "step2m2_burst_overlay_alignment_qa"
STEP2M2_SOURCE_CONTEXT_IMAGES_DIR = STEP2M2_OUTPUT_DIR / "step2m2_source_context_images"
STEP2M2_TARGET_CONTEXT_IMAGES_DIR = STEP2M2_OUTPUT_DIR / "step2m2_target_context_images"
STEP2M2_SOURCE_CROP_IMAGES_DIR = STEP2M2_OUTPUT_DIR / "step2m2_source_crop_images"
STEP2M2_TARGET_CROP_IMAGES_DIR = STEP2M2_OUTPUT_DIR / "step2m2_target_crop_images"
STEP2M2_VALIDATION_SUMMARY_PATH = STEP2M2_OUTPUT_DIR / "step2m2_validation_summary.json"
STEP2M2_SAFETY_GUARDRAIL_AUDIT_PATH = STEP2M2_OUTPUT_DIR / "step2m2_safety_guardrail_audit.json"
STEP2M2_ISSUE_REGISTER_PATH = STEP2M2_OUTPUT_DIR / "step2m2_issue_register.json"
STEP2M2_FREEZE_CANDIDATE_MANIFEST_PATH = STEP2M2_OUTPUT_DIR / "step2m2_freeze_candidate_manifest.json"
STEP2M2_REVIEW_PACK_DIR = STEP2M2_OUTPUT_DIR / "review_pack"
STEP2M2_REVIEW_PACK_MANIFEST_PATH = STEP2M2_REVIEW_PACK_DIR / "step2m2_review_pack_manifest.json"

STEP2M3_OUTPUT_DIR = STEP2_VISUAL_CONTINUITY_DIR / "step2m3_adaptation_safe_continuity_output"
STEP2M3_ACCEPTED_EDGES_JSONL_GZ_PATH = STEP2M3_OUTPUT_DIR / "step2m3_accepted_visual_continuity_edges.jsonl.gz"
STEP2M3_ACCEPTED_EDGE_SAMPLE_PATH = STEP2M3_OUTPUT_DIR / "step2m3_accepted_visual_continuity_edge_sample.json"
STEP2M3_ACCEPTED_EDGE_SUMMARY_PATH = STEP2M3_OUTPUT_DIR / "step2m3_accepted_visual_continuity_edge_summary.json"
STEP2M3_QUARANTINED_EDGES_JSONL_GZ_PATH = STEP2M3_OUTPUT_DIR / "step2m3_quarantined_visual_continuity_edges.jsonl.gz"
STEP2M3_QUARANTINED_EDGE_SAMPLE_PATH = STEP2M3_OUTPUT_DIR / "step2m3_quarantined_visual_continuity_edge_sample.json"
STEP2M3_QUARANTINE_SUMMARY_PATH = STEP2M3_OUTPUT_DIR / "step2m3_quarantine_summary.json"
STEP2M3_GROUP_ROWS_PATH = STEP2M3_OUTPUT_DIR / "step2m3_adaptation_safe_visual_continuity_groups.json"
STEP2M3_GROUP_SAMPLE_PATH = STEP2M3_OUTPUT_DIR / "step2m3_adaptation_safe_visual_continuity_group_sample.json"
STEP2M3_GROUP_SUMMARY_PATH = STEP2M3_OUTPUT_DIR / "step2m3_group_summary.json"
STEP2M3_HANDOFF_MANIFEST_PATH = STEP2M3_OUTPUT_DIR / "step2m3_handoff_manifest.json"
STEP2M3_VALIDATION_SUMMARY_PATH = STEP2M3_OUTPUT_DIR / "step2m3_validation_summary.json"
STEP2M3_SAFETY_GUARDRAIL_AUDIT_PATH = STEP2M3_OUTPUT_DIR / "step2m3_safety_guardrail_audit.json"
STEP2M3_ISSUE_REGISTER_PATH = STEP2M3_OUTPUT_DIR / "step2m3_issue_register.json"
STEP2M3_FREEZE_CANDIDATE_MANIFEST_PATH = STEP2M3_OUTPUT_DIR / "step2m3_freeze_candidate_manifest.json"
STEP2M3_REVIEW_PACK_DIR = STEP2M3_OUTPUT_DIR / "review_pack"
STEP2M3_REVIEW_PACK_MANIFEST_PATH = STEP2M3_REVIEW_PACK_DIR / "step2m3_review_pack_manifest.json"

STEP2M3R_OUTPUT_DIR = STEP2_VISUAL_CONTINUITY_DIR / "step2m3r_topology_qa"
STEP2M3R_GROUP_TOPOLOGY_AUDIT_ROWS_PATH = STEP2M3R_OUTPUT_DIR / "step2m3r_group_topology_audit_rows.json"
STEP2M3R_GROUP_TOPOLOGY_AUDIT_SUMMARY_PATH = STEP2M3R_OUTPUT_DIR / "step2m3r_group_topology_audit_summary.json"
STEP2M3R_GROUP_TOPOLOGY_AUDIT_SAMPLE_PATH = STEP2M3R_OUTPUT_DIR / "step2m3r_group_topology_audit_sample.json"
STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_ROWS_JSONL_GZ_PATH = STEP2M3R_OUTPUT_DIR / "step2m3r_accepted_edge_topology_audit_rows.jsonl.gz"
STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_SUMMARY_PATH = STEP2M3R_OUTPUT_DIR / "step2m3r_accepted_edge_topology_audit_summary.json"
STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_SAMPLE_PATH = STEP2M3R_OUTPUT_DIR / "step2m3r_accepted_edge_topology_audit_sample.json"
STEP2M3R_TOPOLOGY_REVIEW_CANDIDATE_ROWS_PATH = STEP2M3R_OUTPUT_DIR / "step2m3r_topology_review_candidate_rows.json"
STEP2M3R_TOPOLOGY_REVIEW_UI_HTML_PATH = STEP2M3R_OUTPUT_DIR / "step2m3r_topology_review_ui.html"
STEP2M3R_TOPOLOGY_REVIEW_CONTACT_SHEET_PATH = STEP2M3R_OUTPUT_DIR / "step2m3r_topology_review_contact_sheet.jpg"
STEP2M3R_VISUAL_EVIDENCE_DIR = STEP2M3R_OUTPUT_DIR / "step2m3r_visual_evidence"
STEP2M3R_GROUP_TIMELINE_STRIPS_DIR = STEP2M3R_VISUAL_EVIDENCE_DIR / "group_timeline_strips"
STEP2M3R_EDGE_BURST_STRIPS_DIR = STEP2M3R_VISUAL_EVIDENCE_DIR / "edge_burst_strips"
STEP2M3R_GROUP_TIMELINE_ANIMATIONS_DIR = STEP2M3R_VISUAL_EVIDENCE_DIR / "group_timeline_animations"
STEP2M3R_EDGE_BURST_ANIMATIONS_DIR = STEP2M3R_VISUAL_EVIDENCE_DIR / "edge_burst_animations"
STEP2M3R_REVIEWED_TOPOLOGY_DECISIONS_PATH = STEP2M3R_OUTPUT_DIR / "step2m3r_reviewed_topology_decisions.json"
STEP2M3R_REVIEW_PROGRESS_SUMMARY_PATH = STEP2M3R_OUTPUT_DIR / "step2m3r_review_progress_summary.json"
STEP2M3R_REVIEW_DECISION_SUMMARY_PATH = STEP2M3R_OUTPUT_DIR / "step2m3r_review_decision_summary.json"
STEP2M3R_HANDOFF_READINESS_SUMMARY_PATH = STEP2M3R_OUTPUT_DIR / "step2m3r_handoff_readiness_summary.json"
STEP2M3R_VALIDATION_SUMMARY_PATH = STEP2M3R_OUTPUT_DIR / "step2m3r_validation_summary.json"
STEP2M3R_SAFETY_GUARDRAIL_AUDIT_PATH = STEP2M3R_OUTPUT_DIR / "step2m3r_safety_guardrail_audit.json"
STEP2M3R_ISSUE_REGISTER_PATH = STEP2M3R_OUTPUT_DIR / "step2m3r_issue_register.json"
STEP2M3R_FREEZE_CANDIDATE_MANIFEST_PATH = STEP2M3R_OUTPUT_DIR / "step2m3r_freeze_candidate_manifest.json"
STEP2M3R_REVIEW_PACK_DIR = STEP2M3R_OUTPUT_DIR / "review_pack"
STEP2M3R_REVIEW_PACK_MANIFEST_PATH = STEP2M3R_REVIEW_PACK_DIR / "step2m3r_review_pack_manifest.json"

STEP2M3S_OUTPUT_DIR = STEP2_VISUAL_CONTINUITY_DIR / "step2m3s_topology_safe_handoff_subset"
STEP2M3S_REVIEWED_TOPOLOGY_DECISION_ROWS_PATH = STEP2M3S_OUTPUT_DIR / "step2m3s_reviewed_topology_decision_rows.json"
STEP2M3S_REVIEWED_TOPOLOGY_DECISION_SUMMARY_PATH = STEP2M3S_OUTPUT_DIR / "step2m3s_reviewed_topology_decision_summary.json"
STEP2M3S_HANDOFF_SAFE_GROUPS_PATH = STEP2M3S_OUTPUT_DIR / "step2m3s_handoff_safe_visual_continuity_groups.json"
STEP2M3S_HANDOFF_SAFE_GROUP_SAMPLE_PATH = STEP2M3S_OUTPUT_DIR / "step2m3s_handoff_safe_visual_continuity_group_sample.json"
STEP2M3S_HANDOFF_SAFE_GROUP_SUMMARY_PATH = STEP2M3S_OUTPUT_DIR / "step2m3s_handoff_safe_group_summary.json"
STEP2M3S_HANDOFF_SAFE_EDGES_JSONL_GZ_PATH = STEP2M3S_OUTPUT_DIR / "step2m3s_handoff_safe_visual_continuity_edges.jsonl.gz"
STEP2M3S_HANDOFF_SAFE_EDGE_SAMPLE_PATH = STEP2M3S_OUTPUT_DIR / "step2m3s_handoff_safe_visual_continuity_edge_sample.json"
STEP2M3S_HANDOFF_SAFE_EDGE_SUMMARY_PATH = STEP2M3S_OUTPUT_DIR / "step2m3s_handoff_safe_edge_summary.json"
STEP2M3S_TOPOLOGY_QUARANTINED_GROUPS_PATH = STEP2M3S_OUTPUT_DIR / "step2m3s_topology_quarantined_groups.json"
STEP2M3S_TOPOLOGY_QUARANTINED_EDGES_JSONL_GZ_PATH = STEP2M3S_OUTPUT_DIR / "step2m3s_topology_quarantined_edges.jsonl.gz"
STEP2M3S_TOPOLOGY_QUARANTINE_SUMMARY_PATH = STEP2M3S_OUTPUT_DIR / "step2m3s_topology_quarantine_summary.json"
STEP2M3S_HANDOFF_MANIFEST_PATH = STEP2M3S_OUTPUT_DIR / "step2m3s_handoff_manifest.json"
STEP2M3S_VALIDATION_SUMMARY_PATH = STEP2M3S_OUTPUT_DIR / "step2m3s_validation_summary.json"
STEP2M3S_SAFETY_GUARDRAIL_AUDIT_PATH = STEP2M3S_OUTPUT_DIR / "step2m3s_safety_guardrail_audit.json"
STEP2M3S_ISSUE_REGISTER_PATH = STEP2M3S_OUTPUT_DIR / "step2m3s_issue_register.json"
STEP2M3S_FREEZE_CANDIDATE_MANIFEST_PATH = STEP2M3S_OUTPUT_DIR / "step2m3s_freeze_candidate_manifest.json"
STEP2M3S_REVIEW_PACK_DIR = STEP2M3S_OUTPUT_DIR / "review_pack"
STEP2M3S_REVIEW_PACK_MANIFEST_PATH = STEP2M3S_REVIEW_PACK_DIR / "step2m3s_review_pack_manifest.json"

STEP2M3T_OUTPUT_DIR = STEP2_VISUAL_CONTINUITY_DIR / "step2m3t_sparse_pathlets"
STEP2M3T_SPARSE_CANDIDATE_EDGES_JSONL_GZ_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_sparse_candidate_edge_rows.jsonl.gz"
STEP2M3T_SPARSE_CANDIDATE_EDGE_SUMMARY_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_sparse_candidate_edge_summary.json"
STEP2M3T_SPARSE_CANDIDATE_EDGE_SAMPLE_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_sparse_candidate_edge_sample.json"
STEP2M3T_SELECTED_SPARSE_EDGES_JSONL_GZ_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_selected_sparse_visual_continuity_edges.jsonl.gz"
STEP2M3T_SELECTED_SPARSE_EDGE_SUMMARY_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_selected_sparse_visual_continuity_edge_summary.json"
STEP2M3T_SELECTED_SPARSE_EDGE_SAMPLE_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_selected_sparse_visual_continuity_edge_sample.json"
STEP2M3T_SPARSE_PATHLETS_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_sparse_visual_continuity_pathlets.json"
STEP2M3T_SPARSE_PATHLET_SUMMARY_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_sparse_visual_continuity_pathlet_summary.json"
STEP2M3T_SPARSE_PATHLET_SAMPLE_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_sparse_visual_continuity_pathlet_sample.json"
STEP2M3T_TOPOLOGY_QUARANTINED_EDGES_JSONL_GZ_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_topology_quarantined_edges.jsonl.gz"
STEP2M3T_TOPOLOGY_QUARANTINED_PATHLETS_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_topology_quarantined_pathlets.json"
STEP2M3T_TOPOLOGY_QUARANTINE_SUMMARY_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_topology_quarantine_summary.json"
STEP2M3T_REVIEW_CANDIDATE_ROWS_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_review_candidate_rows.json"
STEP2M3T_REVIEW_UI_HTML_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_review_ui.html"
STEP2M3T_REVIEW_CONTACT_SHEET_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_review_contact_sheet.jpg"
STEP2M3T_REVIEW_PROGRESS_SUMMARY_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_review_progress_summary.json"
STEP2M3T_REVIEW_DECISION_SUMMARY_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_review_decision_summary.json"
STEP2M3T_REVIEWED_SPARSE_PATHLET_DECISIONS_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_reviewed_sparse_pathlet_decisions.json"
STEP2M3T_VISUAL_EVIDENCE_DIR = STEP2M3T_OUTPUT_DIR / "step2m3t_visual_evidence"
STEP2M3T_PATHLET_ANIMATIONS_DIR = STEP2M3T_VISUAL_EVIDENCE_DIR / "pathlet_animations"
STEP2M3T_PATHLET_STRIPS_DIR = STEP2M3T_VISUAL_EVIDENCE_DIR / "pathlet_strips"
STEP2M3T_EDGE_BURST_ANIMATIONS_DIR = STEP2M3T_VISUAL_EVIDENCE_DIR / "edge_burst_animations"
STEP2M3T_EDGE_BURST_STRIPS_DIR = STEP2M3T_VISUAL_EVIDENCE_DIR / "edge_burst_strips"
STEP2M3T_HANDOFF_MANIFEST_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_handoff_manifest.json"
STEP2M3T_VALIDATION_SUMMARY_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_validation_summary.json"
STEP2M3T_SAFETY_GUARDRAIL_AUDIT_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_safety_guardrail_audit.json"
STEP2M3T_ISSUE_REGISTER_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_issue_register.json"
STEP2M3T_FREEZE_CANDIDATE_MANIFEST_PATH = STEP2M3T_OUTPUT_DIR / "step2m3t_freeze_candidate_manifest.json"
STEP2M3T_REVIEW_PACK_DIR = STEP2M3T_OUTPUT_DIR / "review_pack"
STEP2M3T_REVIEW_PACK_MANIFEST_PATH = STEP2M3T_REVIEW_PACK_DIR / "step2m3t_review_pack_manifest.json"

STEP2M4_OUTPUT_DIR = STEP2_VISUAL_CONTINUITY_DIR / "step2m4_sparse_handoff_package"
STEP2M4_HANDOFF_PATHLETS_PATH = STEP2M4_OUTPUT_DIR / "step2m4_sparse_handoff_pathlets.json"
STEP2M4_HANDOFF_EDGES_JSONL_GZ_PATH = STEP2M4_OUTPUT_DIR / "step2m4_sparse_handoff_edges.jsonl.gz"
STEP2M4_HANDOFF_SUMMARY_PATH = STEP2M4_OUTPUT_DIR / "step2m4_sparse_handoff_summary.json"
STEP2M4_PATHLET_OVERLAY_FRAMES_DIR = STEP2M4_OUTPUT_DIR / "step2m4_pathlet_overlay_frames"
STEP2M4_PATHLET_OVERLAY_GIFS_DIR = STEP2M4_OUTPUT_DIR / "step2m4_pathlet_overlay_gifs"
STEP2M4_PATHLET_OVERLAY_STRIPS_DIR = STEP2M4_OUTPUT_DIR / "step2m4_pathlet_overlay_strips"
STEP2M4_VIEWER_HTML_PATH = STEP2M4_OUTPUT_DIR / "step2m4_sparse_handoff_viewer.html"
STEP2M4_HANDOFF_MANIFEST_PATH = STEP2M4_OUTPUT_DIR / "step2m4_handoff_manifest.json"
STEP2M4_VALIDATION_SUMMARY_PATH = STEP2M4_OUTPUT_DIR / "step2m4_validation_summary.json"
STEP2M4_SAFETY_GUARDRAIL_AUDIT_PATH = STEP2M4_OUTPUT_DIR / "step2m4_safety_guardrail_audit.json"
STEP2M4_ISSUE_REGISTER_PATH = STEP2M4_OUTPUT_DIR / "step2m4_issue_register.json"
STEP2M4_FREEZE_CANDIDATE_MANIFEST_PATH = STEP2M4_OUTPUT_DIR / "step2m4_freeze_candidate_manifest.json"
STEP2M4_REVIEW_PACK_DIR = STEP2M4_OUTPUT_DIR / "review_pack"
STEP2M4_REVIEW_PACK_MANIFEST_PATH = STEP2M4_REVIEW_PACK_DIR / "step2m4_review_pack_manifest.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path, str(path)).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    payload_rows = rows(payload)
    if len(payload_rows) > 5000:
        text = json.dumps(payload, separators=(",", ":"))
    else:
        text = json.dumps(payload, indent=2)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def write_jsonl_gz(path: Path, row_iterable: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in row_iterable:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
            handle.write("\n")


def read_jsonl_gz_rows(path: Path) -> list[dict[str, Any]]:
    require_file(path, str(path))
    output: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                output.append(json.loads(stripped))
    return output


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def copy_text_file(source: Path, destination: Path) -> None:
    write_text(destination, require_file(source, str(source)).read_text(encoding="utf-8"))


def copy_binary_file(source: Path, destination: Path) -> None:
    require_file(source, str(source))
    ensure_dir(destination.parent)
    shutil.copyfile(source, destination)


def rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("rows", [])
    return value if isinstance(value, list) else []


def compact_edge_payload_artifacts(
    payload: dict[str, Any],
    *,
    legacy_json_path: Path,
    summary_path: Path,
    sample_path: Path,
    jsonl_gz_path: Path,
    sample_limit: int = 80,
) -> dict[str, Any]:
    payload_rows = rows(payload)
    summary = dict(payload.get("summary", {})) if isinstance(payload.get("summary"), dict) else {}
    row_count = len(payload_rows)
    summary.setdefault("edge_rows", row_count)
    if "visual_continuity_edge_candidate_rows" in summary:
        summary["visual_continuity_edge_candidate_rows"] = row_count
    storage_metadata = {
        "rows_embedded": False,
        "compact_storage_manifest": True,
        "full_rows_jsonl_gz_path": str(jsonl_gz_path.resolve()),
        "summary_json_path": str(summary_path.resolve()),
        "sample_json_path": str(sample_path.resolve()),
        "sample_rows": min(sample_limit, row_count),
        "total_rows": row_count,
    }
    base = {
        key: value
        for key, value in payload.items()
        if key != "rows"
    }
    compact_manifest = guardrail_stamp(
        {
            **base,
            "created_at": utc_iso(),
            **storage_metadata,
            "summary": summary,
            "rows": [],
        }
    )
    summary_payload = guardrail_stamp(
        {
            **base,
            "created_at": utc_iso(),
            **storage_metadata,
            "summary": summary,
            "rows": [],
        }
    )
    sample_payload_value = guardrail_stamp(
        {
            **base,
            "created_at": utc_iso(),
            "rows_embedded": True,
            "compact_storage_manifest": False,
            "full_rows_jsonl_gz_path": str(jsonl_gz_path.resolve()),
            "summary_json_path": str(summary_path.resolve()),
            "sample_rows": min(sample_limit, row_count),
            "total_rows": row_count,
            "summary": summary,
            "rows": payload_rows[:sample_limit],
        }
    )
    write_jsonl_gz(jsonl_gz_path, payload_rows)
    write_json(summary_path, summary_payload)
    write_json(sample_path, sample_payload_value)
    write_json(legacy_json_path, compact_manifest)
    return compact_manifest


def read_compact_edge_payload(
    *,
    legacy_json_path: Path,
    summary_path: Path,
    jsonl_gz_path: Path,
) -> dict[str, Any]:
    if jsonl_gz_path.exists():
        payload = read_json(summary_path) if summary_path.exists() else read_json(legacy_json_path)
        payload["rows"] = read_jsonl_gz_rows(jsonl_gz_path)
        payload["rows_embedded"] = True
        payload["compact_storage_manifest"] = False
        return payload
    return read_json(legacy_json_path)


def write_edge_candidate_artifacts(edge_payload: dict[str, Any]) -> dict[str, Any]:
    return compact_edge_payload_artifacts(
        edge_payload,
        legacy_json_path=STEP2M1_EDGE_CANDIDATE_ROWS_PATH,
        summary_path=STEP2M1_EDGE_CANDIDATE_SUMMARY_PATH,
        sample_path=STEP2M1_EDGE_CANDIDATE_SAMPLE_PATH,
        jsonl_gz_path=STEP2M1_EDGE_CANDIDATE_ROWS_JSONL_GZ_PATH,
    )


def read_edge_candidate_payload() -> dict[str, Any]:
    return read_compact_edge_payload(
        legacy_json_path=STEP2M1_EDGE_CANDIDATE_ROWS_PATH,
        summary_path=STEP2M1_EDGE_CANDIDATE_SUMMARY_PATH,
        jsonl_gz_path=STEP2M1_EDGE_CANDIDATE_ROWS_JSONL_GZ_PATH,
    )


def write_human_corrected_edge_artifacts(corrected_payload: dict[str, Any]) -> dict[str, Any]:
    return compact_edge_payload_artifacts(
        corrected_payload,
        legacy_json_path=STEP2M1_HUMAN_CORRECTED_EDGE_ROWS_PATH,
        summary_path=STEP2M1_HUMAN_CORRECTED_EDGE_SUMMARY_PATH,
        sample_path=STEP2M1_HUMAN_CORRECTED_EDGE_SAMPLE_PATH,
        jsonl_gz_path=STEP2M1_HUMAN_CORRECTED_EDGE_ROWS_JSONL_GZ_PATH,
    )


def read_human_corrected_edge_payload() -> dict[str, Any]:
    return read_compact_edge_payload(
        legacy_json_path=STEP2M1_HUMAN_CORRECTED_EDGE_ROWS_PATH,
        summary_path=STEP2M1_HUMAN_CORRECTED_EDGE_SUMMARY_PATH,
        jsonl_gz_path=STEP2M1_HUMAN_CORRECTED_EDGE_ROWS_JSONL_GZ_PATH,
    )


def reviewed_decisions_file_exists_with_rows() -> bool:
    if not STEP2M1_REVIEWED_DECISIONS_PATH.exists():
        return False
    return bool(rows(read_json(STEP2M1_REVIEWED_DECISIONS_PATH)))


def reviewed_decision_template_payload() -> dict[str, Any]:
    return guardrail_stamp(
        {
            "artifact": "step2m1_reviewed_visual_continuity_decisions",
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "review_instructions": "Add rows with human_confirmed=true only after visual review in the Step2.M1 sandbox UI.",
            "allowed_human_review_decisions": [
                "accept_short_window_visual_continuity_edge",
                "reject_edge",
                "unsure_needs_later_review",
                "bulk_accept_safe_bucket",
            ],
            "rows": [],
        }
    )


def ensure_reviewed_decision_template() -> dict[str, Any]:
    if STEP2M1_REVIEWED_DECISIONS_PATH.exists():
        return read_json(STEP2M1_REVIEWED_DECISIONS_PATH)
    payload = reviewed_decision_template_payload()
    write_json(STEP2M1_REVIEWED_DECISIONS_PATH, payload)
    return payload


def sample_payload(path: Path, row_limit: int, artifact: str) -> dict[str, Any]:
    payload = read_json(path) if path.exists() else {"rows": []}
    if payload.get("compact_storage_manifest") is True and payload.get("sample_json_path"):
        sample_path = Path(str(payload.get("sample_json_path", "")))
        if sample_path.exists():
            payload = read_json(sample_path)
    payload_rows = rows(payload)
    return {
        "artifact": artifact,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "sample_rows": min(row_limit, len(payload_rows)),
        "total_rows": safe_int(payload.get("total_rows"), len(payload_rows)),
        "summary": payload.get("summary", payload.get("selection_summary", {})),
        "rows": payload_rows[:row_limit],
    }


def build_and_write_visual_continuity_sandbox(max_frame_gap: int = DEFAULT_MAX_FRAME_GAP) -> dict[str, Any]:
    from football_intelligence.step2_visual_continuity.edge_candidates import build_edge_candidate_payload
    from football_intelligence.step2_visual_continuity.grouping import build_group_payload
    from football_intelligence.step2_visual_continuity.human_corrections import apply_reviewed_decisions_payloads
    from football_intelligence.step2_visual_continuity.nodes import build_node_payload
    from football_intelligence.step2_visual_continuity.review_selection import build_review_candidate_payload
    from football_intelligence.step2_visual_continuity.review_validation import (
        write_review_progress_and_decision_summaries,
    )
    from football_intelligence.step2_visual_continuity.validation import build_and_write_validation_outputs

    f3_payload = read_json(STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_ROWS_PATH)
    g1_manifest = read_json(STEP1G1_FREEZE_CANDIDATE_MANIFEST_PATH) if STEP1G1_FREEZE_CANDIDATE_MANIFEST_PATH.exists() else {}
    node_payload = build_node_payload(f3_payload, g1_manifest)
    edge_payload = build_edge_candidate_payload(node_payload, max_frame_gap=max_frame_gap)
    group_payload = build_group_payload(node_payload, edge_payload)
    review_payload = build_review_candidate_payload(edge_payload)

    write_json(STEP2M1_NODE_ROWS_PATH, node_payload)
    write_edge_candidate_artifacts(edge_payload)
    write_json(STEP2M1_GROUP_ROWS_SANDBOX_PATH, group_payload)
    write_json(STEP2M1_REVIEW_CANDIDATE_ROWS_PATH, review_payload)

    reviewed_payload = ensure_reviewed_decision_template()
    progress, decision = write_review_progress_and_decision_summaries(review_payload, reviewed_payload)

    corrected_payload = None
    audit_payload = None
    training_rows: list[dict[str, Any]] = []
    if rows(reviewed_payload):
        corrected_payload, audit_payload, training_rows, group_payload = apply_reviewed_decisions_payloads(
            node_payload,
            edge_payload,
            review_payload,
            reviewed_payload,
        )
        write_human_corrected_edge_artifacts(corrected_payload)
        write_json(STEP2M1_CORRECTION_AUDIT_ROWS_PATH, audit_payload)
        write_jsonl(STEP2M1_TRAINING_EXAMPLES_PATH, training_rows)
        write_json(STEP2M1_GROUP_ROWS_SANDBOX_PATH, group_payload)

    validation_outputs = build_and_write_validation_outputs(
        f3_payload=f3_payload,
        node_payload=node_payload,
        edge_payload=corrected_payload or edge_payload,
        group_payload=group_payload,
        review_payload=review_payload,
        review_progress=progress,
        review_decision=decision,
        correction_audit_payload=audit_payload,
        corrected_edge_rows_available=corrected_payload is not None,
        post_review_validation_refreshed=corrected_payload is not None,
    )

    return {
        "node_payload": node_payload,
        "edge_payload": corrected_payload or edge_payload,
        "group_payload": group_payload,
        "review_payload": review_payload,
        "review_progress": progress,
        "review_decision": decision,
        "validation_outputs": validation_outputs,
    }


def output_paths_payload() -> dict[str, str]:
    return {
        "step2m1_output_dir": str(STEP2M1_OUTPUT_DIR.resolve()),
        "step2m1_visual_continuity_node_rows_path": str(STEP2M1_NODE_ROWS_PATH.resolve()),
        "step2m1_visual_continuity_edge_candidate_rows_path": str(STEP2M1_EDGE_CANDIDATE_ROWS_PATH.resolve()),
        "step2m1_visual_continuity_edge_candidate_summary_path": str(STEP2M1_EDGE_CANDIDATE_SUMMARY_PATH.resolve()),
        "step2m1_visual_continuity_edge_candidate_sample_path": str(STEP2M1_EDGE_CANDIDATE_SAMPLE_PATH.resolve()),
        "step2m1_visual_continuity_edge_candidate_rows_jsonl_gz_path": str(STEP2M1_EDGE_CANDIDATE_ROWS_JSONL_GZ_PATH.resolve()),
        "step2m1_visual_continuity_group_rows_sandbox_path": str(STEP2M1_GROUP_ROWS_SANDBOX_PATH.resolve()),
        "step2m1_visual_continuity_review_candidate_rows_path": str(STEP2M1_REVIEW_CANDIDATE_ROWS_PATH.resolve()),
        "step2m1_reviewed_visual_continuity_decisions_path": str(STEP2M1_REVIEWED_DECISIONS_PATH.resolve()),
        "step2m1_visual_continuity_review_progress_summary_path": str(STEP2M1_REVIEW_PROGRESS_SUMMARY_PATH.resolve()),
        "step2m1_visual_continuity_review_decision_summary_path": str(STEP2M1_REVIEW_DECISION_SUMMARY_PATH.resolve()),
        "step2m1_human_corrected_visual_continuity_edge_summary_path": str(STEP2M1_HUMAN_CORRECTED_EDGE_SUMMARY_PATH.resolve()),
        "step2m1_human_corrected_visual_continuity_edge_sample_path": str(STEP2M1_HUMAN_CORRECTED_EDGE_SAMPLE_PATH.resolve()),
        "step2m1_human_corrected_visual_continuity_edge_rows_jsonl_gz_path": str(STEP2M1_HUMAN_CORRECTED_EDGE_ROWS_JSONL_GZ_PATH.resolve()),
        "step2m1_visual_continuity_validation_summary_path": str(STEP2M1_VALIDATION_SUMMARY_PATH.resolve()),
        "step2m1_visual_continuity_issue_register_path": str(STEP2M1_ISSUE_REGISTER_PATH.resolve()),
        "step2m1_visual_continuity_safety_guardrail_audit_path": str(STEP2M1_SAFETY_GUARDRAIL_AUDIT_PATH.resolve()),
        "step2m1_visual_continuity_freeze_candidate_manifest_path": str(STEP2M1_FREEZE_CANDIDATE_MANIFEST_PATH.resolve()),
        "step2m1_visual_continuity_review_ui_html_path": str(STEP2M1_REVIEW_UI_HTML_PATH.resolve()),
        "step2m1_visual_continuity_review_contact_sheet_path": str(STEP2M1_REVIEW_CONTACT_SHEET_PATH.resolve()),
        "step2m1_review_pack_dir": str(STEP2M1_REVIEW_PACK_DIR.resolve()),
    }


def print_step2m1_console(outputs: dict[str, Any]) -> None:
    paths = output_paths_payload()
    for key, value in paths.items():
        print(f"{key}: {value}")
    review = outputs.get("review_payload", {})
    progress = outputs.get("review_progress", {})
    validation = outputs.get("validation_outputs", {}).get("validation_summary", {})
    print(f"review_candidate_rows: {len(rows(review))}")
    print(f"step2m1_review_scope_too_large_rebuild_candidate_rules={str(review.get('selection_summary', {}).get('step2m1_review_scope_too_large_rebuild_candidate_rules', False)).lower()}")
    print(f"reviewed_candidates: {progress.get('reviewed_candidates', 0)}")
    print(f"step2m1_high_correction_rate_rebuild_candidate_rules_recommended={str(progress.get('step2m1_high_correction_rate_rebuild_candidate_rules_recommended', False)).lower()}")
    print(f"step2m1_visual_continuity_freeze_candidate_created={str(validation.get('step2m1_visual_continuity_freeze_candidate_created', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("no_auto_promotion=true")
    print("sandbox_only=true")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("goalkeeper_slots_assigned=false")
    print("metric_analysis_performed=false")
    print(f"soccertrack_root: {SOCCERTRACK_ROOT.resolve()}")
