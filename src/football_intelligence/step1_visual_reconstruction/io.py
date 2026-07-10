# ruff: noqa: E501

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from football_intelligence.paths import (
    CLIP_ID,
    MATCH_ID,
    MATCH_ROOT,
    STAGE3C_FRAME_MANIFEST_PATH,
    STAGE3C_OFFICIAL_CANDIDATES_10FPS_PATH,
    STAGE3C_PLAYER_CANDIDATES_10FPS_PATH,
    STAGE3C_REFEREE_CANDIDATES_10FPS_PATH,
    STAGE3C_STAFF_CANDIDATES_10FPS_PATH,
    STAGE3C_UNKNOWN_CANDIDATES_10FPS_PATH,
    ensure_dir,
    require_file,
)
from football_intelligence.step1_visual_reconstruction.person_candidates import (
    build_candidate_inventory_payload,
    candidate_inventory_report_markdown,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
)
from football_intelligence.step1_visual_reconstruction.state_model import (
    build_person_states_payload,
    person_state_report_markdown,
)


STEP1_OUTPUT_DIR = MATCH_ROOT / "calibration" / "step1_visual_reconstruction"
STEP1_PERSON_CANDIDATES_PATH = STEP1_OUTPUT_DIR / "step1_person_candidates.json"
STEP1_PERSON_STATES_PATH = STEP1_OUTPUT_DIR / "step1_person_states.json"
STEP1_CANDIDATE_REPORT_PATH = STEP1_OUTPUT_DIR / "step1_candidate_inventory_report.md"
STEP1_STATE_REPORT_PATH = STEP1_OUTPUT_DIR / "step1_person_state_report.md"
STEP1_CANDIDATE_CONTACT_SHEET_PATH = STEP1_OUTPUT_DIR / "step1_candidate_contact_sheet.jpg"
STEP1_STATE_CONTACT_SHEET_PATH = STEP1_OUTPUT_DIR / "step1_state_contact_sheet.jpg"
STEP1_REVIEW_PACK_MANIFEST_PATH = STEP1_OUTPUT_DIR / "step1_review_pack_manifest.json"
STEP1_REVIEW_PACK_DIR = STEP1_OUTPUT_DIR / "review_pack"

STEP1B2_OUTPUT_DIR = STEP1_OUTPUT_DIR / "step1b2_state_threshold_audit"
STEP1B2_GOLD8_EVAL_SUMMARY_PATH = STEP1B2_OUTPUT_DIR / "step1b2_gold8_person_state_eval_summary.json"
STEP1B2_GOLD8_EVAL_REPORT_PATH = STEP1B2_OUTPUT_DIR / "step1b2_gold8_person_state_eval_report.md"
STEP1B2_ERROR_ROWS_PATH = STEP1B2_OUTPUT_DIR / "step1b2_gold8_error_rows.json"
STEP1B2_THRESHOLD_SWEEP_PATH = STEP1B2_OUTPUT_DIR / "step1b2_threshold_sweep.json"
STEP1B2_THRESHOLD_RECOMMENDATION_PATH = STEP1B2_OUTPUT_DIR / "step1b2_threshold_recommendation.md"
STEP1B2_RENDER_TIER_ROWS_PATH = STEP1B2_OUTPUT_DIR / "step1b2_render_tier_rows.json"
STEP1B2_REVIEW_CONTACT_SHEET_PATH = STEP1B2_OUTPUT_DIR / "step1b2_review_contact_sheet.jpg"
STEP1B2_GOLD8_FRAME_PANELS_DIR = STEP1B2_OUTPUT_DIR / "step1b2_gold8_frame_panels"
STEP1B2_REVIEW_PACK_MANIFEST_PATH = STEP1B2_OUTPUT_DIR / "step1b2_review_pack_manifest.json"
STEP1B2_REVIEW_PACK_DIR = STEP1B2_OUTPUT_DIR / "review_pack"

STEP1B3_OUTPUT_DIR = STEP1_OUTPUT_DIR / "step1b3_reconciliation_sandbox"
STEP1B3_RECONCILIATION_ROWS_PATH = STEP1B3_OUTPUT_DIR / "step1b3_reconciliation_rows.json"
STEP1B3_COUNT_POLICY_ROWS_PATH = STEP1B3_OUTPUT_DIR / "step1b3_count_policy_rows.json"
STEP1B3_GOLD8_EVAL_SUMMARY_PATH = STEP1B3_OUTPUT_DIR / "step1b3_gold8_eval_summary.json"
STEP1B3_GOLD8_EVAL_REPORT_PATH = STEP1B3_OUTPUT_DIR / "step1b3_gold8_eval_report.md"
STEP1B3_ERROR_ROWS_PATH = STEP1B3_OUTPUT_DIR / "step1b3_error_rows.json"
STEP1B3_BEFORE_AFTER_COMPARISON_PATH = STEP1B3_OUTPUT_DIR / "step1b3_before_after_comparison.md"
STEP1B3_REVIEW_CONTACT_SHEET_PATH = STEP1B3_OUTPUT_DIR / "step1b3_review_contact_sheet.jpg"
STEP1B3_GOLD8_FRAME_PANELS_DIR = STEP1B3_OUTPUT_DIR / "step1b3_gold8_frame_panels"
STEP1B3_REVIEW_PACK_MANIFEST_PATH = STEP1B3_OUTPUT_DIR / "step1b3_review_pack_manifest.json"
STEP1B3_REVIEW_PACK_DIR = STEP1B3_OUTPUT_DIR / "review_pack"

STEP1B4_OUTPUT_DIR = STEP1_OUTPUT_DIR / "step1b4_visible_person_base_candidate"
STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH = STEP1B4_OUTPUT_DIR / "step1b4_visible_person_base_rows.json"
STEP1B4_RETAINED_CANDIDATE_PROVENANCE_ROWS_PATH = STEP1B4_OUTPUT_DIR / "step1b4_retained_candidate_provenance_rows.json"
STEP1B4_STEP1C_INPUT_CONTRACT_PATH = STEP1B4_OUTPUT_DIR / "step1b4_step1c_input_contract.md"
STEP1B4_GOLD8_EVAL_SUMMARY_PATH = STEP1B4_OUTPUT_DIR / "step1b4_gold8_eval_summary.json"
STEP1B4_GOLD8_EVAL_REPORT_PATH = STEP1B4_OUTPUT_DIR / "step1b4_gold8_eval_report.md"
STEP1B4_BEFORE_AFTER_COMPARISON_PATH = STEP1B4_OUTPUT_DIR / "step1b4_before_after_b2_b3_b4_comparison.md"
STEP1B4_ERROR_ROWS_PATH = STEP1B4_OUTPUT_DIR / "step1b4_error_rows.json"
STEP1B4_REVIEW_DECISION_TEMPLATE_PATH = STEP1B4_OUTPUT_DIR / "step1b4_review_decision_template.json"
STEP1B4_REVIEW_CONTACT_SHEET_PATH = STEP1B4_OUTPUT_DIR / "step1b4_review_contact_sheet.jpg"
STEP1B4_GOLD8_FRAME_PANELS_DIR = STEP1B4_OUTPUT_DIR / "step1b4_gold8_frame_panels"
STEP1B4_REVIEW_PACK_MANIFEST_PATH = STEP1B4_OUTPUT_DIR / "step1b4_review_pack_manifest.json"
STEP1B4_REVIEW_PACK_DIR = STEP1B4_OUTPUT_DIR / "review_pack"

STEP1C1_OUTPUT_DIR = STEP1_OUTPUT_DIR / "step1c1_team_colour_beliefs"
STEP1C1_COLOUR_FEATURE_ROWS_PATH = STEP1C1_OUTPUT_DIR / "step1c1_colour_feature_rows.json"
STEP1C1_COLOUR_PROTOTYPES_PATH = STEP1C1_OUTPUT_DIR / "step1c1_colour_prototypes.json"
STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH = STEP1C1_OUTPUT_DIR / "step1c1_team_colour_belief_rows.json"
STEP1C1_UNKNOWN_AMBIGUOUS_COLOUR_ROWS_PATH = STEP1C1_OUTPUT_DIR / "step1c1_unknown_ambiguous_colour_rows.json"
STEP1C1_GOLD8_COLOUR_EVAL_SUMMARY_PATH = STEP1C1_OUTPUT_DIR / "step1c1_gold8_colour_eval_summary.json"
STEP1C1_GOLD8_COLOUR_EVAL_REPORT_PATH = STEP1C1_OUTPUT_DIR / "step1c1_gold8_colour_eval_report.md"
STEP1C1_COLOUR_BELIEF_REPORT_PATH = STEP1C1_OUTPUT_DIR / "step1c1_colour_belief_report.md"
STEP1C1_REVIEW_CONTACT_SHEET_PATH = STEP1C1_OUTPUT_DIR / "step1c1_review_contact_sheet.jpg"
STEP1C1_CROP_CONTACT_SHEET_PATH = STEP1C1_OUTPUT_DIR / "step1c1_crop_contact_sheet.jpg"
STEP1C1_GOLD8_FRAME_PANELS_DIR = STEP1C1_OUTPUT_DIR / "step1c1_gold8_frame_panels"
STEP1C1_REVIEW_DECISION_TEMPLATE_PATH = STEP1C1_OUTPUT_DIR / "step1c1_review_decision_template.json"
STEP1C1_REVIEW_PACK_MANIFEST_PATH = STEP1C1_OUTPUT_DIR / "step1c1_review_pack_manifest.json"
STEP1C1_REVIEW_PACK_DIR = STEP1C1_OUTPUT_DIR / "review_pack"

STEP1C1B_OUTPUT_DIR = STEP1_OUTPUT_DIR / "step1c1b_colour_profile_audit"
STEP1C1B_CROP_AUDIT_ROWS_PATH = STEP1C1B_OUTPUT_DIR / "step1c1b_crop_audit_rows.json"
STEP1C1B_CROP_AUDIT_SUMMARY_PATH = STEP1C1B_OUTPUT_DIR / "step1c1b_crop_audit_summary.json"
STEP1C1B_COLOUR_PROFILE_SWEEP_PATH = STEP1C1B_OUTPUT_DIR / "step1c1b_colour_profile_sweep.json"
STEP1C1B_PROFILE_EVAL_SUMMARY_PATH = STEP1C1B_OUTPUT_DIR / "step1c1b_profile_eval_summary.json"
STEP1C1B_PROFILE_EVAL_REPORT_PATH = STEP1C1B_OUTPUT_DIR / "step1c1b_profile_eval_report.md"
STEP1C1B_RECOMMENDED_PROFILE_PATH = STEP1C1B_OUTPUT_DIR / "step1c1b_recommended_profile_for_human_review.md"
STEP1C1B_BEST_SANDBOX_BELIEF_ROWS_PATH = STEP1C1B_OUTPUT_DIR / "step1c1b_profile_belief_rows_best_sandbox.json"
STEP1C1B_BEST_SANDBOX_UNKNOWN_AMBIGUOUS_ROWS_PATH = STEP1C1B_OUTPUT_DIR / "step1c1b_unknown_ambiguous_rows_best_sandbox.json"
STEP1C1B_GOLD8_CLUSTER_CONFUSION_ROWS_PATH = STEP1C1B_OUTPUT_DIR / "step1c1b_gold8_cluster_confusion_rows.json"
STEP1C1B_REVIEW_CONTACT_SHEET_PATH = STEP1C1B_OUTPUT_DIR / "step1c1b_review_contact_sheet.jpg"
STEP1C1B_CROP_COMPARISON_CONTACT_SHEET_PATH = STEP1C1B_OUTPUT_DIR / "step1c1b_crop_comparison_contact_sheet.jpg"
STEP1C1B_CLUSTER_CROP_CONTACT_SHEET_PATH = STEP1C1B_OUTPUT_DIR / "step1c1b_cluster_crop_contact_sheet.jpg"
STEP1C1B_REVIEW_DECISION_TEMPLATE_PATH = STEP1C1B_OUTPUT_DIR / "step1c1b_review_decision_template.json"
STEP1C1B_REVIEW_PACK_MANIFEST_PATH = STEP1C1B_OUTPUT_DIR / "step1c1b_review_pack_manifest.json"
STEP1C1B_REVIEW_PACK_DIR = STEP1C1B_OUTPUT_DIR / "review_pack"

STEP1C1C_OUTPUT_DIR = STEP1_OUTPUT_DIR / "step1c1c_manual_colour_seed_review"
STEP1C1C_COLOUR_SEED_CANDIDATE_ROWS_PATH = STEP1C1C_OUTPUT_DIR / "step1c1c_colour_seed_candidate_rows.json"
STEP1C1C_COLOUR_SEED_CANDIDATE_SUMMARY_PATH = STEP1C1C_OUTPUT_DIR / "step1c1c_colour_seed_candidate_summary.json"
STEP1C1C_MANUAL_COLOUR_SEED_LABEL_TEMPLATE_JSON_PATH = STEP1C1C_OUTPUT_DIR / "step1c1c_manual_colour_seed_label_template.json"
STEP1C1C_MANUAL_COLOUR_SEED_LABEL_TEMPLATE_CSV_PATH = STEP1C1C_OUTPUT_DIR / "step1c1c_manual_colour_seed_label_template.csv"
STEP1C1C_REVIEWED_COLOUR_SEED_LABELS_PATH = STEP1C1C_OUTPUT_DIR / "step1c1c_reviewed_colour_seed_labels.json"
STEP1C1C_SEED_CANDIDATE_CONTACT_SHEET_PATH = STEP1C1C_OUTPUT_DIR / "step1c1c_seed_candidate_contact_sheet.jpg"
STEP1C1C_SEED_CANDIDATE_CROP_SHEET_PATH = STEP1C1C_OUTPUT_DIR / "step1c1c_seed_candidate_crop_sheet.jpg"
STEP1C1C_SEED_VALIDATION_SUMMARY_PATH = STEP1C1C_OUTPUT_DIR / "step1c1c_seed_validation_summary.json"
STEP1C1C_SEEDED_COLOUR_PROTOTYPES_SANDBOX_PATH = STEP1C1C_OUTPUT_DIR / "step1c1c_seeded_colour_prototypes_sandbox.json"
STEP1C1C_SEEDED_COLOUR_BELIEF_ROWS_SANDBOX_PATH = STEP1C1C_OUTPUT_DIR / "step1c1c_seeded_colour_belief_rows_sandbox.json"
STEP1C1C_SEEDED_COLOUR_EVAL_SUMMARY_PATH = STEP1C1C_OUTPUT_DIR / "step1c1c_seeded_colour_eval_summary.json"
STEP1C1C_SEEDED_COLOUR_EVAL_REPORT_PATH = STEP1C1C_OUTPUT_DIR / "step1c1c_seeded_colour_eval_report.md"
STEP1C1C_RECOMMENDED_NEXT_ACTION_PATH = STEP1C1C_OUTPUT_DIR / "step1c1c_recommended_next_action.md"
STEP1C1C_REVIEW_DECISION_TEMPLATE_PATH = STEP1C1C_OUTPUT_DIR / "step1c1c_review_decision_template.json"
STEP1C1C_REVIEW_PACK_MANIFEST_PATH = STEP1C1C_OUTPUT_DIR / "step1c1c_review_pack_manifest.json"
STEP1C1C_REVIEW_PACK_DIR = STEP1C1C_OUTPUT_DIR / "review_pack"

STEP1C1D_OUTPUT_DIR = STEP1_OUTPUT_DIR / "step1c1d_manual_seed_review_ui"
STEP1C1D_REVIEW_UI_MANIFEST_PATH = STEP1C1D_OUTPUT_DIR / "step1c1d_review_ui_manifest.json"
STEP1C1D_REVIEW_PROGRESS_SUMMARY_PATH = STEP1C1D_OUTPUT_DIR / "step1c1d_review_progress_summary.json"
STEP1C1D_REVIEW_SESSION_STATE_PATH = STEP1C1D_OUTPUT_DIR / "step1c1d_review_session_state.json"
STEP1C1D_CANDIDATE_THUMBNAILS_DIR = STEP1C1D_OUTPUT_DIR / "step1c1d_candidate_thumbnails"
STEP1C1D_CONTEXT_IMAGES_DIR = STEP1C1D_OUTPUT_DIR / "step1c1d_context_images"
STEP1C1D_MANUAL_REVIEW_UI_HTML_PATH = STEP1C1D_OUTPUT_DIR / "step1c1d_manual_review_ui.html"
STEP1C1D_REVIEW_PACK_MANIFEST_PATH = STEP1C1D_OUTPUT_DIR / "step1c1d_review_pack_manifest.json"
STEP1C1D_REVIEW_PACK_DIR = STEP1C1D_OUTPUT_DIR / "review_pack"

STEP1C2_OUTPUT_DIR = STEP1_OUTPUT_DIR / "step1c2_colour_stability_sandbox"
STEP1C2_SHORT_BURST_COLOUR_GROUP_ROWS_PATH = STEP1C2_OUTPUT_DIR / "step1c2_short_burst_colour_group_rows.json"
STEP1C2_COLOUR_STABILITY_ROWS_PATH = STEP1C2_OUTPUT_DIR / "step1c2_colour_stability_rows.json"
STEP1C2_COLOUR_FLIP_AUDIT_ROWS_PATH = STEP1C2_OUTPUT_DIR / "step1c2_colour_flip_audit_rows.json"
STEP1C2_GOLD8_COLOUR_STABILITY_EVAL_SUMMARY_PATH = STEP1C2_OUTPUT_DIR / "step1c2_gold8_colour_stability_eval_summary.json"
STEP1C2_GOLD8_COLOUR_STABILITY_EVAL_REPORT_PATH = STEP1C2_OUTPUT_DIR / "step1c2_gold8_colour_stability_eval_report.md"
STEP1C2_COLOUR_STABILITY_REPORT_PATH = STEP1C2_OUTPUT_DIR / "step1c2_colour_stability_report.md"
STEP1C2_REVIEW_CONTACT_SHEET_PATH = STEP1C2_OUTPUT_DIR / "step1c2_review_contact_sheet.jpg"
STEP1C2_GROUP_CROP_CONTACT_SHEET_PATH = STEP1C2_OUTPUT_DIR / "step1c2_group_crop_contact_sheet.jpg"
STEP1C2_REVIEW_DECISION_TEMPLATE_PATH = STEP1C2_OUTPUT_DIR / "step1c2_review_decision_template.json"
STEP1C2_REVIEW_PACK_MANIFEST_PATH = STEP1C2_OUTPUT_DIR / "step1c2_review_pack_manifest.json"
STEP1C2_REVIEW_PACK_DIR = STEP1C2_OUTPUT_DIR / "review_pack"

STEP1C2B_OUTPUT_DIR = STEP1_OUTPUT_DIR / "step1c2b_colour_stability_human_review"
STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH = STEP1C2B_OUTPUT_DIR / "step1c2b_colour_stability_review_candidate_rows.json"
STEP1C2B_REVIEW_CANDIDATE_SUMMARY_PATH = STEP1C2B_OUTPUT_DIR / "step1c2b_colour_stability_review_candidate_summary.json"
STEP1C2B_REVIEWED_DECISIONS_PATH = STEP1C2B_OUTPUT_DIR / "step1c2b_reviewed_colour_stability_decisions.json"
STEP1C2B_REVIEW_PROGRESS_SUMMARY_PATH = STEP1C2B_OUTPUT_DIR / "step1c2b_review_progress_summary.json"
STEP1C2B_REVIEW_UI_MANIFEST_PATH = STEP1C2B_OUTPUT_DIR / "step1c2b_review_ui_manifest.json"
STEP1C2B_MANUAL_REVIEW_UI_HTML_PATH = STEP1C2B_OUTPUT_DIR / "step1c2b_manual_review_ui.html"
STEP1C2B_CANDIDATE_CROP_IMAGES_DIR = STEP1C2B_OUTPUT_DIR / "step1c2b_candidate_crop_images"
STEP1C2B_CANDIDATE_CONTEXT_IMAGES_DIR = STEP1C2B_OUTPUT_DIR / "step1c2b_candidate_context_images"
STEP1C2B_REVIEW_DECISION_SUMMARY_PATH = STEP1C2B_OUTPUT_DIR / "step1c2b_review_decision_summary.json"
STEP1C2B_RECOMMENDED_NEXT_ACTION_PATH = STEP1C2B_OUTPUT_DIR / "step1c2b_recommended_next_action.md"
STEP1C2B_REVIEW_DECISION_TEMPLATE_PATH = STEP1C2B_OUTPUT_DIR / "step1c2b_review_decision_template.json"
STEP1C2B_REVIEW_PACK_MANIFEST_PATH = STEP1C2B_OUTPUT_DIR / "step1c2b_review_pack_manifest.json"
STEP1C2B_REVIEW_PACK_DIR = STEP1C2B_OUTPUT_DIR / "review_pack"

STEP1C2C_OUTPUT_DIR = STEP1_OUTPUT_DIR / "step1c2c_human_corrected_colour_stability"
STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH = STEP1C2C_OUTPUT_DIR / "step1c2c_human_corrected_colour_stability_rows.json"
STEP1C2C_HUMAN_CORRECTION_AUDIT_ROWS_PATH = STEP1C2C_OUTPUT_DIR / "step1c2c_human_correction_audit_rows.json"
STEP1C2C_GOLD8_HUMAN_CORRECTED_COLOUR_EVAL_SUMMARY_PATH = STEP1C2C_OUTPUT_DIR / "step1c2c_gold8_human_corrected_colour_eval_summary.json"
STEP1C2C_GOLD8_HUMAN_CORRECTED_COLOUR_EVAL_REPORT_PATH = STEP1C2C_OUTPUT_DIR / "step1c2c_gold8_human_corrected_colour_eval_report.md"
STEP1C2C_HUMAN_CORRECTION_REPORT_PATH = STEP1C2C_OUTPUT_DIR / "step1c2c_human_correction_report.md"
STEP1C2C_REVIEW_CONTACT_SHEET_PATH = STEP1C2C_OUTPUT_DIR / "step1c2c_review_contact_sheet.jpg"
STEP1C2C_CORRECTION_CROP_CONTACT_SHEET_PATH = STEP1C2C_OUTPUT_DIR / "step1c2c_correction_crop_contact_sheet.jpg"
STEP1C2C_REVIEW_DECISION_TEMPLATE_PATH = STEP1C2C_OUTPUT_DIR / "step1c2c_review_decision_template.json"
STEP1C2C_REVIEW_PACK_MANIFEST_PATH = STEP1C2C_OUTPUT_DIR / "step1c2c_review_pack_manifest.json"
STEP1C2C_REVIEW_PACK_DIR = STEP1C2C_OUTPUT_DIR / "review_pack"

STEP1D1_OUTPUT_DIR = STEP1_OUTPUT_DIR / "step1d1_official_context_beliefs"
STEP1D1_OFFICIAL_CONTEXT_FEATURE_ROWS_PATH = STEP1D1_OUTPUT_DIR / "step1d1_official_context_feature_rows.json"
STEP1D1_OFFICIAL_CONTEXT_BELIEF_ROWS_PATH = STEP1D1_OUTPUT_DIR / "step1d1_official_context_belief_rows.json"
STEP1D1_OFFICIAL_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH = STEP1D1_OUTPUT_DIR / "step1d1_official_context_review_candidate_rows.json"
STEP1D1_GOLD8_OFFICIAL_CONTEXT_EVAL_SUMMARY_PATH = STEP1D1_OUTPUT_DIR / "step1d1_gold8_official_context_eval_summary.json"
STEP1D1_GOLD8_OFFICIAL_CONTEXT_EVAL_REPORT_PATH = STEP1D1_OUTPUT_DIR / "step1d1_gold8_official_context_eval_report.md"
STEP1D1_OFFICIAL_CONTEXT_BELIEF_REPORT_PATH = STEP1D1_OUTPUT_DIR / "step1d1_official_context_belief_report.md"
STEP1D1_REVIEW_CONTACT_SHEET_PATH = STEP1D1_OUTPUT_DIR / "step1d1_review_contact_sheet.jpg"
STEP1D1_CONTEXT_CROP_CONTACT_SHEET_PATH = STEP1D1_OUTPUT_DIR / "step1d1_context_crop_contact_sheet.jpg"
STEP1D1_REVIEW_DECISION_TEMPLATE_PATH = STEP1D1_OUTPUT_DIR / "step1d1_review_decision_template.json"
STEP1D1_REVIEW_PACK_MANIFEST_PATH = STEP1D1_OUTPUT_DIR / "step1d1_review_pack_manifest.json"
STEP1D1_REVIEW_PACK_DIR = STEP1D1_OUTPUT_DIR / "review_pack"

STEP1D1B_OUTPUT_DIR = STEP1_OUTPUT_DIR / "step1d1b_official_context_human_review"
STEP1D1B_REVIEW_UI_MANIFEST_PATH = STEP1D1B_OUTPUT_DIR / "step1d1b_review_ui_manifest.json"
STEP1D1B_REVIEW_PROGRESS_SUMMARY_PATH = STEP1D1B_OUTPUT_DIR / "step1d1b_review_progress_summary.json"
STEP1D1B_REVIEW_DECISION_SUMMARY_PATH = STEP1D1B_OUTPUT_DIR / "step1d1b_review_decision_summary.json"
STEP1D1B_REVIEWED_DECISIONS_PATH = STEP1D1B_OUTPUT_DIR / "step1d1b_reviewed_official_context_decisions.json"
STEP1D1B_CANDIDATE_CROP_IMAGES_DIR = STEP1D1B_OUTPUT_DIR / "step1d1b_candidate_crop_images"
STEP1D1B_CANDIDATE_CONTEXT_IMAGES_DIR = STEP1D1B_OUTPUT_DIR / "step1d1b_candidate_context_images"
STEP1D1B_CANDIDATE_FULL_FRAME_IMAGES_DIR = STEP1D1B_OUTPUT_DIR / "step1d1b_candidate_full_frame_images"
STEP1D1B_MANUAL_REVIEW_UI_HTML_PATH = STEP1D1B_OUTPUT_DIR / "step1d1b_manual_review_ui.html"
STEP1D1B_RECOMMENDED_NEXT_ACTION_PATH = STEP1D1B_OUTPUT_DIR / "step1d1b_recommended_next_action.md"
STEP1D1B_REVIEW_DECISION_TEMPLATE_PATH = STEP1D1B_OUTPUT_DIR / "step1d1b_review_decision_template.json"
STEP1D1B_REVIEW_PACK_MANIFEST_PATH = STEP1D1B_OUTPUT_DIR / "step1d1b_review_pack_manifest.json"
STEP1D1B_REVIEW_PACK_DIR = STEP1D1B_OUTPUT_DIR / "review_pack"

STEP1D1C_OUTPUT_DIR = STEP1_OUTPUT_DIR / "step1d1c_human_corrected_official_context"
STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH = STEP1D1C_OUTPUT_DIR / "step1d1c_human_corrected_official_context_rows.json"
STEP1D1C_HUMAN_CORRECTION_AUDIT_ROWS_PATH = STEP1D1C_OUTPUT_DIR / "step1d1c_human_correction_audit_rows.json"
STEP1D1C_GOLD8_HUMAN_CORRECTED_OFFICIAL_CONTEXT_EVAL_SUMMARY_PATH = STEP1D1C_OUTPUT_DIR / "step1d1c_gold8_human_corrected_official_context_eval_summary.json"
STEP1D1C_GOLD8_HUMAN_CORRECTED_OFFICIAL_CONTEXT_EVAL_REPORT_PATH = STEP1D1C_OUTPUT_DIR / "step1d1c_gold8_human_corrected_official_context_eval_report.md"
STEP1D1C_HUMAN_CORRECTION_REPORT_PATH = STEP1D1C_OUTPUT_DIR / "step1d1c_human_correction_report.md"
STEP1D1C_REVIEW_CONTACT_SHEET_PATH = STEP1D1C_OUTPUT_DIR / "step1d1c_review_contact_sheet.jpg"
STEP1D1C_CORRECTION_CROP_CONTACT_SHEET_PATH = STEP1D1C_OUTPUT_DIR / "step1d1c_correction_crop_contact_sheet.jpg"
STEP1D1C_REVIEW_DECISION_TEMPLATE_PATH = STEP1D1C_OUTPUT_DIR / "step1d1c_review_decision_template.json"
STEP1D1C_REVIEW_PACK_MANIFEST_PATH = STEP1D1C_OUTPUT_DIR / "step1d1c_review_pack_manifest.json"
STEP1D1C_REVIEW_PACK_DIR = STEP1D1C_OUTPUT_DIR / "review_pack"

STEP1E1_OUTPUT_DIR = STEP1_OUTPUT_DIR / "step1e1_goalkeeper_context_beliefs"
STEP1E1_GOALKEEPER_CONTEXT_FEATURE_ROWS_PATH = STEP1E1_OUTPUT_DIR / "step1e1_goalkeeper_context_feature_rows.json"
STEP1E1_GOALKEEPER_CONTEXT_BELIEF_ROWS_PATH = STEP1E1_OUTPUT_DIR / "step1e1_goalkeeper_context_belief_rows.json"
STEP1E1_GOALKEEPER_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH = STEP1E1_OUTPUT_DIR / "step1e1_goalkeeper_context_review_candidate_rows.json"
STEP1E1_GOLD8_GOALKEEPER_CONTEXT_EVAL_SUMMARY_PATH = STEP1E1_OUTPUT_DIR / "step1e1_gold8_goalkeeper_context_eval_summary.json"
STEP1E1_GOLD8_GOALKEEPER_CONTEXT_EVAL_REPORT_PATH = STEP1E1_OUTPUT_DIR / "step1e1_gold8_goalkeeper_context_eval_report.md"
STEP1E1_GOALKEEPER_CONTEXT_REPORT_PATH = STEP1E1_OUTPUT_DIR / "step1e1_goalkeeper_context_report.md"
STEP1E1_REVIEW_CONTACT_SHEET_PATH = STEP1E1_OUTPUT_DIR / "step1e1_review_contact_sheet.jpg"
STEP1E1_GOALKEEPER_CROP_CONTACT_SHEET_PATH = STEP1E1_OUTPUT_DIR / "step1e1_goalkeeper_crop_contact_sheet.jpg"
STEP1E1_REVIEW_DECISION_TEMPLATE_PATH = STEP1E1_OUTPUT_DIR / "step1e1_review_decision_template.json"
STEP1E1_REVIEW_PACK_MANIFEST_PATH = STEP1E1_OUTPUT_DIR / "step1e1_review_pack_manifest.json"
STEP1E1_REVIEW_PACK_DIR = STEP1E1_OUTPUT_DIR / "review_pack"

STEP1E1B_OUTPUT_DIR = STEP1_OUTPUT_DIR / "step1e1b_goalkeeper_context_human_review"
STEP1E1B_GOALKEEPER_CONTEXT_REVIEW_STATE_PATH = STEP1E1B_OUTPUT_DIR / "step1e1b_goalkeeper_context_review_state.json"
STEP1E1B_REVIEWED_GOALKEEPER_CONTEXT_DECISIONS_PATH = STEP1E1B_OUTPUT_DIR / "step1e1b_reviewed_goalkeeper_context_decisions.json"
STEP1E1B_REVIEW_PROGRESS_SUMMARY_PATH = STEP1E1B_OUTPUT_DIR / "step1e1b_review_progress_summary.json"
STEP1E1B_REVIEW_DECISION_SUMMARY_PATH = STEP1E1B_OUTPUT_DIR / "step1e1b_review_decision_summary.json"
STEP1E1B_REVIEW_UI_MANIFEST_PATH = STEP1E1B_OUTPUT_DIR / "step1e1b_review_ui_manifest.json"
STEP1E1B_REVIEW_DECISION_TEMPLATE_PATH = STEP1E1B_OUTPUT_DIR / "step1e1b_review_decision_template.json"
STEP1E1B_REVIEW_PACK_MANIFEST_PATH = STEP1E1B_OUTPUT_DIR / "step1e1b_review_pack_manifest.json"
STEP1E1B_MANUAL_REVIEW_UI_HTML_PATH = STEP1E1B_OUTPUT_DIR / "step1e1b_manual_review_ui.html"
STEP1E1B_RECOMMENDED_NEXT_ACTION_PATH = STEP1E1B_OUTPUT_DIR / "step1e1b_recommended_next_action.md"
STEP1E1B_CANDIDATE_SOURCE_FRAME_IMAGES_DIR = STEP1E1B_OUTPUT_DIR / "step1e1b_candidate_source_frame_images"
STEP1E1B_CANDIDATE_CROP_IMAGES_DIR = STEP1E1B_OUTPUT_DIR / "step1e1b_candidate_crop_images"
STEP1E1B_CANDIDATE_CONTEXT_IMAGES_DIR = STEP1E1B_OUTPUT_DIR / "step1e1b_candidate_context_images"
STEP1E1B_CANDIDATE_FULL_FRAME_IMAGES_DIR = STEP1E1B_OUTPUT_DIR / "step1e1b_candidate_full_frame_images"
STEP1E1B_REVIEW_PACK_DIR = STEP1E1B_OUTPUT_DIR / "review_pack"

STEP1E1C_OUTPUT_DIR = STEP1_OUTPUT_DIR / "step1e1c_human_corrected_goalkeeper_context"
STEP1E1C_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_ROWS_PATH = STEP1E1C_OUTPUT_DIR / "step1e1c_human_corrected_goalkeeper_context_rows.json"
STEP1E1C_HUMAN_GOALKEEPER_CORRECTION_AUDIT_ROWS_PATH = STEP1E1C_OUTPUT_DIR / "step1e1c_human_goalkeeper_correction_audit_rows.json"
STEP1E1C_GOLD8_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_EVAL_SUMMARY_PATH = STEP1E1C_OUTPUT_DIR / "step1e1c_gold8_human_corrected_goalkeeper_context_eval_summary.json"
STEP1E1C_GOLD8_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_EVAL_REPORT_PATH = STEP1E1C_OUTPUT_DIR / "step1e1c_gold8_human_corrected_goalkeeper_context_eval_report.md"
STEP1E1C_HUMAN_GOALKEEPER_CORRECTION_REPORT_PATH = STEP1E1C_OUTPUT_DIR / "step1e1c_human_goalkeeper_correction_report.md"
STEP1E1C_REVIEW_CONTACT_SHEET_PATH = STEP1E1C_OUTPUT_DIR / "step1e1c_review_contact_sheet.jpg"
STEP1E1C_CORRECTION_CROP_CONTACT_SHEET_PATH = STEP1E1C_OUTPUT_DIR / "step1e1c_correction_crop_contact_sheet.jpg"
STEP1E1C_REVIEW_DECISION_TEMPLATE_PATH = STEP1E1C_OUTPUT_DIR / "step1e1c_review_decision_template.json"
STEP1E1C_REVIEW_PACK_MANIFEST_PATH = STEP1E1C_OUTPUT_DIR / "step1e1c_review_pack_manifest.json"
STEP1E1C_REVIEW_PACK_DIR = STEP1E1C_OUTPUT_DIR / "review_pack"

STEP1F1_OUTPUT_DIR = STEP1_OUTPUT_DIR / "step1f1_fused_visual_role_state"
STEP1F1_FUSED_VISUAL_ROLE_STATE_ROWS_PATH = STEP1F1_OUTPUT_DIR / "step1f1_fused_visual_role_state_rows.json"
STEP1F1_FUSED_VISUAL_ROLE_STATE_EVAL_SUMMARY_PATH = STEP1F1_OUTPUT_DIR / "step1f1_fused_visual_role_state_eval_summary.json"
STEP1F1_FUSED_VISUAL_ROLE_STATE_EVAL_REPORT_PATH = STEP1F1_OUTPUT_DIR / "step1f1_fused_visual_role_state_eval_report.md"
STEP1F1_ROLE_STATE_CONFLICT_AUDIT_ROWS_PATH = STEP1F1_OUTPUT_DIR / "step1f1_role_state_conflict_audit_rows.json"
STEP1F1_ROLE_STATE_REPORT_PATH = STEP1F1_OUTPUT_DIR / "step1f1_role_state_report.md"
STEP1F1_REVIEW_CONTACT_SHEET_PATH = STEP1F1_OUTPUT_DIR / "step1f1_review_contact_sheet.jpg"
STEP1F1_ROLE_CROP_CONTACT_SHEET_PATH = STEP1F1_OUTPUT_DIR / "step1f1_role_crop_contact_sheet.jpg"
STEP1F1_REVIEW_DECISION_TEMPLATE_PATH = STEP1F1_OUTPUT_DIR / "step1f1_review_decision_template.json"
STEP1F1_REVIEW_PACK_MANIFEST_PATH = STEP1F1_OUTPUT_DIR / "step1f1_review_pack_manifest.json"
STEP1F1_REVIEW_PACK_DIR = STEP1F1_OUTPUT_DIR / "review_pack"

STEP1F2_OUTPUT_DIR = STEP1_OUTPUT_DIR / "step1f2_fused_visual_role_state_human_review"
STEP1F2_REVIEW_CANDIDATE_ROWS_PATH = STEP1F2_OUTPUT_DIR / "step1f2_fused_role_state_review_candidate_rows.json"
STEP1F2_REVIEWED_DECISIONS_PATH = STEP1F2_OUTPUT_DIR / "step1f2_reviewed_fused_role_state_decisions.json"
STEP1F2_REVIEW_PROGRESS_SUMMARY_PATH = STEP1F2_OUTPUT_DIR / "step1f2_review_progress_summary.json"
STEP1F2_REVIEW_DECISION_SUMMARY_PATH = STEP1F2_OUTPUT_DIR / "step1f2_review_decision_summary.json"
STEP1F2_REVIEW_CANDIDATE_SELECTION_REPORT_PATH = STEP1F2_OUTPUT_DIR / "step1f2_review_candidate_selection_report.md"
STEP1F2_REVIEW_UI_HTML_PATH = STEP1F2_OUTPUT_DIR / "step1f2_review_ui.html"
STEP1F2_REVIEW_PACK_MANIFEST_PATH = STEP1F2_OUTPUT_DIR / "step1f2_review_pack_manifest.json"
STEP1F2_REVIEW_PACK_DIR = STEP1F2_OUTPUT_DIR / "review_pack"
STEP1F2_REVIEW_CONTACT_SHEET_PATH = STEP1F2_OUTPUT_DIR / "step1f2_review_contact_sheet.jpg"
STEP1F2_CANDIDATE_SOURCE_FRAME_IMAGES_DIR = STEP1F2_OUTPUT_DIR / "step1f2_candidate_source_frame_images"
STEP1F2_CANDIDATE_CROP_IMAGES_DIR = STEP1F2_OUTPUT_DIR / "step1f2_candidate_crop_images"
STEP1F2_CANDIDATE_CONTEXT_IMAGES_DIR = STEP1F2_OUTPUT_DIR / "step1f2_candidate_context_images"
STEP1F2_CANDIDATE_FULL_FRAME_IMAGES_DIR = STEP1F2_OUTPUT_DIR / "step1f2_candidate_full_frame_images"

STEP1F3_OUTPUT_DIR = STEP1_OUTPUT_DIR / "step1f3_human_corrected_fused_visual_role_state"
STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_ROWS_PATH = STEP1F3_OUTPUT_DIR / "step1f3_human_corrected_fused_visual_role_state_rows.json"
STEP1F3_HUMAN_FUSED_ROLE_STATE_CORRECTION_AUDIT_ROWS_PATH = STEP1F3_OUTPUT_DIR / "step1f3_human_fused_role_state_correction_audit_rows.json"
STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_EVAL_SUMMARY_PATH = STEP1F3_OUTPUT_DIR / "step1f3_human_corrected_fused_visual_role_state_eval_summary.json"
STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_EVAL_REPORT_PATH = STEP1F3_OUTPUT_DIR / "step1f3_human_corrected_fused_visual_role_state_eval_report.md"
STEP1F3_HUMAN_FUSED_ROLE_STATE_CORRECTION_REPORT_PATH = STEP1F3_OUTPUT_DIR / "step1f3_human_fused_role_state_correction_report.md"
STEP1F3_REVIEW_CONTACT_SHEET_PATH = STEP1F3_OUTPUT_DIR / "step1f3_review_contact_sheet.jpg"
STEP1F3_ROLE_CROP_CONTACT_SHEET_PATH = STEP1F3_OUTPUT_DIR / "step1f3_role_crop_contact_sheet.jpg"
STEP1F3_REVIEW_DECISION_TEMPLATE_PATH = STEP1F3_OUTPUT_DIR / "step1f3_review_decision_template.json"
STEP1F3_REVIEW_PACK_MANIFEST_PATH = STEP1F3_OUTPUT_DIR / "step1f3_review_pack_manifest.json"
STEP1F3_REVIEW_PACK_DIR = STEP1F3_OUTPUT_DIR / "review_pack"

STEP1G1_OUTPUT_DIR = STEP1_OUTPUT_DIR / "step1g1_visual_reconstruction_validation"
STEP1G1_VALIDATION_SUMMARY_PATH = STEP1G1_OUTPUT_DIR / "step1g1_visual_reconstruction_validation_summary.json"
STEP1G1_VALIDATION_REPORT_PATH = STEP1G1_OUTPUT_DIR / "step1g1_visual_reconstruction_validation_report.md"
STEP1G1_FREEZE_CANDIDATE_MANIFEST_PATH = STEP1G1_OUTPUT_DIR / "step1g1_freeze_candidate_manifest.json"
STEP1G1_VISUAL_ISSUE_REGISTER_PATH = STEP1G1_OUTPUT_DIR / "step1g1_visual_issue_register.json"
STEP1G1_ROW_COUNT_AND_PROVENANCE_AUDIT_PATH = STEP1G1_OUTPUT_DIR / "step1g1_row_count_and_provenance_audit.json"
STEP1G1_SAFETY_GUARDRAIL_AUDIT_PATH = STEP1G1_OUTPUT_DIR / "step1g1_safety_guardrail_audit.json"
STEP1G1_GOLD_PROXY_VALIDATION_SUMMARY_PATH = STEP1G1_OUTPUT_DIR / "step1g1_gold_proxy_validation_summary.json"
STEP1G1_FINAL_VISUAL_ROLE_STATE_COUNTS_PATH = STEP1G1_OUTPUT_DIR / "step1g1_final_visual_role_state_counts.json"
STEP1G1_VALIDATION_CONTACT_SHEET_PATH = STEP1G1_OUTPUT_DIR / "step1g1_validation_contact_sheet.jpg"
STEP1G1_FINAL_ROLE_CROP_CONTACT_SHEET_PATH = STEP1G1_OUTPUT_DIR / "step1g1_final_role_crop_contact_sheet.jpg"
STEP1G1_FREEZE_REVIEW_DECISION_TEMPLATE_PATH = STEP1G1_OUTPUT_DIR / "step1g1_freeze_review_decision_template.json"
STEP1G1_REVIEW_PACK_MANIFEST_PATH = STEP1G1_OUTPUT_DIR / "step1g1_review_pack_manifest.json"
STEP1G1_REVIEW_PACK_DIR = STEP1G1_OUTPUT_DIR / "review_pack"

SOCCERTRACK_ROOT = Path(__file__).resolve().parents[3]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path, str(path)).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def load_manifest_frames() -> list[dict[str, Any]]:
    payload = read_json(STAGE3C_FRAME_MANIFEST_PATH)
    return sorted(payload.get("frames", []), key=lambda item: int(float(item.get("frame_sequence", -1))))


def stage3c_source_paths() -> dict[str, Path]:
    return {
        "player": STAGE3C_PLAYER_CANDIDATES_10FPS_PATH,
        "official": STAGE3C_OFFICIAL_CANDIDATES_10FPS_PATH,
        "referee": STAGE3C_REFEREE_CANDIDATES_10FPS_PATH,
        "staff": STAGE3C_STAFF_CANDIDATES_10FPS_PATH,
        "unknown": STAGE3C_UNKNOWN_CANDIDATES_10FPS_PATH,
    }


def load_stage3c_source_payloads() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    payloads: dict[str, dict[str, Any]] = {}
    paths: dict[str, str] = {}
    for source_name, path in stage3c_source_paths().items():
        payloads[source_name] = read_json(path)
        paths[source_name] = str(path.resolve())
    return payloads, paths


def build_and_write_candidate_inventory() -> dict[str, Any]:
    manifest_frames = load_manifest_frames()
    source_payloads, source_paths = load_stage3c_source_payloads()
    payload = build_candidate_inventory_payload(
        manifest_frames=manifest_frames,
        source_payloads=source_payloads,
        source_paths=source_paths,
    )
    write_json(STEP1_PERSON_CANDIDATES_PATH, payload)
    write_text(STEP1_CANDIDATE_REPORT_PATH, candidate_inventory_report_markdown(payload))
    return payload


def load_candidate_inventory() -> dict[str, Any]:
    return read_json(STEP1_PERSON_CANDIDATES_PATH)


def build_and_write_person_states() -> dict[str, Any]:
    candidate_payload = load_candidate_inventory()
    payload = build_person_states_payload(candidate_payload)
    write_json(STEP1_PERSON_STATES_PATH, payload)
    write_text(STEP1_STATE_REPORT_PATH, person_state_report_markdown(payload))
    return payload


def load_person_states() -> dict[str, Any]:
    return read_json(STEP1_PERSON_STATES_PATH)


def output_manifest_payload(
    *,
    candidate_payload: dict[str, Any],
    state_payload: dict[str, Any],
    review_pack_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "project_wide_defaults_changed_note": "No project-wide defaults were modified by Step1.A/B.",
        "stage3d_registries_changed_note": "No Stage 3D.4g, 3D.4h, or 3D.4k registry files were modified by Step1.A/B.",
        "no_metrics_calculated": True,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "outputs": {
            "step1_person_candidates_path": str(STEP1_PERSON_CANDIDATES_PATH.resolve()),
            "step1_person_states_path": str(STEP1_PERSON_STATES_PATH.resolve()),
            "candidate_inventory_report_path": str(STEP1_CANDIDATE_REPORT_PATH.resolve()),
            "person_state_report_path": str(STEP1_STATE_REPORT_PATH.resolve()),
            "candidate_contact_sheet_path": str(STEP1_CANDIDATE_CONTACT_SHEET_PATH.resolve()),
            "state_contact_sheet_path": str(STEP1_STATE_CONTACT_SHEET_PATH.resolve()),
            "review_pack_dir": str(STEP1_REVIEW_PACK_DIR.resolve()),
            "review_pack_manifest_path": str(STEP1_REVIEW_PACK_MANIFEST_PATH.resolve()),
        },
        "summary": {
            "candidate_rows": candidate_payload.get("summary", {}).get("total_rows", 0),
            "state_rows": state_payload.get("summary", {}).get("total_rows", 0),
            "state_counts": state_payload.get("summary", {}).get("state_counts", {}),
            "review_pack_file_count": len(review_pack_entries),
            "review_pack_file_limit": 20,
        },
        "review_pack_entries": review_pack_entries,
    }


def sample_payload(payload: dict[str, Any], *, row_limit: int = 20) -> dict[str, Any]:
    return {
        "artifact": payload.get("artifact"),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "summary": payload.get("summary", {}),
        "rows_sample": payload.get("rows", [])[:row_limit],
        "frame_sample": payload.get("frames", [])[:8],
    }


def copy_text_file(source: Path, destination: Path) -> None:
    write_text(destination, require_file(source, str(source)).read_text(encoding="utf-8"))


def copy_binary_file(source: Path, destination: Path) -> None:
    require_file(source, str(source))
    ensure_dir(destination.parent)
    shutil.copyfile(source, destination)


def review_index_text(candidate_payload: dict[str, Any], state_payload: dict[str, Any]) -> str:
    candidate_rows = candidate_payload.get("summary", {}).get("total_rows", 0)
    state_counts = state_payload.get("summary", {}).get("state_counts", {})
    return "\n".join(
        [
            "# Step1 Visual Reconstruction Review Pack",
            "",
            f"- Match: {MATCH_ID}",
            f"- Clip: {CLIP_ID}",
            f"- Warning: `{VISUAL_ONLY_WARNING}`",
            "- production_ready false",
            "- project_wide_defaults_changed false",
            "- stage3d_registries_changed false",
            "- no metrics calculated",
            "- no identity tracking performed",
            "- no player slots assigned",
            "",
            "## What Changed",
            "",
            "- Added a new `football_intelligence.step1_visual_reconstruction` package.",
            "- Added Step1.A inventory, Step1.B state model, visual QA rendering, and tests.",
            "- Generated candidate/state JSON, reports, contact sheets, and this compact review pack.",
            "",
            "## Output Summary",
            "",
            f"- Candidate rows: {candidate_rows}",
            f"- State counts: {state_counts}",
            "",
            "## Review Focus",
            "",
            "- Check whether the clear/partial/unknown thresholds are too strict or too loose.",
            "- Check if source-disagreement merge behavior should prefer a source or remain generic.",
            "- Check whether the contact-sheet frame selection surfaces enough official/referee and unknown context.",
        ]
    ) + "\n"


def restrictions_text() -> str:
    return "\n".join(
        [
            "# Step1 Scope And Restrictions",
            "",
            f"- `{VISUAL_ONLY_WARNING}`",
            "- Visual-only person reconstruction foundation.",
            "- No speed, distance, fatigue, player-load, team-shape, pass, dribble, tactical, or physical-performance metrics.",
            "- No identity tracking.",
            "- No player slot assignment.",
            "- 2D projection is not treated as metric truth.",
            "- Stage 3C.11, Stage 3C.12, Stage 3C.15, and new candidates are not promoted.",
            "- Stage 3D.4g, Stage 3D.4h, and Stage 3D.4k registries are not changed.",
            "- Project-wide defaults are not changed.",
            "- `production_ready=false` remains explicit on all generated payloads.",
        ]
    ) + "\n"


def tests_added_text() -> str:
    return "\n".join(
        [
            "# Tests Added",
            "",
            "- `tests/test_step1_schema.py`: required fields, visual-only warning, do-not-use flag, production_ready false.",
            "- `tests/test_step1_candidate_inventory.py`: duplicate source_detection_id merge, source rows retained, no player_slot_id requirement.",
            "- `tests/test_step1_state_model.py`: partial candidates retained, unknown source not forced to player, only clear/partial are visible observed candidates.",
            "- `tests/test_step1_restrictions.py`: no promotion imports, no Stage 3D registry/default changes, production_ready false.",
        ]
    ) + "\n"


def next_prompt_text() -> str:
    return "\n".join(
        [
            "You are reviewing the football-intelligence Step1 visual-only person reconstruction foundation.",
            "",
            "Focus on optimizing Step1.A candidate inventory canonicalization and Step1.B clear/partial/unknown state thresholds.",
            f"Respect `{VISUAL_ONLY_WARNING}` and do not introduce metrics, identity tracking, player slots, promotion paths, or Stage 3D registry changes.",
            "",
            "Review the reports, code files, JSON samples, and contact sheets in this pack. Recommend the smallest next improvement that increases visual QA reliability.",
        ]
    ) + "\n"


def build_review_pack(
    *,
    candidate_payload: dict[str, Any],
    state_payload: dict[str, Any],
) -> dict[str, Any]:
    ensure_dir(STEP1_REVIEW_PACK_DIR)
    entries: list[dict[str, Any]] = []

    def add_entry(name: str, description: str, kind: str) -> Path:
        path = STEP1_REVIEW_PACK_DIR / name
        entries.append({"name": name, "kind": kind, "description": description, "path": str(path.resolve())})
        return path

    write_text(add_entry("00_REVIEW_INDEX.md", "Review starting point and output summary.", "markdown"), review_index_text(candidate_payload, state_payload))
    write_text(add_entry("01_SCOPE_AND_RESTRICTIONS.md", "Scope guardrails from the Step1 brief.", "markdown"), restrictions_text())
    output_manifest_path = add_entry("02_OUTPUT_MANIFEST.json", "Generated Step1 output paths and high-level counts.", "json")
    write_json(
        output_manifest_path,
        output_manifest_payload(candidate_payload=candidate_payload, state_payload=state_payload, review_pack_entries=[]),
    )
    copy_text_file(STEP1_CANDIDATE_REPORT_PATH, add_entry("03_CANDIDATE_INVENTORY_REPORT.md", "Step1.A candidate counts and warnings.", "markdown"))
    copy_text_file(STEP1_STATE_REPORT_PATH, add_entry("04_PERSON_STATE_REPORT.md", "Step1.B state counts and restrictions.", "markdown"))
    write_json(add_entry("05_CANDIDATE_SAMPLE.json", "Small sample from step1_person_candidates.json.", "json"), sample_payload(candidate_payload))
    write_json(add_entry("06_STATE_SAMPLE.json", "Small sample from step1_person_states.json.", "json"), sample_payload(state_payload))

    code_files = [
        ("07_SCHEMA.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "schema.py", "Schema and visual-only validation."),
        ("08_PERSON_CANDIDATES.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "person_candidates.py", "Step1.A inventory implementation."),
        ("09_STATE_MODEL.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "state_model.py", "Step1.B state model implementation."),
        ("10_IO.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "io.py", "Output paths and review-pack generation."),
        ("11_QA_RENDER.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "qa_render.py", "Contact sheet renderer."),
        ("12_SCRIPT_STEP1A.py", SOCCERTRACK_ROOT / "scripts" / "step1a_build_person_candidate_inventory.py", "Step1.A runner script."),
        ("13_SCRIPT_STEP1B.py", SOCCERTRACK_ROOT / "scripts" / "step1b_build_person_state_model.py", "Step1.B runner script."),
        ("14_SCRIPT_RENDER_QA.py", SOCCERTRACK_ROOT / "scripts" / "step1a_step1b_render_visual_qa.py", "Visual QA runner script."),
    ]
    for name, source, description in code_files:
        copy_text_file(source, add_entry(name, description, "python"))

    write_text(add_entry("15_TESTS_ADDED.md", "Summary of tests added for this stage.", "markdown"), tests_added_text())
    copy_binary_file(STEP1_CANDIDATE_CONTACT_SHEET_PATH, add_entry("16_CANDIDATE_CONTACT_SHEET.jpg", "Step1.A visual candidate contact sheet.", "image"))
    copy_binary_file(STEP1_STATE_CONTACT_SHEET_PATH, add_entry("17_STATE_CONTACT_SHEET.jpg", "Step1.B state contact sheet.", "image"))
    write_text(add_entry("18_NEXT_OPTIMISATION_PROMPT.txt", "Prompt for the next ChatGPT review pass.", "text"), next_prompt_text())

    if len(entries) != 19:
        raise RuntimeError(f"Internal review pack count before manifest is {len(entries)}, expected 19.")
    manifest_path = add_entry("19_REVIEW_PACK_MANIFEST.json", "Review pack file manifest.", "json")
    manifest = output_manifest_payload(
        candidate_payload=candidate_payload,
        state_payload=state_payload,
        review_pack_entries=entries,
    )
    write_json(output_manifest_path, manifest)
    write_json(manifest_path, manifest)
    write_json(STEP1_REVIEW_PACK_MANIFEST_PATH, manifest)
    if len(entries) > 20:
        raise RuntimeError(f"Review pack contains {len(entries)} files; maximum is 20.")
    return manifest


def print_final_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    print(f"step1_person_candidates_path: {outputs['step1_person_candidates_path']}")
    print(f"step1_person_states_path: {outputs['step1_person_states_path']}")
    print(f"candidate_contact_sheet_path: {outputs['candidate_contact_sheet_path']}")
    print(f"state_contact_sheet_path: {outputs['state_contact_sheet_path']}")
    print(f"visual_only_warning: {VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")


def step1b2_sample_payload(path: Path, *, row_key: str = "rows", row_limit: int = 40) -> dict[str, Any]:
    payload = read_json(path)
    out = {
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "summary": payload.get("summary", payload.get("recommendation_summary", {})),
    }
    rows = payload.get(row_key, [])
    if isinstance(rows, list):
        out[f"{row_key}_sample"] = rows[:row_limit]
    return out


def step1b2_review_index_text(eval_summary: dict[str, Any], sweep: dict[str, Any]) -> str:
    recommended = sweep.get("recommendation", {}).get("recommended_profile_for_visual_review", "")
    return "\n".join(
        [
            "# Step1.B2 Gold-8 State Threshold Audit Review Pack",
            "",
            f"- Match: {MATCH_ID}",
            f"- Clip: {CLIP_ID}",
            f"- Warning: `{VISUAL_ONLY_WARNING}`",
            "- production_ready false",
            "- project_wide_defaults_changed false",
            "- stage3d_registries_changed false",
            "- no metrics calculated",
            "- no identity tracking performed",
            "- no player slots assigned",
            "- no expected 22-role states created",
            "",
            "## What This Pack Reviews",
            "",
            "- Gold-8 image-space visible-person reconstruction against Step1.B states.",
            "- A sandbox threshold sweep with no auto-promotion.",
            "- Presentation-only QA render tiers for a more readable contact sheet.",
            "",
            "## Headline Counts",
            "",
            f"- Gold visible person rows: {eval_summary.get('gold_visible_person_rows', 0)}",
            f"- Matched Gold rows: {eval_summary.get('matched_gold_visible_rows', 0)}",
            f"- Missed Gold rows: {eval_summary.get('missed_gold_visible_rows', 0)}",
            f"- Extra observed rows: {eval_summary.get('extra_observed_candidate_rows', 0)}",
            f"- Recommended visual review profile: {recommended or 'see threshold recommendation'}",
        ]
    ) + "\n"


def step1b2_scope_text() -> str:
    return "\n".join(
        [
            "# Step1.B2 Scope And Restrictions",
            "",
            f"- `{VISUAL_ONLY_WARNING}`",
            "- Gold-8 visual state threshold audit only.",
            "- Visual QA counts are only for debugging person reconstruction.",
            "- No team-colour classification.",
            "- No goalkeeper classification.",
            "- No official/referee specialist exclusion logic.",
            "- No identity tracking.",
            "- No player-slot assignment.",
            "- No expected 22-role states.",
            "- No speed, distance, fatigue, player-load, team-shape, pass, dribble, tactical, physical-performance, or football-conclusion metrics.",
            "- No Stage 3C.11, Stage 3C.12, or Stage 3C.15 promotion path is imported or called.",
            "- No Stage 3D.4g, Stage 3D.4h, or Stage 3D.4k registry file is changed.",
            "- Canonical `step1_person_states.json` is not overwritten by B2 threshold variants.",
            "- `production_ready=false` remains explicit.",
        ]
    ) + "\n"


def step1b2_tests_added_text() -> str:
    return "\n".join(
        [
            "# Step1.B2 Tests Added",
            "",
            "- `tests/test_step1b2_gold8_visual_eval.py`: completed Gold-8 loader, official inclusion, non-person exclusions, strict one-to-one matching.",
            "- `tests/test_step1b2_threshold_audit.py`: sandbox variants do not overwrite canonical Step1.B output and retain visual-only flags.",
            "- `tests/test_step1b2_render_tiers.py`: presentation-only tiering leaves state/observed visibility unchanged.",
            "- `tests/test_step1b2_restrictions.py`: forbidden keys, registry/default invariants, and no Stage 3C promotion imports.",
        ]
    ) + "\n"


def step1b2_manifest_payload(
    *,
    eval_summary: dict[str, Any],
    sweep: dict[str, Any],
    review_pack_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "no_metrics_calculated": True,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "auto_promoted_threshold_profile": False,
        "outputs": {
            "step1b2_gold8_eval_summary_path": str(STEP1B2_GOLD8_EVAL_SUMMARY_PATH.resolve()),
            "step1b2_gold8_eval_report_path": str(STEP1B2_GOLD8_EVAL_REPORT_PATH.resolve()),
            "step1b2_error_rows_path": str(STEP1B2_ERROR_ROWS_PATH.resolve()),
            "step1b2_threshold_sweep_path": str(STEP1B2_THRESHOLD_SWEEP_PATH.resolve()),
            "step1b2_threshold_recommendation_path": str(STEP1B2_THRESHOLD_RECOMMENDATION_PATH.resolve()),
            "step1b2_render_tier_rows_path": str(STEP1B2_RENDER_TIER_ROWS_PATH.resolve()),
            "step1b2_review_contact_sheet_path": str(STEP1B2_REVIEW_CONTACT_SHEET_PATH.resolve()),
            "step1b2_gold8_frame_panels_dir": str(STEP1B2_GOLD8_FRAME_PANELS_DIR.resolve()),
            "step1b2_review_pack_manifest_path": str(STEP1B2_REVIEW_PACK_MANIFEST_PATH.resolve()),
            "review_pack_dir": str(STEP1B2_REVIEW_PACK_DIR.resolve()),
        },
        "summary": {
            "gold_visible_person_rows": eval_summary.get("gold_visible_person_rows", 0),
            "matched_gold_visible_rows": eval_summary.get("matched_gold_visible_rows", 0),
            "missed_gold_visible_rows": eval_summary.get("missed_gold_visible_rows", 0),
            "extra_observed_candidate_rows": eval_summary.get("extra_observed_candidate_rows", 0),
            "recommended_profile_for_visual_review": sweep.get("recommendation", {}).get("recommended_profile_for_visual_review", ""),
            "review_pack_file_count": len(review_pack_entries),
            "review_pack_file_limit": 20,
        },
        "review_pack_entries": review_pack_entries,
    }


def build_step1b2_review_pack() -> dict[str, Any]:
    ensure_dir(STEP1B2_REVIEW_PACK_DIR)
    eval_summary = read_json(STEP1B2_GOLD8_EVAL_SUMMARY_PATH)
    sweep = read_json(STEP1B2_THRESHOLD_SWEEP_PATH)
    entries: list[dict[str, Any]] = []

    def add_entry(name: str, description: str, kind: str) -> Path:
        path = STEP1B2_REVIEW_PACK_DIR / name
        entries.append({"name": name, "kind": kind, "description": description, "path": str(path.resolve())})
        return path

    write_text(add_entry("00_REVIEW_INDEX.md", "Review starting point and summary.", "markdown"), step1b2_review_index_text(eval_summary, sweep))
    write_text(add_entry("01_SCOPE_AND_RESTRICTIONS.md", "B2 scope guardrails.", "markdown"), step1b2_scope_text())
    write_json(add_entry("02_GOLD8_VISUAL_EVAL_SUMMARY.json", "Gold-8 visible-person evaluation summary.", "json"), eval_summary)
    copy_text_file(STEP1B2_GOLD8_EVAL_REPORT_PATH, add_entry("03_GOLD8_VISUAL_EVAL_REPORT.md", "Gold-8 visible-person evaluation report.", "markdown"))
    write_json(add_entry("04_THRESHOLD_SWEEP.json", "Sandbox threshold sweep results.", "json"), sweep)
    copy_text_file(STEP1B2_THRESHOLD_RECOMMENDATION_PATH, add_entry("05_THRESHOLD_RECOMMENDATION.md", "Recommended next visual-review profile, with no auto-promotion.", "markdown"))
    write_json(add_entry("06_ERROR_ROWS_SAMPLE.json", "Sample of B2 visual QA error rows.", "json"), step1b2_sample_payload(STEP1B2_ERROR_ROWS_PATH, row_key="rows", row_limit=60))
    write_json(add_entry("07_RENDER_TIER_SAMPLE.json", "Sample of presentation-only render-tier rows.", "json"), step1b2_sample_payload(STEP1B2_RENDER_TIER_ROWS_PATH, row_key="rows", row_limit=60))
    copy_binary_file(STEP1B2_REVIEW_CONTACT_SHEET_PATH, add_entry("08_REVIEW_CONTACT_SHEET.jpg", "B2 Gold-8 multi-panel contact sheet.", "image"))

    code_files = [
        ("09_gold8_visual_eval.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "gold8_visual_eval.py", "Gold-8 visual eval implementation."),
        ("10_threshold_audit.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "threshold_audit.py", "Threshold sweep implementation."),
        ("11_render_tiers.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "render_tiers.py", "Presentation-only render tiers and renderer."),
        ("12_SCRIPT_AUDIT.py", SOCCERTRACK_ROOT / "scripts" / "step1b2_audit_person_state_thresholds_gold8.py", "B2 audit runner."),
        ("13_SCRIPT_RENDER.py", SOCCERTRACK_ROOT / "scripts" / "step1b2_render_state_threshold_review.py", "B2 render runner."),
    ]
    for name, source, description in code_files:
        copy_text_file(source, add_entry(name, description, "python"))

    write_text(add_entry("14_TESTS_ADDED.md", "Summary of B2 tests.", "markdown"), step1b2_tests_added_text())
    manifest_path = add_entry("15_REVIEW_PACK_MANIFEST.json", "B2 review pack manifest.", "json")
    manifest = step1b2_manifest_payload(eval_summary=eval_summary, sweep=sweep, review_pack_entries=entries)
    write_json(manifest_path, manifest)
    write_json(STEP1B2_REVIEW_PACK_MANIFEST_PATH, manifest)
    if len(entries) > 20:
        raise RuntimeError(f"Step1.B2 review pack contains {len(entries)} files; maximum is 20.")
    return manifest


def print_step1b2_final_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    print(f"step1b2_gold8_eval_summary_path: {outputs['step1b2_gold8_eval_summary_path']}")
    print(f"step1b2_threshold_sweep_path: {outputs['step1b2_threshold_sweep_path']}")
    print(f"step1b2_threshold_recommendation_path: {outputs['step1b2_threshold_recommendation_path']}")
    print(f"step1b2_review_contact_sheet_path: {outputs['step1b2_review_contact_sheet_path']}")
    print(f"step1b2_review_pack_manifest_path: {outputs['step1b2_review_pack_manifest_path']}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")


def step1b3_sample_payload(path: Path, *, row_key: str = "rows", row_limit: int = 50) -> dict[str, Any]:
    payload = read_json(path)
    out = {
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "summary": payload.get("summary", {}),
    }
    rows = payload.get(row_key, [])
    if isinstance(rows, list):
        out[f"{row_key}_sample"] = rows[:row_limit]
    return out


def step1b3_scope_text() -> str:
    return "\n".join(
        [
            "# Step1.B3 Scope And Restrictions",
            "",
            f"- `{VISUAL_ONLY_WARNING}`",
            "- Visual duplicate/source reconciliation and observed-count policy sandbox only.",
            "- Candidate retention is preserved; no Step1.A/B/B2 canonical artifact is overwritten.",
            "- Visual QA counts are only for debugging person reconstruction.",
            "- No team-colour classification.",
            "- No goalkeeper classification.",
            "- No official/referee specialist exclusion logic.",
            "- No identity tracking.",
            "- No player-slot assignment.",
            "- No expected 22-role states.",
            "- No football, physical-performance, tactical, pass, dribble, distance, speed, fatigue, player-load, or team-shape metrics.",
            "- No Stage 3C.11, Stage 3C.12, or Stage 3C.15 promotion path is imported or called.",
            "- No Stage 3D.4g, Stage 3D.4h, or Stage 3D.4k registry file is changed.",
            "- `production_ready=false` remains explicit.",
        ]
    ) + "\n"


def step1b3_review_index_text(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.B3 Reconciliation Sandbox Review Pack",
            "",
            f"- Match: {MATCH_ID}",
            f"- Clip: {CLIP_ID}",
            f"- Warning: `{VISUAL_ONLY_WARNING}`",
            "- production_ready false",
            "- project_wide_defaults_changed false",
            "- stage3d_registries_changed false",
            "- no identity tracking",
            "- no player slots",
            "- no auto-promotion",
            "",
            "## B2 vs B3 Headline",
            "",
            f"- B2 missed Gold rows: {summary.get('b2_missed_gold_visible_rows', 0)}",
            f"- B3 missed Gold rows: {summary.get('b3_missed_gold_visible_rows', 0)}",
            f"- B2 extra observed rows: {summary.get('b2_extra_observed_candidate_rows', 0)}",
            f"- B3 extra observed rows: {summary.get('b3_extra_observed_candidate_rows', 0)}",
            f"- B2 duplicate candidate rows: {summary.get('b2_duplicate_candidate_rows', 0)}",
            f"- B3 duplicate candidate rows: {summary.get('b3_duplicate_candidate_rows', 0)}",
            f"- Recommended for canonical review: {summary.get('b3_recommended_for_canonical_review', False)}",
        ]
    ) + "\n"


def step1b3_tests_added_text() -> str:
    return "\n".join(
        [
            "# Step1.B3 Tests Added",
            "",
            "- `tests/test_step1b3_reconciliation.py`: duplicate/source grouping and adjacent-person separation.",
            "- `tests/test_step1b3_count_policy.py`: duplicate shadows not counted, retained overlaps counted, context retained.",
            "- `tests/test_step1b3_gold8_eval.py`: B3 eval uses the B3 count flag and remains visual-only.",
            "- `tests/test_step1b3_restrictions.py`: forbidden keys, registry/default invariants, and no Stage 3C promotion imports.",
        ]
    ) + "\n"


def step1b3_manifest_payload(summary: dict[str, Any], review_pack_entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "auto_promoted_reconciliation_profile": False,
        "outputs": {
            "step1b3_reconciliation_rows_path": str(STEP1B3_RECONCILIATION_ROWS_PATH.resolve()),
            "step1b3_count_policy_rows_path": str(STEP1B3_COUNT_POLICY_ROWS_PATH.resolve()),
            "step1b3_gold8_eval_summary_path": str(STEP1B3_GOLD8_EVAL_SUMMARY_PATH.resolve()),
            "step1b3_gold8_eval_report_path": str(STEP1B3_GOLD8_EVAL_REPORT_PATH.resolve()),
            "step1b3_error_rows_path": str(STEP1B3_ERROR_ROWS_PATH.resolve()),
            "step1b3_before_after_comparison_path": str(STEP1B3_BEFORE_AFTER_COMPARISON_PATH.resolve()),
            "step1b3_review_contact_sheet_path": str(STEP1B3_REVIEW_CONTACT_SHEET_PATH.resolve()),
            "step1b3_gold8_frame_panels_dir": str(STEP1B3_GOLD8_FRAME_PANELS_DIR.resolve()),
            "step1b3_review_pack_manifest_path": str(STEP1B3_REVIEW_PACK_MANIFEST_PATH.resolve()),
            "review_pack_dir": str(STEP1B3_REVIEW_PACK_DIR.resolve()),
        },
        "summary": {
            **summary,
            "review_pack_file_count": len(review_pack_entries),
            "review_pack_file_limit": 20,
        },
        "review_pack_entries": review_pack_entries,
    }


def build_step1b3_review_pack() -> dict[str, Any]:
    ensure_dir(STEP1B3_REVIEW_PACK_DIR)
    summary = read_json(STEP1B3_GOLD8_EVAL_SUMMARY_PATH)
    entries: list[dict[str, Any]] = []

    def add_entry(name: str, description: str, kind: str) -> Path:
        path = STEP1B3_REVIEW_PACK_DIR / name
        entries.append({"name": name, "kind": kind, "description": description, "path": str(path.resolve())})
        return path

    write_text(add_entry("00_REVIEW_INDEX.md", "Review starting point and B2/B3 headline.", "markdown"), step1b3_review_index_text(summary))
    write_text(add_entry("01_SCOPE_AND_RESTRICTIONS.md", "B3 scope guardrails.", "markdown"), step1b3_scope_text())
    write_json(add_entry("02_B2_B3_COMPARISON_SUMMARY.json", "B2/B3 comparison summary.", "json"), summary)
    copy_text_file(STEP1B3_BEFORE_AFTER_COMPARISON_PATH, add_entry("03_B2_B3_COMPARISON_REPORT.md", "B2/B3 comparison report and recommendation.", "markdown"))
    write_json(add_entry("04_RECONCILIATION_SAMPLE.json", "Sample of B3 reconciliation rows.", "json"), step1b3_sample_payload(STEP1B3_RECONCILIATION_ROWS_PATH, row_limit=60))
    write_json(add_entry("05_COUNT_POLICY_SAMPLE.json", "Sample of B3 count-policy rows.", "json"), step1b3_sample_payload(STEP1B3_COUNT_POLICY_ROWS_PATH, row_limit=60))
    write_json(add_entry("06_ERROR_ROWS_SAMPLE.json", "Sample of B3 eval error rows.", "json"), step1b3_sample_payload(STEP1B3_ERROR_ROWS_PATH, row_limit=60))
    copy_binary_file(STEP1B3_REVIEW_CONTACT_SHEET_PATH, add_entry("07_REVIEW_CONTACT_SHEET.jpg", "B3 six-panel Gold-8 contact sheet.", "image"))

    code_files = [
        ("08_reconciliation.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "reconciliation.py", "B3 reconciliation grouping."),
        ("09_count_policy.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "count_policy.py", "B3 observed-count policy."),
        ("10_reconciliation_eval.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "reconciliation_eval.py", "B3 Gold-8 eval and renderer."),
        ("11_SCRIPT_BUILD.py", SOCCERTRACK_ROOT / "scripts" / "step1b3_build_reconciliation_sandbox.py", "B3 build runner."),
        ("12_SCRIPT_EVAL.py", SOCCERTRACK_ROOT / "scripts" / "step1b3_evaluate_reconciliation_gold8.py", "B3 eval runner."),
        ("13_SCRIPT_RENDER.py", SOCCERTRACK_ROOT / "scripts" / "step1b3_render_reconciliation_review.py", "B3 render runner."),
    ]
    for name, source, description in code_files:
        copy_text_file(source, add_entry(name, description, "python"))

    write_text(add_entry("14_TESTS_ADDED.md", "Summary of B3 tests.", "markdown"), step1b3_tests_added_text())
    manifest_path = add_entry("15_REVIEW_PACK_MANIFEST.json", "B3 review pack manifest.", "json")
    manifest = step1b3_manifest_payload(summary, entries)
    write_json(manifest_path, manifest)
    write_json(STEP1B3_REVIEW_PACK_MANIFEST_PATH, manifest)
    if len(entries) > 20:
        raise RuntimeError(f"Step1.B3 review pack contains {len(entries)} files; maximum is 20.")
    return manifest


def print_step1b3_final_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    summary = manifest["summary"]
    print(f"step1b3_reconciliation_rows_path: {outputs['step1b3_reconciliation_rows_path']}")
    print(f"step1b3_count_policy_rows_path: {outputs['step1b3_count_policy_rows_path']}")
    print(f"step1b3_gold8_eval_summary_path: {outputs['step1b3_gold8_eval_summary_path']}")
    print(f"step1b3_before_after_comparison_path: {outputs['step1b3_before_after_comparison_path']}")
    print(f"step1b3_review_contact_sheet_path: {outputs['step1b3_review_contact_sheet_path']}")
    print(f"step1b3_review_pack_manifest_path: {outputs['step1b3_review_pack_manifest_path']}")
    print(f"b2_missed_gold_visible_rows: {summary.get('b2_missed_gold_visible_rows', 0)}")
    print(f"b3_missed_gold_visible_rows: {summary.get('b3_missed_gold_visible_rows', 0)}")
    print(f"b2_extra_observed_candidate_rows: {summary.get('b2_extra_observed_candidate_rows', 0)}")
    print(f"b3_extra_observed_candidate_rows: {summary.get('b3_extra_observed_candidate_rows', 0)}")
    print(f"b2_duplicate_candidate_rows: {summary.get('b2_duplicate_candidate_rows', 0)}")
    print(f"b3_duplicate_candidate_rows: {summary.get('b3_duplicate_candidate_rows', 0)}")
    print(f"b3_recommended_for_canonical_review={str(summary.get('b3_recommended_for_canonical_review', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")


def step1b4_sample_payload(path: Path, *, row_key: str = "rows", row_limit: int = 50) -> dict[str, Any]:
    payload = read_json(path)
    out = {
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "summary": payload.get("summary", {}),
    }
    rows = payload.get(row_key, [])
    if isinstance(rows, list):
        out[f"{row_key}_sample"] = rows[:row_limit]
    return out


def step1b4_scope_text() -> str:
    return "\n".join(
        [
            "# Step1.B4 Scope And Restrictions",
            "",
            f"- `{VISUAL_ONLY_WARNING}`",
            "- Visible-person base candidate freeze for Step1.C input review only.",
            "- Built from B3 count-policy rows; no Step1.A/B/B2/B3 canonical or sandbox output is overwritten.",
            "- Candidate retention is preserved through `step1b4_retained_candidate_provenance_rows.json`.",
            "- No team-colour classification.",
            "- No goalkeeper classification.",
            "- No official/referee specialist exclusion logic.",
            "- No identity tracking.",
            "- No player-slot assignment.",
            "- No expected 22-role states.",
            "- No football, physical-performance, tactical, pass, dribble, distance, speed, fatigue, player-load, or team-shape metrics.",
            "- No Stage 3C.11, Stage 3C.12, or Stage 3C.15 promotion path is imported or called.",
            "- No Stage 3D.4g, Stage 3D.4h, or Stage 3D.4k registry file is changed.",
            "- `production_ready=false` remains explicit.",
        ]
    ) + "\n"


def step1b4_review_index_text(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.B4 Visible-Person Base Candidate Review Pack",
            "",
            f"- Match: {MATCH_ID}",
            f"- Clip: {CLIP_ID}",
            f"- Warning: `{VISUAL_ONLY_WARNING}`",
            "- production_ready false",
            "- project_wide_defaults_changed false",
            "- stage3d_registries_changed false",
            "- no identity tracking",
            "- no player slots",
            "- no auto-promotion",
            "",
            "## B2/B3/B4 Headline",
            "",
            f"- B3 counted observed rows: {summary.get('b3_counted_observed_visible_rows', 0)}",
            f"- B4 visible-person base rows: {summary.get('b4_visible_person_base_rows', 0)}",
            f"- B4 missed Gold rows: {summary.get('b4_missed_gold_visible_rows', 0)}",
            f"- B4 extra observed rows: {summary.get('b4_extra_observed_candidate_rows', 0)}",
            f"- B4 duplicate candidate rows: {summary.get('b4_duplicate_candidate_rows', 0)}",
            f"- Ready for Step1.C input candidate review: {summary.get('b4_ready_for_step1c_input_candidate', False)}",
        ]
    ) + "\n"


def step1b4_tests_added_text() -> str:
    return "\n".join(
        [
            "# Step1.B4 Tests Added",
            "",
            "- `tests/test_step1b4_visible_person_base.py`: B3 counted-only inclusion, shadow exclusion, provenance retention, stable unique IDs.",
            "- `tests/test_step1b4_input_contract.py`: Step1.C contract guardrails and review decision template defaults.",
            "- `tests/test_step1b4_gold8_eval.py`: B4 eval uses visible-person base rows and remains visual-only.",
            "- `tests/test_step1b4_restrictions.py`: forbidden keys, registry/default invariants, and no Stage 3C promotion imports.",
        ]
    ) + "\n"


def step1b4_manifest_payload(summary: dict[str, Any], review_pack_entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "auto_promoted": False,
        "outputs": {
            "step1b4_visible_person_base_rows_path": str(STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH.resolve()),
            "step1b4_retained_candidate_provenance_rows_path": str(STEP1B4_RETAINED_CANDIDATE_PROVENANCE_ROWS_PATH.resolve()),
            "step1b4_step1c_input_contract_path": str(STEP1B4_STEP1C_INPUT_CONTRACT_PATH.resolve()),
            "step1b4_gold8_eval_summary_path": str(STEP1B4_GOLD8_EVAL_SUMMARY_PATH.resolve()),
            "step1b4_gold8_eval_report_path": str(STEP1B4_GOLD8_EVAL_REPORT_PATH.resolve()),
            "step1b4_before_after_b2_b3_b4_comparison_path": str(STEP1B4_BEFORE_AFTER_COMPARISON_PATH.resolve()),
            "step1b4_error_rows_path": str(STEP1B4_ERROR_ROWS_PATH.resolve()),
            "step1b4_review_decision_template_path": str(STEP1B4_REVIEW_DECISION_TEMPLATE_PATH.resolve()),
            "step1b4_review_contact_sheet_path": str(STEP1B4_REVIEW_CONTACT_SHEET_PATH.resolve()),
            "step1b4_gold8_frame_panels_dir": str(STEP1B4_GOLD8_FRAME_PANELS_DIR.resolve()),
            "step1b4_review_pack_manifest_path": str(STEP1B4_REVIEW_PACK_MANIFEST_PATH.resolve()),
            "review_pack_dir": str(STEP1B4_REVIEW_PACK_DIR.resolve()),
        },
        "summary": {
            **summary,
            "review_pack_file_count": len(review_pack_entries),
            "review_pack_file_limit": 20,
        },
        "review_pack_entries": review_pack_entries,
    }


def build_step1b4_review_pack() -> dict[str, Any]:
    ensure_dir(STEP1B4_REVIEW_PACK_DIR)
    summary = read_json(STEP1B4_GOLD8_EVAL_SUMMARY_PATH)
    entries: list[dict[str, Any]] = []

    def add_entry(name: str, description: str, kind: str) -> Path:
        path = STEP1B4_REVIEW_PACK_DIR / name
        entries.append({"name": name, "kind": kind, "description": description, "path": str(path.resolve())})
        return path

    write_text(add_entry("00_REVIEW_INDEX.md", "Review starting point and B4 headline.", "markdown"), step1b4_review_index_text(summary))
    write_text(add_entry("01_SCOPE_AND_RESTRICTIONS.md", "B4 scope guardrails.", "markdown"), step1b4_scope_text())
    write_json(add_entry("02_B4_EVAL_SUMMARY.json", "B4 Gold-8 evaluation summary.", "json"), summary)
    copy_text_file(STEP1B4_GOLD8_EVAL_REPORT_PATH, add_entry("03_B4_EVAL_REPORT.md", "B4 Gold-8 eval report.", "markdown"))
    copy_text_file(STEP1B4_BEFORE_AFTER_COMPARISON_PATH, add_entry("04_B2_B3_B4_COMPARISON.md", "B2/B3/B4 comparison.", "markdown"))
    write_json(add_entry("05_VISIBLE_PERSON_BASE_SAMPLE.json", "Sample of B4 visible-person base rows.", "json"), step1b4_sample_payload(STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH, row_limit=60))
    write_json(add_entry("06_RETAINED_PROVENANCE_SAMPLE.json", "Sample of retained B3 provenance rows.", "json"), step1b4_sample_payload(STEP1B4_RETAINED_CANDIDATE_PROVENANCE_ROWS_PATH, row_limit=60))
    copy_text_file(STEP1B4_STEP1C_INPUT_CONTRACT_PATH, add_entry("07_STEP1C_INPUT_CONTRACT.md", "Step1.C input contract.", "markdown"))
    write_json(add_entry("08_REVIEW_DECISION_TEMPLATE.json", "Human review decision template.", "json"), read_json(STEP1B4_REVIEW_DECISION_TEMPLATE_PATH))
    copy_binary_file(STEP1B4_REVIEW_CONTACT_SHEET_PATH, add_entry("09_REVIEW_CONTACT_SHEET.jpg", "B4 five-panel Gold-8 contact sheet.", "image"))

    code_files = [
        ("10_visible_person_base.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "visible_person_base.py", "B4 visible-person base builder."),
        ("11_input_contracts.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "input_contracts.py", "Step1.C contract and review template."),
        ("12_visible_person_base_eval.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "visible_person_base_eval.py", "B4 Gold-8 eval and renderer."),
        ("13_SCRIPT_BUILD.py", SOCCERTRACK_ROOT / "scripts" / "step1b4_build_visible_person_base_candidate.py", "B4 build runner."),
        ("14_SCRIPT_EVAL.py", SOCCERTRACK_ROOT / "scripts" / "step1b4_evaluate_visible_person_base_gold8.py", "B4 eval runner."),
        ("15_SCRIPT_RENDER.py", SOCCERTRACK_ROOT / "scripts" / "step1b4_render_visible_person_base_review.py", "B4 render runner."),
    ]
    for name, source, description in code_files:
        copy_text_file(source, add_entry(name, description, "python"))

    write_text(add_entry("16_TESTS_ADDED.md", "Summary of B4 tests.", "markdown"), step1b4_tests_added_text())
    manifest_path = add_entry("17_REVIEW_PACK_MANIFEST.json", "B4 review pack manifest.", "json")
    manifest = step1b4_manifest_payload(summary, entries)
    write_json(manifest_path, manifest)
    write_json(STEP1B4_REVIEW_PACK_MANIFEST_PATH, manifest)
    if len(entries) > 20:
        raise RuntimeError(f"Step1.B4 review pack contains {len(entries)} files; maximum is 20.")
    return manifest


def print_step1b4_final_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    summary = manifest["summary"]
    print(f"step1b4_visible_person_base_rows_path: {outputs['step1b4_visible_person_base_rows_path']}")
    print(f"step1b4_retained_candidate_provenance_rows_path: {outputs['step1b4_retained_candidate_provenance_rows_path']}")
    print(f"step1b4_step1c_input_contract_path: {outputs['step1b4_step1c_input_contract_path']}")
    print(f"step1b4_gold8_eval_summary_path: {outputs['step1b4_gold8_eval_summary_path']}")
    print(f"step1b4_review_contact_sheet_path: {outputs['step1b4_review_contact_sheet_path']}")
    print(f"step1b4_review_pack_manifest_path: {outputs['step1b4_review_pack_manifest_path']}")
    print(f"b4_visible_person_base_rows: {summary.get('b4_visible_person_base_rows', 0)}")
    print(f"b4_missed_gold_visible_rows: {summary.get('b4_missed_gold_visible_rows', 0)}")
    print(f"b4_extra_observed_candidate_rows: {summary.get('b4_extra_observed_candidate_rows', 0)}")
    print(f"b4_duplicate_candidate_rows: {summary.get('b4_duplicate_candidate_rows', 0)}")
    print(f"b4_ready_for_step1c_input_candidate={str(summary.get('b4_ready_for_step1c_input_candidate', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")


def step1c1_sample_payload(path: Path, *, row_key: str = "rows", row_limit: int = 50) -> dict[str, Any]:
    payload = read_json(path)
    out = {
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "summary": payload.get("summary", {}),
    }
    rows = payload.get(row_key, [])
    if isinstance(rows, list):
        out[f"{row_key}_sample"] = rows[:row_limit]
    return out


def step1c1_scope_text() -> str:
    return "\n".join(
        [
            "# Step1.C1 Scope And Restrictions",
            "",
            f"- `{VISUAL_ONLY_WARNING}`",
            "- Visual-only team-colour evidence and belief candidates from Step1.B4 visible-person base rows.",
            "- No goalkeeper classification.",
            "- No official/referee specialist exclusion logic.",
            "- No identity tracking.",
            "- No player-slot assignment.",
            "- No expected 22-role states.",
            "- No football, tactical, physical-performance, pass, dribble, distance, speed, fatigue, player-load, or team-shape metrics.",
            "- Candidate type, original role source, and source labels are provenance only.",
            "- Unknown/context/off-ROI people are not forced into team labels.",
            "- No Stage 3C.11, Stage 3C.12, or Stage 3C.15 promotion path is imported or called.",
            "- No Stage 3D.4g, Stage 3D.4h, or Stage 3D.4k registry file is changed.",
            "- `production_ready=false` remains explicit.",
        ]
    ) + "\n"


def step1c1_review_index_text(eval_summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.C1 Team-Colour Belief Review Pack",
            "",
            f"- Match: {MATCH_ID}",
            f"- Clip: {CLIP_ID}",
            f"- Warning: `{VISUAL_ONLY_WARNING}`",
            "- production_ready false",
            "- project_wide_defaults_changed false",
            "- stage3d_registries_changed false",
            "- no identity tracking",
            "- no player slots",
            "- no goalkeeper or official specialist classification",
            "- no auto-promotion",
            "",
            "## Headline",
            "",
            f"- B4 visible-person base rows: {eval_summary.get('b4_visible_person_base_rows', 0)}",
            f"- Step1.C1 feature rows: {eval_summary.get('step1c1_colour_feature_rows', 0)}",
            f"- Step1.C1 belief rows: {eval_summary.get('step1c1_team_colour_belief_rows', 0)}",
            f"- Unknown/ambiguous rows: {eval_summary.get('unknown_ambiguous_colour_rows', 0)}",
            f"- Crop unusable rows: {eval_summary.get('crop_unusable_rows', 0)}",
            f"- Gold-8 colour eval available: {eval_summary.get('gold8_colour_eval_available', False)}",
        ]
    ) + "\n"


def step1c1_tests_added_text() -> str:
    return "\n".join(
        [
            "# Step1.C1 Tests Added",
            "",
            "- `tests/test_step1c1_colour_features.py`: one feature row per B4 row, missing crop handling, visual-only flags.",
            "- `tests/test_step1c1_team_colour_beliefs.py`: one belief row per B4 row, unknown/context handling, provenance preservation.",
            "- `tests/test_step1c1_eval.py`: Gold colour QA gracefully reports unavailable and does not evaluate roles/slots/identity.",
            "- `tests/test_step1c1_restrictions.py`: forbidden keys, registry/default invariants, and no Stage 3C promotion imports.",
        ]
    ) + "\n"


def step1c1_colour_belief_summary_payload() -> dict[str, Any]:
    belief_payload = read_json(STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH)
    prototypes_payload = read_json(STEP1C1_COLOUR_PROTOTYPES_PATH)
    return {
        "artifact": "step1c1_colour_belief_summary",
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "summary": belief_payload.get("summary", {}),
        "prototype_summary": prototypes_payload.get("summary", {}),
        "prototype_sandbox_only": prototypes_payload.get("prototype_sandbox_only", True),
        "auto_promoted": False,
        "safe_team_mapping_found": prototypes_payload.get("safe_team_mapping_found", False),
        "mapping_reason": prototypes_payload.get("mapping_reason", ""),
    }


def step1c1_manifest_payload(eval_summary: dict[str, Any], review_pack_entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_classification_performed": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "outputs": {
            "step1c1_colour_feature_rows_path": str(STEP1C1_COLOUR_FEATURE_ROWS_PATH.resolve()),
            "step1c1_colour_prototypes_path": str(STEP1C1_COLOUR_PROTOTYPES_PATH.resolve()),
            "step1c1_team_colour_belief_rows_path": str(STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH.resolve()),
            "step1c1_unknown_ambiguous_colour_rows_path": str(STEP1C1_UNKNOWN_AMBIGUOUS_COLOUR_ROWS_PATH.resolve()),
            "step1c1_gold8_colour_eval_summary_path": str(STEP1C1_GOLD8_COLOUR_EVAL_SUMMARY_PATH.resolve()),
            "step1c1_gold8_colour_eval_report_path": str(STEP1C1_GOLD8_COLOUR_EVAL_REPORT_PATH.resolve()),
            "step1c1_colour_belief_report_path": str(STEP1C1_COLOUR_BELIEF_REPORT_PATH.resolve()),
            "step1c1_review_contact_sheet_path": str(STEP1C1_REVIEW_CONTACT_SHEET_PATH.resolve()),
            "step1c1_crop_contact_sheet_path": str(STEP1C1_CROP_CONTACT_SHEET_PATH.resolve()),
            "step1c1_gold8_frame_panels_dir": str(STEP1C1_GOLD8_FRAME_PANELS_DIR.resolve()),
            "step1c1_review_decision_template_path": str(STEP1C1_REVIEW_DECISION_TEMPLATE_PATH.resolve()),
            "step1c1_review_pack_manifest_path": str(STEP1C1_REVIEW_PACK_MANIFEST_PATH.resolve()),
            "review_pack_dir": str(STEP1C1_REVIEW_PACK_DIR.resolve()),
        },
        "summary": {
            **eval_summary,
            "review_pack_file_count": len(review_pack_entries),
            "review_pack_file_limit": 20,
        },
        "review_pack_entries": review_pack_entries,
    }


def build_step1c1_review_pack() -> dict[str, Any]:
    ensure_dir(STEP1C1_REVIEW_PACK_DIR)
    eval_summary = read_json(STEP1C1_GOLD8_COLOUR_EVAL_SUMMARY_PATH)
    entries: list[dict[str, Any]] = []

    def add_entry(name: str, description: str, kind: str) -> Path:
        path = STEP1C1_REVIEW_PACK_DIR / name
        entries.append({"name": name, "kind": kind, "description": description, "path": str(path.resolve())})
        return path

    write_text(add_entry("00_REVIEW_INDEX.md", "Review starting point and C1 headline.", "markdown"), step1c1_review_index_text(eval_summary))
    write_text(add_entry("01_SCOPE_AND_RESTRICTIONS.md", "C1 scope guardrails.", "markdown"), step1c1_scope_text())
    write_json(add_entry("02_COLOUR_BELIEF_SUMMARY.json", "Colour belief summary.", "json"), step1c1_colour_belief_summary_payload())
    copy_text_file(STEP1C1_COLOUR_BELIEF_REPORT_PATH, add_entry("03_COLOUR_BELIEF_REPORT.md", "Colour belief report.", "markdown"))
    write_json(add_entry("04_GOLD8_COLOUR_EVAL_SUMMARY.json", "Gold-8 colour QA eval summary.", "json"), eval_summary)
    copy_text_file(STEP1C1_GOLD8_COLOUR_EVAL_REPORT_PATH, add_entry("05_GOLD8_COLOUR_EVAL_REPORT.md", "Gold-8 colour QA eval report.", "markdown"))
    write_json(add_entry("06_COLOUR_FEATURE_SAMPLE.json", "Sample of colour feature rows.", "json"), step1c1_sample_payload(STEP1C1_COLOUR_FEATURE_ROWS_PATH, row_limit=60))
    write_json(add_entry("07_TEAM_COLOUR_BELIEF_SAMPLE.json", "Sample of team-colour belief rows.", "json"), step1c1_sample_payload(STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH, row_limit=60))
    write_json(add_entry("08_UNKNOWN_AMBIGUOUS_SAMPLE.json", "Sample of unknown/ambiguous colour rows.", "json"), step1c1_sample_payload(STEP1C1_UNKNOWN_AMBIGUOUS_COLOUR_ROWS_PATH, row_limit=60))
    copy_binary_file(STEP1C1_REVIEW_CONTACT_SHEET_PATH, add_entry("09_REVIEW_CONTACT_SHEET.jpg", "C1 colour belief review contact sheet.", "image"))
    copy_binary_file(STEP1C1_CROP_CONTACT_SHEET_PATH, add_entry("10_CROP_CONTACT_SHEET.jpg", "C1 grouped torso crop contact sheet.", "image"))
    write_json(add_entry("11_REVIEW_DECISION_TEMPLATE.json", "Human review decision template.", "json"), read_json(STEP1C1_REVIEW_DECISION_TEMPLATE_PATH))

    code_files = [
        ("12_colour_features.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_features.py", "C1 colour feature extraction."),
        ("13_team_colour_beliefs.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "team_colour_beliefs.py", "C1 belief and prototype builder."),
        ("14_team_colour_eval.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "team_colour_eval.py", "C1 Gold-8 colour QA eval."),
        ("15_team_colour_render.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "team_colour_render.py", "C1 review renderers."),
        ("16_SCRIPT_BUILD.py", SOCCERTRACK_ROOT / "scripts" / "step1c1_extract_colour_features.py", "C1 feature extraction runner."),
        ("17_SCRIPT_EVAL_RENDER.py", SOCCERTRACK_ROOT / "scripts" / "step1c1_eval_render_review.py", "C1 combined eval/render runner."),
    ]
    for name, source, description in code_files:
        copy_text_file(source, add_entry(name, description, "python"))

    write_text(add_entry("18_TESTS_ADDED.md", "Summary of C1 tests.", "markdown"), step1c1_tests_added_text())
    manifest_path = add_entry("19_REVIEW_PACK_MANIFEST.json", "C1 review pack manifest.", "json")
    manifest = step1c1_manifest_payload(eval_summary, entries)
    write_json(manifest_path, manifest)
    write_json(STEP1C1_REVIEW_PACK_MANIFEST_PATH, manifest)
    if len(entries) > 20:
        raise RuntimeError(f"Step1.C1 review pack contains {len(entries)} files; maximum is 20.")
    return manifest


def print_step1c1_final_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    summary = manifest["summary"]
    print(f"step1c1_colour_feature_rows_path: {outputs['step1c1_colour_feature_rows_path']}")
    print(f"step1c1_colour_prototypes_path: {outputs['step1c1_colour_prototypes_path']}")
    print(f"step1c1_team_colour_belief_rows_path: {outputs['step1c1_team_colour_belief_rows_path']}")
    print(f"step1c1_unknown_ambiguous_colour_rows_path: {outputs['step1c1_unknown_ambiguous_colour_rows_path']}")
    print(f"step1c1_gold8_colour_eval_summary_path: {outputs['step1c1_gold8_colour_eval_summary_path']}")
    print(f"step1c1_review_contact_sheet_path: {outputs['step1c1_review_contact_sheet_path']}")
    print(f"step1c1_crop_contact_sheet_path: {outputs['step1c1_crop_contact_sheet_path']}")
    print(f"step1c1_review_pack_manifest_path: {outputs['step1c1_review_pack_manifest_path']}")
    print(f"b4_visible_person_base_rows: {summary.get('b4_visible_person_base_rows', 0)}")
    print(f"step1c1_team_colour_belief_rows: {summary.get('step1c1_team_colour_belief_rows', 0)}")
    print(f"unknown_ambiguous_colour_rows: {summary.get('unknown_ambiguous_colour_rows', 0)}")
    print(f"crop_unusable_rows: {summary.get('crop_unusable_rows', 0)}")
    print(f"high_confidence_visual_colour_rows: {summary.get('high_confidence_visual_colour_rows', 0)}")
    print(f"review_required_rows: {summary.get('review_required_rows', 0)}")
    print(f"gold8_colour_eval_available={str(summary.get('gold8_colour_eval_available', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("goalkeeper_classification_performed=false")
    print("official_specialist_exclusion_performed=false")


def step1c1b_scope_text() -> str:
    return "\n".join(
        [
            "# Step1.C1b Scope And Restrictions",
            "",
            f"- `{VISUAL_ONLY_WARNING}`",
            "- Crop/prototype audit and two-outfield-colour separation diagnostics only.",
            "- Step1.C1 canonical colour feature and belief outputs are inputs only and are not overwritten.",
            "- Sandbox crop profiles and prototype strategies are for human review only.",
            "- No team-colour mapping is auto-promoted.",
            "- No goalkeeper classification.",
            "- No official/referee specialist exclusion logic.",
            "- No identity tracking.",
            "- No player-slot assignment.",
            "- No expected 22-role states.",
            "- No football, tactical, physical-performance, pass, dribble, distance, speed, fatigue, player-load, or team-shape metrics.",
            "- Gold visible_person_type_gold is used only as visual QA proxy context.",
            "- Unknown/context/off-ROI people are not forced into team labels.",
            "- No Stage 3C.11, Stage 3C.12, or Stage 3C.15 promotion path is imported or called.",
            "- No Stage 3D.4g, Stage 3D.4h, or Stage 3D.4k registry file is changed.",
            "- `production_ready=false` remains explicit.",
        ]
    ) + "\n"


def step1c1b_review_index_text(eval_summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.C1b Colour Profile Audit Review Pack",
            "",
            f"- Match: {MATCH_ID}",
            f"- Clip: {CLIP_ID}",
            f"- Warning: `{VISUAL_ONLY_WARNING}`",
            "- production_ready false",
            "- project_wide_defaults_changed false",
            "- stage3d_registries_changed false",
            "- no identity tracking",
            "- no player slots",
            "- no goalkeeper or official specialist classification",
            "- no auto-promotion",
            "",
            "## Headline",
            "",
            f"- C1 unknown/ambiguous rows: {eval_summary.get('c1_unknown_ambiguous_colour_rows', 0)}",
            f"- Best sandbox profile: {eval_summary.get('c1b_best_profile_name', '')}",
            f"- Best prototype strategy: {eval_summary.get('c1b_best_prototype_strategy', '')}",
            f"- Best unknown/ambiguous rows: {eval_summary.get('c1b_best_unknown_ambiguous_colour_rows', 0)}",
            f"- Separation score: {eval_summary.get('c1b_team_1_team_2_separation_score', 0.0)}",
            f"- Safe for team-colour separation review: {eval_summary.get('c1b_safe_for_team_colour_separation_review', False)}",
        ]
    ) + "\n"


def step1c1b_tests_added_text() -> str:
    return "\n".join(
        [
            "# Step1.C1b Tests Added",
            "",
            "- `tests/test_step1c1b_crop_audit.py`: one audit row per C1 feature row, crop issue flags, visual-only flags.",
            "- `tests/test_step1c1b_colour_profile_sweep.py`: every profile keeps B4 row count, sandbox flags, C1 paths unchanged.",
            "- `tests/test_step1c1b_cluster_diagnostics.py`: Gold-8 proxy diagnostics report team collapse without evaluating identity, slots, roles, or metrics.",
            "- `tests/test_step1c1b_restrictions.py`: forbidden row keys, registry/default invariants, no Stage 3C promotion imports, production_ready false.",
        ]
    ) + "\n"


def step1c1b_review_decision_template_payload() -> dict[str, Any]:
    return {
        "reviewer_name": "",
        "reviewed_at": "",
        "contact_sheet_reviewed": False,
        "crop_comparison_reviewed": False,
        "cluster_crop_sheet_reviewed": False,
        "approve_c1b_profile_for_next_stage": False,
        "approve_any_team_colour_mapping": False,
        "selected_profile_name": "",
        "known_issues": [],
        "frames_requiring_manual_followup": [],
        "notes": "",
        "production_ready": PRODUCTION_READY,
        "no_auto_promotion": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
    }


def step1c1b_manifest_payload(eval_summary: dict[str, Any], review_pack_entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_classification_performed": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "outputs": {
            "step1c1b_crop_audit_rows_path": str(STEP1C1B_CROP_AUDIT_ROWS_PATH.resolve()),
            "step1c1b_crop_audit_summary_path": str(STEP1C1B_CROP_AUDIT_SUMMARY_PATH.resolve()),
            "step1c1b_colour_profile_sweep_path": str(STEP1C1B_COLOUR_PROFILE_SWEEP_PATH.resolve()),
            "step1c1b_profile_eval_summary_path": str(STEP1C1B_PROFILE_EVAL_SUMMARY_PATH.resolve()),
            "step1c1b_profile_eval_report_path": str(STEP1C1B_PROFILE_EVAL_REPORT_PATH.resolve()),
            "step1c1b_recommended_profile_path": str(STEP1C1B_RECOMMENDED_PROFILE_PATH.resolve()),
            "step1c1b_best_sandbox_belief_rows_path": str(STEP1C1B_BEST_SANDBOX_BELIEF_ROWS_PATH.resolve()),
            "step1c1b_best_sandbox_unknown_ambiguous_rows_path": str(STEP1C1B_BEST_SANDBOX_UNKNOWN_AMBIGUOUS_ROWS_PATH.resolve()),
            "step1c1b_gold8_cluster_confusion_rows_path": str(STEP1C1B_GOLD8_CLUSTER_CONFUSION_ROWS_PATH.resolve()),
            "step1c1b_review_contact_sheet_path": str(STEP1C1B_REVIEW_CONTACT_SHEET_PATH.resolve()),
            "step1c1b_crop_comparison_contact_sheet_path": str(STEP1C1B_CROP_COMPARISON_CONTACT_SHEET_PATH.resolve()),
            "step1c1b_cluster_crop_contact_sheet_path": str(STEP1C1B_CLUSTER_CROP_CONTACT_SHEET_PATH.resolve()),
            "step1c1b_review_decision_template_path": str(STEP1C1B_REVIEW_DECISION_TEMPLATE_PATH.resolve()),
            "step1c1b_review_pack_manifest_path": str(STEP1C1B_REVIEW_PACK_MANIFEST_PATH.resolve()),
            "review_pack_dir": str(STEP1C1B_REVIEW_PACK_DIR.resolve()),
        },
        "summary": {
            **eval_summary,
            "review_pack_file_count": len(review_pack_entries),
            "review_pack_file_limit": 20,
        },
        "review_pack_entries": review_pack_entries,
    }


def clear_step1c1b_review_pack_dir() -> None:
    ensure_dir(STEP1C1B_REVIEW_PACK_DIR)
    for path in STEP1C1B_REVIEW_PACK_DIR.iterdir():
        if path.is_file():
            path.unlink()


def build_step1c1b_review_pack() -> dict[str, Any]:
    clear_step1c1b_review_pack_dir()
    eval_summary = read_json(STEP1C1B_PROFILE_EVAL_SUMMARY_PATH)
    entries: list[dict[str, Any]] = []

    def add_entry(name: str, description: str, kind: str) -> Path:
        path = STEP1C1B_REVIEW_PACK_DIR / name
        entries.append({"name": name, "kind": kind, "description": description, "path": str(path.resolve())})
        return path

    write_text(add_entry("00_REVIEW_INDEX.md", "Review starting point and C1b headline.", "markdown"), step1c1b_review_index_text(eval_summary))
    write_text(add_entry("01_SCOPE_AND_RESTRICTIONS.md", "C1b scope guardrails.", "markdown"), step1c1b_scope_text())
    write_json(add_entry("02_C1B_PROFILE_EVAL_SUMMARY.json", "C1b profile eval summary.", "json"), eval_summary)
    copy_text_file(STEP1C1B_PROFILE_EVAL_REPORT_PATH, add_entry("03_C1B_PROFILE_EVAL_REPORT.md", "C1b profile eval report.", "markdown"))
    copy_text_file(STEP1C1B_RECOMMENDED_PROFILE_PATH, add_entry("04_RECOMMENDED_PROFILE_FOR_HUMAN_REVIEW.md", "Recommended sandbox profile note.", "markdown"))
    write_json(add_entry("05_CROP_AUDIT_SUMMARY.json", "Crop audit summary.", "json"), read_json(STEP1C1B_CROP_AUDIT_SUMMARY_PATH))
    write_json(add_entry("06_COLOUR_PROFILE_SWEEP.json", "Full C1b profile sweep summary.", "json"), read_json(STEP1C1B_COLOUR_PROFILE_SWEEP_PATH))
    write_json(add_entry("07_GOLD8_CLUSTER_CONFUSION_SAMPLE.json", "Sample of Gold-8 proxy confusion rows.", "json"), step1c1_sample_payload(STEP1C1B_GOLD8_CLUSTER_CONFUSION_ROWS_PATH, row_limit=80))
    write_json(add_entry("08_BEST_SANDBOX_BELIEF_SAMPLE.json", "Sample of best sandbox belief rows.", "json"), step1c1_sample_payload(STEP1C1B_BEST_SANDBOX_BELIEF_ROWS_PATH, row_limit=80))
    write_json(add_entry("09_UNKNOWN_AMBIGUOUS_SAMPLE.json", "Sample of best sandbox unknown/ambiguous rows.", "json"), step1c1_sample_payload(STEP1C1B_BEST_SANDBOX_UNKNOWN_AMBIGUOUS_ROWS_PATH, row_limit=80))
    copy_binary_file(STEP1C1B_REVIEW_CONTACT_SHEET_PATH, add_entry("10_REVIEW_CONTACT_SHEET.jpg", "C1b overlay review contact sheet.", "image"))
    copy_binary_file(STEP1C1B_CROP_COMPARISON_CONTACT_SHEET_PATH, add_entry("11_CROP_COMPARISON_CONTACT_SHEET.jpg", "C1b crop comparison contact sheet.", "image"))
    copy_binary_file(STEP1C1B_CLUSTER_CROP_CONTACT_SHEET_PATH, add_entry("12_CLUSTER_CROP_CONTACT_SHEET.jpg", "C1b cluster crop contact sheet.", "image"))
    write_json(add_entry("13_REVIEW_DECISION_TEMPLATE.json", "Human review decision template.", "json"), read_json(STEP1C1B_REVIEW_DECISION_TEMPLATE_PATH))

    code_files = [
        ("14_colour_crop_audit.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_crop_audit.py", "C1b crop audit builder."),
        ("15_colour_profile_sweep.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_profile_sweep.py", "C1b crop profile and prototype sweep."),
        ("16_colour_cluster_diagnostics.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_cluster_diagnostics.py", "C1b Gold-8 proxy cluster diagnostics."),
        ("17_colour_profile_render.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_profile_render.py", "C1b visual review renderers."),
    ]
    for name, source, description in code_files:
        copy_text_file(source, add_entry(name, description, "python"))

    write_text(add_entry("18_TESTS_ADDED.md", "Summary of C1b tests.", "markdown"), step1c1b_tests_added_text())
    manifest_path = add_entry("19_REVIEW_PACK_MANIFEST.json", "C1b review pack manifest.", "json")
    manifest = step1c1b_manifest_payload(eval_summary, entries)
    write_json(manifest_path, manifest)
    write_json(STEP1C1B_REVIEW_PACK_MANIFEST_PATH, manifest)
    if len(entries) > 20:
        raise RuntimeError(f"Step1.C1b review pack contains {len(entries)} files; maximum is 20.")
    return manifest


def print_step1c1b_final_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    summary = manifest["summary"]
    print(f"step1c1b_crop_audit_rows_path: {outputs['step1c1b_crop_audit_rows_path']}")
    print(f"step1c1b_colour_profile_sweep_path: {outputs['step1c1b_colour_profile_sweep_path']}")
    print(f"step1c1b_profile_eval_summary_path: {outputs['step1c1b_profile_eval_summary_path']}")
    print(f"step1c1b_profile_eval_report_path: {outputs['step1c1b_profile_eval_report_path']}")
    print(f"step1c1b_recommended_profile_path: {outputs['step1c1b_recommended_profile_path']}")
    print(f"step1c1b_best_sandbox_belief_rows_path: {outputs['step1c1b_best_sandbox_belief_rows_path']}")
    print(f"step1c1b_review_contact_sheet_path: {outputs['step1c1b_review_contact_sheet_path']}")
    print(f"step1c1b_crop_comparison_contact_sheet_path: {outputs['step1c1b_crop_comparison_contact_sheet_path']}")
    print(f"step1c1b_cluster_crop_contact_sheet_path: {outputs['step1c1b_cluster_crop_contact_sheet_path']}")
    print(f"step1c1b_review_pack_manifest_path: {outputs['step1c1b_review_pack_manifest_path']}")
    print(f"c1_unknown_ambiguous_colour_rows: {summary.get('c1_unknown_ambiguous_colour_rows', 0)}")
    print(f"c1b_best_profile_name: {summary.get('c1b_best_profile_name', '')}")
    print(f"c1b_best_unknown_ambiguous_colour_rows: {summary.get('c1b_best_unknown_ambiguous_colour_rows', 0)}")
    print(f"c1b_team_1_team_2_separation_score: {summary.get('c1b_team_1_team_2_separation_score', 0.0)}")
    print(f"c1b_safe_for_team_colour_separation_review={str(summary.get('c1b_safe_for_team_colour_separation_review', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("goalkeeper_classification_performed=false")
    print("official_specialist_exclusion_performed=false")


def step1c1c_scope_text() -> str:
    return "\n".join(
        [
            "# Step1.C1c Scope And Restrictions",
            "",
            f"- `{VISUAL_ONLY_WARNING}`",
            "- Manual visual seed review and seed-prototype validation only.",
            "- Step1.A/B/B2/B3/B4/C1/C1b canonical outputs are inputs only and are not overwritten.",
            "- Gold visible_person_type_gold may be used only as prefill_only visual QA context.",
            "- No seed becomes usable without human_confirmed=true in the reviewed seed label file.",
            "- No team-colour mapping is auto-promoted.",
            "- No goalkeeper classification.",
            "- No official/referee specialist exclusion logic.",
            "- No identity tracking.",
            "- No player-slot assignment.",
            "- No expected 22-role states.",
            "- No football, tactical, physical-performance, pass, dribble, distance, speed, fatigue, player-load, or team-shape metrics.",
            "- Unknown/context/off-ROI people are not forced into team labels.",
            "- No Stage 3C.11, Stage 3C.12, or Stage 3C.15 promotion path is imported or called.",
            "- No Stage 3D.4g, Stage 3D.4h, or Stage 3D.4k registry file is changed.",
            "- `production_ready=false` remains explicit.",
        ]
    ) + "\n"


def step1c1c_review_index_text(eval_summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.C1c Manual Colour Seed Review Pack",
            "",
            f"- Match: {MATCH_ID}",
            f"- Clip: {CLIP_ID}",
            f"- Warning: `{VISUAL_ONLY_WARNING}`",
            "- production_ready false",
            "- no identity tracking",
            "- no player slots",
            "- no goalkeeper or official specialist classification",
            "- no auto-promotion",
            "",
            "## Headline",
            "",
            f"- Reviewed seed labels loaded: {eval_summary.get('reviewed_seed_labels_loaded', False)}",
            f"- Reviewed seed labels valid: {eval_summary.get('reviewed_seed_labels_valid', False)}",
            f"- Team 1 confirmed seeds: {eval_summary.get('human_confirmed_team_1_seed_count', 0)}",
            f"- Team 2 confirmed seeds: {eval_summary.get('human_confirmed_team_2_seed_count', 0)}",
            f"- Safe for C2 smoothing review: {eval_summary.get('c1c_safe_for_c2_smoothing_review', False)}",
            f"- Recommendation: {eval_summary.get('c1c_safety_message', '')}",
        ]
    ) + "\n"


def step1c1c_tests_added_text() -> str:
    return "\n".join(
        [
            "# Step1.C1c Tests Added",
            "",
            "- `tests/test_step1c1c_seed_candidates.py`: candidate rows, prefill-only flags, no auto-approval.",
            "- `tests/test_step1c1c_manual_seed_schema.py`: allowed labels, human confirmation gate, Gold prefill cannot become a seed by itself.",
            "- `tests/test_step1c1c_seeded_colour_prototypes.py`: absent labels stay unpromoted, valid labels create sandbox prototypes, context rows are not forced to teams.",
            "- `tests/test_step1c1c_seeded_eval.py`: Gold proxy-only eval and safety gate behavior.",
            "- `tests/test_step1c1c_restrictions.py`: forbidden keys, registry/default invariants, no Stage 3C promotion imports, production_ready false.",
        ]
    ) + "\n"


def step1c1c_review_decision_template_payload() -> dict[str, Any]:
    return {
        "reviewer_name": "",
        "reviewed_at": "",
        "seed_candidate_contact_sheet_reviewed": False,
        "seed_crop_sheet_reviewed": False,
        "manual_seed_labels_completed": False,
        "approve_seeded_colour_profile_for_c2_sandbox": False,
        "approve_any_team_colour_mapping": False,
        "known_issues": [],
        "frames_requiring_manual_followup": [],
        "notes": "",
        "production_ready": PRODUCTION_READY,
        "no_auto_promotion": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
    }


def step1c1c_manifest_payload(eval_summary: dict[str, Any], review_pack_entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_classification_performed": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "outputs": {
            "step1c1c_colour_seed_candidate_rows_path": str(STEP1C1C_COLOUR_SEED_CANDIDATE_ROWS_PATH.resolve()),
            "step1c1c_colour_seed_candidate_summary_path": str(STEP1C1C_COLOUR_SEED_CANDIDATE_SUMMARY_PATH.resolve()),
            "step1c1c_manual_colour_seed_label_template_json_path": str(STEP1C1C_MANUAL_COLOUR_SEED_LABEL_TEMPLATE_JSON_PATH.resolve()),
            "step1c1c_manual_colour_seed_label_template_csv_path": str(STEP1C1C_MANUAL_COLOUR_SEED_LABEL_TEMPLATE_CSV_PATH.resolve()),
            "step1c1c_reviewed_colour_seed_labels_path": str(STEP1C1C_REVIEWED_COLOUR_SEED_LABELS_PATH.resolve()),
            "step1c1c_seed_candidate_contact_sheet_path": str(STEP1C1C_SEED_CANDIDATE_CONTACT_SHEET_PATH.resolve()),
            "step1c1c_seed_candidate_crop_sheet_path": str(STEP1C1C_SEED_CANDIDATE_CROP_SHEET_PATH.resolve()),
            "step1c1c_seed_validation_summary_path": str(STEP1C1C_SEED_VALIDATION_SUMMARY_PATH.resolve()),
            "step1c1c_seeded_colour_prototypes_sandbox_path": str(STEP1C1C_SEEDED_COLOUR_PROTOTYPES_SANDBOX_PATH.resolve()),
            "step1c1c_seeded_colour_belief_rows_sandbox_path": str(STEP1C1C_SEEDED_COLOUR_BELIEF_ROWS_SANDBOX_PATH.resolve()),
            "step1c1c_seeded_colour_eval_summary_path": str(STEP1C1C_SEEDED_COLOUR_EVAL_SUMMARY_PATH.resolve()),
            "step1c1c_seeded_colour_eval_report_path": str(STEP1C1C_SEEDED_COLOUR_EVAL_REPORT_PATH.resolve()),
            "step1c1c_recommended_next_action_path": str(STEP1C1C_RECOMMENDED_NEXT_ACTION_PATH.resolve()),
            "step1c1c_review_decision_template_path": str(STEP1C1C_REVIEW_DECISION_TEMPLATE_PATH.resolve()),
            "step1c1c_review_pack_manifest_path": str(STEP1C1C_REVIEW_PACK_MANIFEST_PATH.resolve()),
            "review_pack_dir": str(STEP1C1C_REVIEW_PACK_DIR.resolve()),
        },
        "summary": {
            **eval_summary,
            "review_pack_file_count": len(review_pack_entries),
            "review_pack_file_limit": 20,
        },
        "review_pack_entries": review_pack_entries,
    }


def clear_step1c1c_review_pack_dir() -> None:
    ensure_dir(STEP1C1C_REVIEW_PACK_DIR)
    for path in STEP1C1C_REVIEW_PACK_DIR.iterdir():
        if path.is_file():
            path.unlink()


def build_step1c1c_review_pack() -> dict[str, Any]:
    clear_step1c1c_review_pack_dir()
    eval_summary = read_json(STEP1C1C_SEEDED_COLOUR_EVAL_SUMMARY_PATH)
    entries: list[dict[str, Any]] = []

    def add_entry(name: str, description: str, kind: str) -> Path:
        path = STEP1C1C_REVIEW_PACK_DIR / name
        entries.append({"name": name, "kind": kind, "description": description, "path": str(path.resolve())})
        return path

    write_json(STEP1C1C_REVIEW_DECISION_TEMPLATE_PATH, step1c1c_review_decision_template_payload())
    write_text(add_entry("00_REVIEW_INDEX.md", "Review starting point and C1c headline.", "markdown"), step1c1c_review_index_text(eval_summary))
    write_text(add_entry("01_SCOPE_AND_RESTRICTIONS.md", "C1c scope guardrails.", "markdown"), step1c1c_scope_text())
    write_json(add_entry("02_SEED_CANDIDATE_SUMMARY.json", "Seed candidate summary.", "json"), read_json(STEP1C1C_COLOUR_SEED_CANDIDATE_SUMMARY_PATH))
    write_json(add_entry("03_MANUAL_LABEL_TEMPLATE.json", "Manual seed label JSON template.", "json"), read_json(STEP1C1C_MANUAL_COLOUR_SEED_LABEL_TEMPLATE_JSON_PATH))
    copy_text_file(STEP1C1C_MANUAL_COLOUR_SEED_LABEL_TEMPLATE_CSV_PATH, add_entry("04_MANUAL_LABEL_TEMPLATE.csv", "Manual seed label CSV template.", "csv"))
    write_json(add_entry("05_SEED_VALIDATION_SUMMARY.json", "Reviewed seed validation summary.", "json"), read_json(STEP1C1C_SEED_VALIDATION_SUMMARY_PATH))
    write_json(add_entry("06_SEEDED_COLOUR_EVAL_SUMMARY.json", "Seeded colour eval summary.", "json"), eval_summary)
    copy_text_file(STEP1C1C_SEEDED_COLOUR_EVAL_REPORT_PATH, add_entry("07_SEEDED_COLOUR_EVAL_REPORT.md", "Seeded colour eval report.", "markdown"))
    copy_text_file(STEP1C1C_RECOMMENDED_NEXT_ACTION_PATH, add_entry("08_RECOMMENDED_NEXT_ACTION.md", "Recommended next action.", "markdown"))
    write_json(add_entry("09_SEED_CANDIDATE_SAMPLE.json", "Sample of seed candidate rows.", "json"), step1c1_sample_payload(STEP1C1C_COLOUR_SEED_CANDIDATE_ROWS_PATH, row_limit=80))
    copy_binary_file(STEP1C1C_SEED_CANDIDATE_CONTACT_SHEET_PATH, add_entry("10_SEED_CANDIDATE_CONTACT_SHEET.jpg", "Seed candidate frame contact sheet.", "image"))
    copy_binary_file(STEP1C1C_SEED_CANDIDATE_CROP_SHEET_PATH, add_entry("11_SEED_CANDIDATE_CROP_SHEET.jpg", "Grouped seed candidate crop sheet.", "image"))
    write_json(add_entry("12_REVIEW_DECISION_TEMPLATE.json", "Human review decision template.", "json"), read_json(STEP1C1C_REVIEW_DECISION_TEMPLATE_PATH))

    code_files = [
        ("13_colour_seed_candidates.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_seed_candidates.py", "C1c seed candidate builder."),
        ("14_manual_colour_seed_schema.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "manual_colour_seed_schema.py", "C1c manual seed schema validator."),
        ("15_seeded_colour_prototypes.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "seeded_colour_prototypes.py", "C1c seeded prototype sandbox."),
        ("16_seeded_colour_eval.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "seeded_colour_eval.py", "C1c seeded eval."),
        ("17_colour_seed_render.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_seed_render.py", "C1c seed review renderers."),
    ]
    for name, source, description in code_files:
        copy_text_file(source, add_entry(name, description, "python"))

    write_text(add_entry("18_TESTS_ADDED.md", "Summary of C1c tests.", "markdown"), step1c1c_tests_added_text())
    manifest_path = add_entry("19_REVIEW_PACK_MANIFEST.json", "C1c review pack manifest.", "json")
    manifest = step1c1c_manifest_payload(eval_summary, entries)
    write_json(manifest_path, manifest)
    write_json(STEP1C1C_REVIEW_PACK_MANIFEST_PATH, manifest)
    if len(entries) > 20:
        raise RuntimeError(f"Step1.C1c review pack contains {len(entries)} files; maximum is 20.")
    return manifest


def print_step1c1c_final_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    summary = manifest["summary"]
    print(f"step1c1c_colour_seed_candidate_rows_path: {outputs['step1c1c_colour_seed_candidate_rows_path']}")
    print(f"step1c1c_manual_colour_seed_label_template_json_path: {outputs['step1c1c_manual_colour_seed_label_template_json_path']}")
    print(f"step1c1c_manual_colour_seed_label_template_csv_path: {outputs['step1c1c_manual_colour_seed_label_template_csv_path']}")
    print(f"step1c1c_seed_candidate_contact_sheet_path: {outputs['step1c1c_seed_candidate_contact_sheet_path']}")
    print(f"step1c1c_seed_candidate_crop_sheet_path: {outputs['step1c1c_seed_candidate_crop_sheet_path']}")
    print(f"step1c1c_seed_validation_summary_path: {outputs['step1c1c_seed_validation_summary_path']}")
    print(f"step1c1c_seeded_colour_eval_summary_path: {outputs['step1c1c_seeded_colour_eval_summary_path']}")
    print(f"step1c1c_recommended_next_action_path: {outputs['step1c1c_recommended_next_action_path']}")
    print(f"step1c1c_review_pack_manifest_path: {outputs['step1c1c_review_pack_manifest_path']}")
    print(f"reviewed_seed_labels_loaded={str(summary.get('reviewed_seed_labels_loaded', False)).lower()}")
    print(f"reviewed_seed_labels_valid={str(summary.get('reviewed_seed_labels_valid', False)).lower()}")
    print(f"human_confirmed_team_1_seed_count: {summary.get('human_confirmed_team_1_seed_count', 0)}")
    print(f"human_confirmed_team_2_seed_count: {summary.get('human_confirmed_team_2_seed_count', 0)}")
    print(f"c1c_safe_for_c2_smoothing_review={str(summary.get('c1c_safe_for_c2_smoothing_review', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("goalkeeper_classification_performed=false")
    print("official_specialist_exclusion_performed=false")
