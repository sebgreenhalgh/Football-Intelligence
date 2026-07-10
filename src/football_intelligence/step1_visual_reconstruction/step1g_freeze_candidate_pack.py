# ruff: noqa: E501

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.paths import CLIP_ID, MATCH_ID
from football_intelligence.step1_visual_reconstruction.io import (
    SOCCERTRACK_ROOT,
    STEP1G1_FINAL_ROLE_CROP_CONTACT_SHEET_PATH,
    STEP1G1_FINAL_VISUAL_ROLE_STATE_COUNTS_PATH,
    STEP1G1_FREEZE_CANDIDATE_MANIFEST_PATH,
    STEP1G1_FREEZE_REVIEW_DECISION_TEMPLATE_PATH,
    STEP1G1_GOLD_PROXY_VALIDATION_SUMMARY_PATH,
    STEP1G1_REVIEW_PACK_DIR,
    STEP1G1_REVIEW_PACK_MANIFEST_PATH,
    STEP1G1_ROW_COUNT_AND_PROVENANCE_AUDIT_PATH,
    STEP1G1_SAFETY_GUARDRAIL_AUDIT_PATH,
    STEP1G1_VALIDATION_CONTACT_SHEET_PATH,
    STEP1G1_VALIDATION_REPORT_PATH,
    STEP1G1_VALIDATION_SUMMARY_PATH,
    STEP1G1_VISUAL_ISSUE_REGISTER_PATH,
    copy_binary_file,
    copy_text_file,
    write_json,
    write_text,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
)
from football_intelligence.step1_visual_reconstruction.step1g_visual_reconstruction_validation import build_and_write_step1g1_validation


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def review_pack_file_names() -> list[str]:
    return [
        "00_REVIEW_INDEX.md",
        "01_SCOPE_AND_RESTRICTIONS.md",
        "02_STEP1G_VALIDATION_SUMMARY.json",
        "03_STEP1G_VALIDATION_REPORT.md",
        "04_FREEZE_CANDIDATE_MANIFEST.json",
        "05_VISUAL_ISSUE_REGISTER.json",
        "06_ROW_COUNT_AND_PROVENANCE_AUDIT.json",
        "07_SAFETY_GUARDRAIL_AUDIT.json",
        "08_GOLD_PROXY_VALIDATION_SUMMARY.json",
        "09_FINAL_ROLE_STATE_COUNTS.json",
        "10_VALIDATION_CONTACT_SHEET.jpg",
        "11_FINAL_ROLE_CROP_CONTACT_SHEET.jpg",
        "12_FREEZE_REVIEW_DECISION_TEMPLATE.json",
        "13_step1g_visual_reconstruction_validation.py",
        "14_step1g_visual_reconstruction_render.py",
        "15_step1g_freeze_candidate_pack.py",
        "16_SCRIPT_VALIDATE_G1.py",
        "17_SCRIPT_RENDER_G1.py",
        "18_TESTS_ADDED.md",
        "19_REVIEW_PACK_MANIFEST.json",
    ]


def review_index_text(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.G1 Freeze-Candidate Review Index",
            "",
            f"- B4/C2c/D1c/E1c/F3 rows: {summary.get('b4_row_count', 0)} / {summary.get('c2c_row_count', 0)} / {summary.get('d1c_row_count', 0)} / {summary.get('e1c_row_count', 0)} / {summary.get('f3_row_count', 0)}",
            f"- Freeze candidate created: {summary.get('step1g1_freeze_candidate_created', False)}",
            f"- Human approved: {summary.get('step1g1_freeze_candidate_human_approved', False)}",
            f"- Safe for Step2 visual continuity candidate: {summary.get('step1g1_safe_for_step2_visual_continuity_candidate', False)}",
            f"- Blocking issue count: {summary.get('blocking_issue_count', 0)}",
            f"- Visual flag: {VISUAL_ONLY_WARNING}",
            "- production_ready=false",
            "- no_auto_promotion=true",
        ]
    ) + "\n"


def scope_text() -> str:
    return "\n".join(
        [
            "# Step1.G1 Scope And Restrictions",
            "",
            "Step1.G1 validates and packages the Step 1 visual reconstruction freeze candidate for human signoff.",
            "",
            "- Validation/reporting only; no upstream artifacts are overwritten.",
            "- Visual-only; no metric, event, tactical, physical-performance, or football-conclusion calculations.",
            "- No identity tracking, player slots, goalkeeper slots, expected role states, exact-count forcing, official/referee exclusion, deletion, or promotion.",
            "- Team and goalkeeper labels remain visual context only.",
            "- Unknown and review-required rows remain retained uncertainty.",
            "- The freeze candidate is not canonical and not production-ready.",
        ]
    ) + "\n"


def tests_added_text() -> str:
    return "\n".join(
        [
            "# Step1.G1 Tests Added",
            "",
            "- `tests/test_step1g1_visual_reconstruction_validation.py` covers row preservation, ID alignment, F3-safe input, issue register, freeze manifest, and human approval default false.",
            "- `tests/test_step1g1_freeze_candidate_pack.py` covers review-pack file count, required filenames, decision template defaults, and contact-sheet nonblank checks.",
            "- `tests/test_step1g1_restrictions.py` covers forbidden fields, no slot/identity/metric/exclusion approvals, registry/default invariants, no promotion strings, production_ready=false, and no_auto_promotion=true.",
        ]
    ) + "\n"


def clear_review_pack_dir() -> None:
    STEP1G1_REVIEW_PACK_DIR.mkdir(parents=True, exist_ok=True)
    for path in STEP1G1_REVIEW_PACK_DIR.iterdir():
        if path.is_file():
            path.unlink()


def build_step1g1_freeze_candidate_review_pack() -> dict[str, Any]:
    payloads = build_and_write_step1g1_validation()
    summary = payloads["validation_summary"]
    clear_review_pack_dir()
    entries: list[dict[str, Any]] = []

    def add_entry(name: str, description: str, kind: str) -> Path:
        path = STEP1G1_REVIEW_PACK_DIR / name
        entries.append({"name": name, "kind": kind, "description": description, "path": str(path.resolve())})
        return path

    write_text(add_entry("00_REVIEW_INDEX.md", "G1 review starting point.", "markdown"), review_index_text(summary))
    write_text(add_entry("01_SCOPE_AND_RESTRICTIONS.md", "G1 scope guardrails.", "markdown"), scope_text())
    copy_text_file(STEP1G1_VALIDATION_SUMMARY_PATH, add_entry("02_STEP1G_VALIDATION_SUMMARY.json", "G1 validation summary.", "json"))
    copy_text_file(STEP1G1_VALIDATION_REPORT_PATH, add_entry("03_STEP1G_VALIDATION_REPORT.md", "G1 validation report.", "markdown"))
    copy_text_file(STEP1G1_FREEZE_CANDIDATE_MANIFEST_PATH, add_entry("04_FREEZE_CANDIDATE_MANIFEST.json", "Freeze-candidate manifest.", "json"))
    copy_text_file(STEP1G1_VISUAL_ISSUE_REGISTER_PATH, add_entry("05_VISUAL_ISSUE_REGISTER.json", "Visual QA issue register.", "json"))
    copy_text_file(STEP1G1_ROW_COUNT_AND_PROVENANCE_AUDIT_PATH, add_entry("06_ROW_COUNT_AND_PROVENANCE_AUDIT.json", "Row/provenance audit.", "json"))
    copy_text_file(STEP1G1_SAFETY_GUARDRAIL_AUDIT_PATH, add_entry("07_SAFETY_GUARDRAIL_AUDIT.json", "Safety guardrail audit.", "json"))
    copy_text_file(STEP1G1_GOLD_PROXY_VALIDATION_SUMMARY_PATH, add_entry("08_GOLD_PROXY_VALIDATION_SUMMARY.json", "Gold proxy validation summary.", "json"))
    copy_text_file(STEP1G1_FINAL_VISUAL_ROLE_STATE_COUNTS_PATH, add_entry("09_FINAL_ROLE_STATE_COUNTS.json", "Final role-state counts.", "json"))
    copy_binary_file(STEP1G1_VALIDATION_CONTACT_SHEET_PATH, add_entry("10_VALIDATION_CONTACT_SHEET.jpg", "G1 validation contact sheet.", "image"))
    copy_binary_file(STEP1G1_FINAL_ROLE_CROP_CONTACT_SHEET_PATH, add_entry("11_FINAL_ROLE_CROP_CONTACT_SHEET.jpg", "G1 final-role crop sheet.", "image"))
    copy_text_file(STEP1G1_FREEZE_REVIEW_DECISION_TEMPLATE_PATH, add_entry("12_FREEZE_REVIEW_DECISION_TEMPLATE.json", "Freeze review decision template.", "json"))
    code_files = [
        ("13_step1g_visual_reconstruction_validation.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "step1g_visual_reconstruction_validation.py", "G1 validation logic."),
        ("14_step1g_visual_reconstruction_render.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "step1g_visual_reconstruction_render.py", "G1 rendering logic."),
        ("15_step1g_freeze_candidate_pack.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "step1g_freeze_candidate_pack.py", "G1 freeze pack builder."),
        ("16_SCRIPT_VALIDATE_G1.py", SOCCERTRACK_ROOT / "scripts" / "step1g1_validate_visual_reconstruction_candidate.py", "G1 validation script."),
        ("17_SCRIPT_RENDER_G1.py", SOCCERTRACK_ROOT / "scripts" / "step1g1_render_visual_reconstruction_validation.py", "G1 render script."),
    ]
    for name, source, description in code_files:
        copy_text_file(source, add_entry(name, description, "python"))
    write_text(add_entry("18_TESTS_ADDED.md", "Summary of G1 tests.", "markdown"), tests_added_text())
    manifest_path = add_entry("19_REVIEW_PACK_MANIFEST.json", "G1 review pack manifest.", "json")
    manifest = {
        "artifact": "step1g1_review_pack_manifest",
        "created_at": utc_iso(),
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "no_auto_promotion": True,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "goalkeeper_slot_assignment_performed": False,
        "expected_22_role_states_created": False,
        "official_specialist_exclusion_performed": False,
        "exact_22_forcing_performed": False,
        "exact_two_goalkeeper_forcing_performed": False,
        "outputs": {
            "step1g1_validation_summary_path": str(STEP1G1_VALIDATION_SUMMARY_PATH.resolve()),
            "step1g1_validation_report_path": str(STEP1G1_VALIDATION_REPORT_PATH.resolve()),
            "step1g1_freeze_candidate_manifest_path": str(STEP1G1_FREEZE_CANDIDATE_MANIFEST_PATH.resolve()),
            "step1g1_visual_issue_register_path": str(STEP1G1_VISUAL_ISSUE_REGISTER_PATH.resolve()),
            "step1g1_safety_guardrail_audit_path": str(STEP1G1_SAFETY_GUARDRAIL_AUDIT_PATH.resolve()),
            "step1g1_review_pack_manifest_path": str(STEP1G1_REVIEW_PACK_MANIFEST_PATH.resolve()),
        },
        "summary": summary,
        "review_pack_file_count": len(entries),
        "review_pack_file_limit": 20,
        "review_pack_entries": entries,
    }
    write_json(manifest_path, manifest)
    write_json(STEP1G1_REVIEW_PACK_MANIFEST_PATH, manifest)
    if len(entries) > 20:
        raise RuntimeError(f"Step1.G1 review pack contains {len(entries)} files; maximum is 20.")
    return manifest


def print_step1g1_final_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    summary = manifest["summary"]
    print(f"step1g1_validation_summary_path: {outputs['step1g1_validation_summary_path']}")
    print(f"step1g1_validation_report_path: {outputs['step1g1_validation_report_path']}")
    print(f"step1g1_freeze_candidate_manifest_path: {outputs['step1g1_freeze_candidate_manifest_path']}")
    print(f"step1g1_visual_issue_register_path: {outputs['step1g1_visual_issue_register_path']}")
    print(f"step1g1_safety_guardrail_audit_path: {outputs['step1g1_safety_guardrail_audit_path']}")
    print(f"step1g1_review_pack_manifest_path: {outputs['step1g1_review_pack_manifest_path']}")
    print(f"b4_row_count: {summary.get('b4_row_count', 0)}")
    print(f"c2c_row_count: {summary.get('c2c_row_count', 0)}")
    print(f"d1c_row_count: {summary.get('d1c_row_count', 0)}")
    print(f"e1c_row_count: {summary.get('e1c_row_count', 0)}")
    print(f"f3_row_count: {summary.get('f3_row_count', 0)}")
    print(f"final_visual_role_state_counts: {summary.get('final_visual_role_state_counts', {})}")
    print(f"issue_register_counts_by_severity: {summary.get('issue_register_counts_by_severity', {})}")
    print(f"blocking_issue_count: {summary.get('blocking_issue_count', 0)}")
    print(f"f3_safe_for_step1g_validation_candidate={str(summary.get('f3_safe_for_step1g_validation_candidate', False)).lower()}")
    print(f"step1g1_freeze_candidate_created={str(summary.get('step1g1_freeze_candidate_created', False)).lower()}")
    print("step1g1_freeze_candidate_human_approved=false")
    print(f"step1g1_safe_for_step2_visual_continuity_candidate={str(summary.get('step1g1_safe_for_step2_visual_continuity_candidate', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("no_auto_promotion=true")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("goalkeeper_slot_assignment_performed=false")
    print("expected_22_role_states_created=false")
    print("official_specialist_exclusion_performed=false")
    print("exact_22_forcing_performed=false")
    print("exact_two_goalkeeper_forcing_performed=false")
