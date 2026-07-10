# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_intelligence.paths import CLIP_ID, MATCH_ID, SOCCERTRACK_ROOT, ensure_dir
from football_intelligence.step2_visual_continuity.io import (
    STEP2M1_CORRECTION_AUDIT_ROWS_PATH,
    STEP2M1_EDGE_CANDIDATE_ROWS_PATH,
    STEP2M1_EDGE_CANDIDATE_SAMPLE_PATH,
    STEP2M1_FREEZE_CANDIDATE_MANIFEST_PATH,
    STEP2M1_GROUP_ROWS_SANDBOX_PATH,
    STEP2M1_ISSUE_REGISTER_PATH,
    STEP2M1_NODE_ROWS_PATH,
    STEP2M1_REVIEW_CANDIDATE_ROWS_PATH,
    STEP2M1_REVIEW_CONTACT_SHEET_PATH,
    STEP2M1_REVIEW_DECISION_SUMMARY_PATH,
    STEP2M1_REVIEW_PACK_DIR,
    STEP2M1_REVIEW_PACK_MANIFEST_PATH,
    STEP2M1_REVIEW_PROGRESS_SUMMARY_PATH,
    STEP2M1_REVIEW_UI_HTML_PATH,
    STEP2M1_REVIEWED_DECISIONS_PATH,
    STEP2M1_SAFETY_GUARDRAIL_AUDIT_PATH,
    STEP2M1_VALIDATION_SUMMARY_PATH,
    copy_binary_file,
    copy_text_file,
    read_json,
    sample_payload,
    write_json,
    write_text,
)
from football_intelligence.step2_visual_continuity.schema import (
    VISUAL_ONLY_WARNING,
    guardrail_stamp,
    utc_iso,
)


def clear_review_pack_dir() -> None:
    ensure_dir(STEP2M1_REVIEW_PACK_DIR)
    for path in STEP2M1_REVIEW_PACK_DIR.iterdir():
        if path.is_file():
            path.unlink()


def scope_text() -> str:
    return "\n".join(
        [
            "# Step2.M1 Scope And Restrictions",
            "",
            f"- `{VISUAL_ONLY_WARNING}`",
            "- Visual-only short-window continuity candidates and sandbox groups.",
            "- Not identity tracking.",
            "- Not player-slot assignment.",
            "- Not goalkeeper-slot assignment.",
            "- Not metric, event, tactical, physical-performance, or football-conclusion analysis.",
            "- No exact-count forcing, no official/referee exclusion, and no bad-detection deletion.",
            "- `production_ready=false`, `no_auto_promotion=true`, `human_approved=false` by default.",
        ]
    ) + "\n"


def review_index_text(validation_summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step2.M1 Visual Continuity Review Pack",
            "",
            f"- Match: {MATCH_ID}",
            f"- Clip: {CLIP_ID}",
            f"- Warning: `{VISUAL_ONLY_WARNING}`",
            f"- Nodes: {validation_summary.get('node_row_count', 0)}",
            f"- Edge candidates: {validation_summary.get('edge_candidate_rows', 0)}",
            f"- Review cards: {validation_summary.get('review_candidate_rows', 0)}",
            f"- Freeze candidate created: {validation_summary.get('step2m1_visual_continuity_freeze_candidate_created', False)}",
            f"- Blocking issues: {validation_summary.get('blocking_issue_count', 0)}",
        ]
    ) + "\n"


def tests_added_text() -> str:
    return "\n".join(
        [
            "# Step2.M1 Tests Added",
            "",
            "- `tests/test_step2m1_visual_continuity_schema.py`: schema stamps and forbidden-key checks.",
            "- `tests/test_step2m1_nodes.py`: one node per F3 row, ID alignment, retained ambiguous/official/bad rows.",
            "- `tests/test_step2m1_edge_features.py`: visual-only edge feature scoring and uncertainty reasons.",
            "- `tests/test_step2m1_edge_candidates.py`: short-window edge IDs, gap cap, and edge guardrails.",
            "- `tests/test_step2m1_grouping.py`: sandbox groups are not identities, slots, goalkeeper slots, or metrics.",
            "- `tests/test_step2m1_review_selection.py`: review scope, hard cap, and safe auto-accept audit sample.",
            "- `tests/test_step2m1_review_validation.py`: human decision validation, high correction-rate gate, bulk safety.",
            "- `tests/test_step2m1_human_corrections.py`: corrections, audit coverage, training-example export.",
            "- `tests/test_step2m1_validation.py`: validation summary and freeze blocking behavior.",
            "- `tests/test_step2m1_restrictions.py`: no forbidden output keys in synthetic payloads.",
            "- `tests/test_step2m1_review_pack.py`: compact review pack file-count guard.",
        ]
    ) + "\n"


def optional_copy_text(source: Path, destination: Path, fallback: str) -> None:
    if source.exists() and source.is_file():
        copy_text_file(source, destination)
    else:
        write_text(destination, fallback)


def optional_copy_binary(source: Path, destination: Path) -> None:
    if source.exists() and source.is_file():
        copy_binary_file(source, destination)
    else:
        destination.write_bytes(b"")


def validate_review_pack_entries(entries: list[dict[str, Any]], limit: int = 20) -> None:
    if len(entries) > limit:
        raise RuntimeError(f"Step2.M1 review pack contains {len(entries)} files; maximum is {limit}.")


def validate_post_review_freshness(validation_summary: dict[str, Any], review_progress: dict[str, Any]) -> None:
    reviewed = int(review_progress.get("reviewed_candidates", 0) or 0)
    if reviewed <= 0:
        return
    mismatches = []
    for key in ["reviewed_candidates", "accepted_count", "rejected_count", "unsure_count", "correction_rate"]:
        if validation_summary.get(key) != review_progress.get(key):
            mismatches.append(key)
    if validation_summary.get("post_review_validation_refreshed") is not True:
        mismatches.append("post_review_validation_refreshed")
    if mismatches:
        raise RuntimeError(
            "Step2.M1 review pack refused stale validation summary; rerun "
            "scripts/step2m1_apply_visual_continuity_reviewed_decisions.py. "
            f"Mismatched fields: {sorted(set(mismatches))}"
        )


def validate_group_sample_consistency(validation_summary: dict[str, Any], group_sample: dict[str, Any]) -> None:
    expected = int(validation_summary.get("visual_continuity_group_rows", 0) or 0)
    observed = int(group_sample.get("total_rows", 0) or 0)
    if expected != observed:
        raise RuntimeError(
            "Step2.M1 group sample row count does not match validation summary: "
            f"validation={expected}, group_sample={observed}"
        )


def manifest_payload(entries: list[dict[str, Any]], validation_summary: dict[str, Any]) -> dict[str, Any]:
    validate_review_pack_entries(entries)
    return guardrail_stamp(
        {
            "artifact": "step2m1_visual_continuity_review_pack_manifest",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "review_pack_file_count": len(entries),
            "review_pack_file_limit": 20,
            "reviewed_candidates": validation_summary.get("reviewed_candidates", 0),
            "accepted_count": validation_summary.get("accepted_count", 0),
            "rejected_count": validation_summary.get("rejected_count", 0),
            "unsure_count": validation_summary.get("unsure_count", 0),
            "correction_rate": validation_summary.get("correction_rate", 0.0),
            "corrected_edge_rows_available": validation_summary.get("corrected_edge_rows_available", False),
            "post_review_validation_refreshed": validation_summary.get("post_review_validation_refreshed", False),
            "visual_continuity_group_rows": validation_summary.get("visual_continuity_group_rows", 0),
            "summary": validation_summary,
            "review_pack_entries": entries,
        }
    )


def build_step2m1_review_pack() -> dict[str, Any]:
    clear_review_pack_dir()
    validation_summary = read_json(STEP2M1_VALIDATION_SUMMARY_PATH) if STEP2M1_VALIDATION_SUMMARY_PATH.exists() else {}
    review_progress = read_json(STEP2M1_REVIEW_PROGRESS_SUMMARY_PATH) if STEP2M1_REVIEW_PROGRESS_SUMMARY_PATH.exists() else {}
    validate_post_review_freshness(validation_summary, review_progress)
    entries: list[dict[str, Any]] = []

    def add_entry(name: str, description: str, kind: str) -> Path:
        path = STEP2M1_REVIEW_PACK_DIR / name
        entries.append({"name": name, "kind": kind, "description": description, "path": str(path.resolve())})
        return path

    write_text(add_entry("00_REVIEW_INDEX.md", "Step2.M1 review starting point.", "markdown"), review_index_text(validation_summary))
    write_text(add_entry("01_SCOPE_AND_RESTRICTIONS.md", "Step2.M1 guardrails.", "markdown"), scope_text())
    write_json(add_entry("02_VALIDATION_SUMMARY.json", "Validation summary.", "json"), validation_summary)
    write_json(add_entry("03_SAFETY_GUARDRAIL_AUDIT.json", "Safety guardrail audit.", "json"), read_json(STEP2M1_SAFETY_GUARDRAIL_AUDIT_PATH) if STEP2M1_SAFETY_GUARDRAIL_AUDIT_PATH.exists() else {})
    write_json(add_entry("04_ISSUE_REGISTER.json", "Issue register.", "json"), read_json(STEP2M1_ISSUE_REGISTER_PATH) if STEP2M1_ISSUE_REGISTER_PATH.exists() else {})
    write_json(add_entry("05_FREEZE_CANDIDATE_MANIFEST.json", "Freeze candidate manifest.", "json"), read_json(STEP2M1_FREEZE_CANDIDATE_MANIFEST_PATH) if STEP2M1_FREEZE_CANDIDATE_MANIFEST_PATH.exists() else {})
    write_json(add_entry("06_REVIEW_PROGRESS_SUMMARY.json", "Review progress summary.", "json"), review_progress)
    write_json(add_entry("07_REVIEW_DECISION_SUMMARY.json", "Review decision summary.", "json"), read_json(STEP2M1_REVIEW_DECISION_SUMMARY_PATH) if STEP2M1_REVIEW_DECISION_SUMMARY_PATH.exists() else {})
    write_json(add_entry("08_NODE_SAMPLE.json", "Node-row sample.", "json"), sample_payload(STEP2M1_NODE_ROWS_PATH, 40, "step2m1_node_sample"))
    edge_sample_path = STEP2M1_EDGE_CANDIDATE_SAMPLE_PATH if STEP2M1_EDGE_CANDIDATE_SAMPLE_PATH.exists() else STEP2M1_EDGE_CANDIDATE_ROWS_PATH
    write_json(add_entry("09_EDGE_SAMPLE.json", "Edge-row sample.", "json"), sample_payload(edge_sample_path, 40, "step2m1_edge_sample"))
    group_sample = sample_payload(STEP2M1_GROUP_ROWS_SANDBOX_PATH, 40, "step2m1_group_sample")
    validate_group_sample_consistency(validation_summary, group_sample)
    write_json(add_entry("10_GROUP_SAMPLE.json", "Group-row sample.", "json"), group_sample)
    write_json(add_entry("11_REVIEW_CANDIDATE_SAMPLE.json", "Review-candidate sample.", "json"), sample_payload(STEP2M1_REVIEW_CANDIDATE_ROWS_PATH, 80, "step2m1_review_candidate_sample"))
    write_json(add_entry("12_REVIEWED_DECISIONS_SAMPLE.json", "Reviewed decision sample or template.", "json"), sample_payload(STEP2M1_REVIEWED_DECISIONS_PATH, 80, "step2m1_reviewed_decision_sample"))
    write_json(add_entry("13_CORRECTION_AUDIT_SAMPLE.json", "Correction audit sample.", "json"), sample_payload(STEP2M1_CORRECTION_AUDIT_ROWS_PATH, 80, "step2m1_correction_audit_sample"))
    optional_copy_text(STEP2M1_REVIEW_UI_HTML_PATH, add_entry("14_REVIEW_UI_HTML_COPY.html", "Review UI HTML copy.", "html"), "<!-- Step2.M1 review UI has not been rendered yet. -->\n")
    optional_copy_binary(STEP2M1_REVIEW_CONTACT_SHEET_PATH, add_entry("15_REVIEW_CONTACT_SHEET.jpg", "Review contact sheet.", "image"))
    write_text(add_entry("16_CODE_ENTRYPOINTS.md", "Step2.M1 scripts and modules.", "markdown"), code_entrypoints_text())
    write_text(add_entry("17_TESTS_ADDED.md", "Synthetic tests added.", "markdown"), tests_added_text())
    manifest_path = add_entry("18_REVIEW_PACK_MANIFEST.json", "Review pack manifest.", "json")
    manifest = manifest_payload(entries, validation_summary)
    write_json(manifest_path, manifest)
    write_json(STEP2M1_REVIEW_PACK_MANIFEST_PATH, manifest)
    return manifest


def code_entrypoints_text() -> str:
    files = [
        "scripts/step2m1_build_visual_continuity_sandbox.py",
        "scripts/step2m1_render_visual_continuity_review.py",
        "scripts/step2m1_launch_visual_continuity_review_ui.py",
        "scripts/step2m1_validate_visual_continuity_review.py",
        "scripts/step2m1_apply_visual_continuity_reviewed_decisions.py",
        "scripts/step2m1_build_visual_continuity_review_pack.py",
        "src/football_intelligence/step2_visual_continuity/",
    ]
    return "\n".join(["# Step2.M1 Code Entrypoints", ""] + [f"- `{SOCCERTRACK_ROOT / item}`" for item in files]) + "\n"


def print_step2m1_review_pack_console(manifest: dict[str, Any]) -> None:
    print(f"step2m1_visual_continuity_review_pack_manifest_path: {STEP2M1_REVIEW_PACK_MANIFEST_PATH.resolve()}")
    print(f"review_pack_file_count: {manifest.get('review_pack_file_count', 0)}")
    print("review_pack_file_limit: 20")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
