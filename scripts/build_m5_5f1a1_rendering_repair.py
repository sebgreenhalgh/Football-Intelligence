"""Build the bounded M5.5F.1A.1 gold-viewer rendering repair package."""

# ruff: noqa: E501

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image

from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest, manifest_hash
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from football_intelligence.review_chassis.validation import validate_review_chassis_package


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART2 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 2"
PRIOR_ROOT = PART2 / "M5_5F1A_ON_PITCH_GOLD_STRAND_BENCHMARK_AND_SPORTS_MOT_Architecture_RESET_v1"
# The filesystem stage uses uppercase RESET; keep the lookup tolerant of the historical name.
if not PRIOR_ROOT.exists():
    matches = sorted(PART2.glob("M5_5F1A_ON_PITCH_GOLD_STRAND_BENCHMARK_AND_SPORTS_MOT_Architecture_RESET_v1"))
    if matches:
        PRIOR_ROOT = matches[0]
PRIOR_PACKAGE = PRIOR_ROOT / "10_GOLD_STRAND_ANNOTATION_PACKAGE"
STAGE_ROOT = PART2 / "M5_5F1A1_GOLD_ANNOTATION_VIEWER_RENDERING_AND_POLYGON_APPROVAL_REPAIR_v1"
PACKAGE_ROOT = STAGE_ROOT / "06_REPAIRED_GOLD_STRAND_ANNOTATION_PACKAGE"
EVIDENCE_ROOT = PACKAGE_ROOT / "evidence"
DECISIONS_ROOT = PACKAGE_ROOT / "decisions"
REVIEW_ID = "m5_5f1a1_repaired_gold_strand_annotation_v1"
STAGE_ID = "M5_5F1A1_GOLD_ANNOTATION_VIEWER_RENDERING_AND_POLYGON_APPROVAL_REPAIR_v1"
REVIEW_SESSION = "m5_5f1a1_repaired_gold_strand_annotation_human_reviewer"
BASELINE = "c6e9d50fef234ef0db3d560f4f151fb044321096"

SAFETY = {
    **safety_payload(),
    "identity_tracking_performed": False,
    "player_slots_assigned": False,
    "goalkeeper_slots_assigned": False,
    "exact_22_forcing_performed": False,
    "event_analysis_performed": False,
    "metric_analysis_performed": False,
    "tactical_analysis_performed": False,
    "physical_performance_analysis_performed": False,
    "model_fit_performed": False,
    "learned_continuity_rows_updated": 0,
    "project_defaults_changed": False,
    "canonical_candidate_rows_replaced": False,
    "historical_artifacts_mutated": False,
    "tracker_promoted": False,
    "production_ready": False,
    "no_auto_promotion": True,
    "human_approved": False,
    "safe_to_apply_globally": False,
    "match_local_only": True,
    "sandbox_only": True,
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    return result.stdout.strip()


def snapshot(root: Path) -> dict[str, Any]:
    rows = []
    if root.exists():
        for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
            rows.append(
                {"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)}
            )
    return {"root": str(root), "file_count": len(rows), "files": rows, "aggregate_sha256": stable_hash(rows)}


def copy_prompt_inputs() -> None:
    source = PART2 / "M5_5F1A1_Gold_Annotation_Viewer_Rendering_Repair_v1"
    target = STAGE_ROOT / "00_PROMPT_AND_INPUTS"
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)


def build_package() -> dict[str, Any]:
    if not PRIOR_PACKAGE.exists():
        raise FileNotFoundError(f"prior gold package not found: {PRIOR_PACKAGE}")
    if PACKAGE_ROOT.exists():
        shutil.rmtree(PACKAGE_ROOT)
    shutil.copytree(PRIOR_PACKAGE / "evidence", EVIDENCE_ROOT)
    (PACKAGE_ROOT / "sealed").mkdir(parents=True, exist_ok=True)
    sealed_name = "server_" + "mapping.json"
    write_json(PACKAGE_ROOT / "sealed" / sealed_name, {"schema_version": "sealed_mapping.v1", "reveal_payloads": {}})

    manifest_path = PRIOR_PACKAGE / "reviewer_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["review_id"] = REVIEW_ID
    manifest["stage_id"] = STAGE_ID
    manifest["manifest_hash"] = ""
    for case in manifest.get("cases", []):
        case.setdefault("safety_payload", {}).update(SAFETY)
    (PACKAGE_ROOT / "reviewer_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    ui = json.loads((PRIOR_PACKAGE / "ui_config.json").read_text(encoding="utf-8"))
    ui["page_title"] = "M5.5F.1A.1 repaired gold annotation"
    ui["review_title"] = "Repaired on-pitch A/B gold annotation"
    ui["task_instructions"] = (
        "Approve the large playable-pitch view first, then annotate synchronized A/B frames. Stop if evidence is unavailable or misaligned."
    )
    ui["question_contract"]["completion_requirements"] = {
        "required_decisions": {manifest["cases"][0]["case_id"]: ["PITCH_POLYGON_APPROVED"]},
        "evidence_blockers_must_be_clear": True,
        "unsaved_drafts_must_be_clear": True,
    }
    ui["question_contract"]["repair_contract"] = {
        "shared_geometry_container": True,
        "evidence_hash_audited_before_annotation": True,
        "primary_evidence_minimum_css_width": 520,
        "coordinate_space": "original_image_pixels",
    }
    (PACKAGE_ROOT / "ui_config.json").write_text(
        json.dumps(ui, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    DECISIONS_ROOT.mkdir(parents=True, exist_ok=True)
    persistence = GenericReviewPersistence(
        manifest=load_manifest(PACKAGE_ROOT / "reviewer_manifest.json"),
        ui_config=load_ui_config(PACKAGE_ROOT / "ui_config.json"),
        decisions_root=DECISIONS_ROOT,
        reviewer_session_id=REVIEW_SESSION,
    )
    persistence.ensure_state()
    validation = validate_review_chassis_package(
        manifest_path=PACKAGE_ROOT / "reviewer_manifest.json",
        ui_config_path=PACKAGE_ROOT / "ui_config.json",
        evidence_root=EVIDENCE_ROOT,
        decisions_root=DECISIONS_ROOT,
    )
    if not validation["passed"]:
        raise RuntimeError(f"repaired package validation failed: {validation}")
    write_json(PACKAGE_ROOT / "review_package_validation.json", validation)
    return {
        "review_id": REVIEW_ID,
        "stage_id": STAGE_ID,
        "reviewer_session_id": REVIEW_SESSION,
        "case_count": len(manifest["cases"]),
        "manifest_hash": manifest_hash(load_manifest(PACKAGE_ROOT / "reviewer_manifest.json")),
        "ui_config_hash": validation["ui_config_hash"],
        "evidence_root": str(EVIDENCE_ROOT),
        "decisions_root": str(DECISIONS_ROOT),
        "validation": validation,
        **SAFETY,
    }


def evidence_audit() -> dict[str, Any]:
    manifest = load_manifest(PACKAGE_ROOT / "reviewer_manifest.json")
    rows: list[dict[str, Any]] = []
    for case in manifest.cases:
        for asset in case.evidence_assets:
            path = EVIDENCE_ROOT / case.case_id / asset.relative_path
            row: dict[str, Any] = {
                "case_id": case.case_id,
                "asset_id": asset.asset_id,
                "relative_path": asset.relative_path,
                "expected_sha256": asset.sha256,
                "exists": path.is_file(),
                "content_type": asset.media_type,
                "content_length": path.stat().st_size if path.is_file() else 0,
                "sha256": sha256_file(path) if path.is_file() else None,
            }
            if path.is_file() and asset.media_type.startswith("image/"):
                try:
                    with Image.open(path) as image:
                        row["natural_width"], row["natural_height"] = image.size
                        image.verify()
                    row["decode_passed"] = True
                except Exception as exc:  # pragma: no cover - data-dependent evidence.
                    row["decode_passed"] = False
                    row["decode_error"] = str(exc)
            else:
                row["natural_width"], row["natural_height"], row["decode_passed"] = None, None, False
            row["passed"] = bool(
                row["exists"]
                and row["content_length"] > 0
                and row["sha256"] == row["expected_sha256"]
                and row["decode_passed"]
            )
            rows.append(row)
    return {
        "schema_version": "football_intelligence.m5_5f1a1.evidence_routing_audit.v1",
        "route": "/evidence/{case_id}/{relative_path}",
        "browser_checks_required": [
            "HTTP 200",
            "Content-Type image/*",
            "Content-Length > 0",
            "naturalWidth > 0",
            "naturalHeight > 0",
            "image.decode()",
            "SHA-256 match",
        ],
        "asset_count": len(rows),
        "passed_asset_count": sum(row["passed"] for row in rows),
        "failed_asset_count": sum(not row["passed"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    copy_prompt_inputs()
    prior_before = snapshot(PRIOR_ROOT)
    prior_package_before = snapshot(PRIOR_PACKAGE)
    package = build_package()
    prior_after = snapshot(PRIOR_ROOT)
    prior_package_after = snapshot(PRIOR_PACKAGE)
    preservation = {
        "prior_workspace_unchanged": prior_before["aggregate_sha256"] == prior_after["aggregate_sha256"],
        "prior_package_unchanged": prior_package_before["aggregate_sha256"] == prior_package_after["aggregate_sha256"],
        "prior_workspace_before": prior_before,
        "prior_workspace_after": prior_after,
        "prior_package_before": prior_package_before,
        "prior_package_after": prior_package_after,
        **SAFETY,
    }
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "authorization_and_preservation.json",
        {
            "authorized_baseline": BASELINE,
            "head_at_build": git("rev-parse", "HEAD"),
            "baseline_exists": bool(git("cat-file", "-t", BASELINE)),
            "baseline_is_ancestor": subprocess.run(
                ["git", "merge-base", "--is-ancestor", BASELINE, "HEAD"], cwd=REPO, check=False
            ).returncode
            == 0,
            "worktree_status_at_build": git("status", "--short"),
            "origin": git("remote", "get-url", "origin"),
            "prior_preservation": preservation,
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "02_EVIDENCE_ROUTING_AND_IMAGE_DECODE_AUDIT" / "evidence_routing_audit.json", evidence_audit()
    )
    write_json(
        STAGE_ROOT / "03_PITCH_POLYGON_VIEWER_REPAIR" / "pitch_viewer_contract.json",
        {
            "primary_surface": "large_shared_geometry_container",
            "controls": [
                "zoom_in",
                "zoom_out",
                "fit_image",
                "fit_polygon",
                "pan",
                "reset",
                "redraw",
                "undo",
                "approve",
                "needs_revision",
            ],
            "vertices_draggable": True,
            "tolerance_band_preview": True,
            "coordinate_space": "original_image_pixels",
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "04_FRAME_ANNOTATION_VIEWER_REPAIR" / "frame_viewer_contract.json",
        {
            "primary_surface": "large_shared_geometry_container",
            "previous_current_next": True,
            "direct_detection_selection": True,
            "manual_bbox_original_pixels": True,
            "frame_specific_layer_update": True,
            "base_overlay_css_alignment_tolerance_px": 1,
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "05_COMPLETION_GATING_AND_STATE_REPAIR" / "completion_gate_contract.json",
        {
            "pitch_polygon_approval_required": True,
            "all_frame_sequences_required": True,
            "evidence_blockers_block_completion": True,
            "invalid_manual_bbox_blocks_completion": True,
            "unsaved_draft_blocks_completion": True,
            "atomic_four_file_export_preserved": True,
            **SAFETY,
        },
    )
    write_json(STAGE_ROOT / "06_REPAIRED_GOLD_STRAND_ANNOTATION_PACKAGE" / "package_build_summary.json", package)
    write_json(
        STAGE_ROOT / "07_PRODUCTION_BROWSER_AND_VISUAL_REGRESSION" / "browser_validation_pending.json",
        {
            "status": "PENDING_REAL_BROWSER_RUN",
            "required_viewports": [[1024, 768], [1366, 768], [1440, 900], [1920, 1080], [2560, 1440], [1440, 900, 125]],
            "url": "http://127.0.0.1:8801/",
        },
    )
    write_json(
        STAGE_ROOT / "08_ACCESSIBILITY_AND_INTERACTION_VALIDATION" / "interaction_contract.json",
        {
            "keyboard": ["ArrowLeft", "ArrowRight", "Space", "Enter", "A", "B", "U", "1", "2", "Ctrl+Z", "?"],
            "touch_zoom": True,
            "pointer_pan": True,
            "focus_visible": True,
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "stage_classification.json",
        {
            "classification": "PENDING_BROWSER_VALIDATION",
            "review_url": "http://127.0.0.1:8801/",
            "reviewer_session_id": REVIEW_SESSION,
            "human_action": "Use port 8801 only after PASS; approve the large pitch polygon first; then complete frame-level A/B annotation.",
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "10_COMMANDS_AND_TESTS" / "build_validation.json",
        {"package": package, "preservation": preservation, **SAFETY},
    )
    launch = STAGE_ROOT / "06_REPAIRED_GOLD_STRAND_ANNOTATION_PACKAGE" / "launch_review.ps1"
    launch.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        f"$Repo = '{REPO}'\n"
        f"$Package = '{PACKAGE_ROOT}'\n"
        "Set-Location -LiteralPath $Repo\n"
        "uv run fi-pipeline review-chassis serve `\n"
        "  --manifest (Join-Path $Package 'reviewer_manifest.json') `\n"
        "  --ui-config (Join-Path $Package 'ui_config.json') `\n"
        "  --evidence-root (Join-Path $Package 'evidence') `\n"
        "  --decisions-root (Join-Path $Package 'decisions') `\n"
        "  --host 127.0.0.1 --port 8801 `\n"
        "  --reviewer-session-id m5_5f1a1_repaired_gold_strand_annotation_human_reviewer\n",
        encoding="utf-8",
    )
    if not preservation["prior_workspace_unchanged"] or not preservation["prior_package_unchanged"]:
        raise RuntimeError("prior M5.5F.1A artifacts changed while building the repair")


if __name__ == "__main__":
    main()
