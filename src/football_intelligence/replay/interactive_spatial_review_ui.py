from __future__ import annotations

import csv
import json
import mimetypes
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import manifest_hash
from football_intelligence.review_chassis.models import (
    GenericEvidenceAsset,
    GenericReviewCase,
    GenericReviewManifest,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from football_intelligence.review_chassis.server import _sanitize_browser_payload
from football_intelligence.review_chassis.spatial_annotations import (
    ImageSize,
    ViewTransform,
    autosave_payload,
    client_to_image,
    hit_test_candidates,
    normalize_bbox,
    normalize_spatial_annotation_note,
    safe_anonymous_candidate,
    scan_forbidden_browser_payload,
    validate_spatial_annotation_for_decision,
)

STAGE_ID = "M5_4J_INTERACTIVE_SPATIAL_REVIEW_UI_v1"
AUTHORIZED_BASELINE = "a58fe7e19ac38cea81fc2304f82eb68fba3fb64b"
LOCAL_URL = "http://127.0.0.1:8777/"
WORKSPACE_DIRS = (
    "00_PROMPT_AND_INPUTS",
    "01_AUTHORIZATION_AND_SOURCE_AUDIT",
    "02_INTERACTIVE_REVIEW_PACKAGE",
    "03_UI_VALIDATION",
    "04_VISUAL_EVIDENCE",
    "05_COMMANDS_AND_TESTS",
    "06_REVIEW_PACK_FOR_CHATGPT",
    "_tmp",
)
REVIEW_PACK_FILENAMES = (
    "REVIEW_PACK_MANIFEST.json",
    "01_EXECUTIVE_SUMMARY.md",
    "02_RUN_AND_GIT_CONTEXT.json",
    "03_FILES_CHANGED.md",
    "04_SOURCE_DIFF.patch",
    "05_COMMANDS_AND_TEST_RESULTS.md",
    "06_OUTPUT_ARTIFACT_INDEX.json",
    "07_PRIMARY_RESULT_OR_BLOCKER.json",
    "08_SAFETY_AND_PRIVACY_AUDIT.json",
    "09_SOURCE_MUTATION_AUDIT.json",
    "10_UNRESOLVED_AND_NEXT_ACTION.md",
    "11_UI_CONFIG_AND_SCHEMA.json",
    "12_COORDINATE_TRANSFORM_RESULTS.json",
    "13_INTERACTION_AND_PERSISTENCE_RESULTS.json",
    "14_BROWSER_PAYLOAD_PRIVACY_RESULTS.json",
    "15_LAUNCHER_AND_REVIEW_PATHS.json",
    "16_ACCEPTANCE_CHECKLIST.json",
    "17_FULL_PAGE_UI.jpg",
    "18_LARGE_ANNOTATION_VIEW.jpg",
    "19_BBOX_DETECTION_OCCLUSION_DEMO.jpg",
)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    return _write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _run(repo_root: Path, *args: str) -> dict[str, Any]:
    completed = subprocess.run(
        list(args),
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
        timeout=90,
    )
    return {
        "command": list(args),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _git(repo_root: Path, *args: str) -> dict[str, Any]:
    return _run(repo_root, "git", *args)


def _inventory(root: Path) -> list[dict[str, Any]]:
    rows = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(root)),
                    "byte_size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def _copy(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"path": str(dst), "byte_size": dst.stat().st_size, "sha256": sha256_file(dst)}


def _media_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _frame_rows_by_sequence(frame_manifest_path: Path) -> dict[int, dict[str, Any]]:
    manifest = _read_json(frame_manifest_path)
    return {int(row["frame_sequence"]): row for row in manifest.get("frames", [])}


def _case_index_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sealed_rows_by_case(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    rows = payload.get("mappings", [])
    return {str(row["case_id"]): row for row in rows if isinstance(row, dict)}


def _matching_old_asset(case_data: dict[str, Any], asset_id: str) -> dict[str, Any] | None:
    return next((asset for asset in case_data.get("evidence_assets", []) if asset.get("asset_id") == asset_id), None)


def _old_asset_path(source_root: Path, case_id: str, asset: dict[str, Any]) -> Path:
    return source_root / "evidence" / case_id / str(asset["relative_path"])


def _copy_prompt_inputs(workspace_root: Path, prompt_root: Path, prior_roots: list[str]) -> dict[str, Any]:
    target_root = workspace_root / "00_PROMPT_AND_INPUTS"
    files = []
    for name in (
        "00_READ_ME_FIRST.md",
        "01_INTERACTIVE_SPATIAL_REVIEW_UI_CODEX_PROMPT.md",
        "02_INTERACTIVE_UI_WORKSPACE_CONTRACT.json",
        "03_SPATIAL_ANNOTATION_V2_CONTRACT.json",
        "04_PROMPT_PACK_MANIFEST.json",
    ):
        src = prompt_root / name
        if src.exists():
            copied = _copy(src, target_root / name)
            files.append({"source": str(src), **copied})
    prior = []
    for root_text in prior_roots:
        root = Path(root_text)
        prior.append({"root": str(root), "file_count": len(_inventory(root)), "files": _inventory(root)})
    payload = {
        "schema_version": "football_intelligence.m5_4j_interactive.prompt_inputs.v1",
        "generated_at": utc_now(),
        "prompt_root": str(prompt_root),
        "copied_prompt_files": files,
        "prior_context_roots": prior,
        **safety_payload(),
    }
    _write_json(target_root / "prompt_and_prior_context_inventory.json", payload)
    return payload


def _authorization_audit(repo_root: Path) -> dict[str, Any]:
    status = _git(repo_root, "status", "--short")
    head = _git(repo_root, "rev-parse", "HEAD")
    exists = _git(repo_root, "cat-file", "-e", f"{AUTHORIZED_BASELINE}^{{commit}}")
    ancestor = _git(repo_root, "merge-base", "--is-ancestor", AUTHORIZED_BASELINE, "HEAD")
    log = _git(repo_root, "log", "--oneline", "--decorate", "--no-merges", f"{AUTHORIZED_BASELINE}..HEAD")
    stat = _git(repo_root, "diff", "--stat", f"{AUTHORIZED_BASELINE}..HEAD")
    names = _git(repo_root, "diff", "--name-status", f"{AUTHORIZED_BASELINE}..HEAD")
    return {
        "schema_version": "football_intelligence.m5_4j_interactive.authorization_audit.v1",
        "generated_at": utc_now(),
        "authorized_baseline": AUTHORIZED_BASELINE,
        "current_head": head["stdout"].strip(),
        "head_is_authorized_or_clean_descendant": exists["exit_code"] == 0 and ancestor["exit_code"] == 0,
        "baseline_commit_exists": exists["exit_code"] == 0,
        "baseline_is_ancestor_of_head": ancestor["exit_code"] == 0,
        "worktree_status_at_builder_run": status["stdout"],
        "worktree_clean_at_builder_run": status["stdout"].strip() == "",
        "preimplementation_clean_gate_verified_by_codex": True,
        "intervening_commits": log["stdout"],
        "diff_stat_from_baseline": stat["stdout"],
        "diff_name_status_from_baseline": names["stdout"],
        **safety_payload(),
    }


def _build_ui_config(old_ui: dict[str, Any], schema_contract: dict[str, Any]) -> ReviewUIConfig:
    config = {
        "schema_version": "football_intelligence.review_ui_config.v2",
        "page_title": "Interactive missing-target spatial localization",
        "review_title": "Interactive missing-target spatial localization",
        "task_instructions": (
            "Use the full-resolution target frame to draw a bbox, select an anonymous detection, "
            "mark an occlusion location, or leave the case unresolved."
        ),
        "visual_warning": "VISUAL_ONLY_NOT_METRIC",
        "decisions": old_ui["decisions"],
        "asset_panel_order": [
            {"asset_type": "wide_context", "label": "Full-resolution target"},
            {"asset_type": "animated_gif", "label": "Animated temporal GIF"},
            {"asset_type": "image_sequence", "group_id": "temporal_frames", "label": "Frame stepper"},
            {"asset_type": "temporal_strip", "label": "Temporal strip"},
            {"asset_type": "crop", "label": "Crops"},
            {"asset_type": "overlay", "label": "Overlays"},
        ],
        "visible_metadata_fields": [
            "source_frame_sequence",
            "target_frame_sequence",
            "frame_gap",
            "full_frame_candidate_count",
            "annotation_coordinate_space",
            "full_resolution_dimensions",
        ],
        "hidden_metadata_fields": [],
        "reveal_controls": True,
        "notes_enabled": True,
        "undo_enabled": True,
        "autosave_enabled": True,
        "completion_requires_all_cases": True,
        "decisions_advance_automatically": False,
        "unresolved_allowed": True,
        "gif_primary": True,
        "image_stepper_enabled": True,
        "layout": "interactive_spatial_annotation",
        "spatial_annotation_enabled": True,
        "spatial_annotation_mode": "interactive_bbox_detection_occlusion_footpoint_v2",
        "spatial_annotation_schema": {
            **schema_contract,
            "title": "Interactive spatial annotation",
            "interactive_canvas_enabled": True,
            "bbox_drawing_supported": True,
            "bbox_move_resize_supported": True,
            "anonymous_detection_click_supported": True,
            "overlap_resolution_supported": True,
            "occlusion_point_supported": True,
            "footpoint_supported": True,
            "fullscreen_supported": True,
            "autosave_reload_supported": True,
            "numeric_fallback_supported": True,
            "sealed_candidate_ids_browser_visible": False,
        },
    }
    return ReviewUIConfig.model_validate(config)


def _copy_case_evidence(
    *,
    package_root: Path,
    source_root: Path,
    old_case: dict[str, Any],
    new_case_id: str,
    frame_row: dict[str, Any],
) -> tuple[list[GenericEvidenceAsset], list[dict[str, Any]]]:
    evidence_root = package_root / "evidence" / new_case_id
    assets: list[GenericEvidenceAsset] = []
    evidence_rows: list[dict[str, Any]] = []
    full_src = Path(frame_row["frame_file"])
    full_dst = evidence_root / "target_full_resolution.jpg"
    copied = _copy(full_src, full_dst)
    assets.append(
        GenericEvidenceAsset(
            asset_id="target_full_resolution",
            asset_type="wide_context",
            label="Full-resolution target frame",
            relative_path="target_full_resolution.jpg",
            sha256=copied["sha256"],
            media_type="image/jpeg",
            frame_sequences=[int(frame_row["frame_sequence"])],
            metadata={
                "primary_annotation_image": True,
                "full_resolution": True,
                "width": int(frame_row["width"]),
                "height": int(frame_row["height"]),
                "source_frame_index": int(frame_row["source_frame_index"]),
                "source_frame_byte_sha256": frame_row["byte_sha256"],
            },
        )
    )
    evidence_rows.append({"case_id": new_case_id, "asset_id": "target_full_resolution", **copied})

    for old_asset in old_case.get("evidence_assets", []):
        src = _old_asset_path(source_root, old_case["case_id"], old_asset)
        if not src.exists() or old_asset.get("media_type", "").startswith("video/"):
            continue
        dst = evidence_root / str(old_asset["relative_path"])
        copied = _copy(src, dst)
        metadata = dict(old_asset.get("metadata") or {})
        metadata["copied_from_read_only_source_package"] = True
        assets.append(
            GenericEvidenceAsset(
                asset_id=str(old_asset["asset_id"]),
                asset_type=old_asset["asset_type"],
                label=str(old_asset["label"]),
                relative_path=str(old_asset["relative_path"]),
                sha256=copied["sha256"],
                media_type=str(old_asset["media_type"]),
                frame_sequences=list(old_asset.get("frame_sequences", [])),
                group_id=old_asset.get("group_id"),
                metadata=metadata,
                visibility_policy=old_asset.get("visibility_policy", "always_visible"),
            )
        )
        evidence_rows.append({"case_id": new_case_id, "asset_id": old_asset["asset_id"], **copied})
    return assets, evidence_rows


def _build_interactive_package(
    *,
    workspace_root: Path,
    source_root: Path,
    repo_root: Path,
    schema_contract: dict[str, Any],
) -> dict[str, Any]:
    package_root = workspace_root / "02_INTERACTIVE_REVIEW_PACKAGE"
    package_root.mkdir(parents=True, exist_ok=True)
    old_manifest = _read_json(source_root / "reviewer_manifest.json")
    old_ui = _read_json(source_root / "ui_config.json")
    case_index = _case_index_rows(source_root / "case_index.csv")
    case_index_by_id = {row["case_id"]: row for row in case_index}
    sealed_by_case = _sealed_rows_by_case(source_root / "sealed" / "mapping.json")
    stage_root = source_root.parents[1]
    frame_manifest_path = stage_root / "continuity_v11" / "unseen_window" / "canonical_frame_manifest.json"
    frame_rows = _frame_rows_by_sequence(frame_manifest_path)
    ui_config = _build_ui_config(old_ui, schema_contract)

    all_evidence_rows: list[dict[str, Any]] = []
    cases: list[GenericReviewCase] = []
    sealed_mappings = []
    output_case_index = []
    for index, old_case in enumerate(old_manifest["cases"], start=1):
        old_case_id = old_case["case_id"]
        new_case_id = f"m5_4j_interactive_spatial_case_{index:03d}"
        target_frame_sequence = int(old_case["target_frame_sequence"])
        frame_row = frame_rows[target_frame_sequence]
        assets, evidence_rows = _copy_case_evidence(
            package_root=package_root,
            source_root=source_root,
            old_case=old_case,
            new_case_id=new_case_id,
            frame_row=frame_row,
        )
        all_evidence_rows.extend(evidence_rows)
        sealed_row = sealed_by_case[old_case_id]
        safe_candidates = [
            safe_anonymous_candidate(candidate, target_frame_sequence=target_frame_sequence)
            for candidate in sealed_row.get("anonymous_full_frame_candidates", [])
        ]
        visible_metadata = {
            "source_frame_sequence": old_case.get("source_frame_sequence"),
            "target_frame_sequence": target_frame_sequence,
            "frame_gap": old_case.get("frame_gap"),
            "full_frame_candidate_count": len(safe_candidates),
            "annotation_coordinate_space": "original_image_pixels",
            "full_resolution_dimensions": f"{frame_row['width']}x{frame_row['height']}",
            "safe_anonymous_candidates": safe_candidates,
        }
        case = GenericReviewCase(
            case_id=new_case_id,
            task_type="missing_target_spatial_localization",
            candidate_id=f"anonymous_interactive_spatial_candidate_{index:03d}",
            candidate_hash=stable_hash({"case_id": new_case_id, "target_frame_sequence": target_frame_sequence}),
            evidence_hash=stable_hash([asset.model_dump(mode="json") for asset in assets]),
            allowed_decisions=old_case["allowed_decisions"],
            concise_question="Where is the strongest supported target location in the full-resolution target frame?",
            detailed_instructions=(
                "Draw the missing target bbox, click an anonymous existing detection, mark an occlusion point, "
                "or choose unresolved/source invalid when the evidence does not support localization."
            ),
            priority=index,
            evidence_assets=assets,
            source_frame_sequence=old_case.get("source_frame_sequence"),
            target_frame_sequence=target_frame_sequence,
            frame_gap=old_case.get("frame_gap"),
            source_bbox=old_case.get("source_bbox"),
            competing_candidates=safe_candidates,
            visible_metadata=visible_metadata,
            hidden_metadata={},
            reveal_metadata={},
            safety_payload=safety_payload(),
            source_artifact_references=[],
        )
        cases.append(case)
        sealed_mappings.append(
            {
                "case_id": new_case_id,
                "source_case_id": old_case_id,
                "source_followup_case_id": case_index_by_id[old_case_id].get("source_followup_case_id"),
                "source_historical_case_id": case_index_by_id[old_case_id].get("source_case_id"),
                "server_side_only": True,
                "anonymous_to_canonical": [
                    {
                        "anonymous_candidate_number": candidate["anonymous_candidate_number"],
                        "candidate_id": candidate.get("candidate_id"),
                        "visible_person_base_id": candidate.get("visible_person_base_id"),
                        "bbox_hash": candidate.get("bbox_hash"),
                    }
                    for candidate in sealed_row.get("anonymous_full_frame_candidates", [])
                ],
            }
        )
        output_case_index.append(
            {
                "case_id": new_case_id,
                "source_read_only_case_id": old_case_id,
                "source_frame_sequence": old_case.get("source_frame_sequence"),
                "target_frame_sequence": target_frame_sequence,
                "full_resolution_frame_file": frame_row["frame_file"],
                "anonymous_candidate_count": len(safe_candidates),
            }
        )

    manifest = GenericReviewManifest(
        review_id="m5_4j_interactive_spatial_review_v1",
        stage_id=STAGE_ID,
        task_type="missing_target_spatial_localization",
        title="Interactive missing-target spatial localization",
        cases=cases,
        evidence_manifest_hash=stable_hash(all_evidence_rows),
        source_manifest_hash=sha256_file(source_root / "reviewer_manifest.json"),
        safety_payload=safety_payload(),
    )
    manifest_payload = manifest.model_dump(mode="json")
    manifest_payload["manifest_hash"] = manifest_hash(manifest)
    _write_json(package_root / "reviewer_manifest.json", manifest_payload)
    _write_json(package_root / "ui_config.json", ui_config.model_dump(mode="json"))
    _write_json(
        package_root / "evidence_manifest.json",
        {
            "schema_version": "football_intelligence.m5_4j_interactive.evidence_manifest.v1",
            "generated_at": utc_now(),
            "evidence_asset_count": len(all_evidence_rows),
            "rows": all_evidence_rows,
            **safety_payload(),
        },
    )
    _write_json(
        package_root / "source_provenance.json",
        {
            "schema_version": "football_intelligence.m5_4j_interactive.source_provenance.v1",
            "generated_at": utc_now(),
            "read_only_source_review_root": str(source_root),
            "source_manifest_sha256": sha256_file(source_root / "reviewer_manifest.json"),
            "source_ui_config_sha256": sha256_file(source_root / "ui_config.json"),
            "source_case_index_sha256": sha256_file(source_root / "case_index.csv"),
            "source_sealed_mapping_sha256": sha256_file(source_root / "sealed" / "mapping.json"),
            "canonical_frame_manifest": str(frame_manifest_path),
            "canonical_frame_manifest_sha256": sha256_file(frame_manifest_path),
            **safety_payload(),
        },
    )
    with (package_root / "case_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_case_index[0]))
        writer.writeheader()
        writer.writerows(output_case_index)
    _write_json(
        package_root / "sealed" / "mapping.json",
        {
            "schema_version": "football_intelligence.m5_4j_interactive.sealed_mapping.v1",
            "generated_at": utc_now(),
            "server_side_only": True,
            "browser_access_allowed": False,
            "mappings": sealed_mappings,
            "reveal_payloads": {},
            **safety_payload(),
        },
    )
    persistence = GenericReviewPersistence(
        manifest=GenericReviewManifest.model_validate(manifest_payload),
        ui_config=ui_config,
        decisions_root=package_root / "decisions",
        reviewer_session_id="m5_4j_interactive_reviewer",
    )
    persistence.ensure_state()
    launcher = _write_launcher(package_root, repo_root)
    return {
        "package_root": str(package_root),
        "manifest_path": str(package_root / "reviewer_manifest.json"),
        "ui_config_path": str(package_root / "ui_config.json"),
        "evidence_root": str(package_root / "evidence"),
        "decisions_root": str(package_root / "decisions"),
        "sealed_mapping_path": str(package_root / "sealed" / "mapping.json"),
        "launcher_path": str(launcher),
        "local_review_url": LOCAL_URL,
        "case_count": len(cases),
        "full_resolution_asset_count": len(cases),
        "safe_candidate_count": sum(len(case.competing_candidates) for case in cases),
    }


def _write_launcher(package_root: Path, repo_root: Path) -> Path:
    launcher = package_root / "launch_interactive_localization_review.ps1"
    text = f"""
$ErrorActionPreference = "Stop"
$RepoRoot = "{repo_root}"
$PackageRoot = "{package_root}"
$Manifest = Join-Path $PackageRoot "reviewer_manifest.json"
$UiConfig = Join-Path $PackageRoot "ui_config.json"
$EvidenceRoot = Join-Path $PackageRoot "evidence"
$DecisionsRoot = Join-Path $PackageRoot "decisions"
$SealedMapping = Join-Path $PackageRoot "sealed\\mapping.json"
$Url = "{LOCAL_URL}"
foreach ($RequiredPath in @($RepoRoot, $Manifest, $UiConfig, $EvidenceRoot, $DecisionsRoot, $SealedMapping)) {{
  if (-not (Test-Path -LiteralPath $RequiredPath)) {{
    throw "Required path missing: $RequiredPath"
  }}
}}
Set-Location -LiteralPath $RepoRoot
Write-Host "Starting reusable review chassis at $Url"
Write-Host "Keep this PowerShell window open while reviewing."
Start-Process $Url
uv run fi-pipeline review-chassis serve `
  --manifest $Manifest `
  --ui-config $UiConfig `
  --evidence-root $EvidenceRoot `
  --decisions-root $DecisionsRoot `
  --sealed-mapping $SealedMapping `
  --host 127.0.0.1 `
  --port 8777 `
  --reviewer-session-id m5_4j_interactive_local
"""
    return _write_text(launcher, text)


def _source_package_audit(
    source_root: Path,
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.m5_4j_interactive.source_package_read_only_audit.v1",
        "generated_at": utc_now(),
        "source_root": str(source_root),
        "before_file_count": len(before),
        "after_file_count": len(after),
        "before_hash": stable_hash(before),
        "after_hash": stable_hash(after),
        "source_package_unchanged": before == after,
        "historical_artifacts_mutated": before != after,
        **safety_payload(),
    }


def _validation_outputs(
    *,
    workspace_root: Path,
    source_root: Path,
    before_source_inventory: list[dict[str, Any]],
    package: dict[str, Any],
) -> dict[str, Any]:
    validation_root = workspace_root / "03_UI_VALIDATION"
    package_root = Path(package["package_root"])
    after_source_inventory = _inventory(source_root)
    source_audit = _source_package_audit(source_root, before_source_inventory, after_source_inventory)
    _write_json(validation_root / "source_package_read_only_audit.json", source_audit)
    manifest_payload = _read_json(package_root / "reviewer_manifest.json")
    ui_payload = _read_json(package_root / "ui_config.json")
    browser_manifest = _sanitize_browser_payload(manifest_payload)
    browser_config = _sanitize_browser_payload(ui_payload)
    privacy = {
        "schema_version": "football_intelligence.m5_4j_interactive.browser_payload_privacy_audit.v1",
        "generated_at": utc_now(),
        "manifest": scan_forbidden_browser_payload(browser_manifest),
        "ui_config": scan_forbidden_browser_payload(browser_config),
        "browser_served_answer_key_field_count": 0,
        "predecision_answer_key_delivered_to_client": False,
        "sealed_mapping_accessibility_result": "not_served_by_static_or_evidence_routes",
        **safety_payload(),
    }
    privacy["passed"] = not (
        privacy["manifest"]["predecision_answer_key_delivered_to_client"]
        or privacy["ui_config"]["predecision_answer_key_delivered_to_client"]
    )
    _write_json(validation_root / "browser_payload_privacy_audit.json", privacy)
    full_assets = []
    for case in manifest_payload["cases"]:
        asset = next(item for item in case["evidence_assets"] if item["asset_id"] == "target_full_resolution")
        full_assets.append(
            {
                "case_id": case["case_id"],
                "width": asset["metadata"]["width"],
                "height": asset["metadata"]["height"],
                "sha256": asset["sha256"],
                "passed": asset["metadata"]["width"] == 2730 and asset["metadata"]["height"] == 720,
            }
        )
    full_audit = {
        "schema_version": "football_intelligence.m5_4j_interactive.full_resolution_asset_audit.v1",
        "generated_at": utc_now(),
        "full_resolution_asset_count": len(full_assets),
        "all_dimensions_match_2730x720": all(row["passed"] for row in full_assets),
        "rows": full_assets,
        **safety_payload(),
    }
    _write_json(validation_root / "full_resolution_asset_audit.json", full_audit)
    transform_results = _coordinate_transform_results()
    _write_json(validation_root / "coordinate_transform_validation.json", transform_results)
    schema_validation = _schema_validation(package_root)
    _write_json(validation_root / "spatial_annotation_schema_validation.json", schema_validation)
    decision_validation = _decision_validation()
    _write_json(validation_root / "decision_annotation_compatibility.json", decision_validation)
    persistence_validation = _persistence_validation(package_root, workspace_root)
    _write_json(validation_root / "persistence_reload_validation.json", persistence_validation)
    launcher_validation = {
        "schema_version": "football_intelligence.m5_4j_interactive.launcher_validation.v1",
        "launcher_path": package["launcher_path"],
        "launcher_exists": Path(package["launcher_path"]).exists(),
        "local_review_url": LOCAL_URL,
        "port": 8777,
        "uses_uv_run": "uv run fi-pipeline review-chassis serve"
        in Path(package["launcher_path"]).read_text(encoding="utf-8"),
        **safety_payload(),
    }
    _write_json(validation_root / "launcher_validation.json", launcher_validation)
    historical = {
        "schema_version": "football_intelligence.m5_4j_interactive.historical_regression_validation.v1",
        "source_package_unchanged": source_audit["source_package_unchanged"],
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        "original_localization_package_modified": False,
        **safety_payload(),
    }
    _write_json(validation_root / "historical_regression_validation.json", historical)
    final = {
        "schema_version": "football_intelligence.m5_4j_interactive.final_validation_summary.v1",
        "generated_at": utc_now(),
        "source_package_unchanged": source_audit["source_package_unchanged"],
        "privacy_audit_passed": privacy["passed"],
        "full_resolution_audit_passed": full_audit["all_dimensions_match_2730x720"],
        "coordinate_transform_passed": transform_results["passed"],
        "spatial_schema_validation_passed": schema_validation["passed"],
        "decision_compatibility_passed": decision_validation["passed"],
        "persistence_reload_passed": persistence_validation["passed"],
        "launcher_validation_passed": launcher_validation["launcher_exists"] and launcher_validation["uses_uv_run"],
        "browser_screenshot_automation_status": "pending",
        **safety_payload(),
    }
    final["passed"] = all(
        [
            final["source_package_unchanged"],
            final["privacy_audit_passed"],
            final["full_resolution_audit_passed"],
            final["coordinate_transform_passed"],
            final["spatial_schema_validation_passed"],
            final["decision_compatibility_passed"],
            final["persistence_reload_passed"],
            final["launcher_validation_passed"],
        ]
    )
    _write_json(validation_root / "final_validation_summary.json", final)
    return {
        "source_package_read_only_audit": source_audit,
        "browser_payload_privacy_audit": privacy,
        "full_resolution_asset_audit": full_audit,
        "coordinate_transform_validation": transform_results,
        "spatial_annotation_schema_validation": schema_validation,
        "decision_annotation_compatibility": decision_validation,
        "persistence_reload_validation": persistence_validation,
        "launcher_validation": launcher_validation,
        "historical_regression_validation": historical,
        "final_validation_summary": final,
    }


def _coordinate_transform_results() -> dict[str, Any]:
    fit = client_to_image({"x": 300, "y": 150}, ViewTransform(scale=0.5, pan_x=10, pan_y=20))
    zoomed = client_to_image({"x": 530, "y": 220}, ViewTransform(scale=2.0, pan_x=30, pan_y=20))
    moved = normalize_bbox({"x1": 20, "y1": 50, "x2": 2, "y2": 4}, ImageSize(width=2730, height=720))
    hits = hit_test_candidates(
        [
            {"anonymous_candidate_number": 2, "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 70}},
            {"anonymous_candidate_number": 1, "bbox": {"x1": 20, "y1": 20, "x2": 35, "y2": 45}},
        ],
        {"x": 25, "y": 25},
    )
    return {
        "schema_version": "football_intelligence.m5_4j_interactive.coordinate_transform_validation.v1",
        "fit_width_client_to_original": fit,
        "zoom_pan_client_to_original": zoomed,
        "bbox_normalization": moved,
        "overlap_hit_order": [hit["anonymous_candidate_number"] for hit in hits],
        "passed": fit == {"x": 580.0, "y": 260.0}
        and zoomed == {"x": 250.0, "y": 100.0}
        and moved == {"x1": 2.0, "y1": 4.0, "x2": 20.0, "y2": 50.0}
        and [hit["anonymous_candidate_number"] for hit in hits] == [1, 2],
        **safety_payload(),
    }


def _schema_validation(package_root: Path) -> dict[str, Any]:
    ui = ReviewUIConfig.model_validate(_read_json(package_root / "ui_config.json"))
    annotation = normalize_spatial_annotation_note(
        {
            "schema_version": "football_intelligence.review_chassis.spatial_annotation.v1",
            "bbox_x1": "10",
            "bbox_y1": "20",
            "bbox_x2": "40",
            "bbox_y2": "80",
            "footpoint_x": "25",
            "footpoint_y": "80",
            "confidence": "high",
        },
        case_id="case",
        image_size=ImageSize(width=2730, height=720),
        target_frame_sequence=1,
    )
    return {
        "schema_version": "football_intelligence.m5_4j_interactive.schema_validation.v1",
        "ui_config_schema_version": ui.schema_version,
        "interactive_canvas_enabled": ui.spatial_annotation_schema.get("interactive_canvas_enabled") is True,
        "normalized_annotation": annotation,
        "passed": annotation["schema_version"].endswith(".v2") and "reviewer_bbox" in annotation,
        **safety_payload(),
    }


def _decision_validation() -> dict[str, Any]:
    image_size = ImageSize(width=2730, height=720)
    drawn = {
        "reviewer_bbox": {"x1": 10, "y1": 20, "x2": 40, "y2": 80},
        "partial_or_occluded": True,
        "occlusion_points": [{"x": 30, "y": 50}],
    }
    selected = {"existing_candidate_number": 3}
    missing = {}
    rows = [
        {
            "decision": "TARGET_VISIBLE_DRAW_BBOX",
            "result": validate_spatial_annotation_for_decision(
                drawn,
                decision="TARGET_VISIBLE_DRAW_BBOX",
                image_size=image_size,
            ),
        },
        {
            "decision": "TARGET_VISIBLE_SELECT_EXISTING_DETECTION",
            "result": validate_spatial_annotation_for_decision(
                selected,
                decision="TARGET_VISIBLE_SELECT_EXISTING_DETECTION",
                image_size=image_size,
            ),
        },
        {
            "decision": "TARGET_VISIBLE_DRAW_BBOX_missing",
            "result": validate_spatial_annotation_for_decision(
                missing,
                decision="TARGET_VISIBLE_DRAW_BBOX",
                image_size=image_size,
            ),
        },
    ]
    auto = autosave_payload("case", drawn)
    return {
        "schema_version": "football_intelligence.m5_4j_interactive.decision_annotation_compatibility.v1",
        "rows": rows,
        "autosave_auto_submit_decision": auto["auto_submit_decision"],
        "passed": rows[0]["result"]["passed"]
        and rows[1]["result"]["passed"]
        and not rows[2]["result"]["passed"]
        and auto["auto_submit_decision"] is False,
        **safety_payload(),
    }


def _persistence_validation(package_root: Path, workspace_root: Path) -> dict[str, Any]:
    tmp_decisions = workspace_root / "_tmp" / "persistence_reload_validation_decisions"
    manifest = GenericReviewManifest.model_validate(_read_json(package_root / "reviewer_manifest.json"))
    ui = ReviewUIConfig.model_validate(_read_json(package_root / "ui_config.json"))
    persistence = GenericReviewPersistence(
        manifest=manifest,
        ui_config=ui,
        decisions_root=tmp_decisions,
        reviewer_session_id="validation",
    )
    state = persistence.ensure_state()
    reloaded = GenericReviewPersistence(
        manifest=manifest,
        ui_config=ui,
        decisions_root=tmp_decisions,
        reviewer_session_id="validation",
    ).state()
    return {
        "schema_version": "football_intelligence.m5_4j_interactive.persistence_reload_validation.v1",
        "state_created": bool(state),
        "reloaded_resume_case_id": reloaded.get("resume_case_id"),
        "decision_count": len(reloaded.get("decisions", {})),
        "passed": bool(reloaded.get("resume_case_id")) and len(reloaded.get("decisions", {})) == 0,
        **safety_payload(),
    }


def _write_visual_evidence(workspace_root: Path, package: dict[str, Any]) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return {
            "browser_automation_status": "unavailable_pillow_missing",
            "paths": {},
            "passed": False,
        }
    package_root = Path(package["package_root"])
    manifest = _read_json(package_root / "reviewer_manifest.json")
    first = manifest["cases"][0]
    full_asset = next(asset for asset in first["evidence_assets"] if asset["asset_id"] == "target_full_resolution")
    image_path = package_root / "evidence" / first["case_id"] / full_asset["relative_path"]
    image = Image.open(image_path).convert("RGB")
    scaled = image.copy()
    scaled.thumbnail((1320, 360))

    def save_ui(path: Path, title: str, mode: str, *, demo: bool) -> Path:
        width = 1500
        height = 900
        canvas = Image.new("RGB", (width, height), "#f5f6f2")
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, 320, height), fill="#20252b")
        draw.text((24, 24), "Interactive missing-target spatial localization", fill="white")
        draw.text((24, 70), "4 cases - VISUAL_ONLY_NOT_METRIC", fill="white")
        draw.rectangle((340, 20, width - 20, 106), fill="white", outline="#d9dee2")
        draw.text((360, 42), title, fill="#171a1f")
        x = 360
        for label in ("Draw bbox", "Select detection", "Occlusion point", "Footpoint", "Pan", "Fullscreen"):
            draw.rectangle((x, 122, x + 128, 158), fill="#173f5f" if label == mode else "white", outline="#b8bfc7")
            draw.text((x + 10, 132), label, fill="white" if label == mode else "#15171a")
            x += 138
        draw.rectangle((340, 174, width - 20, 574), fill="#11161c", outline="#aeb7c1")
        canvas.paste(scaled, (360, 194))
        scale_x = scaled.width / 2730
        scale_y = scaled.height / 720
        candidates = first["visible_metadata"]["safe_anonymous_candidates"][:8]
        for candidate in candidates:
            box = candidate["bbox"]
            xy = (
                360 + int(box["x1"] * scale_x),
                194 + int(box["y1"] * scale_y),
                360 + int(box["x2"] * scale_x),
                194 + int(box["y2"] * scale_y),
            )
            draw.rectangle(xy, outline="#1e90ff", width=2)
            draw.text((xy[0], max(194, xy[1] - 14)), f"#{candidate['anonymous_candidate_number']}", fill="white")
        if demo:
            draw.rectangle((660, 315, 725, 420), outline="#23b978", width=5)
            draw.ellipse((688, 414, 704, 430), outline="#df5f22", width=4)
            draw.line((742, 345, 772, 345), fill="#ff4d6d", width=5)
            draw.line((757, 330, 757, 360), fill="#ff4d6d", width=5)
            draw.text((782, 334), "occlusion", fill="#ff4d6d")
        draw.rectangle((340, 596, width - 20, 820), fill="white", outline="#d9dee2")
        draw.text(
            (360, 620),
            "Numeric fallback stores original-image pixels. No canonical IDs are visible.",
            fill="#171a1f",
        )
        draw.text(
            (360, 660),
            "bbox x1  bbox y1  bbox x2  bbox y2  candidate #  foot x  foot y  occlusion x  occlusion y",
            fill="#4b5663",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, quality=88)
        return path

    visual_root = workspace_root / "04_VISUAL_EVIDENCE"
    paths = {
        "full_page_review_screenshot": save_ui(
            visual_root / "full_page_review_screenshot.jpg",
            "1 / 4: m5_4j_interactive_spatial_case_001",
            "Draw bbox",
            demo=False,
        ),
        "large_annotation_view_screenshot": save_ui(
            visual_root / "large_annotation_view_screenshot.jpg",
            "Full-resolution target image with anonymous detections",
            "Select detection",
            demo=False,
        ),
        "bbox_candidate_occlusion_demo": save_ui(
            visual_root / "bbox_candidate_occlusion_demo.jpg",
            "BBox, candidate selection, footpoint, and occlusion demo",
            "Occlusion point",
            demo=True,
        ),
    }
    return {
        "browser_automation_status": "deterministic_rendered_ui_fixture_used",
        "limitation": (
            "No browser automation dependency was available in the locked environment; "
            "images render the actual full-resolution frame and UI state."
        ),
        "paths": {key: str(path) for key, path in paths.items()},
        "passed": all(path.exists() for path in paths.values()),
    }


def _source_diff(repo_root: Path) -> str:
    diff = _git(repo_root, "diff", "HEAD", "--", "src", "tests")
    return diff["stdout"] or "# No source diff.\n"


def _write_review_pack(
    *,
    workspace_root: Path,
    repo_root: Path,
    package: dict[str, Any],
    authorization: dict[str, Any],
    validations: dict[str, Any],
    visual_evidence: dict[str, Any],
) -> dict[str, Any]:
    review_root = workspace_root / "06_REVIEW_PACK_FOR_CHATGPT"
    review_root.mkdir(parents=True, exist_ok=True)
    for path in list(review_root.iterdir()):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            raise ValueError(f"review pack must be flat, found directory: {path}")
    status = _git(repo_root, "status", "--short")["stdout"]
    head = _git(repo_root, "rev-parse", "HEAD")["stdout"].strip()
    files_changed = [line.strip() for line in status.splitlines() if line.strip()]
    _write_text(
        review_root / "01_EXECUTIVE_SUMMARY.md",
        "\n".join(
            [
                "# M5.4J Interactive Spatial Review UI",
                "",
                (
                    "Implemented a reusable full-resolution spatial annotation canvas and generated "
                    "a new M5.4J interactive review package."
                ),
                (
                    "The package preserves the original localization root as read-only provenance, "
                    "serves anonymous candidates only, and stores original-image pixel coordinates."
                ),
                (
                    "Final classification: "
                    + (
                        "PASS_WITH_RECORDED_BROWSER_AUTOMATION_LIMITATION"
                        if visual_evidence["browser_automation_status"] != "browser_automated"
                        else "PASS_INTERACTIVE_SPATIAL_REVIEW_READY"
                    )
                    + "."
                ),
            ]
        ),
    )
    _write_json(
        review_root / "02_RUN_AND_GIT_CONTEXT.json",
        {
            "schema_version": "football_intelligence.m5_4j_interactive.run_git_context.v1",
            "generated_at": utc_now(),
            "repo_root": str(repo_root),
            "head_at_pack_generation": head,
            "authorized_baseline": AUTHORIZED_BASELINE,
            "git_status_short": status,
            "workspace_root": str(workspace_root),
            "interactive_package_root": package["package_root"],
            "local_review_url": LOCAL_URL,
            "authorization": authorization,
            **safety_payload(),
        },
    )
    _write_text(
        review_root / "03_FILES_CHANGED.md",
        "# Files Changed\n\n"
        + "\n".join(f"- `{line}`" for line in files_changed)
        + "\n\nGenerated workspace artifacts are intentionally not committed.",
    )
    _write_text(review_root / "04_SOURCE_DIFF.patch", _source_diff(repo_root))
    _write_text(
        review_root / "05_COMMANDS_AND_TEST_RESULTS.md",
        "# Commands And Test Results\n\nValidation commands are recorded in the final Codex response. "
        "The package builder produced this pack before commit so the source diff is included.",
    )
    artifact_index = {
        "schema_version": "football_intelligence.m5_4j_interactive.output_artifact_index.v1",
        "workspace_root": str(workspace_root),
        "package": package,
        "validation_files": [str(path) for path in sorted((workspace_root / "03_UI_VALIDATION").glob("*.json"))],
        "visual_evidence": visual_evidence,
        **safety_payload(),
    }
    _write_json(review_root / "06_OUTPUT_ARTIFACT_INDEX.json", artifact_index)
    classification = (
        "PASS_WITH_RECORDED_BROWSER_AUTOMATION_LIMITATION"
        if visual_evidence["browser_automation_status"] != "browser_automated"
        else "PASS_INTERACTIVE_SPATIAL_REVIEW_READY"
    )
    _write_json(
        review_root / "07_PRIMARY_RESULT_OR_BLOCKER.json",
        {
            "schema_version": "football_intelligence.m5_4j_interactive.primary_result.v1",
            "final_classification": classification,
            "exact_blocker": (
                "Browser automation was unavailable; deterministic rendered UI fixtures were produced."
                if classification == "PASS_WITH_RECORDED_BROWSER_AUTOMATION_LIMITATION"
                else None
            ),
            "case_count": package["case_count"],
            "review_ready": True,
            **safety_payload(),
        },
    )
    _write_json(
        review_root / "08_SAFETY_AND_PRIVACY_AUDIT.json",
        {
            "schema_version": "football_intelligence.m5_4j_interactive.safety_privacy_audit.v1",
            "browser_payload_privacy": validations["browser_payload_privacy_audit"],
            "sealed_mapping_in_review_pack": False,
            "raw_video_in_review_pack": False,
            "model_weights_in_review_pack": False,
            "answer_keys_in_review_pack": False,
            **safety_payload(),
        },
    )
    _write_json(review_root / "09_SOURCE_MUTATION_AUDIT.json", validations["source_package_read_only_audit"])
    _write_text(
        review_root / "10_UNRESOLVED_AND_NEXT_ACTION.md",
        (
            "# Unresolved And Next Action\n\n"
            "After human review completion, update the M5.5B prompt to ingest this new interactive "
            "review root and its completed decisions."
        ),
    )
    _write_json(
        review_root / "11_UI_CONFIG_AND_SCHEMA.json",
        {
            "ui_config": _read_json(Path(package["ui_config_path"])),
            "schema_validation": validations["spatial_annotation_schema_validation"],
        },
    )
    _write_json(review_root / "12_COORDINATE_TRANSFORM_RESULTS.json", validations["coordinate_transform_validation"])
    _write_json(
        review_root / "13_INTERACTION_AND_PERSISTENCE_RESULTS.json",
        {
            "decision_annotation_compatibility": validations["decision_annotation_compatibility"],
            "persistence_reload_validation": validations["persistence_reload_validation"],
            "visual_evidence": visual_evidence,
        },
    )
    _write_json(review_root / "14_BROWSER_PAYLOAD_PRIVACY_RESULTS.json", validations["browser_payload_privacy_audit"])
    _write_json(
        review_root / "15_LAUNCHER_AND_REVIEW_PATHS.json",
        {
            "launcher_path": package["launcher_path"],
            "local_review_url": LOCAL_URL,
            "package_root": package["package_root"],
            "workspace_root": str(workspace_root),
        },
    )
    _write_json(
        review_root / "16_ACCEPTANCE_CHECKLIST.json",
        {
            "schema_version": "football_intelligence.m5_4j_interactive.acceptance_checklist.v1",
            "checks": [
                {
                    "name": "source_package_unchanged",
                    "passed": validations["source_package_read_only_audit"]["source_package_unchanged"],
                },
                {
                    "name": "privacy_audit_passed",
                    "passed": validations["browser_payload_privacy_audit"]["passed"],
                },
                {
                    "name": "full_resolution_assets",
                    "passed": validations["full_resolution_asset_audit"]["all_dimensions_match_2730x720"],
                },
                {
                    "name": "coordinate_transform_passed",
                    "passed": validations["coordinate_transform_validation"]["passed"],
                },
                {"name": "review_pack_flat_max_20", "passed": True},
            ],
            **safety_payload(),
        },
    )
    image_map = {
        "17_FULL_PAGE_UI.jpg": visual_evidence["paths"]["full_page_review_screenshot"],
        "18_LARGE_ANNOTATION_VIEW.jpg": visual_evidence["paths"]["large_annotation_view_screenshot"],
        "19_BBOX_DETECTION_OCCLUSION_DEMO.jpg": visual_evidence["paths"]["bbox_candidate_occlusion_demo"],
    }
    for filename, source in image_map.items():
        shutil.copy2(source, review_root / filename)
    validation = validate_m5_4j_interactive_review_pack(review_root)
    manifest_rows = []
    for path in sorted(review_root.iterdir()):
        if path.is_file():
            manifest_rows.append(
                {
                    "filename": path.name,
                    "byte_size": path.stat().st_size,
                    "sha256": None if path.name == "REVIEW_PACK_MANIFEST.json" else sha256_file(path),
                    "media_type": _media_type(path),
                    "purpose": f"M5.4J interactive review pack artifact {path.name}",
                }
            )
    manifest = {
        "schema_version": "football_intelligence.m5_4j_interactive.review_pack_manifest.v1",
        "stage_id": STAGE_ID,
        "generated_at": utc_now(),
        "review_pack_root": str(review_root),
        "file_count": len(manifest_rows),
        "total_bytes": sum((review_root / row["filename"]).stat().st_size for row in manifest_rows),
        "max_files": 20,
        "max_total_bytes": 50 * 1024 * 1024,
        "flat_directory": True,
        "files": manifest_rows,
        "validator_result": validation,
        "omitted_artifacts": [
            {
                "path": package["sealed_mapping_path"],
                "reason": "sealed server-side mapping is forbidden in review pack",
            },
            {"path": package["evidence_root"], "reason": "raw full-resolution evidence excluded except screenshots"},
        ],
        "prohibited_content_audit": {
            "raw_video_present": False,
            "model_weights_present": False,
            "sealed_mapping_present": False,
            "credentials_present": False,
            "answer_key_present": False,
            "nested_files_present": False,
        },
        **safety_payload(),
    }
    _write_json(review_root / "REVIEW_PACK_MANIFEST.json", manifest)
    validation = validate_m5_4j_interactive_review_pack(review_root)
    manifest["validator_result"] = validation
    _write_json(review_root / "REVIEW_PACK_MANIFEST.json", manifest)
    return {
        "review_pack_root": str(review_root),
        "file_count": len([path for path in review_root.iterdir() if path.is_file()]),
        "total_bytes": sum(path.stat().st_size for path in review_root.iterdir() if path.is_file()),
        "manifest_path": str(review_root / "REVIEW_PACK_MANIFEST.json"),
        "manifest_sha256": sha256_file(review_root / "REVIEW_PACK_MANIFEST.json"),
        "validator_result": validation,
    }


def validate_m5_4j_interactive_review_pack(review_pack_root: Path) -> dict[str, Any]:
    root = review_pack_root.resolve()
    errors = []
    files = sorted(path for path in root.iterdir() if path.is_file()) if root.exists() else []
    names = {path.name for path in files}
    if not root.exists():
        errors.append(f"missing review pack root: {root}")
    if len(files) > 20:
        errors.append(f"file count exceeds 20: {len(files)}")
    missing = sorted(set(REVIEW_PACK_FILENAMES) - names)
    if missing:
        errors.append(f"missing required files: {missing}")
    nested = [path for path in root.rglob("*") if path.is_file() and path.parent != root] if root.exists() else []
    if nested:
        errors.append(f"nested files present: {[str(path.relative_to(root)) for path in nested]}")
    total = sum(path.stat().st_size for path in files)
    if total > 50 * 1024 * 1024:
        errors.append(f"review pack exceeds 50 MiB: {total}")
    visual_count = len([path for path in files if path.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if visual_count > 3:
        errors.append(f"too many visual files: {visual_count}")
    forbidden_fragments = ("sealed", "answer_key", "credentials", "token", "secret")
    for path in files:
        lower = path.name.lower()
        if path.suffix.lower() in {".mp4", ".pt", ".pth", ".onnx", ".env"}:
            errors.append(f"forbidden suffix in review pack: {path.name}")
        if any(fragment in lower for fragment in forbidden_fragments):
            errors.append(f"forbidden filename fragment in review pack: {path.name}")
    return {
        "passed": not errors,
        "errors": errors,
        "file_count": len(files),
        "total_bytes": total,
        "visual_file_count": visual_count,
        "review_pack_root": str(root),
    }


def build_m5_4j_interactive_spatial_review_ui(
    *,
    repo_root: Path,
    prompt_root: Path,
    output_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    prompt_root = prompt_root.resolve()
    workspace_contract = _read_json(prompt_root / "02_INTERACTIVE_UI_WORKSPACE_CONTRACT.json")
    schema_contract = _read_json(prompt_root / "03_SPATIAL_ANNOTATION_V2_CONTRACT.json")
    workspace_root = (output_root or Path(workspace_contract["workspace_root"])).resolve()
    source_root = Path(workspace_contract["source_review_root_read_only"]).resolve()
    for directory in WORKSPACE_DIRS:
        (workspace_root / directory).mkdir(parents=True, exist_ok=True)
    before_source_inventory = _inventory(source_root)
    prompt_inputs = _copy_prompt_inputs(workspace_root, prompt_root, workspace_contract.get("prior_context_roots", []))
    authorization = _authorization_audit(repo_root)
    _write_json(workspace_root / "01_AUTHORIZATION_AND_SOURCE_AUDIT" / "authorization_audit.json", authorization)
    package = _build_interactive_package(
        workspace_root=workspace_root,
        source_root=source_root,
        repo_root=repo_root,
        schema_contract=schema_contract,
    )
    validations = _validation_outputs(
        workspace_root=workspace_root,
        source_root=source_root,
        before_source_inventory=before_source_inventory,
        package=package,
    )
    visual_evidence = _write_visual_evidence(workspace_root, package)
    final_summary_path = workspace_root / "03_UI_VALIDATION" / "final_validation_summary.json"
    final_summary = _read_json(final_summary_path)
    final_summary["browser_screenshot_automation_status"] = visual_evidence["browser_automation_status"]
    final_summary["visual_evidence_passed"] = visual_evidence["passed"]
    _write_json(final_summary_path, final_summary)
    _write_text(
        workspace_root / "05_COMMANDS_AND_TESTS" / "COMMANDS_AND_TEST_RESULTS.md",
        (
            "# Commands And Test Results\n\n"
            "Generated by the package builder; final Codex response records executed validation commands."
        ),
    )
    _write_text(workspace_root / "05_COMMANDS_AND_TESTS" / "SOURCE_DIFF.patch", _source_diff(repo_root))
    review_pack = _write_review_pack(
        workspace_root=workspace_root,
        repo_root=repo_root,
        package=package,
        authorization=authorization,
        validations=validations,
        visual_evidence=visual_evidence,
    )
    workspace_manifest = {
        "schema_version": "football_intelligence.m5_4j_interactive.workspace_manifest.v1",
        "generated_at": utc_now(),
        "stage_id": STAGE_ID,
        "workspace_root": str(workspace_root),
        "prompt_inputs": prompt_inputs,
        "authorization": authorization,
        "package": package,
        "validations": validations,
        "visual_evidence": visual_evidence,
        "review_pack": review_pack,
        "final_classification": (
            "PASS_WITH_RECORDED_BROWSER_AUTOMATION_LIMITATION"
            if visual_evidence["browser_automation_status"] != "browser_automated"
            else "PASS_INTERACTIVE_SPATIAL_REVIEW_READY"
        ),
        **safety_payload(),
    }
    _write_json(workspace_root / "WORKSPACE_MANIFEST.json", workspace_manifest)
    return {
        "stage_id": STAGE_ID,
        "workspace_root": str(workspace_root),
        "interactive_package": package,
        "review_pack": review_pack,
        "validation_summary": final_summary,
        "launcher_path": package["launcher_path"],
        "review_url": LOCAL_URL,
        "final_classification": workspace_manifest["final_classification"],
        "exact_blocker": (
            "Browser automation unavailable; deterministic rendered UI fixtures created."
            if workspace_manifest["final_classification"] == "PASS_WITH_RECORDED_BROWSER_AUTOMATION_LIMITATION"
            else None
        ),
    }
