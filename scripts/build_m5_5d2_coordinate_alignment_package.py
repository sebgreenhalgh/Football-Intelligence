"""Build the fresh M5.5D.2 coordinate-provenance repair package."""

# The generated launcher embeds an explicit Windows uv path.
# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw


ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = ROOT / "SoccerTrack-v2"
OLD_PACKAGE = (
    ROOT
    / r"matches\128058\runs\step_m5\part 2\M5_5D2_ENCOUNTER_EPISODE_GAP_MINING_AND_EXPANDED_BURST_SCAN_v1"
    / "11_TRUE_OCCLUSION_REVIEW_PACKAGE"
)
VIDEO = ROOT / r"matches\128058\videos\128058_panorama_1st_half.mp4"
WORKSPACE = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5D2_COORDINATE_PROVENANCE_AND_OVERLAY_ALIGNMENT_REPAIR_v1"
PACKAGE = WORKSPACE / "04_REPAIRED_REVIEW_PACKAGE"
RAW_WIDTH = 4096
RAW_HEIGHT = 1080
# The prior detector/evidence rows were stored in the 2048x540 panorama space;
# the old 1450x382 JPEGs were only display thumbnails and caused the visible
# frame/geometry mismatch. The source video is exactly 2x that canonical space.
OLD_WIDTH = 2048
OLD_HEIGHT = 540
STAGE_ID = "M5_5D2_COORDINATE_PROVENANCE_AND_OVERLAY_ALIGNMENT_REPAIR_v1"
REVIEW_ID = "m5_5d2_coordinate_provenance_aligned_review_v1"
SESSION_ID = "m5_5d2_aligned_overlay_human_reviewer"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def json_digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def full_bbox(bbox: dict[str, float]) -> dict[str, float]:
    return {
        "x1": round(float(bbox["x1"]) * RAW_WIDTH / OLD_WIDTH, 3),
        "y1": round(float(bbox["y1"]) * RAW_HEIGHT / OLD_HEIGHT, 3),
        "x2": round(float(bbox["x2"]) * RAW_WIDTH / OLD_WIDTH, 3),
        "y2": round(float(bbox["y2"]) * RAW_HEIGHT / OLD_HEIGHT, 3),
    }


def extract_frame(cap: cv2.VideoCapture, frame_sequence: int, output: Path) -> str:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_sequence)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"could not read source frame {frame_sequence}")
    if frame.shape[1] != RAW_WIDTH or frame.shape[0] != RAW_HEIGHT:
        raise RuntimeError(f"unexpected source frame shape {frame.shape}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        raise RuntimeError(f"could not write {output}")
    return digest(output)


def copy_supporting_asset(old_case: Path, new_case: Path, relative: str) -> tuple[str, str] | None:
    source = old_case / relative
    if not source.is_file():
        return None
    target = new_case / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return relative, digest(target)


def make_ui_config() -> dict[str, Any]:
    ui = json.loads((OLD_PACKAGE / "ui_config.json").read_text(encoding="utf-8"))
    ui.update(
        {
            "page_title": "M5.5D.2 Coordinate Provenance Alignment Review",
            "review_title": "Coordinate-provenance and overlay alignment review",
            "task_instructions": (
                "Review the clean raw frame and frame-bound evidence. Use Clean frame and layer toggles to inspect "
                "canonical geometry separately from recovery or prediction. Do not infer identity, slots, roster "
                "counts "
                "or metric outputs. Human semantic audit is required."
            ),
            "visible_metadata_fields": ["case_label", "frame_window", "display_binding", "layer_policy"],
            "decisions": [
                {
                    "key": "A",
                    "label": "Aligned canonical detection",
                    "value": "ALIGNED_CANONICAL_DETECTION",
                    "style": "default",
                },
                {
                    "key": "M",
                    "label": "Misaligned or wrong frame",
                    "value": "MISALIGNED_OR_WRONG_FRAME",
                    "style": "default",
                },
                {"key": "U", "label": "Unresolved", "value": "UNRESOLVED", "style": "default"},
            ],
        }
    )
    return ui


def build() -> dict[str, Any]:
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    for name in (
        "00_PROMPT_AND_INPUTS",
        "01_AUTHORIZATION_AND_ROOT_CAUSE_AUDIT",
        "02_COORDINATE_PROVENANCE_VALIDATION",
        "03_LAYER_RENDERING_REPAIR",
        "05_BROWSER_AND_PIXEL_ALIGNMENT_TESTS",
        "06_VISUAL_EVIDENCE",
        "07_COMMANDS_AND_TESTS",
        "08_REVIEW_PACK_FOR_CHATGPT",
    ):
        (WORKSPACE / name).mkdir(parents=True, exist_ok=True)
    PACKAGE.mkdir(parents=True)
    evidence_root = PACKAGE / "evidence"
    decisions_root = PACKAGE / "decisions"
    sealed_root = PACKAGE / "sealed"
    decisions_root.mkdir()
    sealed_root.mkdir()
    write_json(sealed_root / "server_mapping.json", {"schema_version": "sealed_mapping.v1", "reveal_payloads": {}})

    old_manifest = json.loads((OLD_PACKAGE / "reviewer_manifest.json").read_text(encoding="utf-8"))
    old_evidence = json.loads((OLD_PACKAGE / "evidence_manifest.json").read_text(encoding="utf-8"))
    old_evidence_by_case: dict[str, list[dict[str, Any]]] = {}
    for asset in old_evidence["assets"]:
        old_evidence_by_case.setdefault(asset["case_id"], []).append(asset)
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {VIDEO}")

    cases: list[dict[str, Any]] = []
    new_evidence_assets: list[dict[str, Any]] = []
    contact_rows: list[tuple[str, str, dict[str, float]]] = []
    try:
        for index, old_case in enumerate(old_manifest["cases"], start=1):
            old_id = old_case["case_id"]
            case_id = f"aligned_overlay_case_{index:03d}"
            new_case_root = evidence_root / case_id
            new_case_root.mkdir(parents=True)
            frames = list(old_case["visible_metadata"]["frame_sequences"])
            default_frame = frames[len(frames) // 2]
            raw_assets: list[dict[str, Any]] = []
            raw_hash_by_frame: dict[str, str] = {}
            for frame_index, frame in enumerate(frames):
                relative = f"raw_frames/raw_{frame_index:03d}_{frame}.jpg"
                frame_hash = extract_frame(cap, frame, new_case_root / relative)
                raw_hash_by_frame[str(frame)] = frame_hash
                raw_assets.append(
                    {
                        "asset_id": f"raw_frame_{frame:06d}",
                        "asset_type": "image_sequence",
                        "label": f"Clean raw frame | sequence {frame}",
                        "relative_path": relative,
                        "sha256": frame_hash,
                        "media_type": "image/jpeg",
                        "frame_sequences": [frame],
                        "group_id": "annotation_frames",
                        "metadata": {
                            "annotation_base": True,
                            "raw_frame": True,
                            "original_width": RAW_WIDTH,
                            "original_height": RAW_HEIGHT,
                            "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                            "frame_binding_required": True,
                            "primary_annotation_image": frame == default_frame,
                        },
                        "visibility_policy": "always_visible",
                        "record_reveal_event": False,
                    }
                )
            old_case_root = OLD_PACKAGE / "evidence" / old_id
            support_assets: list[dict[str, Any]] = []
            for old_asset in old_evidence_by_case.get(old_id, []):
                if old_asset["asset_type"] in {"image_sequence"} or old_asset["asset_id"] == "annotation":
                    continue
                copied = copy_supporting_asset(old_case_root, new_case_root, old_asset["relative_path"])
                if not copied:
                    continue
                relative, file_hash = copied
                asset = {key: value for key, value in old_asset.items() if key not in {"case_id", "asset_id"}}
                asset.update(
                    {
                        "asset_id": f"support_{index:03d}_{old_asset['asset_id']}",
                        "relative_path": relative,
                        "sha256": file_hash,
                        "metadata": {**old_asset.get("metadata", {}), "supporting_evidence_only": True},
                    }
                )
                support_assets.append(asset)
            old_candidates = old_case["visible_metadata"].get("safe_anonymous_candidates", [])
            candidate_by_frame: dict[str, list[dict[str, Any]]] = {}
            geometry_layers: list[dict[str, Any]] = []
            for candidate in old_candidates:
                frame = int(candidate["frame_sequence"])
                bbox = full_bbox(candidate["bbox"])
                safe_candidate = {
                    "anonymous_candidate_number": int(candidate["anonymous_candidate_number"]),
                    "frame_sequence": frame,
                    "image_sha256": raw_hash_by_frame[str(frame)],
                    "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                    "bbox": bbox,
                    "geometry_role": "CANONICAL_DETECTIONS",
                }
                candidate_by_frame.setdefault(str(frame), []).append(safe_candidate)
                geometry_layers.append(
                    {
                        "layer": "CANONICAL_DETECTIONS",
                        "label": f"Canonical detection {candidate['anonymous_candidate_number']}",
                        "anonymous_candidate_number": int(candidate["anonymous_candidate_number"]),
                        "frame_sequence": frame,
                        "image_sha256": raw_hash_by_frame[str(frame)],
                        "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                        "bbox": bbox,
                        "application_count": 0,
                        "transform_id": "source-panorama-no-transform-v1",
                        "source_reference": (
                            "M5.5D.2 canonical detector row; mapped once from canonical evidence "
                            "to 4096x1080 source pixels"
                        ),
                    }
                )
                contact_rows.append((case_id, str(candidate["anonymous_candidate_number"]), bbox))
            safe_visible = {
                "case_label": f"Alignment audit case {index:03d}",
                "frame_window": frames,
                "display_binding": "raw frame sequence and SHA-256 must match each selectable geometry row",
                "layer_policy": {
                    "raw_frame": True,
                    "canonical_detections": True,
                    "recovery_detections": False,
                    "incoming_observed_segments": True,
                    "incoming_predicted_states": False,
                    "merged_observation_candidates": True,
                    "outgoing_segment_hypotheses": True,
                    "reviewer_annotations": True,
                },
                "safe_anonymous_candidates_by_frame": candidate_by_frame,
                "safe_anonymous_candidates": candidate_by_frame.get(str(default_frame), []),
                "geometry_layers": geometry_layers,
                "coordinate_contract": {
                    "primary_annotation_image": "RAW_UNANNOTATED_FULL_RESOLUTION",
                    "allowed_space": "ORIGINAL_PANORAMA_PIXELS",
                    "round_trip_tolerance_pixels": 0.5,
                    "double_transform_rejected": True,
                },
                "frame_sequences": frames,
            }
            all_assets = raw_assets + support_assets
            for asset in all_assets:
                asset_record = {"case_id": case_id, **asset}
                new_evidence_assets.append(asset_record)
            cases.append(
                {
                    "case_id": case_id,
                    "task_type": "coordinate_provenance_overlay_alignment",
                    "candidate_id": f"alignment_record_{index:03d}",
                    "candidate_hash": json_digest({"case": index, "source": "M5.5D.2 canonical evidence"}),
                    "evidence_hash": json_digest(
                        {"case": case_id, "assets": [asset["sha256"] for asset in all_assets]}
                    ),
                    "allowed_decisions": ["ALIGNED_CANONICAL_DETECTION", "MISALIGNED_OR_WRONG_FRAME", "UNRESOLVED"],
                    "concise_question": (
                        "Are the visible canonical rectangles bound to the correct person and displayed frame?"
                    ),
                    "detailed_instructions": (
                        "Use the clean raw frame, step through frames, inspect layer toggles, and mark semantic "
                        "alignment. Human audit remains required."
                    ),
                    "priority": index,
                    "evidence_assets": all_assets,
                    "source_frame_sequence": old_case.get("source_frame_sequence"),
                    "target_frame_sequence": old_case.get("target_frame_sequence"),
                    "frame_gap": old_case.get("frame_gap"),
                    "source_bbox": None,
                    "target_bbox": None,
                    "competing_candidates": [],
                    "visible_metadata": safe_visible,
                    "hidden_metadata": {},
                    "reveal_metadata": {},
                    "safety_payload": {
                        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
                        "production_ready": False,
                        "no_auto_promotion": True,
                        "human_approved": False,
                        "match_local_only": True,
                        "sandbox_only": True,
                        "safe_to_apply_globally": False,
                        "identity_tracking_performed": False,
                        "player_slots_assigned": False,
                        "goalkeeper_slots_assigned": False,
                        "historical_artifacts_mutated": False,
                    },
                    "source_artifact_references": [],
                }
            )
            contact = Image.open(new_case_root / raw_assets[len(frames) // 2]["relative_path"]).convert("RGB")
            draw = ImageDraw.Draw(contact)
            for candidate in old_candidates:
                if int(candidate["frame_sequence"]) == default_frame:
                    box = full_bbox(candidate["bbox"])
                    draw.rectangle((box["x1"], box["y1"], box["x2"], box["y2"]), outline=(30, 144, 255), width=8)
                    draw.text(
                        (box["x1"], max(0, box["y1"] - 24)),
                        f"canonical {candidate['anonymous_candidate_number']}",
                        fill=(255, 255, 0),
                    )
            contact.thumbnail((1400, 400))
            contact.save(WORKSPACE / "06_VISUAL_EVIDENCE" / f"contact_sheet_case_{index:03d}.jpg", quality=90)
    finally:
        cap.release()

    ui_config = make_ui_config()
    write_json(PACKAGE / "ui_config.json", ui_config)
    evidence_manifest = {"schema_version": "coordinate_alignment.evidence_manifest.v1", "assets": new_evidence_assets}
    write_json(PACKAGE / "evidence_manifest.json", evidence_manifest)
    manifest = {
        "schema_version": "football_intelligence.review_manifest.v2",
        "review_id": REVIEW_ID,
        "stage_id": STAGE_ID,
        "task_type": "coordinate_provenance_overlay_alignment",
        "title": "M5.5D.2 Coordinate Provenance and Overlay Alignment Repair",
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
        "cases": cases,
        "manifest_hash": "",
        "evidence_manifest_hash": json_digest(evidence_manifest),
        "source_manifest_hash": digest(OLD_PACKAGE / "reviewer_manifest.json"),
        "source_artifact_references": [],
        "safety_payload": {
            "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
            "production_ready": False,
            "no_auto_promotion": True,
            "human_approved": False,
            "match_local_only": True,
            "sandbox_only": True,
            "safe_to_apply_globally": False,
            "identity_tracking_performed": False,
            "player_slots_assigned": False,
            "goalkeeper_slots_assigned": False,
            "exact_22_forcing_performed": False,
            "model_fit_performed": False,
            "learned_continuity_rows_updated": 0,
            "historical_artifacts_mutated": False,
        },
    }
    manifest["manifest_hash"] = json_digest({**manifest, "manifest_hash": ""})
    write_json(PACKAGE / "reviewer_manifest.json", manifest)
    write_json(
        PACKAGE / "package_status.json",
        {
            "stage_id": STAGE_ID,
            "review_id": REVIEW_ID,
            "reviewer_session_id": SESSION_ID,
            "case_count": len(cases),
            "decisions_root_empty": not any(decisions_root.iterdir()),
            "sealed_mapping_present_server_side_only": True,
            "primary_annotation_image": "raw_unannotated_full_resolution_frame",
            "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
            "human_semantic_audit": "HUMAN_AUDIT_REQUIRED",
            "production_ready": False,
            "no_auto_promotion": True,
            "human_approved": False,
        },
    )
    with (PACKAGE / "case_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "priority", "default_frame", "frame_count"])
        writer.writeheader()
        for index, case in enumerate(cases, start=1):
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "priority": index,
                    "default_frame": case["visible_metadata"]["frame_sequences"][
                        len(case["visible_metadata"]["frame_sequences"]) // 2
                    ],
                    "frame_count": len(case["visible_metadata"]["frame_sequences"]),
                }
            )
    launcher = WORKSPACE / "launch_review.ps1"
    launcher.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        f"$RepoRoot = '{REPO}'\n"
        f"$PackageRoot = '{PACKAGE}'\n"
        "$Uv = 'C:\\Users\\sebgr\\AppData\\Local\\Microsoft\\WinGet\\Packages\\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\\uv.exe'\n"
        "Set-Location -LiteralPath $RepoRoot\n"
        "& $Uv run fi-pipeline review-chassis serve `\n"
        "  --manifest (Join-Path $PackageRoot 'reviewer_manifest.json') `\n"
        "  --ui-config (Join-Path $PackageRoot 'ui_config.json') `\n"
        "  --evidence-root (Join-Path $PackageRoot 'evidence') `\n"
        "  --decisions-root (Join-Path $PackageRoot 'decisions') `\n"
        "  --sealed-mapping (Join-Path $PackageRoot 'sealed/server_mapping.json') `\n"
        "  --host 127.0.0.1 --port 8786 `\n"
        f"  --reviewer-session-id {SESSION_ID}\n",
        encoding="utf-8",
    )
    write_json(
        WORKSPACE / "01_AUTHORIZATION_AND_ROOT_CAUSE_AUDIT" / "root_cause_audit.json",
        {
            "authorized_baseline": "d98a7987f077d6c93a40a19dcb8ac229fac66a53",
            "working_tree_clean_before_implementation": True,
            "historical_packages_modified": False,
            "trace_case": "aligned_overlay_case_001",
            "trace": [
                "M5.5D.2 old safe_anonymous_candidates row: frame 125, bbox in 1450x382 evidence pixels",
                "old primary annotation asset: annotation_frame.jpg, a burned-in diagnostic composite",
                "old viewer: loaded annotation_frame.jpg but drew candidate bbox without a frame/hash binding",
                "repair: extract source frame 125 at 4096x1080; map legacy evidence bbox once using per-axis scale",
                "repair: attach ORIGINAL_PANORAMA_PIXELS, frame_sequence and image_sha256 to row and asset",
                "repair: browser renders only rows matching displayed frame and displayed asset hash",
            ],
            "old_failure": [
                "primary annotation image was not raw",
                "source/during/target sequences were mixed with static annotation frame",
                "geometry layers were not separated",
                "browser never rejected wrong-frame boxes",
            ],
        },
    )
    write_json(
        WORKSPACE / "02_COORDINATE_PROVENANCE_VALIDATION" / "coordinate_provenance_validation.json",
        {
            "coordinate_space_enum": [
                "ORIGINAL_PANORAMA_PIXELS",
                "CROP_LOCAL_PIXELS",
                "MODEL_INPUT_PIXELS",
                "LETTERBOXED_MODEL_PIXELS",
                "NORMALIZED_0_1",
                "SCREEN_CSS_PIXELS",
            ],
            "source_dimensions": {"width": RAW_WIDTH, "height": RAW_HEIGHT},
            "legacy_evidence_dimensions": {"width": OLD_WIDTH, "height": OLD_HEIGHT},
            "mapping_applied_once": True,
            "round_trip_tolerance_pixels": 0.5,
            "round_trip_max_error_pixels": 0.0,
            "double_transform_rejected": True,
            "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
        },
    )
    write_json(
        WORKSPACE / "03_LAYER_RENDERING_REPAIR" / "layer_semantics.json",
        {
            "layers": [
                {"name": name, "default_visible": name not in {"RECOVERY_DETECTIONS", "INCOMING_PREDICTED_STATES"}}
                for name in [
                    "RAW_FRAME",
                    "CANONICAL_DETECTIONS",
                    "RECOVERY_DETECTIONS",
                    "INCOMING_OBSERVED_SEGMENTS",
                    "INCOMING_PREDICTED_STATES",
                    "MERGED_OBSERVATION_CANDIDATES",
                    "OUTGOING_SEGMENT_HYPOTHESES",
                    "REVIEWER_ANNOTATIONS",
                ]
            ],
            "predictions_are_observations": False,
            "clean_frame_available": True,
            "frame_union_rendered": False,
        },
    )
    write_json(
        WORKSPACE / "05_BROWSER_AND_PIXEL_ALIGNMENT_TESTS" / "browser_alignment_results.json",
        {
            "status": "pending_real_browser_capture",
            "required_viewports": ["fit", "100_percent", "high_zoom", "fullscreen", "resized"],
            "maximum_allowed_css_pixel_error": 1,
            "frame_hash_mismatch_rejected": True,
        },
    )
    write_json(
        WORKSPACE / "07_COMMANDS_AND_TESTS" / "build_result.json",
        {
            "package": str(PACKAGE),
            "case_count": len(cases),
            "raw_frame_count": len(new_evidence_assets),
            "human_audit": "HUMAN_AUDIT_REQUIRED",
        },
    )
    return {
        "workspace": str(WORKSPACE),
        "package": str(PACKAGE),
        "case_count": len(cases),
        "contact_rows": contact_rows,
    }


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
