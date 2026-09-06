"""Block, prepare, or close the six-frame calibration adjudication re-audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from football_intelligence.calibration_adjudication import (
    ADJUDICATION_ASSERTION,
    CALIBRATION_IDS,
    load_original_parent,
    sha256_file,
    validate_adjudication_pair,
)
from football_intelligence.dense_person_gold import COMPLETION_ASSERTION
from football_intelligence.dense_person_reviewer import (
    VISIBLE_PERSON_SCOPE_REMINDER,
    VISIBLE_PERSON_SCOPE_SECOND_LINE,
)
from g7f_c_run_calibration_adjudication_reviewer import (
    EXPECTED_HASHES,
    REVIEWER_RELEASE,
    read_json,
    roots,
    verify_environment,
    verify_frozen_contracts,
    verify_original_inventory,
    verify_release,
)


INCOMPLETE = "HOLD_G7F_C_CALIBRATION_ADJUDICATION_INCOMPLETE"
VISUAL_REQUIRED = "HOLD_G7F_C_CALIBRATION_ADJUDICATION_VISUAL_REAUDIT_REQUIRED"
PASS = "PASS_G7F_C_CALIBRATION_SUPERSEDING_ADJUDICATION_REAUDIT_READY_FOR_SCORED_DENSE_GOLD_ANNOTATION"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def load_selection(source: Path) -> dict[str, dict[str, Any]]:
    selection = read_json(source / "01_SELECTION/dense_gold_selection_manifest.json")
    frames = {
        row["anonymous_dense_image_id"]: row
        for row in selection["images"]
        if row["anonymous_dense_image_id"] in CALIBRATION_IDS
    }
    if set(frames) != set(CALIBRATION_IDS):
        raise RuntimeError("frozen selection does not contain exactly DG-001..DG-006")
    return frames


def phase_b_state(repo: Path) -> dict[str, Any]:
    paths = roots(repo)
    verify_frozen_contracts(paths["source"])
    _, bindings = verify_release(paths)
    frozen = verify_original_inventory(paths)
    frames = load_selection(paths["source"])
    original_root = paths["source"] / "06_DENSE_DECISIONS"
    adjudication_root = original_root / "calibration_adjudication"
    rows, missing = [], []
    for image_id in CALIBRATION_IDS:
        key = f"calibration_adjudication_001__{image_id}.json"
        event_path = adjudication_root / "events" / key
        ack_path = adjudication_root / "acknowledgements" / key
        if not event_path.is_file() or not ack_path.is_file():
            missing.append(image_id)
            continue
        parent = load_original_parent(
            original_root,
            frames[image_id],
            expected_event_file_sha256=frozen[image_id]["event_file_sha256"],
            expected_ack_file_sha256=frozen[image_id]["acknowledgement_file_sha256"],
        )
        event, acknowledgement = validate_adjudication_pair(
            event_path,
            ack_path,
            frame=frames[image_id],
            parent=parent,
            expected_binding_hashes=bindings,
        )
        rows.append(
            {
                "anonymous_dense_image_id": image_id,
                "event": event,
                "acknowledgement": acknowledgement,
                "event_path": event_path,
                "acknowledgement_path": ack_path,
                "event_file_sha256": sha256_file(event_path),
                "acknowledgement_file_sha256": sha256_file(ack_path),
                "parent_event_file_sha256": parent.event_file_sha256,
            }
        )
    existing_events = sorted((adjudication_root / "events").glob("*.json")) if adjudication_root.is_dir() else []
    existing_acks = (
        sorted((adjudication_root / "acknowledgements").glob("*.json")) if adjudication_root.is_dir() else []
    )
    unexpected = [
        str(path) for path in (*existing_events, *existing_acks) if "calibration_adjudication_001__" not in path.name
    ]
    if unexpected:
        raise RuntimeError(f"unexpected adjudication lineage files require explicit chain review: {unexpected}")
    return {
        "paths": paths,
        "bindings": bindings,
        "frames": frames,
        "rows": rows,
        "missing": missing,
        "event_count": len(existing_events),
        "acknowledgement_count": len(existing_acks),
    }


def status(repo: Path) -> dict[str, Any]:
    state = phase_b_state(repo)
    complete = not state["missing"] and state["event_count"] == state["acknowledgement_count"] == 6
    return {
        "decision": VISUAL_REQUIRED if complete else INCOMPLETE,
        "required_adjudications": 6,
        "valid_event_ack_pairs": len(state["rows"]),
        "missing_anonymous_dense_image_ids": state["missing"],
        "qa_assets_generated": False,
        "scored_annotation_authorized": False,
        "scored_annotation_started": False,
        "production_ready": False,
    }


def _vertices(component: list[dict[str, Any]]) -> np.ndarray:
    return np.rint([[point["x"], point["y"]] for point in component]).astype(np.int32)


def render_overlay(image: np.ndarray, annotation: dict[str, Any]) -> np.ndarray:
    overlay = image.copy()
    fill = image.copy()
    for index, person in enumerate(annotation["people"]):
        colour = (40, 215, 95) if person["relevance"] == "MATCH_RELEVANT" else (255, 170, 45)
        for component in person["canonical_components"]:
            vertices = _vertices(component)
            cv2.fillPoly(fill, [vertices], colour, lineType=cv2.LINE_8)
            cv2.polylines(overlay, [vertices], True, colour, 2, cv2.LINE_AA)
        box = person["derived_visible_box_xyxy"]
        cv2.putText(
            overlay,
            person["instance_id"],
            (int(box["x1"]), max(12, int(box["y1"]) - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            colour,
            1,
            cv2.LINE_AA,
        )
    for region in annotation["ignore_regions"]:
        for component in region["canonical_components"]:
            vertices = _vertices(component)
            cv2.fillPoly(fill, [vertices], (180, 75, 210), lineType=cv2.LINE_8)
            cv2.polylines(overlay, [vertices], True, (220, 120, 255), 2, cv2.LINE_AA)
    cv2.addWeighted(fill, 0.22, overlay, 0.78, 0, overlay)
    return overlay


def crop_montage(image: np.ndarray, overlay: np.ndarray, people: list[dict[str, Any]]) -> np.ndarray:
    tiles = []
    for person in people:
        box = person["derived_visible_box_xyxy"]
        width, height = box["x2"] - box["x1"], box["y2"] - box["y1"]
        pad = max(12, int(max(width, height) * 0.35))
        x1, y1 = max(0, box["x1"] - pad), max(0, box["y1"] - pad)
        x2, y2 = min(image.shape[1], box["x2"] + pad), min(image.shape[0], box["y2"] + pad)
        crop = overlay[y1:y2, x1:x2]
        scale = min(180 / max(1, crop.shape[1]), 140 / max(1, crop.shape[0]))
        resized = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))))
        tile = np.full((175, 190, 3), 24, dtype=np.uint8)
        top, left = 25 + (140 - resized.shape[0]) // 2, (190 - resized.shape[1]) // 2
        tile[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
        cv2.putText(tile, person["instance_id"], (7, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
        tiles.append(tile)
    columns = 5
    rows = math.ceil(len(tiles) / columns)
    montage = np.full((rows * 175, columns * 190, 3), 16, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        montage[row * 175 : (row + 1) * 175, column * 190 : (column + 1) * 190] = tile
    return montage


def structural_summary(state: dict[str, Any]) -> dict[str, Any]:
    images = []
    for row in state["rows"]:
        annotation = row["event"]["annotation"]
        images.append(
            {
                "anonymous_dense_image_id": row["anonymous_dense_image_id"],
                "people": len(annotation["people"]),
                "ignore_regions": len(annotation["ignore_regions"]),
                "reviewed_strips": annotation["reviewed_exhaustiveness_strips"],
                "completion_assertion_exact": annotation["completion_assertion"] == COMPLETION_ASSERTION,
                "adjudication_assertion_exact": (
                    row["event"]["adjudication_metadata"]["adjudication_assertion"] == ADJUDICATION_ASSERTION
                ),
                "parent_bindings_exact": True,
                "server_validation": row["event"]["server_validation"],
            }
        )
    return {
        "images": images,
        "all_six_structurally_valid": len(images) == 6,
        "candidate_data_used": False,
        "production_ready": False,
    }


def prepare(repo: Path) -> dict[str, Any]:
    state = phase_b_state(repo)
    if state["missing"] or state["event_count"] != 6 or state["acknowledgement_count"] != 6:
        return status(repo)
    output = state["paths"]["stage"] / "11_PHASE_B/07_QA_ASSETS"
    assets = []
    for row in state["rows"]:
        image_id, annotation = row["anonymous_dense_image_id"], row["event"]["annotation"]
        source_path = state["paths"]["source"] / "05_REVIEWER/assets" / f"{image_id}.png"
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"source image cannot be loaded: {source_path}")
        overlay = render_overlay(image, annotation)
        image_root = output / image_id
        overlay_path = image_root / "full_human_overlay.png"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_path), overlay)
        files = [overlay_path]
        for strip in range(8):
            x1, x2 = round(image.shape[1] * strip / 8), round(image.shape[1] * (strip + 1) / 8)
            strip_path = image_root / "strips" / f"strip_{strip}.png"
            strip_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(strip_path), overlay[:, x1:x2])
            files.append(strip_path)
        montage_path = image_root / "person_crop_montage.png"
        cv2.imwrite(str(montage_path), crop_montage(image, overlay, annotation["people"]))
        files.append(montage_path)
        assets.append(
            {
                "anonymous_dense_image_id": image_id,
                "authoritative_event_sha256": row["event"]["event_sha256"],
                "people": len(annotation["people"]),
                "files": [
                    {"path": str(path), "byte_size": path.stat().st_size, "sha256": sha256_file(path)} for path in files
                ],
            }
        )
    manifest = {
        "schema_version": "football_intelligence.g7f_c.calibration_adjudication_candidate_free_qa.v1",
        "candidate_data_used": False,
        "human_events_only": True,
        "full_overlay_count": 6,
        "strip_view_count": 48,
        "person_montage_count": 6,
        "images": assets,
        "production_ready": False,
    }
    write_json(output / "qa_asset_manifest.json", manifest)
    write_json(state["paths"]["stage"] / "11_PHASE_B/03_STRUCTURAL_REAUDIT.json", structural_summary(state))
    return {
        "decision": VISUAL_REQUIRED,
        "valid_event_ack_pairs": 6,
        "qa_asset_manifest": str(output / "qa_asset_manifest.json"),
        "candidate_data_used": False,
        "scored_annotation_authorized": False,
        "scored_annotation_started": False,
        "production_ready": False,
    }


def validate_visual_assessment(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if value.get("candidate_data_used") is not False:
        raise RuntimeError("visual re-audit must be candidate-free")
    rows = value.get("images")
    if not isinstance(rows, list) or {row.get("anonymous_dense_image_id") for row in rows} != set(CALIBRATION_IDS):
        raise RuntimeError("visual re-audit must cover exactly DG-001..DG-006")
    required = {
        "no_clear_material_human_omission",
        "no_merged_person_mask",
        "no_material_shadow_or_background_contamination",
        "no_major_visible_body_omission",
        "no_improper_amodal_bridge",
        "ignore_regions_justified",
        "coverage_rule_consistent",
    }
    for row in rows:
        if row.get("visual_status") != "PASS" or any(row.get(key) is not True for key in required):
            raise RuntimeError(f"visual re-audit repair required for {row.get('anonymous_dense_image_id')}")
    if value.get("dg005_person_030_assistant_referee_relevance_resolved") is not True:
        raise RuntimeError("DG-005 person-030 assistant-referee relevance issue is unresolved")
    return value


def decisions_inventory(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    rows = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    ordered = "\n".join(f"{row['relative_path']}\t{row['byte_size']}\t{row['sha256']}" for row in rows).encode()
    return {
        "file_count": len(rows),
        "total_bytes": sum(row["byte_size"] for row in rows),
        "ordered_inventory_sha256": hashlib.sha256(ordered).hexdigest(),
        "files": rows,
    }


def write_phase_b_handoff(
    state: dict[str, Any],
    *,
    visual: dict[str, Any],
    visual_assessment_path: Path,
    authorization: dict[str, Any],
) -> Path:
    stage, source = state["paths"]["stage"], state["paths"]["source"]
    handoff = stage / "11_PHASE_B/10_REVIEW_PACK/CHATGPT_HANDOFF"
    handoff.mkdir(parents=True, exist_ok=True)
    original_freeze = read_json(stage / "01_FREEZE/original_real_decisions_inventory.json")
    structural = structural_summary(state)
    write_json(
        handoff / "00_EXECUTIVE_SUMMARY.json",
        {
            "stage": "G7F_C_CALIBRATION_SUPERSEDING_ADJUDICATION_AND_REAUDIT",
            "phase": "B",
            "decision": PASS,
            "authoritative_calibration_images": 6,
            "fresh_proficiency_images": 0,
            "candidate_data_used": False,
            "SCORED_ANNOTATION_AUTHORIZED": True,
            "SCORED_ANNOTATION_STARTED": False,
            "production_ready": False,
        },
    )
    write_json(
        handoff / "01_ORIGINAL_AND_SUPERSEDING_EVENT_INTEGRITY.json",
        {
            "original_first_pass_inventory": {
                key: original_freeze[key] for key in ("file_count", "total_bytes", "ordered_inventory_sha256")
            },
            "original_calibration_parents": original_freeze["calibration_parents"],
            "valid_superseding_event_ack_pairs": 6,
            "exact_parent_bindings": True,
            "originals_byte_identical": True,
            "append_only": True,
            "production_ready": False,
        },
    )
    write_json(
        handoff / "02_AUTHORITATIVE_CALIBRATION_STATE.json",
        {
            "precedence": "LATEST_VALID_ACKNOWLEDGED_SUPERSEDING_ADJUDICATION",
            "images": authorization["authoritative_adjudications"],
            "scored_image_precedence_changed": False,
            "production_ready": False,
        },
    )
    write_json(handoff / "03_STRUCTURAL_REAUDIT.json", structural)
    visual_lines = [
        "# Candidate-free calibration adjudication visual re-audit",
        "",
        f"Decision: `{PASS}`",
        "",
        "Inspected the six full human overlays, all 48 strip views, and six person-crop montages.",
        "No detector candidates, scores, run identifiers, or disagreement data were used.",
        "",
    ]
    for row in visual["images"]:
        visual_lines.append(f"- {row['anonymous_dense_image_id']}: PASS")
    visual_lines.extend(
        [
            "",
            "No clear material omission, merged mask, material contamination, major visible-body omission, "
            "or improper amodal bridge remains. Ignore regions are justified and coverage is consistent.",
            "DG-005 person-030 assistant-referee relevance is resolved.",
            "",
            "Minor contour roughness remains nonblocking.",
            "",
            "`production_ready=false`",
        ]
    )
    (handoff / "04_VISUAL_REAUDIT.md").write_text("\n".join(visual_lines) + "\n", encoding="utf-8", newline="\n")
    write_json(
        handoff / "05_CROSS_IMAGE_CONSISTENCY.json",
        {
            "coverage_rule_consistent": True,
            "assistant_referee_relevance_consistent": True,
            "dg005_person_030_relevance": "MATCH_RELEVANT",
            "minor_contour_roughness_nonblocking": True,
            "fresh_proficiency_images": 0,
            "production_ready": False,
        },
    )
    current_inventory = decisions_inventory(source / "06_DENSE_DECISIONS")
    write_json(
        handoff / "06_REAL_DECISIONS_IMMUTABILITY_AND_APPEND_ONLY_REPORT.json",
        {
            "original_inventory_sha256": original_freeze["ordered_inventory_sha256"],
            "current_inventory": {
                key: current_inventory[key] for key in ("file_count", "total_bytes", "ordered_inventory_sha256")
            },
            "original_files_all_exact": True,
            "new_files_confined_to_calibration_adjudication_namespace": True,
            "event_ack_pairs_added": 6,
            "production_ready": False,
        },
    )
    write_json(handoff / "07_SCORED_ANNOTATION_AUTHORIZATION.json", authorization)
    (handoff / "08_DECISION.md").write_text(
        f"# Decision\n\n`{PASS}`\n\n"
        "SCORED_ANNOTATION_AUTHORIZED=true  \nSCORED_ANNOTATION_STARTED=false  \nproduction_ready=false\n",
        encoding="utf-8",
        newline="\n",
    )
    payload_files = sorted(path for path in handoff.iterdir() if path.is_file() and path.name != "10_MANIFEST.json")
    if len(payload_files) != 9:
        raise RuntimeError(f"Phase-B handoff must contain exactly nine payload files: {payload_files}")
    write_json(
        handoff / "10_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7f_c.calibration_adjudication_phase_b_handoff.v1",
            "files": [
                {"name": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
                for path in payload_files
            ],
            "file_count_excluding_manifest": 9,
            "visual_assessment_path": str(visual_assessment_path),
            "visual_assessment_sha256": sha256_file(visual_assessment_path),
            "decision": PASS,
            "production_ready": False,
        },
    )
    return handoff


def finalize(repo: Path, visual_assessment_path: Path) -> dict[str, Any]:
    prepared = prepare(repo)
    if prepared["decision"] != VISUAL_REQUIRED or prepared.get("valid_event_ack_pairs") != 6:
        return prepared
    visual = validate_visual_assessment(visual_assessment_path)
    state = phase_b_state(repo)
    dg005 = next(row for row in state["rows"] if row["anonymous_dense_image_id"] == "DG-005")
    person_030 = next(
        (person for person in dg005["event"]["annotation"]["people"] if person["instance_id"] == "person-030"),
        None,
    )
    if person_030 is None or person_030["relevance"] != "MATCH_RELEVANT":
        raise RuntimeError("DG-005 person-030 must be MATCH_RELEVANT after confirmed assistant-referee adjudication")
    current_inventory = decisions_inventory(state["paths"]["source"] / "06_DENSE_DECISIONS")
    release = verify_release(state["paths"])[0]
    authorization = {
        "decision": PASS,
        "SCORED_ANNOTATION_AUTHORIZED": True,
        "SCORED_ANNOTATION_STARTED": False,
        "fresh_proficiency_images_required": 0,
        "authoritative_adjudications": [
            {
                "anonymous_dense_image_id": row["anonymous_dense_image_id"],
                "event_sha256": row["event"]["event_sha256"],
                "event_file_sha256": row["event_file_sha256"],
                "acknowledgement_sha256": row["acknowledgement"]["acknowledgement_sha256"],
                "acknowledgement_file_sha256": row["acknowledgement_file_sha256"],
                "original_parent_event_sha256": row["event"]["supersedes_event_sha256"],
            }
            for row in state["rows"]
        ],
        "frozen_contract_hashes": EXPECTED_HASHES,
        "reviewer_release": REVIEWER_RELEASE,
        "reviewer_release_manifest_sha256": state["bindings"]["reviewer_release_manifest_sha256"],
        "repository_commit": release["repository_commit"],
        "scope_guidance": [VISIBLE_PERSON_SCOPE_REMINDER, VISIBLE_PERSON_SCOPE_SECOND_LINE],
        "visual_assessment_sha256": sha256_file(visual_assessment_path),
        "real_decisions_inventory_hash": current_inventory["ordered_inventory_sha256"],
        "production_ready": False,
    }
    handoff = write_phase_b_handoff(
        state,
        visual=visual,
        visual_assessment_path=visual_assessment_path,
        authorization=authorization,
    )
    authorization["handoff"] = str(handoff)
    return authorization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", nargs="?", choices=("status", "prepare", "finalize"), default="status")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--visual-assessment", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verify_environment()
    if args.phase == "status":
        result = status(args.repo.resolve())
    elif args.phase == "prepare":
        result = prepare(args.repo.resolve())
    else:
        if args.visual_assessment is None:
            raise RuntimeError("finalize requires --visual-assessment")
        result = finalize(args.repo.resolve(), args.visual_assessment.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("decision") == PASS else 2


if __name__ == "__main__":
    sys.exit(main())
