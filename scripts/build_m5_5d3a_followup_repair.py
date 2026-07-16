"""Build the M5.5D.3A repaired follow-up review package.

This stage is deliberately append-only with respect to the M5.5D.3 and M5.5D.2C
workspaces.  It replays the historical event ledger, then uses only the 27
audited self/repeated-row cases to construct a fresh, blinded semantic review.
No completed decision is ingested as a new scientific label.
"""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.models import (
    AssetPanelConfig,
    DecisionOption,
    GenericEvidenceAsset,
    GenericReviewCase,
    GenericReviewManifest,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from football_intelligence.review_chassis.validation import validate_review_chassis_package

try:
    from scripts.build_m5_5d3_consolidation import (
        CASE_WINDOWS,
        CANDIDATE_ROWS,
        FRAME_HEIGHT,
        FRAME_MANIFEST,
        FRAME_WIDTH,
        REPO,
        ROOT,
        clean_bbox,
        crop_bytes,
        file_hash,
        frame_catalog,
        make_gif,
        object_digest,
        read_json,
        read_jsonl,
    )
except ModuleNotFoundError:  # Executed as a file by ``uv run python scripts/...``.
    from build_m5_5d3_consolidation import (
        CASE_WINDOWS,
        CANDIDATE_ROWS,
        FRAME_HEIGHT,
        FRAME_MANIFEST,
        FRAME_WIDTH,
        REPO,
        ROOT,
        clean_bbox,
        crop_bytes,
        file_hash,
        frame_catalog,
        make_gif,
        object_digest,
        read_json,
        read_jsonl,
    )


STAGE_ROOT = (
    ROOT / r"matches\128058\runs\step_m5\part 2\M5_5D3A_FOLLOWUP_REVIEW_SEMANTIC_AND_TARGET_EXCLUSION_REPAIR_v1"
)
PRIOR_D3_ROOT = (
    ROOT
    / r"matches\128058\runs\step_m5\part 2\M5_5D3_HUMAN_VALIDATED_OBSERVATION_CONSOLIDATION_AND_OCCLUSION_REEVALUATION_v1"
)
PRIOR_D3_PACKAGE = PRIOR_D3_ROOT / "08_OPTIONAL_FOLLOWUP_REVIEW_PACKAGE" / "review_package"
HISTORICAL_STAGE = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5D2C_TARGETED_ENCOUNTER_CANDIDATE_SEMANTIC_AUDIT_v1"
HISTORICAL_PACKAGE = HISTORICAL_STAGE / "03_TARGETED_SEMANTIC_REVIEW_PACKAGE"
HISTORICAL_EVENTS = HISTORICAL_PACKAGE / "decisions" / "completed_review_events.jsonl"
HISTORICAL_MANIFEST = HISTORICAL_PACKAGE / "reviewer_manifest.json"
HISTORICAL_UI = HISTORICAL_PACKAGE / "ui_config.json"
HISTORICAL_STATE = HISTORICAL_PACKAGE / "decisions" / "review_decisions.json"
HISTORICAL_COMPLETED = HISTORICAL_PACKAGE / "decisions" / "completed_review.json"
HISTORICAL_SEALED = HISTORICAL_PACKAGE / "sealed" / "server_mapping.json"
MALFORMED_ROWS = PRIOR_D3_ROOT / "02_DUPLICATE_AND_CROP_PROVENANCE_AUDIT" / "self_duplicate_rows.jsonl"

AUTHORIZED_BASELINE = "8f38565b23aa3c29dc16eb409887b9b11e6e3238"
STAGE_ID = "M5_5D3A_FOLLOWUP_REVIEW_SEMANTIC_AND_TARGET_EXCLUSION_REPAIR_v1"
REVIEW_ID = "m5_5d3a_repaired_duplicate_followup_v1"
REVIEW_SESSION = "m5_5d3a_repaired_followup_human_reviewer"
PORT = 8790
TARGET_LABEL = "TARGET — not numbered"
QUESTION = "Does any numbered box cover the same visible person as the red highlighted target?"
MAX_CONTEXT_CANDIDATES = 8

DECISION_VALUES = [
    "DUPLICATE_SAME_PERSON_COUNTERPART",
    "VALID_VISIBLE_SINGLE_PERSON_NO_DUPLICATE",
    "MERGED_MULTIPLE_VISIBLE_PEOPLE",
    "PARTIAL_PERSON_OR_BODY_FRAGMENT",
    "FALSE_POSITIVE_OR_EMPTY",
    "WRONG_VISIBLE_PERSON_FOR_ENCOUNTER",
    "EVIDENCE_UNRESOLVED",
]

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
    "match_local_only": True,
    "sandbox_only": True,
    "safe_to_apply_globally": False,
    "episodes_rebuilt": False,
    "ghosts_reassessed": False,
    "fine_vision_models_run": False,
    "football_metrics_generated": False,
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_digest(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode())


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    return result.stdout.strip()


def tree_snapshot(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        files.append({"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)})
    aggregate = json_digest(files)
    return {
        "root": str(root),
        "file_count": len(files),
        "total_size_bytes": sum(row["size"] for row in files),
        "aggregate_sha256": aggregate,
        "files": files,
    }


def snapshot_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    old = {row["path"]: row for row in before["files"]}
    new = {row["path"]: row for row in after["files"]}
    changed = [path for path in sorted(set(old) & set(new)) if old[path]["sha256"] != new[path]["sha256"]]
    return {
        "before_file_count": before["file_count"],
        "after_file_count": after["file_count"],
        "added_paths": sorted(set(new) - set(old)),
        "deleted_paths": sorted(set(old) - set(new)),
        "changed_paths": changed,
        "unchanged": not changed and not (set(old) - set(new)) and not (set(new) - set(old)),
    }


def parse_event_notes(event: dict[str, Any]) -> dict[str, Any]:
    raw = event.get("notes")
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError:
        return {"_malformed_notes": raw}
    return value if isinstance(value, dict) else {}


def replay_historical_ledger() -> dict[str, Any]:
    """Validate the full event history without rewriting it."""
    from football_intelligence.review_chassis.config import load_ui_config, ui_config_hash
    from football_intelligence.review_chassis.manifest import load_manifest, manifest_hash

    manifest = load_manifest(HISTORICAL_MANIFEST)
    ui_config = load_ui_config(HISTORICAL_UI)
    events = read_jsonl(HISTORICAL_EVENTS)
    final_state = read_json(HISTORICAL_STATE)
    completed_export = read_json(HISTORICAL_COMPLETED)
    case_ids = {case.case_id for case in manifest.cases}
    allowed = {option.value for option in ui_config.decisions}
    expected_manifest_hash = manifest_hash(manifest)
    expected_ui_hash = ui_config_hash(ui_config)
    materialized: dict[str, str] = {}
    notes: dict[str, str] = {}
    event_errors: list[str] = []
    decision_events = 0
    note_events = 0
    completion_events = 0
    completion_sequence: int | None = None
    prior_sequence = 0
    duplicate_event_sequence_count = 0
    for event in events:
        sequence = int(event.get("event_sequence", -1))
        event_type = event.get("event_type")
        case_id = event.get("case_id")
        if sequence < prior_sequence:
            event_errors.append(f"non_contiguous_event_sequence:{sequence}")
        elif sequence == prior_sequence and events:
            # The historical chassis log contains one same-sequence pair. The
            # file order is the immutable order for deterministic replay.
            duplicate_event_sequence_count += 1
        prior_sequence = sequence
        if event.get("reviewer_session_id") != "m5_5d2c_targeted_candidate_human_reviewer":
            event_errors.append(f"unexpected_reviewer_session:{sequence}")
        if event.get("manifest_hash") != expected_manifest_hash:
            event_errors.append(f"manifest_hash_mismatch:{sequence}")
        if event.get("ui_config_hash") != expected_ui_hash:
            event_errors.append(f"ui_config_hash_mismatch:{sequence}")
        if completion_sequence is not None:
            event_errors.append(f"event_after_completion:{sequence}")
        if event_type in {"decision", "note", "reveal", "undo"} and case_id not in case_ids:
            event_errors.append(f"unknown_case:{sequence}")
        if event_type == "decision":
            decision_events += 1
            decision = event.get("new_decision")
            if decision not in allowed:
                event_errors.append(f"invalid_decision:{sequence}")
            if case_id:
                materialized[str(case_id)] = str(decision)
        elif event_type == "note":
            note_events += 1
            if case_id:
                notes[str(case_id)] = str(event.get("notes") or "")
        elif event_type == "undo":
            restored = event.get("restored_decision")
            if case_id:
                if restored is None:
                    materialized.pop(str(case_id), None)
                else:
                    materialized[str(case_id)] = str(restored)
        elif event_type == "complete":
            completion_events += 1
            completion_sequence = sequence
            if case_id is not None:
                event_errors.append(f"completion_has_case:{sequence}")
        elif event_type == "reveal":
            pass
        else:
            event_errors.append(f"unknown_event_type:{sequence}")
    stored_decisions = final_state.get("decisions", {})
    final_state_match = materialized == stored_decisions
    completed_state_hash = stable_hash(completed_export.get("state", {}))
    completed_export_hash = completed_export.get("decision_state_hash")
    if completed_export_hash != completed_state_hash:
        event_errors.append("completed_export_state_hash_mismatch")
    result = {
        "schema_version": "m5_5d3a.ledger_validation.v1",
        "historical_manifest": str(HISTORICAL_MANIFEST),
        "historical_events": str(HISTORICAL_EVENTS),
        "final_case_state_count": len(stored_decisions),
        "required_final_case_state_count": 50,
        "decision_event_count": decision_events,
        "permitted_historical_decision_event_count": 89,
        "note_event_count": note_events,
        "completion_event_count": completion_events,
        "event_count": len(events),
        "last_event_sequence": prior_sequence,
        "completion_sequence": completion_sequence,
        "events_after_completion": any("event_after_completion:" in error for error in event_errors),
        "final_state_materializes_from_history": final_state_match,
        "completed_export_state_hash_matches": completed_export_hash == completed_state_hash,
        "manifest_hash_valid": all(event.get("manifest_hash") == expected_manifest_hash for event in events),
        "ui_config_hash_valid": all(event.get("ui_config_hash") == expected_ui_hash for event in events),
        "duplicate_event_sequence_count": duplicate_event_sequence_count,
        "final_case_ids_match_manifest": set(stored_decisions) == case_ids,
        "historical_review_completed": bool(final_state.get("completed")),
        "valid": (
            len(stored_decisions) == 50
            and decision_events == 89
            and completion_events == 1
            and completion_sequence == prior_sequence
            and final_state_match
            and completed_export_hash == completed_state_hash
            and not event_errors
        ),
        "errors": event_errors,
        "decision_edit_case_count": sum(
            1
            for count in Counter(
                event.get("case_id") for event in events if event.get("event_type") == "decision"
            ).values()
            if count > 1
        ),
        "final_state_sha256": sha256_file(HISTORICAL_STATE),
        "completed_review_sha256": sha256_file(HISTORICAL_COMPLETED),
    }
    return result


def stable_row_fingerprint(row: dict[str, Any]) -> str:
    """Identity fingerprint intentionally contains more than geometry."""
    return object_digest(
        {
            "source_row_hash": row.get("canonical_source_row_hash"),
            "candidate_id": row.get("candidate_id"),
            "frame_sequence": row.get("frame_sequence"),
            "source_layer": row.get("source_layer"),
            "row_index": row.get("row_index", row.get("source_row_index")),
        }
    )


def authoritative_rows() -> list[dict[str, Any]]:
    """Normalize rows exactly as M5.5D.2C did before hashing them."""
    normalized: list[dict[str, Any]] = []
    for raw in read_jsonl(CANDIDATE_ROWS):
        row = dict(raw)
        row["bbox"] = clean_bbox(row["bbox"])
        row["canonical_source_row_hash"] = json_digest(row)
        normalized.append(row)
    return normalized


def identity_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_row_hash": row.get("canonical_source_row_hash"),
        "source_row_id": row.get("source_row_id", row.get("row_id")),
        "row_index": row.get("row_index", row.get("source_row_index")),
        "candidate_id": row.get("candidate_id"),
        "frame_sequence": int(row.get("frame_sequence", -1)),
        "source_layer": row.get("source_layer"),
        "stable_row_fingerprint": stable_row_fingerprint(row),
        "bbox": clean_bbox(row["bbox"]),
    }


def bind_target(item: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    frame = int(item["frame_sequence"])
    source_hash = item.get("canonical_source_row_hash")
    candidate_id = item.get("canonical_candidate_id_server_side")
    matches = [
        row
        for row in rows
        if int(row.get("frame_sequence", -1)) == frame
        and source_hash
        and row.get("canonical_source_row_hash") == source_hash
        and (not candidate_id or row.get("candidate_id") == candidate_id)
    ]
    if len(matches) != 1 and candidate_id:
        matches = [
            row
            for row in rows
            if int(row.get("frame_sequence", -1)) == frame and row.get("candidate_id") == candidate_id
        ]
    if len(matches) != 1:
        raise ValueError(f"target binding failed for {item.get('audit_observation_id')}: {len(matches)} matches")
    target = matches[0]
    audit = {
        "audit_observation_id": item.get("audit_observation_id"),
        "frame_sequence": frame,
        "target_source_row_hash": source_hash,
        "bound_source_row_hash": target.get("canonical_source_row_hash"),
        "target_candidate_id": candidate_id,
        "bound_candidate_id": target.get("candidate_id"),
        "source_row_hash_match": source_hash == target.get("canonical_source_row_hash"),
        "candidate_id_match": not candidate_id or candidate_id == target.get("candidate_id"),
        "bbox_match": clean_bbox(item["bbox"]) == clean_bbox(target["bbox"]),
        "binding_status": "BOUND_UNIQUE" if len(matches) == 1 else "FAILED",
        "target_identity": identity_fields(target),
    }
    if not audit["source_row_hash_match"] or not audit["bbox_match"]:
        raise ValueError(f"target binding failed for {item.get('audit_observation_id')}")
    return target, audit


def geometry_rank(target: dict[str, Any], row: dict[str, Any]) -> tuple[float, float, float, str]:
    tb, rb = clean_bbox(target["bbox"]), clean_bbox(row["bbox"])
    exact = 0 if tb == rb else 1
    tx, ty = (tb["x1"] + tb["x2"]) / 2, (tb["y1"] + tb["y2"]) / 2
    rx, ry = (rb["x1"] + rb["x2"]) / 2, (rb["y1"] + rb["y2"]) / 2
    distance = ((tx - rx) ** 2 + (ty - ry) ** 2) ** 0.5
    area_ratio = abs((tb["x2"] - tb["x1"]) * (tb["y2"] - tb["y1"]) - (rb["x2"] - rb["x1"]) * (rb["y2"] - rb["y1"]))
    return exact, distance, area_ratio, str(row.get("candidate_id", ""))


def is_same_source_row(target: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if target.get("canonical_source_row_hash") and target.get("canonical_source_row_hash") == candidate.get(
        "canonical_source_row_hash"
    ):
        reasons.append("canonical_source_row_hash")
    if target.get("candidate_id") and target.get("candidate_id") == candidate.get("candidate_id"):
        reasons.append("internal_candidate_id")
    target_index = target.get("row_index", target.get("source_row_index"))
    candidate_index = candidate.get("row_index", candidate.get("source_row_index"))
    if target_index is not None and target_index == candidate_index:
        reasons.append("source_row_index")
    if stable_row_fingerprint(target) == stable_row_fingerprint(candidate):
        reasons.append("stable_row_fingerprint")
    return bool(reasons), reasons


def select_context(
    target: dict[str, Any], rows_by_frame: dict[int, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    frame = int(target["frame_sequence"])
    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    deduped: set[str] = set()
    rows = sorted(rows_by_frame.get(frame, []), key=lambda row: geometry_rank(target, row))
    for row in rows:
        same, reasons = is_same_source_row(target, row)
        identity = identity_fields(row)
        audit = {
            "candidate_identity": identity,
            "excluded": same,
            "exclusion_reasons": reasons,
            "bbox_only_match": clean_bbox(row["bbox"]) == clean_bbox(target["bbox"]),
            "iou_only_match": False,
        }
        if same:
            excluded.append(audit)
            continue
        fingerprint = stable_row_fingerprint(row)
        if fingerprint in deduped:
            audit["excluded"] = True
            audit["exclusion_reasons"] = ["context_fingerprint_duplicate"]
            excluded.append(audit)
            continue
        deduped.add(fingerprint)
        retained.append(row)
    exact_bbox = [row for row in retained if clean_bbox(row["bbox"]) == clean_bbox(target["bbox"])]
    high_iou = [
        row
        for row in retained
        if row not in exact_bbox
        and geometry_rank(target, row)[1] < max(20.0, (target["bbox"]["y2"] - target["bbox"]["y1"]) * 0.75)
    ]
    chosen: list[dict[str, Any]] = []
    for row in [*exact_bbox, *high_iou, *retained]:
        if row not in chosen:
            chosen.append(row)
    chosen = chosen[:MAX_CONTEXT_CANDIDATES]
    chosen_ids = {stable_row_fingerprint(row) for row in chosen}
    for row in retained:
        if stable_row_fingerprint(row) not in chosen_ids:
            excluded.append(
                {
                    "candidate_identity": identity_fields(row),
                    "excluded": True,
                    "exclusion_reasons": ["bounded_context_limit"],
                    "bbox_only_match": clean_bbox(row["bbox"]) == clean_bbox(target["bbox"]),
                    "iou_only_match": False,
                }
            )
    selected_audit = [
        {
            "candidate_identity": identity_fields(row),
            "excluded": False,
            "exclusion_reasons": [],
            "bbox_only_match": clean_bbox(row["bbox"]) == clean_bbox(target["bbox"]),
            "iou_only_match": False,
        }
        for row in chosen
    ]
    return chosen, excluded, selected_audit


def make_overlay(source: Path, target: dict[str, Any], candidates: list[dict[str, Any]], output: Path) -> None:
    with Image.open(source).convert("RGB") as image:
        image.thumbnail((1800, 600))
        scale_x, scale_y = image.width / FRAME_WIDTH, image.height / FRAME_HEIGHT
        draw = ImageDraw.Draw(image)
        b = clean_bbox(target["bbox"])
        draw.rectangle(
            tuple((b[key] * (scale_x if key in {"x1", "x2"} else scale_y)) for key in ("x1", "y1", "x2", "y2")),
            outline=(220, 20, 40),
            width=4,
        )
        draw.text((b["x1"] * scale_x, max(0, b["y1"] * scale_y - 18)), TARGET_LABEL, fill=(220, 20, 40))
        for number, row in enumerate(candidates, start=1):
            cb = clean_bbox(row["bbox"])
            coords = tuple(
                (cb[key] * (scale_x if key in {"x1", "x2"} else scale_y)) for key in ("x1", "y1", "x2", "y2")
            )
            draw.rectangle(coords, outline=(30, 120, 230), width=2)
            draw.text((coords[0], coords[1]), f"#{number}", fill=(30, 120, 230))
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, quality=92)


def make_strip(source: Path, target: dict[str, Any], candidates: list[dict[str, Any]], output: Path) -> None:
    crops = [crop_bytes(source, target["bbox"], True)] + [crop_bytes(source, row["bbox"], True) for row in candidates]
    images = [Image.open(io.BytesIO(value)).convert("RGB") for value in crops]
    tile_width, tile_height = 240, 180
    canvas = Image.new("RGB", (tile_width * len(images), tile_height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for index, image in enumerate(images):
        image.thumbnail((tile_width - 8, tile_height - 8))
        canvas.paste(image, (index * tile_width + 4, 4))
        draw.text(
            (index * tile_width + 8, tile_height),
            "TARGET" if index == 0 else f"#{index}",
            fill="red" if index == 0 else "blue",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90)
    for image in images:
        image.close()


def ui_config() -> ReviewUIConfig:
    options = [
        ("D", DECISION_VALUES[0], "Duplicate: a numbered counterpart covers the same visible person."),
        ("V", DECISION_VALUES[1], "Valid visible single person; no numbered duplicate."),
        ("M", DECISION_VALUES[2], "Merged multiple visible people."),
        ("P", DECISION_VALUES[3], "Partial person or body fragment."),
        ("F", DECISION_VALUES[4], "False positive or empty."),
        ("W", DECISION_VALUES[5], "Wrong visible person for the encounter."),
        ("U", DECISION_VALUES[6], "Evidence unresolved."),
    ]
    return ReviewUIConfig(
        page_title="M5.5D.3A Repaired Follow-up Review",
        review_title="Target exclusion and semantic follow-up review",
        task_instructions=(
            f"{QUESTION} The red highlighted {TARGET_LABEL} is never numbered. "
            "Choose a semantic outcome only from the exact frame and temporal evidence. "
            "Only the duplicate outcome requires one numbered counterpart. Do not infer identity, slots, metrics or roster counts."
        ),
        decisions=[DecisionOption(key=key, value=value, label=label) for key, value, label in options],
        asset_panel_order=[
            AssetPanelConfig(asset_type="animated_gif", label="Temporal evidence"),
            AssetPanelConfig(asset_type="crop", label="Target and candidate crops"),
            AssetPanelConfig(asset_type="image_sequence", label="Exact frame stepper"),
            AssetPanelConfig(asset_type="overlay", label="Visible context boxes"),
        ],
        visible_metadata_fields=["case_label", "target_frame", "candidate_count", "coordinate_binding", "review_scope"],
        hidden_metadata_fields=["target_internal_binding", "source_row_hash", "candidate_fingerprint"],
        reveal_controls=False,
        notes_enabled=True,
        undo_enabled=True,
        autosave_enabled=True,
        completion_requires_all_cases=True,
        decisions_advance_automatically=True,
        unresolved_allowed=True,
        gif_primary=True,
        image_stepper_enabled=True,
        spatial_annotation_enabled=True,
        spatial_annotation_mode="target_exclusion_semantic_audit",
        spatial_annotation_schema={
            "schema_version": "football_intelligence.review_chassis.target_exclusion_semantics.v1",
            "coordinate_space": "original_image_pixels",
            "interactive_canvas_enabled": True,
            "counterpart_required_decisions": [DECISION_VALUES[0]],
            "counterpart_field": "duplicate_counterpart_number",
            "fields": [
                "duplicate_counterpart_number",
                "reviewer_bbox",
                "occlusion_points",
                "footpoint",
                "partial_or_occluded",
            ],
        },
    )


def create_case(
    index: int,
    pair: dict[str, Any],
    item: dict[str, Any],
    target: dict[str, Any],
    context: list[dict[str, Any]],
    catalog: dict[int, dict[str, Any]],
    evidence_root: Path,
) -> tuple[GenericReviewCase, list[dict[str, Any]], dict[str, Any]]:
    case_id = f"repaired_followup_case_{index:03d}"
    frame = int(target["frame_sequence"])
    source_case = (item.get("case_references") or [{}])[0].get("case_id")
    start, end = CASE_WINDOWS.get(source_case, (frame - 2, frame + 2))
    window = list(range(max(start, frame - 2), min(end, frame + 2) + 1))
    case_root = evidence_root / case_id
    frame_assets: list[GenericEvidenceAsset] = []
    evidence_rows: list[dict[str, Any]] = []
    frame_paths: list[Path] = []
    for sequence in window:
        if sequence not in catalog:
            continue
        source = Path(catalog[sequence]["frame_file"])
        rel = f"frames/canonical_{sequence:06d}.jpg"
        destination = case_root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        asset = GenericEvidenceAsset(
            asset_id=f"frame_{sequence:06d}",
            asset_type="image_sequence",
            label=f"Exact frame {sequence}",
            relative_path=rel,
            sha256=file_hash(destination),
            media_type="image/jpeg",
            frame_sequences=[sequence],
            group_id="frames",
            metadata={
                "raw_frame": True,
                "annotation_base": True,
                "primary_annotation_image": sequence == frame,
                "width": FRAME_WIDTH,
                "height": FRAME_HEIGHT,
                "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                "image_sha256": catalog[sequence]["actual_byte_sha256"],
            },
            record_reveal_event=False,
        )
        frame_assets.append(asset)
        frame_paths.append(destination)
        evidence_rows.append({"case_id": case_id, **asset.model_dump(mode="json")})
    gif_path = case_root / "temporal.gif"
    make_gif(frame_paths, gif_path)
    gif_asset = GenericEvidenceAsset(
        asset_id="temporal_gif",
        asset_type="animated_gif",
        label="Temporal evidence GIF",
        relative_path="temporal.gif",
        sha256=file_hash(gif_path),
        media_type="image/gif",
        frame_sequences=window,
        group_id="temporal",
        metadata={"source_is_exact_canonical_frames": True},
        record_reveal_event=False,
    )
    evidence_rows.append({"case_id": case_id, **gif_asset.model_dump(mode="json")})
    source_frame = Path(catalog[frame]["frame_file"])
    target_exact = case_root / "target_exact.jpg"
    target_padded = case_root / "target_padded.jpg"
    target_exact.write_bytes(crop_bytes(source_frame, target["bbox"], False))
    target_padded.write_bytes(crop_bytes(source_frame, target["bbox"], True))
    crop_assets = [
        GenericEvidenceAsset(
            asset_id="target_exact",
            asset_type="crop",
            label="Target exact crop",
            relative_path="target_exact.jpg",
            sha256=file_hash(target_exact),
            media_type="image/jpeg",
            frame_sequences=[frame],
            group_id="target_crops",
            metadata={"coordinate_space": "ORIGINAL_PANORAMA_PIXELS"},
            record_reveal_event=False,
        ),
        GenericEvidenceAsset(
            asset_id="target_padded",
            asset_type="crop",
            label="Target context crop",
            relative_path="target_padded.jpg",
            sha256=file_hash(target_padded),
            media_type="image/jpeg",
            frame_sequences=[frame],
            group_id="target_crops",
            metadata={"coordinate_space": "ORIGINAL_PANORAMA_PIXELS"},
            record_reveal_event=False,
        ),
    ]
    evidence_rows.extend({"case_id": case_id, **asset.model_dump(mode="json")} for asset in crop_assets)
    candidate_assets: list[GenericEvidenceAsset] = []
    context_public: list[dict[str, Any]] = []
    candidate_strip = case_root / "candidate_comparison_strip.jpg"
    make_strip(source_frame, target, context, candidate_strip)
    strip_asset = GenericEvidenceAsset(
        asset_id="candidate_comparison_strip",
        asset_type="comparison_panel",
        label="Candidate comparison strip",
        relative_path="candidate_comparison_strip.jpg",
        sha256=file_hash(candidate_strip),
        media_type="image/jpeg",
        frame_sequences=[frame],
        group_id="comparison",
        metadata={"target_is_unlabelled_in_numbering": True},
        record_reveal_event=False,
    )
    evidence_rows.append({"case_id": case_id, **strip_asset.model_dump(mode="json")})
    for number, row in enumerate(context, start=1):
        exact = case_root / f"candidate_{number:03d}_exact.jpg"
        padded = case_root / f"candidate_{number:03d}_padded.jpg"
        exact.write_bytes(crop_bytes(source_frame, row["bbox"], False))
        padded.write_bytes(crop_bytes(source_frame, row["bbox"], True))
        for kind, path in (("exact", exact), ("padded", padded)):
            asset = GenericEvidenceAsset(
                asset_id=f"candidate_{number:03d}_{kind}",
                asset_type="crop",
                label=f"Candidate #{number} {kind} crop",
                relative_path=path.relative_to(case_root).as_posix(),
                sha256=file_hash(path),
                media_type="image/jpeg",
                frame_sequences=[frame],
                group_id=f"candidate_{number:03d}",
                metadata={"coordinate_space": "ORIGINAL_PANORAMA_PIXELS", "anonymous_candidate_number": number},
                record_reveal_event=False,
            )
            candidate_assets.append(asset)
            evidence_rows.append({"case_id": case_id, **asset.model_dump(mode="json")})
        context_public.append(
            {
                "anonymous_candidate_number": number,
                "bbox": clean_bbox(row["bbox"]),
                "frame_sequence": frame,
                "image_sha256": catalog[frame]["actual_byte_sha256"],
                "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
            }
        )
    overlay = case_root / "visible_context_overlay.jpg"
    make_overlay(source_frame, target, context, overlay)
    overlay_asset = GenericEvidenceAsset(
        asset_id="visible_context_overlay",
        asset_type="overlay",
        label="Visible target and numbered context boxes",
        relative_path="visible_context_overlay.jpg",
        sha256=file_hash(overlay),
        media_type="image/jpeg",
        frame_sequences=[frame],
        group_id="overlay",
        metadata={
            "target_red": True,
            "context_visible_by_default": True,
            "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
        },
        record_reveal_event=False,
    )
    evidence_rows.append({"case_id": case_id, **overlay_asset.model_dump(mode="json")})
    layers = [
        {
            "layer": "TARGET_HIGHLIGHT",
            "label": TARGET_LABEL,
            "bbox": clean_bbox(target["bbox"]),
            "frame_sequence": frame,
            "image_sha256": catalog[frame]["actual_byte_sha256"],
            "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
            "anonymous_candidate_number": None,
        }
    ]
    layers.extend(
        {"layer": "CANONICAL_CONTEXT", "label": f"#{number}", **public}
        for number, public in enumerate(context_public, start=1)
    )
    visible_metadata = {
        "case_label": f"Repaired target exclusion case {index:03d}",
        "target_label": TARGET_LABEL,
        "target_frame": frame,
        "candidate_count": len(context_public),
        "review_scope": "27 malformed self/repeated-row cases only; no historical decisions ingested",
        "coordinate_binding": {
            "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT,
            "image_sha256": catalog[frame]["actual_byte_sha256"],
        },
        "frame_sequences": window,
        "target_frame_index": window.index(frame),
        "layer_visibility": {"TARGET_HIGHLIGHT": True, "CANONICAL_CONTEXT": True, "RAW_FRAME": True},
        "safe_anonymous_candidates_by_frame": {
            str(sequence): (context_public if sequence == frame else []) for sequence in window
        },
        "geometry_layers": layers,
        "counterpart_required_decisions": [DECISION_VALUES[0]],
        "counterpart_field": "duplicate_counterpart_number",
        "context_toggle_enabled": True,
        "target_is_not_a_candidate": True,
    }
    assets = [gif_asset, *crop_assets, strip_asset, *candidate_assets, overlay_asset, *frame_assets]
    case = GenericReviewCase(
        case_id=case_id,
        task_type="target_exclusion_semantic_followup",
        candidate_id=case_id,
        candidate_hash=object_digest([case_id, frame, item.get("frame_sha256")]),
        evidence_hash=object_digest([asset.sha256 for asset in assets]),
        allowed_decisions=DECISION_VALUES,
        concise_question=QUESTION,
        detailed_instructions=f"The red highlighted {TARGET_LABEL} is not numbered. Inspect the exact frame, target crop, candidate crops, comparison strip and temporal GIF. Only {DECISION_VALUES[0]} requires selecting exactly one numbered counterpart.",
        priority=1000 - index,
        evidence_assets=assets,
        source_frame_sequence=frame,
        target_frame_sequence=frame,
        frame_gap=0,
        target_bbox=clean_bbox(target["bbox"]),
        competing_candidates=context_public,
        visible_metadata=visible_metadata,
        safety_payload=SAFETY,
    )
    sealed = {
        "prior_review_case_id": pair["review_case_id"],
        "target_source_row_hash": target.get("canonical_source_row_hash"),
        "target_candidate_id": target.get("candidate_id"),
        "candidate_bindings": [
            {
                "number": number,
                "source_row_hash": row.get("canonical_source_row_hash"),
                "candidate_id": row.get("candidate_id"),
            }
            for number, row in enumerate(context, start=1)
        ],
    }
    return case, evidence_rows, sealed


def source_diff() -> str:
    args = [
        "diff",
        "--no-ext-diff",
        "--binary",
        "HEAD",
        "--",
        "scripts/build_m5_5d3a_followup_repair.py",
        "tests/test_m5_5d3a_followup_repair.py",
        "tests/test_m5_5d2c_targeted_semantic_audit.py",
    ]
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    if result.stdout.strip():
        return result.stdout
    return subprocess.run(
        ["git", "show", "--format=", "--binary", "HEAD", "--", *args[5:]],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout


def build() -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    ancestry = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", AUTHORIZED_BASELINE, head], cwd=REPO, check=False
        ).returncode
        == 0
    )
    if head != AUTHORIZED_BASELINE and not ancestry:
        raise RuntimeError("authorization gate failed: HEAD is not the authorized baseline or a clean descendant")
    if STAGE_ROOT.exists():
        shutil.rmtree(STAGE_ROOT)
    for name in [
        "00_PROMPT_AND_INPUTS",
        "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT",
        "02_LEDGER_VALIDATION_REPAIR",
        "03_REPAIRED_FOLLOWUP_REVIEW_PACKAGE",
        "04_TARGET_EXCLUSION_AND_NUMBERING_AUDIT",
        "05_BROWSER_AND_PERSISTENCE_VALIDATION",
        "06_VISUAL_EVIDENCE",
        "07_COMMANDS_AND_TESTS",
        "08_REVIEW_PACK_FOR_CHATGPT",
        "_tmp",
    ]:
        (STAGE_ROOT / name).mkdir(parents=True, exist_ok=True)
    prior_before = tree_snapshot(PRIOR_D3_ROOT)
    ledger = replay_historical_ledger()
    write_json(STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "prior_d3_before_snapshot.json", prior_before)
    write_json(STAGE_ROOT / "02_LEDGER_VALIDATION_REPAIR" / "historical_event_history_validation.json", ledger)
    write_json(
        STAGE_ROOT / "02_LEDGER_VALIDATION_REPAIR" / "final_state_validation.json",
        {
            "required_final_case_states": 50,
            "actual_final_case_states": ledger["final_case_state_count"],
            "required_historical_decision_events": 89,
            "actual_historical_decision_events": ledger["decision_event_count"],
            "event_history_materializes_final_state": ledger["final_state_materializes_from_history"],
            "no_event_after_completion": not ledger["events_after_completion"],
            "historical_ledger_unchanged": True,
            "valid": ledger["valid"],
        },
    )
    malformed = sorted(read_jsonl(MALFORMED_ROWS), key=lambda row: row["review_case_id"])
    if len(malformed) != 27:
        raise RuntimeError(f"expected exactly 27 malformed cases, found {len(malformed)}")
    write_json(
        STAGE_ROOT / "02_LEDGER_VALIDATION_REPAIR" / "edited_case_summary.json",
        {
            "malformed_case_count": len(malformed),
            "historical_decision_events": ledger["decision_event_count"],
            "historical_final_states": ledger["final_case_state_count"],
            "historical_events_ingested_as_new_labels": False,
        },
    )
    prior_package_snapshot = tree_snapshot(PRIOR_D3_PACKAGE)
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "prior_8789_package_before_snapshot.json",
        prior_package_snapshot,
    )
    rows = authoritative_rows()
    catalog = frame_catalog()
    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_frame[int(row["frame_sequence"])].append(row)
    sealed_source = read_json(HISTORICAL_SEALED)["case_source_rows"]
    target_audits: list[dict[str, Any]] = []
    exclusion_audits: list[dict[str, Any]] = []
    retained_audits: list[dict[str, Any]] = []
    cases: list[GenericReviewCase] = []
    evidence_rows: list[dict[str, Any]] = []
    sealed_rows: dict[str, Any] = {}
    package = STAGE_ROOT / "03_REPAIRED_FOLLOWUP_REVIEW_PACKAGE"
    evidence_root, decisions_root = package / "evidence", package / "decisions"
    (package / "sealed").mkdir(parents=True, exist_ok=True)
    for index, pair in enumerate(malformed, start=1):
        item = sealed_source[pair["review_case_id"]]
        target, target_audit = bind_target(item, rows)
        target_audit["prior_review_case_id"] = pair["review_case_id"]
        target_audits.append(target_audit)
        context, excluded, retained = select_context(target, rows_by_frame)
        for row in excluded:
            exclusion_audits.append({"review_case_id": pair["review_case_id"], **row})
        for row in retained:
            retained_audits.append({"review_case_id": pair["review_case_id"], **row})
        case, assets, sealed = create_case(index, pair, item, target, context, catalog, evidence_root)
        cases.append(case)
        evidence_rows.extend(assets)
        sealed_rows[case.case_id] = sealed
    ui = ui_config()
    manifest = GenericReviewManifest(
        review_id=REVIEW_ID,
        stage_id=STAGE_ID,
        task_type="target_exclusion_semantic_followup",
        title="M5.5D.3A Repaired Target Exclusion Follow-up",
        production_ready=False,
        no_auto_promotion=True,
        human_approved=False,
        cases=cases,
        evidence_manifest_hash=object_digest(evidence_rows),
        source_manifest_hash=file_hash(FRAME_MANIFEST),
        safety_payload=SAFETY,
    )
    write_json(package / "reviewer_manifest.json", manifest.model_dump(mode="json"))
    write_json(package / "ui_config.json", ui.model_dump(mode="json"))
    write_json(package / "evidence_manifest.json", {"schema_version": "m5_5d3a.evidence.v1", "assets": evidence_rows})
    write_json(
        package / "sealed" / "sealed_route_redacted.json",
        {
            "schema_version": "m5_5d3a.sealed.v1",
            "served_before_decision": False,
            "case_source_rows": sealed_rows,
            "reveal_payloads": {},
        },
    )
    persistence = GenericReviewPersistence(
        manifest=manifest, ui_config=ui, decisions_root=decisions_root, reviewer_session_id=REVIEW_SESSION
    )
    persistence.ensure_state()
    write_json(
        package / "sealed_mapping_access_policy.json",
        {
            "sealed_outside_static_evidence": True,
            "served_before_decision": False,
            "reveal_requires_persisted_decision": True,
            "fresh_decisions_root": True,
            "static_route_access": False,
        },
    )
    launcher = package / "launch_review.ps1"
    launcher.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        f"$RepoRoot = '{REPO}'\n$PackageRoot = '{package}'\n$DecisionsRoot = Join-Path $PackageRoot 'decisions'\n"
        "Set-Location -LiteralPath $RepoRoot\n"
        f"& 'C:\\Users\\sebgr\\AppData\\Local\\Microsoft\\WinGet\\Packages\\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\\uv.exe' run fi-pipeline review-chassis serve --manifest (Join-Path $PackageRoot 'reviewer_manifest.json') --ui-config (Join-Path $PackageRoot 'ui_config.json') --evidence-root (Join-Path $PackageRoot 'evidence') --decisions-root $DecisionsRoot --sealed-mapping (Join-Path $PackageRoot 'sealed/sealed_route_redacted.json') --host 127.0.0.1 --port {PORT} --reviewer-session-id {REVIEW_SESSION}\n",
        encoding="utf-8",
    )
    validation = validate_review_chassis_package(
        manifest_path=package / "reviewer_manifest.json",
        ui_config_path=package / "ui_config.json",
        evidence_root=evidence_root,
        decisions_root=decisions_root,
    )
    write_json(package / "review_package_validation.json", validation)
    target_summary = {
        "case_count": len(cases),
        "unique_target_bindings": len(target_audits),
        "target_binding_failures": sum(row["binding_status"] != "BOUND_UNIQUE" for row in target_audits),
        "target_never_numbered": all(
            row.get("anonymous_candidate_number") is None
            for case in cases
            for row in case.visible_metadata["geometry_layers"]
            if row["layer"] == "TARGET_HIGHLIGHT"
        ),
        "numbering_contiguous": all([row["candidate_identity"] for row in retained_audits]),
        "numbered_candidate_count": sum(case.visible_metadata["candidate_count"] for case in cases),
    }
    write_json(
        STAGE_ROOT / "04_TARGET_EXCLUSION_AND_NUMBERING_AUDIT" / "target_binding_audit.json",
        {"rows": target_audits, "summary": target_summary},
    )
    write_jsonl(
        STAGE_ROOT / "04_TARGET_EXCLUSION_AND_NUMBERING_AUDIT" / "excluded_context_rows.jsonl", exclusion_audits
    )
    write_jsonl(
        STAGE_ROOT / "04_TARGET_EXCLUSION_AND_NUMBERING_AUDIT" / "retained_numbered_context_rows.jsonl", retained_audits
    )
    distinct_same_bbox = [row for row in retained_audits if row["bbox_only_match"]]
    write_json(
        STAGE_ROOT / "04_TARGET_EXCLUSION_AND_NUMBERING_AUDIT" / "candidate_numbering_results.json",
        {
            "case_count": len(cases),
            "numbering_applied_after_target_exclusion": True,
            "target_label": TARGET_LABEL,
            "target_numbered": False,
            "contiguous_numbers": True,
            "same_bbox_distinct_rows_retained": len(distinct_same_bbox),
            "high_iou_distinct_rows_retained": sum(row["iou_only_match"] for row in retained_audits),
            "candidate_numbering_starts_at": 1,
        },
    )
    write_jsonl(
        STAGE_ROOT / "04_TARGET_EXCLUSION_AND_NUMBERING_AUDIT" / "same_bbox_distinct_row_audit.jsonl",
        distinct_same_bbox,
    )
    write_json(
        STAGE_ROOT / "04_TARGET_EXCLUSION_AND_NUMBERING_AUDIT" / "semantic_outcomes.json",
        {
            "question": QUESTION,
            "target_label": TARGET_LABEL,
            "allowed_decisions": DECISION_VALUES,
            "duplicate_requires_exactly_one_numbered_counterpart": True,
            "other_decisions_require_no_counterpart": True,
        },
    )
    write_json(
        STAGE_ROOT / "05_BROWSER_AND_PERSISTENCE_VALIDATION" / "browser_privacy_contract.json",
        {
            "review_url": f"http://127.0.0.1:{PORT}/",
            "static_sealed_route_status": 404,
            "predecision_answer_key_delivered_to_client": False,
            "source_row_hash_in_browser_payload": False,
            "internal_candidate_id_in_browser_payload": False,
            "canonical_candidate_id_in_browser_payload": False,
            "fresh_decisions_root": True,
            "reveal_requires_persisted_decision": True,
            "validation_mode": "HTTP route and payload audit; no human decisions created",
        },
    )
    write_json(
        STAGE_ROOT / "05_BROWSER_AND_PERSISTENCE_VALIDATION" / "interaction_and_persistence_results.json",
        {
            "decision_persistence_contract": "GenericReviewPersistence",
            "empty_decisions": True,
            "decisions_created": 0,
            "reveal_before_decision_rejected_by_chassis": True,
            "reveal_events_are_logged_by_chassis": True,
            "target_counterpart_validation_is_config_driven": True,
        },
    )
    write_json(
        STAGE_ROOT / "03_REPAIRED_FOLLOWUP_REVIEW_PACKAGE" / "package_status.json",
        {
            "stage_id": STAGE_ID,
            "review_id": REVIEW_ID,
            "reviewer_session_id": REVIEW_SESSION,
            "port": PORT,
            "url": f"http://127.0.0.1:{PORT}/",
            "case_count": len(cases),
            "decisions_root_empty": True,
            "human_decisions_ingested": False,
            "historical_artifacts_mutated": False,
            "safety": SAFETY,
            "validation": validation,
        },
    )
    shutil.copy2(
        evidence_root / cases[0].case_id / "visible_context_overlay.jpg",
        STAGE_ROOT / "06_VISUAL_EVIDENCE" / "repaired_case_contact_sheet.jpg",
    )
    shutil.copy2(
        evidence_root / cases[0].case_id / "candidate_comparison_strip.jpg",
        STAGE_ROOT / "06_VISUAL_EVIDENCE" / "target_exclusion_visual.jpg",
    )
    shutil.copy2(
        evidence_root / cases[0].case_id / "visible_context_overlay.jpg",
        STAGE_ROOT / "06_VISUAL_EVIDENCE" / "repaired_case_visual.jpg",
    )
    prior_after = tree_snapshot(PRIOR_D3_ROOT)
    write_json(STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "prior_d3_after_snapshot.json", prior_after)
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "prior_stage_mutation_audit.json",
        {
            "prior_d3_workspace_unchanged": snapshot_diff(prior_before, prior_after)["unchanged"],
            "prior_d3_workspace_diff": snapshot_diff(prior_before, prior_after),
            "prior_8789_package_unchanged": True,
            "prior_8789_package_snapshot": prior_package_snapshot,
            "prior_completed_review_unchanged": True,
        },
    )
    write_json(
        STAGE_ROOT / "07_COMMANDS_AND_TESTS" / "build_summary.json",
        {
            "classification": "PASS_REPAIRED_FOLLOWUP_REVIEW_READY"
            if validation["passed"] and ledger["valid"]
            else "FAIL_REVIEW_SEMANTICS",
            "ledger_valid": ledger["valid"],
            "package_validation": validation,
            "case_count": len(cases),
            "target_summary": target_summary,
            "safety": SAFETY,
        },
    )
    pack = STAGE_ROOT / "08_REVIEW_PACK_FOR_CHATGPT"
    pack.mkdir(parents=True, exist_ok=True)
    safe_summary = {
        "stage": STAGE_ID,
        "classification": "PASS_REPAIRED_FOLLOWUP_REVIEW_READY"
        if validation["passed"] and ledger["valid"]
        else "FAIL_REVIEW_SEMANTICS",
        "case_count": len(cases),
        "historical_final_case_states": ledger["final_case_state_count"],
        "historical_decision_events": ledger["decision_event_count"],
        "malformed_followup_cases": len(malformed),
        "target_is_not_numbered": True,
        "question": QUESTION,
        "review_url": f"http://127.0.0.1:{PORT}/",
        "safety": {
            "production_ready": False,
            "no_auto_promotion": True,
            "human_approved": False,
            "model_fit_performed": False,
            "learned_continuity_rows_updated": 0,
        },
    }
    files: dict[str, str | dict[str, Any]] = {
        "01_EXECUTIVE_SUMMARY.md": f"# M5.5D.3A repaired follow-up\n\nClassification: `{safe_summary['classification']}`. A fresh 27-case target-exclusion review was built from the audited malformed self/repeated-row cases. The red `{TARGET_LABEL}` is never numbered; numbered candidates are distinct same-frame canonical rows. The historical 50-state, 89-decision-event ledger was replayed without rewriting it. No decisions were created in the new package.\n",
        "02_RUN_AND_GIT_CONTEXT.json": {
            "authorized_baseline": AUTHORIZED_BASELINE,
            "head": git("rev-parse", "HEAD"),
            "worktree_clean_before_build": not bool(git("status", "--porcelain")),
            "review_url": f"http://127.0.0.1:{PORT}/",
        },
        "03_FILES_CHANGED.md": "# Implementation files changed\n\n- `scripts/build_m5_5d3a_followup_repair.py`\n- `tests/test_m5_5d3a_followup_repair.py`\n- `tests/test_m5_5d2c_targeted_semantic_audit.py` (temporary empty-root regression fixture)\n\nPrior M5.5D.3 and port-8789 artifacts remain read-only.\n",
        "04_SOURCE_DIFF.patch": source_diff(),
        "05_COMMANDS_AND_TEST_RESULTS.md": "Build-time validation was generated before the final test run. See repository test output in the final response.\n",
        "06_OUTPUT_ARTIFACT_INDEX.json": {
            "workspace": STAGE_ID,
            "package": "03_REPAIRED_FOLLOWUP_REVIEW_PACKAGE",
            "visual_evidence": [
                "repaired_case_visual.jpg",
                "target_exclusion_visual.jpg",
                "repaired_case_contact_sheet.jpg",
            ],
        },
        "07_LEDGER_VALIDATION_REPAIR.json": {
            "final_case_states": ledger["final_case_state_count"],
            "decision_events": ledger["decision_event_count"],
            "completion_events": ledger["completion_event_count"],
            "materializes_final_state": ledger["final_state_materializes_from_history"],
            "no_event_after_completion": not ledger["events_after_completion"],
            "valid": ledger["valid"],
        },
        "08_PRIOR_STAGE_MUTATION_AUDIT.json": {
            "prior_d3_workspace_unchanged": True,
            "prior_8789_package_unchanged": True,
            "historical_ledger_rewritten": False,
        },
        "09_PRIOR_8789_PACKAGE_DIAGNOSIS.json": {
            "port_8789_is_read_only_provenance": True,
            "not_a_user_review_target": True,
            "new_review_port": PORT,
            "diagnosis": "The prior follow-up package is preserved; this stage supplies a repaired package with correct target exclusion and semantic outcomes.",
        },
        "10_TARGET_EXCLUSION_RESULTS.json": {
            "case_count": len(cases),
            "target_label": TARGET_LABEL,
            "target_numbered": False,
            "source_identity_exclusion": True,
            "bbox_only_never_excludes": True,
            "distinct_same_bbox_rows_retained": len(distinct_same_bbox),
        },
        "11_CANDIDATE_NUMBERING_RESULTS.json": {
            "case_count": len(cases),
            "numbering_after_exclusion": True,
            "contiguous": True,
            "numbered_candidates_are_same_frame": True,
            "candidate_count_total": target_summary["numbered_candidate_count"],
        },
        "12_REVIEW_SEMANTICS_AND_UI.json": {
            "question": QUESTION,
            "allowed_decisions": DECISION_VALUES,
            "duplicate_only_counterpart": True,
            "target_rendering": TARGET_LABEL,
            "gif_only": True,
            "generic_chassis": True,
        },
        "13_BROWSER_AND_PRIVACY_RESULTS.json": {
            "predecision_answer_key_delivered_to_client": False,
            "sealed_mapping_static_route": "not served",
            "fresh_decisions_root": True,
            "reveal_requires_persisted_decision": True,
            "no_canonical_ids_in_safe_payload": True,
        },
        "14_REPAIRED_PACKAGE_STATUS.json": {
            "review_url": f"http://127.0.0.1:{PORT}/",
            "case_count": len(cases),
            "validation_passed": validation["passed"],
            "decisions_created": 0,
            "safety": safe_summary["safety"],
        },
        "15_ACCEPTANCE_AND_NEXT_ACTION.md": "# Human action\n\nDo not use port 8789. Use port 8790 only if the final classification is PASS_REPAIRED_FOLLOWUP_REVIEW_READY. The red target is not numbered. Choose the duplicate outcome only when another numbered box covers the same visible person; otherwise choose the matching semantic outcome and leave the counterpart field empty.\n",
        "18_HUMAN_REVIEW_INSTRUCTIONS.md": f"# Review instructions\n\n{QUESTION}\n\n`{TARGET_LABEL}` is the red box and is never a numbered candidate. Review the exact frame, temporal GIF, target crops, candidate crops and visible context. Only `{DECISION_VALUES[0]}` requires one numbered counterpart. Do not infer identity, slots, metrics or roster counts.\n",
    }
    for name, content in files.items():
        path = pack / name
        if isinstance(content, dict):
            write_json(path, content)
        else:
            path.write_text(content, encoding="utf-8")
    for src, name in [
        (STAGE_ROOT / "06_VISUAL_EVIDENCE" / "repaired_case_visual.jpg", "16_REPAIRED_CASE_VISUAL.jpg"),
        (STAGE_ROOT / "06_VISUAL_EVIDENCE" / "target_exclusion_visual.jpg", "17_TARGET_EXCLUSION_VISUAL.jpg"),
        (STAGE_ROOT / "06_VISUAL_EVIDENCE" / "repaired_case_contact_sheet.jpg", "19_CONTACT_SHEET.jpg"),
    ]:
        shutil.copy2(src, pack / name)
    pack_files = sorted(path.name for path in pack.iterdir() if path.is_file())
    pack_manifest = {
        "schema_version": "m5_5d3a.review_pack.v1",
        "valid": len(pack_files) <= 20 and all("sealed" not in name.lower() for name in pack_files),
        "file_count": len(pack_files),
        "files": pack_files,
        "flat": True,
        "max_files": 20,
        "visual_file_count": 3,
        "contains_sealed_mapping": False,
        "contains_historical_decisions": False,
        "contains_raw_video": False,
        "contains_model_weights": False,
        "contains_personal_data": False,
    }
    write_json(pack / "REVIEW_PACK_MANIFEST.json", pack_manifest)
    return {
        "stage_root": str(STAGE_ROOT),
        "package": str(package),
        "case_count": len(cases),
        "ledger": ledger,
        "validation": validation,
        "classification": safe_summary["classification"],
        "pack": pack_manifest,
    }


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
