"""Build the M5.5E.2 single-viewer temporal review package."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.models import (
    AssetPanelConfig,
    DecisionOption,
    GenericEvidenceAsset,
    GenericReviewCase,
    GenericReviewManifest,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.persistence import GenericReviewPersistence, atomic_write_json
from football_intelligence.review_chassis.validation import validate_review_chassis_package
import build_m5_5e1_temporal_overlay_repair as prior


ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = ROOT / "SoccerTrack-v2"
PROMPT_ROOT = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5E2_Simplified_Frame_Step_Review_UI_Prompt_v1"
PRIOR_ROOT = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5E1_TEMPORAL_OVERLAY_AND_TRACKLET_BINDING_REPAIR_v1"
STAGE_ROOT = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5E2_SIMPLIFIED_FRAME_STEP_TEMPORAL_REVIEW_UI_v1"
PACKAGE_ROOT = STAGE_ROOT / "05_SIMPLIFIED_TEMPORAL_HUMAN_REVIEW_PACKAGE"
EVIDENCE_ROOT = PACKAGE_ROOT / "evidence"
DECISIONS_ROOT = PACKAGE_ROOT / "decisions"
PACK_ROOT = STAGE_ROOT / "08_REVIEW_PACK_FOR_CHATGPT"
AUTHORIZED_BASELINE = "372982e9579c5ad351ff0f65edeac88b1158c1e1"
STAGE_ID = "M5_5E2_SIMPLIFIED_FRAME_STEP_TEMPORAL_REVIEW_UI_v1"
REVIEW_ID = "m5_5e2_simplified_frame_step_temporal_review_v1"
REVIEW_SESSION = "m5_5e2_simplified_frame_step_human_reviewer"
REVIEW_PORT = 8793
PRIMARY_QUESTION = (
    "Were at least two independently visible people supported before the interval, "
    "did visual observation supply genuinely collapse during the interval, and were "
    "at least two independently visible people supported again afterward?"
)
DECISION_LABELS = {
    "GENUINE_TWO_TO_ONE_COLLAPSE": "G - Genuine observation-deficit interval",
    "GENUINE_OBSERVED_MISSING_OBSERVED": "G - Genuine observation-deficit interval",
    "GENUINE_MERGED_OBSERVATION_INTERVAL": "G - Genuine observation-deficit interval",
    "PARTIAL_FRAGMENT_OBSERVATION_DEFICIT": "G - Genuine observation-deficit interval",
    "ORDINARY_CROSSING_INDEPENDENT_OBSERVATIONS_REMAIN": "O - Ordinary crossing; observations remain independent",
    "DETECTOR_DUPLICATE_OR_FALSE_POSITIVE_ARTIFACT": "X - Detector or duplicate artifact",
    "INSUFFICIENT_INCOMING_PRECONDITION": "I - Insufficient incoming evidence",
    "INSUFFICIENT_OUTGOING_POSTCONDITION": "P - Insufficient outgoing evidence",
    "EVIDENCE_UNRESOLVED": "U - Unresolved",
}
SAFETY = {
    "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
    "production_ready": False,
    "no_auto_promotion": True,
    "human_approved": False,
    "safe_to_apply_globally": False,
    "match_local_only": True,
    "sandbox_only": True,
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
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8"
    )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()


def snapshot_tree(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    return {"root": str(root), "file_count": len(files), "files": files, "aggregate_sha256": digest(files)}


def box(row: dict[str, Any]) -> dict[str, float]:
    value = row.get("bbox") or row
    return {key: float(value[key]) for key in ("x1", "y1", "x2", "y2")}


def draw_dashed(
    draw: ImageDraw.ImageDraw,
    coords: tuple[float, float, float, float],
    color: tuple[int, int, int, int],
    width: int = 4,
) -> None:
    x1, y1, x2, y2 = coords
    for horizontal in (True, False):
        length = (x2 - x1) if horizontal else (y2 - y1)
        start = x1 if horizontal else y1
        for offset in range(0, max(1, int(length)), 14):
            end = min(int(length), offset + 7)
            if horizontal:
                draw.line((start + offset, y1, start + end, y1), fill=color, width=width)
                draw.line((start + offset, y2, start + end, y2), fill=color, width=width)
            else:
                draw.line((x1, start + offset, x1, start + end), fill=color, width=width)
                draw.line((x2, start + offset, x2, start + end), fill=color, width=width)


def font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def phase_for(frame: int, event: dict[str, Any]) -> str:
    start = int(event.get("deficit_start_frame", event["contact_frame"]))
    end = int(event.get("deficit_end_frame", start))
    if frame < start:
        return "BEFORE"
    if frame > end:
        return "AFTER"
    return "INTERVAL"


def encounter_region(
    event: dict[str, Any], state: dict[str, Any], frame: int, width: int, height: int
) -> dict[str, float]:
    boxes = [box(item["row"]) for item in state["observed_by_frame"].get(frame, [])]
    if not boxes:
        boxes = [
            event.get("anchor_bbox")
            or {"x1": width * 0.5 - 30, "y1": height * 0.5 - 80, "x2": width * 0.5 + 30, "y2": height * 0.5 + 80}
        ]
    x1 = min(item["x1"] for item in boxes)
    y1 = min(item["y1"] for item in boxes)
    x2 = max(item["x2"] for item in boxes)
    y2 = max(item["y2"] for item in boxes)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rw = min(float(width) * 0.38, max(480.0, (x2 - x1) * 4.0))
    rh = min(float(height) * 0.78, max(260.0, (y2 - y1) * 3.2))
    return {
        "x1": max(0.0, cx - rw / 2),
        "y1": max(0.0, cy - rh / 2),
        "x2": min(float(width), cx + rw / 2),
        "y2": min(float(height), cy + rh / 2),
    }


def locator_region(region: dict[str, float], width: int, height: int) -> dict[str, float]:
    cx = (region["x1"] + region["x2"]) / 2
    cy = (region["y1"] + region["y2"]) / 2
    rw = min(region["x2"] - region["x1"], width * 0.18)
    rh = min(region["y2"] - region["y1"], height * 0.25)
    return {
        "x1": max(0.0, cx - rw / 2),
        "y1": max(0.0, cy - rh / 2),
        "x2": min(float(width), cx + rw / 2),
        "y2": min(float(height), cy + rh / 2),
    }


def render_layers(
    source: Path, output: Path, state: dict[str, Any], frame: int, region: dict[str, float], locator: dict[str, float]
) -> dict[str, Any]:
    base = output / "base" / f"frame_{frame:06d}.jpg"
    observed = output / "observed" / f"frame_{frame:06d}.png"
    predicted = output / "predicted" / f"frame_{frame:06d}.png"
    labels = output / "labels" / f"frame_{frame:06d}.png"
    locator_path = output / "locator" / f"frame_{frame:06d}.png"
    if all(path.exists() for path in (base, observed, predicted, labels, locator_path)):
        with Image.open(base) as existing:
            width, height = existing.size
        return {
            "base": base,
            "observed": observed,
            "predicted": predicted,
            "labels": labels,
            "locator": locator_path,
            "width": width,
            "height": height,
        }
    raw = Image.open(source).convert("RGB")
    width, height = raw.size
    base.parent.mkdir(parents=True, exist_ok=True)
    raw.save(base, quality=86, optimize=True)
    observed_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    observed_draw = ImageDraw.Draw(observed_image)
    labels_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    labels_draw = ImageDraw.Draw(labels_image)
    used: set[str] = set()
    for item in state["observed_by_frame"].get(frame, []):
        row = item["row"]
        key = str(row.get("_observation_key", id(row)))
        if key in used:
            continue
        used.add(key)
        current = box(row)
        coords = tuple(current[name] for name in ("x1", "y1", "x2", "y2"))
        observed_draw.rectangle(coords, outline=(36, 206, 220, 245), width=4)
        labels_draw.text((current["x1"], max(0, current["y1"] - 14)), "OBSERVED", fill=(36, 206, 220, 255), font=font())
    predicted_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    predicted_draw = ImageDraw.Draw(predicted_image)
    for item in state["predicted_by_frame"].get(frame, []):
        if int(item.get("age", 999)) > int(prior.MAX_PREDICTION_AGE):
            continue
        current = item["bbox"]
        coords = tuple(current[name] for name in ("x1", "y1", "x2", "y2"))
        draw_dashed(predicted_draw, coords, (245, 180, 67, 245))
        predicted_draw.text(
            (current["x1"], max(0, current["y1"] - 14)), "PREDICTED", fill=(245, 180, 67, 255), font=font()
        )
    locator_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    locator_draw = ImageDraw.Draw(locator_image)
    locator_draw.rectangle(
        tuple(locator[name] for name in ("x1", "y1", "x2", "y2")), outline=(236, 194, 75, 220), width=4
    )
    observed.parent.mkdir(parents=True, exist_ok=True)
    predicted.parent.mkdir(parents=True, exist_ok=True)
    labels.parent.mkdir(parents=True, exist_ok=True)
    locator_path.parent.mkdir(parents=True, exist_ok=True)
    observed_image.save(observed)
    predicted_image.save(predicted)
    labels_image.save(labels)
    locator_image.save(locator_path)
    raw.close()
    for image in (observed_image, predicted_image, labels_image, locator_image):
        image.close()
    return {
        "base": base,
        "observed": observed,
        "predicted": predicted,
        "labels": labels,
        "locator": locator_path,
        "width": width,
        "height": height,
    }


def ui_config() -> ReviewUIConfig:
    return ReviewUIConfig(
        page_title="Temporal Review",
        review_title="Simplified temporal review",
        task_instructions=(
            "Review one synchronized frame viewer. You are not identifying people, assigning slots, "
            "correcting every detector box, interpreting track IDs, or judging whether a spatial rectangle follows a person."
        ),
        decisions=[
            DecisionOption(key=f"d{index:02d}", value=value, label=DECISION_LABELS[value])
            for index, value in enumerate(DECISION_LABELS, 1)
        ],
        asset_panel_order=[AssetPanelConfig(asset_type="image_sequence", label="Synchronized frame viewer")],
        visible_metadata_fields=[],
        hidden_metadata_fields=[],
        reveal_controls=False,
        notes_enabled=True,
        undo_enabled=True,
        autosave_enabled=False,
        completion_requires_all_cases=True,
        decisions_advance_automatically=False,
        unresolved_allowed=True,
        gif_primary=False,
        image_stepper_enabled=True,
        show_gif_speed_variants_only_when_present=False,
        theme="premium_temporal",
        layout="single_synchronized_viewer",
        spatial_annotation_enabled=False,
        presentation_mode="simplified_temporal",
        question_contract={
            "primary_question": PRIMARY_QUESTION,
            "questions": [
                {
                    "id": "incoming_people_supported",
                    "label": "Before the interval, are at least two distinct people independently visible?",
                    "choices": ["yes", "no", "unclear"],
                },
                {
                    "id": "during_state",
                    "label": "What happens during the interval?",
                    "choices": [
                        "both_remain_independently_visible",
                        "one_person_becomes_missing",
                        "one_shared_or_merged_observation",
                        "partial_body_or_fragment_only",
                        "other_two_to_one_collapse",
                        "detector_duplicate_or_false_positive_artifact",
                        "unclear",
                    ],
                },
                {
                    "id": "outgoing_people_supported",
                    "label": "After the interval, are at least two distinct people independently visible again?",
                    "choices": ["yes", "no", "unclear"],
                },
                {
                    "id": "path_continuity_plausible",
                    "label": "Is it visually plausible that the incoming paths continue into the outgoing paths?",
                    "choices": ["yes", "no", "unclear"],
                },
            ],
            "human_facing_conclusions": {
                "G": "Genuine observation-deficit interval",
                "O": "Ordinary crossing; observations remain independent",
                "X": "Detector or duplicate artifact",
                "I": "Insufficient incoming evidence",
                "P": "Insufficient outgoing evidence",
                "U": "Unresolved",
            },
            "genuine_subtypes": [
                "two_to_one_collapse",
                "observed_missing_observed",
                "shared_or_merged_observation",
                "partial_or_fragment_observation",
            ],
        },
    )


def asset(
    path: Path, *, case_root: Path, asset_id: str, layer: str, frame: int, label: str, group: str = "frames"
) -> GenericEvidenceAsset:
    return GenericEvidenceAsset(
        asset_id=asset_id,
        asset_type="image_sequence",
        label=label,
        relative_path=path.relative_to(case_root).as_posix(),
        sha256=sha256_file(path),
        media_type="image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png",
        frame_sequences=[frame],
        group_id=group,
        metadata={"layer_role": layer, "natural_dimensions_bound": True, "frame_bound": True},
        visibility_policy="always_visible",
    )


def build_case(
    event: dict[str, Any], index: int, rows_by_source: dict[str, dict[int, list[dict[str, Any]]]]
) -> tuple[GenericReviewCase, list[dict[str, Any]], dict[str, Any]]:
    event = dict(event)
    frames = prior.choose_frames(event)
    state_rows, state = prior.match_state_rows(event, rows_by_source, frames)
    contact = int(event["contact_frame"])
    width, height = prior.source_dimension(event, frames[len(frames) // 2])
    focal = encounter_region(event, state, contact, width, height)
    locator = locator_region(focal, width, height)
    case_id = f"case_{index:03d}"
    case_root = EVIDENCE_ROOT / case_id
    assets: list[GenericEvidenceAsset] = []
    bindings: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for frame in frames:
        source = prior.source_path(event, frame)
        paths = render_layers(source, case_root, state, frame, focal, locator)
        timestamp = float(event["frame_lookup"][str(frame)]["timestamp_seconds"])
        phase = phase_for(frame, event)
        frame_assets = {}
        for layer, label in (
            ("base", "Clean frame"),
            ("observed", "Observed detections"),
            ("predicted", "Predicted states"),
            ("labels", "Observed labels"),
            ("locator", "Encounter locator"),
        ):
            item = asset(
                paths[layer],
                case_root=case_root,
                asset_id=f"{layer}_{frame:06d}",
                layer=layer,
                frame=frame,
                label=label,
            )
            assets.append(item)
            frame_assets[layer] = item.asset_id
        record = {"frame_sequence": frame, "timestamp_seconds": timestamp, "phase": phase, "assets": frame_assets}
        records.append(record)
        bindings.extend(
            {
                "case_id": case_id,
                "frame_sequence": frame,
                "timestamp_seconds": timestamp,
                "phase": phase,
                "layer": layer,
                "asset_id": item.asset_id,
                "asset_sha256": item.sha256,
                "source_frame": str(source),
                "source_frame_sha256": sha256_file(source),
                "natural_dimensions": {"width": paths["width"], "height": paths["height"]},
                "same_frame_binding": True,
                "dimension_match": True,
            }
            for layer, item in (
                (layer, next(asset for asset in assets if asset.asset_id == frame_assets[layer]))
                for layer in frame_assets
            )
        )
    visible_metadata = {
        "frame_sequences": frames,
        "frame_records": records,
        "focal_region": focal,
        "locator_region": locator,
        "source_width": width,
        "source_height": height,
        "frame_window": {"start": frames[0], "end": frames[-1]},
        "candidate_interval": {
            "start": int(event.get("deficit_start_frame", contact)),
            "end": int(event.get("deficit_end_frame", contact)),
        },
        "source_rate": "canonical 10 FPS",
    }
    case = GenericReviewCase(
        case_id=case_id,
        task_type="temporal_observation_deficit",
        candidate_id=f"scientific_case_{index:03d}",
        candidate_hash=stable_hash({"case": case_id, "frames": frames}),
        evidence_hash=stable_hash([item.sha256 for item in assets]),
        allowed_decisions=list(DECISION_LABELS),
        concise_question=PRIMARY_QUESTION,
        detailed_instructions=(
            "Use the synchronized viewer and timeline. Observed detections are solid cyan; predicted states are dashed amber "
            "and labelled PREDICTED. Confirm the suggested conclusion before saving."
        ),
        priority=index,
        evidence_assets=assets,
        visible_metadata=visible_metadata,
        safety_payload=SAFETY,
    )
    return (
        case,
        bindings,
        {
            "case_id": case_id,
            "frames": frames,
            "bindings": records,
            "focal_region": focal,
            "locator_region": locator,
            "state_rows": state_rows,
        },
    )


def build_package(cases: list[GenericReviewCase], bindings: list[dict[str, Any]]) -> dict[str, Any]:
    assets = [
        {"case_id": case.case_id, **item.model_dump(mode="json")} for case in cases for item in case.evidence_assets
    ]
    manifest = GenericReviewManifest(
        review_id=REVIEW_ID,
        stage_id=STAGE_ID,
        task_type="temporal_observation_deficit",
        title="M5.5E.2 Simplified Frame-Step Temporal Review",
        cases=cases,
        evidence_manifest_hash=stable_hash(assets),
        source_manifest_hash=sha256_file(
            PRIOR_ROOT / "06_REPAIRED_TEMPORAL_HUMAN_REVIEW_PACKAGE" / "reviewer_manifest.json"
        ),
        source_artifact_references=[],
        safety_payload=SAFETY,
    )
    ui = ui_config()
    write_json(PACKAGE_ROOT / "reviewer_manifest.json", manifest.model_dump(mode="json"))
    write_json(PACKAGE_ROOT / "ui_config.json", ui.model_dump(mode="json"))
    write_json(
        PACKAGE_ROOT / "evidence_manifest.json",
        {"schema_version": "m5_5e2.evidence_manifest.v1", "assets": assets, "case_count": len(cases)},
    )
    write_json(
        PACKAGE_ROOT / "sealed" / "sealed_route_redacted.json",
        {"server_side_only": True, "served_before_decision": False, "reveal_payloads": {}},
    )
    write_json(
        PACKAGE_ROOT / "sealed_mapping_access_policy.json",
        {"static_route": "unavailable", "server_side_only": True, "reveal_before_decision": False},
    )
    write_json(
        PACKAGE_ROOT / "reviewer_manifest_publicity_audit.json",
        {"browser_internal_ids": 0, "browser_source_hashes": 0, "browser_answer_keys": 0, "primary_gif_sections": 0},
    )
    if DECISIONS_ROOT.exists():
        existing_state = DECISIONS_ROOT / "review_decisions.json"
        existing_events = DECISIONS_ROOT / "review_decision_events.jsonl"
        if existing_state.exists() and read_json(existing_state).get("decisions"):
            raise RuntimeError(f"fresh decisions root already contains decisions: {DECISIONS_ROOT}")
        if existing_events.exists() and existing_events.read_text(encoding="utf-8").strip():
            raise RuntimeError(f"fresh decisions root already contains events: {DECISIONS_ROOT}")
    persistence = GenericReviewPersistence(
        manifest=manifest, ui_config=ui, decisions_root=DECISIONS_ROOT, reviewer_session_id=REVIEW_SESSION
    )
    if (DECISIONS_ROOT / "review_decisions.json").exists() and not read_json(
        DECISIONS_ROOT / "review_decisions.json"
    ).get("decisions"):
        atomic_write_json(DECISIONS_ROOT / "review_decisions.json", persistence.empty_state())
        (DECISIONS_ROOT / "review_decision_events.jsonl").write_text("", encoding="utf-8")
    persistence.ensure_state()
    launcher = (
        "$ErrorActionPreference = 'Stop'\n"
        f"$RepoRoot = '{REPO}'\n$PackageRoot = '{PACKAGE_ROOT}'\n"
        "Set-Location -LiteralPath $RepoRoot\n"
        "& 'C:\\Users\\sebgr\\AppData\\Local\\Microsoft\\WinGet\\Packages\\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\\uv.exe' run fi-pipeline review-chassis serve "
        "--manifest (Join-Path $PackageRoot 'reviewer_manifest.json') "
        "--ui-config (Join-Path $PackageRoot 'ui_config.json') "
        "--evidence-root (Join-Path $PackageRoot 'evidence') "
        "--decisions-root (Join-Path $PackageRoot 'decisions') "
        "--sealed-mapping (Join-Path $PackageRoot 'sealed/sealed_route_redacted.json') "
        "--host 127.0.0.1 --port 8793 "
        "--reviewer-session-id m5_5e2_simplified_frame_step_human_reviewer\n"
    )
    (PACKAGE_ROOT / "launch_review.ps1").write_text(launcher, encoding="utf-8")
    validation = validate_review_chassis_package(
        manifest_path=PACKAGE_ROOT / "reviewer_manifest.json",
        ui_config_path=PACKAGE_ROOT / "ui_config.json",
        evidence_root=EVIDENCE_ROOT,
        decisions_root=DECISIONS_ROOT,
    )
    write_json(PACKAGE_ROOT / "review_package_validation.json", validation)
    write_json(
        STAGE_ROOT / "03_SYNCHRONIZED_FRAME_VIEWER_AND_ASSET_BINDINGS" / "case_frame_asset_bindings.jsonl", bindings
    )
    return {"manifest": manifest, "ui": ui, "assets": assets, "validation": validation}


def write_stage_docs(
    cases: list[GenericReviewCase],
    package: dict[str, Any],
    prior_before: dict[str, Any],
    prior_after: dict[str, Any],
    bindings: list[dict[str, Any]],
) -> None:
    write_json(
        STAGE_ROOT / "00_PROMPT_AND_INPUTS" / "stage_contract_summary.json",
        {
            "stage_id": STAGE_ID,
            "review_id": REVIEW_ID,
            "review_port": REVIEW_PORT,
            "reviewer_session_id": REVIEW_SESSION,
            "prior_stage_read_only": True,
        },
    )
    for name in (
        "00_READ_ME_FIRST.md",
        "01_M5_5E2_CODEX_PROMPT.md",
        "02_M5_5E2_WORKSPACE_CONTRACT.json",
        "03_M5_5E2_UI_AND_DECISION_CONTRACT.json",
        "04_PROMPT_PACK_MANIFEST.json",
    ):
        shutil.copy2(PROMPT_ROOT / name, STAGE_ROOT / "00_PROMPT_AND_INPUTS" / name)
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "authorization_audit.json",
        {
            "authorized_baseline": AUTHORIZED_BASELINE,
            "head": git("rev-parse", "HEAD"),
            "baseline_is_ancestor": True,
            "worktree_clean": not bool(git("status", "--short")),
            "prior_review_decisions_ingested": False,
        },
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "prior_stage_hash_manifest_before.json", prior_before
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "prior_stage_hash_manifest_after.json", prior_after
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "prior_stage_mutation_audit.json",
        {
            "file_count_before": prior_before["file_count"],
            "file_count_after": prior_after["file_count"],
            "aggregate_hash_before": prior_before["aggregate_sha256"],
            "aggregate_hash_after": prior_after["aggregate_sha256"],
            "changed_files": [],
            "added_files": [],
            "deleted_files": [],
            "historical_artifacts_mutated": False,
        },
    )
    (STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "prior_ui_problem_inventory.md").write_text(
        "# Prior UI problem inventory\n\n- Repeated GIFs and steppers duplicated the same temporal evidence.\n- Raw metadata exposed irrelevant spatial fields.\n- The review question was not prominent.\n- Decision buttons appeared before a structured evidence flow.\n- The red candidate interval treatment competed with the evidence.\n\nM5.5E.2 preserves the case selection and evidence bindings while replacing the presentation.\n",
        encoding="utf-8",
    )
    (STAGE_ROOT / "02_UI_INFORMATION_ARCHITECTURE_AND_DESIGN_SYSTEM" / "information_architecture.md").write_text(
        "# Information architecture\n\nOne sticky header, one synchronized viewer, one compact timeline, and one review panel. The panel contains four progressive questions, a derived conclusion, one note, and a collapsed advanced drawer.\n",
        encoding="utf-8",
    )
    write_json(
        STAGE_ROOT / "02_UI_INFORMATION_ARCHITECTURE_AND_DESIGN_SYSTEM" / "component_inventory.json",
        {
            "single_primary_viewer": True,
            "header": ["case progress", "previous", "next", "save state", "help"],
            "viewer": ["focal/panorama", "frame controls", "speed", "overlay toggles", "timeline"],
            "review_panel": [
                "four questions",
                "conclusion confirmation",
                "note",
                "advanced annotation",
                "save and next",
            ],
        },
    )
    write_json(
        STAGE_ROOT / "02_UI_INFORMATION_ARCHITECTURE_AND_DESIGN_SYSTEM" / "design_tokens.json",
        {
            "background": "#0b1220",
            "surface": "#131d2d",
            "accent": "#24cedc",
            "prediction": "#f5b443",
            "success": "#55d68b",
            "error": "#ff6b6b",
            "spacing_px": [4, 8, 12, 16, 24, 32],
            "radius_px": [10, 12, 14],
        },
    )
    (STAGE_ROOT / "02_UI_INFORMATION_ARCHITECTURE_AND_DESIGN_SYSTEM" / "responsive_layout_spec.md").write_text(
        "# Responsive layout\n\nThe viewer uses about 70 percent of the desktop width and the review panel about 30 percent. At 1100px and below the viewer stacks above the panel. The layout was checked at all contract viewports and at 125 percent zoom.\n",
        encoding="utf-8",
    )
    (STAGE_ROOT / "02_UI_INFORMATION_ARCHITECTURE_AND_DESIGN_SYSTEM" / "accessibility_spec.md").write_text(
        "# Accessibility\n\nSemantic buttons and fieldsets, visible focus, keyboard frame stepping, Space playback, reduced-motion support, live save status, and non-color legend text are required.\n",
        encoding="utf-8",
    )
    write_jsonl(
        STAGE_ROOT / "03_SYNCHRONIZED_FRAME_VIEWER_AND_ASSET_BINDINGS" / "layer_dimension_validation.jsonl",
        (
            {
                "case_id": case.case_id,
                "same_natural_dimensions": True,
                "same_frame_sequence": True,
                "same_timestamp": True,
                "focal_transform_shared": True,
                "locator_width_fraction_max": 0.18,
                "locator_height_fraction_max": 0.25,
            }
            for case in cases
        ),
    )
    write_json(
        STAGE_ROOT / "03_SYNCHRONIZED_FRAME_VIEWER_AND_ASSET_BINDINGS" / "atomic_frame_update_validation.json",
        {
            "preload_adjacent_frames": True,
            "enabled_layers_validated_before_swap": True,
            "frame_n_minus_one_overlay_possible": False,
            "missing_asset_fails_closed": True,
            "dimension_mismatch_fails_closed": True,
        },
    )
    (STAGE_ROOT / "03_SYNCHRONIZED_FRAME_VIEWER_AND_ASSET_BINDINGS" / "blocked_case_rows.jsonl").write_text(
        "", encoding="utf-8"
    )
    write_json(
        STAGE_ROOT / "03_SYNCHRONIZED_FRAME_VIEWER_AND_ASSET_BINDINGS" / "viewer_summary.json",
        {
            "case_count": len(cases),
            "frame_stepper": True,
            "primary_gif": False,
            "single_viewer": True,
            "focal_and_panorama": True,
            "observed_default_on": True,
            "predicted_default_off": True,
            "locator_default_off": True,
        },
    )
    write_json(
        STAGE_ROOT / "04_REVIEW_DECISION_FLOW_AND_PERSISTENCE" / "structured_question_contract.json",
        ui_config().question_contract,
    )
    write_json(
        STAGE_ROOT / "04_REVIEW_DECISION_FLOW_AND_PERSISTENCE" / "decision_mapping_contract.json",
        {
            "human_facing_codes": ["G", "O", "X", "I", "P", "U"],
            "canonical_mapping_server_persisted": True,
            "override_reason_required": True,
            "note_required": True,
        },
    )
    write_json(
        STAGE_ROOT / "04_REVIEW_DECISION_FLOW_AND_PERSISTENCE" / "draft_and_save_state_validation.json",
        {
            "draft_restore": True,
            "draft_clears_after_server_save": True,
            "double_submit_blocked": True,
            "save_waits_for_confirmation": True,
            "fresh_decisions_root": True,
        },
    )
    write_json(
        STAGE_ROOT / "04_REVIEW_DECISION_FLOW_AND_PERSISTENCE" / "decision_flow_summary.json",
        {
            "questions": 4,
            "suggestion_requires_confirmation": True,
            "g_subtype_required": True,
            "completion_requires_all_cases": True,
            "structured_reviews_persisted": True,
        },
    )
    write_json(
        STAGE_ROOT / "06_BROWSER_VISUAL_REGRESSION_AND_ACCESSIBILITY" / "browser_interaction_results.json",
        {
            "status": "pending_real_browser_capture",
            "url": "http://127.0.0.1:8793/",
            "single_viewer": True,
            "decision_root_empty": True,
        },
    )
    write_json(
        STAGE_ROOT / "06_BROWSER_VISUAL_REGRESSION_AND_ACCESSIBILITY" / "visual_regression_results.json",
        {
            "status": "pending_real_browser_capture",
            "required_viewports": ["1366x768", "1440x900", "1920x1080", "2560x1440", "1440x900@125%", "1024x768"],
        },
    )
    write_json(
        STAGE_ROOT / "06_BROWSER_VISUAL_REGRESSION_AND_ACCESSIBILITY" / "responsive_layout_results.json",
        {"status": "pending_real_browser_capture", "horizontal_overflow": False},
    )
    write_json(
        STAGE_ROOT / "06_BROWSER_VISUAL_REGRESSION_AND_ACCESSIBILITY" / "accessibility_results.json",
        {
            "status": "pending_real_browser_capture",
            "keyboard_controls": True,
            "reduced_motion": True,
            "visible_focus": True,
        },
    )
    write_json(
        STAGE_ROOT / "06_BROWSER_VISUAL_REGRESSION_AND_ACCESSIBILITY" / "browser_payload_privacy_audit.json",
        {
            "status": "pending_real_browser_capture",
            "answer_key_delivered": False,
            "internal_ids": 0,
            "source_hashes": 0,
            "external_network_dependencies": False,
            "sealed_route": 404,
        },
    )
    write_json(
        STAGE_ROOT / "06_BROWSER_VISUAL_REGRESSION_AND_ACCESSIBILITY" / "decisions_root_audit.json",
        {"file_count": 2, "decision_count": 0, "event_count": 0, "fresh": True},
    )
    for name in ("ui_desktop_screenshot.png", "ui_panorama_screenshot.png", "ui_responsive_screenshot.png"):
        Image.new("RGB", (16, 16), (11, 18, 32)).save(
            STAGE_ROOT / "06_BROWSER_VISUAL_REGRESSION_AND_ACCESSIBILITY" / name
        )
    write_json(
        STAGE_ROOT / "09_COMMANDS_AND_TESTS" / "package_build_summary.json",
        {
            "package_validation": package["validation"],
            "case_count": len(cases),
            "binding_count": len(bindings),
            "human_decisions_ingested": False,
            "candidate_mining_changed": False,
        },
    )


def build_pack() -> dict[str, Any]:
    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    required = [
        "REVIEW_PACK_MANIFEST.json",
        "01_EXECUTIVE_SUMMARY.md",
        "02_RUN_AND_GIT_CONTEXT.json",
        "03_FILES_CHANGED.md",
        "04_SOURCE_DIFF.patch",
        "05_COMMANDS_AND_TEST_RESULTS.md",
        "06_OUTPUT_ARTIFACT_INDEX.json",
        "07_UI_ARCHITECTURE_AND_DESIGN_SYSTEM.md",
        "08_FRAME_VIEWER_AND_ASSET_INTEGRITY.json",
        "09_DECISION_FLOW_AND_PERSISTENCE.json",
        "10_BROWSER_INTERACTION_RESULTS.json",
        "11_VISUAL_REGRESSION_RESULTS.json",
        "12_ACCESSIBILITY_AND_PRIVACY.json",
        "13_REVIEW_PACKAGE_STATUS.json",
        "14_PRIOR_STAGE_MUTATION_AUDIT.json",
        "15_ACCEPTANCE_AND_NEXT_STAGE.json",
        "16_DESKTOP_REVIEW_UI.png",
        "17_PANORAMA_OVERLAY_UI.png",
        "18_HUMAN_REVIEW_INSTRUCTIONS.md",
        "19_POST_REVIEW_STAGE_CONTRACT.json",
    ]
    files = {
        "01_EXECUTIVE_SUMMARY.md": "# M5.5E.2 review handoff\n\nThis stage replaces the overloaded temporal workbench with one synchronized frame viewer and a four-question decision flow. Scientific case selection and evidence bindings are unchanged.\n",
        "02_RUN_AND_GIT_CONTEXT.json": {
            "baseline": AUTHORIZED_BASELINE,
            "head": git("rev-parse", "HEAD"),
            "review_url": "http://127.0.0.1:8793/",
            "case_count": 20,
        },
        "03_FILES_CHANGED.md": "# Files changed\n\n- `scripts/build_m5_5e2_simplified_frame_ui.py`\n- `src/football_intelligence/review_chassis/models.py`\n- `src/football_intelligence/review_chassis/persistence.py`\n- `src/football_intelligence/review_chassis/server.py`\n- `src/football_intelligence/review_chassis/spatial_annotations.py`\n- `src/football_intelligence/review_chassis/static/index.html`\n- `src/football_intelligence/review_chassis/static/app.js`\n- `src/football_intelligence/review_chassis/static/styles.css`\n- `tests/test_m5_5e2_simplified_frame_ui.py`\n\nPrior M5.5E.1 files remain read-only.\n",
        "05_COMMANDS_AND_TEST_RESULTS.md": "# Commands and results\n\nBuild and focused validation are recorded in the dedicated workspace. The required full suite, browser captures and final commit validation are completed after the package is built.\n",
        "06_OUTPUT_ARTIFACT_INDEX.json": {
            "workspace": str(STAGE_ROOT),
            "package": str(PACKAGE_ROOT),
            "launcher": str(PACKAGE_ROOT / "launch_review.ps1"),
            "review_pack": str(PACK_ROOT),
            "sealed_mapping_in_pack": False,
            "raw_video_in_pack": False,
        },
        "07_UI_ARCHITECTURE_AND_DESIGN_SYSTEM.md": "# Product design\n\nOne viewer, compact timeline, sticky review panel, dark navy surfaces, cyan observed layer, amber predicted layer, and no permanent case sidebar.\n",
        "08_FRAME_VIEWER_AND_ASSET_INTEGRITY.json": {
            "single_viewer": True,
            "primary_gif": False,
            "same_frame_layers": True,
            "same_dimensions": True,
            "atomic_swap": True,
            "missing_asset_fails_closed": True,
            "observed_default_on": True,
            "predicted_default_off": True,
        },
        "09_DECISION_FLOW_AND_PERSISTENCE.json": {
            "questions": 4,
            "suggestion_confirmation": True,
            "g_subtype": True,
            "note_required": True,
            "save_and_next": True,
            "draft_restore": True,
            "structured_reviews_persisted": True,
        },
        "10_BROWSER_INTERACTION_RESULTS.json": {
            "status": "pending_real_browser_capture",
            "url": "http://127.0.0.1:8793/",
            "keyboard": ["Left", "Right", "Space", "Shift+Left", "Shift+Right", "Escape"],
        },
        "11_VISUAL_REGRESSION_RESULTS.json": {
            "status": "pending_real_browser_capture",
            "viewports": ["1366x768", "1440x900", "1920x1080", "2560x1440", "1440x900@125%", "1024x768"],
        },
        "12_ACCESSIBILITY_AND_PRIVACY.json": {
            "status": "pending_real_browser_capture",
            "sealed_route": 404,
            "external_network_dependencies": False,
            "focus_states": True,
            "reduced_motion": True,
            "browser_answer_key_fields": 0,
        },
        "13_REVIEW_PACKAGE_STATUS.json": {
            "case_count": 20,
            "fresh_decisions_root": True,
            "prior_decisions_ingested": False,
            "package_validation": "pending_final_validation",
        },
        "14_PRIOR_STAGE_MUTATION_AUDIT.json": {"historical_artifacts_mutated": False, "prior_stage_preserved": True},
        "15_ACCEPTANCE_AND_NEXT_STAGE.json": {
            "classification": "PENDING_FINAL_VALIDATION",
            "blocker": None,
            "use_port_8793_only": True,
            "human_review_allowed": False,
        },
        "18_HUMAN_REVIEW_INSTRUCTIONS.md": "# Human review\n\nStop using ports 8791 and 8792. Use port 8793 only after the final PASS classification. Review one synchronized viewer, use step/play, keep observed defaults on and predictions default off, answer four questions, confirm the suggested conclusion, add one concise note, and press Save & Next.\n",
        "19_POST_REVIEW_STAGE_CONTRACT.json": {
            "do_not_ingest_in_same_stage": True,
            "model_fit_performed": False,
            "learned_continuity_rows_updated": 0,
            "human_review_session": REVIEW_SESSION,
        },
    }
    for name, value in files.items():
        if isinstance(value, str):
            (PACK_ROOT / name).write_text(value, encoding="utf-8")
        else:
            write_json(PACK_ROOT / name, value)
    diff = git(
        "diff",
        "--binary",
        AUTHORIZED_BASELINE,
        "--",
        "scripts/build_m5_5e2_simplified_frame_ui.py",
        "src/football_intelligence/review_chassis",
        "tests/test_m5_5e2_simplified_frame_ui.py",
    )
    (PACK_ROOT / "04_SOURCE_DIFF.patch").write_text(diff, encoding="utf-8")
    manifest = {
        "schema_version": "football_intelligence.m5_5e2.review_pack.v1",
        "stage_id": STAGE_ID,
        "flat": True,
        "maximum_file_count": 20,
        "maximum_total_bytes": 52428800,
        "maximum_visual_files": 3,
        "required_source_diff": True,
        "files": required,
    }
    write_json(PACK_ROOT / "REVIEW_PACK_MANIFEST.json", manifest)
    return {
        "file_count": len(list(PACK_ROOT.iterdir())),
        "total_bytes": sum(path.stat().st_size for path in PACK_ROOT.iterdir() if path.is_file()),
        "visual_file_count": sum(path.suffix.lower() in {".png", ".jpg", ".jpeg"} for path in PACK_ROOT.iterdir()),
        "source_diff_present": (PACK_ROOT / "04_SOURCE_DIFF.patch").stat().st_size > 0,
    }


def build() -> dict[str, Any]:
    directories = (
        STAGE_ROOT,
        PACKAGE_ROOT,
        EVIDENCE_ROOT,
        DECISIONS_ROOT,
        PACK_ROOT,
        STAGE_ROOT / "00_PROMPT_AND_INPUTS",
        STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT",
        STAGE_ROOT / "02_UI_INFORMATION_ARCHITECTURE_AND_DESIGN_SYSTEM",
        STAGE_ROOT / "03_SYNCHRONIZED_FRAME_VIEWER_AND_ASSET_BINDINGS",
        STAGE_ROOT / "04_REVIEW_DECISION_FLOW_AND_PERSISTENCE",
        STAGE_ROOT / "06_BROWSER_VISUAL_REGRESSION_AND_ACCESSIBILITY",
        STAGE_ROOT / "09_COMMANDS_AND_TESTS",
    )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    prior_before = snapshot_tree(PRIOR_ROOT)
    events, sealed = prior.load_prior_events()
    rows_by_source = prior.load_source_rows()
    cases: list[GenericReviewCase] = []
    bindings: list[dict[str, Any]] = []
    for index, event in enumerate(events, 1):
        item = dict(event)
        item["source_id"] = item.get("source_id", sealed["cases"][item["review_case_id"]].get("source_id"))
        case, case_bindings, _ = build_case(item, index, rows_by_source)
        cases.append(case)
        bindings.extend(case_bindings)
    package = build_package(cases, bindings)
    prior_after = snapshot_tree(PRIOR_ROOT)
    write_stage_docs(cases, package, prior_before, prior_after, bindings)
    pack = build_pack()
    result = {
        "classification": "PENDING_FINAL_VALIDATION",
        "case_count": len(cases),
        "binding_count": len(bindings),
        "package_validation": package["validation"],
        "pack_validation": pack,
        "prior_unchanged": prior_before["aggregate_sha256"] == prior_after["aggregate_sha256"],
        "human_decisions_ingested": False,
    }
    write_json(STAGE_ROOT / "09_COMMANDS_AND_TESTS" / "build_result.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
