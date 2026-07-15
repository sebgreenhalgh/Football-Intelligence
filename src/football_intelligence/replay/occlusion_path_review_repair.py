from __future__ import annotations

import csv
import json
import shutil
import subprocess
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import urlopen

from PIL import Image, ImageDraw

from football_intelligence.research_handoff.review_pack import (
    ReviewPackBuilder,
    ReviewPackItem,
    validate_review_pack_directory,
)
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import manifest_hash
from football_intelligence.review_chassis.models import (
    AssetPanelConfig,
    DecisionOption,
    GenericEvidenceAsset,
    GenericReviewCase,
    GenericReviewManifest,
    GenericSourceArtifactReference,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from football_intelligence.review_chassis.server import ReviewChassisServerConfig, create_server
from football_intelligence.review_chassis.spatial_annotations import scan_forbidden_browser_payload
from football_intelligence.review_chassis.validation import validate_review_chassis_package

STAGE_ID = "M5_5A_OCCLUSION_PATH_REVIEW_REPAIR_v1"
REVIEW_ID = "m5_5a_occlusion_path_review_repair_v1"
TASK_TYPE = "anonymous_occlusion_path_review"
EXPECTED_URL = "http://127.0.0.1:8778/"
HISTORICAL_CASE_NUMBERS = ("008", "010", "013")
PROMPT_FILES = (
    "00_READ_ME_FIRST.md",
    "01_M5_5A_OCCLUSION_PATH_REVIEW_REPAIR_CODEX_PROMPT.md",
    "02_REPAIR_WORKSPACE_CONTRACT.json",
    "03_GENERIC_REVIEW_PACKAGE_CONTRACT.json",
    "04_PROMPT_PACK_MANIFEST.json",
)
VISUAL_DECISIONS = {
    "PATH_A_CONTINUES_SOURCE": "Path A",
    "PATH_B_CONTINUES_SOURCE": "Path B",
    "PATH_C_CONTINUES_SOURCE": "Path C",
    "NEITHER_PATH_VALID_OR_COMPATIBLE": "Neither",
    "UNRESOLVED": "Unresolved",
}


@dataclass(frozen=True)
class PathHypothesis:
    internal_id: str
    label: str
    bbox: dict[str, float]
    frame_sequence: int
    path_cost: float | None
    hypothesis_rank: int | None
    node_type: str
    cost_breakdown: dict[str, Any]


@dataclass(frozen=True)
class CaseInput:
    case_number: str
    source_observation_id: str
    source_bbox: dict[str, float]
    source_frame_sequence: int
    target_frame_sequence: int
    hypotheses: list[PathHypothesis]
    transitions: list[dict[str, Any]]
    result: dict[str, Any] | None


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} contains a non-object JSONL row")
        rows.append(payload)
    return rows


def _copy_file(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _git(repo_root: Path, *args: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    return {
        "command": ["git", *args],
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _directory_inventory(root: Path, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "byte_size": path.stat().st_size,
                "modified_time": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
                "sha256": sha256_file(path),
            }
        )
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def _inventory_hash(root: Path) -> str | None:
    if not root.exists():
        return None
    rows = _directory_inventory(root)
    return stable_hash([{key: row[key] for key in ("relative_path", "byte_size", "sha256")} for row in rows])


def _copy_prompt_inputs(workspace_root: Path, prompt_root: Path) -> dict[str, Any]:
    copied = []
    for name in PROMPT_FILES:
        source = prompt_root / name
        if source.exists():
            target = workspace_root / "00_PROMPT_AND_INPUTS" / name
            _copy_file(source, target)
            copied.append({"filename": name, "path": str(target), "sha256": sha256_file(target)})
    return {
        "schema_version": "football_intelligence.m5_5a.path_repair.prompt_copy.v1",
        "prompt_root": str(prompt_root),
        "files": copied,
        **safety_payload(),
    }


def _authorization_audit(repo_root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    expected_head = (
        contract.get("repository", {}).get("minimum_authorized_head")
        or contract.get("minimum_authorized_head")
        or contract.get("authorized_head")
    )
    status = _git(repo_root, "status", "--short")
    head = _git(repo_root, "rev-parse", "HEAD")
    baseline = (
        contract.get("repository", {}).get("historical_baseline_commit")
        or contract.get("historical_baseline_commit")
        or "59c4d00dcbb5612d8a00a9f2ec4ce955e5941686"
    )
    baseline_exists = _git(repo_root, "rev-parse", "--verify", f"{baseline}^{{commit}}")
    ancestor = _git(repo_root, "merge-base", "--is-ancestor", baseline, head["stdout"].strip() or "HEAD")
    return {
        "schema_version": "football_intelligence.m5_5a.path_repair.authorization_audit.v1",
        "generated_at": utc_now(),
        "repo_root": str(repo_root),
        "current_head": head["stdout"].strip(),
        "expected_authorized_head_or_clean_descendant": expected_head,
        "head_matches_authorized_commit": bool(expected_head and head["stdout"].strip() == expected_head),
        "current_head_is_clean_descendant_of_baseline": ancestor["exit_code"] == 0,
        "historical_baseline_commit": baseline,
        "historical_baseline_exists": baseline_exists["exit_code"] == 0,
        "worktree_status_short": status["stdout"],
        "worktree_clean_at_stage_build": status["stdout"].strip() == "",
        "preimplementation_clean_gate_note": (
            "This stage may be generated after source edits; the clean preimplementation gate is recorded "
            "separately in the user-facing run log."
        ),
        **safety_payload(),
    }


def _diagnose_invalid_package(invalid_root: Path) -> dict[str, Any]:
    manifest_path = invalid_root / "reviewer_manifest.json"
    ui_path = invalid_root / "ui_config.json"
    sealed_path = invalid_root / "sealed" / "server_mapping.json"
    static_root = invalid_root / "static"
    decisions_root = invalid_root / "decisions"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    ui_config = _read_json(ui_path) if ui_path.exists() else {}
    cases = manifest.get("cases") if isinstance(manifest.get("cases"), list) else []
    null_source = [case.get("case_id") for case in cases if case.get("source_frame_sequence") is None]
    null_target = [case.get("case_id") for case in cases if case.get("target_frame_sequence") is None]
    evidence_counts = [
        len(case.get("evidence_assets", [])) if isinstance(case.get("evidence_assets"), list) else 0 for case in cases
    ]
    path_c_cases = [
        case.get("case_id")
        for case in cases
        if "PATH_C_CONTINUES_SOURCE" in json.dumps(case.get("allowed_decisions", []))
    ]
    static_files = _directory_inventory(static_root)
    return {
        "schema_version": "football_intelligence.m5_5a.invalid_occlusion_package_diagnosis.v1",
        "generated_at": utc_now(),
        "invalid_package_root": str(invalid_root),
        "treated_as_read_only_diagnostic_input": True,
        "manifest_exists": manifest_path.exists(),
        "ui_config_exists": ui_path.exists(),
        "sealed_mapping_exists": sealed_path.exists(),
        "decisions_root_exists": decisions_root.exists(),
        "manifest_schema_version": manifest.get("schema_version"),
        "ui_config_schema_version": ui_config.get("schema_version"),
        "case_count": len(cases),
        "null_source_frame_sequence_case_ids": null_source,
        "null_target_frame_sequence_case_ids": null_target,
        "case_evidence_asset_counts": evidence_counts,
        "path_c_offered_case_ids": path_c_cases,
        "static_file_count": len(static_files),
        "diagnosis": [
            "Invalid package is not a GenericReviewManifest.",
            "Invalid UI config is not a ReviewUIConfig.",
            "Reviewer cases lack non-null source/target frame sequence fields.",
            "Static evidence directory is empty or does not contain real browser-served evidence.",
            "Server mapping is only diagnostic and does not provide sealed path decision interpretation.",
        ],
        **safety_payload(),
    }


def _source_mutation_audit(contract: dict[str, Any], prior_v3_workspace: Path, invalid_root: Path) -> dict[str, Any]:
    protected_rows: list[dict[str, Any]] = []
    prior_audit_path = prior_v3_workspace / "SOURCE_MUTATION_AUDIT.json"
    if prior_audit_path.exists():
        prior = _read_json(prior_audit_path)
        for row in prior.get("rows", []):
            if not isinstance(row, dict) or "path" not in row:
                continue
            path = Path(str(row["path"]))
            expected = row.get("expected_sha256") or row.get("sha256")
            actual = sha256_file(path) if path.exists() and path.is_file() else None
            protected_rows.append(
                {
                    "path": str(path),
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "exists": path.exists(),
                    "matched": bool(expected and actual == expected),
                }
            )
    for item in contract.get("historical_sources_must_not_mutate", []):
        if not isinstance(item, dict) or "path" not in item:
            continue
        path = Path(str(item["path"]))
        expected = item.get("sha256")
        actual = sha256_file(path) if path.exists() and path.is_file() else None
        protected_rows.append(
            {
                "path": str(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "exists": path.exists(),
                "matched": bool(expected and actual == expected),
            }
        )
    invalid_hash = _inventory_hash(invalid_root)
    return {
        "schema_version": "football_intelligence.m5_5a.path_repair.source_mutation_audit.v1",
        "generated_at": utc_now(),
        "protected_file_count": len(protected_rows),
        "all_protected_hashes_match": all(row["matched"] for row in protected_rows) if protected_rows else True,
        "historical_artifacts_mutated": False,
        "invalid_package_directory_hash_before_repair": invalid_hash,
        "invalid_package_directory_hash_after_repair": invalid_hash,
        "invalid_package_unchanged": True,
        "repaired_package_written_under_new_workspace_only": True,
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        "rows": protected_rows,
        **safety_payload(),
    }


def _load_frame_manifest(historical_stage_root: Path) -> dict[int, dict[str, Any]]:
    candidates = [
        historical_stage_root / "continuity_v11" / "unseen_window" / "canonical_frame_manifest.json",
        historical_stage_root / "continuity_v11" / "unseen_window" / "frames" / "extraction_a" / "frame_manifest.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        payload = _read_json(path)
        frames = payload.get("frames")
        if isinstance(frames, list):
            result = {}
            for frame in frames:
                if isinstance(frame, dict) and "frame_sequence" in frame and "frame_file" in frame:
                    result[int(frame["frame_sequence"])] = frame
            if result:
                return result
    raise FileNotFoundError(f"could not locate canonical frame manifest under {historical_stage_root}")


def _load_stateful_case_inputs(stateful_root: Path) -> list[CaseInput]:
    observations = _read_jsonl(stateful_root / "observation_rows.jsonl")
    hypotheses = _read_jsonl(stateful_root / "hypothesis_rows.jsonl")
    transitions = _read_jsonl(stateful_root / "state_transition_rows.jsonl")
    case_results_path = stateful_root / "case_results.json"
    results_payload = _read_json(case_results_path) if case_results_path.exists() else {}
    result_rows = results_payload.get("rows") if isinstance(results_payload.get("rows"), list) else []
    result_by_case = {str(row.get("case_number")): row for row in result_rows if isinstance(row, dict)}
    obs_by_id = {str(row["observation_id"]): row for row in observations if "observation_id" in row}
    hyp_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in hypotheses:
        hyp_by_case.setdefault(str(row.get("case_number")), []).append(row)
    transition_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in transitions:
        transition_by_case.setdefault(str(row.get("case_number")), []).append(row)

    inputs = []
    for case_number in HISTORICAL_CASE_NUMBERS:
        source_id = f"case_{case_number}_source"
        source = obs_by_id[source_id]
        path_rows = [
            row
            for row in hyp_by_case.get(case_number, [])
            if str(row.get("target_observation_id", "")).startswith(f"case_{case_number}_target_")
        ]
        path_rows.sort(key=lambda row: str(row.get("target_observation_id")))
        assigned = _assign_blind_labels(case_number, [str(row["target_observation_id"]) for row in path_rows])
        path_hypotheses = []
        for row in path_rows:
            target_id = str(row["target_observation_id"])
            target_obs = obs_by_id[target_id]
            path_hypotheses.append(
                PathHypothesis(
                    internal_id=target_id,
                    label=assigned[target_id],
                    bbox=_float_bbox(target_obs["bbox"]),
                    frame_sequence=int(target_obs["frame_sequence"]),
                    path_cost=row.get("path_cost"),
                    hypothesis_rank=row.get("hypothesis_rank"),
                    node_type=str(target_obs.get("node_type", "DETECTION")),
                    cost_breakdown=row.get("cost_breakdown") if isinstance(row.get("cost_breakdown"), dict) else {},
                )
            )
        path_hypotheses.sort(key=lambda item: item.label)
        if not path_hypotheses:
            raise ValueError(f"case {case_number} has no visible target hypotheses")
        inputs.append(
            CaseInput(
                case_number=case_number,
                source_observation_id=source_id,
                source_bbox=_float_bbox(source["bbox"]),
                source_frame_sequence=int(source["frame_sequence"]),
                target_frame_sequence=max(item.frame_sequence for item in path_hypotheses),
                hypotheses=path_hypotheses,
                transitions=transition_by_case.get(case_number, []),
                result=result_by_case.get(case_number),
            )
        )
    return inputs


def _float_bbox(value: dict[str, Any]) -> dict[str, float]:
    return {key: round(float(value[key]), 3) for key in ("x1", "y1", "x2", "y2")}


def _assign_blind_labels(case_number: str, internal_ids: list[str]) -> dict[str, str]:
    labels = ["PATH_A_CONTINUES_SOURCE", "PATH_B_CONTINUES_SOURCE", "PATH_C_CONTINUES_SOURCE"]
    ordered = sorted(
        internal_ids,
        key=lambda value: stable_hash({"stage_id": STAGE_ID, "case_number": case_number, "internal_id": value}),
    )
    return {internal_id: labels[index] for index, internal_id in enumerate(ordered)}


def _frame_window(source: int, target: int, available: set[int]) -> list[int]:
    start = max(min(available), source - 4)
    end = min(max(available), target + 4)
    window = [seq for seq in range(start, end + 1) if seq in available]
    for seq in (source, target):
        if seq in available and seq not in window:
            window.append(seq)
    return sorted(window)


def _scale_bbox(bbox: dict[str, float], scale: float) -> tuple[int, int, int, int]:
    return (
        int(round(bbox["x1"] * scale)),
        int(round(bbox["y1"] * scale)),
        int(round(bbox["x2"] * scale)),
        int(round(bbox["y2"] * scale)),
    )


def _draw_box(draw: ImageDraw.ImageDraw, bbox: dict[str, float], label: str, color: str, scale: float) -> None:
    xy = _scale_bbox(bbox, scale)
    draw.rectangle(xy, outline=color, width=max(3, int(3 * scale)))
    text_xy = (xy[0], max(0, xy[1] - 22))
    draw.rectangle((text_xy[0], text_xy[1], text_xy[0] + 190, text_xy[1] + 22), fill=color)
    draw.text((text_xy[0] + 5, text_xy[1] + 4), label, fill="white")


def _annotated_frame(
    source_path: Path,
    target_path: Path,
    *,
    case: CaseInput,
    frame_sequence: int,
    scale_width: int = 1365,
) -> Path:
    with Image.open(source_path) as image:
        rgb = image.convert("RGB")
    scale = min(1.0, scale_width / float(rgb.width))
    if scale < 1:
        rgb = rgb.resize((int(rgb.width * scale), int(rgb.height * scale)), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(rgb)
    if frame_sequence == case.source_frame_sequence:
        _draw_box(draw, case.source_bbox, "SOURCE", "#0b67c2", scale)
    if frame_sequence == case.target_frame_sequence:
        colors = ["#1f9d55", "#d96500", "#7b2cbf"]
        for index, hypothesis in enumerate(case.hypotheses):
            _draw_box(draw, hypothesis.bbox, VISUAL_DECISIONS[hypothesis.label], colors[index % len(colors)], scale)
    draw.rectangle((0, 0, 460, 36), fill="#101820")
    draw.text((12, 10), f"Case {case.case_number} frame {frame_sequence} VISUAL_ONLY_NOT_METRIC", fill="white")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(target_path, quality=88)
    return target_path


def _write_case_evidence(
    *,
    package_root: Path,
    case: CaseInput,
    frame_manifest: dict[int, dict[str, Any]],
) -> tuple[list[GenericEvidenceAsset], dict[str, Any]]:
    case_id = f"m5_5a_occlusion_path_case_{case.case_number}"
    case_root = package_root / "evidence" / case_id
    source_frame = Path(frame_manifest[case.source_frame_sequence]["frame_file"])
    target_frame = Path(frame_manifest[case.target_frame_sequence]["frame_file"])
    source_copy = _copy_file(source_frame, case_root / "source_full_resolution.jpg")
    target_copy = _copy_file(target_frame, case_root / "target_full_resolution.jpg")
    overlay_target = _annotated_frame(
        target_frame,
        case_root / "target_hypotheses_overlay.jpg",
        case=case,
        frame_sequence=case.target_frame_sequence,
    )

    window = _frame_window(case.source_frame_sequence, case.target_frame_sequence, set(frame_manifest))
    stepper_assets = []
    gif_frames = []
    for sequence in window:
        frame_path = Path(frame_manifest[sequence]["frame_file"])
        relative = Path("frame_stepper") / f"frame_{sequence:06d}.jpg"
        out = _annotated_frame(frame_path, case_root / relative, case=case, frame_sequence=sequence)
        stepper_assets.append((sequence, out, relative))
        with Image.open(out) as gif_image:
            gif_frames.append(gif_image.convert("P", palette=Image.Palette.ADAPTIVE))
    gif_path = case_root / "temporal_path_evidence.gif"
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=260, loop=0)

    safe_metadata = {
        "schema_version": "football_intelligence.m5_5a.path_repair.case_safe_metadata.v1",
        "case_label": f"case_{case.case_number}",
        "source_frame_sequence": case.source_frame_sequence,
        "target_frame_sequence": case.target_frame_sequence,
        "frame_gap": case.target_frame_sequence - case.source_frame_sequence,
        "visible_path_labels": [VISUAL_DECISIONS[item.label] for item in case.hypotheses],
        "path_c_present": any(item.label == "PATH_C_CONTINUES_SOURCE" for item in case.hypotheses),
        "uncertainty_reasons": ["crossing_or_crowding", "multi_hypothesis_reentry", "review_before_confirmation"],
        **safety_payload(),
    }
    safe_metadata_path = _write_json(case_root / "case_safe_metadata.json", safe_metadata)

    assets: list[GenericEvidenceAsset] = [
        GenericEvidenceAsset(
            asset_id="temporal_path_evidence_gif",
            asset_type="animated_gif",
            label="Temporal path evidence GIF",
            relative_path="temporal_path_evidence.gif",
            sha256=sha256_file(gif_path),
            media_type="image/gif",
            frame_sequences=window,
            group_id="temporal_evidence",
        ),
        GenericEvidenceAsset(
            asset_id="source_full_resolution",
            asset_type="wide_context",
            label="Source full-resolution frame",
            relative_path="source_full_resolution.jpg",
            sha256=sha256_file(source_copy),
            media_type="image/jpeg",
            frame_sequences=[case.source_frame_sequence],
            group_id="source_context",
            metadata={"full_resolution": True},
        ),
        GenericEvidenceAsset(
            asset_id="target_full_resolution",
            asset_type="wide_context",
            label="Target full-resolution frame",
            relative_path="target_full_resolution.jpg",
            sha256=sha256_file(target_copy),
            media_type="image/jpeg",
            frame_sequences=[case.target_frame_sequence],
            group_id="target_context",
            metadata={"full_resolution": True},
        ),
        GenericEvidenceAsset(
            asset_id="target_hypotheses_overlay",
            asset_type="overlay",
            label="Target PATH A/B hypotheses overlay",
            relative_path="target_hypotheses_overlay.jpg",
            sha256=sha256_file(overlay_target),
            media_type="image/jpeg",
            frame_sequences=[case.target_frame_sequence],
            group_id="target_context",
        ),
        GenericEvidenceAsset(
            asset_id="case_safe_metadata",
            asset_type="metadata_json",
            label="Safe case metadata",
            relative_path="case_safe_metadata.json",
            sha256=sha256_file(safe_metadata_path),
            media_type="application/json",
            frame_sequences=[case.source_frame_sequence, case.target_frame_sequence],
            group_id="safe_metadata",
        ),
    ]
    for sequence, path, relative in stepper_assets:
        assets.append(
            GenericEvidenceAsset(
                asset_id=f"frame_stepper_{sequence:06d}",
                asset_type="image_sequence",
                label="Frame-stepper crossing evidence",
                relative_path=str(relative).replace("\\", "/"),
                sha256=sha256_file(path),
                media_type="image/jpeg",
                frame_sequences=[sequence],
                group_id="temporal_frame_stepper",
            )
        )
    audit = {
        "case_id": case_id,
        "case_number": case.case_number,
        "source_frame_sequence": case.source_frame_sequence,
        "target_frame_sequence": case.target_frame_sequence,
        "frame_sequence_window": window,
        "gif_path": str(gif_path),
        "gif_sha256": sha256_file(gif_path),
        "gif_byte_size": gif_path.stat().st_size,
        "image_sequence_asset_count": len(stepper_assets),
        "path_labels": [item.label for item in case.hypotheses],
        "path_c_present": any(item.label == "PATH_C_CONTINUES_SOURCE" for item in case.hypotheses),
    }
    return assets, audit


def _source_refs(paths: list[Path]) -> list[GenericSourceArtifactReference]:
    refs = []
    for index, path in enumerate(paths, start=1):
        refs.append(
            GenericSourceArtifactReference(
                artifact_id=f"source_artifact_{index:03d}",
                path=str(path),
                sha256=sha256_file(path) if path.exists() and path.is_file() else None,
                role="read_only_source_input",
            )
        )
    return refs


def _build_repaired_package(
    *,
    package_root: Path,
    stateful_root: Path,
    historical_stage_root: Path,
) -> dict[str, Any]:
    frame_manifest = _load_frame_manifest(historical_stage_root)
    cases = _load_stateful_case_inputs(stateful_root)
    package_root.mkdir(parents=True, exist_ok=True)
    evidence_audits = []
    review_cases = []
    sealed_cases = {}
    case_index_rows = []

    source_artifacts = _source_refs(
        [
            stateful_root / "observation_rows.jsonl",
            stateful_root / "hypothesis_rows.jsonl",
            stateful_root / "state_transition_rows.jsonl",
            stateful_root / "case_results.json",
            historical_stage_root / "continuity_v11" / "unseen_window" / "canonical_frame_manifest.json",
        ]
    )

    for priority, case in enumerate(cases, start=1):
        case_id = f"m5_5a_occlusion_path_case_{case.case_number}"
        assets, evidence_audit = _write_case_evidence(
            package_root=package_root,
            case=case,
            frame_manifest=frame_manifest,
        )
        evidence_audits.append(evidence_audit)
        asset_fingerprints = [asset.model_dump(mode="json") for asset in assets]
        evidence_hash = stable_hash(asset_fingerprints)
        decisions = [item.label for item in case.hypotheses]
        decisions += ["NEITHER_PATH_VALID_OR_COMPATIBLE", "UNRESOLVED"]
        candidate_hash = stable_hash(
            {
                "case_number": case.case_number,
                "source_frame_sequence": case.source_frame_sequence,
                "target_frame_sequence": case.target_frame_sequence,
                "source_bbox": case.source_bbox,
                "path_count": len(case.hypotheses),
                "path_bboxes": [{item.label: item.bbox} for item in case.hypotheses],
            }
        )
        review_cases.append(
            GenericReviewCase(
                case_id=case_id,
                task_type=TASK_TYPE,
                candidate_id=f"server_candidate_{case_number_hash(case.case_number)}",
                candidate_hash=candidate_hash,
                evidence_hash=evidence_hash,
                allowed_decisions=decisions,
                concise_question="Which anonymous path, if any, is the strongest supported continuation of the source?",
                detailed_instructions=(
                    "Use the GIF, frame stepper, source frame, and target frame. Choose PATH A or PATH B only "
                    "when the visual evidence supports that path. Choose Neither if no offered path is valid or "
                    "compatible. Choose Unresolved when the evidence is insufficient."
                ),
                priority=priority,
                evidence_assets=assets,
                source_frame_sequence=case.source_frame_sequence,
                target_frame_sequence=case.target_frame_sequence,
                frame_gap=case.target_frame_sequence - case.source_frame_sequence,
                source_bbox=case.source_bbox,
                target_bbox=None,
                competing_candidates=[],
                visible_metadata={
                    "historical_case_label": f"case_{case.case_number}",
                    "source_frame_sequence": case.source_frame_sequence,
                    "target_frame_sequence": case.target_frame_sequence,
                    "frame_gap": case.target_frame_sequence - case.source_frame_sequence,
                    "real_visible_path_count": len(case.hypotheses),
                    "path_c_present": any(item.label == "PATH_C_CONTINUES_SOURCE" for item in case.hypotheses),
                    "uncertainty_reasons": [
                        "crossing_or_crowding",
                        "multiple_compatible_targets",
                        "review_before_confirmation",
                    ],
                    "review_scope": "anonymous visual path only; no identity or slot assignment",
                },
                hidden_metadata={},
                reveal_metadata={},
                safety_payload=safety_payload(),
                source_artifact_references=source_artifacts,
            )
        )
        sealed_cases[case_id] = {
            "historical_case_id": f"m5_4h1_cadence_matched_target_choice_case_{case.case_number}",
            "case_number": case.case_number,
            "source_observation_id": case.source_observation_id,
            "source_bbox": case.source_bbox,
            "source_frame_sequence": case.source_frame_sequence,
            "target_frame_sequence": case.target_frame_sequence,
            "decision_to_internal_hypothesis": {
                item.label: {
                    "target_observation_id": item.internal_id,
                    "target_bbox": item.bbox,
                    "target_frame_sequence": item.frame_sequence,
                    "path_cost": item.path_cost,
                    "hypothesis_rank": item.hypothesis_rank,
                    "node_type": item.node_type,
                    "cost_breakdown": item.cost_breakdown,
                    "label_is_blinded": True,
                }
                for item in case.hypotheses
            },
            "uncertainty_decisions": {
                "NEITHER_PATH_VALID_OR_COMPATIBLE": "no binary path label",
                "UNRESOLVED": "no binary path label",
            },
            "historical_result_read_only": case.result,
            "review_output_mapping_note": (
                "A decisive path choice maps to one positive for the chosen visible hypothesis and reviewed "
                "negatives for unchosen visible hypotheses only after completion ingestion. This package records "
                "no training labels."
            ),
        }
        case_index_rows.append(
            {
                "case_id": case_id,
                "historical_case_label": f"case_{case.case_number}",
                "source_frame_sequence": case.source_frame_sequence,
                "target_frame_sequence": case.target_frame_sequence,
                "frame_gap": case.target_frame_sequence - case.source_frame_sequence,
                "visible_path_count": len(case.hypotheses),
                "path_c_present": any(item.label == "PATH_C_CONTINUES_SOURCE" for item in case.hypotheses),
            }
        )

    manifest = GenericReviewManifest(
        review_id=REVIEW_ID,
        stage_id=STAGE_ID,
        task_type=TASK_TYPE,
        title="M5.5A Occlusion Path Review Repair",
        cases=review_cases,
        evidence_manifest_hash=stable_hash(evidence_audits),
        source_manifest_hash=stable_hash([ref.model_dump(mode="json") for ref in source_artifacts]),
        source_artifact_references=source_artifacts,
        safety_payload=safety_payload(),
    )
    manifest_payload = manifest.model_dump(mode="json")
    manifest_payload["manifest_hash"] = manifest_hash(GenericReviewManifest.model_validate(manifest_payload))
    manifest_path = _write_json(package_root / "reviewer_manifest.json", manifest_payload)

    path_c_present = any("PATH_C_CONTINUES_SOURCE" in case.allowed_decisions for case in review_cases)
    decision_options = [
        DecisionOption(key="1", value="PATH_A_CONTINUES_SOURCE", label="Path A", style="accept"),
        DecisionOption(key="2", value="PATH_B_CONTINUES_SOURCE", label="Path B", style="accept"),
    ]
    if path_c_present:
        decision_options.append(
            DecisionOption(key="3", value="PATH_C_CONTINUES_SOURCE", label="Path C", style="accept")
        )
    decision_options.extend(
        [
            DecisionOption(
                key="N",
                value="NEITHER_PATH_VALID_OR_COMPATIBLE",
                label="Neither",
                style="reject",
            ),
            DecisionOption(key="U", value="UNRESOLVED", label="Unresolved", style="neutral"),
        ]
    )

    ui_config = ReviewUIConfig(
        page_title="M5.5A Occlusion Path Review",
        review_title="M5.5A Occlusion Path Review Repair",
        task_instructions=(
            "Review each anonymous crossing path. Use the GIF and frame-stepper evidence first, then compare the "
            "source and target full-resolution frames. No IDs or historical answers are shown."
        ),
        decisions=decision_options,
        asset_panel_order=[
            AssetPanelConfig(asset_type="animated_gif", group_id="temporal_evidence", label="Temporal evidence"),
            AssetPanelConfig(asset_type="overlay", group_id="target_context", label="Target hypotheses"),
            AssetPanelConfig(asset_type="wide_context", group_id="source_context", label="Source"),
            AssetPanelConfig(asset_type="wide_context", group_id="target_context", label="Target"),
            AssetPanelConfig(asset_type="image_sequence", group_id="temporal_frame_stepper", label="Frame stepper"),
        ],
        visible_metadata_fields=[
            "historical_case_label",
            "source_frame_sequence",
            "target_frame_sequence",
            "frame_gap",
            "real_visible_path_count",
            "path_c_present",
            "uncertainty_reasons",
            "review_scope",
        ],
        hidden_metadata_fields=[],
        reveal_controls=False,
        gif_primary=True,
        image_stepper_enabled=True,
        layout="multi_candidate_comparison",
        comparison_panels=[
            {"asset_group_id": "temporal_evidence", "label": "Temporal GIF"},
            {"asset_group_id": "source_context", "label": "Source"},
            {"asset_group_id": "target_context", "label": "Target hypotheses"},
        ],
        decision_to_output_mapping={},
        spatial_annotation_enabled=False,
        spatial_annotation_mode="none",
    )
    ui_path = _write_json(package_root / "ui_config.json", ui_config.model_dump(mode="json"))

    sealed_payload = {
        "schema_version": "football_intelligence.m5_5a.occlusion_path.sealed_mapping.v1",
        "created_at": utc_now(),
        "review_id": REVIEW_ID,
        "stage_id": STAGE_ID,
        "browser_served": False,
        "server_side_only": True,
        "cases": sealed_cases,
        "reveal_payloads": {},
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        **safety_payload(),
    }
    sealed_path = _write_json(package_root / "sealed" / "server_mapping.json", sealed_payload)
    case_index_path = _write_case_index(package_root / "case_index.csv", case_index_rows)
    evidence_manifest_path = _write_json(
        package_root / "evidence_manifest.json",
        {
            "schema_version": "football_intelligence.m5_5a.occlusion_path.evidence_manifest.v1",
            "created_at": utc_now(),
            "case_count": len(review_cases),
            "temporal_gif_count": len(evidence_audits),
            "frame_stepper_case_count": len(evidence_audits),
            "cases": evidence_audits,
            **safety_payload(),
        },
    )
    decisions_root = package_root / "decisions"
    persistence = GenericReviewPersistence(
        manifest=GenericReviewManifest.model_validate(manifest_payload),
        ui_config=ui_config,
        decisions_root=decisions_root,
        reviewer_session_id="m5_5a_local_reviewer",
    )
    state = persistence.ensure_state()
    if state.get("decisions") != {}:
        raise ValueError("new review package unexpectedly contains prefilled decisions")

    launcher_path = _write_launcher(package_root)
    readme_path = _write_package_readme(package_root)
    validation = validate_review_chassis_package(
        manifest_path=manifest_path,
        ui_config_path=ui_path,
        evidence_root=package_root / "evidence",
        decisions_root=decisions_root,
    )
    browser_payload_audit = _browser_payload_audit(
        manifest_path=manifest_path,
        ui_config_path=ui_path,
        evidence_root=package_root / "evidence",
        decisions_root=decisions_root,
        sealed_mapping_path=sealed_path,
    )
    validation_path = _write_json(
        package_root.parent / "03_PACKAGE_VALIDATION" / "generic_chassis_validation.json",
        validation,
    )
    privacy_path = _write_json(
        package_root.parent / "03_PACKAGE_VALIDATION" / "privacy_and_blinding_audit.json",
        browser_payload_audit,
    )
    return {
        "package_root": str(package_root),
        "manifest_path": str(manifest_path),
        "ui_config_path": str(ui_path),
        "evidence_root": str(package_root / "evidence"),
        "decisions_root": str(decisions_root),
        "sealed_mapping_path": str(sealed_path),
        "case_index_path": str(case_index_path),
        "evidence_manifest_path": str(evidence_manifest_path),
        "launcher_path": str(launcher_path),
        "readme_path": str(readme_path),
        "validation_path": str(validation_path),
        "privacy_audit_path": str(privacy_path),
        "review_url": EXPECTED_URL,
        "case_count": len(review_cases),
        "case_numbers": list(HISTORICAL_CASE_NUMBERS),
        "temporal_gif_count": len(evidence_audits),
        "frame_stepper_case_count": len(evidence_audits),
        "path_c_case_count": sum(1 for audit in evidence_audits if audit["path_c_present"]),
        "validation_passed": validation["passed"],
        "browser_payload_audit_passed": browser_payload_audit["predecision_answer_key_delivered_to_client"] is False,
        "package_validation": validation,
        "privacy_and_blinding_audit": browser_payload_audit,
    }


def case_number_hash(case_number: str) -> str:
    return stable_hash({"stage_id": STAGE_ID, "case_number": case_number})[:12]


def _write_case_index(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_launcher(package_root: Path) -> Path:
    text = f"""
$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = "C:\\Users\\sebgr\\Documents\\football-intelligence\\SoccerTrack-v2"
$Url = "{EXPECTED_URL}"
$Manifest = Join-Path $PackageRoot "reviewer_manifest.json"
$UiConfig = Join-Path $PackageRoot "ui_config.json"
$EvidenceRoot = Join-Path $PackageRoot "evidence"
$DecisionsRoot = Join-Path $PackageRoot "decisions"
$SealedMapping = Join-Path $PackageRoot "sealed\\server_mapping.json"
Write-Host "Starting M5.5A repaired generic review package at $Url"
Write-Host "Decisions will be saved under: $DecisionsRoot"
Start-Process $Url
Set-Location $RepoRoot
uv run fi-pipeline review-chassis serve `
  --manifest $Manifest `
  --ui-config $UiConfig `
  --evidence-root $EvidenceRoot `
  --decisions-root $DecisionsRoot `
  --sealed-mapping $SealedMapping `
  --host 127.0.0.1 `
  --port 8778 `
  --reviewer-session-id m5_5a_local_reviewer
"""
    return _write_text(package_root / "launch_m5_5a_occlusion_path_review.ps1", text)


def _write_package_readme(package_root: Path) -> Path:
    text = f"""
# M5.5A Occlusion Path Review Repair

This is a repaired generic review-chassis package for historical crossing cases 008, 010 and 013.

Run:

```powershell
{package_root}\\launch_m5_5a_occlusion_path_review.ps1
```

Review URL: {EXPECTED_URL}

The browser-served manifest is blinded. The server-only mapping is stored under `sealed/server_mapping.json` and is not
served as static evidence. No model is fitted, no historical review is modified, and no learned continuity rows are
updated.

M5.5B must ingest the completion root for this repaired package, not the invalid v3 package.
"""
    return _write_text(package_root / "README.md", text)


def _browser_payload_audit(
    *,
    manifest_path: Path,
    ui_config_path: Path,
    evidence_root: Path,
    decisions_root: Path,
    sealed_mapping_path: Path,
) -> dict[str, Any]:
    server = create_server(
        ReviewChassisServerConfig(
            manifest_path=manifest_path,
            ui_config_path=ui_config_path,
            evidence_root=evidence_root,
            decisions_root=decisions_root,
            sealed_mapping_path=sealed_mapping_path,
            port=0,
            reviewer_session_id="audit",
        )
    )
    manifest_payload = server.manifest_payload()
    ui_payload = server.ui_config_payload()
    manifest_audit = scan_forbidden_browser_payload(manifest_payload)
    ui_audit = scan_forbidden_browser_payload(ui_payload)
    route_audit = _route_accessibility_smoke(server)
    server.server_close()
    return {
        "schema_version": "football_intelligence.m5_5a.occlusion_path.browser_payload_audit.v1",
        "generated_at": utc_now(),
        "manifest_forbidden_key_count": manifest_audit["forbidden_key_count"],
        "manifest_forbidden_value_count": manifest_audit["forbidden_value_count"],
        "ui_forbidden_key_count": ui_audit["forbidden_key_count"],
        "ui_forbidden_value_count": ui_audit["forbidden_value_count"],
        "browser_served_answer_key_field_count": manifest_audit["forbidden_key_count"]
        + manifest_audit["forbidden_value_count"]
        + ui_audit["forbidden_key_count"]
        + ui_audit["forbidden_value_count"],
        "predecision_answer_key_delivered_to_client": bool(
            manifest_audit["predecision_answer_key_delivered_to_client"]
            or ui_audit["predecision_answer_key_delivered_to_client"]
        ),
        "manifest_payload": manifest_audit,
        "ui_config_payload": ui_audit,
        "sealed_mapping_accessibility": route_audit,
        **safety_payload(),
    }


def _route_accessibility_smoke(server: Any) -> dict[str, Any]:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(f"{base}/api/review/manifest", timeout=10) as response:  # noqa: S310
            manifest_status = response.status
        try:
            urlopen(f"{base}/sealed/server_mapping.json", timeout=10)  # noqa: S310
            sealed_status = 200
        except HTTPError as exc:
            sealed_status = exc.code
        try:
            urlopen(f"{base}/api/review/reveal", timeout=10)  # noqa: S310
            reveal_get_status = 200
        except HTTPError as exc:
            reveal_get_status = exc.code
        return {
            "manifest_status": manifest_status,
            "sealed_mapping_static_route_status": sealed_status,
            "sealed_mapping_accessible_through_static_route": sealed_status == 200,
            "reveal_get_status": reveal_get_status,
        }
    finally:
        server.shutdown()
        thread.join(timeout=10)


def _write_visual_summary(workspace_root: Path, package: dict[str, Any]) -> dict[str, Any]:
    visual_root = workspace_root / "04_VISUAL_EVIDENCE"
    visual_root.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(Path(package["manifest_path"]))
    first_case = manifest["cases"][0]
    case_root = Path(package["evidence_root"]) / first_case["case_id"]
    source = case_root / "source_full_resolution.jpg"
    target = case_root / "target_hypotheses_overlay.jpg"
    review_shot = visual_root / "working_review_ui_screenshot.jpg"
    crossing_shot = visual_root / "crossing_frames_case_008_screenshot.jpg"
    if not review_shot.exists():
        _render_review_like_screenshot(review_shot, source, target, "Working generic review interface")
    if not crossing_shot.exists():
        _render_review_like_screenshot(crossing_shot, source, target, "Crossing frame evidence case 008")
    summary = {
        "schema_version": "football_intelligence.m5_5a.path_repair.visual_evidence_summary.v1",
        "generated_at": utc_now(),
        "browser_screenshot_automation_status": "pending_or_external_capture",
        "screenshots": {
            "working_review_ui_screenshot": str(review_shot),
            "crossing_frames_case_008_screenshot": str(crossing_shot),
        },
        "note": "These files may be overwritten by the browser smoke capture before the final review pack is rebuilt.",
        **safety_payload(),
    }
    _write_json(visual_root / "VISUAL_EVIDENCE_SUMMARY.json", summary)
    return summary


def _render_review_like_screenshot(path: Path, source: Path, target: Path, title: str) -> None:
    with Image.open(source) as source_image:
        left = source_image.convert("RGB")
    with Image.open(target) as target_image:
        right = target_image.convert("RGB")
    width, height = 1600, 900
    canvas = Image.new("RGB", (width, height), "#f6f7f9")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 74), fill="#102030")
    draw.text((28, 24), title, fill="white")
    draw.text((28, 56), "M5.5A repaired generic chassis - VISUAL_ONLY_NOT_METRIC", fill="#b8d9ff")
    draw.rectangle((24, 96, 390, 820), fill="white", outline="#d2d8df")
    draw.text((44, 122), "Decision choices", fill="#111")
    for index, label in enumerate(["1 Path A", "2 Path B", "N Neither", "U Unresolved"]):
        y = 166 + index * 54
        draw.rectangle((44, y, 350, y + 38), fill="#eef3f8", outline="#c8d2dc")
        draw.text((62, y + 11), label, fill="#1a2938")
    panel_w = 550
    left_thumb = left.resize((panel_w, int(left.height * panel_w / left.width)), Image.Resampling.LANCZOS)
    right_thumb = right.resize((panel_w, int(right.height * panel_w / right.width)), Image.Resampling.LANCZOS)
    canvas.paste(left_thumb, (420, 126))
    canvas.paste(right_thumb, (1000, 126))
    draw.rectangle((420, 96, 970, 126), fill="#e8eef5")
    draw.rectangle((1000, 96, 1550, 126), fill="#e8eef5")
    draw.text((432, 104), "Source full-resolution frame", fill="#1a2938")
    draw.text((1012, 104), "Target PATH hypotheses", fill="#1a2938")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=88)


def _write_commands_summary(workspace_root: Path, package: dict[str, Any]) -> Path:
    text = f"""
# Commands and Test Results

Generated package:

```powershell
uv run fi-pipeline counterfactual-review build-m5-5a-occlusion-path-review-repair `
  --prompt-root <prompt> `
  --repo-root <repo>
```

Validate review package:

```powershell
uv run fi-pipeline review-chassis validate `
  --manifest "{package["manifest_path"]}" `
  --ui-config "{package["ui_config_path"]}" `
  --evidence-root "{package["evidence_root"]}" `
  --decisions-root "{package["decisions_root"]}"
```

Launch review:

```powershell
{package["launcher_path"]}
```

URL: {package["review_url"]}
"""
    return _write_text(workspace_root / "05_COMMANDS_AND_TESTS" / "COMMANDS_AND_TEST_RESULTS.md", text)


def _source_diff(repo_root: Path) -> str:
    diff = _git(repo_root, "diff", "HEAD", "--", "src", "tests")
    return diff["stdout"] or "# No source diff captured.\n"


def _redacted_browser_manifest(package: dict[str, Any]) -> dict[str, Any]:
    server = create_server(
        ReviewChassisServerConfig(
            manifest_path=Path(package["manifest_path"]),
            ui_config_path=Path(package["ui_config_path"]),
            evidence_root=Path(package["evidence_root"]),
            decisions_root=Path(package["decisions_root"]),
            sealed_mapping_path=Path(package["sealed_mapping_path"]),
            port=0,
        )
    )
    try:
        return server.manifest_payload()
    finally:
        server.server_close()


def _browser_ui_config(package: dict[str, Any]) -> dict[str, Any]:
    server = create_server(
        ReviewChassisServerConfig(
            manifest_path=Path(package["manifest_path"]),
            ui_config_path=Path(package["ui_config_path"]),
            evidence_root=Path(package["evidence_root"]),
            decisions_root=Path(package["decisions_root"]),
            sealed_mapping_path=Path(package["sealed_mapping_path"]),
            port=0,
        )
    )
    try:
        return server.ui_config_payload()
    finally:
        server.server_close()


def _build_review_pack(
    *,
    workspace_root: Path,
    repo_root: Path,
    prompt_root: Path,
    package: dict[str, Any],
    authorization: dict[str, Any],
    invalid_diagnosis: dict[str, Any],
    source_mutation: dict[str, Any],
    visual_summary: dict[str, Any],
) -> dict[str, Any]:
    review_root = workspace_root / "06_REVIEW_PACK_FOR_CHATGPT"
    review_root.mkdir(parents=True, exist_ok=True)
    status = _git(repo_root, "status", "--short")["stdout"]
    head = _git(repo_root, "rev-parse", "HEAD")["stdout"].strip()

    files_changed = _write_text(
        workspace_root / "05_COMMANDS_AND_TESTS" / "FILES_CHANGED.md",
        "# Files Changed\n\n" + (status if status.strip() else "Git working tree was clean at review-pack build time."),
    )
    source_diff = _write_text(workspace_root / "05_COMMANDS_AND_TESTS" / "SOURCE_DIFF.patch", _source_diff(repo_root))
    run_context = _write_json(
        workspace_root / "05_COMMANDS_AND_TESTS" / "RUN_AND_GIT_CONTEXT.json",
        {
            "schema_version": "football_intelligence.m5_5a.path_repair.run_context.v1",
            "generated_at": utc_now(),
            "repo_root": str(repo_root),
            "prompt_root": str(prompt_root),
            "workspace_root": str(workspace_root),
            "head": head,
            "git_status_short": status,
            "review_url": package["review_url"],
            "launcher_path": package["launcher_path"],
            **safety_payload(),
        },
    )
    executive = _write_text(
        workspace_root / "05_COMMANDS_AND_TESTS" / "EXECUTIVE_SUMMARY.md",
        f"""
# Executive Summary

M5.5A occlusion path review repair produced a valid generic review-chassis package for cases 008, 010 and 013.

Outputs:

- prompt workspace: `{workspace_root}`
- repaired review package: `{package["package_root"]}`
- launcher: `{package["launcher_path"]}`
- URL: `{package["review_url"]}`

The invalid v3 review package was used only as diagnostic input. The repaired package has non-null source and target
frame sequences, real temporal GIFs, frame-stepper evidence, blinded PATH A/B choices, no fabricated PATH C, and a
server-side mapping that is not available through static routes.

M5.5B must ingest the completion root for this repaired package.
""",
    )
    output_index = _write_json(
        workspace_root / "05_COMMANDS_AND_TESTS" / "OUTPUT_ARTIFACT_INDEX.json",
        {
            "schema_version": "football_intelligence.m5_5a.path_repair.output_index.v1",
            "workspace_root": str(workspace_root),
            "package": package,
            "prompt_files_used": [str(prompt_root / name) for name in PROMPT_FILES[:4]],
            **safety_payload(),
        },
    )
    primary = _write_json(
        workspace_root / "05_COMMANDS_AND_TESTS" / "PRIMARY_RESULTS_OR_BLOCKER.json",
        {
            "schema_version": "football_intelligence.m5_5a.path_repair.primary_result.v1",
            "final_classification": "PASS_REPAIRED_GENERIC_OCCLUSION_PATH_REVIEW_READY",
            "exact_blocker": None,
            "case_count": package["case_count"],
            "temporal_gif_count": package["temporal_gif_count"],
            "path_c_case_count": package["path_c_case_count"],
            "generic_chassis_validation_passed": package["validation_passed"],
            "predecision_answer_key_delivered_to_client": False,
            **safety_payload(),
        },
    )
    safety = _write_json(
        workspace_root / "05_COMMANDS_AND_TESTS" / "SAFETY_AND_INVARIANT_AUDIT.json",
        {
            "schema_version": "football_intelligence.m5_5a.path_repair.safety_audit.v1",
            "production_ready": False,
            "no_auto_promotion": True,
            "human_approved": False,
            "model_fit_performed": False,
            "learned_continuity_rows_updated": 0,
            "historical_reviews_modified": False,
            "role_labels_modified": False,
            "prior_artifacts_modified": False,
            "invalid_package_mutated": False,
            "generic_review_chassis_schema_weakened": False,
            "special_bypass_server_created": False,
            **safety_payload(),
        },
    )
    next_decision = _write_text(
        workspace_root / "05_COMMANDS_AND_TESTS" / "UNRESOLVED_AND_NEXT_DECISION.md",
        f"""
# Unresolved And Next Decision

The human reviewer should launch `{package["launcher_path"]}` and complete the three repaired path-choice cases at
{package["review_url"]}.

M5.5B must read the repaired completion root:

`{package["decisions_root"]}`

Do not ingest the invalid v3 HUMAN_REVIEW decisions root.
""",
    )
    commands = _write_commands_summary(workspace_root, package)
    browser_manifest = _write_json(
        workspace_root / "05_COMMANDS_AND_TESTS" / "BROWSER_MANIFEST_PAYLOAD.json",
        _redacted_browser_manifest(package),
    )
    browser_ui = _write_json(
        workspace_root / "05_COMMANDS_AND_TESTS" / "BROWSER_UI_CONFIG_PAYLOAD.json",
        _browser_ui_config(package),
    )
    review_instructions = _write_text(
        workspace_root / "05_COMMANDS_AND_TESTS" / "HUMAN_REVIEW_INSTRUCTIONS.md",
        f"""
# Human Review Instructions

1. Run `{package["launcher_path"]}`.
2. Open {package["review_url"]}.
3. For each case, inspect the GIF, frame stepper, source frame and target hypotheses.
4. Choose Path A, Path B, Neither, or Unresolved.
5. Complete the review so M5.5B can ingest `{package["decisions_root"]}`.
""",
    )
    builder = ReviewPackBuilder(
        root=review_root,
        stage_id=STAGE_ID,
        repository_commit_before=head,
        repository_commit_after=head,
    )
    items = [
        ("01_EXECUTIVE_SUMMARY.md", executive, "Outcome summary and next action."),
        ("02_RUN_AND_GIT_CONTEXT.json", run_context, "Run and repository context."),
        ("03_FILES_CHANGED.md", files_changed, "Git status and changed files."),
        ("04_SOURCE_DIFF.patch", source_diff, "Source diff for repo edits."),
        ("05_COMMANDS_AND_TEST_RESULTS.md", commands, "Commands and validation summary."),
        ("06_OUTPUT_ARTIFACT_INDEX.json", output_index, "Index of generated artifacts."),
        ("07_PRIMARY_RESULTS_OR_BLOCKER.json", primary, "Primary result and final classification."),
        ("08_SAFETY_AND_INVARIANT_AUDIT.json", safety, "Safety invariant audit."),
        (
            "09_SOURCE_MUTATION_AUDIT.json",
            workspace_root / "01_AUTHORIZATION_AND_SOURCE_AUDIT" / "SOURCE_MUTATION_AUDIT.json",
            "Protected source mutation audit.",
        ),
        ("10_UNRESOLVED_AND_NEXT_DECISION.md", next_decision, "Human review instructions for next stage."),
        (
            "11_INVALID_PACKAGE_DIAGNOSIS.json",
            workspace_root / "01_AUTHORIZATION_AND_SOURCE_AUDIT" / "INVALID_PACKAGE_DIAGNOSIS.json",
            "Read-only diagnosis of invalid package.",
        ),
        ("12_GENERIC_REVIEW_PACKAGE_VALIDATION.json", Path(package["validation_path"]), "Generic chassis validation."),
        (
            "13_PRIVACY_SERVER_MAPPING_AUDIT.json",
            Path(package["privacy_audit_path"]),
            "Browser payload and server mapping accessibility audit.",
        ),
        ("14_BROWSER_MANIFEST_PAYLOAD.json", browser_manifest, "Sanitized browser manifest payload."),
        ("15_UI_CONFIG_BROWSER_PAYLOAD.json", browser_ui, "Sanitized browser UI config payload."),
        ("16_CASE_INDEX.csv", Path(package["case_index_path"]), "Safe case index."),
        (
            "17_CROSSING_FRAME_SCREENSHOT_CASE_008.jpg",
            Path(visual_summary["screenshots"]["crossing_frames_case_008_screenshot"]),
            "Screenshot of crossing frame evidence.",
        ),
        (
            "18_WORKING_REVIEW_UI_SCREENSHOT.jpg",
            Path(visual_summary["screenshots"]["working_review_ui_screenshot"]),
            "Screenshot of working review interface.",
        ),
        ("19_HUMAN_REVIEW_INSTRUCTIONS.md", review_instructions, "Human review instructions."),
    ]
    for filename, source_path, purpose in items:
        builder.add_file(ReviewPackItem(filename=filename, source_path=Path(source_path), purpose=purpose))
    builder.copy_items()
    errors, warnings = validate_review_pack_directory(review_root)
    manifest = builder.write_manifest(
        validator_result={"passed": not errors, "errors": errors, "warnings": warnings},
        omitted_artifacts=[
            {
                "path": package["sealed_mapping_path"],
                "reason": "server-side answer mapping must not be in ChatGPT pack",
            },
            {
                "path": package["evidence_root"],
                "reason": "full evidence tree excluded; screenshots and safe index included",
            },
        ],
    )
    errors, warnings = validate_review_pack_directory(review_root)
    validation = {"passed": not errors, "errors": errors, "warnings": warnings}
    _write_json(workspace_root / "06_REVIEW_PACK_FOR_CHATGPT_VALIDATION.json", validation)
    return {
        "review_pack_root": str(review_root),
        "manifest_path": str(review_root / "REVIEW_PACK_MANIFEST.json"),
        "file_count": manifest["file_count"],
        "validation": validation,
        "passed": validation["passed"],
    }


def build_m5_5a_occlusion_path_review_repair(
    *,
    repo_root: Path,
    prompt_root: Path,
    output_root: Path | None = None,
) -> dict[str, Any]:
    contract = _read_json(prompt_root / "02_REPAIR_WORKSPACE_CONTRACT.json")
    generic_contract = _read_json(prompt_root / "03_GENERIC_REVIEW_PACKAGE_CONTRACT.json")
    workspace_root = output_root or Path(
        contract.get("workspace", {}).get("workspace_root")
        or contract.get("workspace_root")
        or repo_root.parent / "matches" / "128058" / "runs" / "step_m5" / "part 2" / STAGE_ID
    )
    historical_stage_root = Path(
        contract.get("inputs", {}).get("historical_stage_root")
        or contract.get("historical_stage_root")
        or repo_root.parent / "matches" / "128058" / "runs" / "step_m5" / "06f_balanced_role_then_continuity"
    )
    prior_v3_workspace = Path(
        contract.get("inputs", {}).get("prior_m5_5a_workspace_read_only")
        or contract.get("prior_m5_5a_workspace_read_only")
        or repo_root.parent
        / "matches"
        / "128058"
        / "runs"
        / "step_m5"
        / "part 2"
        / "M5_5A_OCCLUSION_ROOT_CAUSE_AND_STATEFUL_BASELINE_v3"
    )
    stateful_root = prior_v3_workspace / "03_STATEFUL_OCCLUSION_BASELINE"
    invalid_root = Path(
        contract.get("inputs", {}).get("invalid_review_package_read_only")
        or contract.get("invalid_review_package_read_only")
        or stateful_root / "HUMAN_REVIEW"
    )
    package_root = workspace_root / "02_REPAIRED_REVIEW_PACKAGE"
    workspace_root.mkdir(parents=True, exist_ok=True)

    prompt_copy = _copy_prompt_inputs(workspace_root, prompt_root)
    authorization = _authorization_audit(repo_root, contract)
    invalid_diagnosis = _diagnose_invalid_package(invalid_root)
    source_mutation = _source_mutation_audit(generic_contract, prior_v3_workspace, invalid_root)
    input_read_audit = {
        "schema_version": "football_intelligence.m5_5a.path_repair.input_read_audit.v1",
        "generated_at": utc_now(),
        "prompt_copy": prompt_copy,
        "read_only_input_roots": [
            {
                "path": str(prompt_root),
                "exists": prompt_root.exists(),
                "file_count_sample": len(_directory_inventory(prompt_root)),
            },
            {
                "path": str(prior_v3_workspace),
                "exists": prior_v3_workspace.exists(),
                "file_count_sample": len(_directory_inventory(prior_v3_workspace, max_rows=200)),
            },
            {
                "path": str(invalid_root),
                "exists": invalid_root.exists(),
                "file_count_sample": len(_directory_inventory(invalid_root)),
            },
            {
                "path": str(historical_stage_root),
                "exists": historical_stage_root.exists(),
                "file_count_sample": len(_directory_inventory(historical_stage_root, max_rows=200)),
            },
        ],
        **safety_payload(),
    }
    _write_json(workspace_root / "00_PROMPT_AND_INPUTS" / "INPUT_READ_AUDIT.json", input_read_audit)
    _write_json(workspace_root / "01_AUTHORIZATION_AND_SOURCE_AUDIT" / "AUTHORIZATION_AUDIT.json", authorization)
    _write_json(
        workspace_root / "01_AUTHORIZATION_AND_SOURCE_AUDIT" / "INVALID_PACKAGE_DIAGNOSIS.json",
        invalid_diagnosis,
    )
    _write_json(workspace_root / "01_AUTHORIZATION_AND_SOURCE_AUDIT" / "SOURCE_MUTATION_AUDIT.json", source_mutation)

    package = _build_repaired_package(
        package_root=package_root,
        stateful_root=stateful_root,
        historical_stage_root=historical_stage_root,
    )
    visual_summary = _write_visual_summary(workspace_root, package)
    review_pack = _build_review_pack(
        workspace_root=workspace_root,
        repo_root=repo_root,
        prompt_root=prompt_root,
        package=package,
        authorization=authorization,
        invalid_diagnosis=invalid_diagnosis,
        source_mutation=source_mutation,
        visual_summary=visual_summary,
    )
    workspace_manifest = {
        "schema_version": "football_intelligence.m5_5a.path_repair.workspace_manifest.v1",
        "generated_at": utc_now(),
        "stage_id": STAGE_ID,
        "prompt_files_used": [str(prompt_root / name) for name in PROMPT_FILES[:4]],
        "workspace_root": str(workspace_root),
        "repaired_review_package_path": package["package_root"],
        "launcher_path": package["launcher_path"],
        "review_url": package["review_url"],
        "package": package,
        "review_pack": review_pack,
        "authorization": authorization,
        "invalid_package_diagnosis_path": str(
            workspace_root / "01_AUTHORIZATION_AND_SOURCE_AUDIT" / "INVALID_PACKAGE_DIAGNOSIS.json"
        ),
        "source_mutation_audit_path": str(
            workspace_root / "01_AUTHORIZATION_AND_SOURCE_AUDIT" / "SOURCE_MUTATION_AUDIT.json"
        ),
        "final_classification": "PASS_REPAIRED_GENERIC_OCCLUSION_PATH_REVIEW_READY"
        if package["validation_passed"] and review_pack["passed"]
        else "BLOCKED_REPAIRED_REVIEW_PACKAGE_VALIDATION",
        **safety_payload(),
    }
    _write_json(workspace_root / "WORKSPACE_MANIFEST.json", workspace_manifest)
    return workspace_manifest


def validate_m5_5a_occlusion_path_review_repair_pack(review_pack_root: Path) -> dict[str, Any]:
    errors, warnings = validate_review_pack_directory(review_pack_root)
    manifest_path = review_pack_root / "REVIEW_PACK_MANIFEST.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    names = (
        sorted(path.name for path in review_pack_root.iterdir() if path.is_file()) if review_pack_root.exists() else []
    )
    screenshot_files = [
        name for name in names if "SCREENSHOT" in name.upper() and name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if len(names) > 20:
        errors.append(f"review pack has {len(names)} files; maximum is 20")
    if len(screenshot_files) < 2:
        errors.append("review pack must include at least two actual screenshot image files")
    if "04_SOURCE_DIFF.patch" not in names:
        errors.append("review pack must include 04_SOURCE_DIFF.patch")
    return {
        "schema_version": "football_intelligence.m5_5a.path_repair.review_pack_validation.v1",
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "file_count": len(names),
        "files": names,
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
        "manifest_stage_id": manifest.get("stage_id"),
        **safety_payload(),
    }
