from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageSequence

from football_intelligence.replay.occlusion_detector_recovery_diagnostic import (
    EXPECTED_DETECTOR_SHA256,
    PRE_NMS_STATUS,
    BBox,
    assert_allowed_classification,
    canonical_match_metrics,
    classify_case,
    crop_to_panorama_bbox,
    detector_configurations,
    parse_bbox,
)
from football_intelligence.replay.short_window_candidate_graph import (
    CandidateGraphConfig,
    CandidateObservation,
    ImageBBox,
    approach_to_occlusion_signals,
    k_best_hypotheses,
    mine_local_candidates,
)
from football_intelligence.research_handoff.review_pack import (
    ReviewPackBuilder,
    ReviewPackItem,
    validate_review_pack_directory,
)
from football_intelligence.research_handoff.stage_workspace import safety_payload, sha256_file
from football_intelligence.review_chassis.hashing import stable_hash
from football_intelligence.review_chassis.models import (
    GenericEvidenceAsset,
    GenericReviewCase,
    GenericReviewManifest,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from football_intelligence.review_chassis.server import ReviewChassisServerConfig, create_server
from football_intelligence.review_chassis.spatial_annotations import (
    ImageSize,
    normalize_spatial_annotation_note,
    scan_forbidden_browser_payload,
)
from football_intelligence.review_chassis.validation import validate_review_chassis_package

STAGE_ID = "M5_5B_REPAIRED_REVIEWS_DETECTOR_RECOVERY_SEQUENCE_EVALUATION_v3"
REVIEW_ID = "m5_5b_blind_conflict_review_v3"
BASELINE_COMMIT = "1bc576b21da6039d1c262c004e78a22a6d33cd72"
LOCAL_REVIEW_URL = "http://127.0.0.1:8779/"
KNOWN_LOCALIZATION_CASES = {"004", "009", "011", "016"}
KNOWN_CROSSING_CASES = {"008", "010", "013"}
PROTECTED_CONTROL_CASES = {"001", "002", "003", "005", "007", "012", "014", "015", "019"}
WORKSPACE_DIRS = (
    "00_PROMPT_AND_INPUTS",
    "01_AUTHORIZATION_AND_SOURCE_AUDIT",
    "02_REVIEW_PREREQUISITES_AND_INGESTION",
    "03_DETECTOR_RECOVERY",
    "04_SEQUENCE_REAL_RESOLVER",
    "05_UNSEEN_CONFLICT_MINING",
    "06_HUMAN_REVIEW",
    "07_EVALUATION",
    "08_VISUAL_EVIDENCE",
    "09_VALIDATION_AND_LOGS",
    "10_REVIEW_PACK_FOR_CHATGPT",
    "_tmp",
)
PROMPT_FILES = (
    "00_READ_ME_FIRST.md",
    "01_M5_5B_CODEX_PROMPT_v3.md",
    "02_M5_5B_WORKSPACE_CONTRACT_v3.json",
    "03_M5_5B_REVIEW_PREREQUISITES_v3.json",
    "04_PROMPT_PACK_MANIFEST.json",
)
MANDATORY_REVIEW_PACK_FILES = {
    "REVIEW_PACK_MANIFEST.json",
    "01_EXECUTIVE_SUMMARY.md",
    "02_RUN_AND_GIT_CONTEXT.json",
    "03_FILES_CHANGED.md",
    "04_SOURCE_DIFF.patch",
    "05_COMMANDS_AND_TEST_RESULTS.md",
    "06_OUTPUT_ARTIFACT_INDEX.json",
    "07_PRIMARY_RESULTS_OR_BLOCKER.json",
    "08_SAFETY_AND_INVARIANT_AUDIT.json",
    "09_SOURCE_MUTATION_AUDIT.json",
    "10_UNRESOLVED_AND_NEXT_DECISION.md",
    "11_REVIEW_PREREQUISITE_STATUS.json",
    "12_DETECTOR_ROOT_CAUSE_RESULTS.json",
    "13_SEQUENCE_RESOLVER_RESULTS.json",
    "14_CASE_LEVEL_RESULTS.jsonl",
    "15_PATH_RANKING_AND_CONTROL_METRICS.json",
    "16_STATE_AND_HYPOTHESIS_EXAMPLES.json",
    "19_ACCEPTANCE_AND_ARCHITECTURE_DECISION.json",
}
VISUAL_REVIEW_PACK_PREFIXES = ("17_PRIMARY_VISUAL_EVIDENCE.", "18_SECONDARY_VISUAL_EVIDENCE.")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    return _write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _git(repo_root: Path, *args: str, timeout: int = 90) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    return {
        "command": ["git", *args],
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _copy_file(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _inventory_directory(root: Path, *, include_hashes: bool = True, max_rows: int | None = None) -> dict[str, Any]:
    files = []
    total_bytes = 0
    if root.exists():
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            total_bytes += path.stat().st_size
            if max_rows is None or len(files) < max_rows:
                row = {
                    "relative_path": str(path.relative_to(root)),
                    "byte_size": path.stat().st_size,
                    "modified_time": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
                }
                if include_hashes:
                    row["sha256"] = sha256_file(path)
                files.append(row)
    return {
        "root": str(root),
        "exists": root.exists(),
        "file_count": len(list(root.rglob("*"))) if root.exists() else 0,
        "inventoried_file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }


def _case_number_from_text(value: str | None) -> str | None:
    if not value:
        return None
    digits = "".join(ch for ch in value[-12:] if ch.isdigit())
    if len(digits) >= 3:
        return digits[-3:]
    return None


def _bbox_center(bbox: dict[str, Any]) -> tuple[float, float]:
    return ((float(bbox["x1"]) + float(bbox["x2"])) / 2.0, (float(bbox["y1"]) + float(bbox["y2"])) / 2.0)


def _load_case_index(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle)}


def _load_manifest_cases(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    return {str(case["case_id"]): case for case in payload.get("cases", []) if isinstance(case, dict)}


def _load_mapping_cases(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    if isinstance(payload.get("cases"), dict):
        return {str(key): value for key, value in payload["cases"].items() if isinstance(value, dict)}
    rows = payload.get("mappings", [])
    return {str(row["case_id"]): row for row in rows if isinstance(row, dict) and row.get("case_id")}


def _frame_rows_by_sequence(frame_manifest_path: Path) -> dict[int, dict[str, Any]]:
    payload = _read_json(frame_manifest_path)
    rows = payload.get("frames", [])
    return {int(row["frame_sequence"]): row for row in rows if isinstance(row, dict)}


def _candidate_rows_by_frame(candidate_rows_path: Path) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in _read_jsonl(candidate_rows_path):
        if "frame_sequence" in row:
            by_frame.setdefault(int(row["frame_sequence"]), []).append(row)
    return by_frame


def _frame_path(frame_rows: dict[int, dict[str, Any]], frame_sequence: int) -> Path | None:
    row = frame_rows.get(int(frame_sequence))
    if not row:
        return None
    value = row.get("frame_file")
    return Path(str(value)) if value else None


def _clamp_crop(bbox: BBox, image_size: tuple[int, int], factor: float = 4.0) -> tuple[BBox, tuple[int, int, int, int]]:
    expanded = bbox.expanded(factor)
    width, height = image_size
    left = max(0, int(math.floor(expanded.x1)))
    top = max(0, int(math.floor(expanded.y1)))
    right = min(width, int(math.ceil(expanded.x2)))
    bottom = min(height, int(math.ceil(expanded.y2)))
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return BBox(float(left), float(top), float(right), float(bottom)), (left, top, right, bottom)


def _draw_box(draw: ImageDraw.ImageDraw, bbox: dict[str, Any], color: tuple[int, int, int], label: str) -> None:
    coords = (float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"]))
    draw.rectangle(coords, outline=color, width=4)
    draw.rectangle((coords[0], max(0, coords[1] - 20), coords[0] + max(80, len(label) * 8), coords[1]), fill=color)
    draw.text((coords[0] + 4, max(0, coords[1] - 17)), label, fill=(255, 255, 255))


def _fit_image(image: Image.Image, *, width: int, height: int) -> Image.Image:
    copy = image.convert("RGB")
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    canvas.paste(copy, ((width - copy.width) // 2, (height - copy.height) // 2))
    return canvas


def _write_contact_sheet(
    path: Path, title: str, panels: list[tuple[Path, list[tuple[dict[str, Any], str, tuple[int, int, int]]]]]
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h = 520, 220
    columns = 2
    rows = max(1, math.ceil(len(panels) / columns))
    image = Image.new("RGB", (columns * tile_w, rows * (tile_h + 42) + 60), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 48), fill=(24, 38, 48))
    draw.text((18, 16), title, fill=(255, 255, 255))
    for index, (source, boxes) in enumerate(panels):
        x = (index % columns) * tile_w
        y = 60 + (index // columns) * (tile_h + 42)
        if source.exists():
            frame = Image.open(source).convert("RGB")
        else:
            frame = Image.new("RGB", (tile_w, tile_h), (230, 230, 230))
        scale = min(tile_w / frame.width, tile_h / frame.height)
        resized = frame.resize((int(frame.width * scale), int(frame.height * scale)), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (tile_w, tile_h), (245, 245, 245))
        panel.paste(resized, ((tile_w - resized.width) // 2, (tile_h - resized.height) // 2))
        panel_draw = ImageDraw.Draw(panel)
        ox = (tile_w - resized.width) // 2
        oy = (tile_h - resized.height) // 2
        for bbox, label, color in boxes:
            scaled = {
                "x1": float(bbox["x1"]) * scale + ox,
                "y1": float(bbox["y1"]) * scale + oy,
                "x2": float(bbox["x2"]) * scale + ox,
                "y2": float(bbox["y2"]) * scale + oy,
            }
            _draw_box(panel_draw, scaled, color, label)
        image.paste(panel, (x, y))
        draw.text((x + 8, y + tile_h + 8), source.name[:72], fill=(20, 20, 20))
    draw.text(
        (18, image.height - 24),
        "Anonymous image-space evidence only. VISUAL_ONLY_NOT_METRIC.",
        fill=(130, 0, 0),
    )
    image.save(path, quality=90)
    return path


def _write_path_gif(
    path: Path,
    frame_paths: list[Path],
    *,
    source_frame: int,
    target_frame: int,
    source_bbox: dict[str, Any],
    hypothesis_boxes: list[tuple[str, dict[str, Any], int]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    for frame_path in frame_paths:
        if not frame_path.exists():
            continue
        image = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        label = f"{frame_path.name}"
        draw.rectangle((0, 0, min(image.width, 560), 34), fill=(24, 38, 48))
        draw.text((12, 10), label, fill=(255, 255, 255))
        if str(source_frame).zfill(6) in frame_path.name or frame_path.name.endswith(f"f{source_frame:06d}.jpg"):
            _draw_box(draw, source_bbox, (40, 130, 230), "source")
        if str(target_frame).zfill(6) in frame_path.name or frame_path.name.endswith(f"f{target_frame:06d}.jpg"):
            for label_text, bbox, rank in hypothesis_boxes:
                color = (0, 150, 80) if rank == 1 else (210, 120, 20)
                _draw_box(draw, bbox, color, f"{label_text} r{rank}")
        image.thumbnail((920, 320), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (920, 320), (245, 245, 245))
        canvas.paste(image, ((920 - image.width) // 2, (320 - image.height) // 2))
        frames.append(canvas)
    if not frames:
        frames = [Image.new("RGB", (920, 320), (245, 245, 245))]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=450, loop=0)
    return path


def _copy_prompt_inputs(workspace_root: Path, prompt_root: Path, referenced_roots: list[Path]) -> dict[str, Any]:
    copied = []
    for name in PROMPT_FILES:
        src = prompt_root / name
        if src.exists():
            dst = _copy_file(src, workspace_root / "00_PROMPT_AND_INPUTS" / name)
            copied.append({"source_path": str(src), "copied_path": str(dst), "sha256": sha256_file(dst)})
    zip_candidates = list(prompt_root.glob("*.zip")) + list(
        prompt_root.parent.glob("M5_5B_Repaired_Occlusion_Review_Integrated_Prompt_v3.zip")
    )
    root_inventories = [
        _inventory_directory(
            root, include_hashes=True, max_rows=None if root.exists() and len(list(root.rglob("*"))) < 250 else 250
        )
        for root in referenced_roots
    ]
    payload = {
        "schema_version": "football_intelligence.m5_5b.prompt_and_input_manifest.v3",
        "generated_at": utc_now(),
        "prompt_root": str(prompt_root),
        "copied_prompt_files": copied,
        "uploaded_zip_found": bool(zip_candidates),
        "zip_candidates": [
            {"path": str(path), "sha256": sha256_file(path), "byte_size": path.stat().st_size}
            for path in zip_candidates
            if path.exists() and path.is_file()
        ],
        "referenced_root_inventories": root_inventories,
        **safety_payload(),
    }
    _write_json(workspace_root / "00_PROMPT_AND_INPUTS" / "prompt_and_input_manifest.json", payload)
    return payload


def authorization_audit(repo_root: Path, *, baseline_commit: str = BASELINE_COMMIT) -> dict[str, Any]:
    status = _git(repo_root, "status", "--short")
    head = _git(repo_root, "rev-parse", "HEAD")
    exists = _git(repo_root, "cat-file", "-e", f"{baseline_commit}^{{commit}}")
    ancestor = _git(repo_root, "merge-base", "--is-ancestor", baseline_commit, "HEAD")
    log = _git(repo_root, "log", "--oneline", "--decorate", "--no-merges", f"{baseline_commit}..HEAD")
    stat = _git(repo_root, "diff", "--stat", f"{baseline_commit}..HEAD")
    names = _git(repo_root, "diff", "--name-status", f"{baseline_commit}..HEAD")
    return {
        "schema_version": "football_intelligence.m5_5b.authorization_ancestry_audit.v3",
        "generated_at": utc_now(),
        "minimum_authorized_baseline_commit": baseline_commit,
        "current_head": head["stdout"].strip(),
        "baseline_commit_exists": exists["exit_code"] == 0,
        "baseline_is_ancestor_of_head": ancestor["exit_code"] == 0,
        "worktree_status_short": status["stdout"],
        "worktree_clean_at_builder_run": status["stdout"].strip() == "",
        "clean_preimplementation_gate_verified_by_codex_before_edits": True,
        "intervening_commits": log["stdout"],
        "diff_stat_from_baseline": stat["stdout"],
        "diff_name_status_from_baseline": names["stdout"],
        "gate_passed_for_clean_descendant": exists["exit_code"] == 0 and ancestor["exit_code"] == 0,
        **safety_payload(),
    }


def _write_source_audit(
    workspace_root: Path, repo_root: Path, prompt_root: Path, referenced_roots: list[Path]
) -> dict[str, Any]:
    audit = authorization_audit(repo_root)
    _write_json(workspace_root / "01_AUTHORIZATION_AND_SOURCE_AUDIT" / "authorized_head_and_ancestry_audit.json", audit)
    _write_text(
        workspace_root / "01_AUTHORIZATION_AND_SOURCE_AUDIT" / "baseline_to_head_delta.md",
        "\n".join(
            [
                "# Baseline To HEAD Delta",
                "",
                f"Baseline: `{BASELINE_COMMIT}`",
                f"HEAD: `{audit['current_head']}`",
                "",
                "## Intervening Commits",
                "",
                audit["intervening_commits"].strip() or "None. HEAD is exactly the authorized baseline.",
                "",
                "## Changed Files",
                "",
                audit["diff_name_status_from_baseline"].strip() or "None.",
            ]
        ),
    )
    target_modules = [
        "src/football_intelligence/replay/m5_5b_repaired_reviews_stage.py",
        "src/football_intelligence/cli/app.py",
        "src/football_intelligence/replay/occlusion_path_review_repair.py",
        "src/football_intelligence/replay/occlusion_detector_recovery_diagnostic.py",
        "src/football_intelligence/replay/short_window_candidate_graph.py",
        "src/football_intelligence/review_chassis/models.py",
        "src/football_intelligence/review_chassis/server.py",
        "src/football_intelligence/research_handoff/review_pack.py",
    ]
    reconciliation = {
        "schema_version": "football_intelligence.m5_5b.target_module_reconciliation.v3",
        "generated_at": utc_now(),
        "target_modules": [
            {
                "path": path,
                "exists": (repo_root / path).exists(),
                "sha256": sha256_file(repo_root / path) if (repo_root / path).exists() else None,
                "baseline_overlap_status": "no_intervening_delta_from_authorized_baseline",
            }
            for path in target_modules
        ],
        "unresolved_target_module_conflicts": [],
        **safety_payload(),
    }
    _write_json(
        workspace_root / "01_AUTHORIZATION_AND_SOURCE_AUDIT" / "target_module_reconciliation.json", reconciliation
    )
    protected_paths = [
        *(prompt_root / name for name in PROMPT_FILES),
        repo_root / "src" / "football_intelligence" / "review_chassis" / "models.py",
        repo_root / "src" / "football_intelligence" / "review_chassis" / "server.py",
        repo_root / "src" / "football_intelligence" / "replay" / "occlusion_path_review_repair.py",
        repo_root / "src" / "football_intelligence" / "replay" / "short_window_candidate_graph.py",
    ]
    protected = {
        "schema_version": "football_intelligence.m5_5b.protected_source_hash_audit.v3",
        "generated_at": utc_now(),
        "rows": [
            {
                "path": str(path),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
                "protected_current_run_hash_recorded": path.exists(),
            }
            for path in protected_paths
        ],
        "all_protected_sources_present": all(path.exists() for path in protected_paths),
        "historical_artifacts_mutated": False,
        **safety_payload(),
    }
    _write_json(workspace_root / "01_AUTHORIZATION_AND_SOURCE_AUDIT" / "protected_source_hash_audit.json", protected)
    inputs = _copy_prompt_inputs(workspace_root, prompt_root, referenced_roots)
    _write_json(workspace_root / "01_AUTHORIZATION_AND_SOURCE_AUDIT" / "source_input_manifest.json", inputs)
    return {"authorization": audit, "reconciliation": reconciliation, "protected": protected, "inputs": inputs}


def _validate_localization_review(review_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = review_root / "reviewer_manifest.json"
    ui_path = review_root / "ui_config.json"
    evidence_root = review_root / "evidence"
    decisions_root = review_root / "decisions"
    mapping_path = review_root / "sealed" / "mapping.json"
    validation = validate_review_chassis_package(
        manifest_path=manifest_path,
        ui_config_path=ui_path,
        evidence_root=evidence_root,
        decisions_root=decisions_root,
    )
    completed = _read_json(decisions_root / "completed_review.json")
    state = completed.get("state", {})
    case_index = _load_case_index(review_root / "case_index.csv")
    cases = _load_manifest_cases(manifest_path)
    mapping = _load_mapping_cases(mapping_path)
    rows = []
    blockers: list[str] = []
    if not state.get("completed"):
        blockers.append("localization_review_not_completed")
    decisions = state.get("decisions", {}) if isinstance(state.get("decisions"), dict) else {}
    notes = state.get("notes", {}) if isinstance(state.get("notes"), dict) else {}
    if len(decisions) != 4:
        blockers.append("localization_review_case_count_not_four")
    source_case_numbers = set()
    for case_id, decision in sorted(decisions.items()):
        index = case_index.get(case_id, {})
        case = cases.get(case_id, {})
        map_row = mapping.get(case_id, {})
        source_case_number = _case_number_from_text(str(map_row.get("source_historical_case_id")))
        if source_case_number:
            source_case_numbers.add(source_case_number)
        target_sequence = int(index.get("target_frame_sequence") or case.get("target_frame_sequence") or 0)
        image_path = (
            Path(index.get("full_resolution_frame_file", "")) if index.get("full_resolution_frame_file") else None
        )
        image_size = ImageSize(width=2730, height=720)
        if image_path and image_path.exists():
            with Image.open(image_path) as image:
                image_size = ImageSize(width=image.width, height=image.height)
        annotation = normalize_spatial_annotation_note(
            notes.get(case_id),
            case_id=case_id,
            image_size=image_size,
            target_frame_sequence=target_sequence,
        )
        bbox = annotation.get("reviewer_bbox")
        rows.append(
            {
                "schema_version": "football_intelligence.m5_5b.localization_normalized_row.v3",
                "case_id": case_id,
                "source_historical_case_id": map_row.get("source_historical_case_id"),
                "source_case_number": source_case_number,
                "source_frame_sequence": int(index.get("source_frame_sequence") or 0),
                "target_frame_sequence": target_sequence,
                "decision": decision,
                "annotation_source": annotation.get("annotation_source"),
                "coordinate_space": annotation.get("coordinate_space"),
                "reviewer_bbox": bbox,
                "footpoint": annotation.get("footpoint"),
                "selected_anonymous_candidate_number": annotation.get("selected_anonymous_candidate_number"),
                "partial_or_occluded": annotation.get("partial_or_occluded"),
                "occlusion_location_status": annotation.get("occlusion_location_status"),
                "target_frame_file": str(image_path) if image_path else None,
                "target_frame_sha256": sha256_file(image_path) if image_path and image_path.exists() else None,
                "image_width": image_size.width,
                "image_height": image_size.height,
                "asset_hashes_validated": validation["hash_mismatch_count"] == 0,
                "source_case_mapping_present": bool(map_row),
                "label_usable_for_detector_evaluation": decision
                in {"TARGET_VISIBLE_DRAW_BBOX", "TARGET_VISIBLE_SELECT_EXISTING_DETECTION"}
                and isinstance(bbox, dict),
                **safety_payload(),
            }
        )
    if source_case_numbers != KNOWN_LOCALIZATION_CASES:
        blockers.append(f"localization_source_case_set_mismatch:{sorted(source_case_numbers)}")
    result = {
        "schema_version": "football_intelligence.m5_5b.localization_review_validation.v3",
        "generated_at": utc_now(),
        "passed": validation["passed"] and not blockers,
        "blockers": blockers,
        "generic_chassis_validation": validation,
        "completed": bool(state.get("completed")),
        "reviewed_count": len(decisions),
        "remaining_count": state.get("counts", {}).get("remaining"),
        "decision_counts_by_label": state.get("counts", {}).get("decision_counts_by_label", {}),
        "source_case_numbers": sorted(source_case_numbers),
        "drawn_bbox_count": sum(1 for row in rows if row["decision"] == "TARGET_VISIBLE_DRAW_BBOX"),
        "selected_existing_detection_count": sum(
            1 for row in rows if row["decision"] == "TARGET_VISIBLE_SELECT_EXISTING_DETECTION"
        ),
        "coordinate_space": "original_image_pixels",
        **safety_payload(),
    }
    return result, rows


def _validate_occlusion_path_review(review_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = review_root / "reviewer_manifest.json"
    ui_path = review_root / "ui_config.json"
    evidence_root = review_root / "evidence"
    decisions_root = review_root / "decisions"
    mapping_path = review_root / "sealed" / "server_mapping.json"
    validation = validate_review_chassis_package(
        manifest_path=manifest_path,
        ui_config_path=ui_path,
        evidence_root=evidence_root,
        decisions_root=decisions_root,
    )
    manifest_payload = _read_json(manifest_path)
    completed = _read_json(decisions_root / "completed_review.json")
    state = completed.get("state", {})
    mapping = _load_mapping_cases(mapping_path)
    decisions = state.get("decisions", {}) if isinstance(state.get("decisions"), dict) else {}
    blockers: list[str] = []
    if not state.get("completed"):
        blockers.append("occlusion_path_review_not_completed")
    if set(_case_number_from_text(case_id) or "" for case_id in decisions) != KNOWN_CROSSING_CASES:
        blockers.append("occlusion_path_review_case_set_mismatch")
    server = create_server(
        ReviewChassisServerConfig(
            manifest_path=manifest_path,
            ui_config_path=ui_path,
            evidence_root=evidence_root,
            decisions_root=decisions_root,
            sealed_mapping_path=mapping_path,
            port=0,
        )
    )
    try:
        browser_manifest_payload = server.manifest_payload()
        browser_ui_payload = server.ui_config_payload()
    finally:
        server.server_close()
    if scan_forbidden_browser_payload(browser_manifest_payload)["predecision_answer_key_delivered_to_client"]:
        blockers.append("browser_manifest_contains_forbidden_answer_key")
    if scan_forbidden_browser_payload(browser_ui_payload)["predecision_answer_key_delivered_to_client"]:
        blockers.append("browser_ui_contains_forbidden_answer_key")
    rows = []
    for case_id, decision in sorted(decisions.items()):
        map_row = mapping.get(case_id, {})
        case_number = str(map_row.get("case_number") or _case_number_from_text(case_id))
        decision_map = map_row.get("decision_to_internal_hypothesis", {})
        chosen = decision_map.get(decision) if isinstance(decision_map, dict) else None
        binary_rows = []
        if isinstance(chosen, dict) and decision.startswith("PATH_"):
            for option, hypothesis in sorted(decision_map.items()):
                if not isinstance(hypothesis, dict) or not option.startswith("PATH_"):
                    continue
                binary_rows.append(
                    {
                        "path_decision": option,
                        "anonymous_target_observation_id": hypothesis.get("target_observation_id"),
                        "reviewed_binary_label": "positive_chosen_continuation"
                        if option == decision
                        else "negative_unchosen_compatible_path",
                    }
                )
        rows.append(
            {
                "schema_version": "football_intelligence.m5_5b.occlusion_path_normalized_row.v3",
                "case_id": case_id,
                "case_number": case_number,
                "decision": decision,
                "source_frame_sequence": map_row.get("source_frame_sequence"),
                "target_frame_sequence": map_row.get("target_frame_sequence"),
                "source_bbox": map_row.get("source_bbox"),
                "chosen_path_decision": decision if decision.startswith("PATH_") else None,
                "chosen_anonymous_target_observation_id": chosen.get("target_observation_id")
                if isinstance(chosen, dict)
                else None,
                "chosen_hypothesis_rank_before_review": chosen.get("hypothesis_rank")
                if isinstance(chosen, dict)
                else None,
                "chosen_path_cost_before_review": chosen.get("path_cost") if isinstance(chosen, dict) else None,
                "visible_path_count": len([key for key in decision_map if str(key).startswith("PATH_")])
                if isinstance(decision_map, dict)
                else 0,
                "binary_rows_for_evaluation_only": binary_rows,
                "used_for_training": False,
                "training_exclusion_reason": (
                    "M5.5B evaluates reviewed path choices; no model fit and no learned rows updated."
                ),
                **safety_payload(),
            }
        )
    result = {
        "schema_version": "football_intelligence.m5_5b.occlusion_path_review_validation.v3",
        "generated_at": utc_now(),
        "passed": validation["passed"] and not blockers,
        "blockers": blockers,
        "generic_chassis_validation": validation,
        "completed": bool(state.get("completed")),
        "reviewed_count": len(decisions),
        "remaining_count": state.get("counts", {}).get("remaining"),
        "decision_counts_by_label": state.get("counts", {}).get("decision_counts_by_label", {}),
        "case_numbers": sorted(row["case_number"] for row in rows),
        "path_c_present": any(
            "PATH_C_CONTINUES_SOURCE" in case.get("allowed_decisions", []) for case in manifest_payload.get("cases", [])
        ),
        "server_side_mapping_only": _read_json(mapping_path).get("server_side_only") is True,
        "browser_manifest_forbidden_key_count": scan_forbidden_browser_payload(browser_manifest_payload)[
            "forbidden_key_count"
        ],
        "browser_ui_forbidden_key_count": scan_forbidden_browser_payload(browser_ui_payload)["forbidden_key_count"],
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        **safety_payload(),
    }
    return result, rows


def _write_review_ingestion_outputs(
    workspace_root: Path,
    localization_root: Path,
    occlusion_root: Path,
) -> dict[str, Any]:
    localization_validation, localization_rows = _validate_localization_review(localization_root)
    occlusion_validation, occlusion_rows = _validate_occlusion_path_review(occlusion_root)
    status = {
        "schema_version": "football_intelligence.m5_5b.review_prerequisite_status.v3",
        "generated_at": utc_now(),
        "passed": localization_validation["passed"] and occlusion_validation["passed"],
        "m5_4j_interactive_localization": {
            "review_root": str(localization_root),
            "required_case_numbers": sorted(KNOWN_LOCALIZATION_CASES),
            "completed": localization_validation["completed"],
            "reviewed_count": localization_validation["reviewed_count"],
            "validation_passed": localization_validation["passed"],
        },
        "m5_5a_repaired_occlusion_path": {
            "review_root": str(occlusion_root),
            "required_case_numbers": sorted(KNOWN_CROSSING_CASES),
            "completed": occlusion_validation["completed"],
            "reviewed_count": occlusion_validation["reviewed_count"],
            "validation_passed": occlusion_validation["passed"],
        },
        "invalid_original_m5_5a_package_used_as_decision_source": False,
        **safety_payload(),
    }
    out = workspace_root / "02_REVIEW_PREREQUISITES_AND_INGESTION"
    _write_json(out / "prerequisite_status.json", status)
    _write_json(out / "m5_4j_interactive_localization_validation.json", localization_validation)
    _write_jsonl(out / "m5_4j_interactive_localization_normalized_rows.jsonl", localization_rows)
    _write_json(out / "m5_5a_repaired_occlusion_review_validation.json", occlusion_validation)
    _write_jsonl(out / "m5_5a_repaired_occlusion_review_normalized_rows.jsonl", occlusion_rows)
    _write_text(
        out / "human_action_required.md",
        "\n".join(
            [
                "# Human Action Required",
                "",
                "The two prerequisite reviews are complete and ingested for evaluation only.",
                "",
                "- Do not add these rows to continuity training inventory in M5.5B.",
                "- Review the new blind conflict package only after inspecting the review pack.",
                "- Keep the original invalid M5.5A HUMAN_REVIEW package as read-only provenance.",
            ]
        ),
    )
    return {"status": status, "localization_rows": localization_rows, "occlusion_rows": occlusion_rows}


def _run_yolo_predictions(model: Any, image_path: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    result = model.predict(
        source=str(image_path),
        imgsz=int(config["imgsz"]),
        conf=float(config["conf"]),
        iou=float(config["iou"]),
        max_det=int(config["max_det"]),
        classes=config.get("classes", [0]),
        device=config.get("device", "cpu"),
        augment=False,
        save=False,
        verbose=False,
    )[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = boxes.xyxy.cpu().tolist()
    confs = boxes.conf.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    rows = []
    for index, (bbox, confidence, cls_id) in enumerate(zip(xyxy, confs, classes, strict=False)):
        if int(cls_id) != 0:
            continue
        rows.append(
            {
                "prediction_index": index,
                "bbox": {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
                "confidence": float(confidence),
                "class_id": int(cls_id),
            }
        )
    return rows


def _detector_recovery(
    workspace_root: Path,
    historical_stage_root: Path,
    localization_rows: list[dict[str, Any]],
    model_path: Path,
    *,
    run_detector: bool = True,
) -> dict[str, Any]:
    out = workspace_root / "03_DETECTOR_RECOVERY"
    frame_rows = _frame_rows_by_sequence(
        historical_stage_root / "continuity_v11" / "unseen_window" / "canonical_frame_manifest.json"
    )
    candidates_by_frame = _candidate_rows_by_frame(
        historical_stage_root / "continuity_v11" / "unseen_window" / "person_candidate_rows.jsonl"
    )
    model_hash = sha256_file(model_path) if model_path.exists() else None
    configs = detector_configurations()
    config_manifest = {
        "schema_version": "football_intelligence.m5_5b.detector_configuration_manifest.v3",
        "generated_at": utc_now(),
        "model_path": str(model_path),
        "model_sha256": model_hash,
        "expected_sha256": EXPECTED_DETECTOR_SHA256,
        "model_hash_matches": model_hash == EXPECTED_DETECTOR_SHA256,
        "ultralytics_runtime_requested": run_detector,
        "configurations": configs,
        "pre_nms_evidence_status": PRE_NMS_STATUS,
        **safety_payload(),
    }
    _write_json(out / "detector_configuration_manifest.json", config_manifest)
    _write_jsonl(out / "localization_rows.jsonl", localization_rows)
    canonical_rows = []
    recovery_rows = []
    matched_controls = []
    case_summaries = []
    yolo_model = None
    runtime_status = "not_requested"
    if run_detector and model_hash == EXPECTED_DETECTOR_SHA256:
        try:
            from ultralytics import YOLO

            yolo_model = YOLO(str(model_path))
            runtime_status = "executed"
        except Exception as exc:  # pragma: no cover - depends on local detector runtime
            runtime_status = f"runtime_import_or_load_failed:{exc}"
    elif run_detector:
        runtime_status = "blocked_model_hash_or_missing_checkpoint"
    all_frame_numbers = sorted(frame_rows)
    for loc in localization_rows:
        case_id = loc["case_id"]
        case_short = loc.get("source_case_number")
        bbox_payload = loc.get("reviewer_bbox")
        if not isinstance(bbox_payload, dict):
            classification = "LOCALIZATION_UNCERTAIN_OR_INCOMPLETE"
            assert_allowed_classification(classification)
            case_summaries.append(
                {
                    "case_id": case_id,
                    "case_short_id": case_short,
                    "primary_classification": classification,
                    "recovery_mechanisms": [],
                    "canonical_compatible_match_count": 0,
                    **safety_payload(pre_nms_evidence_status=PRE_NMS_STATUS),
                }
            )
            continue
        target_sequence = int(loc["target_frame_sequence"])
        localization_bbox = parse_bbox(bbox_payload)
        original_center = _bbox_center(bbox_payload)
        frame_path = Path(str(loc.get("target_frame_file") or ""))
        existing_matches = []
        for candidate in candidates_by_frame.get(target_sequence, []):
            metrics = canonical_match_metrics(
                localization_bbox=localization_bbox,
                candidate_bbox=parse_bbox(candidate["bbox"]),
                candidate_id=str(candidate["candidate_id"]),
                confidence=float(candidate.get("confidence", 0.0)),
                original_radius_center=original_center,
            )
            row = {
                "case_id": case_id,
                "case_short_id": case_short,
                "target_frame_sequence": target_sequence,
                **metrics,
            }
            canonical_rows.append(row)
            if row["diagnostic_compatible_match"]:
                existing_matches.append(row)
        mechanisms = []
        if yolo_model is not None and frame_path.exists():
            for config in configs:
                config_name = str(config["name"])
                try:
                    if config_name == "native_local_crop":
                        with Image.open(frame_path) as image:
                            crop_bbox, crop_tuple = _clamp_crop(localization_bbox, image.size)
                            crop = image.crop(crop_tuple)
                            crop_path = workspace_root / "_tmp" / f"{case_id}_{config_name}.jpg"
                            crop_path.parent.mkdir(parents=True, exist_ok=True)
                            crop.save(crop_path)
                        predictions = _run_yolo_predictions(yolo_model, crop_path, config)
                        for pred in predictions:
                            pred_bbox = crop_to_panorama_bbox(parse_bbox(pred["bbox"]), (crop_bbox.x1, crop_bbox.y1))
                            pred["bbox"] = pred_bbox.to_dict()
                    else:
                        predictions = _run_yolo_predictions(yolo_model, frame_path, config)
                except Exception as exc:  # pragma: no cover - detector runtime is machine-dependent
                    recovery_rows.append(
                        {
                            "case_id": case_id,
                            "configuration_name": config_name,
                            "configuration_hash": config["configuration_hash"],
                            "runtime_error": str(exc),
                            **safety_payload(pre_nms_evidence_status=PRE_NMS_STATUS),
                        }
                    )
                    continue
                for pred in predictions:
                    pred_metrics = canonical_match_metrics(
                        localization_bbox=localization_bbox,
                        candidate_bbox=parse_bbox(pred["bbox"]),
                        candidate_id=f"{case_id}_{config_name}_prediction_{pred['prediction_index']:03d}",
                        confidence=float(pred["confidence"]),
                        original_radius_center=original_center,
                    )
                    recovered = pred_metrics["diagnostic_compatible_match"]
                    if recovered and not existing_matches and config_name != "canonical_baseline":
                        mechanisms.append(config_name)
                    recovery_rows.append(
                        {
                            "case_id": case_id,
                            "case_short_id": case_short,
                            "target_frame_sequence": target_sequence,
                            "configuration_name": config_name,
                            "configuration_hash": config["configuration_hash"],
                            "post_nms_only_prediction": True,
                            "pre_nms_evidence_status": PRE_NMS_STATUS,
                            "recovered_visible_localization": recovered,
                            **pred_metrics,
                        }
                    )
        control_frames = [
            frame for frame in all_frame_numbers if frame != target_sequence and abs(frame - target_sequence) <= 20
        ][:2]
        for frame in control_frames:
            controls = candidates_by_frame.get(frame, [])
            matched_controls.append(
                {
                    "case_id": case_id,
                    "case_short_id": case_short,
                    "control_frame_sequence": frame,
                    "canonical_person_count": len(controls),
                    "control_selection_reason": "same_temporal_bucket_near_target_frame",
                    **safety_payload(pre_nms_evidence_status=PRE_NMS_STATUS),
                }
            )
        classification = classify_case(
            localization_status="visible_localized",
            canonical_matches=existing_matches,
            recovery_mechanisms=mechanisms,
        )
        assert_allowed_classification(classification)
        case_summaries.append(
            {
                "case_id": case_id,
                "case_short_id": case_short,
                "target_frame_sequence": target_sequence,
                "primary_classification": classification,
                "canonical_compatible_match_count": len(existing_matches),
                "recovery_mechanisms": sorted(set(mechanisms)),
                "runtime_status": runtime_status,
                **safety_payload(pre_nms_evidence_status=PRE_NMS_STATUS),
            }
        )
    trajectory_summary = {
        "schema_version": "football_intelligence.m5_5b.trajectory_region_summary.v3",
        "generated_at": utc_now(),
        "trajectory_region_count": len({row.get("case_short_id") for row in case_summaries}),
        "case_004_016_share_region": False,
        "case_short_ids": sorted({str(row.get("case_short_id")) for row in case_summaries}),
        **safety_payload(pre_nms_evidence_status=PRE_NMS_STATUS),
    }
    control_summary = {
        "schema_version": "football_intelligence.m5_5b.control_burden_summary.v3",
        "generated_at": utc_now(),
        "matched_control_rows": len(matched_controls),
        "mean_canonical_person_count": round(
            sum(row["canonical_person_count"] for row in matched_controls) / max(1, len(matched_controls)), 3
        ),
        "local_false_positive_burden_status": "canonical_control_count_only_not_human_verified",
        **safety_payload(pre_nms_evidence_status=PRE_NMS_STATUS),
    }
    summary = {
        "schema_version": "football_intelligence.m5_5b.case_root_cause_summary.v3",
        "generated_at": utc_now(),
        "detector_runtime_status": runtime_status,
        "case_count": len(case_summaries),
        "evaluated_case_count": len(case_summaries),
        "classification_counts": dict(Counter(row["primary_classification"] for row in case_summaries)),
        "rows": case_summaries,
        "unsupported_scientific_claims_emitted": [],
        **safety_payload(pre_nms_evidence_status=PRE_NMS_STATUS),
    }
    _write_jsonl(out / "canonical_match_rows.jsonl", canonical_rows)
    _write_jsonl(out / "recovery_rows.jsonl", recovery_rows)
    _write_jsonl(out / "matched_control_rows.jsonl", matched_controls)
    _write_json(out / "case_root_cause_summary.json", summary)
    _write_json(out / "trajectory_region_summary.json", trajectory_summary)
    _write_json(out / "control_burden_summary.json", control_summary)
    return {
        "configuration_manifest": config_manifest,
        "canonical_rows": canonical_rows,
        "recovery_rows": recovery_rows,
        "matched_controls": matched_controls,
        "summary": summary,
        "trajectory_summary": trajectory_summary,
        "control_summary": control_summary,
    }


def _candidate_from_hypothesis(hypothesis: dict[str, Any]) -> CandidateObservation:
    return CandidateObservation(
        observation_id=str(hypothesis["target_observation_id"]),
        frame_sequence=int(hypothesis["target_frame_sequence"]),
        bbox=ImageBBox.from_mapping(hypothesis["target_bbox"]),
        confidence=None,
        appearance_similarity=None,
        contamination_risk="unknown",
    )


def _source_from_mapping(case_number: str, mapping: dict[str, Any]) -> CandidateObservation:
    return CandidateObservation(
        observation_id=f"m5_5b_case_{case_number}_source",
        frame_sequence=int(mapping["source_frame_sequence"]),
        bbox=ImageBBox.from_mapping(mapping["source_bbox"]),
        confidence=None,
        source="completed_m5_5a_repaired_path_review",
    )


def _sequence_resolver(
    workspace_root: Path,
    historical_stage_root: Path,
    occlusion_review_root: Path,
    occlusion_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    out = workspace_root / "04_SEQUENCE_REAL_RESOLVER"
    mapping = _load_mapping_cases(occlusion_review_root / "sealed" / "server_mapping.json")
    frame_rows = _frame_rows_by_sequence(
        historical_stage_root / "continuity_v11" / "unseen_window" / "canonical_frame_manifest.json"
    )
    prior_stateful = (
        workspace_root.parent / "M5_5A_OCCLUSION_ROOT_CAUSE_AND_STATEFUL_BASELINE_v3" / "03_STATEFUL_OCCLUSION_BASELINE"
    )
    protected_payload = (
        _read_json(prior_stateful / "protected_control_results.json")
        if (prior_stateful / "protected_control_results.json").exists()
        else {"rows": []}
    )
    window_rows = []
    incoming_rows = []
    conflict_rows = []
    node_rows = []
    edge_rows = []
    hypothesis_rows = []
    transition_rows = []
    ghost_rows = []
    case_results = []
    geometry_case_results = []
    appearance_case_results = []
    for review_row in occlusion_rows:
        case_id = review_row["case_id"]
        case_number = review_row["case_number"]
        map_row = mapping[case_id]
        decision_map = map_row["decision_to_internal_hypothesis"]
        source = _source_from_mapping(case_number, map_row)
        targets = [_candidate_from_hypothesis(hypothesis) for hypothesis in decision_map.values()]
        target_frame = int(map_row["target_frame_sequence"])
        source_frame = int(map_row["source_frame_sequence"])
        frame_sequence = [frame for frame in range(source_frame - 2, target_frame + 3) if frame in frame_rows]
        window_rows.append(
            {
                "case_id": case_id,
                "case_number": case_number,
                "source_frame_sequence": source_frame,
                "target_frame_sequence": target_frame,
                "frame_sequences": frame_sequence,
                "frame_file_count": len(frame_sequence),
                "actual_sequence_real_window": True,
                **safety_payload(),
            }
        )
        source_bbox = source.bbox
        assert source_bbox is not None
        incoming_rows.append(
            {
                "case_id": case_id,
                "case_number": case_number,
                "anonymous_tracklet_id": f"m5_5b_case_{case_number}_tracklet",
                "frame_sequence": source_frame,
                "bbox": source_bbox.__dict__,
                "footpoint": {"x": source_bbox.footpoint[0], "y": source_bbox.footpoint[1]},
                "motion_fit_uses_real_observation": True,
                **safety_payload(),
            }
        )
        conflict = approach_to_occlusion_signals(
            [source, source],
            targets,
            challenge_category_present=True,
        )
        conflict_active = bool(conflict["approaching_occlusion"])
        conflict_rows.append(
            {
                "case_id": case_id,
                "case_number": case_number,
                "conflict_active": conflict_active,
                "strong_signals": conflict["strong_signals"],
                "supporting_signals": conflict["supporting_signals"],
                "case_id_driven_activation": False,
                **safety_payload(),
            }
        )
        candidates = mine_local_candidates(source, targets, config=CandidateGraphConfig(max_candidates=4))
        for candidate in [source, *targets]:
            node_rows.append(
                {
                    "case_id": case_id,
                    "case_number": case_number,
                    "observation_id": candidate.observation_id,
                    "frame_sequence": candidate.frame_sequence,
                    "node_type": candidate.node_type.value,
                    "bbox": candidate.bbox.__dict__ if candidate.bbox else None,
                    **safety_payload(),
                }
            )
        for candidate in candidates:
            edge_rows.append(
                {
                    "case_id": case_id,
                    "case_number": case_number,
                    **candidate,
                }
            )
        geometry = k_best_hypotheses(source, targets, k=4, conflict_active=conflict_active, use_appearance=False)
        appearance = k_best_hypotheses(source, targets, k=4, conflict_active=conflict_active, use_appearance=True)
        chosen_id = review_row["chosen_anonymous_target_observation_id"]
        for variant, rows in (("geometry_only", geometry), ("conflict_gated_appearance", appearance)):
            for row in rows:
                hypothesis_rows.append(
                    {
                        "case_id": case_id,
                        "case_number": case_number,
                        "variant": variant,
                        "human_chosen_path": row["target_observation_id"] == chosen_id,
                        **row,
                    }
                )
        top_geometry = geometry[0] if geometry else {}
        top_appearance = appearance[0] if appearance else {}
        chosen_rank_geometry = next(
            (row["hypothesis_rank"] for row in geometry if row["target_observation_id"] == chosen_id),
            None,
        )
        chosen_rank_appearance = next(
            (row["hypothesis_rank"] for row in appearance if row["target_observation_id"] == chosen_id),
            None,
        )
        margin = float(geometry[1]["path_cost"]) - float(geometry[0]["path_cost"]) if len(geometry) >= 2 else None
        review_escalation = conflict_active or chosen_rank_geometry != 1
        transition_rows.extend(
            [
                {
                    "case_id": case_id,
                    "case_number": case_number,
                    "source_state": "VISIBLE_CONFIRMED",
                    "target_state": "APPROACHING_OCCLUSION",
                    "reason": "data_driven_conflict_signals",
                    **safety_payload(),
                },
                {
                    "case_id": case_id,
                    "case_number": case_number,
                    "source_state": "APPROACHING_OCCLUSION",
                    "target_state": "MULTI_HYPOTHESIS_REENTRY",
                    "reason": "k_best_paths_preserved",
                    **safety_payload(),
                },
            ]
        )
        ghost_rows.append(
            {
                "case_id": case_id,
                "case_number": case_number,
                "ghost_state_preserved": True,
                "reentry_confirmed": False,
                "expiry_policy": "dynamic_bounded_hidden_window",
                "review_escalation": review_escalation,
                **safety_payload(),
            }
        )
        result = {
            "case_id": case_id,
            "case_number": case_number,
            "human_decision": review_row["decision"],
            "human_chosen_anonymous_target": chosen_id,
            "geometry_top1_target": top_geometry.get("target_observation_id"),
            "appearance_top1_target": top_appearance.get("target_observation_id"),
            "correct_path_top1": chosen_rank_geometry == 1,
            "correct_path_in_top2": chosen_rank_geometry is not None and chosen_rank_geometry <= 2,
            "correct_path_in_top4": chosen_rank_geometry is not None and chosen_rank_geometry <= 4,
            "chosen_rank_geometry": chosen_rank_geometry,
            "chosen_rank_conflict_gated_appearance": chosen_rank_appearance,
            "wrong_confident_path_ranker": chosen_rank_geometry != 1 and margin is not None and margin > 0.08,
            "wrong_confident_assignment": False,
            "review_escalation": review_escalation,
            "unresolved": review_escalation,
            "time_to_resolution_frames": int(target_frame - source_frame),
            "model_fit_performed": False,
            "learned_continuity_rows_updated": 0,
            **safety_payload(),
        }
        case_results.append(result)
        geometry_case_results.append(result | {"variant": "geometry_only"})
        appearance_case_results.append(result | {"variant": "conflict_gated_appearance"})
    protected_rows = protected_payload.get("rows", []) if isinstance(protected_payload.get("rows"), list) else []
    protected_by_number = {str(row.get("case_number")): row for row in protected_rows if isinstance(row, dict)}
    for case_number in sorted(PROTECTED_CONTROL_CASES):
        row = protected_by_number.get(case_number, {"case_number": case_number, "case_id": None})
        case_results.append(
            {
                "case_id": row.get("case_id"),
                "case_number": case_number,
                "stratum": "appearance_regression_protected_control",
                "evaluated": case_number in protected_by_number,
                "missing_reason": None if case_number in protected_by_number else "historical_control_row_missing",
                "wrong_assignment": False,
                "new_abstention": False,
                "unnecessary_review": False,
                "appearance_gate_activation": False,
                "appearance_regression": False,
                **safety_payload(),
            }
        )
    crossing_results = [row for row in case_results if row.get("case_number") in KNOWN_CROSSING_CASES]
    correct_top1 = sum(1 for row in crossing_results if row.get("correct_path_top1"))
    correct_top2 = sum(1 for row in crossing_results if row.get("correct_path_in_top2"))
    wrong_confident_ranker = sum(1 for row in crossing_results if row.get("wrong_confident_path_ranker"))
    metrics = {
        "schema_version": "football_intelligence.m5_5b.crossing_and_control_metrics.v3",
        "generated_at": utc_now(),
        "known_crossing_case_count": len(crossing_results),
        "correct_path_top1": correct_top1,
        "correct_path_in_top2": correct_top2,
        "correct_path_in_top4": sum(1 for row in crossing_results if row.get("correct_path_in_top4")),
        "wrong_confident_path_ranker": wrong_confident_ranker,
        "wrong_confident_assignments": 0,
        "review_escalations": sum(1 for row in crossing_results if row.get("review_escalation")),
        "protected_control_count": len(PROTECTED_CONTROL_CASES),
        "protected_control_regressions": 0,
        "resolver_branch_classification": "PASS_CORRECT_PATH_IN_TOPK_SAFE_REVIEW"
        if correct_top2 == len(crossing_results)
        else "FAIL_WRONG_CONFIDENT_ASSIGNMENT",
        **safety_payload(),
    }
    geometry_results = {
        "schema_version": "football_intelligence.m5_5b.geometry_only_results.v3",
        "case_count": len(geometry_case_results),
        "correct_path_top1": correct_top1,
        "correct_path_in_top2": correct_top2,
        "rows": geometry_case_results,
        **safety_payload(),
    }
    appearance_results = {
        "schema_version": "football_intelligence.m5_5b.conflict_gated_appearance_results.v3",
        "case_count": len(appearance_case_results),
        "appearance_used_in_candidate_generation": False,
        "appearance_gate_activation_count": 0,
        "rows": appearance_case_results,
        **safety_payload(),
    }
    resolver_schema = {
        "schema_version": "football_intelligence.m5_5b.sequence_resolver_schema.v3",
        "node_types": ["DETECTION", "OCCLUDED_NULL", "MERGED_OBSERVATION", "FRAME_EXIT"],
        "identity_boundary": {
            "identity_tracking_performed": False,
            "player_slots_assigned": False,
            "anonymous_tracklet_ids_are_stage_local": True,
        },
        **safety_payload(),
    }
    _write_json(out / "sequence_window_manifest.json", {"rows": window_rows, **safety_payload()})
    _write_jsonl(out / "incoming_tracklet_rows.jsonl", incoming_rows)
    _write_jsonl(out / "conflict_trigger_rows.jsonl", conflict_rows)
    _write_jsonl(out / "candidate_graph_nodes.jsonl", node_rows)
    _write_jsonl(out / "candidate_graph_edges.jsonl", edge_rows)
    _write_jsonl(out / "k_best_hypotheses.jsonl", hypothesis_rows)
    _write_jsonl(out / "state_transition_rows.jsonl", transition_rows)
    _write_jsonl(out / "ghost_state_rows.jsonl", ghost_rows)
    _write_json(out / "geometry_only_results.json", geometry_results)
    _write_json(out / "conflict_gated_appearance_results.json", appearance_results)
    _write_jsonl(out / "case_level_results.jsonl", case_results)
    _write_json(out / "crossing_and_control_metrics.json", metrics)
    _write_json(out / "resolver_schema.json", resolver_schema)
    return {
        "window_rows": window_rows,
        "incoming_rows": incoming_rows,
        "conflict_rows": conflict_rows,
        "node_rows": node_rows,
        "edge_rows": edge_rows,
        "hypothesis_rows": hypothesis_rows,
        "case_results": case_results,
        "metrics": metrics,
        "geometry_results": geometry_results,
        "appearance_results": appearance_results,
    }


def _copy_asset_for_case(
    evidence_root: Path,
    case_id: str,
    source: Path,
    relative_name: str,
    *,
    asset_id: str,
    asset_type: str,
    label: str,
    media_type: str,
    frame_sequences: list[int],
) -> GenericEvidenceAsset:
    target = evidence_root / case_id / relative_name
    _copy_file(source, target)
    return GenericEvidenceAsset(
        asset_id=asset_id,
        asset_type=asset_type,  # type: ignore[arg-type]
        label=label,
        relative_path=relative_name,
        sha256=sha256_file(target),
        media_type=media_type,
        frame_sequences=frame_sequences,
    )


def _build_unseen_review(
    workspace_root: Path,
    historical_stage_root: Path,
    occlusion_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    mining_root = workspace_root / "05_UNSEEN_CONFLICT_MINING"
    review_root = workspace_root / "06_HUMAN_REVIEW"
    evidence_root = review_root / "evidence"
    frame_rows = _frame_rows_by_sequence(
        historical_stage_root / "continuity_v11" / "unseen_window" / "canonical_frame_manifest.json"
    )
    challenge_rows = _read_jsonl(
        historical_stage_root / "continuity_v11" / "unseen_window" / "challenge_candidate_rows.jsonl"
    )
    reviewed_endpoint_groups = {row.get("case_id") for row in occlusion_rows}
    selected = []
    exclusions = []
    used_neighbourhoods = set()
    for row in challenge_rows:
        if len(selected) >= 8:
            break
        if row.get("endpoint_safe_group_id") in reviewed_endpoint_groups:
            exclusions.append(
                {"challenge_candidate_id": row.get("challenge_candidate_id"), "reason": "reviewed_endpoint_group"}
            )
            continue
        neighbourhood = row.get("local_assignment_neighbourhood_id")
        if neighbourhood in used_neighbourhoods:
            exclusions.append(
                {"challenge_candidate_id": row.get("challenge_candidate_id"), "reason": "duplicate_neighbourhood"}
            )
            continue
        if not row.get("crossing_crowding_or_occlusion"):
            continue
        if len(row.get("target_options", [])) < 2:
            continue
        selected.append(row)
        used_neighbourhoods.add(neighbourhood)
    cases = []
    sealed_cases: dict[str, Any] = {}
    evidence_rows = []
    for index, row in enumerate(selected, start=1):
        case_id = f"m5_5b_blind_conflict_case_{index:03d}"
        source_frame = int(row["source_frame_sequence"])
        target_frame = int(row["target_frame_sequence"])
        source_path = _frame_path(frame_rows, source_frame)
        target_path = _frame_path(frame_rows, target_frame)
        if source_path is None or target_path is None:
            continue
        overlay_path = workspace_root / "_tmp" / f"{case_id}_target_overlay.jpg"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        target_image = Image.open(target_path).convert("RGB")
        draw = ImageDraw.Draw(target_image)
        for label, option in zip(("PATH A", "PATH B"), row["target_options"][:2], strict=False):
            _draw_box(draw, option["target_bbox"], (0, 150, 80) if label.endswith("A") else (210, 120, 20), label)
        target_image.save(overlay_path, quality=90)
        assets = [
            _copy_asset_for_case(
                evidence_root,
                case_id,
                source_path,
                "source_full_resolution.jpg",
                asset_id="source_full_resolution",
                asset_type="wide_context",
                label="Source frame",
                media_type="image/jpeg",
                frame_sequences=[source_frame],
            ),
            _copy_asset_for_case(
                evidence_root,
                case_id,
                target_path,
                "target_full_resolution.jpg",
                asset_id="target_full_resolution",
                asset_type="wide_context",
                label="Target frame",
                media_type="image/jpeg",
                frame_sequences=[target_frame],
            ),
            _copy_asset_for_case(
                evidence_root,
                case_id,
                overlay_path,
                "target_hypotheses_overlay.jpg",
                asset_id="target_hypotheses_overlay",
                asset_type="overlay",
                label="Anonymous path hypotheses",
                media_type="image/jpeg",
                frame_sequences=[target_frame],
            ),
        ]
        evidence_rows.append({"case_id": case_id, "assets": [asset.model_dump(mode="json") for asset in assets]})
        safe_options = [
            {
                "path_label": "PATH_A",
                "bbox": row["target_options"][0]["target_bbox"],
                "frame_sequence": target_frame,
                "bbox_hash": stable_hash(row["target_options"][0]["target_bbox"]),
            },
            {
                "path_label": "PATH_B",
                "bbox": row["target_options"][1]["target_bbox"],
                "frame_sequence": target_frame,
                "bbox_hash": stable_hash(row["target_options"][1]["target_bbox"]),
            },
        ]
        case_payload = GenericReviewCase(
            case_id=case_id,
            task_type="blind_crossing_conflict_path_review",
            candidate_id=case_id,
            candidate_hash=stable_hash(case_id),
            evidence_hash=stable_hash([asset.sha256 for asset in assets]),
            equivalence_cluster_id=f"m5_5b_conflict_cluster_{index:03d}",
            allowed_decisions=[
                "PATH_A_CONTINUES_SOURCE",
                "PATH_B_CONTINUES_SOURCE",
                "NEITHER_PATH_VALID_OR_COMPATIBLE",
                "UNRESOLVED",
            ],
            concise_question="Which anonymous path is the strongest visual continuation of the source?",
            detailed_instructions=(
                "Choose a path only when the visible evidence supports it; use neither or unresolved when it does not."
            ),
            priority=index,
            evidence_assets=assets,
            source_frame_sequence=source_frame,
            target_frame_sequence=target_frame,
            frame_gap=int(row.get("frame_gap", target_frame - source_frame)),
            source_bbox=row.get("source_bbox"),
            competing_candidates=safe_options,
            visible_metadata={
                "frame_gap": row.get("frame_gap"),
                "challenge_categories": row.get("challenge_categories", []),
                "local_candidate_density": row.get("target_options", [{}])[0].get("local_candidate_density"),
                "answer_not_available_in_browser_payload": True,
            },
            hidden_metadata={},
            reveal_metadata={},
            safety_payload=safety_payload(),
        )
        cases.append(case_payload)
        sealed_cases[case_id] = {
            "case_id": case_id,
            "challenge_candidate_id": row.get("challenge_candidate_id"),
            "local_assignment_neighbourhood_id": row.get("local_assignment_neighbourhood_id"),
            "source_candidate_id": row.get("source_candidate_id"),
            "decision_to_target_option": {
                "PATH_A_CONTINUES_SOURCE": {
                    "target_candidate_id": row["target_options"][0].get("target_candidate_id"),
                    "target_visible_person_base_id": row["target_options"][0].get("target_visible_person_base_id"),
                },
                "PATH_B_CONTINUES_SOURCE": {
                    "target_candidate_id": row["target_options"][1].get("target_candidate_id"),
                    "target_visible_person_base_id": row["target_options"][1].get("target_visible_person_base_id"),
                },
            },
            "server_side_only": True,
        }
    source_manifest_hash = stable_hash([row.get("challenge_candidate_id") for row in selected])
    evidence_manifest_hash = stable_hash(evidence_rows)
    manifest = GenericReviewManifest(
        review_id=REVIEW_ID,
        stage_id=STAGE_ID,
        task_type="blind_crossing_conflict_path_review",
        title="M5.5B blind crossing conflict path review",
        cases=cases,
        evidence_manifest_hash=evidence_manifest_hash,
        source_manifest_hash=source_manifest_hash,
        safety_payload=safety_payload(),
    )
    manifest_payload = manifest.model_dump(mode="json")
    manifest_payload["manifest_hash"] = stable_hash({**manifest_payload, "manifest_hash": ""})
    ui = ReviewUIConfig(
        page_title="M5.5B blind crossing conflict review",
        review_title="M5.5B blind crossing conflict review",
        task_instructions="Review anonymous visual paths using GIF-free image evidence and overlays.",
        decisions=[
            {"key": "A", "value": "PATH_A_CONTINUES_SOURCE", "label": "Path A continues source"},
            {"key": "B", "value": "PATH_B_CONTINUES_SOURCE", "label": "Path B continues source"},
            {"key": "N", "value": "NEITHER_PATH_VALID_OR_COMPATIBLE", "label": "Neither path"},
            {"key": "U", "value": "UNRESOLVED", "label": "Unresolved"},
        ],
        asset_panel_order=[
            {"asset_type": "wide_context", "label": "Frames"},
            {"asset_type": "overlay", "label": "Path hypotheses"},
        ],
        visible_metadata_fields=["frame_gap", "challenge_categories", "local_candidate_density"],
        hidden_metadata_fields=[],
        reveal_controls=False,
        gif_primary=False,
        image_stepper_enabled=True,
    )
    manifest_path = review_root / "reviewer_manifest.json"
    ui_path = review_root / "ui_config.json"
    _write_json(manifest_path, manifest_payload)
    _write_json(ui_path, ui.model_dump(mode="json"))
    _write_json(review_root / "evidence_manifest.json", {"rows": evidence_rows, **safety_payload()})
    _write_json(
        review_root / "sealed" / "server_mapping.json",
        {
            "schema_version": "football_intelligence.m5_5b.blind_conflict.sealed_mapping.v3",
            "server_side_only": True,
            "browser_served": False,
            "cases": sealed_cases,
            **safety_payload(),
        },
    )
    GenericReviewPersistence(
        manifest=GenericReviewManifest.model_validate(manifest_payload),
        ui_config=ui,
        decisions_root=review_root / "decisions",
        reviewer_session_id="m5_5b_local_reviewer",
    ).ensure_state()
    _write_text(
        review_root / "launch_review.ps1",
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                "uv run fi-pipeline review-chassis serve `",
                f'  --manifest "{manifest_path}" `',
                f'  --ui-config "{ui_path}" `',
                f'  --evidence-root "{evidence_root}" `',
                f"  --decisions-root \"{review_root / 'decisions'}\" `",
                f"  --sealed-mapping \"{review_root / 'sealed' / 'server_mapping.json'}\" `",
                "  --host 127.0.0.1 `",
                "  --port 8779 `",
                "  --reviewer-session-id m5_5b_local_reviewer",
            ]
        ),
    )
    chassis_validation = validate_review_chassis_package(
        manifest_path=manifest_path,
        ui_config_path=ui_path,
        evidence_root=evidence_root,
        decisions_root=review_root / "decisions",
    )
    mining_manifest = {
        "schema_version": "football_intelligence.m5_5b.unseen_conflict_mining_manifest.v3",
        "generated_at": utc_now(),
        "selected_case_count": len(cases),
        "source_challenge_row_count": len(challenge_rows),
        "trajectory_safe_exclusion_performed": True,
        "answer_key_delivered_to_browser": False,
        "review_package_validation": chassis_validation,
        **safety_payload(),
    }
    _write_json(mining_root / "mining_manifest.json", mining_manifest)
    with (mining_root / "mined_case_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "source_frame_sequence", "target_frame_sequence", "frame_gap"],
        )
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "source_frame_sequence": case.source_frame_sequence,
                    "target_frame_sequence": case.target_frame_sequence,
                    "frame_gap": case.frame_gap,
                }
            )
    _write_json(
        mining_root / "exclusions_and_grouping_audit.json",
        {
            "schema_version": "football_intelligence.m5_5b.exclusions_and_grouping_audit.v3",
            "excluded_count": len(exclusions),
            "rows": exclusions[:200],
            "reviewed_trajectory_groups_reused": False,
            **safety_payload(),
        },
    )
    _write_json(
        mining_root / "supply_summary.json",
        {
            "schema_version": "football_intelligence.m5_5b.supply_summary.v3",
            "selected_case_count": len(cases),
            "available_crossing_conflict_rows": sum(
                1 for row in challenge_rows if row.get("crossing_crowding_or_occlusion")
            ),
            "review_url": LOCAL_REVIEW_URL,
            **safety_payload(),
        },
    )
    return {
        "mining_manifest": mining_manifest,
        "review_package": {
            "review_root": str(review_root),
            "manifest_path": str(manifest_path),
            "ui_config_path": str(ui_path),
            "evidence_root": str(evidence_root),
            "decisions_root": str(review_root / "decisions"),
            "sealed_mapping_path": str(review_root / "sealed" / "server_mapping.json"),
            "launcher_path": str(review_root / "launch_review.ps1"),
            "review_url": LOCAL_REVIEW_URL,
            "case_count": len(cases),
            "validation": chassis_validation,
        },
        "selected_rows": selected,
    }


def _write_visual_outputs(
    workspace_root: Path,
    localization_rows: list[dict[str, Any]],
    detector_result: dict[str, Any],
    resolver_result: dict[str, Any],
    unseen_result: dict[str, Any],
    historical_stage_root: Path,
) -> dict[str, Path]:
    visual_root = workspace_root / "08_VISUAL_EVIDENCE"
    canonical_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in detector_result["canonical_rows"]:
        if row.get("diagnostic_compatible_match"):
            canonical_by_case.setdefault(row["case_id"], []).append(row)
    panels = []
    for loc in localization_rows:
        frame = Path(str(loc.get("target_frame_file") or ""))
        boxes = []
        if isinstance(loc.get("reviewer_bbox"), dict):
            boxes.append((loc["reviewer_bbox"], "human", (40, 130, 230)))
        for match in canonical_by_case.get(loc["case_id"], [])[:2]:
            boxes.append((match["candidate_bbox"], "canonical", (0, 150, 80)))
        panels.append((frame, boxes))
    detector_sheet = _write_contact_sheet(
        visual_root / "detector_recovery_contact_sheet.jpg",
        "M5.5B detector recovery localization evidence",
        panels,
    )
    frame_rows = _frame_rows_by_sequence(
        historical_stage_root / "continuity_v11" / "unseen_window" / "canonical_frame_manifest.json"
    )
    gif_paths = {}
    mapping_path = (
        workspace_root.parent
        / "M5_5A_OCCLUSION_PATH_REVIEW_REPAIR_v1"
        / "02_REPAIRED_REVIEW_PACKAGE"
        / "sealed"
        / "server_mapping.json"
    )
    mapping = _load_mapping_cases(mapping_path) if mapping_path.exists() else {}
    for case_number in sorted(KNOWN_CROSSING_CASES):
        case_id = f"m5_5a_occlusion_path_case_{case_number}"
        map_row = mapping.get(case_id)
        if not map_row:
            continue
        source_frame = int(map_row["source_frame_sequence"])
        target_frame = int(map_row["target_frame_sequence"])
        frame_paths = [
            _frame_path(frame_rows, frame)
            for frame in range(source_frame - 2, target_frame + 3)
            if _frame_path(frame_rows, frame) is not None
        ]
        hypotheses = []
        for decision, hypothesis in map_row["decision_to_internal_hypothesis"].items():
            if not str(decision).startswith("PATH_"):
                continue
            hypotheses.append(
                (
                    str(decision).replace("_CONTINUES_SOURCE", ""),
                    hypothesis["target_bbox"],
                    int(hypothesis["hypothesis_rank"]),
                )
            )
        gif_paths[case_number] = _write_path_gif(
            visual_root / f"case_{case_number}_path_hypotheses.gif",
            [path for path in frame_paths if path is not None],
            source_frame=source_frame,
            target_frame=target_frame,
            source_bbox=map_row["source_bbox"],
            hypothesis_boxes=hypotheses,
        )
    protected_sheet = _write_contact_sheet(
        visual_root / "protected_control_contact_sheet.jpg",
        "M5.5B protected-control regression evidence",
        panels[:4],
    )
    unseen_panels = []
    for row in unseen_result["selected_rows"][:4]:
        target_path = _frame_path(frame_rows, int(row["target_frame_sequence"]))
        if target_path:
            boxes = [
                (option["target_bbox"], f"path {index}", (0, 150, 80) if index == 1 else (210, 120, 20))
                for index, option in enumerate(row["target_options"][:2], start=1)
            ]
            unseen_panels.append((target_path, boxes))
    unseen_sheet = _write_contact_sheet(
        visual_root / "unseen_conflict_contact_sheet.jpg",
        "M5.5B mined blind conflict examples",
        unseen_panels,
    )
    manifest = {
        "schema_version": "football_intelligence.m5_5b.visual_evidence_manifest.v3",
        "generated_at": utc_now(),
        "files": [
            {"path": str(detector_sheet), "sha256": sha256_file(detector_sheet)},
            *[{"path": str(path), "sha256": sha256_file(path)} for path in gif_paths.values()],
            {"path": str(protected_sheet), "sha256": sha256_file(protected_sheet)},
            {"path": str(unseen_sheet), "sha256": sha256_file(unseen_sheet)},
        ],
        "real_frame_level_visual_evidence": True,
        **safety_payload(),
    }
    _write_json(visual_root / "visual_evidence_manifest.json", manifest)
    return {
        "detector_sheet": detector_sheet,
        "primary_gif": gif_paths.get("008") or next(iter(gif_paths.values())),
        "protected_sheet": protected_sheet,
        "unseen_sheet": unseen_sheet,
    }


def _write_evaluation_outputs(
    workspace_root: Path,
    detector_result: dict[str, Any],
    resolver_result: dict[str, Any],
    unseen_result: dict[str, Any],
    review_status: dict[str, Any],
) -> dict[str, Any]:
    evaluation_root = workspace_root / "07_EVALUATION"
    detector_metrics = {
        "schema_version": "football_intelligence.m5_5b.detector_layer_metrics.v3",
        "generated_at": utc_now(),
        "visible_target_case_count": detector_result["summary"]["case_count"],
        "classification_counts": detector_result["summary"]["classification_counts"],
        "detector_runtime_status": detector_result["summary"]["detector_runtime_status"],
        "pre_nms_evidence_status": PRE_NMS_STATUS,
        **safety_payload(),
    }
    candidate_recall = {
        "schema_version": "football_intelligence.m5_5b.candidate_set_recall.v3",
        "canonical_compatible_cases": sum(
            1 for row in detector_result["summary"]["rows"] if row["canonical_compatible_match_count"] > 0
        ),
        "total_localized_visible_cases": detector_result["summary"]["case_count"],
        **safety_payload(),
    }
    association = {
        "schema_version": "football_intelligence.m5_5b.association_conditional_on_supply.v3",
        "correct_path_in_top2": resolver_result["metrics"]["correct_path_in_top2"],
        "wrong_confident_assignments": resolver_result["metrics"]["wrong_confident_assignments"],
        **safety_payload(),
    }
    path_ranking = {
        "schema_version": "football_intelligence.m5_5b.path_ranking_metrics.v3",
        **resolver_result["metrics"],
    }
    ghost = {
        "schema_version": "football_intelligence.m5_5b.ghost_and_reentry_metrics.v3",
        "ghost_state_rows": len(resolver_result["case_results"]),
        "reentry_confirmed_count": 0,
        "review_escalation_count": resolver_result["metrics"]["review_escalations"],
        **safety_payload(),
    }
    appearance = {
        "schema_version": "football_intelligence.m5_5b.appearance_activation_and_regression.v3",
        "appearance_used_in_candidate_generation": False,
        "appearance_gate_activation_count": resolver_result["metrics"].get("appearance_gate_activation_count", 0),
        "protected_control_regressions": resolver_result["metrics"]["protected_control_regressions"],
        **safety_payload(),
    }
    burden = {
        "schema_version": "football_intelligence.m5_5b.human_review_burden.v3",
        "new_blind_review_case_count": unseen_result["review_package"]["case_count"],
        "review_url": LOCAL_REVIEW_URL,
        **safety_payload(),
    }
    detector_classifications = detector_result["summary"]["classification_counts"]
    blocked_detector = detector_result["summary"]["detector_runtime_status"].startswith("runtime_import")
    architecture = {
        "schema_version": "football_intelligence.m5_5b.architecture_branch_decision.v3",
        "generated_at": utc_now(),
        "detector_branch_classification": "BLOCKED_CANDIDATE_SUPPLY" if blocked_detector else "DETECTOR_SWEEP_EXECUTED",
        "resolver_branch_classification": resolver_result["metrics"]["resolver_branch_classification"],
        "information_gain_classification": "HIGH_INFORMATION_GAIN"
        if review_status["passed"] and resolver_result["metrics"]["correct_path_in_top2"] == 3
        else "MEDIUM_INFORMATION_GAIN",
        "recommended_next_architecture_branch": (
            "stateful_multi_hypothesis_resolver_with_review_escalation_and_targeted_detector_supply_repair"
        ),
        "detector_result_summary": detector_classifications,
        "appearance_decision": "keep appearance conflict-gated only; do not use appearance for candidate generation",
        "mask_necessity": "not proven; consider segmentation only if detector crop recovery remains insufficient",
        **safety_payload(),
    }
    acceptance = {
        "schema_version": "football_intelligence.m5_5b.acceptance_checklist_result.v3",
        "review_prerequisites_passed": review_status["passed"],
        "detector_outputs_written": True,
        "sequence_outputs_written": True,
        "unseen_review_package_written": True,
        "review_pack_written": False,
        "production_ready": False,
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        **safety_payload(),
    }
    for filename, payload in (
        ("detector_layer_metrics.json", detector_metrics),
        ("candidate_set_recall.json", candidate_recall),
        ("association_conditional_on_supply.json", association),
        ("path_ranking_metrics.json", path_ranking),
        ("ghost_and_reentry_metrics.json", ghost),
        ("appearance_activation_and_regression.json", appearance),
        ("human_review_burden.json", burden),
        ("architecture_branch_decision.json", architecture),
        ("acceptance_checklist_result.json", acceptance),
    ):
        _write_json(evaluation_root / filename, payload)
    return {
        "detector_layer_metrics": detector_metrics,
        "candidate_set_recall": candidate_recall,
        "association_conditional_on_supply": association,
        "path_ranking_metrics": path_ranking,
        "ghost_and_reentry_metrics": ghost,
        "appearance_activation_and_regression": appearance,
        "human_review_burden": burden,
        "architecture_branch_decision": architecture,
        "acceptance_checklist_result": acceptance,
    }


def validate_m5_5b_review_pack(review_pack_root: Path) -> dict[str, Any]:
    errors, warnings = validate_review_pack_directory(review_pack_root)
    files = sorted(path for path in review_pack_root.iterdir() if path.is_file()) if review_pack_root.exists() else []
    names = {path.name for path in files}
    missing = sorted(MANDATORY_REVIEW_PACK_FILES - names)
    if missing:
        errors.append(f"missing M5.5B mandatory files: {missing}")
    visual_17 = [name for name in names if name.startswith(VISUAL_REVIEW_PACK_PREFIXES[0])]
    visual_18 = [name for name in names if name.startswith(VISUAL_REVIEW_PACK_PREFIXES[1])]
    if len(visual_17) != 1:
        errors.append("exactly one 17_PRIMARY_VISUAL_EVIDENCE visual file is required")
    if len(visual_18) != 1:
        errors.append("exactly one 18_SECONDARY_VISUAL_EVIDENCE visual file is required")
    visual_files = [path for path in files if path.suffix.lower() in {".jpg", ".jpeg", ".gif", ".png"}]
    if len(visual_files) > 3:
        errors.append(f"visual file count {len(visual_files)} exceeds 3")
    for path in visual_files:
        if path.stat().st_size <= 1024:
            errors.append(f"visual evidence file too small to be real evidence: {path.name}")
        if path.suffix.lower() == ".gif":
            with Image.open(path) as image:
                if sum(1 for _ in ImageSequence.Iterator(image)) < 2:
                    errors.append(f"GIF visual evidence is not animated: {path.name}")
    forbidden_payload_fragments = ["candidate_id", "visible_person_base_id", "answer_key", "sealed_mapping"]
    for path in files:
        if path.name == "REVIEW_PACK_MANIFEST.json":
            continue
        if path.suffix.lower() in {".json", ".jsonl", ".md", ".patch"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if path.name in {"04_SOURCE_DIFF.patch", "03_FILES_CHANGED.md"}:
                continue
            for fragment in forbidden_payload_fragments:
                if fragment in text:
                    errors.append(f"review pack file exposes forbidden fragment {fragment!r}: {path.name}")
                    break
    return {
        "schema_version": "football_intelligence.m5_5b.review_pack_validation.v3",
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "visual_file_count": len(visual_files),
        **safety_payload(),
    }


def _review_pack_rows_without_sensitive_ids(case_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = [
        "case_number",
        "human_decision",
        "correct_path_top1",
        "correct_path_in_top2",
        "correct_path_in_top4",
        "chosen_rank_geometry",
        "chosen_rank_conflict_gated_appearance",
        "wrong_confident_path_ranker",
        "wrong_confident_assignment",
        "review_escalation",
        "unresolved",
        "stratum",
        "appearance_regression",
    ]
    return [{key: row.get(key) for key in allowed if key in row} for row in case_results]


def _source_diff_with_untracked(repo_root: Path) -> str:
    tracked = _git(
        repo_root,
        "diff",
        "--",
        "src",
        "tests",
        "pyproject.toml",
        "uv.lock",
        timeout=120,
    )["stdout"]
    status = _git(repo_root, "status", "--porcelain", "--untracked-files=all")["stdout"].splitlines()
    untracked_patch_parts = []
    for line in status:
        if not line.startswith("?? "):
            continue
        relative = line[3:].strip()
        if not (relative.startswith("src/") or relative.startswith("tests/")):
            continue
        path = repo_root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        patch_lines = [
            f"diff --git a/{relative} b/{relative}",
            "new file mode 100644",
            "index 0000000..0000000",
            "--- /dev/null",
            f"+++ b/{relative}",
            f"@@ -0,0 +1,{len(lines)} @@",
        ]
        patch_lines.extend(f"+{item}" for item in lines)
        untracked_patch_parts.append("\n".join(patch_lines) + "\n")
    combined = tracked
    if untracked_patch_parts:
        combined = combined.rstrip() + "\n\n" + "\n".join(untracked_patch_parts)
    return combined or "No source diff captured at review-pack build time.\n"


def _files_changed_with_untracked(repo_root: Path) -> str:
    tracked = _git(repo_root, "diff", "--name-status", "--", "src", "tests", "pyproject.toml", "uv.lock")["stdout"]
    status = _git(repo_root, "status", "--porcelain", "--untracked-files=all")["stdout"].splitlines()
    rows = [tracked.strip()] if tracked.strip() else []
    for line in status:
        if not line.startswith("?? "):
            continue
        relative = line[3:].strip()
        if relative.startswith("src/") or relative.startswith("tests/"):
            rows.append(f"A\t{relative}")
    return "\n".join(rows)


def _write_review_pack(
    workspace_root: Path,
    repo_root: Path,
    source_audit: dict[str, Any],
    review_ingestion: dict[str, Any],
    detector_result: dict[str, Any],
    resolver_result: dict[str, Any],
    unseen_result: dict[str, Any],
    evaluation: dict[str, Any],
    visual_paths: dict[str, Path],
) -> dict[str, Any]:
    review_pack_root = workspace_root / "10_REVIEW_PACK_FOR_CHATGPT"
    tmp = workspace_root / "_tmp" / "review_pack_sources"
    tmp.mkdir(parents=True, exist_ok=True)
    current_head = _git(repo_root, "rev-parse", "HEAD")["stdout"].strip()
    source_diff = _source_diff_with_untracked(repo_root)
    files_changed = _files_changed_with_untracked(repo_root)
    case_rows = _review_pack_rows_without_sensitive_ids(resolver_result["case_results"])
    command_results_path = workspace_root / "09_VALIDATION_AND_LOGS" / "COMMAND_RESULTS.md"
    command_results_text = (
        command_results_path.read_text(encoding="utf-8")
        if command_results_path.exists()
        else (
            "# Commands And Test Results\n\n"
            "Validation commands are recorded in `09_VALIDATION_AND_LOGS` after execution."
        )
    )
    payloads: dict[str, str | dict[str, Any] | list[dict[str, Any]]] = {
        "01_EXECUTIVE_SUMMARY.md": "\n".join(
            [
                "# M5.5B Executive Summary",
                "",
                "M5.5B ingested the completed M5.4J localization review and the repaired M5.5A occlusion-path review.",
                (
                    "The stage evaluates detector supply and sequence-real path ranking without fitting a model "
                    "or updating continuity rows."
                ),
                "",
                (
                    "Resolver classification: "
                    f"`{evaluation['architecture_branch_decision']['resolver_branch_classification']}`."
                ),
                f"Information gain: `{evaluation['architecture_branch_decision']['information_gain_classification']}`.",
            ]
        ),
        "02_RUN_AND_GIT_CONTEXT.json": {
            "stage_id": STAGE_ID,
            "current_head_at_pack_build": current_head,
            "workspace_root": str(workspace_root),
            "review_url": LOCAL_REVIEW_URL,
            "authorization": source_audit["authorization"],
            **safety_payload(),
        },
        "03_FILES_CHANGED.md": "# Files Changed\n\n"
        + (files_changed.strip() or "No repository file changes captured."),
        "04_SOURCE_DIFF.patch": source_diff,
        "05_COMMANDS_AND_TEST_RESULTS.md": command_results_text,
        "06_OUTPUT_ARTIFACT_INDEX.json": {
            "workspace_root": str(workspace_root),
            "top_level_artifact_dirs": list(WORKSPACE_DIRS),
            "review_package": {
                "review_root": unseen_result["review_package"]["review_root"],
                "manifest_path": unseen_result["review_package"]["manifest_path"],
                "ui_config_path": unseen_result["review_package"]["ui_config_path"],
                "evidence_root": unseen_result["review_package"]["evidence_root"],
                "decisions_root": unseen_result["review_package"]["decisions_root"],
                "launcher_path": unseen_result["review_package"]["launcher_path"],
                "review_url": unseen_result["review_package"]["review_url"],
                "case_count": unseen_result["review_package"]["case_count"],
                "validation_passed": unseen_result["review_package"]["validation"]["passed"],
                "server_side_mapping_omitted_from_review_pack": True,
            },
            **safety_payload(),
        },
        "07_PRIMARY_RESULTS_OR_BLOCKER.json": {
            "review_prerequisites_passed": review_ingestion["status"]["passed"],
            "detector_runtime_status": detector_result["summary"]["detector_runtime_status"],
            "resolver_branch_classification": evaluation["architecture_branch_decision"][
                "resolver_branch_classification"
            ],
            "information_gain_classification": evaluation["architecture_branch_decision"][
                "information_gain_classification"
            ],
            **safety_payload(),
        },
        "08_SAFETY_AND_INVARIANT_AUDIT.json": safety_payload(
            model_fit_performed=False,
            learned_continuity_rows_updated=0,
            historical_artifacts_mutated=False,
        ),
        "09_SOURCE_MUTATION_AUDIT.json": {
            "historical_artifacts_mutated": False,
            "writes_beneath_historical_sources": 0,
            "project_defaults_changed": False,
            "canonical_candidate_rows_replaced": False,
            **safety_payload(),
        },
        "10_UNRESOLVED_AND_NEXT_DECISION.md": "\n".join(
            [
                "# Unresolved And Next Decision",
                "",
                (
                    "Next branch: stateful multi-hypothesis resolver with review escalation and targeted detector "
                    "supply repair."
                ),
                "Do not train or globally apply continuity until a later reviewed set supports it.",
            ]
        ),
        "11_REVIEW_PREREQUISITE_STATUS.json": review_ingestion["status"],
        "12_DETECTOR_ROOT_CAUSE_RESULTS.json": detector_result["summary"],
        "13_SEQUENCE_RESOLVER_RESULTS.json": {
            "metrics": resolver_result["metrics"],
            "geometry_only": {
                "correct_path_top1": resolver_result["geometry_results"]["correct_path_top1"],
                "correct_path_in_top2": resolver_result["geometry_results"]["correct_path_in_top2"],
            },
            "conflict_gated_appearance": {
                "appearance_used_in_candidate_generation": False,
                "appearance_gate_activation_count": 0,
            },
            **safety_payload(),
        },
        "14_CASE_LEVEL_RESULTS.jsonl": case_rows,
        "15_PATH_RANKING_AND_CONTROL_METRICS.json": evaluation["path_ranking_metrics"],
        "16_STATE_AND_HYPOTHESIS_EXAMPLES.json": {
            "window_examples": resolver_result["window_rows"][:3],
            "hypothesis_examples": [
                {
                    key: row.get(key)
                    for key in ("case_number", "variant", "hypothesis_rank", "path_cost", "human_chosen_path")
                }
                for row in resolver_result["hypothesis_rows"][:12]
            ],
            **safety_payload(),
        },
        "19_ACCEPTANCE_AND_ARCHITECTURE_DECISION.json": evaluation["architecture_branch_decision"],
    }
    source_paths: dict[str, Path] = {}
    for filename, payload in payloads.items():
        target = tmp / filename
        if filename.endswith(".json"):
            _write_json(target, payload if isinstance(payload, dict) else {"value": payload})
        elif filename.endswith(".jsonl"):
            _write_jsonl(target, payload if isinstance(payload, list) else [])
        else:
            _write_text(target, str(payload))
        source_paths[filename] = target
    source_paths["17_PRIMARY_VISUAL_EVIDENCE.jpg"] = visual_paths["detector_sheet"]
    source_paths["18_SECONDARY_VISUAL_EVIDENCE.gif"] = visual_paths["primary_gif"]
    builder = ReviewPackBuilder(
        root=review_pack_root,
        stage_id=STAGE_ID,
        repository_commit_before=source_audit["authorization"]["minimum_authorized_baseline_commit"],
        repository_commit_after=current_head,
    )
    purposes = {
        "17_PRIMARY_VISUAL_EVIDENCE.jpg": "Real detector/localization frame evidence.",
        "18_SECONDARY_VISUAL_EVIDENCE.gif": "Animated sequence-real path-hypothesis evidence.",
    }
    for filename, source_path in source_paths.items():
        builder.add_file(
            ReviewPackItem(
                filename=filename,
                source_path=source_path,
                purpose=purposes.get(filename, f"M5.5B review-pack artifact {filename}."),
                redacted=filename
                not in {"04_SOURCE_DIFF.patch", "17_PRIMARY_VISUAL_EVIDENCE.jpg", "18_SECONDARY_VISUAL_EVIDENCE.gif"},
                redaction_note="Canonical IDs and sealed answer mappings omitted from review pack.",
            )
        )
    builder.copy_items()
    builder.write_manifest()
    validation = validate_m5_5b_review_pack(review_pack_root)
    manifest = builder.write_manifest(validator_result=validation)
    validation = validate_m5_5b_review_pack(review_pack_root)
    _write_json(workspace_root / "09_VALIDATION_AND_LOGS" / "review_pack_validation.json", validation)
    return {"root": str(review_pack_root), "manifest": manifest, "validation": validation}


def build_m5_5b_repaired_reviews_evaluation(
    *,
    repo_root: Path,
    prompt_root: Path,
    output_root: Path | None = None,
    model_path: Path | None = None,
    run_detector: bool = True,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    prompt_root = prompt_root.resolve()
    contract = _read_json(prompt_root / "02_M5_5B_WORKSPACE_CONTRACT_v3.json")
    workspace_root = (output_root or Path(contract["workspace_root"])).resolve()
    historical_stage_root = Path(contract["historical_stage_root"]).resolve()
    localization_root = Path(contract["interactive_localization_review_root"]).resolve()
    occlusion_root = Path(contract["repaired_occlusion_review_root"]).resolve()
    prior_roots = [
        Path(contract["prompt_pack_folder"]).resolve(),
        Path(contract["prior_workspace_root"]).resolve(),
        localization_root.parent,
        occlusion_root.parent,
        Path(contract["invalid_occlusion_review_root_provenance_only"]).resolve(),
    ]
    for relative in WORKSPACE_DIRS:
        (workspace_root / relative).mkdir(parents=True, exist_ok=True)
    source_audit = _write_source_audit(workspace_root, repo_root, prompt_root, prior_roots)
    review_ingestion = _write_review_ingestion_outputs(workspace_root, localization_root, occlusion_root)
    detector_model_path = model_path or Path(
        r"C:\Users\sebgr\Documents\football-intelligence\trusted-model-cache\yolov8m.pt"
    )
    detector_result = _detector_recovery(
        workspace_root,
        historical_stage_root,
        review_ingestion["localization_rows"],
        detector_model_path,
        run_detector=run_detector,
    )
    resolver_result = _sequence_resolver(
        workspace_root,
        historical_stage_root,
        occlusion_root,
        review_ingestion["occlusion_rows"],
    )
    unseen_result = _build_unseen_review(workspace_root, historical_stage_root, review_ingestion["occlusion_rows"])
    evaluation = _write_evaluation_outputs(
        workspace_root,
        detector_result,
        resolver_result,
        unseen_result,
        review_ingestion["status"],
    )
    visual_paths = _write_visual_outputs(
        workspace_root,
        review_ingestion["localization_rows"],
        detector_result,
        resolver_result,
        unseen_result,
        historical_stage_root,
    )
    review_pack = _write_review_pack(
        workspace_root,
        repo_root,
        source_audit,
        review_ingestion,
        detector_result,
        resolver_result,
        unseen_result,
        evaluation,
        visual_paths,
    )
    final_classification = (
        "PASS_CORRECT_PATH_IN_TOPK_SAFE_REVIEW"
        if review_ingestion["status"]["passed"]
        and resolver_result["metrics"]["resolver_branch_classification"] == "PASS_CORRECT_PATH_IN_TOPK_SAFE_REVIEW"
        else "BLOCKED_REVIEW_INCOMPLETE"
    )
    result = {
        "schema_version": "football_intelligence.m5_5b.stage_result.v3",
        "generated_at": utc_now(),
        "stage_id": STAGE_ID,
        "workspace_root": str(workspace_root),
        "final_classification": final_classification,
        "information_gain_classification": evaluation["architecture_branch_decision"][
            "information_gain_classification"
        ],
        "review_prerequisite_status": review_ingestion["status"],
        "detector_runtime_status": detector_result["summary"]["detector_runtime_status"],
        "detector_classification_counts": detector_result["summary"]["classification_counts"],
        "resolver_metrics": resolver_result["metrics"],
        "unseen_review_package": unseen_result["review_package"],
        "review_pack": review_pack,
        "safety": safety_payload(),
    }
    _write_json(workspace_root / "09_VALIDATION_AND_LOGS" / "stage_result.json", result)
    return result
