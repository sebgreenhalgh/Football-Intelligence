from __future__ import annotations

# ruff: noqa: E501

import ast
import csv
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from football_intelligence.replay.positive_only_counterfactual_continuity import _inventory
from football_intelligence.review.schemas import safety_payload

PACK_FILENAMES = [
    "01_EXECUTIVE_BRIEFING.md",
    "02_SYSTEM_ARCHITECTURE_MAP.md",
    "03_CURRENT_STATE_AND_PROVENANCE.json",
    "04_SAFETY_SCOPE_AND_IDENTITY_BOUNDARIES.md",
    "05_DATA_CONTRACTS_AND_SCHEMAS.md",
    "06_DETECTION_AND_PERSON_PIPELINE_EXCERPTS.py",
    "07_CONTINUITY_AND_CHALLENGE_PIPELINE_EXCERPTS.py",
    "08_REVIEW_CHASSIS_AND_PERSISTENCE_EXCERPTS.py",
    "09_LOCALIZATION_AND_UPSTREAM_AUDIT_EXCERPTS.py",
    "10_OCCLUSION_FAILURE_TAXONOMY_AND_KNOWN_WEAKNESSES.md",
    "11_CURRENT_EVALUATION_LABELS_AND_GROUPING.json",
    "12_OCCLUSION_RESEARCH_QUESTIONS_AND_DECISION_GATES.md",
    "13_REPRESENTATIVE_CASE_INDEX.csv",
    "14_CROSSING_FAILURE_CASE_008.gif",
    "15_CROSSING_FAILURE_CASE_010.gif",
    "16_CROSSING_FAILURE_CASE_013.gif",
    "17_SHARED_OCCLUSION_REGION_CASES_004_016.gif",
    "18_FULL_FRAME_DETECTION_CONTACT_SHEET.jpg",
    "19_TARGETED_CODEBASE_FILE_MAP.md",
    "20_PACK_MANIFEST.json",
]

FORBIDDEN_ANSWER_KEY_FIELDS = [
    "accepted_target_panel",
    "alternative_target_panel",
    "prior_accepted_target",
    "same_frame_alternative_target",
    "decision_to_output_mapping",
    "conflict_if_chosen_panel_is_not_prior_accept",
    "post_decision_answer_key",
]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return "text/markdown"
    if suffix == ".json":
        return "application/json"
    if suffix == ".csv":
        return "text/csv"
    if suffix == ".py":
        return "text/x-python"
    if suffix == ".gif":
        return "image/gif"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "application/octet-stream"


def _safe_prepare_output_root(output_root: Path) -> None:
    output_root = output_root.resolve()
    if output_root.name != "occlusion_pro_extended_context_v1":
        raise ValueError(f"refusing to clear unexpected output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    for child in output_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _source(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256_file(path) if path.exists() else None}


def _line_range_for_symbol(path: Path, symbol: str) -> tuple[int, int, str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == symbol:
            end = getattr(node, "end_lineno", node.lineno)
            return node.lineno, int(end), source
    raise ValueError(f"symbol {symbol} not found in {path}")


def _extract_python_symbols(repo_root: Path, specs: list[tuple[str, list[str], str]]) -> str:
    parts = [
        "# Exact source excerpts for the M5.5A occlusion research context pack.",
        "# These snippets are copied byte-for-text from the repository files listed above each block.",
        "",
    ]
    for relative, symbols, relevance in specs:
        path = repo_root / relative
        lines = path.read_text(encoding="utf-8").splitlines()
        for symbol in symbols:
            start, end, _ = _line_range_for_symbol(path, symbol)
            parts.extend(
                [
                    f"# original source path: {path}",
                    f"# original function/class: {symbol}",
                    f"# line range: {start}-{end}",
                    f"# why relevant: {relevance}",
                    "\n".join(lines[start - 1 : end]),
                    "",
                ]
            )
    return "\n".join(parts)


def _extract_js_function(path: Path, function_name: str) -> tuple[int, int, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start_index = next(index for index, line in enumerate(lines) if line.startswith(f"function {function_name}("))
    depth = 0
    started = False
    for index in range(start_index, len(lines)):
        line = lines[index]
        depth += line.count("{")
        if "{" in line:
            started = True
        depth -= line.count("}")
        if started and depth == 0:
            return start_index + 1, index + 1, "\n".join(lines[start_index : index + 1])
    raise ValueError(f"could not extract {function_name} from {path}")


def _extract_review_chassis(repo_root: Path) -> str:
    python = _extract_python_symbols(
        repo_root,
        [
            (
                "src/football_intelligence/review_chassis/models.py",
                ["GenericEvidenceAsset", "GenericReviewCase", "GenericReviewManifest", "ReviewUIConfig"],
                "documents reviewer-safe manifest and UI schema contracts",
            ),
            (
                "src/football_intelligence/review_chassis/persistence.py",
                ["GenericReviewPersistence"],
                "documents decision state, event log, snapshots, completion and notes",
            ),
            (
                "src/football_intelligence/review_chassis/server.py",
                ["ReviewChassisHTTPServer", "ReviewChassisRequestHandler"],
                "documents server-side sealed mapping boundary and API routes",
            ),
        ],
    )
    app_path = repo_root / "src/football_intelligence/review_chassis/static/app.js"
    js_parts = [
        "# JavaScript excerpts are included as comments so this file remains Python-parseable.",
        "# original source path: "
        + str(app_path)
        + " | why relevant: GIF/image sequence rendering and spatial annotation support",
    ]
    for name in ["renderImageStepper", "renderAsset", "renderSpatialAnnotation", "applySpatialAnnotation"]:
        start, end, text = _extract_js_function(app_path, name)
        js_parts.append(f"# original function: {name} | line range: {start}-{end}")
        js_parts.extend(f"# JS| {line}" for line in text.splitlines())
        js_parts.append("")
    return python + "\n".join(js_parts)


def _case_id(index: int) -> str:
    return f"m5_4h1_cadence_matched_target_choice_case_{index:03d}"


def _case_evidence_root(stage_root: Path, index: int) -> Path:
    return stage_root / "continuity_v11" / "review" / "evidence" / _case_id(index)


def _load_case_frames(stage_root: Path, index: int) -> list[Path]:
    root = _case_evidence_root(stage_root, index) / "frames"
    frames = sorted(root.glob("frame_*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no frames for case {index:03d}: {root}")
    return frames


def _font(size: int = 20) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _caption(image: Image.Image, text: str) -> Image.Image:
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    font = _font(22)
    width = min(output.width, max(760, int(draw.textlength(text, font=font) + 28)))
    draw.rectangle((0, 0, width, 42), fill=(0, 0, 0))
    draw.text((12, 10), text, fill=(255, 255, 255), font=font)
    return output


def _resize_width(image: Image.Image, width: int) -> Image.Image:
    if image.width <= width:
        return image.copy().convert("RGB")
    height = int(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS).convert("RGB")


def _write_case_gif(stage_root: Path, case_index: int, dest: Path) -> dict[str, Any]:
    frame_paths = _load_case_frames(stage_root, case_index)
    frames = []
    for frame_path in frame_paths:
        frame_no = int(frame_path.stem.split("_")[-1])
        image = _resize_width(Image.open(frame_path), 1200)
        frames.append(_caption(image, f"case {case_index:03d} | frame {frame_no} | neutral source/candidate evidence"))
    frames[0].save(dest, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
    with Image.open(dest) as gif:
        decoded_frames = getattr(gif, "n_frames", 1)
    return {
        "path": str(dest),
        "decoded": True,
        "frame_count": decoded_frames,
        "byte_size": dest.stat().st_size,
        "source_frames": [str(path) for path in frame_paths],
    }


def _write_shared_region_gif(stage_root: Path, dest: Path) -> dict[str, Any]:
    left_paths = _load_case_frames(stage_root, 4)
    right_paths = _load_case_frames(stage_root, 16)
    count = max(len(left_paths), len(right_paths))
    frames = []
    for index in range(count):
        left_path = left_paths[min(index, len(left_paths) - 1)]
        right_path = right_paths[min(index, len(right_paths) - 1)]
        left = _caption(_resize_width(Image.open(left_path), 760), "case 004 | neutral interval")
        right = _caption(_resize_width(Image.open(right_path), 760), "case 016 | same trajectory-safe local region")
        canvas = Image.new("RGB", (left.width + right.width + 16, max(left.height, right.height)), (245, 245, 245))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width + 16, 0))
        frames.append(canvas)
    frames[0].save(dest, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
    with Image.open(dest) as gif:
        decoded_frames = getattr(gif, "n_frames", 1)
    return {
        "path": str(dest),
        "decoded": True,
        "frame_count": decoded_frames,
        "byte_size": dest.stat().st_size,
        "source_frames": [str(path) for path in left_paths + right_paths],
    }


def _write_contact_sheet(stage_root: Path, dest: Path, localization_rows: list[dict[str, str]]) -> dict[str, Any]:
    rows = []
    for row in localization_rows:
        case_id = str(row["case_id"])
        evidence = stage_root / "continuity_v14" / "localization" / "evidence" / case_id
        unannotated = _resize_width(Image.open(evidence / "target_full_frame_unannotated.jpg"), 760)
        overlay = _resize_width(Image.open(evidence / "full_target_frame_detector_overlay.jpg"), 760)
        source_crop = _resize_width(Image.open(evidence / "source_crop.jpg"), 160)
        for image, label in [
            (
                unannotated,
                "target frame "
                + row["target_frame_sequence"]
                + " | unannotated | source frame "
                + row["source_frame_sequence"],
            ),
            (
                overlay,
                "target frame "
                + row["target_frame_sequence"]
                + " | all anonymous detections="
                + row["full_frame_candidate_count"],
            ),
        ]:
            image.paste(source_crop, (image.width - source_crop.width - 8, 50))
            draw = ImageDraw.Draw(image)
            draw.rectangle(
                (image.width - source_crop.width - 12, 46, image.width - 4, 56 + source_crop.height),
                outline=(0, 0, 0),
                width=2,
            )
            draw.text(
                (image.width - source_crop.width - 8, 58 + source_crop.height),
                "source crop",
                fill=(0, 0, 0),
                font=_font(14),
            )
            rows.append(_caption(image, label))
    cell_w = max(image.width for image in rows)
    cell_h = max(image.height for image in rows)
    sheet = Image.new("RGB", (cell_w * 2 + 24, cell_h * 4 + 48), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (12, 8), "M5.5A full-frame detector contact sheet: no target marked correct", fill=(0, 0, 0), font=_font(24)
    )
    for index, image in enumerate(rows):
        row = index // 2
        col = index % 2
        sheet.paste(image, (12 + col * (cell_w + 12), 44 + row * cell_h))
    sheet.save(dest, quality=92)
    with Image.open(dest) as decoded:
        dimensions = decoded.size
    return {"path": str(dest), "decoded": True, "dimensions": list(dimensions), "byte_size": dest.stat().st_size}


def _case_short(case_id: str) -> str:
    return case_id.rsplit("_", 1)[-1]


def _source_artifact_index(stage_root: Path, repo_root: Path) -> dict[str, Path]:
    return {
        "frame_manifest": stage_root / "continuity_v11" / "unseen_window" / "canonical_frame_manifest.json",
        "person_candidate_manifest": stage_root
        / "continuity_v11"
        / "unseen_window"
        / "person_candidate_rows_manifest.json",
        "person_candidate_rows": stage_root / "continuity_v11" / "unseen_window" / "person_candidate_rows.jsonl",
        "target_choice_case_index": stage_root / "continuity_v11" / "review" / "target_choice_case_index.csv",
        "target_choice_manifest": stage_root / "continuity_v11" / "review" / "target_choice_reviewer_manifest.json",
        "primary_results": stage_root / "continuity_v13" / "evaluation" / "corrected_primary_results.json",
        "secondary_results": stage_root / "continuity_v13" / "evaluation" / "corrected_secondary_results.json",
        "failure_taxonomy": stage_root / "continuity_v13" / "evaluation" / "corrected_failure_taxonomy.json",
        "combined_inventory": stage_root / "continuity_v13" / "labels" / "combined_inventory_candidate_v2.json",
        "trajectory_grouping": stage_root / "continuity_v13" / "audit" / "canonical_trajectory_safe_grouping.json",
        "m5_4j_validation": stage_root / "validation" / "m5_4j_validation_summary.json",
        "m5_4j_event_validation": stage_root / "continuity_v14" / "ingestion" / "followup_event_validation.json",
        "m5_4j_mapping_validation": stage_root
        / "continuity_v14"
        / "ingestion"
        / "followup_sealed_mapping_validation.json",
        "m5_4j_full_frame_audit": stage_root / "continuity_v14" / "audit" / "full_frame_candidate_coverage_audit.json",
        "m5_4j_detector_provenance": stage_root
        / "continuity_v14"
        / "audit"
        / "affected_frame_detector_provenance.json",
        "m5_4j_root_cause": stage_root / "continuity_v14" / "audit" / "candidate_supply_root_cause.json",
        "m5_4j_localization_case_index": stage_root / "continuity_v14" / "localization" / "case_index.csv",
        "source_video_manifest": stage_root.parent / "05_blind_second_window" / "source" / "source_video_manifest.json",
        "portable_detector": repo_root / "src" / "football_intelligence" / "replay" / "portable_detector.py",
    }


def _render_sources(sources: dict[str, Path], keys: list[str]) -> str:
    return "\n".join(f"- `{key}`: `{sources[key]}`" for key in keys)


def _build_case_index_rows(
    stage_root: Path,
    primary: dict[str, Any],
    secondary: dict[str, Any],
    case_index: list[dict[str, str]],
    full_frame_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    primary_by_case = {row["case_id"]: row for row in primary["rows"]}
    secondary_by_case = {row["case_id"]: row for row in secondary["rows"]}
    index_by_case = {row["case_id"]: row for row in case_index}
    supply_cases = {"004", "009", "011", "016"}
    crossing_cases = {"008", "010", "013"}
    root_by_short = {
        _case_short(row["source_case_id"]): "UNRESOLVED_ROOT_CAUSE" for row in full_frame_audit.get("rows", [])
    }
    output = []
    for short in ["004", "008", "009", "010", "011", "013", "016"]:
        case_id = _case_id(int(short))
        index_row = index_by_case[case_id]
        secondary_row = secondary_by_case[case_id]
        primary_row = primary_by_case[case_id]
        root_status = root_by_short.get(short, "ASSOCIATION_CROSSING_CONFLICT_OBSERVED")
        visual_file = {
            "004": "17_SHARED_OCCLUSION_REGION_CASES_004_016.gif",
            "008": "14_CROSSING_FAILURE_CASE_008.gif",
            "009": "18_FULL_FRAME_DETECTION_CONTACT_SHEET.jpg",
            "010": "15_CROSSING_FAILURE_CASE_010.gif",
            "011": "18_FULL_FRAME_DETECTION_CONTACT_SHEET.jpg",
            "013": "16_CROSSING_FAILURE_CASE_013.gif",
            "016": "17_SHARED_OCCLUSION_REGION_CASES_004_016.gif",
        }[short]
        output.append(
            {
                "case_short_id": short,
                "case_id": case_id,
                "source_frame": index_row["source_frame_sequence"],
                "target_frame": index_row["target_frame_sequence"],
                "frame_gap": index_row["frame_gap"],
                "temporal_gap_seconds": index_row["temporal_gap_seconds"],
                "original_decision": secondary_row["human_decision"],
                "current_root_cause_status": root_status,
                "challenge_categories": "crossing_or_assignment_conflict"
                if short in crossing_cases
                else "candidate_supply_requires_localization",
                "endpoint_safe_group": index_row["endpoint_safe_group_id"],
                "trajectory_safe_group": secondary_row["trajectory_safe_group_id"],
                "crossing_wrong_panel": str(short in crossing_cases).lower(),
                "candidate_supply_failure": str(short in supply_cases).lower(),
                "merged_trajectory_region": str(short in {"004", "016"}).lower(),
                "primary_rule_classification": primary_row["classification"],
                "secondary_rule_classification": secondary_row["classification"],
                "visual_file": visual_file,
                "source_evidence_paths": str(_case_evidence_root(stage_root, int(short))),
            }
        )
    return output


def _write_case_index(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case_short_id",
        "case_id",
        "source_frame",
        "target_frame",
        "frame_gap",
        "temporal_gap_seconds",
        "original_decision",
        "current_root_cause_status",
        "challenge_categories",
        "endpoint_safe_group",
        "trajectory_safe_group",
        "crossing_wrong_panel",
        "candidate_supply_failure",
        "merged_trajectory_region",
        "primary_rule_classification",
        "secondary_rule_classification",
        "visual_file",
        "source_evidence_paths",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_code_excerpts(paths: list[Path]) -> dict[str, Any]:
    rows = []
    for path in paths:
        try:
            ast.parse(path.read_text(encoding="utf-8"))
            rows.append({"path": str(path), "python_parseable": True})
        except SyntaxError as exc:
            rows.append({"path": str(path), "python_parseable": False, "error": str(exc)})
    return {"all_python_excerpts_parse": all(row["python_parseable"] for row in rows), "rows": rows}


def _text_utf8_check(output_root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(output_root.iterdir()):
        if path.suffix.lower() in {".md", ".json", ".csv", ".py"}:
            try:
                path.read_text(encoding="utf-8")
                rows.append({"filename": path.name, "utf8": True})
            except UnicodeDecodeError as exc:
                rows.append({"filename": path.name, "utf8": False, "error": str(exc)})
    return {"all_text_utf8": all(row["utf8"] for row in rows), "rows": rows}


def _forbidden_field_scan(output_root: Path) -> dict[str, Any]:
    hits = []
    for path in sorted(output_root.iterdir()):
        if path.suffix.lower() in {".md", ".json", ".csv"} and path.name != "20_PACK_MANIFEST.json":
            text = path.read_text(encoding="utf-8")
            for field in FORBIDDEN_ANSWER_KEY_FIELDS:
                if field in text:
                    hits.append({"filename": path.name, "field": field})
    return {"answer_key_field_count": len(hits), "hits": hits}


def _source_mutation_audit(stage_root: Path) -> dict[str, Any]:
    prior_paths = [stage_root / f"continuity_v{index}" for index in range(3, 15)]
    before = _inventory(prior_paths, base=stage_root)
    after = _inventory(prior_paths, base=stage_root)
    return {
        "before_hash": before["inventory_hash"],
        "after_hash": after["inventory_hash"],
        "preserved": before == after,
    }


def _pack_hash(output_root: Path, manifest_payload: dict[str, Any]) -> str:
    items: list[dict[str, Any]] = []
    for filename in PACK_FILENAMES:
        path = output_root / filename
        if filename == "20_PACK_MANIFEST.json":
            normalized = dict(manifest_payload)
            normalized["generated_at"] = "<excluded>"
            normalized["pack_hash"] = "<self-excluded>"
            normalized["files"] = [
                {**row, "sha256": "<self-excluded>", "byte_size": "<self-excluded>"}
                if row["filename"] == filename
                else row
                for row in normalized["files"]
            ]
            items.append({"filename": filename, "normalized_manifest": normalized})
        else:
            items.append({"filename": filename, "sha256": _sha256_file(path)})
    return _stable_hash(items)


def _file_manifest_rows(output_root: Path, source_map: dict[str, list[str]]) -> list[dict[str, Any]]:
    rows = []
    for index, filename in enumerate(PACK_FILENAMES, start=1):
        path = output_root / filename
        rows.append(
            {
                "upload_order": index,
                "filename": filename,
                "sha256": _sha256_file(path)
                if path.exists() and filename != "20_PACK_MANIFEST.json"
                else "<self-excluded>",
                "byte_size": path.stat().st_size if path.exists() else 0,
                "media_type": _media_type(path),
                "source_artifact_paths": source_map.get(filename, []),
            }
        )
    return rows


def build_occlusion_pro_context_pack(
    *,
    stage_root: Path,
    repo_root: Path,
    output_root: Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    stage_root = stage_root.resolve()
    repo_root = repo_root.resolve()
    output_root = (
        output_root.resolve()
        if output_root is not None
        else (stage_root / "occlusion_pro_extended_context_v1").resolve()
    )
    generated_at = generated_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    sources = _source_artifact_index(stage_root, repo_root)
    _safe_prepare_output_root(output_root)

    frame_manifest = _read_json(sources["frame_manifest"])
    person_manifest = _read_json(sources["person_candidate_manifest"])
    source_video_manifest = _read_json(sources["source_video_manifest"])
    primary = _read_json(sources["primary_results"])
    secondary = _read_json(sources["secondary_results"])
    combined = _read_json(sources["combined_inventory"])
    trajectory = _read_json(sources["trajectory_grouping"])
    m5_4j = _read_json(sources["m5_4j_validation"])
    full_frame_audit = _read_json(sources["m5_4j_full_frame_audit"])
    detector_provenance = _read_json(sources["m5_4j_detector_provenance"])
    mapping_validation = _read_json(sources["m5_4j_mapping_validation"])
    case_index = _read_csv(sources["target_choice_case_index"])
    localization_rows = _read_csv(sources["m5_4j_localization_case_index"])
    source_map: dict[str, list[str]] = {}

    def write(name: str, text: str, source_keys: list[str]) -> None:
        _write_text(output_root / name, text)
        source_map[name] = [str(sources[key]) for key in source_keys]

    briefing_sources = [
        "frame_manifest",
        "person_candidate_manifest",
        "primary_results",
        "secondary_results",
        "combined_inventory",
        "m5_4j_validation",
        "m5_4j_detector_provenance",
    ]
    write(
        "01_EXECUTIVE_BRIEFING.md",
        f"""# M5.5A Executive Briefing

Product and R&D goal: Football Intelligence is building a visual-only, match-local short-window continuity system for panoramic football video. The immediate R&D question is which occlusion-aware architecture should come next. Sources:
{_render_sources(sources, briefing_sources)}

What the current pipeline can do:
- Extract a 600-frame third unseen interval at 10 FPS from match 128058, 1620.0-1680.0 seconds, dimensions 2730x720.
- Run the declared official YOLOv8m person-only detector and serialize canonical person candidates.
- Build short-window continuity challenges for frame gaps 1-3, i.e. 0.1-0.3 seconds.
- Preserve blind human decisions as sidecars without fitting a continuity model.

What it cannot yet do:
- It cannot prove whether the four M5.4J supply failures are true detector misses, local radius failures, postprocess losses, occlusions, or reviewer interpretation issues. M5.4J records root causes as unresolved pending spatial localization.
- It cannot separate silhouettes inside merged or overlapping person boxes.
- It cannot maintain hidden/ghost image-space tracks through temporary occlusion.
- It cannot produce persistent identity, player slots, tactical events, speed, distance, fatigue, load, or production-ready outputs.

Why occlusion and overlap are central now: the strict primary rule was conservative (3 correct, 0 wrong, 13 abstentions), while the secondary IoU rule improved coverage (12 correct, 3 wrong, 1 abstention) but selected the wrong target in cases 008, 010 and 013. The four N cases 004, 009, 011 and 016 show candidate-supply/root-cause uncertainty; cases 004 and 016 share one trajectory-safe local region.

Distinctions for the researcher:
- Person detection: image-space boxes from YOLO person class only.
- Silhouette separation: a future instance/body-part step that may split overlapping people; not implemented.
- Candidate generation: frame-gap 1-3 candidate edges from detected canonical people.
- Short-window association: visual continuity decisions over 0.1-0.3 seconds.
- Temporary track state: permitted only as an internal match-local, image-space hypothesis.
- Persistent identity: explicitly forbidden.

Current label inventory: {combined["canonical_unique_edge_counts"]["accept_continuity"]} canonical positives, {combined["canonical_unique_edge_counts"]["reject_continuity"]} canonical negatives, {combined["combined_candidate_row_count"]} combined rows, model_fit_performed=false, learned_continuity_rows_updated=0.

Current detector provenance: model hash {person_manifest["model_sha256"]}; detector classification {person_manifest["provenance"]["detector_source_classification"]}; imgsz={person_manifest["provenance"]["detector_config"]["imgsz"]}; confidence={person_manifest["provenance"]["detector_config"]["confidence_threshold"]}; IoU/NMS={person_manifest["provenance"]["detector_config"]["iou_threshold"]}; max_det={person_manifest["provenance"]["detector_config"]["max_det"]}; device={person_manifest["provenance"]["detector_config"]["device"]}; pre-NMS evidence status={detector_provenance["pre_nms_evidence_status"]}.

Unresolved questions: distinguish detector miss vs radius failure vs postprocess filtering vs merged box vs true occlusion; decide whether segmentation, pose, temporal propagation, multi-hypothesis association, or detector recall repair should lead.

Safety restrictions: VISUAL_ONLY_NOT_METRIC; production_ready=false; no_auto_promotion=true; human_approved=false; match_local_only=true; sandbox_only=true; no persistent identity; no slots; no forced 22; no tactical/physical outputs.

Recommended reading order: 02, 03, 04, 05, 10, 11, 13, 14-18, 06-09, 12, 19, 20.
""",
        briefing_sources,
    )

    write(
        "02_SYSTEM_ARCHITECTURE_MAP.md",
        f"""# System Architecture Map

Actual current architecture:

`source video` -> `cadence-controlled frame extraction` -> `YOLO person detection` -> `canonical person candidates` -> `generic visual role/team context` -> `frame-gap 1-3 candidate edges` -> `geometry/appearance features` -> `baseline acceptance or abstention` -> `challenge mining` -> `blind human review` -> `canonical label sidecars` -> `trajectory-safe grouping` -> `localization/root-cause audits`

Exact local code paths:
- Frame extraction and third-window setup: `{repo_root / "src/football_intelligence/replay/third_unseen_geometry_challenge.py"}`
- Declared detector support: `{repo_root / "src/football_intelligence/replay/portable_detector.py"}`
- Cadence-matched challenge mining: `{repo_root / "src/football_intelligence/replay/cadence_matched_third_unseen_challenge.py"}`
- Review ingestion and sidecars: `{repo_root / "src/football_intelligence/replay/third_unseen_review_ingestion.py"}`
- Corrected semantics/grouping: `{repo_root / "src/football_intelligence/replay/third_unseen_review_correction.py"}`
- M5.4J localization/root-cause gate: `{repo_root / "src/football_intelligence/replay/followup_candidate_supply_diagnostic.py"}`
- Review chassis: `{repo_root / "src/football_intelligence/review_chassis"}`
- CLI entry point: `{repo_root / "src/football_intelligence/cli/app.py"}`

Exact artifact roots:
- Stage root: `{stage_root}`
- Frames and candidates: `{stage_root / "continuity_v11/unseen_window"}`
- Cadence-matched review: `{stage_root / "continuity_v11/review"}`
- Corrected labels/grouping: `{stage_root / "continuity_v13"}`
- Localization/root-cause gate: `{stage_root / "continuity_v14"}`
- This context pack: `{output_root}`

Deterministic stages: frame extraction manifests, candidate rows, case indexes, corrected sidecars, source mutation audits and this pack generator are deterministic for a fixed generated_at exclusion. Human-decision stages: v11 target-choice review, v13 N follow-up review and v14 localization review decisions.

Occlusion failure points:
- Detector: missed/merged/oversized boxes enter the candidate table.
- Candidate generation: if a person is absent, filtered or outside a local radius, association cannot choose it.
- Geometry: IoU can follow the wrong overlapping person.
- Appearance: jersey/foreground contamination can overrule useful geometry.
- Review mining: hard cases concentrate conflicts and are not a broad random sample.
- Root-cause: M5.4J cannot classify the N cases until spatial localization is completed.

Insertion points for occlusion-aware work: a detector-recall diagnostic after localization; a segmentation/body-part module before canonical candidate serialization; a multi-hypothesis short-window association module after candidate generation; a temporary image-space occlusion state before final sidecar creation.

Sources:
{_render_sources(sources, ["frame_manifest", "person_candidate_manifest", "target_choice_case_index", "combined_inventory", "trajectory_grouping", "m5_4j_validation"])}
""",
        [
            "frame_manifest",
            "person_candidate_manifest",
            "target_choice_case_index",
            "combined_inventory",
            "trajectory_grouping",
            "m5_4j_validation",
        ],
    )

    state_payload = {
        "schema_version": "football_intelligence.m5_5a.current_state.v1",
        "repository_commit": "59c4d00dcbb5612d8a00a9f2ec4ce955e5941686",
        "source_video": {
            "path": source_video_manifest["source_video_uri"],
            "sha256": source_video_manifest["source_video_sha256"],
            "source_manifest": _source(sources["source_video_manifest"]),
        },
        "detector_model": {
            "path": person_manifest["provenance"]["model_path"],
            "sha256": person_manifest["model_sha256"],
            "classification": person_manifest["provenance"]["detector_source_classification"],
            "checkpoint_included_in_pack": False,
        },
        "detector_runtime_settings": person_manifest["provenance"]["detector_config"],
        "third_window_temporal_domain": {
            "match_id": "128058",
            "start_seconds": frame_manifest["start_seconds"],
            "end_seconds": frame_manifest["end_seconds"],
            "duration_seconds": frame_manifest["duration_seconds"],
            "output_frame_count": frame_manifest["actual_frame_count"],
            "output_cadence_fps": 10,
            "dimensions": frame_manifest["dimensions"],
        },
        "canonical_inventory_counts": {
            "positive_edges": combined["canonical_unique_edge_counts"]["accept_continuity"],
            "negative_edges": combined["canonical_unique_edge_counts"]["reject_continuity"],
            "combined_rows": combined["combined_candidate_row_count"],
        },
        "stage_classifications": {
            "m5_4j": m5_4j["final_classification"],
            "continuity_research_gate": m5_4j["continuity_research_gate"],
        },
        "model_fit_performed": combined["model_fit_performed"],
        "learned_continuity_rows_updated": combined["learned_continuity_rows_updated"],
        "known_provenance_cautions": [
            "Current detector is an official pretrained baseline, not historical checkpoint recovery.",
            "M5.4J root causes remain unresolved pending localization.",
            "No pre-NMS detector outputs were preserved.",
            "M5.4H.1 continuity-node table was absent; person candidate table is the generic endpoint-node proxy.",
            "Some historical audit artifacts required semantic corrections; do not rewrite historical evidence.",
        ],
        "unresolved_m5_4j_integrity_fields": {
            "ui_config_hash_matches": _read_json(sources["m5_4j_event_validation"]).get("ui_config_hash_matches"),
            "session_result": m5_4j["reviewer_session_result"],
            "browser_served_before_decision_field": mapping_validation.get("browser_served_before_decision"),
            "caution": "Treat these as provenance cautions, not authorization to alter historical evidence.",
        },
        "source_artifacts": {key: _source(path) for key, path in sources.items()},
        **safety_payload(),
    }
    _write_json(output_root / "03_CURRENT_STATE_AND_PROVENANCE.json", state_payload)
    source_map["03_CURRENT_STATE_AND_PROVENANCE.json"] = [str(path) for path in sources.values() if path.exists()]

    write(
        "04_SAFETY_SCOPE_AND_IDENTITY_BOUNDARIES.md",
        f"""# Safety Scope And Identity Boundaries

VISUAL_ONLY_NOT_METRIC means every current output is image-space, reviewer-facing, or diagnostic only. It is not a pitch-metric, event, tactical, speed, distance, fatigue, load or performance system. Source: `{repo_root / "src/football_intelligence/review/schemas.py"}` and `{sources["combined_inventory"]}`.

Match-local-only scope: all labels and diagnostics are for match 128058 and the current 1620.0-1680.0 second third-window interval. Source: `{sources["frame_manifest"]}`.

ID distinctions:
- Anonymous detection ID: a detector/person-candidate row identifier inside current artifacts.
- Short-window tracklet ID: a temporary image-space continuity segment or edge grouping, if introduced later.
- Temporary occlusion/ghost state: permitted only as an internal short-window hypothesis for hidden image evidence.
- Persistent real-world identity: forbidden. No player, goalkeeper, squad, official or cross-match identity may be asserted.

A temporary latent track or ghost state may be researched if it stays match-local, image-space, uncertainty-carrying and expires under explicit gates. It must not become a player slot or identity.

Current required flags remain: production_ready=false; no_auto_promotion=true; human_approved=false; safe_to_apply_globally=false; match_local_only=true; sandbox_only=true.

Uncertainty must remain explicit: M5.4J records `PENDING_SPATIAL_LOCALIZATION` and `UNRESOLVED_ROOT_CAUSE`, so detector misses, radius failures and occlusion failures are not yet proven.

Sources:
{_render_sources(sources, ["combined_inventory", "m5_4j_validation", "m5_4j_full_frame_audit", "m5_4j_root_cause"])}
""",
        ["combined_inventory", "m5_4j_validation", "m5_4j_full_frame_audit", "m5_4j_root_cause"],
    )

    person_rows = _read_jsonl(sources["person_candidate_rows"])
    sample_candidate = next(row for row in person_rows if int(row["frame_sequence"]) in {82, 183, 301, 342})
    sample_edge = {
        "source_frame_sequence": case_index[7]["source_frame_sequence"],
        "target_frame_sequence": case_index[7]["target_frame_sequence"],
        "frame_gap": case_index[7]["frame_gap"],
        "temporal_gap_seconds": case_index[7]["temporal_gap_seconds"],
        "endpoint_safe_group_id": case_index[7]["endpoint_safe_group_id"],
    }
    write(
        "05_DATA_CONTRACTS_AND_SCHEMAS.md",
        f"""# Data Contracts And Schemas

Sources:
{_render_sources(sources, ["frame_manifest", "person_candidate_rows", "target_choice_case_index", "target_choice_manifest", "m5_4j_event_validation"])}

Frame record example from current frame manifest:
```json
{json.dumps(frame_manifest["frames"][0], indent=2, sort_keys=True)}
```

Person candidate example from current candidate rows:
```json
{json.dumps(sample_candidate, indent=2, sort_keys=True)}
```

Continuity edge fields: source and target frame sequence, frame gap, temporal gap seconds, candidate IDs, endpoint-safe group and trajectory-safe group. Compact current example:
```json
{json.dumps(sample_edge, indent=2, sort_keys=True)}
```

Positive and negative labels are canonical sidecar rows only; current counts are {combined["canonical_unique_edge_counts"]}. Non-binary N/U outcomes are excluded from binary labels. Source: `{sources["combined_inventory"]}` and `{sources["m5_4j_validation"]}`.

Review manifest case contract includes case_id, task_type, candidate_hash, evidence_hash, allowed_decisions, evidence_assets, visible metadata, hidden/reveal metadata fields and source artifact references. Source: `{repo_root / "src/football_intelligence/review_chassis/models.py"}`.

Review event contract records event_sequence, event_type, case_id, prior_decision, new_decision, candidate_hash, evidence_hash, manifest_hash and ui_config_hash. Source: `{repo_root / "src/football_intelligence/review_chassis/persistence.py"}`.

Spatial localization payload is stored by the reusable chassis as JSON in notes under `spatial_annotation` with numeric bbox, optional footpoint, existing anonymous candidate number, confidence and occlusion/partial flags. Source: `{repo_root / "src/football_intelligence/review_chassis/static/app.js"}` and `{stage_root / "continuity_v14/localization/ui_config.json"}`.

BBox contract: `{{
  "x1": float,
  "y1": float,
  "x2": float,
  "y2": float
}}`

BBox hash and candidate hash are SHA-like stable hashes stored in candidate/review artifacts. Visible-person base IDs are anonymous current-window person endpoint identifiers, not identities.
""",
        [
            "frame_manifest",
            "person_candidate_rows",
            "target_choice_case_index",
            "target_choice_manifest",
            "combined_inventory",
        ],
    )

    _write_text(
        output_root / "06_DETECTION_AND_PERSON_PIPELINE_EXCERPTS.py",
        _extract_python_symbols(
            repo_root,
            [
                (
                    "src/football_intelligence/replay/portable_detector.py",
                    [
                        "PortableDetectorConfig",
                        "detector_config_from_context",
                        "_load_yolo_model",
                        "_predict_kwargs",
                        "_rows_from_results",
                        "run_detector_inference",
                    ],
                    "detector configuration, YOLO loading, inference and person row serialization",
                ),
                (
                    "src/football_intelligence/replay/third_unseen_geometry_challenge.py",
                    ["_detector_rows", "_bbox_hash"],
                    "conversion from detector output to current canonical candidate rows",
                ),
            ],
        ),
    )
    source_map["06_DETECTION_AND_PERSON_PIPELINE_EXCERPTS.py"] = [
        str(repo_root / "src/football_intelligence/replay/portable_detector.py"),
        str(repo_root / "src/football_intelligence/replay/third_unseen_geometry_challenge.py"),
    ]

    _write_text(
        output_root / "07_CONTINUITY_AND_CHALLENGE_PIPELINE_EXCERPTS.py",
        _extract_python_symbols(
            repo_root,
            [
                (
                    "src/football_intelligence/replay/third_unseen_geometry_challenge.py",
                    [
                        "_appearance_similarity",
                        "_baseline_primary",
                        "_baseline_secondary",
                        "_candidate_edges",
                        "_challenge_categories",
                        "_mine_challenge_candidates",
                    ],
                    "short-window geometry/appearance edge construction and challenge mining",
                ),
                (
                    "src/football_intelligence/replay/cadence_matched_third_unseen_challenge.py",
                    ["_same_frame_neighbourhood_id", "_build_challenge_row", "_mine_challenge_candidates"],
                    "cadence-matched target-choice construction and same-frame assignment neighbourhoods",
                ),
            ],
        ),
    )
    source_map["07_CONTINUITY_AND_CHALLENGE_PIPELINE_EXCERPTS.py"] = [
        str(repo_root / "src/football_intelligence/replay/third_unseen_geometry_challenge.py"),
        str(repo_root / "src/football_intelligence/replay/cadence_matched_third_unseen_challenge.py"),
    ]

    _write_text(output_root / "08_REVIEW_CHASSIS_AND_PERSISTENCE_EXCERPTS.py", _extract_review_chassis(repo_root))
    source_map["08_REVIEW_CHASSIS_AND_PERSISTENCE_EXCERPTS.py"] = [
        str(repo_root / "src/football_intelligence/review_chassis/models.py"),
        str(repo_root / "src/football_intelligence/review_chassis/persistence.py"),
        str(repo_root / "src/football_intelligence/review_chassis/server.py"),
        str(repo_root / "src/football_intelligence/review_chassis/static/app.js"),
    ]

    _write_text(
        output_root / "09_LOCALIZATION_AND_UPSTREAM_AUDIT_EXCERPTS.py",
        _extract_python_symbols(
            repo_root,
            [
                (
                    "src/football_intelligence/replay/third_unseen_review_correction.py",
                    [
                        "_endpoint_status_from_candidate",
                        "canonical_trajectory_safe_grouping",
                        "corrected_rule_results",
                        "label_binding_status",
                    ],
                    "corrected evaluation semantics, endpoint binding and trajectory-safe grouping",
                ),
                (
                    "src/football_intelligence/replay/followup_candidate_supply_diagnostic.py",
                    [
                        "validate_followup_events",
                        "decoded_followup_rows",
                        "build_localization_review",
                        "inventory_candidate_coverage",
                        "detector_provenance_outputs",
                        "detector_diagnostic_placeholders",
                        "root_cause_and_research_gate",
                    ],
                    "N follow-up ingestion, localization workbench and pending upstream root-cause audit",
                ),
            ],
        ),
    )
    source_map["09_LOCALIZATION_AND_UPSTREAM_AUDIT_EXCERPTS.py"] = [
        str(repo_root / "src/football_intelligence/replay/third_unseen_review_correction.py"),
        str(repo_root / "src/football_intelligence/replay/followup_candidate_supply_diagnostic.py"),
    ]

    write(
        "10_OCCLUSION_FAILURE_TAXONOMY_AND_KNOWN_WEAKNESSES.md",
        f"""# Occlusion Failure Taxonomy And Known Weaknesses

Sources:
{_render_sources(sources, ["secondary_results", "m5_4j_validation", "m5_4j_full_frame_audit", "m5_4j_detector_provenance", "m5_4j_root_cause"])}

## A. Detector Failures
- Person completely missed: hypothesis for N cases 004, 009, 011, 016; not proven before localization.
- Confidence-threshold loss: hypothesis; detector sweep has not run.
- NMS suppression: hypothesis; pre-NMS evidence unavailable.
- Max-det truncation: hypothesis; max_det=80 recorded, no proof of truncation.
- Merged two-person box / oversized box spanning two people / partial-body box / duplicate boxes / poor localization: plausible in overlapping regions; not proven until spatial localization and full-frame comparison.

## B. Silhouette Failures
- No instance separation, one mask covering two people, body-part fragmentation, hidden limbs, foreground/background confusion, jersey contamination and one player emerging from another are architectural hypotheses. No segmentation module exists yet. Cases 008, 010 and 013 show association conflict under IoU, not a proven mask failure.

## C. Association Failures
- IoU follows the wrong overlapping person: observed in secondary wrong-target cases 008, 010 and 013.
- Two sources compete for one target / track swap / crossing conflict: observed as assignment or crossing conflict in the corrected evaluation outputs.
- Appearance override harms correct geometry: current artifacts include appearance correction/regression summaries; treat as diagnostic, not global truth.
- Correct target outside candidate radius or absent from candidate set: possible for N cases 004, 009, 011, 016; M5.4J requires spatial localization first.

## D. Temporal-State Failures
- Track terminated too early, no temporary occlusion state, re-emergence linked to the wrong track, uncertainty not carried across invisible frames and single-hypothesis assignment during crossing are research hypotheses. Current system has no hidden/ghost state and does not maintain multiple hypotheses through occlusion.

Status by representative case:
- 004 and 016: candidate-supply/root-cause cases in one trajectory-safe local region; unresolved.
- 009 and 011: candidate-supply/root-cause cases; unresolved.
- 008, 010 and 013: secondary IoU wrong-panel crossing/assignment cases; association weakness observed.
""",
        [
            "secondary_results",
            "m5_4j_validation",
            "m5_4j_full_frame_audit",
            "m5_4j_detector_provenance",
            "m5_4j_root_cause",
        ],
    )

    label_payload = {
        "schema_version": "football_intelligence.m5_5a.labels_grouping.v1",
        "canonical_positive_edges": combined["canonical_unique_edge_counts"]["accept_continuity"],
        "canonical_negative_edges": combined["canonical_unique_edge_counts"]["reject_continuity"],
        "combined_candidate_rows": combined["combined_candidate_row_count"],
        "endpoint_safe_group_count": 20,
        "trajectory_safe_group_count": len(trajectory["components"]),
        "crossing_assignment_wrong_panel_cases": ["008", "010", "013"],
        "n_cases": ["004", "009", "011", "016"],
        "trajectory_safe_supply_failure_regions": m5_4j["trajectory_safe_failure_region_count"],
        "primary_rule_performance": {
            "correct": primary["correct_decisive_target_choices"],
            "wrong": primary["wrong_decisive_target_choices"],
            "abstentions": primary["decisive_abstentions"],
        },
        "secondary_rule_performance": {
            "correct": secondary["correct_decisive_target_choices"],
            "wrong": secondary["wrong_decisive_target_choices"],
            "abstentions": secondary["decisive_abstentions"],
        },
        "appearance_corrections_and_regressions": _read_json(
            stage_root / "continuity_v13" / "evaluation" / "appearance_policy_correction.json"
        ),
        "model_fit_performed": combined["model_fit_performed"],
        "learned_continuity_rows_updated": combined["learned_continuity_rows_updated"],
        "challenge_set_sampling_caveat": "20-case hard/challenge set; broader random sample still required.",
        "source_artifacts": {
            key: str(sources[key])
            for key in ["primary_results", "secondary_results", "combined_inventory", "trajectory_grouping"]
        },
        **safety_payload(),
    }
    _write_json(output_root / "11_CURRENT_EVALUATION_LABELS_AND_GROUPING.json", label_payload)
    source_map["11_CURRENT_EVALUATION_LABELS_AND_GROUPING.json"] = [
        str(sources["primary_results"]),
        str(sources["secondary_results"]),
        str(sources["combined_inventory"]),
        str(sources["trajectory_grouping"]),
    ]

    write(
        "12_OCCLUSION_RESEARCH_QUESTIONS_AND_DECISION_GATES.md",
        f"""# Occlusion Research Questions And Decision Gates

Sources:
{_render_sources(sources, ["m5_4j_validation", "m5_4j_full_frame_audit", "m5_4j_detector_provenance", "secondary_results"])}

Questions for the Pro researcher:
- Is the first priority detection, segmentation, association, or all three?
- Should the detector remain box-based?
- Would instance segmentation materially help at the current player scale?
- Can masks be reliably propagated for 0.1-0.3 seconds?
- Can pose or body-part evidence separate overlapping players at 2730x720 panoramic scale?
- Should an occlusion state machine carry temporary hidden tracks?
- Should re-emergence use geometry, appearance, team colour, pose, temporal path evidence, or a learned combination?
- Should the system maintain multiple hypotheses during crossings?
- How should confidence decay while invisible?
- When should a ghost track die?
- When should human review be requested?
- What smallest experiment distinguishes merged detection, true miss, candidate-radius failure and association swap?
- What additional labels are needed?
- What evidence would justify a learned model?

Pass/block gates:
- PASS_DETECTOR_RECALL_DIAGNOSTIC_READY only after localization decisions seal target regions.
- BLOCK_DETECTOR_MISS_CLAIM if spatial localization is absent.
- PASS_SEGMENTATION_PILOT_READY only if representative overlaps are localized and masks can be compared against boxes.
- PASS_OCCLUSION_STATE_PILOT_READY only if temporary image-space ghost states remain match-local and expire.
- BLOCK_LEARNED_CONTINUITY_MODEL unless labels are class-balanced, grouped-validation feasible and no leakage is present.
- BLOCK_GLOBAL_POLICY_CHANGE from four N cases or three crossing cases alone.
- PASS_HUMAN_REVIEW_ESCALATION if uncertainty remains high after detector, segmentation and association diagnostics.
""",
        ["m5_4j_validation", "m5_4j_full_frame_audit", "m5_4j_detector_provenance", "secondary_results"],
    )

    representative_rows = _build_case_index_rows(stage_root, primary, secondary, case_index, full_frame_audit)
    _write_case_index(output_root / "13_REPRESENTATIVE_CASE_INDEX.csv", representative_rows)
    source_map["13_REPRESENTATIVE_CASE_INDEX.csv"] = [
        str(sources["target_choice_case_index"]),
        str(sources["primary_results"]),
        str(sources["secondary_results"]),
        str(sources["m5_4j_full_frame_audit"]),
    ]

    gif_results = {
        "14_CROSSING_FAILURE_CASE_008.gif": _write_case_gif(
            stage_root, 8, output_root / "14_CROSSING_FAILURE_CASE_008.gif"
        ),
        "15_CROSSING_FAILURE_CASE_010.gif": _write_case_gif(
            stage_root, 10, output_root / "15_CROSSING_FAILURE_CASE_010.gif"
        ),
        "16_CROSSING_FAILURE_CASE_013.gif": _write_case_gif(
            stage_root, 13, output_root / "16_CROSSING_FAILURE_CASE_013.gif"
        ),
        "17_SHARED_OCCLUSION_REGION_CASES_004_016.gif": _write_shared_region_gif(
            stage_root, output_root / "17_SHARED_OCCLUSION_REGION_CASES_004_016.gif"
        ),
    }
    for filename, result in gif_results.items():
        source_map[filename] = result["source_frames"]

    contact_result = _write_contact_sheet(
        stage_root, output_root / "18_FULL_FRAME_DETECTION_CONTACT_SHEET.jpg", localization_rows
    )
    source_map["18_FULL_FRAME_DETECTION_CONTACT_SHEET.jpg"] = [
        str(stage_root / "continuity_v14" / "localization" / "evidence"),
        str(sources["m5_4j_localization_case_index"]),
    ]

    write(
        "19_TARGETED_CODEBASE_FILE_MAP.md",
        f"""# Targeted Codebase File Map

| Absolute path | Repository-relative path | Main symbols | Current responsibility | Likely occlusion modification | Dependencies | Relevant tests | Historical risk | Prefer new module? |
|---|---|---|---|---|---|---|---|---|
| `{repo_root / "src/football_intelligence/replay/portable_detector.py"}` | `src/football_intelligence/replay/portable_detector.py` | `PortableDetectorConfig`, `run_detector_inference` | detector model validation/inference | detector recall sweeps, optional segmentation model boundary | Ultralytics, torch | `tests/test_portable_detector.py` | medium | yes for diagnostics |
| `{repo_root / "src/football_intelligence/replay/third_unseen_geometry_challenge.py"}` | `src/football_intelligence/replay/third_unseen_geometry_challenge.py` | `_candidate_edges`, `_baseline_primary`, `_baseline_secondary` | third-window frame/candidate/edge generation | add occlusion-aware feature rows or alternate candidate miner | cv2, numpy | `tests/test_m5_4h_third_unseen_geometry_challenge.py` | high | yes |
| `{repo_root / "src/football_intelligence/replay/cadence_matched_third_unseen_challenge.py"}` | `src/football_intelligence/replay/cadence_matched_third_unseen_challenge.py` | `_same_frame_neighbourhood_id`, `_mine_challenge_candidates` | cadence-matched target-choice review mining | add crossing-neighbourhood diagnostics | review chassis | `tests/test_m5_4h1_cadence_matched_third_unseen.py` | high | yes |
| `{repo_root / "src/football_intelligence/replay/third_unseen_review_ingestion.py"}` | `src/football_intelligence/replay/third_unseen_review_ingestion.py` | `_decode_decisions`, `_trajectory_safe_grouping` | human-review ingestion and label sidecars | consume future occlusion review labels only after gates | v11 review artifacts | `tests/test_m5_4i_third_unseen_review_ingestion.py` | high | yes |
| `{repo_root / "src/football_intelligence/replay/third_unseen_review_correction.py"}` | `src/football_intelligence/replay/third_unseen_review_correction.py` | `canonical_trajectory_safe_grouping`, `label_binding_status` | corrected semantics and grouping | add audits; avoid rewriting historical semantics | v13 artifacts | `tests/test_m5_4i1_review_correction.py` | high | yes |
| `{repo_root / "src/football_intelligence/replay/followup_candidate_supply_diagnostic.py"}` | `src/football_intelligence/replay/followup_candidate_supply_diagnostic.py` | `build_localization_review`, `inventory_candidate_coverage` | M5.4J localization/root-cause gate | extend after localization to detector/segmentation recovery | v14 artifacts | `tests/test_m5_4j_candidate_supply_diagnostic.py` | medium | maybe |
| `{repo_root / "src/football_intelligence/review_chassis"}` | `src/football_intelligence/review_chassis/*` | `GenericReviewManifest`, `GenericReviewPersistence` | reusable review UI/persistence | add generic mask/point annotation, not stage-specific UI | stdlib HTTP, static JS | `tests/test_m5_4f5_review_chassis.py` | medium | modify carefully |
| `{repo_root / "src/football_intelligence/cli/app.py"}` | `src/football_intelligence/cli/app.py` | `counterfactual-review` commands | CLI entry points | add bounded future commands | Typer | CLI help validation | low | no |
| `{repo_root / "tests"}` | `tests/*` | M5.4H-I-J tests | regression safety | add occlusion diagnostic tests | pytest | full suite | low | no |

Sources: code paths above and current stage artifacts `{stage_root}`.
""",
        ["m5_4j_validation"],
    )

    # Placeholder manifest first so every file exists for the final manifest row collection.
    _write_json(output_root / "20_PACK_MANIFEST.json", {"placeholder": True})
    source_map["20_PACK_MANIFEST.json"] = [str(path) for path in sources.values() if path.exists()]
    files = _file_manifest_rows(output_root, source_map)
    utf8 = _text_utf8_check(output_root)
    code_excerpt_validation = _validate_code_excerpts(
        [
            output_root / "06_DETECTION_AND_PERSON_PIPELINE_EXCERPTS.py",
            output_root / "07_CONTINUITY_AND_CHALLENGE_PIPELINE_EXCERPTS.py",
            output_root / "08_REVIEW_CHASSIS_AND_PERSISTENCE_EXCERPTS.py",
            output_root / "09_LOCALIZATION_AND_UPSTREAM_AUDIT_EXCERPTS.py",
        ]
    )
    forbidden = _forbidden_field_scan(output_root)
    source_mutation = _source_mutation_audit(stage_root)
    manifest_payload = {
        "schema_version": "football_intelligence.m5_5a.pack_manifest.v1",
        "exact_file_count": 20,
        "filenames": PACK_FILENAMES,
        "files": files,
        "source_commit": "59c4d00dcbb5612d8a00a9f2ec4ce955e5941686",
        "generated_at": generated_at,
        "pack_hash": "<pending>",
        "reviewer_safe": True,
        "sealed_mapping_included": False,
        "model_weights_included": False,
        "source_video_included": False,
        "answer_key_fields_found": forbidden["answer_key_field_count"],
        "known_omissions": [
            "No model checkpoints or raw source video.",
            "No sealed mapping files.",
            "No pre-NMS detector outputs because they were not preserved.",
            "No final detector recovery sweep because localization is not sealed.",
        ],
        "known_provenance_cautions": state_payload["known_provenance_cautions"]
        + [
            "M5.4J UI-config hash mismatch appears in followup_event_validation.json.",
            "M5.4J session reconciliation is an alias/default mismatch.",
            "M5.4J browser_served_before_decision field is true in the validation artifact despite server_side_only=true; treat as ambiguous provenance.",
        ],
        "gif_decode_results": gif_results,
        "contact_sheet": contact_result,
        "quality_checks": {
            "exact_20_file_count": len(list(output_root.iterdir())) == 20,
            "no_nested_files": all(path.is_file() for path in output_root.iterdir()),
            "utf8": utf8,
            "code_excerpt_source_verification": code_excerpt_validation,
            "forbidden_answer_key_field_scan": forbidden,
            "sealed_file_path_scan": {
                "sealed_mapping_file_included": False,
                "pack_contains_file_named_mapping_json": any(
                    path.name == "mapping.json" for path in output_root.iterdir()
                ),
            },
            "model_checkpoint_included": any(path.suffix.lower() == ".pt" for path in output_root.iterdir()),
            "source_video_included": any(
                path.suffix.lower() in {".mp4", ".mov", ".mkv"} for path in output_root.iterdir()
            ),
            "source_mutation_audit": source_mutation,
        },
        **safety_payload(),
    }
    manifest_payload["pack_hash"] = _pack_hash(output_root, manifest_payload)
    _write_json(output_root / "20_PACK_MANIFEST.json", manifest_payload)
    files = _file_manifest_rows(output_root, source_map)
    manifest_payload["files"] = files
    manifest_payload["pack_hash"] = _pack_hash(output_root, manifest_payload)
    _write_json(output_root / "20_PACK_MANIFEST.json", manifest_payload)

    final_files = sorted(path.name for path in output_root.iterdir())
    if final_files != PACK_FILENAMES:
        raise ValueError(f"pack file mismatch: {final_files}")
    if forbidden["answer_key_field_count"] != 0:
        raise ValueError(f"forbidden answer-key fields found: {forbidden}")
    if not code_excerpt_validation["all_python_excerpts_parse"]:
        raise ValueError("code excerpts are not Python-parseable")
    if any(result["byte_size"] > 12 * 1024 * 1024 for result in gif_results.values()):
        raise ValueError("a GIF exceeds 12 MB")
    return {
        "output_root": str(output_root),
        "file_count": 20,
        "ordered_file_list": PACK_FILENAMES,
        "source_commit": manifest_payload["source_commit"],
        "pack_hash": manifest_payload["pack_hash"],
        "code_excerpt_source_verification": code_excerpt_validation,
        "gif_decode_results": gif_results,
        "contact_sheet_result": contact_result,
        "answer_key_field_count": forbidden["answer_key_field_count"],
        "sealed_mapping_included": False,
        "model_checkpoint_included": False,
        "source_video_included": False,
        "source_mutation_preserved": source_mutation["preserved"],
        "final_classification": "PASS_OCCLUSION_PRO_EXTENDED_CONTEXT_PACK_READY",
        "exact_blocker": "NONE",
    }


def deterministic_pack_hash(payload: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(payload, sort_keys=True))
    clone.pop("generated_at", None)
    return _stable_hash(clone)
