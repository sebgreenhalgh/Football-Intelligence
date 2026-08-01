"""Build the bounded G7E-B temporal reviewer and deterministic review assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from football_intelligence.temporal_review import (
    PROTOCOL_ID,
    REVIEW_ID,
    REVIEW_REVISION,
    TRANCHES,
    canonical_bytes,
    deterministic_tranche_assignment,
    sha256_file,
    validate_tranche_assignment,
)

EXPECTED_HEAD = "4f6e3a9a4e7402411b644e088ee440daf937c70c"
BURST_SHA256 = "619b6847fdde14ae13ec8a2618ac90c7ac9fc7f4d7445336bfde529e5746909d"
FRAME_SHA256 = "96688c685cc495a05af4c70003ea02b3e3f5b2dd66cc2c4813e581b10c42723d"
PANORAMA_LONG_EDGE = 2560
PANORAMA_JPEG_QUALITY = 92
FOCUS_JPEG_QUALITY = 95
STAGE_NAME = "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def git_head(repository: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True, encoding="utf-8").strip()


def rectangle_for_burst(burst: dict[str, Any]) -> list[int]:
    width = int(burst["source_width"])
    height = int(burst["source_height"])
    boxes = [candidate["source_box_xyxy"] for candidate in burst.get("focus_candidates", [])]
    if boxes:
        left = min(float(box[0]) for box in boxes)
        top = min(float(box[1]) for box in boxes)
        right = max(float(box[2]) for box in boxes)
        bottom = max(float(box[3]) for box in boxes)
        box_width = max(1.0, right - left)
        box_height = max(1.0, bottom - top)
        crop_width = min(width, max(512.0, box_width * 6.0, box_height * 4.0))
        crop_height = min(height, max(384.0, box_height * 4.5, box_width * 1.8))
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
    else:
        crop_width = min(width, max(960.0, width * 0.42))
        crop_height = min(height, max(540.0, height * 0.68))
        center_x = width / 2.0
        center_y = height * 0.52
    x1 = max(0, min(width - int(round(crop_width)), int(round(center_x - crop_width / 2))))
    y1 = max(0, min(height - int(round(crop_height)), int(round(center_y - crop_height / 2))))
    x2 = min(width, x1 + int(round(crop_width)))
    y2 = min(height, y1 + int(round(crop_height)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid focus crop for {burst['burst_id']}")
    for box in boxes:
        if not (x1 <= box[0] <= box[2] <= x2 and y1 <= box[1] <= box[3] <= y2):
            raise ValueError(f"focus crop does not contain candidate for {burst['burst_id']}")
    return [x1, y1, x2, y2]


def encode_jpeg(path: Path, image: np.ndarray, quality: int) -> dict[str, Any]:
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_OPTIMIZE, 0],
    )
    if not ok:
        raise ValueError(f"JPEG encoding failed: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None or decoded.size == 0 or float(decoded.std()) < 2.0:
        raise ValueError(f"blank derivative: {path}")
    return {
        "path": path,
        "width": int(decoded.shape[1]),
        "height": int(decoded.shape[0]),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
        "pixel_standard_deviation": round(float(decoded.std()), 6),
    }


def panorama_derivative(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = PANORAMA_LONG_EDGE / max(width, height)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, target, interpolation=cv2.INTER_AREA)


def safe_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "source_box_xyxy": [float(value) for value in candidate["source_box_xyxy"]],
        "footpoint_xy": [float(value) for value in candidate["footpoint_xy"]],
    }


def build_assets(
    project_root: Path,
    package: Path,
    asset_manifest_path: Path,
    bursts: list[dict[str, Any]],
    frames: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    burst_by_id = {row["burst_id"]: row for row in bursts}
    crop_by_burst = {burst_id: rectangle_for_burst(row) for burst_id, row in burst_by_id.items()}
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in frames:
        grouped[row["source_video_relative_path"]][int(row["frame_index_zero_based"])].append(row)

    asset_rows: list[dict[str, Any]] = []
    review_frames: dict[str, dict[str, Any]] = {}
    unique_source_hashes: set[str] = set()
    source_frame_failures: list[str] = []
    panorama_cache: dict[str, dict[str, Any]] = {}
    decoded_count = 0

    for video_relative in sorted(grouped):
        video_path = project_root / Path(video_relative)
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        expected_sizes = {
            int(row["source_video_byte_size"]) for rows in grouped[video_relative].values() for row in rows
        }
        if expected_sizes != {video_path.stat().st_size}:
            raise ValueError(f"source video size mismatch: {video_relative}")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"cannot open source video: {video_relative}")
        try:
            for frame_index in sorted(grouped[video_relative]):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, image = capture.read()
                if not ok or image is None:
                    raise ValueError(f"decode failed: {video_relative}@{frame_index}")
                decoded_count += 1
                references = grouped[video_relative][frame_index]
                expected_width = int(references[0]["source_width"])
                expected_height = int(references[0]["source_height"])
                if image.shape[1] != expected_width or image.shape[0] != expected_height:
                    raise ValueError(f"decoded dimensions mismatch: {video_relative}@{frame_index}")
                rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                source_hash = hashlib.sha256(rgb.tobytes(order="C")).hexdigest()
                expected_hashes = {str(row["frame_pixel_sha256"]) for row in references}
                if expected_hashes != {source_hash}:
                    source_frame_failures.append(f"{video_relative}@{frame_index}")
                    raise ValueError(f"source frame pixel hash mismatch: {video_relative}@{frame_index}")
                unique_source_hashes.add(source_hash)

                panorama = panorama_cache.get(source_hash)
                if panorama is None:
                    panorama_relative = Path("panorama") / source_hash[:2] / f"{source_hash}.jpg"
                    panorama = encode_jpeg(
                        package / "assets" / panorama_relative,
                        panorama_derivative(image),
                        PANORAMA_JPEG_QUALITY,
                    )
                    panorama["relative"] = panorama_relative.as_posix()
                    panorama_cache[source_hash] = panorama

                for reference in sorted(references, key=lambda row: row["frame_reference_id"]):
                    burst = burst_by_id[reference["burst_id"]]
                    x1, y1, x2, y2 = crop_by_burst[reference["burst_id"]]
                    crop = image[y1:y2, x1:x2]
                    focus_relative = Path("focus") / reference["burst_id"] / f"{reference['frame_reference_id']}.jpg"
                    focus = encode_jpeg(
                        package / "assets" / focus_relative,
                        crop,
                        FOCUS_JPEG_QUALITY,
                    )
                    row = {
                        "schema_version": "football_intelligence.g7e_b.review_asset.v1",
                        "burst_id": reference["burst_id"],
                        "frame_reference_id": reference["frame_reference_id"],
                        "burst_frame_sequence": int(reference["burst_frame_sequence"]),
                        "source_video_relative_path": video_relative,
                        "source_frame_index_zero_based": frame_index,
                        "source_timestamp_seconds": float(reference["resolved_timestamp_seconds"]),
                        "source_width": expected_width,
                        "source_height": expected_height,
                        "source_frame_pixel_sha256": source_hash,
                        "panorama": {
                            "derivative_path": f"03_TEMPORAL_REVIEWER/assets/{panorama['relative']}",
                            "browser_url": f"/assets/{panorama['relative']}",
                            "width": panorama["width"],
                            "height": panorama["height"],
                            "byte_size": panorama["byte_size"],
                            "sha256": panorama["sha256"],
                            "jpeg_quality": PANORAMA_JPEG_QUALITY,
                            "resize_contract": "LONG_EDGE_2560_PRESERVE_ASPECT_NO_LETTERBOX",
                        },
                        "focus": {
                            "derivative_path": f"03_TEMPORAL_REVIEWER/assets/{focus_relative.as_posix()}",
                            "browser_url": f"/assets/{focus_relative.as_posix()}",
                            "width": focus["width"],
                            "height": focus["height"],
                            "byte_size": focus["byte_size"],
                            "sha256": focus["sha256"],
                            "jpeg_quality": FOCUS_JPEG_QUALITY,
                            "crop_source_xyxy": [x1, y1, x2, y2],
                            "resize_contract": "NATIVE_SOURCE_PIXELS_NO_RESIZE",
                            "fallback_used": not bool(burst.get("focus_candidates")),
                            "fallback_contract": (
                                "NONE" if burst.get("focus_candidates") else "DETERMINISTIC_CENTRAL_EVIDENCE_REGION"
                            ),
                        },
                        "production_ready": False,
                    }
                    asset_rows.append(row)
                    review_frames[reference["frame_reference_id"]] = row
        finally:
            capture.release()

    asset_rows.sort(key=lambda row: (row["burst_id"], row["burst_frame_sequence"]))
    write_jsonl(asset_manifest_path, asset_rows)
    report = {
        "schema_version": "football_intelligence.g7e_b.review_asset_generation_report.v1",
        "classification": "PASS_G7E_B_REVIEW_ASSETS",
        "frame_references": len(asset_rows),
        "unique_source_frames": len(unique_source_hashes),
        "source_videos": len(grouped),
        "decoded_source_frames": decoded_count,
        "unique_panorama_derivatives": len(panorama_cache),
        "focus_derivatives": len(asset_rows),
        "source_frame_hash_failures": source_frame_failures,
        "all_assets_non_blank": all(
            row["panorama"]["byte_size"] > 0 and row["focus"]["byte_size"] > 0 for row in asset_rows
        ),
        "panorama_contract": {
            "format": "JPEG",
            "quality": PANORAMA_JPEG_QUALITY,
            "long_edge_pixels": PANORAMA_LONG_EDGE,
        },
        "focus_contract": {
            "format": "JPEG",
            "quality": FOCUS_JPEG_QUALITY,
            "native_source_pixels": True,
        },
        "full_resolution_frames_retained": 0,
        "review_asset_manifest_sha256": sha256_file(asset_manifest_path),
        "production_ready": False,
    }
    return asset_rows, review_frames, report


def make_review_cases(
    assigned: list[dict[str, Any]],
    review_frames: dict[str, dict[str, Any]],
    g7e_a_root: Path,
) -> dict[str, Any]:
    burst_manifest = g7e_a_root / "02_BURST_SELECTION/temporal_burst_manifest.jsonl"
    frame_manifest = g7e_a_root / "02_BURST_SELECTION/temporal_frame_manifest.jsonl"
    cases: list[dict[str, Any]] = []
    for burst in assigned:
        crop = rectangle_for_burst(burst)
        frames = []
        for frame_reference_id in burst["frame_reference_ids"]:
            asset = review_frames[frame_reference_id]
            frames.append(
                {
                    "frame_reference_id": frame_reference_id,
                    "burst_frame_sequence": asset["burst_frame_sequence"],
                    "relative_offset_seconds": float(burst["relative_offsets_seconds"][asset["burst_frame_sequence"]]),
                    "resolved_timestamp_seconds": asset["source_timestamp_seconds"],
                    "source_width": asset["source_width"],
                    "source_height": asset["source_height"],
                    "source_frame_pixel_sha256": asset["source_frame_pixel_sha256"],
                    "panorama_url": asset["panorama"]["browser_url"],
                    "panorama_sha256": asset["panorama"]["sha256"],
                    "panorama_width": asset["panorama"]["width"],
                    "panorama_height": asset["panorama"]["height"],
                    "focus_url": asset["focus"]["browser_url"],
                    "focus_sha256": asset["focus"]["sha256"],
                    "focus_width": asset["focus"]["width"],
                    "focus_height": asset["focus"]["height"],
                }
            )
        cases.append(
            {
                "schema_version": "football_intelligence.g7e_b.blind_temporal_review_case.v1",
                "review_id": REVIEW_ID,
                "review_revision": REVIEW_REVISION,
                "burst_id": burst["burst_id"],
                "tranche_id": burst["tranche_id"],
                "tranche_position": burst["tranche_position"],
                "match_id": burst["match_id"],
                "half": burst["half"],
                "source_width": burst["source_width"],
                "source_height": burst["source_height"],
                "focus_crop_source_xyxy": crop,
                "candidates": [safe_candidate(candidate) for candidate in burst.get("focus_candidates", [])],
                "frames": frames,
                "burst_manifest_path": str(burst_manifest.resolve()),
                "source_manifest_hashes": {
                    "temporal_burst_manifest_sha256": sha256_file(burst_manifest),
                    "temporal_frame_manifest_sha256": sha256_file(frame_manifest),
                },
                "blind_first": True,
                "candidate_ids_hidden_by_default": True,
                "team_classification_present": False,
                "permanent_identity_present": False,
                "production_ready": False,
            }
        )
    return {
        "schema_version": "football_intelligence.g7e_b.blind_temporal_review_cases.v1",
        "review_id": REVIEW_ID,
        "review_revision": REVIEW_REVISION,
        "case_count": len(cases),
        "cases": cases,
        "protected_selection_truth_present": False,
        "human_answers_present": False,
        "production_ready": False,
    }


def practice_cases(review_cases: dict[str, Any], assigned: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {case["burst_id"]: case for case in review_cases["cases"]}
    wanted = (
        "OCCLUSION_OR_MERGE_RISK",
        "FRAGMENT_OR_DUPLICATE_RISK",
        "STABLE_OPEN_PLAY_CONTROL",
    )
    chosen: list[dict[str, Any]] = []
    used_matches: set[str] = set()
    for position, selection_class in enumerate(wanted, start=1):
        options = sorted(
            (row for row in assigned if row["primary_selection_class"] == selection_class),
            key=lambda row: (
                str(row["match_id"]) in used_matches,
                hashlib.sha256(f"PRACTICE|{selection_class}|{row['burst_id']}".encode()).hexdigest(),
            ),
        )
        source = options[0]
        used_matches.add(str(source["match_id"]))
        case = json.loads(json.dumps(by_id[source["burst_id"]]))
        case["tranche_id"] = None
        case.pop("tranche_position", None)
        case["practice_position"] = position
        case["practice_only"] = True
        chosen.append(case)
    return {
        "schema_version": "football_intelligence.g7e_b.practice_cases.v1",
        "case_count": 3,
        "separate_non_truth_mode": True,
        "cases": chosen,
        "production_ready": False,
    }


def balance_report(assigned: list[dict[str, Any]]) -> dict[str, Any]:
    tranches: dict[str, Any] = {}
    for tranche_id in TRANCHES:
        rows = [row for row in assigned if row["tranche_id"] == tranche_id]
        tags = Counter(tag for row in rows for tag in row.get("secondary_evidence_tags", []))
        tranches[tranche_id] = {
            "burst_count": len(rows),
            "class_counts": dict(sorted(Counter(row["primary_selection_class"] for row in rows).items())),
            "match_counts": dict(sorted(Counter(str(row["match_id"]) for row in rows).items())),
            "half_counts": dict(sorted(Counter(row["half"] for row in rows).items())),
            "perspective_counts": dict(sorted(Counter(row["perspective_band"] for row in rows).items())),
            "low_light_bursts": sum(1 for row in rows if str(row["match_id"]) == "117092"),
            "calibration_seeds": {
                "nested_must_protect": tags["NESTED_MUST_PROTECT"],
                "nested_safe_fragment": tags["HUMAN_SAFE_FRAGMENT"],
                "missed_person": tags["MISSED_PERSON_MARK"],
                "goalmouth_or_endline": sum(
                    1 for row in rows if row["primary_selection_class"] == "GOALMOUTH_OR_ENDLINE_CROWD"
                ),
                "official_or_boundary": sum(
                    1 for row in rows if row["primary_selection_class"] == "OFFICIAL_OR_BOUNDARY_CONTINUITY"
                ),
                "stable_controls": sum(
                    1 for row in rows if row["primary_selection_class"] == "STABLE_OPEN_PLAY_CONTROL"
                ),
            },
            "requirements_pass": True,
        }
    validation = validate_tranche_assignment(assigned)
    return {
        "schema_version": "football_intelligence.g7e_b.tranche_balance_report.v1",
        "classification": "PASS_G7E_B_TRANCHE_BALANCE" if validation["valid"] else "FAIL_G7E_B_TRANCHE_BALANCE",
        "assignment_algorithm": "DETERMINISTIC_MATCHING_DECOMPOSITION_AND_SHA256_RANK_SEARCH",
        "assignment_attempt": assigned[0]["assignment_attempt"],
        "validation": validation,
        "tranches": tranches,
        "production_ready": False,
    }


def branch_contract() -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.g7e_b.reviewer_branch_contract.v1",
        "review_revision": REVIEW_REVISION,
        "one_question_at_a_time": True,
        "subject_tokens": ["SUBJECT_A", "SUBJECT_B", "SUBJECT_C"],
        "subject_scope": "BURST_LOCAL_ONLY_RESET_EVERY_BURST",
        "questions": [
            "focus_confirmation",
            "subject_anchor",
            "visibility_timeline",
            "candidate_supply_timeline",
            "conditional_candidate_relationship",
            "conditional_confirmed_occlusion_sequence",
            "conditional_burst_local_continuity",
            "role",
            "participation",
            "certainty",
            "conditional_additional_subject",
            "whole_burst_missed_person_check",
            "conditional_source_coordinate_marking",
            "summary_and_acknowledged_save",
        ],
        "focus_no_relevant_branch": [
            "focus_confirmation",
            "whole_burst_missed_person_check",
            "summary_and_acknowledged_save",
        ],
        "focus_not_sure_branch": [
            "focus_confirmation",
            "whole_burst_missed_person_check",
            "summary_and_acknowledged_save",
        ],
        "occlusion_helper_requires_explicit_confirmation": True,
        "candidate_relationship_condition": "MULTIPLE_OR_MERGED_OR_FRAGMENT_SUPPLY",
        "continuity_condition": "DISAPPEAR_REAPPEAR_OR_ASSOCIATION_CHANGE_OR_MULTIPLE_OR_UNCERTAIN",
        "team_classification": "INTENTIONALLY_EXCLUDED",
        "permanent_identity": "FORBIDDEN",
        "selection_class_visible_to_browser": False,
        "server_backed_draft_after_every_valid_answer": True,
        "immutable_event_then_acknowledgement": True,
        "production_ready": False,
    }


def event_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "football_intelligence.g7e_b.burst_annotation_event.v1",
        "title": "G7E-B immutable burst-local temporal annotation event",
        "type": "object",
        "required": [
            "event_id",
            "review_revision",
            "tranche_id",
            "burst_id",
            "burst_manifest_path",
            "burst_manifest_sha256",
            "source_frame_hashes",
            "focus_answer",
            "subjects",
            "candidate_mappings",
            "whole_burst_missed_person_answer",
            "whole_burst_missed_person_marks",
            "summary_confirmed",
            "production_ready",
        ],
        "properties": {
            "review_revision": {"const": REVIEW_REVISION},
            "protocol_id": {"const": PROTOCOL_ID},
            "tranche_id": {"enum": list(TRANCHES)},
            "subjects": {"type": "array", "maxItems": 3},
            "source_frame_hashes": {"type": "array", "minItems": 9, "maxItems": 9},
            "production_ready": {"const": False},
        },
        "forbidden_concepts": ["team", "shirt_number", "track_id", "cross_burst_identity", "permanent_identity"],
    }


def instructions() -> str:
    return "\n".join(
        [
            "# G7E-B temporal burst review",
            "",
            "Launch `launch_temporal_burst_review.ps1`, then open http://127.0.0.1:8818/.",
            "",
            "Start with the three resettable practice bursts if useful. Practice is prominently labelled and never",
            "enters human truth. Real review starts with Tranche 1; later tranches remain locked until the current",
            "tranche has a valid server completion receipt and you explicitly unlock the next one.",
            "",
            "For each burst, play or step through all nine frames, answer one plain-English question at a time,",
            "and use Not sure whenever appropriate. Subject A/B/C labels apply only inside the current burst.",
            "Never infer identity across bursts. Team is intentionally not requested.",
            "",
            "Progress is saved to the server after every valid answer. Final truth appears only after",
            "`SAVED — SERVER ACKNOWLEDGED`. After Tranche 1 reaches 20/20, stop at the pause screen and provide",
            "the tranche completion receipt for validation.",
            "",
            "If an image, hash, mapping, draft, or persistence check fails, controls lock and the reviewer stops",
            "safely. Do not work around a blocking error.",
            "",
        ]
    )


def copy_reviewer_files(repository: Path, package: Path) -> None:
    source = repository / "src/football_intelligence"
    shutil.copyfile(source / "g7e_b_temporal_review.html", package / "index.html")
    shutil.copyfile(source / "g7e_b_temporal_review.css", package / "review.css")
    shutil.copyfile(source / "g7e_b_temporal_review.js", package / "review.js")
    wrapper = "\n".join(
        [
            '"""Launch the bounded G7E-B reviewer server."""',
            "import sys",
            f"sys.path.insert(0, {str((repository / 'src').resolve())!r})",
            "from football_intelligence.temporal_review import main",
            'if __name__ == "__main__":',
            "    main()",
            "",
        ]
    )
    (package / "review_server.py").write_text(wrapper, encoding="utf-8", newline="\n")


def launcher_text(repository: Path, stage: Path) -> str:
    python = repository / ".venv/Scripts/python.exe"
    package = stage / "03_TEMPORAL_REVIEWER"
    return "\n".join(
        [
            '$ErrorActionPreference = "Stop"',
            f'& "{python}" "{package / "review_server.py"}" `',
            f'  --package "{package}" `',
            f'  --decisions-root "{package / "human_decisions"}" `',
            f'  --practice-root "{package / "practice_decisions"}" `',
            "  --port 8818",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--stage-root", type=Path)
    parser.add_argument("--refresh-code-only", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    repository = project_root / "SoccerTrack-v2"
    stage = (
        args.stage_root or project_root / "experiments/football_observation_reasoner/part 7" / STAGE_NAME
    ).resolve()
    g7e_a = (
        project_root
        / "experiments/football_observation_reasoner/part 7"
        / "G7E_A_TARGETED_TEMPORAL_BURST_SELECTION_AND_ANNOTATION_DESIGN_v1"
    )
    burst_manifest = g7e_a / "02_BURST_SELECTION/temporal_burst_manifest.jsonl"
    frame_manifest = g7e_a / "02_BURST_SELECTION/temporal_frame_manifest.jsonl"

    if args.refresh_code_only:
        package = stage / "03_TEMPORAL_REVIEWER"
        if not (package / "review_cases.json").is_file():
            raise SystemExit("cannot refresh reviewer before the deterministic package exists")
        copy_reviewer_files(repository, package)
        (stage / "launch_temporal_burst_review.ps1").write_text(
            launcher_text(repository, stage), encoding="utf-8", newline="\n"
        )
        print("PASS_G7E_B_REVIEWER_CODE_REFRESH")
        return

    if git_head(repository) != EXPECTED_HEAD:
        raise SystemExit("FAIL_BASELINE_OR_WORKTREE: unexpected HEAD")
    if sha256_file(burst_manifest) != BURST_SHA256 or sha256_file(frame_manifest) != FRAME_SHA256:
        raise SystemExit("FAIL_G7E_B_INPUT_PROVENANCE: frozen manifest hash mismatch")
    decision = json.loads((g7e_a / "01_SELECTION_AND_ANNOTATION_CONTRACT/decision.json").read_text(encoding="utf-8"))
    if decision.get("decision") != "PASS_G7E_A_TEMPORAL_BURST_SELECTION_AND_ANNOTATION_DESIGN_FROZEN":
        raise SystemExit("FAIL_G7E_B_INPUT_PROVENANCE: G7E-A decision mismatch")

    bursts = load_jsonl(burst_manifest)
    frames = load_jsonl(frame_manifest)
    if len(bursts) != 120 or len(frames) != 1080 or len({row["frame_pixel_sha256"] for row in frames}) != 1044:
        raise SystemExit("FAIL_G7E_B_INPUT_PROVENANCE: cardinality mismatch")
    assigned = deterministic_tranche_assignment(bursts)
    validation = validate_tranche_assignment(assigned)
    if not validation["valid"]:
        raise SystemExit(f"FAIL_G7E_B_TRANCHE_BALANCE: {validation['errors']}")

    for folder in (
        "00_INPUT_CLOSURE",
        "01_TRANCHE_CONTRACT",
        "02_REVIEW_ASSET_PACKAGE",
        "03_TEMPORAL_REVIEWER",
        "04_BROWSER_ACCEPTANCE",
        "05_VISUAL_QA",
        "06_TESTS_AND_LOGS",
        "07_REVIEW_PACK",
    ):
        (stage / folder).mkdir(parents=True, exist_ok=True)
    package = stage / "03_TEMPORAL_REVIEWER"

    tranche_rows = [
        {
            "schema_version": "football_intelligence.g7e_b.tranche_burst_assignment.v1",
            **row,
            "assignment_algorithm": "DETERMINISTIC_MATCHING_DECOMPOSITION_AND_SHA256_RANK_SEARCH",
        }
        for row in assigned
    ]
    tranche_manifest = stage / "01_TRANCHE_CONTRACT/tranche_manifest.jsonl"
    write_jsonl(tranche_manifest, tranche_rows)
    shutil.copyfile(tranche_manifest, package / "tranche_manifest.jsonl")
    report = balance_report(assigned)
    report["tranche_manifest_sha256"] = sha256_file(tranche_manifest)
    write_json(stage / "01_TRANCHE_CONTRACT/tranche_balance_report.json", report)
    write_json(stage / "01_TRANCHE_CONTRACT/reviewer_branch_contract.json", branch_contract())
    write_json(stage / "01_TRANCHE_CONTRACT/reviewer_event_schema.json", event_schema())

    asset_manifest = stage / "02_REVIEW_ASSET_PACKAGE/review_asset_manifest.jsonl"
    asset_rows, review_frames, asset_report = build_assets(project_root, package, asset_manifest, bursts, frames)
    write_json(stage / "02_REVIEW_ASSET_PACKAGE/asset_generation_report.json", asset_report)

    cases = make_review_cases(assigned, review_frames, g7e_a)
    practices = practice_cases(cases, assigned)
    write_json(package / "review_cases.json", cases)
    write_json(package / "practice_cases.json", practices)
    write_json(package / "reviewer_branch_contract.json", branch_contract())
    write_json(package / "reviewer_event_schema.json", event_schema())
    (package / "HUMAN_REVIEW_INSTRUCTIONS.md").write_text(instructions(), encoding="utf-8", newline="\n")
    copy_reviewer_files(repository, package)
    (stage / "launch_temporal_burst_review.ps1").write_text(
        launcher_text(repository, stage), encoding="utf-8", newline="\n"
    )

    closure = {
        "schema_version": "football_intelligence.g7e_b.input_closure.v1",
        "classification": "PASS_G7E_B_INPUT_PROVENANCE",
        "repository_head": EXPECTED_HEAD,
        "g7e_a_decision": decision["decision"],
        "temporal_burst_manifest": {"path": str(burst_manifest), "sha256": BURST_SHA256, "rows": 120},
        "temporal_frame_manifest": {"path": str(frame_manifest), "sha256": FRAME_SHA256, "rows": 1080},
        "unique_source_frames": 1044,
        "source_videos": 12,
        "matches": sorted({str(row["match_id"]) for row in bursts}),
        "ontology_protocol_id": PROTOCOL_ID,
        "continuity_scope": "BURST_LOCAL_CONTINUITY_ONLY",
        "permanent_identity": "FORBIDDEN",
        "team_classification": "INTENTIONALLY_EXCLUDED",
        "selection_manifests_mutated": False,
        "validation_or_holdout_access": False,
        "inference_run": False,
        "production_ready": False,
    }
    write_json(stage / "00_INPUT_CLOSURE/input_closure.json", closure)
    build_report = {
        "schema_version": "football_intelligence.g7e_b.build_report.v1",
        "classification": "PASS_G7E_B_BUILD_READY_FOR_BROWSER_ACCEPTANCE",
        "review_id": REVIEW_ID,
        "review_revision": REVIEW_REVISION,
        "tranches": 6,
        "cases": 120,
        "practice_cases": 3,
        "asset_rows": len(asset_rows),
        "tranche_manifest_sha256": sha256_file(tranche_manifest),
        "review_asset_manifest_sha256": sha256_file(asset_manifest),
        "review_cases_sha256": sha256_file(package / "review_cases.json"),
        "practice_cases_sha256": sha256_file(package / "practice_cases.json"),
        "real_human_event_count": 0,
        "practice_event_count": 0,
        "inference_run": False,
        "production_ready": False,
    }
    write_json(stage / "06_TESTS_AND_LOGS/build_report.json", build_report)
    print(json.dumps(build_report, indent=2))


if __name__ == "__main__":
    main()
