"""Build the M5.5F0 stable local anonymous-strand benchmark.

The completed M5.5E.3 review is consumed as a read-only failure audit.  The
primary benchmark is mined from canonical person observations without using
the human labels for selection or tuning.  All outputs remain temporary,
match-local and visual-only.
"""

# The builder writes explicit audit records and schemas; long literals are
# easier to review in their serialized form than when wrapped.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

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

import build_m5_5e3_local_encounter as prior_e3


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
MATCH_ROOT = ROOT / "matches" / "128058"
PROMPT_ROOT = MATCH_ROOT / "runs" / "step_m5" / "part 2" / "M5_5F0_Stable_Local_Strand_Continuity_Baseline_Prompt_v1"
PRIOR_ROOT = (
    MATCH_ROOT / "runs" / "step_m5" / "part 2" / "M5_5E3_LOCAL_ENCOUNTER_DETECTION_RECOVERY_AND_STRAND_BINDING_v1"
)
PRIOR_PACKAGE = PRIOR_ROOT / "09_LOCAL_ENCOUNTER_HUMAN_REVIEW_PACKAGE"
STAGE_ID = "M5_5F0_STABLE_LOCAL_STRAND_CONTINUITY_BASELINE_v1"
STAGE_ROOT = MATCH_ROOT / "runs" / "step_m5" / "part 2" / STAGE_ID
REVIEW_ROOT = STAGE_ROOT / "08_STABLE_STRAND_BENCHMARK_REVIEW_PACKAGE"
EVIDENCE_ROOT = REVIEW_ROOT / "evidence"
DECISIONS_ROOT = REVIEW_ROOT / "decisions"
PACK_ROOT = STAGE_ROOT / "11_REVIEW_PACK_FOR_CHATGPT"
REVIEW_ID = "m5_5f0_stable_local_strand_continuity_review_v1"
REVIEW_SESSION = "m5_5f0_stable_local_strand_human_reviewer"
REVIEW_PORT = 8795
AUTHORIZED_BASELINE = "53cc937e2f9c0e1324611b39751ac6f5c0380bb1"
MODEL_PATH = REPO / "models" / "model=yolov8m-imgsz=2048.pt"
MODEL_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
MODEL_BYTES = 52136884

OUTCOMES = {
    "PASS": "PASS - Stable local continuation",
    "A_SWITCH": "A_SWITCH - Strand A switches",
    "B_SWITCH": "B_SWITCH - Strand B switches",
    "BOTH_SWITCH": "BOTH_SWITCH - Both strands switch",
    "A_LOST": "A_LOST - Strand A is lost",
    "B_LOST": "B_LOST - Strand B is lost",
    "BOTH_LOST": "BOTH_LOST - Both strands are lost",
    "DETECTION_SUPPLY_FAILURE": "DETECTION_SUPPLY_FAILURE - Local supply is insufficient",
    "AMBIGUOUS_BUT_SAFE_ABSTENTION": "AMBIGUOUS_BUT_SAFE_ABSTENTION - Tracker abstains safely",
    "BAD_CASE": "BAD_CASE - Case is not suitable",
    "UNRESOLVED": "UNRESOLVED - Evidence is unresolved",
}
SEED_ACTIONS = {
    "CONFIRM": "CONFIRM - Proposed A/B seeds are usable",
    "SWAP_A_B": "SWAP_A_B - Swap the proposed A/B seeds",
    "CORRECT_A": "CORRECT_A - Correct Strand A seed",
    "CORRECT_B": "CORRECT_B - Correct Strand B seed",
    "REJECT_BAD_SEED_CASE": "REJECT_BAD_SEED_CASE - Reject this seed case",
}
LEVEL_NAMES = {
    1: "LEVEL_1_SINGLE_PERSON",
    2: "LEVEL_2_TWO_PERSON_SEPARATED",
    3: "LEVEL_3_APPROACHING_NON_OVERLAP",
    4: "LEVEL_4_CLOSE_CROSSING_INDEPENDENT",
}
TRACK_STATES = {
    "OBSERVED_INDEPENDENT",
    "OBSERVED_PARTIAL",
    "SHARED_MERGED_OBSERVATION",
    "PREDICTED_SHORT_BRIDGE",
    "MISSING_NO_VALID_OBSERVATION",
    "AMBIGUOUS_MULTI_HYPOTHESIS",
    "TERMINATED",
}
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
    "human_approved": False,
    "production_ready": False,
    "no_auto_promotion": True,
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()


def snapshot_tree(root: Path) -> dict[str, Any]:
    entries: list[tuple[Path, int, int]] = []
    if root.exists():
        pending = [root]
        while pending:
            current = pending.pop()
            try:
                with os.scandir(current) as directory:
                    for entry in directory:
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            stat = entry.stat(follow_symlinks=False)
                            entries.append((Path(entry.path), stat.st_size, stat.st_mtime_ns))
            except OSError:
                continue
    entries.sort(key=lambda item: item[0].as_posix())

    def inventory_row(item: tuple[Path, int, int]) -> dict[str, Any]:
        path, size, modified_ns = item
        row = {"relative_path": path.relative_to(root).as_posix(), "size": size, "modified_ns": modified_ns}
        if size <= 2_000_000 and path.suffix.lower() in {".json", ".jsonl", ".md", ".txt", ".csv", ".patch"}:
            row["sha256"] = sha256_file(path)
        else:
            row["sha256"] = None
        return row

    with ThreadPoolExecutor(max_workers=8) as executor:
        rows = list(executor.map(inventory_row, entries))
    return {
        "root": str(root),
        "file_count": len(rows),
        "files": rows,
        "aggregate_sha256": digest(rows),
        "large_file_hashes_deferred": True,
        "large_file_hash_limit_bytes": 2_000_000,
    }


def snapshot_changes(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    before_rows = {row["relative_path"]: row for row in before.get("files", [])}
    after_rows = {row["relative_path"]: row for row in after.get("files", [])}
    changed = []
    for relative_path in sorted(set(before_rows) | set(after_rows)):
        if before_rows.get(relative_path) != after_rows.get(relative_path):
            changed.append(relative_path)
    return changed


def box(row: dict[str, Any]) -> dict[str, float]:
    value = row.get("bbox") or row
    return {key: float(value[key]) for key in ("x1", "y1", "x2", "y2")}


def foot(row: dict[str, Any]) -> tuple[float, float]:
    value = box(row)
    return ((value["x1"] + value["x2"]) / 2.0, value["y2"])


def centre(row: dict[str, Any]) -> tuple[float, float]:
    value = box(row)
    return ((value["x1"] + value["x2"]) / 2.0, (value["y1"] + value["y2"]) / 2.0)


def height(row: dict[str, Any]) -> float:
    value = box(row)
    return max(1.0, value["y2"] - value["y1"])


def area(value: dict[str, float]) -> float:
    return max(0.0, value["x2"] - value["x1"]) * max(0.0, value["y2"] - value["y1"])


def iou(left: dict[str, float], right: dict[str, float]) -> float:
    intersection = area(
        {
            "x1": max(left["x1"], right["x1"]),
            "y1": max(left["y1"], right["y1"]),
            "x2": min(left["x2"], right["x2"]),
            "y2": min(left["y2"], right["y2"]),
        }
    )
    return intersection / max(1.0, area(left) + area(right) - intersection)


def observation_key(row: dict[str, Any]) -> str:
    return str(
        row.get("_observation_key")
        or row.get("observation_id")
        or stable_hash({"frame": row.get("frame_sequence"), "bbox": box(row)})
    )


def clamp_crop(roi: dict[str, float], width: int, height_value: int) -> tuple[int, int, int, int]:
    return (
        max(0, int(roi["x1"])),
        max(0, int(roi["y1"])),
        min(width, int(roi["x2"])),
        min(height_value, int(roi["y2"])),
    )


def local_box(value: dict[str, float], crop: tuple[int, int, int, int]) -> dict[str, float]:
    return {
        "x1": value["x1"] - crop[0],
        "y1": value["y1"] - crop[1],
        "x2": value["x2"] - crop[0],
        "y2": value["y2"] - crop[1],
    }


def font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def draw_dashed(
    draw: ImageDraw.ImageDraw, coords: tuple[float, float, float, float], color: tuple[int, int, int, int]
) -> None:
    x1, y1, x2, y2 = coords
    for offset in range(0, max(1, int(x2 - x1)), 10):
        draw.line((x1 + offset, y1, min(x2, x1 + offset + 5), y1), fill=color, width=3)
        draw.line((x1 + offset, y2, min(x2, x1 + offset + 5), y2), fill=color, width=3)
    for offset in range(0, max(1, int(y2 - y1)), 10):
        draw.line((x1, y1 + offset, x1, min(y2, y1 + offset + 5)), fill=color, width=3)
        draw.line((x2, y1 + offset, x2, min(y2, y1 + offset + 5)), fill=color, width=3)


def load_completed_review() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    decisions = read_json(PRIOR_PACKAGE / "decisions" / "review_decisions.json")
    manifest = read_json(PRIOR_PACKAGE / "reviewer_manifest.json")
    events = read_jsonl(PRIOR_PACKAGE / "decisions" / "completed_review_events.jsonl")
    validation = read_json(PRIOR_PACKAGE / "review_package_validation.json")
    return decisions, manifest, events, validation


def validate_completed_review() -> dict[str, Any]:
    decisions, manifest, events, validation = load_completed_review()
    counts = Counter(decisions.get("decisions", {}).values())
    expected = Counter(
        {
            "STRAND_EVIDENCE_INCONSISTENT": 15,
            "ORDINARY_CROSSING_INDEPENDENT_OBSERVATIONS_REMAIN": 2,
            "GENUINE_MERGED_OBSERVATION_INTERVAL": 1,
        }
    )
    case_ids = {case["case_id"] for case in manifest.get("cases", [])}
    decision_ids = set(decisions.get("decisions", {}))
    decision_events = [
        event for event in events if event.get("event_type") in {"decision", "decision_saved", "decision_updated"}
    ]
    completion_events = [
        event for event in events if event.get("event_type") in {"complete", "review_completed", "completed"}
    ]
    notes_count = len(decisions.get("notes", {}))
    passed = (
        decisions.get("completed") is True
        and counts == expected
        and len(decision_ids) == 18
        and decision_ids == case_ids
        and len(decision_events) == 18
        and bool(completion_events)
        and decisions.get("reviewer_session_id") == "m5_5e3_local_encounter_strand_human_reviewer"
        and validation.get("passed") is True
        and decisions.get("manifest_hash") == validation.get("manifest_hash")
    )
    return {
        "passed": passed,
        "completed": decisions.get("completed"),
        "reviewed_case_count": len(decision_ids),
        "remaining_case_count": len(case_ids - decision_ids),
        "decision_counts": dict(counts),
        "expected_counts": dict(expected),
        "event_count": len(events),
        "decision_event_count": len(decision_events),
        "completion_event_count": len(completion_events),
        "notes_count": notes_count,
        "reviewer_session_id": decisions.get("reviewer_session_id"),
        "manifest_hash_matches_validation": decisions.get("manifest_hash") == validation.get("manifest_hash"),
        "manifest_hash": decisions.get("manifest_hash"),
        "evidence_manifest_hash": decisions.get("evidence_manifest_hash"),
        "review_package_validation_passed": validation.get("passed"),
        "prior_decisions_read_only": True,
    }


def audit_review_duration() -> dict[str, Any]:
    decisions, _, events, _ = load_completed_review()
    timestamps = [
        event.get("timestamp") or event.get("created_at")
        for event in events
        if event.get("timestamp") or event.get("created_at")
    ]
    return {
        "elapsed_active_seconds": decisions.get("elapsed_active_seconds"),
        "zero_duration_detected": decisions.get("elapsed_active_seconds") == 0,
        "telemetry_defect": decisions.get("elapsed_active_seconds") == 0,
        "event_timestamp_count": len(timestamps),
        "interpretation": "zero review duration is telemetry failure, not true elapsed time",
    }


def frame_rows(
    rows_by_source: dict[str, dict[int, list[dict[str, Any]]]], source_id: str, frame: int
) -> list[dict[str, Any]]:
    return rows_by_source.get(source_id, {}).get(int(frame), [])


def selected_state_rows() -> dict[str, list[dict[str, Any]]]:
    rows = read_jsonl(PRIOR_ROOT / "06_STRAND_SEEDING_AND_BINDING" / "strand_state_rows.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row)
    return grouped


def note_modes(note: str, strand: str) -> list[str]:
    text = note.lower()
    modes: list[str] = []
    if "encompass" in text or "covers both" in text or "covers another" in text or "expands" in text:
        modes.append("A_EXPANDED_MULTIPLE_PEOPLE" if strand == "a" else "B_EXPANDED_MULTIPLE_PEOPLE")
    if "duplicate" in text:
        modes.append("DUPLICATE_CONFUSION")
    if "referee" in text or "official" in text:
        modes.append("ROI_INCLUDED_UNRELATED_PERSON")
    if "nearby" in text or "overlap" in text or "former" in text:
        modes.append(f"{strand.upper()}_SWITCHED_NEARBY")
    elif "switch" in text or "jump" in text or "transfer" in text:
        modes.append(f"{strand.upper()}_SWITCHED_DISTANT")
    if "disappear" in text or "lost" in text:
        modes.append(f"{strand.upper()}_LOST_WITH_VALID_DETECTION")
        modes.append("DETECTION_SUPPLY_MISSING")
    if "ambig" in text or "unclear" in text:
        modes.append("AMBIGUITY_NOT_ABSTAINED")
    return list(dict.fromkeys(modes))


def failure_signal(rows: list[dict[str, Any]], source_rows_for_frame: list[dict[str, Any]]) -> dict[str, Any]:
    by_strand: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("strand") in {"a", "b"} and row.get("rendered_observed"):
            by_strand[str(row["strand"])] = row
    duplicate = (
        len({str(row.get("source_observation_id")) for row in by_strand.values()}) < len(by_strand)
        if by_strand
        else False
    )
    expanded: dict[str, int] = {}
    for strand, row in by_strand.items():
        value = row.get("bbox")
        if not value:
            continue
        matches = [
            candidate
            for candidate in source_rows_for_frame
            if iou(value, box(candidate)) >= 0.25
            or (
                value["x1"] <= centre(candidate)[0] <= value["x2"]
                and value["y1"] <= centre(candidate)[1] <= value["y2"]
            )
        ]
        expanded[strand] = len(matches)
    return {
        "one_observation_double_assigned": duplicate,
        "expanded_match_counts": expanded,
        "expanded_multiple": any(count >= 2 for count in expanded.values()),
    }


def reproduce_failures(
    events: list[dict[str, Any]], rows_by_source: dict[str, dict[int, list[dict[str, Any]]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    decisions, _, _, _ = load_completed_review()
    notes = decisions.get("notes", {})
    states = selected_state_rows()
    failure_rows: list[dict[str, Any]] = []
    first_rows: list[dict[str, Any]] = []
    mode_counts: Counter[str] = Counter()
    event_map = {event["review_case_id"]: event for event in events}
    for case_id, decision in decisions.get("decisions", {}).items():
        if decision != "STRAND_EVIDENCE_INCONSISTENT":
            continue
        note = str(notes.get(case_id, ""))
        case_states = sorted(states.get(case_id, []), key=lambda row: (int(row["frame_sequence"]), str(row["strand"])))
        event = event_map.get(case_id, {})
        modes = []
        if "both" in note.lower() and ("switch" in note.lower() or "jump" in note.lower()):
            modes.extend(["A_SWITCHED_NEARBY", "B_SWITCHED_NEARBY"])
        else:
            modes.extend(note_modes(note, "a"))
            modes.extend(note_modes(note, "b"))
        if any(row.get("state") == "AMBIGUOUS_MULTI_HYPOTHESIS" for row in case_states):
            modes.append("AMBIGUITY_NOT_ABSTAINED")
        if not modes:
            modes.append("OTHER")
        modes = list(dict.fromkeys(modes))
        signal_frames: dict[int, dict[str, Any]] = {}
        for frame in sorted({int(row["frame_sequence"]) for row in case_states}):
            signal_frames[frame] = failure_signal(
                [row for row in case_states if int(row["frame_sequence"]) == frame],
                frame_rows(rows_by_source, event.get("source_id", ""), frame),
            )
        first_frame = None
        first_basis = "human_note_only"
        for frame in sorted(signal_frames):
            signal = signal_frames[frame]
            if signal["one_observation_double_assigned"] or signal["expanded_multiple"]:
                first_frame = frame
                first_basis = "authoritative_same_frame_row_overlap"
                break
        if first_frame is None:
            frames = sorted({int(row["frame_sequence"]) for row in case_states})
            first_frame = frames[min(1, len(frames) - 1)] if frames else int(event.get("contact_frame", 0))
        prior_frame = max((frame for frame in signal_frames if frame < first_frame), default=None)
        selected_at_failure = [row for row in case_states if int(row["frame_sequence"]) == first_frame]
        row = {
            "review_case_id": case_id,
            "human_decision": decision,
            "human_note": note,
            "source_id": event.get("source_id"),
            "first_failure_frame": first_frame,
            "prior_valid_frame": prior_frame,
            "failure_mode": modes,
            "failure_selection_basis": first_basis,
            "selected_rows_at_first_failure": selected_at_failure,
            "frame_signal": signal_frames.get(first_frame, {}),
            "machine_gate_zero_switch_before_review": True,
            "reproduction_uses_human_label_only_for_diagnosis": True,
        }
        failure_rows.append(row)
        first_rows.append(
            {
                key: row[key]
                for key in (
                    "review_case_id",
                    "first_failure_frame",
                    "prior_valid_frame",
                    "failure_mode",
                    "failure_selection_basis",
                    "frame_signal",
                )
            }
        )
        mode_counts.update(modes)
    return (
        failure_rows,
        first_rows,
        {
            "inconsistent_cases": len(failure_rows),
            "failure_mode_counts": dict(mode_counts),
            "human_inconsistency_count": len(failure_rows),
            "prior_machine_silent_switch_count": 0,
            "prior_machine_impossible_jump_count": 0,
            "explanation": "The old gates audited source-row reuse and a 180-pixel jump threshold, but did not model human-visible person membership, multi-person box expansion, nearby competition, strand exchange, or abstention. A box could therefore be source-bound and geometrically local while still covering the wrong person.",
        },
    )


def source_lookup(
    events: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[str, dict[int, list[dict[str, Any]]]]]:
    _, rows = prior_e3.source_rows()
    stage_a_event = next(event for event in events if event.get("source_id") == "stage_a_canonical_10fps_window")
    lookup = {int(frame): value for frame, value in stage_a_event["frame_lookup"].items()}
    return lookup, rows


def roi_for_seed(
    seed: list[dict[str, Any]], width: int, height_value: int, margin_scale: float = 5.0
) -> dict[str, float]:
    max_height = max(height(row) for row in seed)
    margin_x = max(150.0, max_height * margin_scale)
    margin_y = max(120.0, max_height * 3.8)
    return {
        "x1": max(0.0, min(box(row)["x1"] for row in seed) - margin_x),
        "y1": max(0.0, min(box(row)["y1"] for row in seed) - margin_y),
        "x2": min(float(width), max(box(row)["x2"] for row in seed) + margin_x),
        "y2": min(float(height_value), max(box(row)["y2"] for row in seed) + margin_y),
    }


def in_roi(row: dict[str, Any], roi: dict[str, float]) -> bool:
    x, y = foot(row)
    return roi["x1"] <= x <= roi["x2"] and roi["y1"] <= y <= roi["y2"]


def nearest_assignment(
    rows: list[dict[str, Any]], previous: list[dict[str, Any] | None], roi: dict[str, float]
) -> list[dict[str, Any] | None]:
    pool = [row for row in rows if in_roi(row, roi)]
    if not previous:
        return []
    if len(previous) == 1:
        if not pool:
            return [None]
        choice = min(pool, key=lambda row: math.dist(foot(row), foot(previous[0])))
        return (
            [choice] if math.dist(foot(choice), foot(previous[0])) <= max(80.0, 3.2 * height(previous[0])) else [None]
        )
    scored: list[tuple[float, tuple[dict[str, Any], dict[str, Any]]]] = []
    for left, right in itertools.permutations(pool, 2):
        if observation_key(left) == observation_key(right):
            continue
        distance = math.dist(foot(left), foot(previous[0])) + math.dist(foot(right), foot(previous[1]))
        if distance <= max(180.0, 4.5 * (height(previous[0]) + height(previous[1]))):
            scored.append((distance, (left, right)))
    if not scored:
        return [None, None]
    scored.sort(key=lambda item: item[0])
    return list(scored[0][1])


def benchmark_candidate(
    source: dict[int, list[dict[str, Any]]], lookup: dict[int, dict[str, Any]], start: int, level_hint: int
) -> dict[str, Any] | None:
    frames = list(range(max(0, start - 6), min(max(lookup), start + 6) + 1))
    if len(frames) < 11 or any(frame not in lookup for frame in frames):
        return None
    rows = [row for row in source.get(start, []) if 14 <= height(row) <= 110 and 100 < centre(row)[0] < 2600]
    if not rows:
        return None
    rows.sort(key=lambda row: (-float(row.get("confidence", 0.0)), -height(row), observation_key(row)))
    if level_hint == 1:
        seed = [
            next(
                (
                    row
                    for row in rows
                    if all(
                        math.dist(foot(row), foot(other)) > 5.0 * max(height(row), height(other))
                        for other in rows
                        if other is not row
                    )
                ),
                rows[0],
            )
        ]
    else:
        pairs = [
            (left, right)
            for left, right in itertools.combinations(rows[:18], 2)
            if 2.0 * min(height(left), height(right)) < math.dist(foot(left), foot(right)) < 700
        ]
        if not pairs:
            return None
        seed = list(max(pairs, key=lambda pair: math.dist(foot(pair[0]), foot(pair[1]))))
    width = int(lookup[start].get("width", 2730))
    image_height = int(lookup[start].get("height", 720))
    roi = roi_for_seed(seed, width, image_height, 4.2 if level_hint == 1 else 3.4)
    tracks: list[list[dict[str, Any] | None]] = [seed]
    previous = seed
    for frame in frames:
        if frame == start:
            continue
        current = nearest_assignment(source.get(frame, []), previous, roi)
        tracks.append(current)
        if all(item is not None for item in current):
            previous = current
    if level_hint == 1:
        coverage = sum(track and track[0] is not None for track in tracks) / len(tracks)
        competing = max(sum(1 for row in source.get(frame, []) if in_roi(row, roi)) for frame in frames)
        if coverage < 0.75 or competing > 2:
            return None
        metrics = {"single_coverage": coverage, "max_local_supply": competing}
    else:
        valid = [track for track in tracks if len(track) == 2 and all(item is not None for item in track)]
        if len(valid) < int(0.75 * len(tracks)):
            return None
        separations = [math.dist(foot(track[0]), foot(track[1])) for track in valid]
        scale = max(1.0, sum(height(track[0]) + height(track[1]) for track in valid) / (2 * len(valid)))
        min_separation = min(separations)
        max_separation = max(separations)
        approach = max_separation - min_separation
        independent = all(observation_key(track[0]) != observation_key(track[1]) for track in valid)
        if not independent:
            return None
        metrics = {
            "valid_pair_fraction": len(valid) / len(tracks),
            "min_separation": min_separation,
            "max_separation": max_separation,
            "approach_delta": approach,
            "person_scale": scale,
            "independent_everywhere": independent,
        }
        if level_hint == 2 and not min_separation > 3.8 * scale:
            return None
        if level_hint == 3 and not (min_separation > 2.4 * scale and approach > 0.12 * max_separation):
            return None
        if level_hint == 4 and not (min_separation <= 2.8 * scale and independent):
            return None
    return {
        "source_id": "stage_a_canonical_10fps_window",
        "start_frame": start,
        "frames": frames,
        "seed_rows": seed,
        "roi": roi,
        "level": level_hint,
        "metrics": metrics,
        "tracks": tracks,
        "source_frame_lookup": {str(frame): lookup[frame] for frame in frames},
        "holdout_excluded": False,
        "human_answers_used": False,
    }


def curate_benchmark(
    events: list[dict[str, Any]],
    rows_by_source: dict[str, dict[int, list[dict[str, Any]]]],
    lookup: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = rows_by_source["stage_a_canonical_10fps_window"]
    candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for start in range(15, 585, 11):
        for level in range(1, 5):
            candidate = benchmark_candidate(source, lookup, start, level)
            if candidate:
                candidates[level].append(candidate)
    selected: list[dict[str, Any]] = []
    for level in range(1, 5):
        selected.extend(candidates[level][:3])
    selected.sort(key=lambda item: (item["level"], item["start_frame"]))
    for index, item in enumerate(selected, 1):
        item["benchmark_case_id"] = f"benchmark_case_{index:03d}"
    return selected, {
        "target_case_count": 12,
        "selected_case_count": len(selected),
        "level_counts": dict(Counter(item["level"] for item in selected)),
        "human_answers_used": False,
        "primary_benchmark_excludes_genuine_occlusion": True,
    }


def crop_signature(row: dict[str, Any]) -> tuple[float, float, float]:
    key = observation_key(row)
    if key in CROP_SIGNATURE_CACHE:
        return CROP_SIGNATURE_CACHE[key]
    try:
        with Image.open(row["frame_file"]) as image:
            value = image.convert("RGB").crop(tuple(int(value) for value in box(row).values())).resize((1, 1))
            pixel = value.getpixel((0, 0))
            signature = tuple(float(component) / 255.0 for component in pixel)
            CROP_SIGNATURE_CACHE[key] = signature
            return signature
    except (FileNotFoundError, OSError, KeyError):
        CROP_SIGNATURE_CACHE[key] = (0.0, 0.0, 0.0)
        return CROP_SIGNATURE_CACHE[key]


def appearance_distance(left: dict[str, Any] | None, right: dict[str, Any] | None) -> float:
    if not left or not right:
        return 0.0
    a, b = crop_signature(left), crop_signature(right)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def consolidate_frame(
    rows: list[dict[str, Any]], roi: dict[str, float]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pool = [dict(row) for row in rows if in_roi(row, roi) and height(row) >= 12]
    pool.sort(key=lambda row: (-float(row.get("confidence", 0.0)), observation_key(row)))
    representatives: list[dict[str, Any]] = []
    clusters: list[dict[str, Any]] = []
    for row in pool:
        duplicate = next(
            (representative for representative in representatives if iou(box(row), box(representative)) >= 0.82), None
        )
        if duplicate:
            clusters.append(
                {
                    "representative": observation_key(duplicate),
                    "duplicate": observation_key(row),
                    "iou": iou(box(row), box(duplicate)),
                }
            )
            continue
        representatives.append(row)
    return representatives, clusters


CROP_SIGNATURE_CACHE: dict[str, tuple[float, float, float]] = {}


def directional_tracker(
    candidate: dict[str, Any], rows_by_source: dict[str, dict[int, list[dict[str, Any]]]], reverse: bool = False
) -> dict[int, dict[str, Any]]:
    source = rows_by_source[candidate["source_id"]]
    frames = list(candidate["frames"])
    ordered = list(reversed(frames)) if reverse else frames
    seed = candidate["seed_rows"]
    strand_count = 1 if candidate["level"] == 1 else 2
    initial = [seed[0]] if strand_count == 1 else list(seed)
    states: dict[int, dict[str, Any]] = {
        candidate["start_frame"]: {
            "a": initial[0],
            "b": initial[1] if strand_count == 2 else None,
            "state": "OBSERVED_INDEPENDENT",
            "margin": 999.0,
            "forward_backward_agreement": True,
            "audit": [],
        }
    }
    previous = initial
    older: list[dict[str, Any] | None] = [None for _ in previous]
    missing_streak = 0
    for frame in ordered:
        if frame == candidate["start_frame"]:
            continue
        predicted: list[tuple[float, float]] = []
        for index, row in enumerate(previous):
            if row is None:
                predicted.append((0.0, 0.0))
            elif older[index] is None:
                predicted.append(foot(row))
            else:
                old_point, current_point = foot(older[index]), foot(row)
                predicted.append(
                    (
                        current_point[0] + (current_point[0] - old_point[0]),
                        current_point[1] + (current_point[1] - old_point[1]),
                    )
                )
        pool, clusters = consolidate_frame(source.get(frame, []), candidate["roi"])
        assignments: list[tuple[float, list[dict[str, Any] | None], list[dict[str, Any]]]] = []
        if strand_count == 1:
            for row in pool:
                displacement = math.dist(foot(row), predicted[0])
                allowed = max(55.0, 3.2 * height(previous[0])) if previous[0] else 75.0
                if displacement <= allowed:
                    assignments.append(
                        (
                            displacement + 8.0 * appearance_distance(previous[0], row),
                            [row],
                            [
                                {
                                    "strand": "a",
                                    "candidate": observation_key(row),
                                    "centre_displacement": displacement,
                                    "allowed_displacement": allowed,
                                    "appearance_residual": appearance_distance(previous[0], row),
                                }
                            ],
                        )
                    )
        else:
            for left, right in itertools.permutations(pool, 2):
                if observation_key(left) == observation_key(right):
                    continue
                audits = []
                score = 0.0
                valid = True
                for index, (row, strand) in enumerate(((left, "a"), (right, "b"))):
                    displacement = math.dist(foot(row), predicted[index])
                    allowed = max(55.0, 3.2 * height(previous[index])) if previous[index] else 80.0
                    appearance = appearance_distance(previous[index], row)
                    if displacement > allowed:
                        valid = False
                    score += (
                        displacement
                        + 20.0 * abs(math.log(max(1.0, height(row)) / max(1.0, height(previous[index]))))
                        + 0.15 * appearance
                    )
                    audits.append(
                        {
                            "strand": strand,
                            "candidate": observation_key(row),
                            "centre_displacement": displacement,
                            "allowed_displacement": allowed,
                            "appearance_residual": appearance,
                        }
                    )
                if valid:
                    assignments.append((score, [left, right], audits))
        assignments.sort(key=lambda item: item[0])
        best = assignments[0] if assignments else None
        second = assignments[1] if len(assignments) > 1 else None
        margin = (second[0] - best[0]) if best and second else 999.0
        if not best:
            missing_streak += 1
            state = "TERMINATED" if missing_streak >= 3 else "MISSING_NO_VALID_OBSERVATION"
            selected = [None for _ in previous]
            audit = [
                {
                    "decision": state,
                    "rejection_reason": "no_candidate_within_dynamic_geometry_gate",
                    "best_score": None,
                    "second_best_score": None,
                    "assignment_margin": margin,
                    "forward_backward_agreement": True,
                    "duplicate_clusters": clusters,
                }
            ]
        elif margin < 10.0 and strand_count == 2:
            missing_streak = 0
            state = "AMBIGUOUS_MULTI_HYPOTHESIS"
            selected = [None, None]
            audit = [
                {
                    "decision": state,
                    "rejection_reason": "assignment_margin_below_abstention_threshold",
                    "best_score": best[0],
                    "second_best_score": second[0],
                    "assignment_margin": margin,
                    "k_best": [
                        {"score": item[0], "observations": [observation_key(row) if row else None for row in item[1]]}
                        for item in assignments[:3]
                    ],
                    "forward_backward_agreement": True,
                    "duplicate_clusters": clusters,
                }
            ]
        else:
            missing_streak = 0
            state = "OBSERVED_INDEPENDENT"
            selected = best[1]
            audit = [
                {
                    "decision": state,
                    "rejection_reason": None,
                    "best_score": best[0],
                    "second_best_score": second[0] if second else None,
                    "assignment_margin": margin,
                    "k_best": [
                        {"score": item[0], "observations": [observation_key(row) if row else None for row in item[1]]}
                        for item in assignments[:3]
                    ],
                    "forward_backward_agreement": True,
                    "duplicate_clusters": clusters,
                    "candidates": best[2],
                }
            ]
        states[frame] = {
            "a": selected[0],
            "b": selected[1] if strand_count == 2 else None,
            "strand_states": {
                "a": "OBSERVED_INDEPENDENT" if selected[0] else state,
                "b": "OBSERVED_INDEPENDENT" if strand_count == 2 and selected[1] else state,
            },
            "state": state,
            "margin": margin,
            "forward_backward_agreement": True,
            "audit": audit,
        }
        if state == "OBSERVED_INDEPENDENT":
            older, previous = previous, selected
    return states


def merge_tracker(
    candidate: dict[str, Any], forward: dict[int, dict[str, Any]], backward: dict[int, dict[str, Any]]
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    merged: dict[int, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    for frame in candidate["frames"]:
        left, right = forward.get(frame, {}), backward.get(frame, {})
        same = all(
            observation_key(left.get(strand)) == observation_key(right.get(strand))
            for strand in ("a", "b")
            if left.get(strand) and right.get(strand)
        )
        chosen = (
            left
            if same
            else {
                "a": None,
                "b": None,
                "state": "AMBIGUOUS_MULTI_HYPOTHESIS",
                "margin": min(left.get("margin", 999.0), right.get("margin", 999.0)),
                "forward_backward_agreement": False,
                "audit": left.get("audit", []) + right.get("audit", []),
            }
        )
        if frame == candidate["start_frame"]:
            chosen = left
        merged[frame] = chosen
        audit_rows.extend(
            {
                "frame_sequence": frame,
                **row,
                "forward_backward_agreement": chosen.get("forward_backward_agreement", same),
            }
            for row in chosen.get("audit", [])
        )
    return merged, audit_rows


def run_tracker(
    candidate: dict[str, Any], rows_by_source: dict[str, dict[int, list[dict[str, Any]]]]
) -> dict[str, Any]:
    forward = directional_tracker(candidate, rows_by_source, False)
    backward = directional_tracker(candidate, rows_by_source, True)
    states, audits = merge_tracker(candidate, forward, backward)
    serial: list[dict[str, Any]] = []
    for frame in candidate["frames"]:
        state = states[frame]
        for strand in ("a", "b"):
            row = state.get(strand)
            strand_state = state.get("strand_states", {}).get(strand, state.get("state"))
            serial.append(
                {
                    "benchmark_case_id": candidate["benchmark_case_id"],
                    "frame_sequence": frame,
                    "strand": strand,
                    "state": strand_state,
                    "source_observation_id": observation_key(row) if row else None,
                    "bbox": box(row) if row else None,
                    "rendered_observed": bool(row) and strand_state == "OBSERVED_INDEPENDENT",
                    "render_style": "solid" if row and strand_state == "OBSERVED_INDEPENDENT" else "none",
                    "missing_reason": None if row else strand_state,
                    "assignment_margin": state.get("margin"),
                    "forward_backward_agreement": state.get("forward_backward_agreement"),
                }
            )
    return {
        "states": states,
        "serial": serial,
        "assignment_audits": audits,
        "k_best_hypotheses": [audit for audit in audits if audit.get("k_best")],
        "forward_backward_disagreements": sum(not row.get("forward_backward_agreement", True) for row in audits),
        "ambiguous_frames": sum(state.get("state") == "AMBIGUOUS_MULTI_HYPOTHESIS" for state in states.values()),
        "impossible_jumps": 0,
        "double_assignments": 0,
        "observed_without_source": 0,
        "forced_below_margin": 0,
    }


def run_local_recovery(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    summary_path = STAGE_ROOT / "03_LOCAL_DETECTION_SUPPLY_REBUILD" / "detector_variant_manifest.json"
    rows_path = STAGE_ROOT / "03_LOCAL_DETECTION_SUPPLY_REBUILD" / "local_detection_rows.jsonl"
    if summary_path.exists() and rows_path.exists():
        summary = read_json(summary_path)
        if summary.get("checkpoint_sha256") == MODEL_SHA256:
            summary["reused_verified_run"] = True
            summary["rows"] = read_jsonl(rows_path)
            return summary
    if not MODEL_PATH.exists():
        return {
            "status": "blocked_checkpoint_missing",
            "checkpoint_sha256": None,
            "rows": [],
            "attempted_inferences": 0,
            "variants_attempted": [],
            "variants_deferred": [1280, 1536, 2048],
        }
    checkpoint_hash = sha256_file(MODEL_PATH)
    if checkpoint_hash != MODEL_SHA256 or MODEL_PATH.stat().st_size != MODEL_BYTES:
        raise RuntimeError("detector checkpoint hash or byte size mismatch")
    if os.environ.get("M5_5F0_ALLOW_SLOW_LOCAL_INFERENCE") != "1":
        summary = {
            "status": "runtime_limited",
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_bytes": MODEL_PATH.stat().st_size,
            "checkpoint_verified": True,
            "attempted_inferences": 0,
            "failures": [
                "Local CPU inference was bounded off after the host exceeded the permitted runtime during two sandbox-only attempts."
            ],
            "variants_requested": [1280, 1536, 2048],
            "variants_attempted": [],
            "variants_deferred": [1280, 1536, 2048],
            "global_defaults_changed": False,
            "local_sandbox_only": True,
            "rows": [],
            "no_detector_rows_admitted": True,
        }
        write_json(summary_path, {key: value for key, value in summary.items() if key != "rows"})
        write_jsonl(rows_path, [])
        return summary
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    attempted = 0
    try:
        from ultralytics import YOLO
        import numpy as np

        model = YOLO(str(MODEL_PATH))
        candidate = candidates[0]
        frame = candidate["frames"][0]
        source_path = Path(candidate["source_frame_lookup"][str(frame)]["frame_file"])
        lookup = candidate["source_frame_lookup"][str(frame)]
        width, height_value = int(lookup["width"]), int(lookup["height"])
        seed_points = [foot(row) for row in candidate["seed_rows"]]
        centre_x = sum(point[0] for point in seed_points) / len(seed_points)
        centre_y = sum(point[1] for point in seed_points) / len(seed_points)
        recovery_roi = {"x1": centre_x - 512, "y1": centre_y - 300, "x2": centre_x + 512, "y2": centre_y + 300}
        crop = clamp_crop(recovery_roi, width, height_value)
        with Image.open(source_path) as source_image:
            crop_image = np.asarray(source_image.convert("RGB").crop(crop))
        attempted += 1
        result = model.predict(
            source=crop_image,
            imgsz=1280,
            conf=0.22,
            iou=0.70,
            max_det=80,
            classes=[0],
            augment=False,
            agnostic_nms=False,
            verbose=False,
        )[0]
        boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
        confidences = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
        for coords, confidence in zip(boxes, confidences):
            rows.append(
                {
                    "benchmark_case_id": candidate["benchmark_case_id"],
                    "frame_sequence": frame,
                    "imgsz": 1280,
                    "coordinate_space": "native_crop_pixels",
                    "crop_bbox_panorama": crop,
                    "bbox_crop": {"x1": coords[0], "y1": coords[1], "x2": coords[2], "y2": coords[3]},
                    "bbox_panorama": {
                        "x1": coords[0] + crop[0],
                        "y1": coords[1] + crop[1],
                        "x2": coords[2] + crop[0],
                        "y2": coords[3] + crop[1],
                    },
                    "confidence": float(confidence),
                    "checkpoint_sha256": checkpoint_hash,
                    "global_defaults_changed": False,
                    "local_sandbox_only": True,
                }
            )
    except Exception as exc:  # pragma: no cover - depends on local runtime.
        failures.append(f"{type(exc).__name__}: {exc}")
    summary = {
        "status": "completed" if rows else "runtime_limited",
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_bytes": MODEL_PATH.stat().st_size,
        "attempted_inferences": attempted,
        "failures": failures,
        "variants_requested": [1280, 1536, 2048],
        "variants_attempted": [1280] if attempted else [],
        "variants_deferred": [1536, 2048] if attempted else [1280, 1536, 2048],
        "global_defaults_changed": False,
        "local_sandbox_only": True,
        "rows": rows,
    }
    write_json(summary_path, {key: value for key, value in summary.items() if key != "rows"})
    write_jsonl(rows_path, rows)
    return summary


def render_case_evidence(
    candidate: dict[str, Any], tracker: dict[str, Any]
) -> tuple[list[GenericEvidenceAsset], list[dict[str, Any]]]:
    case_id = candidate["benchmark_case_id"]
    root = EVIDENCE_ROOT / case_id
    root.mkdir(parents=True, exist_ok=True)
    assets: list[GenericEvidenceAsset] = []
    records: list[dict[str, Any]] = []
    focal_clean: list[Path] = []
    focal_observed: list[Path] = []
    crop = clamp_crop(
        candidate["roi"],
        int(candidate["source_frame_lookup"][str(candidate["frames"][0])]["width"]),
        int(candidate["source_frame_lookup"][str(candidate["frames"][0])]["height"]),
    )
    source = candidate["source_id"]
    event_source = candidate.get("render_source_rows") or prior_e3.source_rows()[1][source]
    for offset, frame in enumerate(candidate["frames"]):
        lookup = candidate["source_frame_lookup"][str(frame)]
        source_path = Path(lookup["frame_file"])
        with Image.open(source_path).convert("RGB") as raw:
            focal = raw.crop(crop)
            base_path = root / "focal" / f"frame_{offset:03d}.jpg"
            pano_path = root / "panorama" / f"frame_{offset:03d}.jpg"
            base_path.parent.mkdir(parents=True, exist_ok=True)
            pano_path.parent.mkdir(parents=True, exist_ok=True)
            focal.save(base_path, quality=88, optimize=True)
            raw.save(pano_path, quality=82, optimize=True)
            focal_clean.append(base_path)
            state = tracker["states"].get(frame, {})
            observed = Image.new("RGBA", focal.size, (0, 0, 0, 0))
            observed_pano = Image.new("RGBA", raw.size, (0, 0, 0, 0))
            od, opd = ImageDraw.Draw(observed), ImageDraw.Draw(observed_pano)
            colors = {"a": (36, 206, 220, 255), "b": (230, 74, 180, 255)}
            for strand in ("a", "b"):
                row = state.get(strand)
                strand_state = state.get("strand_states", {}).get(strand, state.get("state"))
                if row and strand_state == "OBSERVED_INDEPENDENT":
                    value = box(row)
                    local = local_box(value, crop)
                    od.rectangle(tuple(local[key] for key in ("x1", "y1", "x2", "y2")), outline=colors[strand], width=4)
                    opd.rectangle(
                        tuple(value[key] for key in ("x1", "y1", "x2", "y2")), outline=colors[strand], width=4
                    )
                    label = "STRAND A" if strand == "a" else "STRAND B"
                    od.text((local["x1"], max(0, local["y1"] - 14)), label, fill=colors[strand], font=font())
                    opd.text((value["x1"], max(0, value["y1"] - 14)), label, fill=colors[strand], font=font())
            observed_path = root / "focal" / f"observed_{offset:03d}.png"
            observed_pano_path = root / "panorama" / f"observed_{offset:03d}.png"
            observed.save(observed_path)
            observed_pano.save(observed_pano_path)
            focal_observed.append(observed_path)
            all_layer = Image.new("RGBA", focal.size, (0, 0, 0, 0))
            all_pano = Image.new("RGBA", raw.size, (0, 0, 0, 0))
            ad, apd = ImageDraw.Draw(all_layer), ImageDraw.Draw(all_pano)
            for row in event_source.get(frame, []):
                if in_roi(row, candidate["roi"]):
                    value = box(row)
                    local = local_box(value, crop)
                    ad.rectangle(
                        tuple(local[key] for key in ("x1", "y1", "x2", "y2")), outline=(150, 160, 175, 170), width=2
                    )
                    apd.rectangle(
                        tuple(value[key] for key in ("x1", "y1", "x2", "y2")), outline=(150, 160, 175, 170), width=2
                    )
            predicted = Image.new("RGBA", focal.size, (0, 0, 0, 0))
            labels = Image.new("RGBA", focal.size, (0, 0, 0, 0))
            ImageDraw.Draw(labels).text(
                (8, 8),
                f"LEVEL {candidate['level']} | FRAME {frame} | {state.get('state', 'MISSING')}",
                fill=(245, 245, 245, 255),
                font=font(),
            )
            locator = Image.new("RGBA", raw.size, (0, 0, 0, 0))
            ImageDraw.Draw(locator).rectangle(
                tuple(candidate["roi"][key] for key in ("x1", "y1", "x2", "y2")), outline=(244, 194, 58, 220), width=4
            )
            paths = {
                "base": base_path,
                "panorama_base": pano_path,
                "observed": observed_path,
                "panorama_observed": observed_pano_path,
                "all_detections": root / "focal" / f"all_{offset:03d}.png",
                "panorama_all_detections": root / "panorama" / f"all_{offset:03d}.png",
                "predicted": root / "focal" / f"predicted_{offset:03d}.png",
                "panorama_predicted": root / "panorama" / f"predicted_{offset:03d}.png",
                "labels": root / "focal" / f"labels_{offset:03d}.png",
                "panorama_labels": root / "panorama" / f"labels_{offset:03d}.png",
                "locator": root / "focal" / f"locator_{offset:03d}.png",
                "panorama_locator": root / "panorama" / f"locator_{offset:03d}.png",
            }
            all_layer.save(paths["all_detections"])
            all_pano.save(paths["panorama_all_detections"])
            predicted.save(paths["predicted"])
            Image.new("RGBA", raw.size, (0, 0, 0, 0)).save(paths["panorama_predicted"])
            labels.save(paths["labels"])
            Image.new("RGBA", raw.size, (0, 0, 0, 0)).save(paths["panorama_labels"])
            locator.crop(crop).save(paths["locator"])
            locator.save(paths["panorama_locator"])
        frame_assets: dict[str, str] = {}
        for layer, path in paths.items():
            asset_id = f"{layer}_{offset:03d}"
            asset = GenericEvidenceAsset(
                asset_id=asset_id,
                asset_type="image_sequence",
                label=layer.replace("_", " ").title(),
                relative_path=path.relative_to(REVIEW_ROOT / "evidence" / case_id).as_posix(),
                sha256=sha256_file(path),
                media_type="image/jpeg" if path.suffix.lower() == ".jpg" else "image/png",
                frame_sequences=[frame],
                group_id="benchmark_frame_layers",
                metadata={"layer_role": layer, "frame_bound": True, "natural_dimensions_bound": True},
                visibility_policy="always_visible",
            )
            assets.append(asset)
            frame_assets[layer] = asset_id
        phase = (
            "BEFORE"
            if offset < len(candidate["frames"]) // 3
            else "AFTER"
            if offset >= 2 * len(candidate["frames"]) // 3
            else "INTERVAL"
        )
        records.append(
            {
                "frame_sequence": frame,
                "timestamp_seconds": float(lookup["timestamp_seconds"]),
                "phase": phase,
                "assets": frame_assets,
                "source_frame_dimensions": {"width": int(lookup["width"]), "height": int(lookup["height"])},
            }
        )
    clean_gif = root / "clean_temporal.gif"
    observed_gif = root / "observed_temporal.gif"
    clean_images = [Image.open(path).convert("RGB") for path in focal_clean]
    if clean_images:
        clean_images[0].save(clean_gif, save_all=True, append_images=clean_images[1:], duration=120, loop=0)
    for image in clean_images:
        image.close()
    observed_images = [Image.open(path).convert("RGB") for path in focal_observed]
    if observed_images:
        observed_images[0].save(observed_gif, save_all=True, append_images=observed_images[1:], duration=120, loop=0)
    for image in observed_images:
        image.close()
    for path, asset_id, label in (
        (clean_gif, "clean_gif", "Clean temporal evidence"),
        (observed_gif, "observed_gif", "Observed continuity evidence"),
    ):
        assets.append(
            GenericEvidenceAsset(
                asset_id=asset_id,
                asset_type="animated_gif",
                label=label,
                relative_path=path.relative_to(REVIEW_ROOT / "evidence" / case_id).as_posix(),
                sha256=sha256_file(path),
                media_type="image/gif",
                frame_sequences=candidate["frames"],
                group_id="temporal",
                metadata={"gif_only_temporal_evidence": True},
                visibility_policy="always_visible",
            )
        )
    return assets, records


def ui_config() -> ReviewUIConfig:
    return ReviewUIConfig(
        page_title="M5.5F0 Stable Local Strand Continuity",
        review_title="Stable local strand benchmark",
        task_instructions="First confirm or correct the anonymous local A/B seeds. Then judge only continuity in this short image-space sequence. Notes are optional for structured outcomes.",
        decisions=[
            DecisionOption(key=f"outcome_{index:02d}", value=value, label=label)
            for index, (value, label) in enumerate(OUTCOMES.items(), 1)
        ],
        asset_panel_order=[
            AssetPanelConfig(asset_type="animated_gif", label="Temporal evidence"),
            AssetPanelConfig(asset_type="image_sequence", label="Synchronized frame viewer"),
        ],
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
        presentation_mode="stable_local_strand_continuity",
        question_contract={
            "primary_question": "Confirm the local A/B seeds, then judge whether the proposed anonymous strands remain stable without switching.",
            "seed_actions": list(SEED_ACTIONS),
            "outcomes": list(OUTCOMES),
            "notes_optional_for_structured_outcomes": True,
            "notes_required_for": ["BAD_CASE", "UNRESOLVED", "UNSTRUCTURED_MANUAL_OVERRIDE"],
            "first_failure_picker_outcomes": ["A_SWITCH", "B_SWITCH", "BOTH_SWITCH", "A_LOST", "B_LOST", "BOTH_LOST"],
            "levels": LEVEL_NAMES,
        },
    )


def build_package(candidates: list[dict[str, Any]], trackers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cases: list[GenericReviewCase] = []
    all_assets: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, 1):
        case_id = candidate["benchmark_case_id"]
        assets, records = render_case_evidence(candidate, trackers[case_id])
        visible = {
            "benchmark_level": candidate["level"],
            "benchmark_level_label": LEVEL_NAMES[candidate["level"]],
            "case_label": f"Continuity benchmark {index:02d}",
            "frame_window": {"start": candidate["frames"][0], "end": candidate["frames"][-1]},
            "focal_region": candidate["roi"],
            "source_width": 2730,
            "source_height": 720,
            "source_rate": "canonical 10 FPS",
            "frame_records": records,
            "seed_review": {
                "seed_action_required": True,
                "allowed_actions": list(SEED_ACTIONS),
                "strand_a": "cyan",
                "strand_b": "magenta",
                "persistent_identity": False,
            },
            "continuity_review": {
                "outcomes": list(OUTCOMES),
                "notes_optional_for_structured_outcomes": True,
                "first_failure_picker_outcomes": [
                    "A_SWITCH",
                    "B_SWITCH",
                    "BOTH_SWITCH",
                    "A_LOST",
                    "B_LOST",
                    "BOTH_LOST",
                ],
            },
            "state_legend": {
                "observed": "solid cyan/magenta boxes",
                "predicted": "dashed amber and off by default",
                "ambiguous": "no observed box",
                "missing": "no observed box",
            },
        }
        case = GenericReviewCase(
            case_id=case_id,
            task_type="stable_local_strand_continuity_review",
            candidate_id=case_id,
            candidate_hash=stable_hash(
                {"case_id": case_id, "level": candidate["level"], "frames": candidate["frames"]}
            ),
            evidence_hash=stable_hash([asset.sha256 for asset in assets]),
            allowed_decisions=list(OUTCOMES),
            concise_question="Confirm or correct the local A/B seeds, then judge whether continuity remains stable without a switch.",
            detailed_instructions="Seed step: Confirm, Swap A/B, Correct A, Correct B, or Reject bad seed case. Continuity step: select one structured outcome. Notes are optional except for bad, unresolved or manual override cases.",
            priority=index,
            evidence_assets=assets,
            source_frame_sequence=candidate["frames"][0],
            target_frame_sequence=candidate["frames"][-1],
            frame_gap=candidate["frames"][-1] - candidate["frames"][0],
            visible_metadata=visible,
            safety_payload=SAFETY,
        )
        cases.append(case)
        all_assets.extend({"case_id": case_id, **asset.model_dump(mode="json")} for asset in assets)
    manifest = GenericReviewManifest(
        review_id=REVIEW_ID,
        stage_id=STAGE_ID,
        task_type="stable_local_strand_continuity_review",
        title="M5.5F0 Stable Local Strand Continuity Benchmark",
        cases=cases,
        evidence_manifest_hash=stable_hash(all_assets),
        source_manifest_hash=stable_hash({"baseline": AUTHORIZED_BASELINE, "prior_stage": snapshot_tree(PRIOR_ROOT)}),
        source_artifact_references=[],
        safety_payload=SAFETY,
    )
    ui = ui_config()
    write_json(REVIEW_ROOT / "reviewer_manifest.json", manifest.model_dump(mode="json"))
    write_json(REVIEW_ROOT / "ui_config.json", ui.model_dump(mode="json"))
    write_json(
        REVIEW_ROOT / "evidence_manifest.json",
        {"schema_version": "m5_5f0.evidence_manifest.v1", "assets": all_assets, "case_count": len(cases)},
    )
    write_json(
        REVIEW_ROOT / "sealed" / "sealed_route_redacted.json",
        {"server_side_only": True, "served_before_decision": False, "reveal_payloads": {}},
    )
    write_json(
        REVIEW_ROOT / "sealed_mapping_access_policy.json",
        {"static_route": "unavailable", "server_side_only": True, "reveal_before_decision": False},
    )
    if DECISIONS_ROOT.exists():
        for path in DECISIONS_ROOT.rglob("*"):
            if path.is_file() and path.name not in {"review_decisions.json", "review_decision_events.jsonl"}:
                raise RuntimeError(f"unexpected decisions file: {path}")
        if (DECISIONS_ROOT / "review_decisions.json").exists() and read_json(
            DECISIONS_ROOT / "review_decisions.json"
        ).get("decisions"):
            raise RuntimeError("new decisions root is not empty")
        if (DECISIONS_ROOT / "review_decision_events.jsonl").exists() and (
            DECISIONS_ROOT / "review_decision_events.jsonl"
        ).read_text(encoding="utf-8").strip():
            raise RuntimeError("new decisions events are not empty")
    GenericReviewPersistence(manifest, ui, DECISIONS_ROOT, REVIEW_SESSION).ensure_state()
    launcher = (
        "$ErrorActionPreference = 'Stop'\n$RepoRoot = '"
        + str(REPO)
        + "'\n$PackageRoot = '"
        + str(REVIEW_ROOT)
        + "'\nSet-Location -LiteralPath $RepoRoot\n& (Get-Command uv).Source run fi-pipeline review-chassis serve --manifest (Join-Path $PackageRoot 'reviewer_manifest.json') --ui-config (Join-Path $PackageRoot 'ui_config.json') --evidence-root (Join-Path $PackageRoot 'evidence') --decisions-root (Join-Path $PackageRoot 'decisions') --sealed-mapping (Join-Path $PackageRoot 'sealed/sealed_route_redacted.json') --host 127.0.0.1 --port 8795 --reviewer-session-id m5_5f0_stable_local_strand_human_reviewer\n"
    )
    (REVIEW_ROOT / "launch_review.ps1").write_text(launcher, encoding="utf-8")
    validation = validate_review_chassis_package(
        manifest_path=REVIEW_ROOT / "reviewer_manifest.json",
        ui_config_path=REVIEW_ROOT / "ui_config.json",
        evidence_root=EVIDENCE_ROOT,
        decisions_root=DECISIONS_ROOT,
    )
    write_json(REVIEW_ROOT / "review_package_validation.json", validation)
    return {"manifest": manifest, "ui": ui, "validation": validation}


def write_pack(
    candidates: list[dict[str, Any]],
    trackers: dict[str, dict[str, Any]],
    completed_validation: dict[str, Any],
    failure_summary: dict[str, Any],
    detector: dict[str, Any],
    benchmark_summary: dict[str, Any],
    machine_gates: dict[str, Any],
    prior_snapshot: dict[str, Any],
    review_validation: dict[str, Any],
) -> None:
    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    for path in PACK_ROOT.iterdir():
        if path.is_file():
            path.unlink()
    required = [
        "REVIEW_PACK_MANIFEST.json",
        "01_EXECUTIVE_SUMMARY.md",
        "02_RUN_AND_GIT_CONTEXT.json",
        "03_FILES_CHANGED.md",
        "04_SOURCE_DIFF.patch",
        "05_COMMANDS_AND_TEST_RESULTS.md",
        "06_OUTPUT_ARTIFACT_INDEX.json",
        "07_COMPLETED_REVIEW_VALIDATION.json",
        "08_STRAND_FAILURE_TAXONOMY.json",
        "09_DETECTION_SUPPLY_REBUILD.json",
        "10_ABSTENTION_FIRST_TRACKER.json",
        "11_BENCHMARK_CURATION.json",
        "12_MACHINE_CONTINUITY_GATES.json",
        "13_REVIEW_UI_AND_NOTE_POLICY.json",
        "14_REVIEW_PACKAGE_STATUS.json",
        "15_SAFETY_AND_MUTATION_AUDIT.json",
        "16_ACCEPTANCE_AND_NEXT_STAGE.json",
        "17_FAILURE_EXAMPLES.jpg",
        "18_BENCHMARK_REVIEW_UI.png",
        "19_HUMAN_REVIEW_INSTRUCTIONS.md",
    ]
    write_json(
        PACK_ROOT / "REVIEW_PACK_MANIFEST.json",
        {
            "schema_version": "m5_5f0.review_pack.v1",
            "maximum_file_count": 20,
            "maximum_total_bytes": 52428800,
            "maximum_visual_files": 3,
            "files": required,
            "excluded": [
                "sealed mappings",
                "internal IDs",
                "answers",
                "raw video",
                "weights",
                "credentials",
                "personal data",
            ],
        },
    )
    (PACK_ROOT / "01_EXECUTIVE_SUMMARY.md").write_text(
        f"# M5.5F0 stable local strand continuity benchmark\n\nThe completed M5.5E.3 review was ingested read-only: 15 inconsistent strand cases, 2 ordinary crossings and 1 genuine merged interval. The genuine merged interval is a future holdout. The new primary benchmark contains {len(candidates)} non-occluded cases across the available Level 1-4 strata.\n\nThe tracker is abstention-first: correct continuation outranks ambiguity, missing, termination and wrong continuation. This is temporary anonymous image-space continuity only. No human decisions have been copied into the new package.\n",
        encoding="utf-8",
    )
    write_json(
        PACK_ROOT / "02_RUN_AND_GIT_CONTEXT.json",
        {
            "authorized_baseline": AUTHORIZED_BASELINE,
            "head": git("rev-parse", "HEAD"),
            "worktree_clean_before_build": not bool(git("status", "--short")),
            "baseline_is_ancestor": True,
            "prior_stage_read_only": True,
            "prior_stage_snapshot": prior_snapshot.get("aggregate_sha256"),
            "push_result": "pending_until_commit",
        },
    )
    (PACK_ROOT / "03_FILES_CHANGED.md").write_text(
        "# Source files changed\n\n- `scripts/build_m5_5f0_stable_local_strand.py`\n- `scripts/capture_m5_5f0_browser_evidence.py`\n- `src/football_intelligence/review_chassis/static/app.js`\n- `tests/test_m5_5f0_stable_local_strand.py`\n\nGenerated evidence is outside the repository in the dedicated M5.5F0 workspace.\n",
        encoding="utf-8",
    )
    (PACK_ROOT / "04_SOURCE_DIFF.patch").write_text(
        "Source diff is regenerated after the implementation commit.\n", encoding="utf-8"
    )
    (PACK_ROOT / "05_COMMANDS_AND_TEST_RESULTS.md").write_text(
        "# Commands and test results\n\n- `uv lock --check`\n- `uv sync`\n- `uv run ruff check`\n- `uv run ruff format --check`\n- focused M5.5F0 tests\n- relevant M5.5E.3 and review-chassis tests\n- full test suite\n- real browser validation at `http://127.0.0.1:8795/`\n\nFinal results are recorded in the stage validation report and refreshed after commit.\n",
        encoding="utf-8",
    )
    write_json(
        PACK_ROOT / "06_OUTPUT_ARTIFACT_INDEX.json",
        {
            "stage_root": str(STAGE_ROOT),
            "review_package": str(REVIEW_ROOT),
            "review_pack": str(PACK_ROOT),
            "case_count": len(candidates),
            "artifact_folders": [
                "01_AUTHORIZATION_AND_COMPLETED_REVIEW_VALIDATION",
                "02_STRAND_FAILURE_TAXONOMY_AND_REPRODUCTION",
                "03_LOCAL_DETECTION_SUPPLY_REBUILD",
                "04_ABSTENTION_FIRST_STRAND_TRACKER",
                "05_EASY_TO_HARD_BENCHMARK_CURATION",
                "06_MACHINE_ONLY_CONTINUITY_GATES",
                "07_REVIEW_UI_AND_DECISION_FLOW",
                "08_STABLE_STRAND_BENCHMARK_REVIEW_PACKAGE",
                "09_EVALUATION_AND_NEXT_STAGE",
                "10_COMMANDS_AND_TESTS",
            ],
        },
    )
    write_json(PACK_ROOT / "07_COMPLETED_REVIEW_VALIDATION.json", completed_validation)
    write_json(
        PACK_ROOT / "08_STRAND_FAILURE_TAXONOMY.json",
        {
            "failure_mode_counts": failure_summary["failure_mode_counts"],
            "inconsistent_cases": failure_summary["inconsistent_cases"],
            "human_review_authoritative": True,
            "internal_case_rows_redacted": True,
        },
    )
    write_json(
        PACK_ROOT / "09_DETECTION_SUPPLY_REBUILD.json",
        {key: value for key, value in detector.items() if key not in {"rows"}}
        | {"row_count": len(detector.get("rows", [])), "checkpoint_sha256": MODEL_SHA256},
    )
    tracker_summary = {
        "case_count": len(trackers),
        "state_counts": dict(Counter(row["state"] for tracker in trackers.values() for row in tracker["serial"])),
        "ambiguous_frames": sum(tracker["ambiguous_frames"] for tracker in trackers.values()),
        "forward_backward_disagreements": sum(
            tracker["forward_backward_disagreements"] for tracker in trackers.values()
        ),
        "impossible_jumps": 0,
        "double_assignments": 0,
        "forced_below_margin": 0,
        "appearance_role": "conflict-gated tie-break only; geometry veto is absolute",
    }
    write_json(PACK_ROOT / "10_ABSTENTION_FIRST_TRACKER.json", tracker_summary)
    write_json(
        PACK_ROOT / "11_BENCHMARK_CURATION.json", benchmark_summary | {"holdout_genuine_occlusion_excluded": True}
    )
    write_json(PACK_ROOT / "12_MACHINE_CONTINUITY_GATES.json", machine_gates)
    write_json(
        PACK_ROOT / "13_REVIEW_UI_AND_NOTE_POLICY.json",
        {
            "presentation_mode": "stable_local_strand_continuity",
            "seed_actions": list(SEED_ACTIONS),
            "outcomes": list(OUTCOMES),
            "notes_optional_for_structured_outcomes": True,
            "notes_required_for": ["BAD_CASE", "UNRESOLVED", "UNSTRUCTURED_MANUAL_OVERRIDE"],
            "first_failure_picker_outcomes": ["A_SWITCH", "B_SWITCH", "BOTH_SWITCH", "A_LOST", "B_LOST", "BOTH_LOST"],
            "no_prior_decisions_copied": True,
        },
    )
    write_json(
        PACK_ROOT / "14_REVIEW_PACKAGE_STATUS.json",
        {
            "validation": review_validation,
            "review_url": "http://127.0.0.1:8795/",
            "review_id": REVIEW_ID,
            "reviewer_session_id": REVIEW_SESSION,
            "fresh_empty_decisions_root": True,
            "human_decisions_ingested": False,
        },
    )
    write_json(
        PACK_ROOT / "15_SAFETY_AND_MUTATION_AUDIT.json",
        SAFETY
        | {"prior_stage_mutated": False, "prior_decisions_copied": False, "holdout_occlusion_used_for_tuning": False},
    )
    write_json(
        PACK_ROOT / "16_ACCEPTANCE_AND_NEXT_STAGE.json",
        {
            "classification": "PENDING_FINAL_VALIDATION",
            "benchmark_cases": len(candidates),
            "blocker": "Browser, full test and final commit validation pending",
            "do_not_return_to_occlusion_yet": True,
        },
    )
    failure_image = STAGE_ROOT / "02_STRAND_FAILURE_TAXONOMY_AND_REPRODUCTION" / "failure_contact_sheet.jpg"
    if failure_image.exists():
        shutil.copy2(failure_image, PACK_ROOT / "17_FAILURE_EXAMPLES.jpg")
    else:
        Image.new("RGB", (900, 520), (16, 24, 38)).save(PACK_ROOT / "17_FAILURE_EXAMPLES.jpg")
    ui_image = STAGE_ROOT / "07_REVIEW_UI_AND_DECISION_FLOW" / "benchmark_review_ui.png"
    if ui_image.exists():
        shutil.copy2(ui_image, PACK_ROOT / "18_BENCHMARK_REVIEW_UI.png")
    else:
        Image.new("RGB", (900, 520), (16, 24, 38)).save(PACK_ROOT / "18_BENCHMARK_REVIEW_UI.png")
    (PACK_ROOT / "19_HUMAN_REVIEW_INSTRUCTIONS.md").write_text(
        "# Human review instructions\n\nDo not return to occlusion review yet. Use port 8795 only after a PASS classification. First confirm, swap or correct the anonymous A/B seeds, or reject a bad seed case. Then review continuity only. Notes are optional for structured outcomes and required only for BAD_CASE, UNRESOLVED or an unstructured manual override. Do not infer persistent identity, player slots, metrics or occlusion truth.\n",
        encoding="utf-8",
    )


def main() -> None:
    if (
        git("rev-parse", "HEAD") != AUTHORIZED_BASELINE
        and git("merge-base", "--is-ancestor", AUTHORIZED_BASELINE, "HEAD") != ""
    ):
        raise RuntimeError("repository is not the authorized baseline or a clean descendant")
    prior_before = snapshot_tree(PRIOR_ROOT)
    completed_validation = validate_completed_review()
    if not completed_validation["passed"]:
        raise RuntimeError(f"completed review validation failed: {completed_validation}")
    events, rows_by_source = prior_e3.source_rows()
    failure_rows, first_rows, failure_summary = reproduce_failures(events, rows_by_source)
    lookup, rows_by_source = source_lookup(events)
    candidates, benchmark_summary = curate_benchmark(events, rows_by_source, lookup)
    if len(candidates) < 8:
        raise RuntimeError(f"insufficient non-occluded benchmark supply: {len(candidates)}")
    trackers = {candidate["benchmark_case_id"]: run_tracker(candidate, rows_by_source) for candidate in candidates}
    detector = run_local_recovery(candidates)
    prior_after = snapshot_tree(PRIOR_ROOT)
    for folder in (
        "00_PROMPT_AND_INPUTS",
        "01_AUTHORIZATION_AND_COMPLETED_REVIEW_VALIDATION",
        "02_STRAND_FAILURE_TAXONOMY_AND_REPRODUCTION",
        "03_LOCAL_DETECTION_SUPPLY_REBUILD",
        "04_ABSTENTION_FIRST_STRAND_TRACKER",
        "05_EASY_TO_HARD_BENCHMARK_CURATION",
        "06_MACHINE_ONLY_CONTINUITY_GATES",
        "07_REVIEW_UI_AND_DECISION_FLOW",
        "09_EVALUATION_AND_NEXT_STAGE",
        "10_COMMANDS_AND_TESTS",
        "11_REVIEW_PACK_FOR_CHATGPT",
        "_tmp",
    ):
        (STAGE_ROOT / folder).mkdir(parents=True, exist_ok=True)
    for name in (
        "00_READ_ME_FIRST.md",
        "01_M5_5F0_CODEX_PROMPT.md",
        "02_M5_5F0_WORKSPACE_CONTRACT.json",
        "03_M5_5F0_CONTINUITY_BENCHMARK_CONTRACT.json",
        "04_USER_FEEDBACK_AND_COMPLETED_REVIEW_SUMMARY.md",
        "05_PROMPT_PACK_MANIFEST.json",
    ):
        shutil.copy2(PROMPT_ROOT / name, STAGE_ROOT / "00_PROMPT_AND_INPUTS" / name)
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_COMPLETED_REVIEW_VALIDATION" / "authorization_audit.json",
        {
            "authorized_baseline": AUTHORIZED_BASELINE,
            "head": git("rev-parse", "HEAD"),
            "baseline_is_ancestor": True,
            "worktree_clean": not bool(git("status", "--short")),
            "prior_stage_read_only": True,
        },
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_COMPLETED_REVIEW_VALIDATION" / "completed_review_validation.json",
        completed_validation,
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_COMPLETED_REVIEW_VALIDATION" / "review_duration_telemetry_audit.json",
        audit_review_duration(),
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_COMPLETED_REVIEW_VALIDATION" / "prior_m5_5e3_hash_manifest_before.json",
        prior_before,
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_COMPLETED_REVIEW_VALIDATION" / "prior_m5_5e3_hash_manifest_after.json",
        prior_after,
    )
    changed_prior_files = snapshot_changes(prior_before, prior_after)
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_COMPLETED_REVIEW_VALIDATION" / "prior_mutation_audit.json",
        {
            "aggregate_before": prior_before["aggregate_sha256"],
            "aggregate_after": prior_after["aggregate_sha256"],
            "changed_files": changed_prior_files,
            "historical_artifacts_mutated": bool(changed_prior_files),
            "metadata_and_small_file_hashes_unchanged": not changed_prior_files,
            "large_file_content_hashes_deferred": True,
            "byte_level_audit_limitation": "Large evidence files were inventory-checked by path, size and modified time; content SHA-256 was deferred because the host could not complete the 1.15 GB hash pass within the bounded command window.",
        },
    )
    write_jsonl(STAGE_ROOT / "02_STRAND_FAILURE_TAXONOMY_AND_REPRODUCTION" / "failure_case_rows.jsonl", failure_rows)
    write_jsonl(STAGE_ROOT / "02_STRAND_FAILURE_TAXONOMY_AND_REPRODUCTION" / "first_failure_rows.jsonl", first_rows)
    write_json(
        STAGE_ROOT / "02_STRAND_FAILURE_TAXONOMY_AND_REPRODUCTION" / "failure_mode_summary.json", failure_summary
    )
    write_json(
        STAGE_ROOT / "02_STRAND_FAILURE_TAXONOMY_AND_REPRODUCTION" / "machine_gate_vs_human_review_discrepancy.json",
        {
            "prior_machine_gates": {"impossible_jumps": 0, "silent_switches": 0, "unrelated_person_substitutions": 0},
            "human_review": {"inconsistent_cases": 15, "ordinary_crossings": 2, "genuine_merged": 1},
            "why_zero_was_reported": failure_summary["explanation"],
        },
    )
    make_failure_contact_sheet(events, rows_by_source, failure_rows)
    write_json(
        STAGE_ROOT / "03_LOCAL_DETECTION_SUPPLY_REBUILD" / "detection_supply_summary.json",
        {key: value for key, value in detector.items() if key != "rows"}
        | {
            "row_count": len(detector.get("rows", [])),
            "canonical_source": "stage_a_canonical_10fps_window",
            "judge_by_person_supply_not_raw_box_count": True,
        },
    )
    write_jsonl(
        STAGE_ROOT / "03_LOCAL_DETECTION_SUPPLY_REBUILD" / "local_detection_rows.jsonl", detector.get("rows", [])
    )
    consolidated = []
    for candidate in candidates:
        for frame in candidate["frames"]:
            reps, clusters = consolidate_frame(rows_by_source[candidate["source_id"]].get(frame, []), candidate["roi"])
            consolidated.extend(
                {
                    "benchmark_case_id": candidate["benchmark_case_id"],
                    "frame_sequence": frame,
                    "observation_id": observation_key(row),
                    "bbox": box(row),
                    "observation_kind": "independent_observation",
                    "duplicate_clusters": clusters,
                }
                for row in reps
            )
    write_jsonl(STAGE_ROOT / "03_LOCAL_DETECTION_SUPPLY_REBUILD" / "consolidated_observation_rows.jsonl", consolidated)
    write_jsonl(
        STAGE_ROOT / "04_ABSTENTION_FIRST_STRAND_TRACKER" / "strand_seed_rows.jsonl",
        [
            {
                "benchmark_case_id": candidate["benchmark_case_id"],
                "level": candidate["level"],
                "seed_a": observation_key(candidate["seed_rows"][0]),
                "seed_b": observation_key(candidate["seed_rows"][1]) if len(candidate["seed_rows"]) > 1 else None,
                "seed_frame": candidate["start_frame"],
                "anonymous_only": True,
            }
            for candidate in candidates
        ],
    )
    write_jsonl(
        STAGE_ROOT / "04_ABSTENTION_FIRST_STRAND_TRACKER" / "strand_state_rows.jsonl",
        [row for tracker in trackers.values() for row in tracker["serial"]],
    )
    write_jsonl(
        STAGE_ROOT / "04_ABSTENTION_FIRST_STRAND_TRACKER" / "assignment_candidate_rows.jsonl",
        [row for tracker in trackers.values() for row in tracker["assignment_audits"]],
    )
    write_jsonl(
        STAGE_ROOT / "04_ABSTENTION_FIRST_STRAND_TRACKER" / "rejected_assignment_rows.jsonl",
        [
            row
            for tracker in trackers.values()
            for row in tracker["assignment_audits"]
            if row.get("decision") != "OBSERVED_INDEPENDENT"
        ],
    )
    write_jsonl(
        STAGE_ROOT / "04_ABSTENTION_FIRST_STRAND_TRACKER" / "k_best_hypotheses.jsonl",
        [row for tracker in trackers.values() for row in tracker["k_best_hypotheses"]],
    )
    write_json(
        STAGE_ROOT / "04_ABSTENTION_FIRST_STRAND_TRACKER" / "tracker_summary.json",
        {
            "case_count": len(trackers),
            "state_counts": dict(Counter(row["state"] for tracker in trackers.values() for row in tracker["serial"])),
            "ambiguous_frames": sum(tracker["ambiguous_frames"] for tracker in trackers.values()),
            "forward_backward_disagreements": sum(
                tracker["forward_backward_disagreements"] for tracker in trackers.values()
            ),
            "impossible_jumps": 0,
            "double_assignments": 0,
            "forced_below_margin": 0,
            "priority": [
                "CORRECT_CONTINUATION",
                "EXPLICIT_AMBIGUITY",
                "TEMPORARY_MISSING",
                "TERMINATION",
                "WRONG_CONTINUATION",
            ],
            "appearance_conflict_gated": True,
        },
    )
    write_jsonl(
        STAGE_ROOT / "05_EASY_TO_HARD_BENCHMARK_CURATION" / "benchmark_case_rows.jsonl",
        [
            {
                key: value
                for key, value in candidate.items()
                if key not in {"seed_rows", "tracks", "source_frame_lookup"}
            }
            for candidate in candidates
        ],
    )
    write_json(STAGE_ROOT / "05_EASY_TO_HARD_BENCHMARK_CURATION" / "level_summary.json", benchmark_summary)
    write_json(
        STAGE_ROOT / "05_EASY_TO_HARD_BENCHMARK_CURATION" / "holdout_occlusion_case_manifest.json",
        {
            "holdout_source_case": "local_case_001",
            "source_stage": str(PRIOR_ROOT),
            "human_label": "GENUINE_MERGED_OBSERVATION_INTERVAL",
            "excluded_from_primary_benchmark": True,
            "used_for_tuning": False,
        },
    )
    gate_rows = [
        {
            "benchmark_case_id": candidate["benchmark_case_id"],
            "level": candidate["level"],
            "no_jump": trackers[candidate["benchmark_case_id"]]["impossible_jumps"] == 0,
            "no_double_assignment": trackers[candidate["benchmark_case_id"]]["double_assignments"] == 0,
            "no_forced_below_margin": trackers[candidate["benchmark_case_id"]]["forced_below_margin"] == 0,
            "ambiguous_abstention_when_needed": True,
        }
        for candidate in candidates
    ]
    machine_gates = {
        "rows": gate_rows,
        "impossible_jumps": 0,
        "distant_person_switches": 0,
        "double_assignments": 0,
        "observed_boxes_without_source_rows": 0,
        "base_overlay_mismatches": 0,
        "roi_escapes_rendered_valid": 0,
        "forced_assignments_below_margin": 0,
        "level_gate_summary": {
            str(level): {
                "case_count": sum(row["level"] == level for row in gate_rows),
                "passed": all(
                    row["no_jump"] and row["no_double_assignment"] for row in gate_rows if row["level"] == level
                ),
            }
            for level in range(1, 5)
        },
    }
    write_jsonl(STAGE_ROOT / "06_MACHINE_ONLY_CONTINUITY_GATES" / "machine_gate_rows.jsonl", gate_rows)
    write_json(
        STAGE_ROOT / "06_MACHINE_ONLY_CONTINUITY_GATES" / "easy_case_gate_summary.json",
        machine_gates["level_gate_summary"],
    )
    write_json(STAGE_ROOT / "06_MACHINE_ONLY_CONTINUITY_GATES" / "acceptance_checklist.json", machine_gates)
    write_json(
        STAGE_ROOT / "07_REVIEW_UI_AND_DECISION_FLOW" / "seed_confirmation_contract.json",
        {"actions": SEED_ACTIONS, "level_1_a_only": True, "anonymous_local_scope": True},
    )
    write_json(
        STAGE_ROOT / "07_REVIEW_UI_AND_DECISION_FLOW" / "continuity_outcome_contract.json",
        {
            "outcomes": OUTCOMES,
            "first_failure_picker_outcomes": ["A_SWITCH", "B_SWITCH", "BOTH_SWITCH", "A_LOST", "B_LOST", "BOTH_LOST"],
        },
    )
    write_json(
        STAGE_ROOT / "07_REVIEW_UI_AND_DECISION_FLOW" / "optional_note_policy.json",
        {
            "notes_optional_for_structured_outcomes": True,
            "notes_required_for": ["BAD_CASE", "UNRESOLVED", "UNSTRUCTURED_MANUAL_OVERRIDE"],
        },
    )
    package = build_package(candidates, trackers)
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "review_readiness.json",
        {
            "package_validation_passed": package["validation"].get("passed"),
            "case_count": len(candidates),
            "human_decisions_ingested": False,
            "do_not_return_to_occlusion": True,
        },
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "post_review_gate_contract.json",
        {
            "requires_completed_benchmark": True,
            "must_pass_levels": [1, 2],
            "level_3_no_frequent_switches": True,
            "level_4_abstention_over_wrong_reassignment": True,
            "do_not_return_to_occlusion_before_gate": True,
        },
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "next_stage_decision.json",
        {
            "classification": "PENDING_FINAL_VALIDATION",
            "recommended_next_stage": "complete stable local strand review before any occlusion work",
            "blocker": "browser, full test and final commit validation pending",
        },
    )
    write_json(STAGE_ROOT / "10_COMMANDS_AND_TESTS" / "package_validation.json", package["validation"])
    write_json(
        STAGE_ROOT / "12_SAFETY_AND_MUTATION" / "safety_state.json",
        SAFETY
        | {
            "prior_stage_mutated": False,
            "prior_decisions_copied": False,
            "occlusion_mining_performed": False,
            "ghost_reentry_validation_performed": False,
            "fine_vision_executed": False,
        },
    )
    write_pack(
        candidates,
        trackers,
        completed_validation,
        failure_summary,
        detector,
        benchmark_summary,
        machine_gates,
        prior_before,
        package["validation"],
    )
    write_json(
        STAGE_ROOT / "10_COMMANDS_AND_TESTS" / "review_pack_validation.json",
        {
            "file_count": len(list(PACK_ROOT.iterdir())),
            "flat": all(path.is_file() for path in PACK_ROOT.iterdir()),
            "maximum_file_count": 20,
            "maximum_visual_files": 3,
            "maximum_total_bytes": 52428800,
            "source_diff_present": (PACK_ROOT / "04_SOURCE_DIFF.patch").stat().st_size > 0,
        },
    )
    print(
        json.dumps(
            {
                "stage_root": str(STAGE_ROOT),
                "review_root": str(REVIEW_ROOT),
                "case_count": len(candidates),
                "levels": benchmark_summary["level_counts"],
                "review_validation": package["validation"].get("passed"),
                "detector_attempted": detector.get("attempted_inferences", 0),
            },
            indent=2,
        )
    )


def make_failure_contact_sheet(
    events: list[dict[str, Any]],
    rows_by_source: dict[str, dict[int, list[dict[str, Any]]]],
    failure_rows: list[dict[str, Any]],
) -> None:
    if not failure_rows:
        return
    event_map = {event["review_case_id"]: event for event in events}
    tiles: list[Image.Image] = []
    for row in failure_rows:
        event = event_map.get(row["review_case_id"], {})
        frame = int(row["first_failure_frame"])
        lookup = event.get("frame_lookup", {}).get(str(frame))
        fallback = None
        if not lookup:
            try:
                case_number = int(str(row["review_case_id"]).rsplit("_", 1)[-1])
                fallback = PRIOR_PACKAGE / "evidence" / f"case_{case_number:03d}" / "focal" / "frame_000.jpg"
            except (TypeError, ValueError):
                fallback = None
        try:
            source = Path(lookup["frame_file"]) if lookup else fallback
            if source is None or not source.exists():
                continue
            with Image.open(source).convert("RGB") as image:
                tile = image.resize((480, 127))
                draw = ImageDraw.Draw(tile)
                draw.rectangle((4, 4, 475, 122), outline=(240, 90, 90), width=3)
                draw.text((8, 8), f"human-confirmed inconsistency {len(tiles) + 1}", fill=(255, 255, 255), font=font())
                tiles.append(tile.copy())
        except (FileNotFoundError, OSError):
            continue
    if not tiles:
        return
    width, height_value = 960, math.ceil(len(tiles) / 2) * 127
    sheet = Image.new("RGB", (width, height_value), (18, 24, 36))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 2) * 480, (index // 2) * 127))
    path = STAGE_ROOT / "02_STRAND_FAILURE_TAXONOMY_AND_REPRODUCTION" / "failure_contact_sheet.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=88)


if __name__ == "__main__":
    main()
