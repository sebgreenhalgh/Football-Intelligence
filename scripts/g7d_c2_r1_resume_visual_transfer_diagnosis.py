"""Build the bounded G7D-C2 R1 targeted visual-transfer diagnosis."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from football_intelligence import g7d_c1_r1_novice_review as r1
from football_intelligence.g7d_c1_r8_latest_completion_receipt import (
    resolve_current_completion_receipt,
    resolve_latest_event_set,
)
from football_intelligence.g7d_c2_visual_transfer_diagnosis import (
    TARGETED_WARNING,
    box_metrics,
    candidate_flags,
    choose_next_stage,
    classification_metrics,
    grouped_flag_summary,
    polygon_location,
    summary_flags,
)

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
C1 = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_v1"
C1_PACKAGE = C1 / "02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE"
C2_STOP = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_C2_VISUAL_TRANSFER_DIAGNOSIS_FINALIZATION_v1"
B3 = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B3_FROZEN_CROSS_MATCH_REPLAY_v1"
B2C = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B2C_RESUME_FROZEN_128058_BASELINE_v1"
STAGE = (
    PROJECT
    / "experiments/football_observation_reasoner/part 6/G7D_C2_R1_RESUME_VISUAL_TRANSFER_DIAGNOSIS_FINALIZATION_v1"
)
EXPECTED_HEAD = "e0f003fabe3b0bca02ae68197b017257de4d5cd2"
RECEIPT_ID = "completion-r8-bbbaabc5fdbff19754baee53"
RECEIPT_SHA256 = "0a07908ead0845fcb82256a32bab94663e0a53d5d5835ca944c46b601a8ef0fc"
EVENT_SET_DIGEST = "bbbaabc5fdbff19754baee53dce8342a91f49c92967bf319398b1ba30e7b4e08"
LATEST_S01 = "8e145c713516fb829dc8f32bfe0ecea2"
SUPERSEDED_S01 = "d6cff7afef94bad7d411d659dacb0e2d"
POLYGON_HASHES = {
    "118575": "fbd7f3a473acc197b4c893d90bbaa4c5d484d1e883e8df1ac4601daf4396dec1",
    "117092": "92ca8040eedd3b0ec0bb685648691f0c314d8527f3fa8f2db1823b4461e4b338",
}
FLAG_NAMES = [
    "contains_any_person",
    "contains_single_person",
    "contains_multiple_people",
    "is_duplicate",
    "is_background_or_object",
    "is_relevant_active_population",
    "box_is_useful_single_person",
    "box_is_strict_good_single_person",
    "box_quality_issue",
    "has_occlusion",
]
HEAD_LABELS = {
    "candidate_state": [
        "CLEAN_INDEPENDENT_PERSON",
        "DUPLICATE_OF_PERSON",
        "MERGED_MULTIPLE_PEOPLE",
        "PARTIAL_PERSON",
        "BACKGROUND",
        "AMBIGUOUS_UNRESOLVED",
    ],
    "role": ["OUTFIELD_PLAYER", "GOALKEEPER", "REFEREE", "OTHER_MATCH_OFFICIAL", "STAFF_OR_SPECTATOR", "UNKNOWN_ROLE"],
    "team": ["TEAM_1", "TEAM_2", "NO_TEAM", "UNKNOWN_TEAM"],
    "participation": [
        "ACTIVE_ON_PITCH",
        "OFF_PITCH_SUBSTITUTE_OR_WARMING",
        "OFF_PITCH_NON_PLAYER",
        "UNKNOWN_PARTICIPATION",
    ],
    "pitch": ["ON_PITCH", "OFF_PITCH", "BOUNDARY_UNCERTAIN", "UNKNOWN_PITCH_STATE"],
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def sha256(path: Path) -> str:
    return r1.sha256_file(path)


def relative(path: Path) -> str:
    return path.relative_to(PROJECT).as_posix()


def artifact(path: Path) -> dict[str, Any]:
    return {"project_relative_path": relative(path), "byte_size": path.stat().st_size, "sha256": sha256(path)}


def manifest(directory: Path, manifest_name: str) -> None:
    paths = sorted(path for path in directory.iterdir() if path.is_file() and path.name != manifest_name)
    write_json(
        directory / manifest_name,
        {
            "schema_version": "football_intelligence.g7d_c2_r1.directory_manifest.v1",
            "self_hash_omitted": True,
            "file_count": len(paths),
            "files": [artifact(path) for path in paths],
        },
    )


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def continuation_gate() -> dict[str, Any]:
    if git("rev-parse", "HEAD") != EXPECTED_HEAD or git("branch", "--show-current") != "main":
        raise RuntimeError("FAIL_BASELINE_OR_WORKTREE")
    stop_files = sorted(path.name for path in C2_STOP.iterdir() if path.is_file())
    if stop_files != ["00_STAGE_STOP.md", "artifact_manifest.json", "human_event_chain_failure.json"]:
        raise RuntimeError("FAIL_G7D_C2_R1_CONTINUATION_PROVENANCE")
    stop = (C2_STOP / "00_STAGE_STOP.md").read_text(encoding="utf-8")
    failure = load_json(C2_STOP / "human_event_chain_failure.json")
    if "FAIL_G7D_C2_HUMAN_EVENT_CHAIN" not in stop or failure.get("classification") != "FAIL_G7D_C2_HUMAN_EVENT_CHAIN":
        raise RuntimeError("FAIL_G7D_C2_R1_CONTINUATION_PROVENANCE")
    split = load_json(PROJECT / "datasets/soccertrack_v2/splits/split_v1/split_manifest.json")
    required_train = {"117092", "118575", "128058"}
    if split.get("status") != "FROZEN_HUMAN_APPROVED" or split.get("frozen") is not True:
        raise RuntimeError("FAIL_G7D_C2_R1_CONTINUATION_PROVENANCE")
    if not required_train.issubset(set(split["membership"]["TRAIN_DEVELOPMENT"])):
        raise RuntimeError("FAIL_G7D_C2_R1_CONTINUATION_PROVENANCE")
    return {
        "classification": "PASS_G7D_C2_R1_CONTINUATION_PROVENANCE",
        "repository_head": EXPECTED_HEAD,
        "branch": "main",
        "original_c2_classification": failure["classification"],
        "original_c2_files": stop_files,
        "partial_analysis_outputs_found": False,
        "split_status": split["status"],
        "split_frozen": split["frozen"],
        "required_train_development_matches": sorted(required_train),
    }


def provenance_gate() -> tuple[dict[str, Any], Any, Path, dict[str, Any]]:
    latest = resolve_latest_event_set(C1_PACKAGE)
    receipt_path, receipt = resolve_current_completion_receipt(C1_PACKAGE)
    if (
        latest.digest != EVENT_SET_DIGEST
        or receipt["completion_receipt_id"] != RECEIPT_ID
        or sha256(receipt_path) != RECEIPT_SHA256
        or len(latest.candidate_events) != 192
        or len(latest.scene_events) != 24
        or receipt.get("all_cases_complete") is not True
    ):
        raise RuntimeError("FAIL_G7D_C2_R1_HUMAN_EVENT_CHAIN")
    ids = {row["event_id"] for row in [*latest.candidate_events, *latest.scene_events]}
    if LATEST_S01 not in ids or SUPERSEDED_S01 in ids:
        raise RuntimeError("FAIL_G7D_C2_R1_HUMAN_EVENT_CHAIN")
    b3_input = load_json(B3 / "01_INPUT_CLOSURE/input_validation.json")
    execution = load_json(B3 / "03_REPLAY_RUNTIME/execution_receipt.json")
    shortlist = load_json(B3 / "05_RISK_SHORTLIST/diagnostic_shortlist.json")
    if (
        b3_input.get("total_frame_count") != 64
        or execution.get("successful_frame_count") != 64
        or execution.get("aggregation") != "NONE"
        or execution.get("fold_order") != [0, 1, 2, 3, 4]
        or shortlist.get("total_scene_count") != 24
        or shortlist.get("per_match_count") != {"117092": 12, "118575": 12}
    ):
        raise RuntimeError("FAIL_G7D_C2_R1_CONTINUATION_PROVENANCE")
    for match_id, expected in POLYGON_HASHES.items():
        path = PROJECT / f"matches/{match_id}/calibration/pitch_polygon_v1/pitch_polygon.json"
        if sha256(path) != expected or b3_input["matches"][match_id]["polygon"]["sha256"] != expected:
            raise RuntimeError("FAIL_G7D_C2_R1_PITCH_ANALYSIS")
    return (
        {
            "classification": "PASS_G7D_C2_R1_FROZEN_PROVENANCE",
            "receipt": artifact(receipt_path),
            "completion_receipt_id": RECEIPT_ID,
            "latest_event_set_digest": latest.digest,
            "latest_counts": {"candidate": 192, "scene": 24, "total": 216},
            "latest_s01t01_event_id": LATEST_S01,
            "superseded_s01t01_excluded": True,
            "all_acknowledgements_valid": True,
            "b3_frame_count": 64,
            "b3_scene_count": 24,
            "targets_per_scene": 8,
            "target_count": 192,
            "fold_order": [0, 1, 2, 3, 4],
            "aggregation": "NONE",
            "runtime_manifest_sha256": b3_input["runtime_manifest_sha256"],
            "baseline_contract": b3_input["baseline_contract"],
            "polygon_hashes": POLYGON_HASHES,
        },
        latest,
        receipt_path,
        receipt,
    )


def canonical_candidate_state(value: str) -> str:
    return {
        "CLEAN_SINGLE_PERSON": "CLEAN_INDEPENDENT_PERSON",
        "LOOSE_BACKGROUND_AROUND_PERSON": "CLEAN_INDEPENDENT_PERSON",
        "PARTIAL_SINGLE_PERSON": "PARTIAL_PERSON",
        "MERGES_MULTIPLE_PEOPLE": "MERGED_MULTIPLE_PEOPLE",
        "DUPLICATE_OF_ANOTHER_CANDIDATE": "DUPLICATE_OF_PERSON",
        "NO_PERSON_BACKGROUND_OR_OBJECT": "BACKGROUND",
        "UNCERTAIN": "AMBIGUOUS_UNRESOLVED",
    }[value]


def head_truth(decision: Mapping[str, Any], head: str) -> str | None:
    if head == "candidate_state":
        return canonical_candidate_state(decision["proposal_validity"])
    if head == "role":
        return {
            "OTHER_OFFICIAL": "OTHER_MATCH_OFFICIAL",
            "UNKNOWN_PERSON_ROLE": None,
            "NOT_A_PERSON": None,
        }.get(decision["role"], decision["role"])
    if head == "team":
        return None if decision["team"] in {"NOT_APPLICABLE", "UNKNOWN_TEAM"} else decision["team"]
    if head == "participation":
        return {
            "ACTIVE": "ACTIVE_ON_PITCH",
            "WARMING_UP": "OFF_PITCH_SUBSTITUTE_OR_WARMING",
            "NON_PLAYER": "OFF_PITCH_NON_PLAYER",
            "UNKNOWN": None,
            "NOT_APPLICABLE": None,
        }[decision["participation"]]
    if head == "pitch":
        return {"ON_PITCH": "ON_PITCH", "OFF_PITCH": "OFF_PITCH", "BOUNDARY": "BOUNDARY_UNCERTAIN"}.get(
            decision["pitch_state"]
        )
    raise KeyError(head)


def normalize_human(
    latest: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    focus = load_json(C1 / "01_REVIEW_INPUTS/focus_candidate_manifest.json")
    shortlist = load_json(B3 / "05_RISK_SHORTLIST/diagnostic_shortlist.json")
    scene_category = {row["frame_sha256"]: row["primary_quota"] for row in shortlist["scenes"]}
    target_meta = {}
    scene_meta = {}
    for selection in focus["selections"]:
        scene_meta[selection["scene_id"]] = selection
        for target in selection["targets"]:
            target_meta[target["target_id"]] = {
                **target,
                "scene_id": selection["scene_id"],
                "frame_sha256": selection["frame_sha256"],
            }
    b3_rows = load_jsonl(B3 / "03_REPLAY_RUNTIME/foldwise_candidate_records.jsonl")
    b3_by_key = {(row["frame_sha256"], row["candidate_local_id"]): row for row in b3_rows}
    cases = load_json(C1_PACKAGE / "review_cases.json")["cases"]
    case_by_scene = {case["scene_id"]: case for case in cases}
    conditions = {
        match_id: load_json(PROJECT / f"matches/{match_id}/calibration/match_setup.json")["conditions"]
        for match_id in POLYGON_HASHES
    }
    candidates = []
    for reference in latest.candidate_events:
        event = load_json(C1_PACKAGE / reference["event_relative_path"])
        target_id = reference["identity"]
        meta = target_meta[target_id]
        case = case_by_scene[meta["scene_id"]]
        b3_row = b3_by_key[(meta["frame_sha256"], meta["candidate_local_id"])]
        historical_record_hash = hashlib.sha256(
            json.dumps(b3_row, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if historical_record_hash != meta["candidate_record_sha256"]:
            raise RuntimeError("FAIL_G7D_C2_R1_FOLDWISE_JOIN")
        decision = event["payload"]["decision"]
        flags = candidate_flags(decision)
        candidates.append(
            {
                "target_id": target_id,
                "event_id": event["event_id"],
                "event_sha256": reference["event_sha256"],
                "scene_id": case["scene_id"],
                "frame_id": case["frame_id"],
                "frame_sha256": case["frame_sha256"],
                "candidate_local_id": meta["candidate_local_id"],
                "candidate_record_sha256": meta["candidate_record_sha256"],
                "match": case["match_id"],
                "half": case["half"],
                "lighting": conditions[case["match_id"]]["lighting"],
                "match_condition": "DAYLIGHT"
                if conditions[case["match_id"]]["lighting"] == "DAYLIGHT"
                else "NIGHT_LOW_LIGHT",
                "target_slot": meta["slot"],
                "scene_category": scene_category[case["frame_sha256"]],
                "perspective_band": b3_row["perspective_band"],
                "source_box_xyxy": b3_row["source_box_xyxy"],
                "approximate_footpoint_xy": b3_row["approximate_footpoint_xy"],
                "canonical_decision": decision,
                "analysis_flags": flags,
                "fold_outputs": b3_row["fold_outputs"],
            }
        )
    scenes = []
    marks = []
    for reference in latest.scene_events:
        event = load_json(C1_PACKAGE / reference["event_relative_path"])
        scene_id = reference["identity"]
        case = case_by_scene[scene_id]
        review = event["payload"]["review"]
        row = {
            "scene_id": scene_id,
            "event_id": event["event_id"],
            "event_sha256": reference["event_sha256"],
            "frame_id": case["frame_id"],
            "frame_sha256": case["frame_sha256"],
            "match": case["match_id"],
            "half": case["half"],
            "lighting": conditions[case["match_id"]]["lighting"],
            "scene_category": scene_category[case["frame_sha256"]],
            "canonical_review": review,
        }
        scenes.append(row)
        for index, mark in enumerate(review["missed_people_source_xy"], 1):
            marks.append(
                {"scene_id": scene_id, "mark_index": index, "match": case["match_id"], "half": case["half"], **mark}
            )
    candidates.sort(key=lambda row: row["target_id"])
    scenes.sort(key=lambda row: row["scene_id"])
    validation = {
        "classification": "PASS_G7D_C2_R1_CANONICAL_LABELS",
        "candidate_count": len(candidates),
        "scene_count": len(scenes),
        "missed_person_mark_count": len(marks),
        "canonical_values_copied_exactly": True,
        "derived_flags_only": FLAG_NAMES,
        "target_manifest_sha256": sha256(C1 / "01_REVIEW_INPUTS/focus_candidate_manifest.json"),
        "fold_outputs_per_candidate": sorted({len(row["fold_outputs"]) for row in candidates}),
        "warning": TARGETED_WARNING,
    }
    if len(candidates) != 192 or len(scenes) != 24 or validation["fold_outputs_per_candidate"] != [5]:
        raise RuntimeError("FAIL_G7D_C2_R1_CANONICAL_LABELS")
    return candidates, scenes, marks, validation


def counts(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def candidate_diagnosis(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    dimensions = ["match", "half", "match_condition", "target_slot", "scene_category", "perspective_band"]
    state = {
        "warning": TARGETED_WARNING,
        "overall": summary_flags(candidates, FLAG_NAMES),
        "by_slice": grouped_flag_summary(candidates, dimensions, FLAG_NAMES),
        "canonical_proposal_validity": counts(row["canonical_decision"]["proposal_validity"] for row in candidates),
    }
    box = {
        "warning": TARGETED_WARNING,
        "overall": counts(row["canonical_decision"]["box_quality"] for row in candidates),
        "by_match": {
            match: counts(row["canonical_decision"]["box_quality"] for row in candidates if row["match"] == match)
            for match in sorted({row["match"] for row in candidates})
        },
    }
    semantic = {"warning": TARGETED_WARNING}
    semantic.update(
        {
            field: counts(row["canonical_decision"][field] for row in candidates)
            for field in ("role", "team", "participation")
        }
    )
    pitch = {
        "warning": TARGETED_WARNING,
        "pitch_state": counts(row["canonical_decision"]["pitch_state"] for row in candidates),
    }
    occlusion = {
        "warning": TARGETED_WARNING,
        "occlusion": counts(row["canonical_decision"]["occlusion"] for row in candidates),
    }
    certainty = {
        "warning": TARGETED_WARNING,
        "certainty": counts(row["canonical_decision"]["certainty"] for row in candidates),
    }
    slots = {
        "warning": TARGETED_WARNING,
        "slots": {
            slot: summary_flags([row for row in candidates if row["target_slot"] == slot], FLAG_NAMES)
            for slot in sorted({row["target_slot"] for row in candidates})
        },
    }
    conditions = {
        "warning": TARGETED_WARNING,
        "conditions": {
            condition: summary_flags([row for row in candidates if row["match_condition"] == condition], FLAG_NAMES)
            for condition in sorted({row["match_condition"] for row in candidates})
        },
    }
    return {
        "candidate_state_summary.json": state,
        "box_quality_summary.json": box,
        "role_team_participation_summary.json": semantic,
        "pitch_state_summary.json": pitch,
        "occlusion_summary.json": occlusion,
        "certainty_summary.json": certainty,
        "target_slot_summary.json": slots,
        "match_condition_summary.json": conditions,
    }


def pitch_diagnosis(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    polygons = {
        match: load_json(PROJECT / f"matches/{match}/calibration/pitch_polygon_v1/pitch_polygon.json")
        for match in POLYGON_HASHES
    }
    evaluated = []
    for row in candidates:
        polygon = polygons[row["match"]]
        geometry = polygon_location(
            row["approximate_footpoint_xy"], polygon["vertices_source_xy"], polygon["source_height"]
        )
        evaluated.append({**row, "pitch_geometry": geometry})
    policies = {}
    for policy in (
        "A_STRICT_INSIDE",
        "B_INSIDE_OR_BOUNDARY_BAND",
        "C_FAR_OUTSIDE_WITH_RELEVANT_EXCEPTIONS",
        "D_POPULATION_AWARE_OUTSIDE",
    ):
        rows = []
        for row in evaluated:
            geometry = row["pitch_geometry"]
            useful_relevant = (
                row["analysis_flags"]["box_is_useful_single_person"]
                and row["analysis_flags"]["is_relevant_active_population"]
            )
            clutter = (
                row["analysis_flags"]["is_background_or_object"]
                or row["canonical_decision"]["role"] == "STAFF_OR_SPECTATOR"
                or row["canonical_decision"]["participation"] == "NON_PLAYER"
            )
            if policy == "A_STRICT_INSIDE":
                retained = geometry["inside_polygon"]
            elif policy == "B_INSIDE_OR_BOUNDARY_BAND":
                retained = geometry["inside_polygon"] or geometry["geometry_band"] == "NEAR_BOUNDARY"
            elif policy == "C_FAR_OUTSIDE_WITH_RELEVANT_EXCEPTIONS":
                retained = geometry["geometry_band"] != "FAR_OUTSIDE_POLYGON" or useful_relevant
            else:
                retained = geometry["inside_polygon"] or not clutter or useful_relevant
            rows.append({"retained": retained, "clutter": clutter, "useful_relevant": useful_relevant, "row": row})
        clutter_support = sum(item["clutter"] for item in rows)
        relevant_support = sum(item["useful_relevant"] for item in rows)
        removed_clutter = sum(item["clutter"] and not item["retained"] for item in rows)
        lost_relevant = sum(item["useful_relevant"] and not item["retained"] for item in rows)
        policies[policy] = {
            "reviewed_support": len(rows),
            "reviewed_clutter_support": clutter_support,
            "reviewed_clutter_removed": removed_clutter,
            "reviewed_clutter_removed_rate": removed_clutter / clutter_support if clutter_support else None,
            "reviewed_useful_relevant_support": relevant_support,
            "reviewed_useful_relevant_retained": relevant_support - lost_relevant,
            "reviewed_useful_relevant_lost": lost_relevant,
            "reviewed_useful_relevant_loss_rate": lost_relevant / relevant_support if relevant_support else None,
            "boundary_exceptions": sum(
                item["retained"] and item["row"]["pitch_geometry"]["geometry_band"] == "NEAR_BOUNDARY" for item in rows
            ),
            "match_results": {
                match: {
                    "support": sum(item["row"]["match"] == match for item in rows),
                    "clutter_removed": sum(
                        item["row"]["match"] == match and item["clutter"] and not item["retained"] for item in rows
                    ),
                    "useful_relevant_lost": sum(
                        item["row"]["match"] == match and item["useful_relevant"] and not item["retained"]
                        for item in rows
                    ),
                }
                for match in sorted(POLYGON_HASHES)
            },
            "warning": TARGETED_WARNING,
        }
    off_pitch = {
        "warning": TARGETED_WARNING,
        "human_pitch_state": counts(row["canonical_decision"]["pitch_state"] for row in evaluated),
        "geometry_band": counts(row["pitch_geometry"]["geometry_band"] for row in evaluated),
        "staff_or_spectator": sum(row["canonical_decision"]["role"] == "STAFF_OR_SPECTATOR" for row in evaluated),
        "non_player": sum(row["canonical_decision"]["participation"] == "NON_PLAYER" for row in evaluated),
        "background_or_object": sum(row["analysis_flags"]["is_background_or_object"] for row in evaluated),
        "relevant_active_population": sum(row["analysis_flags"]["is_relevant_active_population"] for row in evaluated),
    }
    scope = {
        "warning": TARGETED_WARNING,
        "reviewed_clutter_union": sum(
            row["analysis_flags"]["is_background_or_object"]
            or row["canonical_decision"]["role"] == "STAFF_OR_SPECTATOR"
            or row["canonical_decision"]["participation"] == "NON_PLAYER"
            for row in evaluated
        ),
        "reviewed_useful_relevant": sum(
            row["analysis_flags"]["box_is_useful_single_person"]
            and row["analysis_flags"]["is_relevant_active_population"]
            for row in evaluated
        ),
        "role": counts(row["canonical_decision"]["role"] for row in evaluated),
        "participation": counts(row["canonical_decision"]["participation"] for row in evaluated),
    }
    boundary = {
        "warning": TARGETED_WARNING,
        "boundary_band_definition": "distance <= 1.5% of source height",
        "far_outside_definition": "outside and distance >= 5% of source height",
        "human_boundary_count": sum(row["canonical_decision"]["pitch_state"] == "BOUNDARY" for row in evaluated),
        "relevant_people_outside_geometry": sum(
            not row["pitch_geometry"]["inside_polygon"] and row["analysis_flags"]["is_relevant_active_population"]
            for row in evaluated
        ),
        "examples": [
            row["target_id"]
            for row in evaluated
            if not row["pitch_geometry"]["inside_polygon"] and row["analysis_flags"]["is_relevant_active_population"]
        ][:20],
    }
    return {"evaluated": evaluated, "off_pitch": off_pitch, "scope": scope, "policies": policies, "boundary": boundary}


def fold_person_support(row: Mapping[str, Any]) -> float:
    values = []
    for fold in row["fold_outputs"]:
        head = next(item for item in fold["head_outputs"] if item["head_name"] == "candidate_state")
        indexes = [index for index, label in enumerate(head["class_order"]) if label != "BACKGROUND"]
        values.append(sum(head["calibrated_probabilities"][index] for index in indexes))
    return sum(values) / len(values)


def nested_diagnosis(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    b3_rows = load_jsonl(B3 / "03_REPLAY_RUNTIME/foldwise_candidate_records.jsonl")
    shortlist = load_json(B3 / "05_RISK_SHORTLIST/diagnostic_shortlist.json")
    selected_hashes = {row["frame_sha256"]: row for row in shortlist["scenes"]}
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in b3_rows:
        if row["frame_sha256"] in selected_hashes:
            by_frame[row["frame_sha256"]].append(row)
    human = {(row["frame_sha256"], row["candidate_local_id"]): row for row in candidates}
    pairs = []
    suppressed_candidates = set()
    for frame_hash, rows in sorted(by_frame.items()):
        scene = selected_hashes[frame_hash]
        for inner in rows:
            inner_area = (inner["source_box_xyxy"][2] - inner["source_box_xyxy"][0]) * (
                inner["source_box_xyxy"][3] - inner["source_box_xyxy"][1]
            )
            for outer in rows:
                if inner is outer:
                    continue
                outer_area = (outer["source_box_xyxy"][2] - outer["source_box_xyxy"][0]) * (
                    outer["source_box_xyxy"][3] - outer["source_box_xyxy"][1]
                )
                if inner_area > outer_area:
                    continue
                metrics = box_metrics(inner["source_box_xyxy"], outer["source_box_xyxy"])
                if metrics["intersection_over_inner_area"] < 0.90:
                    continue
                inner_human = human.get((frame_hash, inner["candidate_local_id"]))
                outer_human = human.get((frame_hash, outer["candidate_local_id"]))
                if inner_human and inner_human["analysis_flags"]["is_background_or_object"]:
                    classification = "BALL_EQUIPMENT_BACKGROUND_OR_FRAGMENT"
                elif inner_human and inner_human["analysis_flags"]["is_duplicate"]:
                    classification = "DUPLICATE_OF_SAME_PERSON"
                elif inner_human and inner_human["analysis_flags"]["box_is_useful_single_person"]:
                    classification = (
                        "CORRECT_INNER_PERSON_INSIDE_BAD_OUTER"
                        if outer_human
                        and outer_human["canonical_decision"]["box_quality"] in {"MERGED_BOX", "MISLOCALIZED"}
                        else "LEGITIMATE_SMALLER_PERSON"
                    )
                else:
                    classification = "UNCERTAIN"
                inner_support = fold_person_support(inner)
                outer_support = fold_person_support(outer)
                protected = bool(
                    inner_human
                    and (
                        inner_human["analysis_flags"]["box_is_useful_single_person"]
                        or inner_human["analysis_flags"]["is_relevant_active_population"]
                    )
                ) or bool(
                    outer_human and outer_human["canonical_decision"]["box_quality"] in {"MERGED_BOX", "MISLOCALIZED"}
                )
                evidence = bool(
                    inner_human
                    and (
                        inner_human["analysis_flags"]["is_background_or_object"]
                        or inner_human["analysis_flags"]["is_duplicate"]
                    )
                ) or (
                    inner_support < 0.20
                    and metrics["inner_bottom_region"] >= 0.60
                    and metrics["centre_distance_outer_height"] <= 0.75
                )
                suppress = bool(
                    metrics["intersection_over_inner_area"] >= 0.95
                    and metrics["inner_outer_area_ratio"] <= 0.35
                    and outer_support >= 0.45
                    and evidence
                    and not protected
                )
                if suppress:
                    suppressed_candidates.add((frame_hash, inner["candidate_local_id"]))
                pairs.append(
                    {
                        "scene_frame_id": scene["frame_id"],
                        "frame_sha256": frame_hash,
                        "match_id": scene["match_id"],
                        "inner_candidate_local_id": inner["candidate_local_id"],
                        "outer_candidate_local_id": outer["candidate_local_id"],
                        **metrics,
                        "inner_fold_person_support": inner_support,
                        "outer_fold_person_support": outer_support,
                        "inner_human_target_id": inner_human["target_id"] if inner_human else None,
                        "outer_human_target_id": outer_human["target_id"] if outer_human else None,
                        "reviewed_classification": classification,
                        "protected_exception": protected,
                        "conservative_suppression_simulated": suppress,
                    }
                )
    reviewed_nested = [pair for pair in pairs if pair["inner_human_target_id"]]
    reviewed_suppressed = {
        pair["inner_human_target_id"] for pair in reviewed_nested if pair["conservative_suppression_simulated"]
    }
    useful_reviewed = {row["target_id"] for row in candidates if row["analysis_flags"]["box_is_useful_single_person"]}
    summary = {
        "warning": TARGETED_WARNING,
        "shortlisted_scene_candidate_count": sum(len(rows) for rows in by_frame.values()),
        "containment_pair_count_ge_090": len(pairs),
        "containment_pair_count_ge_095": sum(pair["intersection_over_inner_area"] >= 0.95 for pair in pairs),
        "reviewed_inner_pair_count": len(reviewed_nested),
        "reviewed_classification": counts(pair["reviewed_classification"] for pair in reviewed_nested),
        "unique_simulated_suppressed_candidates": len(suppressed_candidates),
        "reviewed_simulated_suppressed_targets": sorted(reviewed_suppressed),
        "reviewed_simulated_suppressed_count": len(reviewed_suppressed),
        "reviewed_useful_targets_lost": len(reviewed_suppressed & useful_reviewed),
        "protected_exception_pair_count": sum(pair["protected_exception"] for pair in pairs),
    }
    simulation = {
        "policy": "CONTAINMENT_095_SIZE_RATIO_035_OUTER_PERSON_SUPPORT_WITH_INNER_EVIDENCE_AND_PROTECTIONS",
        "applied_to_production": False,
        **summary,
    }
    examples = {
        "warning": TARGETED_WARNING,
        "protected_examples": [pair for pair in pairs if pair["protected_exception"]][:20],
        "simulated_suppression_examples": [pair for pair in pairs if pair["conservative_suppression_simulated"]][:20],
    }
    return {"pairs": pairs, "summary": summary, "simulation": simulation, "examples": examples}


def iou(first: Sequence[float], second: Sequence[float]) -> float:
    return box_metrics(first, second)["box_iou"]


def scene_diagnosis(scenes: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    b3_rows = load_jsonl(B3 / "03_REPLAY_RUNTIME/foldwise_candidate_records.jsonl")
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in b3_rows:
        by_frame[row["frame_sha256"]].append(row)
    candidates_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        candidates_by_scene[row["scene_id"]].append(row)
    rows = []
    for scene in scenes:
        all_candidates = by_frame[scene["frame_sha256"]]
        edges = sum(
            iou(first["source_box_xyxy"], second["source_box_xyxy"]) >= 0.10
            for index, first in enumerate(all_candidates)
            for second in all_candidates[index + 1 :]
        )
        possible = len(all_candidates) * (len(all_candidates) - 1) / 2
        reviewed = candidates_by_scene[scene["scene_id"]]
        review = scene["canonical_review"]
        rows.append(
            {
                **{
                    key: scene[key]
                    for key in ("scene_id", "frame_id", "frame_sha256", "match", "half", "lighting", "scene_category")
                },
                "candidate_count": len(all_candidates),
                "overlap_edge_count_iou_010": edges,
                "overlap_graph_density": edges / possible if possible else 0.0,
                "far_candidate_rate": sum(row["perspective_band"] == "FAR" for row in all_candidates)
                / len(all_candidates),
                "reviewed_merged_count": sum(row["analysis_flags"]["contains_multiple_people"] for row in reviewed),
                "reviewed_truncated_count": sum(
                    row["canonical_decision"]["box_quality"] == "TOO_TIGHT_OR_TRUNCATED" for row in reviewed
                ),
                "reviewed_background_count": sum(row["analysis_flags"]["is_background_or_object"] for row in reviewed),
                "reviewed_missed_person_count": len(review["missed_people_source_xy"]),
                "duplicate_burden": review["duplicate_or_overlap_burden"],
                "off_pitch_burden": review["off_pitch_proposal_burden"],
                "occlusion_burden": review["occlusion_burden"],
                "human_bottlenecks": review["bottlenecks"],
            }
        )
    sorted_counts = sorted(row["candidate_count"] for row in rows)
    sorted_density = sorted(row["overlap_graph_density"] for row in rows)
    count_cut = sorted_counts[math.ceil(len(rows) * 2 / 3) - 1]
    density_cut = sorted_density[math.ceil(len(rows) * 2 / 3) - 1]
    for row in rows:
        row["density_band"] = (
            "HIGH_DENSITY"
            if row["candidate_count"] >= count_cut or row["overlap_graph_density"] >= density_cut
            else "LOWER_DENSITY"
        )
    marks = [mark for scene in scenes for mark in scene["canonical_review"]["missed_people_source_xy"]]
    coverage = {
        "scene_count": len(rows),
        "scenes_zero_missed_relevant_people": sum(row["reviewed_missed_person_count"] == 0 for row in rows),
        "scenes_one_or_more_missed_relevant_people": sum(row["reviewed_missed_person_count"] > 0 for row in rows),
        "total_missed_person_marks": len(marks),
        "allowed_wording": (
            f"{sum(row['reviewed_missed_person_count'] > 0 for row in rows)} of 24 reviewed scenes contained "
            "at least one human-marked missed relevant person."
        ),
        "dataset_recall_claimed": False,
    }
    missed = {
        "scene_coverage": coverage,
        "by_role": counts(mark["role"] for mark in marks),
        "by_certainty": counts(mark["certainty"] for mark in marks),
        "by_match": {
            match: sum(row["reviewed_missed_person_count"] for row in rows if row["match"] == match)
            for match in sorted(POLYGON_HASHES)
        },
    }
    return {
        "rows": rows,
        "coverage": coverage,
        "missed": missed,
        "proposal_burden": {"off_pitch_background_burden": counts(row["off_pitch_burden"] for row in rows)},
        "duplicate_burden": {"duplicate_overlap_burden": counts(row["duplicate_burden"] for row in rows)},
        "occlusion_burden": {"occlusion_burden": counts(row["occlusion_burden"] for row in rows)},
        "bottlenecks": {"human_bottlenecks": counts(item for row in rows for item in row["human_bottlenecks"])},
        "density": {"count_cut": count_cut, "overlap_density_cut": density_cut, "scenes": rows},
    }


def slice_rates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"scene_support": 0}
    return {
        "scene_support": len(rows),
        "merged_targeted_rate": sum(row["reviewed_merged_count"] for row in rows) / (8 * len(rows)),
        "truncation_targeted_rate": sum(row["reviewed_truncated_count"] for row in rows) / (8 * len(rows)),
        "background_targeted_rate": sum(row["reviewed_background_count"] for row in rows) / (8 * len(rows)),
        "scenes_with_missed_relevant_person_rate": sum(row["reviewed_missed_person_count"] > 0 for row in rows)
        / len(rows),
        "mean_missed_marks": sum(row["reviewed_missed_person_count"] for row in rows) / len(rows),
        "high_duplicate_burden_rate": sum(row["duplicate_burden"] == "HIGH" for row in rows) / len(rows),
        "moderate_or_high_occlusion_burden_rate": sum(row["occlusion_burden"] in {"MODERATE", "HIGH"} for row in rows)
        / len(rows),
        "warning": TARGETED_WARNING,
    }


def crowding_diagnosis(scene: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    rows = scene["rows"]
    high = [row for row in rows if row["density_band"] == "HIGH_DENSITY"]
    lower = [row for row in rows if row["density_band"] == "LOWER_DENSITY"]
    far = [row for row in rows if row["far_candidate_rate"] >= 0.50]
    near = [row for row in rows if row["far_candidate_rate"] < 0.50]
    by_scene = {row["scene_id"]: row for row in rows}
    high_targets = [row for row in candidates if by_scene[row["scene_id"]]["density_band"] == "HIGH_DENSITY"]
    lower_targets = [row for row in candidates if by_scene[row["scene_id"]]["density_band"] == "LOWER_DENSITY"]

    def fold_disagreement(target_rows: Sequence[Mapping[str, Any]]) -> float:
        return sum(
            len(
                {
                    next(head for head in fold["head_outputs"] if head["head_name"] == "candidate_state")["top_class"]
                    for fold in row["fold_outputs"]
                }
            )
            > 1
            for row in target_rows
        ) / len(target_rows)

    summary = {
        "warning": TARGETED_WARNING,
        "high_density": slice_rates(high),
        "lower_density": slice_rates(lower),
        "far_dominant": slice_rates(far),
        "not_far_dominant": slice_rates(near),
        "candidate_state_fold_disagreement": {
            "high_density": fold_disagreement(high_targets),
            "lower_density": fold_disagreement(lower_targets),
        },
        "set_piece_causality_claimed": False,
    }
    high_miss_rate = summary["high_density"]["scenes_with_missed_relevant_person_rate"]
    lower_miss_rate = summary["lower_density"]["scenes_with_missed_relevant_person_rate"]
    support_indicators = {
        "miss_rate_elevated": high_miss_rate > lower_miss_rate,
        "merged_rate_elevated": summary["high_density"]["merged_targeted_rate"]
        > summary["lower_density"]["merged_targeted_rate"],
        "occlusion_burden_elevated": summary["high_density"]["moderate_or_high_occlusion_burden_rate"]
        > summary["lower_density"]["moderate_or_high_occlusion_burden_rate"],
        "fold_disagreement_elevated": summary["candidate_state_fold_disagreement"]["high_density"]
        > summary["candidate_state_fold_disagreement"]["lower_density"],
    }
    support_rate = sum(support_indicators.values()) / len(support_indicators)
    temporal = {
        "classification": "SUPPORTS_BOUNDED_TEMPORAL_FOLLOW_UP"
        if support_rate >= 0.25
        else "TEMPORAL_NEED_NOT_ESTABLISHED",
        "support_indicator_rate": support_rate,
        "support_indicators": support_indicators,
        "wording": "Associations observed in targeted crowded/far-side slices; no set-piece causality inferred.",
        "bounded_workload_if_selected": {
            "burst_count_min": 120,
            "burst_count_max": 180,
            "frames_per_burst_min": 5,
            "frames_per_burst_max": 11,
            "quotas": [
                "crowded_far_side",
                "occlusion_entry_exit",
                "merged_boxes",
                "missed_people",
                "fragmentation",
                "stable_controls",
            ],
        },
    }
    contract = {
        "proxies": [
            "candidate_count",
            "box_overlap_graph_density",
            "merged_box_labels",
            "partial_or_severe_occlusion",
            "missed_person_marks",
            "far_perspective_band",
            "human_duplicate_and_occlusion_burden",
        ],
        "high_density_rule": "upper-tercile candidate count OR upper-tercile IoU>=0.10 overlap graph density",
        "set_piece_label_used": False,
    }
    return {
        "contract": contract,
        "summary": summary,
        "far": {"far_dominant": summary["far_dominant"]},
        "temporal": temporal,
    }


def foldwise_diagnosis(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    output = {"warning": TARGETED_WARNING, "aggregation": "NONE", "folds": {}}
    dimensions = ["match", "lighting", "perspective_band", "scene_category", "target_slot"]
    for fold_id in range(5):
        fold_result = {}
        for head, labels in HEAD_LABELS.items():
            selected = []
            for row in candidates:
                truth = head_truth(row["canonical_decision"], head)
                if truth is None:
                    continue
                fold = next(item for item in row["fold_outputs"] if item["fold_id"] == fold_id)
                head_output = next(item for item in fold["head_outputs"] if item["head_name"] == head)
                selected.append((row, truth, head_output))
            metric = classification_metrics(
                [item[1] for item in selected], [item[2]["calibrated_probabilities"] for item in selected], labels
            )
            slices = {}
            for dimension in dimensions:
                slices[dimension] = {}
                for value in sorted({str(item[0][dimension]) for item in selected}):
                    subset = [item for item in selected if str(item[0][dimension]) == value]
                    slices[dimension][value] = classification_metrics(
                        [item[1] for item in subset], [item[2]["calibrated_probabilities"] for item in subset], labels
                    )
            for dimension, selector in {
                "occlusion": lambda row: row["canonical_decision"]["occlusion"],
                "box_quality": lambda row: row["canonical_decision"]["box_quality"],
                "candidate_state": lambda row: row["canonical_decision"]["proposal_validity"],
                "certainty": lambda row: row["canonical_decision"]["certainty"],
            }.items():
                slices[dimension] = {}
                for value in sorted({selector(item[0]) for item in selected}):
                    subset = [item for item in selected if selector(item[0]) == value]
                    slices[dimension][value] = classification_metrics(
                        [item[1] for item in subset], [item[2]["calibrated_probabilities"] for item in subset], labels
                    )
            fold_result[head] = {"overall": metric, "error_slices": slices}
        output["folds"][str(fold_id)] = fold_result
    return output


def bottleneck_diagnosis(
    candidates: list[dict[str, Any]], scenes: dict[str, Any], nested: dict[str, Any], foldwise: dict[str, Any]
) -> dict[str, Any]:
    buckets: dict[str, set[str]] = defaultdict(set)
    for row in candidates:
        target_id = row["target_id"]
        decision = row["canonical_decision"]
        if row["analysis_flags"]["is_background_or_object"] or decision["role"] == "STAFF_OR_SPECTATOR":
            buckets["BACKGROUND_OR_OFF_PITCH_CLUTTER"].add(target_id)
        if row["analysis_flags"]["is_duplicate"]:
            buckets["DUPLICATE_PROPOSALS"].add(target_id)
        if row["analysis_flags"]["contains_multiple_people"]:
            buckets["MERGED_BOX"].add(target_id)
        if decision["box_quality"] == "TOO_TIGHT_OR_TRUNCATED":
            buckets["TRUNCATED_OR_PARTIAL_BOX"].add(target_id)
        if decision["box_quality"] == "MISLOCALIZED":
            buckets["MISLOCALIZED_BOX"].add(target_id)
        if row["analysis_flags"]["has_occlusion"]:
            buckets["OCCLUSION"].add(target_id)
        if row["perspective_band"] == "FAR" and row["analysis_flags"]["box_quality_issue"]:
            buckets["SCALE_OR_PERSPECTIVE"].add(target_id)
    for target_id in nested["summary"]["reviewed_simulated_suppressed_targets"]:
        buckets["NESTED_FRAGMENT_PROPOSAL"].add(target_id)
    useful = [row for row in candidates if row["analysis_flags"]["box_is_useful_single_person"]]
    semantic_map = {
        "role": "ROLE_SEMANTICS",
        "team": "TEAM_SEMANTICS",
        "participation": "PARTICIPATION_SEMANTICS",
        "pitch": "PITCH_STATE",
    }
    for row in useful:
        for head, bucket in semantic_map.items():
            truth = head_truth(row["canonical_decision"], head)
            if truth is None:
                continue
            for fold in row["fold_outputs"]:
                prediction = next(item for item in fold["head_outputs"] if item["head_name"] == head)["top_class"]
                if prediction != truth:
                    buckets[bucket].add(row["target_id"])
                    break
    for scene in scenes["rows"]:
        if scene["reviewed_missed_person_count"]:
            buckets["PROPOSAL_SUPPLY_MISS"].add(scene["scene_id"])
    result = {
        key: {"affected_count": len(values), "example_ids": sorted(values)[:20]}
        for key, values in sorted(buckets.items())
    }
    return {
        "warning": TARGETED_WARNING,
        "bottlenecks": result,
        "semantic_blame_requires_useful_human_person_box": True,
        "intervention_matrix": {
            "BACKGROUND_OR_OFF_PITCH_CLUTTER": "pitch/population-aware proposal gate sandbox",
            "NESTED_FRAGMENT_PROPOSAL": "conservative containment suppression sandbox",
            "PROPOSAL_SUPPLY_MISS": "bounded temporal annotation and proposal discovery",
            "MERGED_BOX": "temporal separation evidence",
            "ROLE_SEMANTICS": "static semantic repair only after proposal adequacy",
            "TEAM_SEMANTICS": "match-local appearance/temporal evidence",
            "PARTICIPATION_SEMANTICS": "temporal participation evidence",
            "PITCH_STATE": "deterministic polygon geometry remains authoritative",
        },
        "fold_aggregation_used": False,
    }


def make_visuals(
    candidate: dict[str, dict[str, Any]],
    pitch: dict[str, Any],
    nested: dict[str, Any],
    scene: dict[str, Any],
    crowding: dict[str, Any],
    foldwise: dict[str, Any],
) -> list[Path]:
    out = STAGE / "10_VISUAL_EVIDENCE"
    out.mkdir(parents=True, exist_ok=True)
    warning = TARGETED_WARNING
    plt.style.use("seaborn-v0_8-whitegrid")

    state_counts = candidate["candidate_state_summary.json"]["canonical_proposal_validity"]
    labels = list(state_counts)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].barh(labels, [state_counts[label] for label in labels], color="#4169a1")
    axes[0].set_title("Canonical candidate validity (n=192)")
    nested_values = [
        nested["summary"]["containment_pair_count_ge_090"],
        nested["summary"]["containment_pair_count_ge_095"],
        nested["summary"]["unique_simulated_suppressed_candidates"],
        nested["summary"]["protected_exception_pair_count"],
    ]
    axes[1].bar([">=90%", ">=95%", "simulated suppress", "protected"], nested_values, color="#d9822b")
    axes[1].set_title("Nested-candidate containment findings")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle(f"Proposal and box diagnosis\n{warning}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    first = out / "01_PROPOSAL_AND_BOX_DIAGNOSIS.png"
    fig.savefig(first, dpi=150, metadata={"Software": "Football Intelligence deterministic C2-R1"})
    plt.close(fig)

    policy = pitch["policies"]["C_FAR_OUTSIDE_WITH_RELEVANT_EXCEPTIONS"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), layout="constrained")
    pitch_counts = pitch["off_pitch"]["human_pitch_state"]
    pitch_labels = list(pitch_counts)
    axes[0].bar(pitch_labels, [pitch_counts[key] for key in pitch_labels], color="#4c956c")
    axes[0].set_title(
        f"Human pitch-state labels (n=192)\nC policy clutter removed {policy['reviewed_clutter_removed']}"
    )
    axes[0].tick_params(axis="x", rotation=20)
    coverage = scene["coverage"]
    axes[1].bar(
        ["zero marks", ">=1 mark", "total marks"],
        [
            coverage["scenes_zero_missed_relevant_people"],
            coverage["scenes_one_or_more_missed_relevant_people"],
            coverage["total_missed_person_marks"],
        ],
        color=["#6c8ebf", "#c75c5c", "#8f6bb3"],
    )
    axes[1].set_title("Whole-scene missed-person review (24 scenes)")
    fig.suptitle(f"Pitch and scene burden\n{warning}", fontsize=14, fontweight="bold")
    second = out / "02_SCENE_AND_PITCH_BURDEN.png"
    fig.savefig(second, dpi=150, metadata={"Software": "Football Intelligence deterministic C2-R1"})
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    heads = list(HEAD_LABELS)
    for fold_id in range(5):
        agreements = [foldwise["folds"][str(fold_id)][head]["overall"]["exact_agreement"] for head in heads]
        axes[0].plot(heads, agreements, marker="o", label=f"fold {fold_id}")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Separate fold-local exact agreement")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(ncol=2)
    high = crowding["summary"]["high_density"]
    low = crowding["summary"]["lower_density"]
    metrics = ["merged_targeted_rate", "truncation_targeted_rate", "scenes_with_missed_relevant_person_rate"]
    x = range(len(metrics))
    axes[1].bar([value - 0.18 for value in x], [high[key] for key in metrics], width=0.36, label="high density")
    axes[1].bar([value + 0.18 for value in x], [low[key] for key in metrics], width=0.36, label="lower density")
    axes[1].set_xticks(list(x), ["merged", "truncated", "scenes with marks"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Crowding proxy slices")
    axes[1].legend()
    fig.suptitle(f"Foldwise semantics and crowding\n{warning} · no ensemble", fontsize=14, fontweight="bold")
    fig.tight_layout()
    third = out / "03_FOLDWISE_AND_CROWDING_DIAGNOSIS.png"
    fig.savefig(third, dpi=150, metadata={"Software": "Football Intelligence deterministic C2-R1"})
    plt.close(fig)
    return [first, second, third]


def write_outputs() -> dict[str, Any]:
    continuation = continuation_gate()
    provenance, latest, receipt_path, receipt = provenance_gate()
    write_json(STAGE / "00_CONTINUATION_PROVENANCE/continuation_validation.json", {**continuation, **provenance})
    candidates, scenes, marks, human_validation = normalize_human(latest)
    human_dir = STAGE / "01_HUMAN_REVIEW_CLOSURE"
    write_json(human_dir / "human_event_selection.json", provenance)
    write_jsonl(
        human_dir / "candidate_human_labels.jsonl",
        [{key: value for key, value in row.items() if key != "fold_outputs"} for row in candidates],
    )
    write_jsonl(human_dir / "scene_human_labels.jsonl", scenes)
    write_jsonl(human_dir / "missed_person_marks.jsonl", marks)
    write_json(human_dir / "human_review_validation_report.json", human_validation)
    manifest(human_dir, "human_review_artifact_manifest.json")

    candidate_outputs = candidate_diagnosis(candidates)
    candidate_dir = STAGE / "02_CANDIDATE_DIAGNOSIS"
    for name, value in candidate_outputs.items():
        write_json(candidate_dir / name, value)

    pitch = pitch_diagnosis(candidates)
    write_json(candidate_dir / "nested_candidate_summary.json", {"deferred_to": "04_NESTED_CANDIDATE_DIAGNOSIS"})
    manifest(candidate_dir, "candidate_diagnosis_manifest.json")
    pitch_dir = STAGE / "03_PITCH_AND_POPULATION_DIAGNOSIS"
    write_json(pitch_dir / "off_pitch_candidate_summary.json", pitch["off_pitch"])
    write_json(pitch_dir / "population_scope_summary.json", pitch["scope"])
    write_json(
        pitch_dir / "pitch_polygon_filter_simulation.json",
        {"policies": pitch["policies"], "applied_to_production": False},
    )
    write_json(pitch_dir / "boundary_exception_summary.json", pitch["boundary"])
    manifest(pitch_dir, "pitch_population_diagnosis_manifest.json")

    nested = nested_diagnosis(candidates)
    nested_dir = STAGE / "04_NESTED_CANDIDATE_DIAGNOSIS"
    write_jsonl(nested_dir / "box_containment_pairs.jsonl", nested["pairs"])
    write_json(nested_dir / "nested_candidate_summary.json", nested["summary"])
    write_json(nested_dir / "nested_suppression_simulation.json", nested["simulation"])
    write_json(nested_dir / "nested_exception_examples.json", nested["examples"])
    manifest(nested_dir, "nested_candidate_manifest.json")
    write_json(candidate_dir / "nested_candidate_summary.json", nested["summary"])
    manifest(candidate_dir, "candidate_diagnosis_manifest.json")

    scene = scene_diagnosis(scenes, candidates)
    scene_dir = STAGE / "05_SCENE_DIAGNOSIS"
    scene_files = {
        "scene_coverage_summary.json": scene["coverage"],
        "missed_relevant_person_summary.json": scene["missed"],
        "proposal_burden_summary.json": scene["proposal_burden"],
        "duplicate_overlap_burden_summary.json": scene["duplicate_burden"],
        "occlusion_burden_summary.json": scene["occlusion_burden"],
        "human_bottleneck_summary.json": scene["bottlenecks"],
        "crowding_density_summary.json": scene["density"],
    }
    for name, value in scene_files.items():
        write_json(scene_dir / name, value)
    manifest(scene_dir, "scene_diagnosis_manifest.json")

    crowding = crowding_diagnosis(scene, candidates)
    crowd_dir = STAGE / "06_CROWDING_AND_TEMPORAL_HYPOTHESIS"
    crowd_files = {
        "crowding_proxy_contract.json": crowding["contract"],
        "crowding_slice_summary.json": crowding["summary"],
        "far_side_occlusion_summary.json": crowding["far"],
        "temporal_evidence_need.json": crowding["temporal"],
    }
    for name, value in crowd_files.items():
        write_json(crowd_dir / name, value)
    manifest(crowd_dir, "crowding_hypothesis_manifest.json")

    foldwise = foldwise_diagnosis(candidates)
    fold_dir = STAGE / "07_FOLDWISE_SEMANTIC_DIAGNOSIS"
    write_json(fold_dir / "foldwise_semantic_diagnosis.json", foldwise)
    manifest(fold_dir, "foldwise_semantic_diagnosis_manifest.json")

    bottleneck = bottleneck_diagnosis(candidates, scene, nested, foldwise)
    bottleneck_dir = STAGE / "08_BOTTLENECK_ATTRIBUTION"
    write_json(bottleneck_dir / "bottleneck_attribution.json", bottleneck)
    write_json(bottleneck_dir / "intervention_matrix.json", bottleneck["intervention_matrix"])
    manifest(bottleneck_dir, "bottleneck_attribution_manifest.json")

    safe_policy = pitch["policies"]["C_FAR_OUTSIDE_WITH_RELEVANT_EXCEPTIONS"]
    useful = [row for row in candidates if row["analysis_flags"]["box_is_useful_single_person"]]
    semantic_errors = []
    for row in useful:
        mismatched = False
        for fold in row["fold_outputs"]:
            for head in ("role", "team", "participation"):
                truth = head_truth(row["canonical_decision"], head)
                if truth is None:
                    continue
                prediction = next(item for item in fold["head_outputs"] if item["head_name"] == head)["top_class"]
                mismatched |= prediction != truth
        semantic_errors.append(mismatched)
    decision_inputs = {
        "pitch_clutter_removed_rate": safe_policy["reviewed_clutter_removed_rate"],
        "pitch_relevant_useful_loss_rate": safe_policy["reviewed_useful_relevant_loss_rate"],
        "nested_separable_burden_rate": nested["summary"]["reviewed_simulated_suppressed_count"] / 192,
        "crowding_temporal_support_rate": crowding["temporal"]["support_indicator_rate"],
        "useful_box_semantic_error_rate": sum(semantic_errors) / len(semantic_errors),
    }
    recommendation = choose_next_stage(decision_inputs)
    recommendation["sequence"] = {
        "immediate_primary": recommendation["primary_stage"],
        "conditional_secondary": recommendation["conditional_secondary_stage"],
        "medium_term_temporal_density_work": "bounded 120–180 bursts only if temporal follow-up remains selected",
    }
    recommendation["bounded_workload"] = crowding["temporal"]["bounded_workload_if_selected"]
    next_dir = STAGE / "09_NEXT_STAGE_RECOMMENDATION"
    write_json(next_dir / "next_stage_recommendation.json", recommendation)
    write_json(next_dir / "decision_evidence.json", decision_inputs)
    manifest(next_dir, "next_stage_recommendation_manifest.json")

    visuals = make_visuals(candidate_outputs, pitch, nested, scene, crowding, foldwise)
    analysis_contract = {
        "classification": "TARGETED_REVIEW_DIAGNOSIS_ONLY",
        "warning": TARGETED_WARNING,
        "visual_only_not_metric": True,
        "production_ready": False,
        "folds_kept_separate": [0, 1, 2, 3, 4],
        "aggregation": "NONE",
        "inference_rerun": False,
        "human_review_repeated": False,
        "thresholds_changed": False,
        "production_filter_or_suppression_applied": False,
        "validation_or_holdout_access": False,
    }
    evidence_dir = STAGE / "11_FINALIZATION_EVIDENCE"
    focused_test_path = evidence_dir / "focused_test_results.json"
    focused_tests = (
        load_json(focused_test_path).get("classification") if focused_test_path.exists() else "PENDING_FOCUSED_TEST_RUN"
    )
    write_json(evidence_dir / "event_and_receipt_selection.json", provenance)
    write_json(evidence_dir / "analysis_contract.json", analysis_contract)
    final_report = {
        "classification": "PASS_G7D_C2_R1_VISUAL_TRANSFER_DIAGNOSIS_FINALIZED",
        "candidate_count": 192,
        "scene_count": 24,
        "missed_person_mark_count": len(marks),
        "visual_count": len(visuals),
        "primary_next_stage": recommendation["primary_stage"],
        "conditional_secondary_stage": recommendation["conditional_secondary_stage"],
        "operator_hypotheses": {
            "off_pitch_clutter": pitch["scope"],
            "nested_fragments": nested["summary"],
            "crowding_temporal": crowding["temporal"],
        },
        "tests": focused_tests,
        **analysis_contract,
    }
    write_json(evidence_dir / "finalization_validation_report.json", final_report)

    handoff = STAGE / "12_REVIEW_PACK/CHATGPT_HANDOFF"
    handoff.mkdir(parents=True, exist_ok=True)
    write_json(
        handoff / "01_EXECUTIVE_SUMMARY.json",
        {
            **final_report,
            "receipt_id": RECEIPT_ID,
            "event_set_digest": EVENT_SET_DIGEST,
            "key_candidate_counts": candidate_outputs["candidate_state_summary.json"]["canonical_proposal_validity"],
            "scene_coverage": scene["coverage"],
            "decision_inputs": decision_inputs,
        },
    )
    write_json(handoff / "02_HUMAN_REVIEW_CLOSURE.json", {"provenance": provenance, "validation": human_validation})
    write_json(
        handoff / "03_CANDIDATE_AND_NESTED_DIAGNOSIS.json",
        {
            "candidate": candidate_outputs["candidate_state_summary.json"],
            "box_quality": candidate_outputs["box_quality_summary.json"],
            "nested": nested["summary"],
            "nested_simulation": nested["simulation"],
        },
    )
    write_json(
        handoff / "04_PITCH_AND_SCENE_DIAGNOSIS.json",
        {
            "pitch": pitch["off_pitch"],
            "population": pitch["scope"],
            "policies": pitch["policies"],
            "scene": scene["coverage"],
            "burdens": {
                "proposal": scene["proposal_burden"],
                "duplicate": scene["duplicate_burden"],
                "occlusion": scene["occlusion_burden"],
            },
        },
    )
    write_json(handoff / "05_CROWDING_AND_TEMPORAL_DIAGNOSIS.json", crowding)
    write_json(handoff / "06_FOLDWISE_SEMANTIC_DIAGNOSIS.json", foldwise)
    write_json(
        handoff / "07_BOTTLENECK_AND_NEXT_STAGE.json", {"bottleneck": bottleneck, "recommendation": recommendation}
    )
    (handoff / "08_DECISION.md").write_text(
        f"# G7D-C2 R1 decision\n\nPrimary: `{recommendation['primary_stage']}`.\n\n{recommendation['primary_reason']} "
        f"Conditional secondary: `{recommendation['conditional_secondary_stage']}`. No policy was applied, no fold "
        "was aggregated, and no production claim is made.\n",
        encoding="utf-8",
        newline="\n",
    )
    (handoff / "09_ANALYSIS_CONTRACT.md").write_text(
        f"# Analysis contract\n\nEvery candidate rate is **{TARGETED_WARNING}**. The 24 scene checks cover "
        "only reviewed frames. Folds 0–4 remain separate. Pitch filtering and nested suppression are retrospective "
        "simulations, not production changes. No inference, training, tuning, threshold change, validation/holdout "
        "access, identity, tracking, tactical, physical, or production metric is present.\n",
        encoding="utf-8",
        newline="\n",
    )
    shutil.copy2(visuals[0], handoff / "10_PROPOSAL_VISUAL.png")
    shutil.copy2(visuals[2], handoff / "11_SCENE_AND_FOLD_VISUAL.png")
    write_json(
        handoff / "12_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7d_c2_r1.handoff_manifest.v1",
            "self_hash_omitted": True,
            "file_count": 11,
            "files": [
                {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256(path)}
                for path in sorted(handoff.iterdir())
                if path.is_file() and path.name != "12_MANIFEST.json"
            ],
        },
    )
    (STAGE / "12_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only the CHATGPT_HANDOFF folder.\n", encoding="utf-8", newline="\n"
    )

    evidence_paths = [
        path
        for path in STAGE.rglob("*")
        if path.is_file() and "12_REVIEW_PACK" not in path.parts and path.name != "finalization_artifact_manifest.json"
    ]
    write_json(
        evidence_dir / "finalization_artifact_manifest.json",
        {
            "schema_version": "football_intelligence.g7d_c2_r1.finalization_artifact_manifest.v1",
            "self_hash_omitted": True,
            "selected_event_count": 216,
            "selected_acknowledgement_count": 216,
            "current_completion_receipt": artifact(receipt_path),
            "artifacts": [artifact(path) for path in sorted(evidence_paths)],
        },
    )
    return {
        "classification": final_report["classification"],
        "candidate_count": 192,
        "scene_count": 24,
        "missed_marks": len(marks),
        "primary_stage": recommendation["primary_stage"],
        "conditional_secondary": recommendation["conditional_secondary_stage"],
        "decision_inputs": decision_inputs,
        "visual_count": 3,
    }


def main() -> int:
    result = write_outputs()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
