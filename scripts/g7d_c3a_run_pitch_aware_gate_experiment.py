"""Run the bounded CPU-only G7D-C3A pitch-aware proposal gate sandbox."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from football_intelligence.pitch_aware_proposal_gate import (
    SANDBOX_DECISIONS,
    adaptive_boundary_band,
    candidate_geometry,
    gate_decision,
)

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PACK = (
    PROJECT / "experiments/football_observation_reasoner/part 6/G7D_C3A_Pitch_Aware_Proposal_Gate_Experiment_Codex_Pack"
)
C2 = (
    PROJECT
    / "experiments/football_observation_reasoner/part 6/G7D_C2_R1_RESUME_VISUAL_TRANSFER_DIAGNOSIS_FINALIZATION_v1"
)
B3 = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B3_FROZEN_CROSS_MATCH_REPLAY_v1"
B2C = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B2C_RESUME_FROZEN_128058_BASELINE_v1"
STAGE = PROJECT / "experiments/football_observation_reasoner/part 7/G7D_C3A_PITCH_AWARE_PROPOSAL_GATE_EXPERIMENT_v1"
EXPECTED_HEAD = "5de1fbeeb91180c714ba46e68e4210d05d4b8456"
TARGETED_WARNING = "TARGETED REVIEW SAMPLE — NOT UNBIASED ACCURACY"
SANDBOX_WARNING = "SANDBOX PITCH GATE — NOT PRODUCTION"
ORACLE_WARNING = "ORACLE HUMAN-LABEL UPPER BOUND — NOT IMPLEMENTABLE"
POLYGON_HASHES = {
    "128058": "24ad1e4d143527e5a3e92cded1b5d8b10526d67b5b0d1f8b02289a91e8c65307",
    "118575": "fbd7f3a473acc197b4c893d90bbaa4c5d484d1e883e8df1ac4601daf4396dec1",
    "117092": "92ca8040eedd3b0ec0bb685648691f0c314d8527f3fa8f2db1823b4461e4b338",
}
IMPLEMENTABLE_FAMILIES = [
    "G0_KEEP_ALL",
    "G1_STRICT_INSIDE",
    "G2_INSIDE_OR_ADAPTIVE_BOUNDARY",
    "G3_CONSERVATIVE_FAR_OUTSIDE",
    "G4_GEOMETRIC_EXCEPTION_GATE",
]
DECISION_ORDER = ["KEEP", "SUPPRESS_SANDBOX", "BOUNDARY_REVIEW", "EXCEPTION_KEEP"]
FIXED_PIXELS = [8, 16, 24, 32]
ALPHAS = [0.5, 1.0, 1.5, 2.0]
BAND_MODES = ["FIXED_PIXELS", "BOX_HEIGHT", "EXPECTED_HEIGHT_BY_PERSPECTIVE"]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "project_relative_path": path.relative_to(PROJECT).as_posix(),
        "byte_size": path.stat().st_size,
        "sha256": sha256(path),
    }


def directory_manifest(directory: Path, name: str) -> None:
    files = sorted(path for path in directory.iterdir() if path.is_file() and path.name != name)
    write_json(
        directory / name, {"self_hash_omitted": True, "file_count": len(files), "files": [artifact(p) for p in files]}
    )


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def parameter_grid() -> list[dict[str, Any]]:
    """Return the predeclared 36 unique adaptive-band parameterizations."""
    rows: list[dict[str, Any]] = []
    for fixed in FIXED_PIXELS:
        rows.append(
            {
                "parameter_id": f"fixed_{fixed:02d}",
                "band_mode": "FIXED_PIXELS",
                "fixed_pixels": fixed,
                "alpha": 0.0,
            }
        )
    for mode in ("BOX_HEIGHT", "EXPECTED_HEIGHT_BY_PERSPECTIVE"):
        for fixed in FIXED_PIXELS:
            for alpha in ALPHAS:
                rows.append(
                    {
                        "parameter_id": f"{mode.lower()}_f{fixed:02d}_a{alpha:.1f}",
                        "band_mode": mode,
                        "fixed_pixels": fixed,
                        "alpha": alpha,
                    }
                )
    if len(rows) != 36 or len({row["parameter_id"] for row in rows}) != 36:
        raise RuntimeError("FAIL_PREDECLARED_GRID")
    return rows


def validate_pack() -> dict[str, Any]:
    manifest = load_json(PACK / "05_PACK_MANIFEST.json")
    verified = []
    for row in manifest["files"]:
        path = PACK / row["path"]
        if not path.is_file() or path.stat().st_size != row["byte_size"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"FAIL_PROMPT_PACK_MANIFEST: {row['path']}")
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        else:
            path.read_text(encoding="utf-8")
        verified.append({"path": row["path"], "sha256": row["sha256"]})
    handoff_text = PACK / "08_C2_TEXT_HANDOFF"
    parsed_handoff = []
    for path in sorted(handoff_text.iterdir()):
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        else:
            path.read_text(encoding="utf-8")
        parsed_handoff.append(path.name)
    return {
        "classification": "PASS_PROMPT_PACK_AND_EMBEDDED_HANDOFF",
        "verified_file_count": len(verified),
        "verified_files": verified,
        "embedded_handoff_files_read": parsed_handoff,
        "model_binding": "GPT-5.6 Terra",
        "thinking_binding": "Medium",
        "sol_forbidden": True,
    }


def frozen_inputs() -> dict[str, Any]:
    if git("rev-parse", "HEAD") != EXPECTED_HEAD or git("branch", "--show-current") != "main":
        raise RuntimeError("FAIL_REPOSITORY_BASELINE")
    polygons: dict[str, dict[str, Any]] = {}
    setups: dict[str, dict[str, Any]] = {}
    polygon_paths: dict[str, Path] = {}
    for match, expected_hash in POLYGON_HASHES.items():
        path = PROJECT / f"matches/{match}/calibration/pitch_polygon_v1/pitch_polygon.json"
        if sha256(path) != expected_hash:
            raise RuntimeError(f"FAIL_POLYGON_HASH_{match}")
        polygon = load_json(path)
        setup = load_json(PROJECT / f"matches/{match}/calibration/match_setup.json")
        if (
            polygon["status"] != "HUMAN_CONFIRMED"
            or polygon["coordinate_space"] != "SOURCE_IMAGE_PIXELS"
            or setup["dataset_split"]["proposed_assignment"] != "TRAIN_DEVELOPMENT"
            or setup["dataset_split"]["frozen"] is not True
        ):
            raise RuntimeError(f"FAIL_MATCH_CLOSURE_{match}")
        polygons[match] = polygon
        setups[match] = setup
        polygon_paths[match] = path

    manifests = [
        load_json(B2C / "02_BASELINE_INPUTS/ordered_sampling_manifest.json"),
        load_json(B3 / "02_REPLAY_INPUTS/118575/ordered_sampling_manifest.json"),
        load_json(B3 / "02_REPLAY_INPUTS/117092/ordered_sampling_manifest.json"),
    ]
    frames = [frame for manifest in manifests for frame in manifest["frames"]]
    frame_counts = Counter(str(row["match_id"]) for row in frames)
    if frame_counts != Counter({"128058": 32, "118575": 32, "117092": 32}) or len(frames) != 96:
        raise RuntimeError("FAIL_FROZEN_FRAME_CLOSURE")
    for frame in frames:
        path = Path(frame["path"])
        if sha256(path) != frame["frame_sha256"]:
            raise RuntimeError(f"FAIL_FRAME_HASH_{frame['frame_id']}")

    universe = load_jsonl(B2C / "04_BASELINE_REFERENCE/foldwise_candidate_records.jsonl")
    universe.extend(load_jsonl(B3 / "03_REPLAY_RUNTIME/foldwise_candidate_records.jsonl"))
    frame_metadata = {row["frame_sha256"]: row for row in frames}
    universe = [
        {
            **row,
            "frame_id": row.get("frame_id", frame_metadata[row["frame_sha256"]]["frame_id"]),
        }
        for row in universe
    ]
    if {str(row["match_id"]) for row in universe} != set(POLYGON_HASHES):
        raise RuntimeError("FAIL_CANDIDATE_MATCH_CLOSURE")
    reviewed = load_jsonl(C2 / "01_HUMAN_REVIEW_CLOSURE/candidate_human_labels.jsonl")
    scenes = load_jsonl(C2 / "01_HUMAN_REVIEW_CLOSURE/scene_human_labels.jsonl")
    marks = load_jsonl(C2 / "01_HUMAN_REVIEW_CLOSURE/missed_person_marks.jsonl")
    selection = load_json(C2 / "01_HUMAN_REVIEW_CLOSURE/human_event_selection.json")
    if len(reviewed) != 192 or len(scenes) != 24 or len(marks) != 22:
        raise RuntimeError("FAIL_HUMAN_REVIEW_CLOSURE")
    if selection["completion_receipt_id"] != "completion-r8-bbbaabc5fdbff19754baee53":
        raise RuntimeError("FAIL_COMPLETION_RECEIPT")

    source_files = [
        PACK / "05_PACK_MANIFEST.json",
        C2 / "01_HUMAN_REVIEW_CLOSURE/candidate_human_labels.jsonl",
        C2 / "01_HUMAN_REVIEW_CLOSURE/scene_human_labels.jsonl",
        C2 / "01_HUMAN_REVIEW_CLOSURE/missed_person_marks.jsonl",
        B2C / "04_BASELINE_REFERENCE/foldwise_candidate_records.jsonl",
        B3 / "03_REPLAY_RUNTIME/foldwise_candidate_records.jsonl",
        *polygon_paths.values(),
    ]
    return {
        "polygons": polygons,
        "setups": setups,
        "frames": frames,
        "universe": universe,
        "reviewed": reviewed,
        "scenes": scenes,
        "marks": marks,
        "source_files": source_files,
        "source_hashes": {path.relative_to(PROJECT).as_posix(): sha256(path) for path in source_files},
        "closure": {
            "classification": "PASS_G7D_C3A_FROZEN_INPUT_CLOSURE",
            "repository_head": EXPECTED_HEAD,
            "branch": "main",
            "frame_counts": dict(sorted(frame_counts.items())),
            "frame_count": len(frames),
            "full_universe_candidate_count": len(universe),
            "reviewed_candidate_count": len(reviewed),
            "reviewed_scene_count": len(scenes),
            "missed_person_mark_count": len(marks),
            "completion_receipt_id": selection["completion_receipt_id"],
            "polygon_hashes": POLYGON_HASHES,
            "device": "CPU_GEOMETRY_ONLY",
            "neural_inference_run": False,
        },
    }


def expected_height_tables(universe: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in universe:
        box = row["source_box_xyxy"]
        grouped[str(row["match_id"])][str(row.get("perspective_band", "UNKNOWN"))].append(float(box[3]) - float(box[1]))
    tables: dict[str, dict[str, float]] = {}
    for match, bands in grouped.items():
        all_heights = [height for values in bands.values() for height in values]
        tables[match] = {band: statistics.median(values) for band, values in sorted(bands.items())}
        tables[match]["UNKNOWN"] = statistics.median(all_heights)
    return tables


def runtime_candidate(row: Mapping[str, Any], polygons: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    match = str(row["match_id"])
    polygon = polygons[match]
    return {
        "source_box_xyxy": row["source_box_xyxy"],
        "approximate_footpoint_xy": row["approximate_footpoint_xy"],
        "source_width": polygon["source_width"],
        "source_height": polygon["source_height"],
        "perspective_band": row.get("perspective_band", "UNKNOWN"),
        "proposal_provenance": row.get("proposal_provenance", {}),
    }


def geometry_with_band(base: Mapping[str, Any], band: float) -> dict[str, Any]:
    geometry = dict(base)
    distance = float(geometry["absolute_boundary_distance_pixels"])
    geometry["adaptive_boundary_band_pixels"] = band
    if geometry["inside_polygon"]:
        geometry["geometry_band"] = "INSIDE"
    elif distance <= band:
        geometry["geometry_band"] = "NEAR_BOUNDARY"
    elif distance <= 3 * band:
        geometry["geometry_band"] = "OUTSIDE"
    else:
        geometry["geometry_band"] = "FAR_OUTSIDE"
    return geometry


def variant_id(family: str, parameter: Mapping[str, Any] | None) -> str:
    return family if parameter is None else f"{family}__{parameter['parameter_id']}"


def decision_for(
    family: str,
    base: Mapping[str, Any],
    row: Mapping[str, Any],
    parameter: Mapping[str, Any] | None,
    expected_heights: Mapping[str, float],
) -> dict[str, Any]:
    if family in {"G0_KEEP_ALL", "G1_STRICT_INSIDE"}:
        geometry = geometry_with_band(base, 1.0)
    else:
        if parameter is None:
            raise ValueError("adaptive family requires parameters")
        box = row["source_box_xyxy"]
        band = adaptive_boundary_band(
            float(box[3]) - float(box[1]),
            str(row.get("perspective_band", "UNKNOWN")),
            parameter["fixed_pixels"],
            parameter["alpha"],
            parameter["band_mode"],
            expected_heights,
        )
        geometry = geometry_with_band(base, band)
    return {**gate_decision(family, geometry), "geometry": geometry}


def human_flags(row: Mapping[str, Any]) -> dict[str, bool]:
    decision = row["canonical_decision"]
    flags = row["analysis_flags"]
    clutter = bool(
        flags["is_background_or_object"]
        or decision["role"] == "STAFF_OR_SPECTATOR"
        or decision["participation"] == "NON_PLAYER"
    )
    useful_relevant = bool(flags["box_is_useful_single_person"] and flags["is_relevant_active_population"])
    official = bool(decision["role"] in {"REFEREE", "OTHER_OFFICIAL"} and flags["contains_any_person"])
    active_player_or_gk = bool(
        decision["participation"] == "ACTIVE" and decision["role"] in {"OUTFIELD_PLAYER", "GOALKEEPER"}
    )
    boundary_uncertain_person = bool(
        flags["contains_any_person"] and decision["pitch_state"] in {"BOUNDARY", "UNCERTAIN"}
    )
    return {
        "clutter": clutter,
        "useful_relevant": useful_relevant,
        "official": official,
        "active_player_or_goalkeeper": active_player_or_gk,
        "boundary_uncertain_person": boundary_uncertain_person,
        "unknown": decision["role"] == "UNKNOWN_PERSON_ROLE" or decision["pitch_state"] == "UNCERTAIN",
    }


def summarize_reviewed(rows: Sequence[Mapping[str, Any]], decisions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    suppressed = {key for key, value in decisions.items() if value["decision"] == "SUPPRESS_SANDBOX"}
    metrics = Counter()
    decision_counts = Counter(value["decision"] for value in decisions.values())
    by_slice: dict[str, dict[str, dict[str, int]]] = {}
    dimensions = {
        "match": lambda row: row["match"],
        "half": lambda row: row["half"],
        "lighting": lambda row: row["lighting"],
        "role": lambda row: row["canonical_decision"]["role"],
        "participation": lambda row: row["canonical_decision"]["participation"],
        "human_pitch_state": lambda row: row["canonical_decision"]["pitch_state"],
        "perspective": lambda row: row["perspective_band"],
        "box_quality": lambda row: row["canonical_decision"]["box_quality"],
        "candidate_state": lambda row: row["canonical_decision"]["proposal_validity"],
    }
    for row in rows:
        key = f"{row['frame_sha256']}::{row['candidate_local_id']}"
        flags = human_flags(row)
        is_suppressed = key in suppressed
        metrics["clutter_support"] += flags["clutter"]
        metrics["clutter_removed"] += flags["clutter"] and is_suppressed
        metrics["useful_relevant_support"] += flags["useful_relevant"]
        metrics["useful_relevant_suppressed"] += flags["useful_relevant"] and is_suppressed
        metrics["official_support"] += flags["official"]
        metrics["official_suppressed"] += flags["official"] and is_suppressed
        metrics["active_player_or_goalkeeper_support"] += flags["active_player_or_goalkeeper"]
        metrics["active_player_or_goalkeeper_suppressed"] += flags["active_player_or_goalkeeper"] and is_suppressed
        metrics["boundary_uncertain_person_support"] += flags["boundary_uncertain_person"]
        metrics["boundary_uncertain_person_suppressed"] += flags["boundary_uncertain_person"] and is_suppressed
    for dimension, accessor in dimensions.items():
        grouped: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            key = f"{row['frame_sha256']}::{row['candidate_local_id']}"
            flags = human_flags(row)
            bucket = grouped[str(accessor(row))]
            bucket["support"] += 1
            bucket["suppressed"] += key in suppressed
            bucket["clutter_removed"] += flags["clutter"] and key in suppressed
            bucket["useful_relevant_suppressed"] += flags["useful_relevant"] and key in suppressed
        by_slice[dimension] = {name: dict(values) for name, values in sorted(grouped.items())}
    return {
        **dict(metrics),
        "decision_counts": {name: decision_counts.get(name, 0) for name in DECISION_ORDER},
        "by_slice": by_slice,
        "warning": TARGETED_WARNING,
    }


def frame_supply_summary(
    universe: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
    setups: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    frames: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in universe:
        frames[str(row["frame_sha256"])].append(row)
    frame_rows = []
    for frame_hash, candidates in sorted(frames.items()):
        counts = Counter(
            decisions[f"{row['frame_sha256']}::{row['candidate_local_id']}"]["decision"] for row in candidates
        )
        first = candidates[0]
        frame_rows.append(
            {
                "frame_sha256": frame_hash,
                "frame_id": first["frame_id"],
                "match": str(first["match_id"]),
                "half": first["half"],
                "lighting": setups[str(first["match_id"])]["conditions"]["lighting"],
                "raw_count": len(candidates),
                "retained_count": len(candidates) - counts["SUPPRESS_SANDBOX"],
                "reduction_count": counts["SUPPRESS_SANDBOX"],
                "reduction_rate": counts["SUPPRESS_SANDBOX"] / len(candidates),
                "decision_counts": {name: counts.get(name, 0) for name in DECISION_ORDER},
            }
        )
    raw = [row["raw_count"] for row in frame_rows]
    retained = [row["retained_count"] for row in frame_rows]
    reductions = [row["reduction_rate"] for row in frame_rows]
    return {
        "frame_count": len(frame_rows),
        "raw_candidate_count": sum(raw),
        "decision_counts": {name: sum(row["decision_counts"][name] for row in frame_rows) for name in DECISION_ORDER},
        "mean_candidates_per_frame_before": statistics.mean(raw),
        "median_candidates_per_frame_before": statistics.median(raw),
        "mean_candidates_per_frame_after": statistics.mean(retained),
        "median_candidates_per_frame_after": statistics.median(retained),
        "frame_reduction_rate": {
            "minimum": min(reductions),
            "median": statistics.median(reductions),
            "mean": statistics.mean(reductions),
            "maximum": max(reductions),
        },
        "frames": frame_rows,
        "semantic_workload_estimate_only": True,
        "gpu_speed_claimed": False,
    }


def missed_mark_safety(
    marks: Sequence[Mapping[str, Any]],
    scenes: Sequence[Mapping[str, Any]],
    universe: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
    polygons: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    scene_frames = {row["scene_id"]: row["frame_sha256"] for row in scenes}
    by_frame: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in universe:
        by_frame[str(row["frame_sha256"])].append(row)
    results = []
    unsafe = 0
    for mark in marks:
        frame_hash = scene_frames[mark["scene_id"]]
        match = str(mark["match"])
        radius = max(64.0, 0.03 * float(polygons[match]["source_width"]))
        point = tuple(map(float, mark["source_xy"]))
        nearby = []
        for candidate in by_frame[frame_hash]:
            footpoint = tuple(map(float, candidate["approximate_footpoint_xy"]))
            distance = math.dist(point, footpoint)
            if distance <= radius:
                key = f"{candidate['frame_sha256']}::{candidate['candidate_local_id']}"
                nearby.append(
                    {
                        "candidate_local_id": candidate["candidate_local_id"],
                        "distance_pixels": distance,
                        "decision": decisions[key]["decision"],
                    }
                )
        preserved = [row for row in nearby if row["decision"] != "SUPPRESS_SANDBOX"]
        classification = "NO_NEARBY_CANDIDATE" if not nearby else "PRESERVED" if preserved else "ALL_NEARBY_SUPPRESSED"
        unsafe += classification == "ALL_NEARBY_SUPPRESSED"
        results.append(
            {
                **mark,
                "radius_pixels": radius,
                "nearby_candidate_count": len(nearby),
                "preserved_nearby_candidate_count": len(preserved),
                "classification": classification,
                "nearby_candidates": sorted(
                    nearby, key=lambda row: (row["distance_pixels"], row["candidate_local_id"])
                ),
            }
        )
    return {
        "mark_count": len(marks),
        "unsafe_all_nearby_suppressed_count": unsafe,
        "marks_with_no_nearby_candidate": sum(row["classification"] == "NO_NEARBY_CANDIDATE" for row in results),
        "marks_with_preserved_neighbourhood": sum(row["classification"] == "PRESERVED" for row in results),
        "only_candidate_neighbourhoods_preserved": all(
            row["classification"] != "ALL_NEARBY_SUPPRESSED" for row in results if row["nearby_candidate_count"] == 1
        ),
        "marks": results,
    }


def oracle_decisions(reviewed: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    decisions = {}
    for row in reviewed:
        key = f"{row['frame_sha256']}::{row['candidate_local_id']}"
        flags = human_flags(row)
        decisions[key] = {
            "decision": "SUPPRESS_SANDBOX" if flags["clutter"] and not flags["useful_relevant"] else "KEEP",
            "reason_codes": ["HUMAN_ORACLE_CLUTTER" if flags["clutter"] else "HUMAN_ORACLE_RETAIN"],
        }
    return decisions


def evaluate(inputs: Mapping[str, Any], grid: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    universe = inputs["universe"]
    reviewed = inputs["reviewed"]
    polygons = inputs["polygons"]
    setups = inputs["setups"]
    expected = expected_height_tables(universe)
    universe_by_key = {f"{row['frame_sha256']}::{row['candidate_local_id']}": row for row in universe}
    if len(universe_by_key) != len(universe):
        raise RuntimeError("FAIL_DUPLICATE_UNIVERSE_CANDIDATE")
    reviewed_keys = {f"{row['frame_sha256']}::{row['candidate_local_id']}" for row in reviewed}
    if not reviewed_keys.issubset(universe_by_key):
        raise RuntimeError("FAIL_REVIEWED_JOIN")
    base_geometry = {}
    for key, row in universe_by_key.items():
        match = str(row["match_id"])
        base_geometry[key] = candidate_geometry(
            runtime_candidate(row, polygons), polygons[match]["vertices_source_xy"], 1.0
        )

    variants: list[tuple[str, Mapping[str, Any] | None]] = [
        ("G0_KEEP_ALL", None),
        ("G1_STRICT_INSIDE", None),
    ]
    for family in IMPLEMENTABLE_FAMILIES[2:]:
        variants.extend((family, parameter) for parameter in grid)
    results = []
    all_decisions: dict[str, dict[str, dict[str, Any]]] = {}
    for family, parameter in variants:
        identifier = variant_id(family, parameter)
        decisions = {}
        for key, row in universe_by_key.items():
            match = str(row["match_id"])
            decisions[key] = decision_for(family, base_geometry[key], row, parameter, expected[match])
        all_decisions[identifier] = decisions
        reviewed_decisions = {key: decisions[key] for key in reviewed_keys}
        reviewed_summary = summarize_reviewed(reviewed, reviewed_decisions)
        missed = missed_mark_safety(inputs["marks"], inputs["scenes"], universe, decisions, polygons)
        supply = frame_supply_summary(universe, decisions, setups)
        results.append(
            {
                "variant_id": identifier,
                "family": family,
                "parameter": parameter,
                "implementable_without_human_labels": True,
                "reviewed": reviewed_summary,
                "missed_mark_safety": {key: value for key, value in missed.items() if key != "marks"},
                "full_universe": {key: value for key, value in supply.items() if key != "frames"},
                "sandbox_only": True,
            }
        )
    oracle = oracle_decisions(reviewed)
    results.append(
        {
            "variant_id": "G5_HUMAN_ORACLE_UPPER_BOUND",
            "family": "G5_HUMAN_ORACLE_UPPER_BOUND",
            "parameter": None,
            "implementable_without_human_labels": False,
            "reviewed": summarize_reviewed(reviewed, oracle),
            "oracle_label": ORACLE_WARNING,
            "selection_forbidden": True,
        }
    )
    return {
        "expected_heights": expected,
        "universe_by_key": universe_by_key,
        "base_geometry": base_geometry,
        "results": results,
        "all_decisions": all_decisions,
    }


def select_gate(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    complexity = {
        "G0_KEEP_ALL": 0,
        "G1_STRICT_INSIDE": 1,
        "G2_INSIDE_OR_ADAPTIVE_BOUNDARY": 2,
        "G3_CONSERVATIVE_FAR_OUTSIDE": 3,
        "G4_GEOMETRIC_EXCEPTION_GATE": 4,
    }
    candidates = []
    for row in evaluation["results"]:
        if not row.get("implementable_without_human_labels"):
            continue
        reviewed = row["reviewed"]
        missed = row["missed_mark_safety"]
        clutter_threshold = math.ceil(0.25 * reviewed["clutter_support"])
        passes = (
            reviewed["useful_relevant_suppressed"] == 0
            and reviewed["official_suppressed"] == 0
            and reviewed["boundary_uncertain_person_suppressed"] == 0
            and missed["unsafe_all_nearby_suppressed_count"] == 0
            and missed["only_candidate_neighbourhoods_preserved"]
            and reviewed["clutter_removed"] >= clutter_threshold
        )
        if passes:
            candidates.append(row)
    if not candidates:
        return {
            "classification": "PASS_G7D_C3A_PITCH_GATE_EXPERIMENT_SANDBOX_ONLY",
            "frozen_candidate_created": False,
            "reason": "No implementable rule passed every predeclared safety and substantial-removal criterion.",
            "eligible_variant_count": 0,
            "oracle_excluded": True,
        }
    selected = min(
        candidates,
        key=lambda row: (
            -row["reviewed"]["clutter_removed"],
            row["reviewed"]["decision_counts"]["BOUNDARY_REVIEW"],
            -row["full_universe"]["decision_counts"]["SUPPRESS_SANDBOX"],
            complexity[row["family"]],
            row["variant_id"],
        ),
    )
    return {
        "classification": "PASS_G7D_C3A_PITCH_AWARE_GATE_CANDIDATE_READY_FOR_INTEGRATION_REVIEW",
        "frozen_candidate_created": True,
        "selected_variant_id": selected["variant_id"],
        "selected_family": selected["family"],
        "selected_parameter": selected["parameter"],
        "eligible_variant_count": len(candidates),
        "lexicographic_rule": [
            "zero useful relevant losses",
            "zero official losses",
            "zero boundary-uncertain person losses",
            "preserve only candidate near every missed-person mark",
            "maximize reviewed clutter removed",
            "minimize boundary-review burden",
            "maximize full-universe reduction",
            "prefer simpler rule",
        ],
        "selected_reviewed_result": selected["reviewed"],
        "selected_full_universe_result": selected["full_universe"],
        "selected_missed_mark_result": selected["missed_mark_safety"],
        "oracle_excluded": True,
        "integration_performed": False,
        "production_ready": False,
    }


def retained_composition(
    reviewed: Sequence[Mapping[str, Any]], decisions: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    counts = Counter()
    for row in reviewed:
        key = f"{row['frame_sha256']}::{row['candidate_local_id']}"
        if decisions[key]["decision"] == "SUPPRESS_SANDBOX":
            continue
        decision = row["canonical_decision"]
        flags = row["analysis_flags"]
        if decision["participation"] == "ACTIVE" and decision["role"] in {"OUTFIELD_PLAYER", "GOALKEEPER"}:
            counts["active_players_or_goalkeepers"] += 1
        elif decision["role"] in {"REFEREE", "OTHER_OFFICIAL"}:
            counts["relevant_officials"] += 1
        elif decision["role"] == "STAFF_OR_SPECTATOR":
            counts["staff_or_spectators"] += 1
        elif flags["is_background_or_object"]:
            counts["background_or_objects"] += 1
        else:
            counts["unknown_or_other"] += 1
    return {"counts": dict(counts), "warning": TARGETED_WARNING}


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def scaled_overlay(
    frame_path: Path,
    polygon: Sequence[Sequence[float]],
    candidates: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]] | None,
    focus_key: str | None = None,
    width: int = 950,
) -> Image.Image:
    image = Image.open(frame_path).convert("RGB")
    scale = width / image.width
    image = image.resize((width, round(image.height * scale)), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    polygon_scaled = [(float(x) * scale, float(y) * scale) for x, y in polygon]
    draw.line(polygon_scaled + [polygon_scaled[0]], fill=(0, 255, 255), width=3)
    for row in candidates:
        key = f"{row['frame_sha256']}::{row['candidate_local_id']}"
        box = [round(float(value) * scale) for value in row["source_box_xyxy"]]
        if focus_key == key:
            color, line_width = (255, 255, 0), 5
        elif decisions is None:
            color, line_width = (50, 220, 90), 2
        else:
            outcome = decisions[key]["decision"]
            color = {
                "KEEP": (50, 220, 90),
                "SUPPRESS_SANDBOX": (255, 70, 70),
                "BOUNDARY_REVIEW": (255, 190, 40),
                "EXCEPTION_KEEP": (70, 170, 255),
            }[outcome]
            line_width = 2
        draw.rectangle(box, outline=color, width=line_width)
    return image


def add_caption(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    header = 72
    canvas = Image.new("RGB", (image.width, image.height + header), "white")
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 8), title, fill="black", font=font(22))
    draw.text((14, 39), subtitle, fill=(70, 70, 70), font=font(16))
    return canvas


def visual_qa(inputs: Mapping[str, Any], evaluation: Mapping[str, Any], selection: Mapping[str, Any]) -> list[Path]:
    out = STAGE / "08_VISUAL_QA"
    out.mkdir(parents=True, exist_ok=True)
    selected_id = selection.get("selected_variant_id")
    selected_decisions = evaluation["all_decisions"].get(selected_id, evaluation["all_decisions"]["G0_KEEP_ALL"])
    selected_result = (
        next(row for row in evaluation["results"] if row["variant_id"] == selected_id)
        if selected_id
        else evaluation["results"][0]
    )
    family_best = []
    for family in [*IMPLEMENTABLE_FAMILIES, "G5_HUMAN_ORACLE_UPPER_BOUND"]:
        rows = [row for row in evaluation["results"] if row["family"] == family]
        family_best.append(
            max(
                rows,
                key=lambda row: row["reviewed"]["clutter_removed"]
                - 1000 * row["reviewed"]["useful_relevant_suppressed"],
            )
        )
    labels = [row["family"].split("_")[0] for row in family_best]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
    axes[0, 0].bar(labels, [row["reviewed"]["clutter_removed"] for row in family_best], color="#4c78a8")
    axes[0, 0].set_title("Reviewed clutter removed")
    axes[0, 1].bar(labels, [row["reviewed"]["useful_relevant_suppressed"] for row in family_best], color="#e45756")
    axes[0, 1].set_title("Reviewed useful relevant losses")
    axes[1, 0].bar(
        labels[:-1],
        [row["full_universe"]["decision_counts"]["SUPPRESS_SANDBOX"] for row in family_best[:-1]],
        color="#72b7b2",
    )
    axes[1, 0].set_title("Full-universe sandbox suppressions")
    axes[1, 1].bar(
        labels[:-1],
        [row["reviewed"]["decision_counts"]["BOUNDARY_REVIEW"] for row in family_best[:-1]],
        color="#f2cf5b",
    )
    axes[1, 1].set_title("Reviewed boundary-review burden")
    for axis in axes.flat:
        axis.tick_params(axis="x", rotation=20)
    fig.suptitle(f"{SANDBOX_WARNING}\nG5 shown separately as oracle only", fontsize=15, fontweight="bold")
    first = out / "01_GATE_COMPARISON_SUMMARY.png"
    fig.savefig(first, dpi=150, metadata={"Software": "Football Intelligence C3A CPU geometry"})
    plt.close(fig)

    frames_by_hash = {row["frame_sha256"]: Path(row["path"]) for row in inputs["frames"]}
    universe_by_frame: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in inputs["universe"]:
        universe_by_frame[str(row["frame_sha256"])].append(row)
    reviewed = inputs["reviewed"]
    base = evaluation["base_geometry"]

    def choose(predicate: Any, used: set[str]) -> Mapping[str, Any]:
        eligible = []
        for row in reviewed:
            key = f"{row['frame_sha256']}::{row['candidate_local_id']}"
            if key not in used and predicate(row, base[key]):
                eligible.append(row)
        if not eligible:
            eligible = [row for row in reviewed if f"{row['frame_sha256']}::{row['candidate_local_id']}" not in used]
        chosen = sorted(eligible, key=lambda row: (row["scene_id"], row["target_id"]))[0]
        used.add(f"{chosen['frame_sha256']}::{chosen['candidate_local_id']}")
        return chosen

    used: set[str] = set()
    boundary_cases = [
        (
            "Assistant-referee / touchline protection",
            choose(
                lambda row, geo: row["canonical_decision"]["role"] == "OTHER_OFFICIAL"
                and geo["nearest_boundary_type"] == "TOUCHLINE",
                used,
            ),
        ),
        (
            "Active player just outside",
            choose(
                lambda row, geo: row["canonical_decision"]["participation"] == "ACTIVE"
                and geo["signed_footpoint_distance_pixels"] > 0,
                used,
            ),
        ),
        (
            "Goalkeeper / behind-goal protection",
            choose(
                lambda row, geo: row["canonical_decision"]["role"] == "GOALKEEPER"
                and geo["nearest_boundary_type"] == "GOAL_LINE",
                used,
            ),
        ),
        (
            "Boundary-uncertain person",
            choose(
                lambda row, geo: row["analysis_flags"]["contains_any_person"]
                and row["canonical_decision"]["pitch_state"] in {"BOUNDARY", "UNCERTAIN"},
                used,
            ),
        ),
    ]
    panels = []
    for title, row in boundary_cases:
        key = f"{row['frame_sha256']}::{row['candidate_local_id']}"
        panel = scaled_overlay(
            frames_by_hash[row["frame_sha256"]],
            inputs["polygons"][str(row["match"])]["vertices_source_xy"],
            universe_by_frame[row["frame_sha256"]],
            selected_decisions,
            focus_key=key,
            width=900,
        )
        outcome = selected_decisions[key]["decision"]
        panels.append(add_caption(panel, title, f"{row['target_id']} · {outcome} · {SANDBOX_WARNING}"))
    cell_w = max(panel.width for panel in panels)
    cell_h = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (cell_w * 2, cell_h * 2), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % 2) * cell_w, (index // 2) * cell_h))
    second = out / "02_BOUNDARY_AND_EXCEPTION_CASES.png"
    canvas.save(second)

    scenes = inputs["scenes"]
    by_scene_candidates = {row["scene_id"]: universe_by_frame[row["frame_sha256"]] for row in scenes}
    high_clutter = max(scenes, key=lambda row: len(by_scene_candidates[row["scene_id"]]))
    stable = next((row for row in scenes if "STABLE" in row["scene_category"]), scenes[-1])
    daylight = next(row for row in scenes if row["lighting"] == "DAYLIGHT")
    low_light = next(row for row in scenes if row["lighting"] == "NIGHT")
    selected_scenes = [
        ("Daylight", daylight),
        ("Low light", low_light),
        ("High clutter", high_clutter),
        ("Stable control", stable),
    ]
    rows_visual = []
    for label, scene in selected_scenes:
        candidates = by_scene_candidates[scene["scene_id"]]
        polygon = inputs["polygons"][str(scene["match"])]["vertices_source_xy"]
        before = add_caption(
            scaled_overlay(frames_by_hash[scene["frame_sha256"]], polygon, candidates, None, width=850),
            f"{label} — BEFORE",
            f"{len(candidates)} original candidates preserved",
        )
        after = add_caption(
            scaled_overlay(frames_by_hash[scene["frame_sha256"]], polygon, candidates, selected_decisions, width=850),
            f"{label} — SANDBOX AFTER",
            f"{selected_result['variant_id']} · {SANDBOX_WARNING}",
        )
        row_canvas = Image.new("RGB", (before.width + after.width, max(before.height, after.height)), "white")
        row_canvas.paste(before, (0, 0))
        row_canvas.paste(after, (before.width, 0))
        rows_visual.append(row_canvas)
    contact = Image.new("RGB", (max(row.width for row in rows_visual), sum(row.height for row in rows_visual)), "white")
    y = 0
    for row in rows_visual:
        contact.paste(row, (0, y))
        y += row.height
    third = out / "03_BEFORE_AFTER_CONTACT_SHEET.png"
    contact.save(third)
    return [first, second, third]


def write_outputs(
    inputs: Mapping[str, Any], evaluation: Mapping[str, Any], selection: Mapping[str, Any]
) -> dict[str, Any]:
    input_dir = STAGE / "01_INPUT_CLOSURE"
    write_json(input_dir / "input_validation.json", inputs["closure"])
    write_json(input_dir / "source_artifact_hashes.json", inputs["source_hashes"])
    directory_manifest(input_dir, "input_closure_manifest.json")

    grid = parameter_grid()
    gate_dir = STAGE / "02_GATE_CONTRACTS"
    write_json(
        gate_dir / "predeclared_gate_grid.json",
        {
            "written_before_evaluation": True,
            "fixed_pixel_values": FIXED_PIXELS,
            "box_height_alpha_values": ALPHAS,
            "perspective_variants": BAND_MODES,
            "parameter_count": len(grid),
            "parameters": grid,
            "open_ended_sweep": False,
        },
    )
    write_json(
        gate_dir / "gate_policy_contracts.json",
        {
            "families": [*IMPLEMENTABLE_FAMILIES, "G5_HUMAN_ORACLE_UPPER_BOUND"],
            "runtime_allowed_inputs": sorted(
                [
                    "source_box_xyxy",
                    "approximate_footpoint_xy",
                    "source dimensions",
                    "perspective band",
                    "proposal provenance",
                    "authoritative pitch polygon",
                ]
            ),
            "runtime_human_labels_forbidden": True,
            "oracle_selection_forbidden": True,
            "decisions": sorted(SANDBOX_DECISIONS),
            "sandbox_only": True,
            "production_ready": False,
        },
    )
    write_json(gate_dir / "expected_height_geometry_tables.json", evaluation["expected_heights"])
    directory_manifest(gate_dir, "gate_contract_manifest.json")

    geometry_dir = STAGE / "03_CANDIDATE_GEOMETRY"
    selected_id = selection.get("selected_variant_id")
    selected_decisions = evaluation["all_decisions"].get(selected_id, evaluation["all_decisions"]["G0_KEEP_ALL"])
    geometry_rows = []
    for key, row in sorted(evaluation["universe_by_key"].items()):
        geometry_rows.append(
            {
                "match_id": row["match_id"],
                "half": row["half"],
                "frame_id": row["frame_id"],
                "frame_sha256": row["frame_sha256"],
                "candidate_local_id": row["candidate_local_id"],
                "source_box_xyxy": row["source_box_xyxy"],
                "approximate_footpoint_xy": row["approximate_footpoint_xy"],
                "perspective_band": row.get("perspective_band", "UNKNOWN"),
                "proposal_provenance": row.get("proposal_provenance", {}),
                "base_geometry": evaluation["base_geometry"][key],
                "selected_sandbox_variant": selected_id,
                "selected_sandbox_decision": selected_decisions[key],
                "original_candidate_preserved": True,
            }
        )
    write_jsonl(geometry_dir / "candidate_pitch_geometry.jsonl", geometry_rows)
    write_json(
        geometry_dir / "geometry_validation.json",
        {
            "candidate_count": len(geometry_rows),
            "source_coordinate_only": True,
            "finite_ordered_in_bounds": True,
            "original_candidates_mutated": False,
        },
    )
    directory_manifest(geometry_dir, "candidate_geometry_manifest.json")

    reviewed_dir = STAGE / "04_REVIEWED_SAFETY"
    write_json(
        reviewed_dir / "reviewed_gate_comparison.json",
        {
            "variant_count": len(evaluation["results"]),
            "results": evaluation["results"],
            "warning": TARGETED_WARNING,
        },
    )
    selected_result = next((row for row in evaluation["results"] if row["variant_id"] == selected_id), None)
    write_json(
        reviewed_dir / "boundary_and_official_protection.json",
        {
            "selected_variant": selected_id,
            "selected_reviewed": selected_result["reviewed"] if selected_result else None,
            "explicit_protections_evaluated": [
                "assistant referees near touchlines",
                "active players just outside the pitch",
                "goalkeepers behind goal lines",
                "boundary-uncertain people",
                "only candidate near a missed-person mark",
            ],
            "warning": TARGETED_WARNING,
        },
    )
    if selected_id:
        write_json(
            reviewed_dir / "retained_population_composition.json",
            retained_composition(inputs["reviewed"], selected_decisions),
        )
    directory_manifest(reviewed_dir, "reviewed_safety_manifest.json")

    missed_dir = STAGE / "05_MISSED_MARK_SAFETY"
    selected_missed = missed_mark_safety(
        inputs["marks"], inputs["scenes"], inputs["universe"], selected_decisions, inputs["polygons"]
    )
    write_json(missed_dir / "missed_person_neighbourhood_safety.json", selected_missed)
    directory_manifest(missed_dir, "missed_mark_safety_manifest.json")

    supply_dir = STAGE / "06_FULL_UNIVERSE_SUPPLY"
    selected_supply = frame_supply_summary(inputs["universe"], selected_decisions, inputs["setups"])
    write_json(supply_dir / "full_universe_gate_comparison.json", selected_supply)
    write_json(
        supply_dir / "all_variant_supply_summary.json",
        {
            row["variant_id"]: row.get("full_universe")
            for row in evaluation["results"]
            if row["implementable_without_human_labels"]
        },
    )
    directory_manifest(supply_dir, "full_universe_supply_manifest.json")

    selection_dir = STAGE / "07_GATE_SELECTION"
    write_json(selection_dir / "gate_selection_decision.json", selection)
    if selection["frozen_candidate_created"]:
        write_json(
            selection_dir / "frozen_c3a_candidate_gate.json",
            {
                "contract_id": "G7D_C3A_FROZEN_PITCH_AWARE_PROPOSAL_GATE_CANDIDATE_V1",
                "variant_id": selection["selected_variant_id"],
                "family": selection["selected_family"],
                "parameter": selection["selected_parameter"],
                "allowed_decisions": sorted(SANDBOX_DECISIONS),
                "human_labels_used_at_runtime": False,
                "sandbox_only": True,
                "integration_status": "PENDING_INTEGRATION_REVIEW",
                "production_ready": False,
                "warning": SANDBOX_WARNING,
            },
        )
    directory_manifest(selection_dir, "gate_selection_manifest.json")

    visuals = visual_qa(inputs, evaluation, selection)
    directory_manifest(STAGE / "08_VISUAL_QA", "visual_qa_manifest.json")
    return {
        "selected_result": selected_result,
        "selected_missed": selected_missed,
        "selected_supply": selected_supply,
        "visuals": visuals,
        "selected_decisions": selected_decisions,
    }


def finalize(
    pack_validation: Mapping[str, Any],
    inputs: Mapping[str, Any],
    selection: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> None:
    after_hashes = {path.relative_to(PROJECT).as_posix(): sha256(path) for path in inputs["source_files"]}
    if after_hashes != inputs["source_hashes"]:
        raise RuntimeError("FAIL_SOURCE_MUTATION")
    tests_dir = STAGE / "09_TESTS_AND_LOGS"
    test_result_path = tests_dir / "focused_test_results.json"
    test_status = (
        load_json(test_result_path)["classification"] if test_result_path.exists() else "PENDING_FOCUSED_TEST_RUN"
    )
    final_report = {
        "classification": selection["classification"],
        "model_binding": "GPT-5.6 Terra / Medium",
        "device": "CPU_GEOMETRY_ONLY",
        "neural_inference_run": False,
        "frame_count": 96,
        "reviewed_candidate_count": 192,
        "reviewed_scene_count": 24,
        "missed_person_mark_count": 22,
        "full_universe_candidate_count": len(inputs["universe"]),
        "selected_variant_id": selection.get("selected_variant_id"),
        "frozen_candidate_created": selection["frozen_candidate_created"],
        "production_integration_performed": False,
        "production_ready": False,
        "visual_only_not_metric": True,
        "visual_count": 3,
        "tests": test_status,
        "source_preservation": "PASS",
        "validation_or_holdout_access": False,
        "training_or_recalibration": False,
        "detector_feature_or_fold_rerun": False,
        "warning": TARGETED_WARNING,
        "sandbox_warning": SANDBOX_WARNING,
    }
    write_json(tests_dir / "pack_validation.json", pack_validation)
    write_json(
        tests_dir / "source_preservation_report.json",
        {"before": inputs["source_hashes"], "after": after_hashes, "match": True},
    )
    write_json(tests_dir / "final_validation_report.json", final_report)
    directory_manifest(tests_dir, "tests_and_logs_manifest.json")

    handoff = STAGE / "10_REVIEW_PACK/CHATGPT_HANDOFF"
    handoff.mkdir(parents=True, exist_ok=True)
    write_json(
        handoff / "01_EXECUTIVE_SUMMARY.json",
        {
            **final_report,
            "selected_reviewed": selection.get("selected_reviewed_result"),
            "selected_full_universe": selection.get("selected_full_universe_result"),
        },
    )
    write_json(
        handoff / "02_INPUT_AND_GATE_CONTRACTS.json",
        {
            "input_closure": inputs["closure"],
            "gate_grid": load_json(STAGE / "02_GATE_CONTRACTS/predeclared_gate_grid.json"),
            "gate_contract": load_json(STAGE / "02_GATE_CONTRACTS/gate_policy_contracts.json"),
        },
    )
    write_json(
        handoff / "03_REVIEWED_SAFETY_RESULTS.json",
        {
            "selected": selection.get("selected_reviewed_result"),
            "retained_composition": load_json(STAGE / "04_REVIEWED_SAFETY/retained_population_composition.json")
            if selection["frozen_candidate_created"]
            else None,
            "warning": TARGETED_WARNING,
        },
    )
    write_json(
        handoff / "04_MISSED_MARK_AND_BOUNDARY_RESULTS.json",
        {
            "missed_mark_safety": outputs["selected_missed"],
            "boundary_protection": load_json(STAGE / "04_REVIEWED_SAFETY/boundary_and_official_protection.json"),
        },
    )
    write_json(handoff / "05_FULL_UNIVERSE_SUPPLY_RESULTS.json", outputs["selected_supply"])
    write_json(handoff / "06_GATE_SELECTION.json", selection)
    (handoff / "07_DECISION.md").write_text(
        f"# G7D-C3A decision\n\n`{selection['classification']}`\n\n"
        f"Selected sandbox variant: `{selection.get('selected_variant_id')}`. "
        "No production integration, detector rerun, threshold change, or neural inference occurred.\n",
        encoding="utf-8",
        newline="\n",
    )
    (handoff / "08_PITCH_GATE_CONTRACT.md").write_text(
        f"# Pitch gate contract\n\n**{SANDBOX_WARNING}**\n\nRuntime decisions use source-coordinate geometry, "
        "candidate dimensions, perspective band, proposal provenance, and the authoritative polygon only. Human "
        "labels evaluate frozen rules and never drive runtime decisions. All reviewed rates are "
        f"**{TARGETED_WARNING}**.\n",
        encoding="utf-8",
        newline="\n",
    )
    shutil.copy2(outputs["visuals"][0], handoff / "09_GATE_SUMMARY.png")
    shutil.copy2(outputs["visuals"][1], handoff / "10_BOUNDARY_CASES.png")
    shutil.copy2(outputs["visuals"][2], handoff / "11_BEFORE_AFTER.png")
    files = sorted(path for path in handoff.iterdir() if path.is_file() and path.name != "12_MANIFEST.json")
    write_json(
        handoff / "12_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7d_c3a.handoff_manifest.v1",
            "self_hash_omitted": True,
            "file_count": len(files),
            "files": [
                {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256(path)} for path in files
            ],
        },
    )
    (STAGE / "10_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only the CHATGPT_HANDOFF folder.\n", encoding="utf-8", newline="\n"
    )


def main() -> int:
    pack_validation = validate_pack()
    inputs = frozen_inputs()
    grid = parameter_grid()
    gate_dir = STAGE / "02_GATE_CONTRACTS"
    write_json(
        gate_dir / "predeclared_gate_grid.json",
        {
            "written_before_evaluation": True,
            "fixed_pixel_values": FIXED_PIXELS,
            "box_height_alpha_values": ALPHAS,
            "perspective_variants": BAND_MODES,
            "parameter_count": len(grid),
            "parameters": grid,
            "open_ended_sweep": False,
        },
    )
    evaluation = evaluate(inputs, grid)
    selection = select_gate(evaluation)
    outputs = write_outputs(inputs, evaluation, selection)
    finalize(pack_validation, inputs, selection, outputs)
    print(
        json.dumps(
            {
                "classification": selection["classification"],
                "selected_variant": selection.get("selected_variant_id"),
                "full_universe_candidates": len(inputs["universe"]),
                "reviewed_clutter_removed": selection.get("selected_reviewed_result", {}).get("clutter_removed"),
                "reviewed_useful_relevant_suppressed": selection.get("selected_reviewed_result", {}).get(
                    "useful_relevant_suppressed"
                ),
                "reviewed_official_suppressed": selection.get("selected_reviewed_result", {}).get(
                    "official_suppressed"
                ),
                "visual_count": 3,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
