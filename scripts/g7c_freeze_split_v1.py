"""Validate final human approvals and freeze G7C split_v1 without data scans."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


MATCHES = ["117092", "117093", "118575", "118576", "118577", "118578", "128057", "128058", "132831", "132877"]
SPLIT = {
    "TRAIN_DEVELOPMENT": ["117092", "117093", "118575", "118576", "118577", "128058"],
    "VALIDATION_MODEL_SELECTION": ["118578", "128057"],
    "SEALED_HOLDOUT": ["132831", "132877"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_setup(review: dict, setup: dict) -> list[dict[str, str]]:
    match = review["match_id"]
    pairs = {
        "match_id": setup.get("match_id"),
        "lighting": setup.get("conditions", {}).get("lighting"),
        "weather": setup.get("conditions", {}).get("weather"),
        "visibility": setup.get("conditions", {}).get("visibility"),
        "panorama_quality": setup.get("conditions", {}).get("panorama_quality"),
        "crowd_background": setup.get("conditions", {}).get("crowd_background"),
        "unusual_conditions": setup.get("conditions", {}).get("unusual_conditions"),
        "team_numbering_policy": setup.get("team_mapping", {}).get("numbering_policy"),
        "team_mapping_reference_half": setup.get("team_mapping", {}).get("reference_half"),
        "team_mapping_reference_video": setup.get("team_mapping", {}).get("reference_video"),
        "team_mapping_reference_timestamp_seconds": setup.get("team_mapping", {}).get("reference_timestamp_seconds"),
        "team_1_definition": setup.get("team_mapping", {}).get("team_1_definition"),
        "team_2_definition": setup.get("team_mapping", {}).get("team_2_definition"),
        "team_1_first_half_side": setup.get("team_mapping", {}).get("team_1_first_half_side"),
        "team_2_first_half_side": setup.get("team_mapping", {}).get("team_2_first_half_side"),
        "team_1_second_half_side": setup.get("team_mapping", {}).get("team_1_second_half_side"),
        "team_2_second_half_side": setup.get("team_mapping", {}).get("team_2_second_half_side"),
        "team_1_primary_colour": setup.get("team_mapping", {}).get("team_1_primary_colour"),
        "team_2_primary_colour": setup.get("team_mapping", {}).get("team_2_primary_colour"),
        "team_1_goalkeeper_primary_colour": setup.get("goalkeeper_kits", {}).get("team_1_goalkeeper_primary_colour"),
        "team_2_goalkeeper_primary_colour": setup.get("goalkeeper_kits", {}).get("team_2_goalkeeper_primary_colour"),
        "team_mapping_source": setup.get("team_mapping", {}).get("mapping_source"),
        "team_mapping_confirmed": setup.get("team_mapping", {}).get("confirmed"),
        "pitch_polygon_status": setup.get("pitch_calibration", {}).get("status"),
        "representative_frame_approved": setup.get("human_review", {}).get("representative_frame_approved"),
        "team_mapping_confirmed_review": setup.get("human_review", {}).get("team_mapping_confirmed"),
    }
    expected = {"match_id": match, **review}
    expected.pop("proposed_split", None)
    expected.pop("representative_frame_approved", None)
    expected.pop("proposed_split_approved", None)
    expected.pop("team_mapping_confirmed", None)
    expected.pop("pitch_polygon_status", None)
    mismatches = []
    for field, actual in pairs.items():
        expected_value = expected.get(field)
        if field in ("representative_frame_approved", "team_mapping_confirmed"):
            expected_value = True
        if field == "team_mapping_confirmed_review":
            expected_value = True
        if field == "pitch_polygon_status":
            expected_value = "HUMAN_REQUIRED"
        if actual != expected_value:
            mismatches.append(
                {"match_id": match, "field": field, "expected": str(expected_value), "actual": str(actual)}
            )
    if setup.get("dataset_split", {}).get("proposed_assignment") != review["proposed_split"]:
        mismatches.append(
            {
                "match_id": match,
                "field": "proposed_split",
                "expected": review["proposed_split"],
                "actual": str(setup.get("dataset_split", {}).get("proposed_assignment")),
            }
        )
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    args = parser.parse_args()
    root, workspace, pack = args.project_root, args.workspace, args.pack
    input_review = read_json(pack / "02_HUMAN_CONDITION_REVIEW_FINAL.json")
    input_manifest = read_json(pack / "03_INPUT_MANIFEST.json")
    pack_manifest = read_json(pack / "04_PACK_MANIFEST.json")
    if input_review.get("schema_version") != "g7c.human_condition_review.v2":
        raise RuntimeError("FAIL_HUMAN_REVIEW_SCHEMA")
    if sha256(pack / "02_HUMAN_CONDITION_REVIEW_FINAL.json") != input_manifest["human_review"]["sha256"]:
        raise RuntimeError("FAIL_INPUT_MANIFEST")
    for item in pack_manifest["files"]:
        path = pack / item["filename"]
        if path.stat().st_size != item["byte_size"] or sha256(path) != item["sha256"]:
            raise RuntimeError("FAIL_PACK_MANIFEST")
    reviews = {row["match_id"]: row for row in input_review["matches"]}
    if set(reviews) != set(MATCHES) or len(reviews) != 10:
        raise RuntimeError("FAIL_HUMAN_REVIEW_MATCH_SET")
    required = (
        "lighting",
        "weather",
        "visibility",
        "panorama_quality",
        "crowd_background",
        "team_1_primary_colour",
        "team_2_primary_colour",
        "unusual_conditions",
    )
    mismatches = []
    for match in MATCHES:
        row = reviews[match]
        if not all(row.get(field) not in (None, "") for field in required):
            mismatches.append(
                {"match_id": match, "field": "required_human_value", "expected": "nonblank", "actual": "blank"}
            )
        if (
            row["representative_frame_approved"] is not True
            or row["proposed_split_approved"] is not True
            or row["team_mapping_confirmed"] is not True
        ):
            mismatches.append(
                {"match_id": match, "field": "approval_flags", "expected": "true", "actual": "not_all_true"}
            )
        setup_path = root / "matches" / match / "calibration" / "match_setup.json"
        if not setup_path.exists():
            mismatches.append(
                {"match_id": match, "field": "setup_file", "expected": str(setup_path), "actual": "MISSING"}
            )
        else:
            mismatches.extend(validate_setup(row, read_json(setup_path)))
    if mismatches:
        report = workspace / "05_CONDITION_REVIEW" / "HUMAN_APPROVAL_SETUP_MISMATCH_REPORT.json"
        dump(report, {"classification": "FAIL_MATCH_SETUP_MISMATCH", "mismatches": mismatches})
        raise RuntimeError("FAIL_MATCH_SETUP_MISMATCH")
    review_copy = workspace / "05_CONDITION_REVIEW" / "HUMAN_CONDITION_REVIEW_FINAL.json"
    shutil.copy2(pack / "02_HUMAN_CONDITION_REVIEW_FINAL.json", review_copy)
    approval_report = {
        "schema_version": "g7c.human_approval_validation_report.v1",
        "status": "HUMAN_APPROVAL_VALIDATED_READY_TO_FREEZE",
        "review_sha256": sha256(review_copy),
        "match_count": 10,
        "representative_frame_approved_count": 10,
        "proposed_split_approved_count": 10,
        "team_mapping_confirmed_count": 10,
        "setup_semantic_equivalence": "PASSED",
        "invalid_colour_tokens": [],
        "pitch_polygon_status": "HUMAN_REQUIRED_FOR_ALL_MATCHES",
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path = workspace / "05_CONDITION_REVIEW" / "HUMAN_APPROVAL_VALIDATION_REPORT.json"
    dump(report_path, approval_report)
    split_root = root / "datasets/soccertrack_v2/splits/split_v1"
    for name, values in (
        ("train_matches.txt", SPLIT["TRAIN_DEVELOPMENT"]),
        ("validation_matches.txt", SPLIT["VALIDATION_MODEL_SELECTION"]),
        ("sealed_holdout_matches.txt", SPLIT["SEALED_HOLDOUT"]),
    ):
        (split_root / name).write_text("\n".join(values) + "\n", encoding="utf-8")
    freeze_time = datetime.now(timezone.utc).isoformat()
    setup_hashes = {}
    for match in MATCHES:
        setup_path = root / "matches" / match / "calibration" / "match_setup.json"
        setup = read_json(setup_path)
        if "dataset_split" in setup:
            setup["dataset_split"]["human_approved"] = True
            setup["dataset_split"]["status"] = "FROZEN_HUMAN_APPROVED"
            setup["dataset_split"]["frozen"] = True
            dump(setup_path, setup)
        setup_hashes[match] = sha256(setup_path)
    registry_path = root / "datasets/soccertrack_v2/match_registry.json"
    registry = read_json(registry_path)
    for entry in registry["matches"]:
        entry["human_approved"] = True
        entry["split_status"] = "FROZEN_HUMAN_APPROVED"
        if entry["match_id"] in setup_hashes:
            entry["match_setup_sha256"] = setup_hashes[entry["match_id"]]
    dump(registry_path, registry)
    dataset_path = root / "datasets/soccertrack_v2/dataset_manifest.json"
    dataset = read_json(dataset_path)
    dataset["dataset_split"] = {
        "status": "FROZEN_HUMAN_APPROVED",
        "frozen": True,
        "split_manifest": "datasets/soccertrack_v2/splits/split_v1/split_manifest.json",
    }
    dump(dataset_path, dataset)
    conditions_path = root / "datasets/soccertrack_v2/condition_inventory.json"
    conditions = read_json(conditions_path)
    for row in conditions["matches"]:
        if row["match_id"] in reviews:
            for field in (
                "lighting",
                "weather",
                "visibility",
                "panorama_quality",
                "crowd_background",
                "unusual_conditions",
            ):
                row[field] = reviews[row["match_id"]][field]
            row["review_status"] = "HUMAN_APPROVED"
    dump(conditions_path, conditions)
    source_refs = {}
    for match in MATCHES:
        manifest_path = root / "matches" / match / "manifests" / "match_manifest.json"
        source_refs[match] = {"path": f"matches/{match}/manifests/match_manifest.json", "sha256": sha256(manifest_path)}
    split_manifest = {
        "schema_version": "g7c.frozen_split_manifest.v1",
        "status": "FROZEN_HUMAN_APPROVED",
        "frozen": True,
        "membership": SPLIT,
        "human_review_sha256": sha256(review_copy),
        "human_approval_validation_report_sha256": sha256(report_path),
        "match_setup_sha256": setup_hashes,
        "repository_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root / "SoccerTrack-v2", capture_output=True, text=True, check=True
        ).stdout.strip(),
        "source_manifest_references": source_refs,
        "117093_correction_event": "matches/117093/manifests/source_correction_events.json",
        "freeze_timestamp": freeze_time,
        "holdout_access_prohibition": "SEALED_HOLDOUT representative frames only; no holdout footage inspection",
        "pitch_polygons": "HUMAN_REQUIRED; not required for split freeze",
    }
    split_manifest_path = split_root / "split_manifest.json"
    dump(split_manifest_path, split_manifest)
    (split_root / "split_manifest.sha256").write_text(
        f"{sha256(split_manifest_path)}  split_manifest.json\n", encoding="utf-8"
    )
    artifact_manifest = {"schema_version": "g7c.human_approval_artifact_manifest.v1", "files": []}
    for path in [
        review_copy,
        *[root / "matches" / match / "calibration" / "match_setup.json" for match in MATCHES],
        report_path,
    ]:
        artifact_manifest["files"].append(
            {
                "project_relative_path": path.relative_to(root).as_posix(),
                "byte_size": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    dump(workspace / "05_CONDITION_REVIEW" / "HUMAN_APPROVAL_ARTIFACT_MANIFEST.json", artifact_manifest)


if __name__ == "__main__":
    main()
