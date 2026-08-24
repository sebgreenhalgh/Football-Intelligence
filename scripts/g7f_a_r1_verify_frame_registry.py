from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED = {
    "frame_instances": 1080,
    "unique_source_hashes": 1044,
    "old_known_hashes": 892,
    "formerly_omitted_hashes": 152,
    "affected_instances": 155,
    "formerly_blocked_candidates": 7575,
    "frozen_candidates": 49803,
}
REGRESSION_HASH = "4854dec974a57d5f65177268ef055ee12965e72c54ce147fdfd24c7cb2158b48"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def verify(original_workspace: Path, reviewer_package: Path, r1_workspace: Path) -> dict[str, Any]:
    cases = read_json(reviewer_package / "review_cases.json")["cases"]
    states = read_json(reviewer_package / "candidate_states_by_reference.json")["frames"]
    gold = original_workspace / "02_NORMALIZED_GOLD"
    subject_rows = read_jsonl(gold / "gold_subject_frames.jsonl")
    missed_rows = read_jsonl(gold / "gold_missed_observations.jsonl")
    old_hashes = {row["source_frame_sha256"] for row in [*subject_rows, *missed_rows]}
    subject_keys = {
        (row["burst_id"], row["frame_sequence"])
        for row in subject_rows
        if row["human_confirmed_source_coordinate"] is not None
    }
    missed_keys = {(row["burst_id"], row["frame_sequence"]) for row in missed_rows}

    expected_instances: dict[tuple[str, int], dict[str, Any]] = {}
    expected_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_fingerprints: dict[str, str] = {}
    for case in cases:
        assert len(case["frames"]) == 9
        for sequence, frame in enumerate(case["frames"]):
            assert frame["burst_frame_sequence"] == sequence
            state = states[frame["frame_reference_id"]]
            assert state["frame_pixel_sha256"] == frame["source_frame_pixel_sha256"]
            assert state["candidates"] == case["frame_candidates"][sequence]
            frame_hash = frame["source_frame_pixel_sha256"]
            fingerprint = hashlib.sha256(
                json.dumps(state["candidates"], sort_keys=True, separators=(",", ":")).encode("ascii")
            ).hexdigest()
            if frame_hash in candidate_fingerprints:
                assert candidate_fingerprints[frame_hash] == fingerprint
            candidate_fingerprints[frame_hash] = fingerprint
            key = (case["burst_id"], sequence)
            expected = {
                "burst_id": case["burst_id"],
                "frame_sequence": sequence,
                "frame_reference_id": frame["frame_reference_id"],
                "source_frame_sha256": frame_hash,
                "source_width": frame["source_width"],
                "source_height": frame["source_height"],
                "historical_frozen_candidate_count": len(state["candidates"]),
                "has_reviewed_subject_point": key in subject_keys,
                "has_missed_observation_point": key in missed_keys,
            }
            assert key not in expected_instances
            expected_instances[key] = expected
            expected_by_hash[frame_hash].append(expected)

    generated_instance_rows = read_jsonl(r1_workspace / "02_NORMALIZED_GOLD" / "gold_frame_instances.jsonl")
    generated_instances = {(row["burst_id"], row["frame_sequence"]): row for row in generated_instance_rows}
    assert len(generated_instances) == len(generated_instance_rows)
    assert set(generated_instances) == set(expected_instances)
    for key, expected in expected_instances.items():
        generated = generated_instances[key]
        for field, value in expected.items():
            assert generated[field] == value, f"instance mismatch {key}/{field}"

    generated_source_rows = read_jsonl(r1_workspace / "02_NORMALIZED_GOLD" / "gold_source_frame_registry.jsonl")
    generated_sources = {row["source_frame_sha256"]: row for row in generated_source_rows}
    assert len(generated_sources) == len(generated_source_rows)
    assert set(generated_sources) == set(expected_by_hash)
    for frame_hash, expected_rows in expected_by_hash.items():
        generated = generated_sources[frame_hash]
        dimensions = {(row["source_width"], row["source_height"]) for row in expected_rows}
        candidate_counts = {row["historical_frozen_candidate_count"] for row in expected_rows}
        assert len(dimensions) == len(candidate_counts) == 1
        width, height = next(iter(dimensions))
        assert (generated["source_width"], generated["source_height"]) == (width, height)
        assert generated["instance_count"] == len(expected_rows)
        assert generated["frozen_candidate_count"] == next(iter(candidate_counts))
        assert generated["historical_frozen_candidate_count_across_instances"] == sum(
            row["historical_frozen_candidate_count"] for row in expected_rows
        )
        expected_membership = [
            {
                "burst_id": row["burst_id"],
                "frame_sequence": row["frame_sequence"],
                "frame_reference_id": row["frame_reference_id"],
            }
            for row in sorted(expected_rows, key=lambda item: (item["burst_id"], item["frame_sequence"]))
        ]
        assert generated["frame_instances"] == expected_membership

    full_hashes = set(expected_by_hash)
    omitted_hashes = full_hashes - old_hashes
    affected = [row for row in expected_instances.values() if row["source_frame_sha256"] in omitted_hashes]
    counts = {
        "frame_instances": len(expected_instances),
        "unique_source_hashes": len(full_hashes),
        "old_known_hashes": len(old_hashes),
        "formerly_omitted_hashes": len(omitted_hashes),
        "affected_instances": len(affected),
        "formerly_blocked_candidates": sum(row["historical_frozen_candidate_count"] for row in affected),
        "frozen_candidates": sum(row["historical_frozen_candidate_count"] for row in expected_instances.values()),
    }
    assert counts == EXPECTED
    assert REGRESSION_HASH in omitted_hashes
    return {
        "schema_version": "football_intelligence.g7f_a_r1.independent_frame_population_verification.v1",
        "classification": "PASS_G7F_A_R1_INDEPENDENT_FRAME_POPULATION_VERIFICATION",
        "method": (
            "Independent direct derivation from frozen review_cases and candidate_states; " "no registry helper used"
        ),
        "counts": counts,
        "exact_instance_registry_equality": True,
        "exact_source_registry_equality": True,
        "old_known_hashes_are_subset": old_hashes < full_hashes,
        "regression_hash_verified": REGRESSION_HASH,
        "frame_instance_registry_sha256": sha256_file(
            r1_workspace / "02_NORMALIZED_GOLD" / "gold_frame_instances.jsonl"
        ),
        "source_frame_registry_sha256": sha256_file(
            r1_workspace / "02_NORMALIZED_GOLD" / "gold_source_frame_registry.jsonl"
        ),
        "production_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Independently verify G7F-A R1 frame registries")
    parser.add_argument("--original-workspace", required=True, type=Path)
    parser.add_argument("--reviewer-package", required=True, type=Path)
    parser.add_argument("--r1-workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = verify(
        args.original_workspace.resolve(),
        args.reviewer_package.resolve(),
        args.r1_workspace.resolve(),
    )
    write_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
