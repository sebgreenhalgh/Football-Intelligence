"""Launch the G7F-C R1 reviewer against the real calibration decision root."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from football_intelligence.dense_person_reviewer import DensePersonReviewerConfig, run_server


REVIEWER_RELEASE = "G7F_C_DENSE_PERSON_REVIEWER_R1"
EXPECTED_HASHES = {
    "01_SELECTION/dense_gold_selection_manifest.json": (
        "f454e0e93ec2cb01f5edeba1b6545a9e607a26ec8ec58528bccc7f5e12618eef"
    ),
    "02_ONTOLOGY/dense_person_gold_ontology.json": ("410e7a704ed750fc6e08b064d08578075c751f556d2439f7e15a350430beac5b"),
    "03_METRICS/dense_gold_metric_protocol.json": ("c28a9d467719dae1944abe2594080e45300b40d46d79e68d636ee1601c8262cd"),
    "04_SCHEMAS/dense_person_frame_annotation.schema.json": (
        "0449998705cffec7d718f7c00cb2249f8b6a35f933a7768aa76db12ea20e8d0c"
    ),
    "04_SCHEMAS/dense_person_frame_acknowledgement.schema.json": (
        "cc902281a2f5fa14f97a3ad186c8bd565ed748855c6fd445cf35f906cbf94d75"
    ),
}


def roots(repo: Path) -> dict[str, Path]:
    part9 = repo.parent / "experiments/football_observation_reasoner/part 9"
    return {
        "repo": repo,
        "source": part9 / "G7F_C_DENSE_PERSON_GOLD_DISCRIMINATION_SET_v1",
        "repair": part9 / "G7F_C_R1_REVIEWER_INTERACTION_AND_UI_REPAIR_v1",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_runtime_and_contracts(paths: dict[str, Path]) -> dict[str, str]:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"G7F-C R1 requires Python 3.12; found {sys.version.split()[0]}")
    source = paths["source"]
    actual = {relative: sha256_file(source / relative) for relative in EXPECTED_HASHES}
    if actual != EXPECTED_HASHES:
        raise RuntimeError(f"frozen G7F-C contract hash mismatch: {actual}")
    bindings_path = paths["repair"] / "05_REVIEWER/reviewer_binding_hashes.json"
    if not bindings_path.is_file():
        raise RuntimeError(f"R1 release bindings are missing: {bindings_path}")
    bindings = read_json(bindings_path)
    expected_binding_values = {
        "selection_manifest_sha256": EXPECTED_HASHES["01_SELECTION/dense_gold_selection_manifest.json"],
        "ontology_sha256": EXPECTED_HASHES["02_ONTOLOGY/dense_person_gold_ontology.json"],
        "metric_protocol_sha256": EXPECTED_HASHES["03_METRICS/dense_gold_metric_protocol.json"],
        "event_schema_sha256": EXPECTED_HASHES["04_SCHEMAS/dense_person_frame_annotation.schema.json"],
        "ack_schema_sha256": EXPECTED_HASHES["04_SCHEMAS/dense_person_frame_acknowledgement.schema.json"],
    }
    for key, value in expected_binding_values.items():
        if bindings.get(key) != value:
            raise RuntimeError(f"R1 binding mismatch for {key}")
    return bindings


def reviewer_config(repo: Path, *, port: int = 8791) -> DensePersonReviewerConfig:
    paths = roots(repo)
    bindings = verify_runtime_and_contracts(paths)
    source = paths["source"]
    return DensePersonReviewerConfig(
        selection_manifest_path=source / "01_SELECTION/dense_gold_selection_manifest.json",
        assets_root=source / "05_REVIEWER/assets",
        decisions_root=source / "06_DENSE_DECISIONS",
        binding_hashes=bindings,
        reviewer_release=REVIEWER_RELEASE,
        reveal_payload_path=source / "05_REVIEWER/sealed_candidate_reveal_payloads.json",
        pass_kind="FIRST_PASS",
        allowed_selection_statuses=("CALIBRATION_ONLY",),
        port=port,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check", "serve"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--port", type=int, default=8791)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = reviewer_config(args.repo.resolve(), port=args.port)
    if args.phase == "check":
        print(
            json.dumps(
                {
                    "reviewer_release": REVIEWER_RELEASE,
                    "workflow": "CALIBRATION_ONLY",
                    "scored_annotation_authorized": False,
                    "production_ready": False,
                },
                sort_keys=True,
            )
        )
        return 0
    print(f"G7F-C R1 calibration reviewer: http://{config.host}:{config.port}/")
    print("Scored annotation is not authorized. production_ready=false")
    run_server(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
