"""Check or launch the candidate-blind G7F-C scored FIRST_PASS reviewer."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from football_intelligence.dense_person_reviewer import DensePersonReviewerConfig, run_server


REVIEWER_RELEASE = "G7F_C_SCORED_DENSE_PERSON_REVIEWER_R1"
PASS = "PASS_G7F_C_SCORED_DENSE_GOLD_FIRST_PASS_REVIEWER_READY_FOR_HUMAN_ANNOTATION"
CALIBRATION_PASS = "PASS_G7F_C_CALIBRATION_SUPERSEDING_ADJUDICATION_REAUDIT_READY_FOR_SCORED_DENSE_GOLD_ANNOTATION"
BASELINE = "455133a8ca9af1e245633a713e44659fee17595d"
VISUAL_ASSESSMENT_SHA256 = "8b438a55097bcd250ea3d8634dfed713a0df5c78f033314e8baaeab64812c1bb"
REAL_DECISIONS_INVENTORY_SHA256 = "7d6d1993e50de5d3cac88b1489efb6a2a9bc96ec6290dc6a80728aca0f76b7fe"
AUTHORITATIVE_SEQUENCES = {
    "DG-001": 1,
    "DG-002": 1,
    "DG-003": 1,
    "DG-004": 2,
    "DG-005": 2,
    "DG-006": 1,
}
FROZEN_HASHES = {
    "01_SELECTION/dense_gold_selection_manifest.json": (
        "f454e0e93ec2cb01f5edeba1b6545a9e607a26ec8ec58528bccc7f5e12618eef"
    ),
    "02_ONTOLOGY/dense_person_gold_ontology.json": (
        "410e7a704ed750fc6e08b064d08578075c751f556d2439f7e15a350430beac5b"
    ),
    "03_METRICS/dense_gold_metric_protocol.json": (
        "c28a9d467719dae1944abe2594080e45300b40d46d79e68d636ee1601c8262cd"
    ),
    "04_SCHEMAS/dense_person_frame_annotation.schema.json": (
        "0449998705cffec7d718f7c00cb2249f8b6a35f933a7768aa76db12ea20e8d0c"
    ),
    "04_SCHEMAS/dense_person_frame_acknowledgement.schema.json": (
        "cc902281a2f5fa14f97a3ad186c8bd565ed748855c6fd445cf35f906cbf94d75"
    ),
}
SEALED_HASHES = {
    "05_REVIEWER/sealed_blind_repeat_qa_selection.json": (
        "12c7185a3ac4d7d5e8ac1f2086396b63ed9c827d210da4451b086b1404ca50b8"
    ),
    "05_REVIEWER/sealed_candidate_reveal_payloads.json": (
        "8c69b80195752a84d2357d3d672c1e15d968e667625cfb481f541ee7d80cb226"
    ),
}


def roots(repo: Path) -> dict[str, Path]:
    part9 = repo.parent / "experiments/football_observation_reasoner/part 9"
    phase_b = part9 / "G7F_C_CALIBRATION_SUPERSEDING_ADJUDICATION_AND_REAUDIT_v1/11_PHASE_B"
    return {
        "repo": repo,
        "part9": part9,
        "source": part9 / "G7F_C_DENSE_PERSON_GOLD_DISCRIMINATION_SET_v1",
        "phase_b": phase_b,
        "phase_b_handoff": phase_b / "10_REVIEW_PACK/CHATGPT_HANDOFF",
        "release": part9 / "G7F_C_SCORED_DENSE_GOLD_FIRST_PASS_REVIEWER_RELEASE_v1",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen_contracts(all_paths: dict[str, Path]) -> dict[str, str]:
    source = all_paths["source"]
    actual = {relative: sha256_file(source / relative) for relative in FROZEN_HASHES}
    if actual != FROZEN_HASHES:
        raise RuntimeError(f"frozen G7F-C contract mismatch: {actual}")
    actual_sealed = {relative: sha256_file(source / relative) for relative in SEALED_HASHES}
    if actual_sealed != SEALED_HASHES:
        raise RuntimeError("sealed candidate/repeat contract hash mismatch")
    return {**actual, **actual_sealed}


def verify_phase_b_authorization(all_paths: dict[str, Path]) -> dict[str, Any]:
    handoff = all_paths["phase_b_handoff"]
    summary = read_json(handoff / "00_EXECUTIVE_SUMMARY.json")
    authorization = read_json(handoff / "07_SCORED_ANNOTATION_AUTHORIZATION.json")
    assessment = all_paths["phase_b"] / "04_CANDIDATE_FREE_VISUAL_ASSESSMENT.json"
    checks = {
        "decision": summary.get("decision") == CALIBRATION_PASS == authorization.get("decision"),
        "authorized": summary.get("SCORED_ANNOTATION_AUTHORIZED") is True
        and authorization.get("SCORED_ANNOTATION_AUTHORIZED") is True,
        "not_started": summary.get("SCORED_ANNOTATION_STARTED") is False
        and authorization.get("SCORED_ANNOTATION_STARTED") is False,
        "production_ready_false": summary.get("production_ready") is False
        and authorization.get("production_ready") is False,
        "candidate_free": summary.get("candidate_data_used") is False,
        "visual_assessment": sha256_file(assessment) == VISUAL_ASSESSMENT_SHA256
        and authorization.get("visual_assessment_sha256") == VISUAL_ASSESSMENT_SHA256,
        "authoritative_sequences": authorization.get("authoritative_sequences") == AUTHORITATIVE_SEQUENCES,
        "real_decisions_inventory": authorization.get("real_decisions_inventory_hash")
        == REAL_DECISIONS_INVENTORY_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"closed Phase-B authorization mismatch: {checks}")
    return {"checks": checks, "authorization": authorization}


def scored_queue(selection: dict[str, Any]) -> list[dict[str, Any]]:
    images = selection.get("images")
    if not isinstance(images, list) or len(images) != 54:
        raise RuntimeError("frozen selection must contain exactly 54 images")
    counts = Counter(str(row.get("selection_status")) for row in images)
    expected = {"CALIBRATION_ONLY": 6, "SCORED_DENSE_GOLD": 48}
    if dict(counts) != expected:
        raise RuntimeError(f"selection status counts differ from 6/48 contract: {counts}")
    rows = sorted(
        (row for row in images if row["selection_status"] == "SCORED_DENSE_GOLD"),
        key=lambda row: (int(row["review_queue_position"]), str(row["anonymous_dense_image_id"])),
    )
    calibration_ids = {
        row["anonymous_dense_image_id"]
        for row in images
        if row["selection_status"] == "CALIBRATION_ONLY"
    }
    queue_ids = [str(row["anonymous_dense_image_id"]) for row in rows]
    source_hashes = [str(row["source_frame_sha256"]) for row in rows]
    if len(rows) != 48 or calibration_ids.intersection(queue_ids):
        raise RuntimeError("scored queue contains calibration contamination")
    if len(set(queue_ids)) != 48 or len(set(source_hashes)) != 48:
        raise RuntimeError("scored queue IDs and source-frame hashes must be unique")
    if calibration_ids != set(AUTHORITATIVE_SEQUENCES):
        raise RuntimeError("frozen calibration identities differ from DG-001..DG-006")
    return rows


def scored_decision_files(decisions_root: Path, scored_ids: set[str]) -> list[str]:
    found: list[str] = []
    for path in sorted(decisions_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(decisions_root).as_posix()
        if any(image_id in path.name for image_id in scored_ids):
            found.append(relative)
            continue
        if relative.startswith("action_receipts/"):
            try:
                response = read_json(path).get("response", {})
            except (json.JSONDecodeError, OSError):
                continue
            if response.get("anonymous_dense_image_id") in scored_ids:
                found.append(relative)
    return found


def release_bindings(all_paths: dict[str, Path]) -> dict[str, str]:
    path = all_paths["release"] / "05_REVIEWER/reviewer_binding_hashes.json"
    if not path.is_file():
        raise RuntimeError(f"scored reviewer release bindings are missing: {path}")
    bindings = read_json(path)
    expected = {
        "selection_manifest_sha256": FROZEN_HASHES["01_SELECTION/dense_gold_selection_manifest.json"],
        "ontology_sha256": FROZEN_HASHES["02_ONTOLOGY/dense_person_gold_ontology.json"],
        "metric_protocol_sha256": FROZEN_HASHES["03_METRICS/dense_gold_metric_protocol.json"],
        "event_schema_sha256": FROZEN_HASHES["04_SCHEMAS/dense_person_frame_annotation.schema.json"],
        "ack_schema_sha256": FROZEN_HASHES["04_SCHEMAS/dense_person_frame_acknowledgement.schema.json"],
        "sealed_repeat_qa_sha256": SEALED_HASHES["05_REVIEWER/sealed_blind_repeat_qa_selection.json"],
        "sealed_reveal_payload_sha256": SEALED_HASHES["05_REVIEWER/sealed_candidate_reveal_payloads.json"],
    }
    if any(bindings.get(key) != value for key, value in expected.items()):
        raise RuntimeError("scored reviewer binding hashes differ from frozen contracts")
    return bindings


def reviewer_config(repo: Path, *, port: int = 8791) -> DensePersonReviewerConfig:
    all_paths = roots(repo.resolve())
    verify_frozen_contracts(all_paths)
    verify_phase_b_authorization(all_paths)
    selection_path = all_paths["source"] / "01_SELECTION/dense_gold_selection_manifest.json"
    scored_queue(read_json(selection_path))
    return DensePersonReviewerConfig(
        selection_manifest_path=selection_path,
        assets_root=all_paths["source"] / "05_REVIEWER/assets",
        decisions_root=all_paths["source"] / "06_DENSE_DECISIONS",
        binding_hashes=release_bindings(all_paths),
        reviewer_release=REVIEWER_RELEASE,
        reveal_payload_path=None,
        pass_kind="FIRST_PASS",
        allowed_selection_statuses=("SCORED_DENSE_GOLD",),
        candidate_reveal_enabled=False,
        port=port,
    )


def check(repo: Path) -> dict[str, Any]:
    all_paths = roots(repo.resolve())
    verify_frozen_contracts(all_paths)
    verify_phase_b_authorization(all_paths)
    config = reviewer_config(repo)
    selection = read_json(config.selection_manifest_path)
    queue = scored_queue(selection)
    scored_ids = {str(row["anonymous_dense_image_id"]) for row in queue}
    existing = scored_decision_files(config.decisions_root, scored_ids)
    if existing:
        raise RuntimeError(f"release check requires never-started scored decisions; found {existing}")
    return {
        "decision": PASS,
        "reviewer_release": REVIEWER_RELEASE,
        "queue_images": len(queue),
        "workflow": "SCORED FIRST_PASS",
        "candidate_blind": True,
        "candidate_reveal_enabled": False,
        "scored_annotation_authorized": True,
        "scored_annotation_started": False,
        "blind_repeat_authorized": False,
        "production_ready": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check", "serve"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--port", type=int, default=8791)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"scored reviewer requires Python 3.12; found {sys.version.split()[0]}")
    if args.phase == "check":
        print(json.dumps(check(args.repo.resolve()), sort_keys=True))
        return 0
    config = reviewer_config(args.repo.resolve(), port=args.port)
    print(f"G7F-C scored FIRST_PASS reviewer: http://{config.host}:{config.port}/")
    print("Candidate reveal and BLIND_REPEAT are disabled. production_ready=false")
    run_server(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
