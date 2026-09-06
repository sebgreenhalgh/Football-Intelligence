"""Run atomic adjudication navigation acceptance with TEMP decisions only."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import multiprocessing
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from football_intelligence.calibration_adjudication import CALIBRATION_IDS, PASS_KIND, sha256_file
from football_intelligence.calibration_adjudication_reviewer import (
    CalibrationAdjudicationHTTPServer,
    CalibrationAdjudicationReviewerConfig,
)


RELEASE = "G7F_C_CALIBRATION_ADJUDICATION_REVIEWER_R2"
ASSERTION = "I have reviewed the full image and annotated every individually evaluable visible human."
ADJUDICATION_ASSERTION = (
    "I re-reviewed the full image, addressed the calibration audit findings, and annotated every individually "
    "evaluable visible human."
)


def paths(repo: Path) -> dict[str, Path]:
    part9 = repo.parent / "experiments/football_observation_reasoner/part 9"
    return {
        "repo": repo,
        "source": part9 / "G7F_C_DENSE_PERSON_GOLD_DISCRIMINATION_SET_v1",
        "audit": part9 / "G7F_C_CALIBRATION_6_HUMAN_ANNOTATION_QUALITY_GATE_v1",
        "phase_a": part9 / "G7F_C_CALIBRATION_SUPERSEDING_ADJUDICATION_AND_REAUDIT_v1",
        "stage": part9 / "G7F_C_ADJUDICATION_R1_NAVIGATION_STATE_DESYNC_REPAIR_v1",
    }


def inventory(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    rows = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    ordered = "\n".join(f"{row['relative_path']}\t{row['byte_size']}\t{row['sha256']}" for row in rows).encode()
    return {
        "file_count": len(rows),
        "total_bytes": sum(row["byte_size"] for row in rows),
        "ordered_inventory_sha256": hashlib.sha256(ordered).hexdigest(),
        "files": rows,
    }


def original_decision_inventory(root: Path) -> dict[str, Any]:
    files = sorted([*(root / "events").glob("*.json"), *(root / "acknowledgements").glob("*.json")])
    return {
        path.relative_to(root).as_posix(): {
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    }


def copy_original_calibration(source: Path, target: Path) -> dict[str, dict[str, str]]:
    frozen: dict[str, dict[str, str]] = {}
    for image_id in CALIBRATION_IDS:
        event_source = source / f"events/first_pass__{image_id}.json"
        acknowledgement_source = source / f"acknowledgements/first_pass__{image_id}.json"
        event_target = target / f"events/first_pass__{image_id}.json"
        acknowledgement_target = target / f"acknowledgements/first_pass__{image_id}.json"
        event_target.parent.mkdir(parents=True, exist_ok=True)
        acknowledgement_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(event_source, event_target)
        shutil.copyfile(acknowledgement_source, acknowledgement_target)
        frozen[image_id] = {
            "event_file_sha256": sha256_file(event_target),
            "acknowledgement_file_sha256": sha256_file(acknowledgement_target),
        }
    return frozen


def config(
    all_paths: dict[str, Path],
    decisions: Path,
    frozen: dict[str, dict[str, str]],
    assets: Path,
) -> CalibrationAdjudicationReviewerConfig:
    bindings = json.loads(
        (all_paths["phase_a"] / "05_REVIEWER/reviewer_binding_hashes.json").read_text(encoding="utf-8")
    )
    audit_path = all_paths["audit"] / "03_MANUAL_VISUAL_ASSESSMENT.json"
    return CalibrationAdjudicationReviewerConfig(
        selection_manifest_path=all_paths["source"] / "01_SELECTION/dense_gold_selection_manifest.json",
        assets_root=assets,
        original_decisions_root=decisions,
        adjudication_root=decisions / "calibration_adjudication",
        binding_hashes=bindings,
        reviewer_release=RELEASE,
        audit_assessment_path=audit_path,
        audit_checklist_sha256=sha256_file(audit_path),
        frozen_parent_file_hashes=frozen,
        port=0,
    )


def ready_document(document: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    result["reviewed_exhaustiveness_strips"] = list(range(8))
    result["completion_assertion"] = ASSERTION
    result["unfinished_polygon"] = None
    return result


def seed_resume_state(server: CalibrationAdjudicationHTTPServer) -> dict[str, Any]:
    metadata = {
        "repair_checklist_addressed": True,
        "adjudication_assertion": ADJUDICATION_ASSERTION,
    }
    for image_id in ("DG-001", "DG-002"):
        state = server.store.state(image_id)
        synthetic_document = copy.deepcopy(state["document"])
        synthetic_document["people"] = []
        synthetic_document["ignore_regions"] = []
        server.store.apply_action(
            {
                "action_id": f"seed-finalize-{image_id}",
                "action_type": "FINALIZE",
                "anonymous_dense_image_id": image_id,
                "pass_kind": PASS_KIND,
                "expected_revision": state["revision"],
                "document": ready_document(synthetic_document),
                "adjudication_metadata": metadata,
            }
        )

    state = server.store.state("DG-003")
    document = copy.deepcopy(state["document"])
    template = copy.deepcopy(document["people"][0])
    numbers = [int(person["instance_id"].split("-")[-1]) for person in document["people"]]
    next_number = max(numbers) + 1
    while len(document["people"]) < 80:
        person = copy.deepcopy(template)
        person["instance_id"] = f"person-{next_number:03d}"
        document["people"].append(person)
        next_number += 1
    document["reviewed_exhaustiveness_strips"] = [0, 2, 4]
    document["completion_assertion"] = None
    document["unfinished_polygon"] = None
    response = server.store.apply_action(
        {
            "action_id": "seed-large-dg003-draft",
            "action_type": "SAVE_DRAFT",
            "anonymous_dense_image_id": "DG-003",
            "pass_kind": PASS_KIND,
            "expected_revision": state["revision"],
            "document": document,
            "adjudication_metadata": {
                "repair_checklist_addressed": False,
                "adjudication_assertion": None,
            },
        }
    )
    dg005 = server.store.state("DG-005")
    dg005_document = copy.deepcopy(dg005["document"])
    dg005_document["people"] = []
    dg005_document["ignore_regions"] = []
    server.store.apply_action(
        {
            "action_id": "seed-lightweight-dg005-draft",
            "action_type": "SAVE_DRAFT",
            "anonymous_dense_image_id": "DG-005",
            "pass_kind": PASS_KIND,
            "expected_revision": dg005["revision"],
            "document": dg005_document,
            "adjudication_metadata": {
                "repair_checklist_addressed": False,
                "adjudication_assertion": None,
            },
        }
    )
    return {
        "finalized": ["DG-001", "DG-002"],
        "mutable": "DG-003",
        "dg003_people": len(document["people"]),
        "dg003_strips": document["reviewed_exhaustiveness_strips"],
        "dg003_revision": response["revision"],
    }


def serve_child(
    server_config: CalibrationAdjudicationReviewerConfig,
    ready: multiprocessing.connection.Connection,
) -> None:
    server = CalibrationAdjudicationHTTPServer(server_config)
    ready.send(int(server.server_address[1]))
    ready.close()
    try:
        server.serve_forever()
    finally:
        server.server_close()


def run_edge(
    repo: Path,
    server_config: CalibrationAdjudicationReviewerConfig,
    scenario: str,
    screenshot: Path | None = None,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=serve_child, args=(server_config, send), daemon=True)
    process.start()
    send.close()
    if not receive.poll(20):
        process.terminate()
        process.join(timeout=5)
        raise RuntimeError(f"temporary Edge {scenario} server did not start")
    port = receive.recv()
    receive.close()
    command = [
        "node",
        str(repo / "scripts/g7f_c_adjudication_r1_edge_state_machine_acceptance.js"),
        f"http://127.0.0.1:{port}/",
        scenario,
    ]
    if screenshot is not None:
        command.append(str(screenshot))
    try:
        result = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=180, check=False)
    finally:
        process.terminate()
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
    if result.returncode:
        raise RuntimeError(f"Edge {scenario} acceptance failed:\n{result.stdout}\n{result.stderr}")
    payload = json.loads(result.stdout)
    if not payload.get("passed") or not all(check.get("passed") for check in payload.get("checks", [])):
        raise RuntimeError(f"Edge {scenario} acceptance did not pass: {payload}")
    return payload


def run(repo: Path, output: Path) -> dict[str, Any]:
    all_paths = paths(repo)
    output.mkdir(parents=True, exist_ok=True)
    real_root = all_paths["source"] / "06_DENSE_DECISIONS"
    real_before = inventory(real_root)
    with tempfile.TemporaryDirectory(prefix="g7f_c_adj_r2_", dir=output) as temp_name:
        temp_root = Path(temp_name)
        decisions = temp_root / "decisions"
        assets = temp_root / "assets"
        assets.mkdir(parents=True)
        pixel = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
        for image_id in CALIBRATION_IDS:
            (assets / f"{image_id}.png").write_bytes(pixel)
        frozen = copy_original_calibration(real_root, decisions)
        original_before = original_decision_inventory(decisions)

        server_config = config(all_paths, decisions, frozen, assets)
        seed_server = CalibrationAdjudicationHTTPServer(server_config)
        seeded = seed_resume_state(seed_server)
        seed_server.server_close()
        initial_event_count = len(list((decisions / "calibration_adjudication/events").glob("*.json")))
        lifecycle = run_edge(repo, server_config, "lifecycle", output / "atomic_navigation_acceptance.png")

        event_paths = sorted((decisions / "calibration_adjudication/events").glob("*.json"))
        acknowledgement_paths = sorted((decisions / "calibration_adjudication/acknowledgements").glob("*.json"))
        if len(event_paths) != initial_event_count + 1 or len(acknowledgement_paths) != initial_event_count + 1:
            raise RuntimeError("double-click acceptance did not create exactly one temporary event/ack pair")
        if not (decisions / "calibration_adjudication/drafts/calibration_adjudication_001__DG-003.json").is_file():
            raise RuntimeError("temporary DG-003 draft disappeared")

        resume = run_edge(repo, server_config, "resume")
        original_after = original_decision_inventory(decisions)
        if original_before != original_after:
            raise RuntimeError("temporary copies of original FIRST_PASS decisions changed")

        temp_summary = {
            "initial_finalized_adjudications": initial_event_count,
            "final_event_ack_pairs": len(event_paths),
            "exactly_one_event_from_double_finalize": len(event_paths) == initial_event_count + 1,
            "dg003_draft_people": 80,
            "original_first_pass_copies_byte_identical": True,
        }

    real_after = inventory(real_root)
    if real_before != real_after:
        raise RuntimeError("atomic navigation acceptance changed the real decision tree")
    payload = {
        "schema_version": "football_intelligence.g7f_c.adjudication_r1.atomic_navigation_acceptance.v1",
        "reviewer_release": RELEASE,
        "seeded_temp_resume_state": seeded,
        "lifecycle": lifecycle,
        "refresh_resume_after_server_restart": resume,
        "temporary_decisions": temp_summary,
        "real_decisions": {
            "before": {key: real_before[key] for key in ("file_count", "total_bytes", "ordered_inventory_sha256")},
            "after": {key: real_after[key] for key in ("file_count", "total_bytes", "ordered_inventory_sha256")},
            "byte_identical": True,
        },
        "phase_b_run": False,
        "scored_annotation_authorized": False,
        "production_ready": False,
    }
    (output / "atomic_navigation_acceptance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.repo.resolve(), args.output.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
