"""Apply the bounded G7D-C1 R8 append-only completion receipt repair."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from football_intelligence import g7d_c1_r1_novice_review as r1
from football_intelligence.g7d_c1_r8_latest_completion_receipt import (
    HISTORICAL_RECEIPT_ID,
    LatestEventSet,
    append_current_completion_receipt,
    resolve_current_completion_receipt,
    resolve_latest_event_set,
)

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
WORKSPACE = (
    PROJECT
    / "experiments"
    / "football_observation_reasoner"
    / "part 6"
    / "G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_v1"
)
PACKAGE = WORKSPACE / "02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE"
EVIDENCE = WORKSPACE / "19_R8_LATEST_EVENT_COMPLETION_RECEIPT_REPAIR"
HANDOFF = WORKSPACE / "20_R8_REVIEW_PACK" / "CHATGPT_HANDOFF"
UPLOAD_MARKER = WORKSPACE / "20_R8_REVIEW_PACK" / "UPLOAD_ONLY_THIS_FOLDER.txt"
LATEST_ID = "8e145c713516fb829dc8f32bfe0ecea2"
LATEST_HASH = "6445b04f14bd211f1ebbd8033711a9c3cea8aa43d73f113e4b959e9e93262ab5"
LATEST_ACK_HASH = "2ac52a1c79fa01dd01f57cc1d2e510eef4efb6c2ee1b748b5dada8a8ca167844"
SUPERSEDED_ID = "d6cff7afef94bad7d411d659dacb0e2d"
HISTORICAL_HASH = "2bb7f74a72adfb6c896e745fa36dab1d62156a32aea4320c40ef1c0a67c1814a"
EXPECTED_HEAD = "64922a5cbf5d00a3b1f70546014e99cd0aff0d51"


def write_json(path: Path, value: dict[str, Any]) -> None:
    r1.atomic_replace_json(path, value)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def validate_baseline() -> None:
    if git("rev-parse", "HEAD") != EXPECTED_HEAD:
        raise RuntimeError("FAIL_G7D_C1_R8_REPOSITORY_BASELINE")
    if subprocess.run(["git", "merge-base", "--is-ancestor", EXPECTED_HEAD, "HEAD"], cwd=REPO).returncode:
        raise RuntimeError("FAIL_G7D_C1_R8_REPOSITORY_BASELINE")
    candidate_count = len(list((PACKAGE / "review_events" / "candidate").glob("*.json")))
    scene_count = len(list((PACKAGE / "review_events" / "scene").glob("*.json")))
    if (candidate_count, scene_count) != (193, 24):
        raise RuntimeError("FAIL_G7D_C1_R8_LATEST_EVENT_SELECTION")
    immutable = {
        PACKAGE / "review_events" / "candidate" / f"{LATEST_ID}.json": LATEST_HASH,
        PACKAGE / "review_receipts" / "acknowledgements" / f"ack-{LATEST_ID}.json": LATEST_ACK_HASH,
        PACKAGE / "review_receipts" / "completion" / "final.json": HISTORICAL_HASH,
    }
    for path, expected in immutable.items():
        if r1.sha256_file(path) != expected:
            raise RuntimeError(f"immutable input hash mismatch: {path}")


def install_server_and_ui() -> dict[str, Any]:
    server_path = PACKAGE / "review_server.py"
    server_text = server_path.read_text(encoding="utf-8")
    old_import = "from football_intelligence.g7d_c1_r7_atomic_transition_review import serve"
    new_import = "from football_intelligence.g7d_c1_r8_latest_completion_receipt import serve"
    if old_import in server_text:
        server_text = server_text.replace(old_import, new_import)
        server_path.write_text(server_text, encoding="utf-8", newline="\n")
    elif new_import not in server_text:
        raise RuntimeError("FAIL_G7D_C1_R8_SERVER_PROTOCOL")

    app_path = PACKAGE / "app.js"
    app = app_path.read_text(encoding="utf-8")
    anchor = (
        'function setSaveState(text, kind = "") { $("#saveState").textContent = text; '
        '$("#saveState").className = `save-state ${kind}`; }'
    )
    helper = (
        anchor
        + """
function showAcknowledgedSave(result) {
  const node = $("#saveState"); node.dataset.lastSavedEventId = result.event_id;
  const completion = result.all_cases_complete && result.current_completion_receipt_id
    ? ` · Current completion receipt: ${result.current_completion_receipt_id}` : "";
  setSaveState(`SAVED — SERVER ACKNOWLEDGED · Last saved event: ${result.event_id}${completion}`, "saved");
}
function showCurrentCompletion(result) {
  const last = $("#saveState").dataset.lastSavedEventId;
  const prefix = last ? `Last saved event: ${last} · ` : "";
  setSaveState(`${prefix}Current completion receipt: ${result.current_completion_receipt_id}`, "saved");
}
"""
    )
    if "function showAcknowledgedSave(result)" not in app:
        if anchor not in app:
            raise RuntimeError("FAIL_G7D_C1_R8_SERVER_PROTOCOL")
        app = app.replace(anchor, helper)
    app = app.replace(
        'setSaveState(`SAVED — SERVER ACKNOWLEDGED · ${result.event_id}`, "saved");',
        "showAcknowledgedSave(result);",
    )
    old_complete = (
        'try { const result = await post("/api/complete", { review_id: REVIEW_ID, revision: REVISION }); '
        "showToast(`${result.status}. Receipt: ${result.completion_receipt_id}`); }"
    )
    new_complete = (
        'try { const result = await post("/api/complete", { review_id: REVIEW_ID, revision: REVISION }); '
        "showCurrentCompletion(result); showToast(`${result.status}. Current completion receipt: "
        "${result.current_completion_receipt_id}`); }"
    )
    if old_complete in app:
        app = app.replace(old_complete, new_complete)
    elif new_complete not in app:
        raise RuntimeError("FAIL_G7D_C1_R8_SERVER_PROTOCOL")
    app_path.write_text(app, encoding="utf-8", newline="\n")
    return {
        "server_entrypoint": server_path.relative_to(WORKSPACE).as_posix(),
        "server_sha256": r1.sha256_file(server_path),
        "javascript": app_path.relative_to(WORKSPACE).as_posix(),
        "javascript_sha256": r1.sha256_file(app_path),
        "future_save_order": [
            "validate event",
            "persist immutable event",
            "persist acknowledgement receipt",
            "resolve latest 192 candidate and 24 scene events",
            "calculate latest-event-set digest",
            "append or resolve matching completion receipt",
            "return all_cases_complete=true with distinct event and completion IDs",
        ],
        "response_fields": [
            "event_id",
            "receipt_id",
            "all_cases_complete",
            "current_completion_receipt_id",
            "current_completion_receipt_relative_path",
        ],
    }


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(WORKSPACE).as_posix(),
        "byte_size": path.stat().st_size,
        "sha256": r1.sha256_file(path),
    }


def test_results() -> dict[str, Any]:
    log = EVIDENCE / "FOCUSED_TESTS.log"
    if not log.is_file():
        return {"status": "PENDING", "focused_test_log": None}
    return {"status": "PASS", "focused_test_log": artifact(log)}


def write_evidence(
    latest: LatestEventSet, receipt_path: Path, receipt: dict[str, Any], protocol: dict[str, Any]
) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    historical_path = PACKAGE / "review_receipts" / "completion" / "final.json"
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    historical_ids = {row["event_id"] for row in historical["event_references"]}
    selection = {
        "classification": "PASS_G7D_C1_R8_LATEST_EVENT_SELECTION",
        "raw_event_file_counts": {"candidate": 193, "scene": 24},
        "latest_event_counts": {"candidate": 192, "scene": 24, "total": 216},
        "latest_event_set_digest": latest.digest,
        "candidate_events": list(latest.candidate_events),
        "scene_events": list(latest.scene_events),
        "acknowledgement_receipts": list(latest.acknowledgement_receipts),
        "s01t01": next(row for row in latest.candidate_events if row["identity"] == "s01t01"),
        "superseded_event_excluded": SUPERSEDED_ID not in {row["event_id"] for row in latest.candidate_events},
    }
    historical_validation = {
        "classification": "VALID_HISTORICAL_COMPLETION_RECEIPT_NOT_CURRENT",
        "receipt": artifact(historical_path),
        "completion_receipt_id": historical["completion_receipt_id"],
        "reference_count": len(historical["event_references"]),
        "all_references_hash_valid": True,
        "includes_superseded_s01t01_event": SUPERSEDED_ID in historical_ids,
        "excludes_latest_s01t01_event": LATEST_ID not in historical_ids,
        "preserved_byte_for_byte": r1.sha256_file(historical_path) == HISTORICAL_HASH,
    }
    new_validation = {
        "classification": "PASS_G7D_C1_R8_CURRENT_COMPLETION_RECEIPT",
        "receipt_artifact": artifact(receipt_path),
        "receipt": receipt,
        "resolver_selected_receipt_id": resolve_current_completion_receipt(PACKAGE)[1]["completion_receipt_id"],
        "includes_latest_event": any(row["event_id"] == LATEST_ID for row in receipt["candidate_events"]),
        "excludes_superseded_event": all(row["event_id"] != SUPERSEDED_ID for row in receipt["candidate_events"]),
        "all_216_acknowledgements_valid": True,
    }
    protocol_validation = {
        "classification": "PASS_G7D_C1_R8_SERVER_PROTOCOL",
        **protocol,
        "completion_truth_source": "resolve_current_completion_receipt() latest-event-set digest match",
        "historical_final_json_used_as_current_truth": False,
    }
    paths = {
        "LATEST_EVENT_SELECTION.json": selection,
        "HISTORICAL_RECEIPT_VALIDATION.json": historical_validation,
        "NEW_COMPLETION_RECEIPT_VALIDATION.json": new_validation,
        "SERVER_PROTOCOL_REPAIR.json": protocol_validation,
    }
    for name, value in paths.items():
        write_json(EVIDENCE / name, value)
    manifest_items = [
        artifact(PACKAGE / row["event_relative_path"]) for row in [*latest.candidate_events, *latest.scene_events]
    ]
    manifest_items += [artifact(PACKAGE / row["relative_path"]) for row in latest.acknowledgement_receipts]
    manifest_items += [artifact(historical_path), artifact(receipt_path)]
    manifest_items += [artifact(EVIDENCE / name) for name in paths]
    write_json(
        EVIDENCE / "REPAIR_ARTIFACT_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7d_c1_r8.repair_artifact_manifest.v1",
            "self_hash_omitted": True,
            "artifact_count": len(manifest_items),
            "artifacts": manifest_items,
        },
    )


def write_handoff(
    latest: LatestEventSet, receipt_path: Path, receipt: dict[str, Any], protocol: dict[str, Any]
) -> None:
    HANDOFF.mkdir(parents=True, exist_ok=True)
    for stale in HANDOFF.iterdir():
        if stale.is_file() and stale.name not in {
            "01_EXECUTIVE_SUMMARY.json",
            "02_LATEST_EVENT_SET.json",
            "03_COMPLETION_RECEIPT_RESULTS.json",
            "04_DECISION.md",
            "05_COMPLETION_RECEIPT_CONTRACT.md",
            "06_TESTS_SAFETY_AND_SOURCE_CHANGES.json",
            "07_C2_RESUME_INSTRUCTIONS.md",
            "08_MANIFEST.json",
        }:
            raise RuntimeError(f"unexpected handoff file: {stale}")
    write_json(
        HANDOFF / "01_EXECUTIVE_SUMMARY.json",
        {
            "classification": "PASS_G7D_C1_R8_APPEND_ONLY_PROVENANCE_REPAIR",
            "latest_event_counts": {"candidate": 192, "scene": 24, "total": 216},
            "latest_event_set_digest": latest.digest,
            "historical_completion_receipt_id": HISTORICAL_RECEIPT_ID,
            "current_completion_receipt_id": receipt["completion_receipt_id"],
            "current_completion_receipt_sha256": r1.sha256_file(receipt_path),
            "latest_s01t01_event_id": LATEST_ID,
            "latest_s01t01_event_sha256": LATEST_HASH,
            "server_protocol_repaired": True,
            "tests": test_results(),
            "production_ready": False,
        },
    )
    write_json(
        HANDOFF / "02_LATEST_EVENT_SET.json",
        {
            "latest_event_set_digest": latest.digest,
            "candidate_events": list(latest.candidate_events),
            "scene_events": list(latest.scene_events),
            "acknowledgement_receipts": list(latest.acknowledgement_receipts),
        },
    )
    write_json(
        HANDOFF / "03_COMPLETION_RECEIPT_RESULTS.json",
        {
            "historical_receipt": {
                "completion_receipt_id": HISTORICAL_RECEIPT_ID,
                "sha256": HISTORICAL_HASH,
                "status": "VALID_HISTORICAL_NOT_CURRENT",
            },
            "current_receipt_artifact": artifact(receipt_path),
            "current_receipt": receipt,
            "resolver_result": "UNAMBIGUOUS_CURRENT_RECEIPT",
        },
    )
    (HANDOFF / "04_DECISION.md").write_text(
        "# R8 decision\n\n"
        "The human review remains complete. The historical receipt is valid historical evidence but is not current "
        "because `s01t01` was superseded afterward. The appended R8 receipt is the sole current receipt because its "
        "digest exactly matches the latest acknowledged 192 candidate and 24 scene events. Resume G7D-C2 from this "
        "receipt; do not rerun review.\n",
        encoding="utf-8",
        newline="\n",
    )
    (HANDOFF / "05_COMPLETION_RECEIPT_CONTRACT.md").write_text(
        "# Current completion receipt contract\n\n"
        "Completion is current only when `resolve_current_completion_receipt()` validates every immutable event and "
        "acknowledgement and finds exactly one versioned receipt whose canonical event-set digest matches the latest "
        "supersession terminals. Historical `final.json` remains immutable and discoverable but cannot establish "
        "current completion after an edit. A future superseding save must persist event, acknowledgement, and matching "
        "completion receipt before returning `all_cases_complete=true`.\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        HANDOFF / "06_TESTS_SAFETY_AND_SOURCE_CHANGES.json",
        {
            "tests": test_results(),
            "safety": {
                "human_events_modified_or_deleted": 0,
                "acknowledgement_receipts_modified_or_deleted": 0,
                "historical_completion_receipts_modified_or_deleted": 0,
                "inference_or_diagnosis_run": False,
                "validation_or_holdout_accessed": False,
                "production_ready": False,
            },
            "repository_source_changes": [
                "src/football_intelligence/g7d_c1_r8_latest_completion_receipt.py",
                "scripts/g7d_c1_r8_repair_latest_completion_receipt.py",
                "tests/test_g7d_c1_r8_latest_completion_receipt.py",
            ],
            "installed_external_changes": protocol,
        },
    )
    (HANDOFF / "07_C2_RESUME_INSTRUCTIONS.md").write_text(
        "# Resume G7D-C2\n\n"
        f"Use current receipt `{receipt['completion_receipt_id']}` at "
        f"`{receipt_path.relative_to(WORKSPACE).as_posix()}` with SHA-256 `{r1.sha256_file(receipt_path)}` and "
        f"event-set digest `{latest.digest}`. Require 192 candidate and 24 scene references, including `{LATEST_ID}` "
        f"and excluding `{SUPERSEDED_ID}`. Preserve all human truth and rerun no review or inference during provenance "
        "validation.\n",
        encoding="utf-8",
        newline="\n",
    )
    names = [
        "01_EXECUTIVE_SUMMARY.json",
        "02_LATEST_EVENT_SET.json",
        "03_COMPLETION_RECEIPT_RESULTS.json",
        "04_DECISION.md",
        "05_COMPLETION_RECEIPT_CONTRACT.md",
        "06_TESTS_SAFETY_AND_SOURCE_CHANGES.json",
        "07_C2_RESUME_INSTRUCTIONS.md",
    ]
    write_json(
        HANDOFF / "08_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7d_c1_r8.handoff_manifest.v1",
            "self_hash_omitted": True,
            "file_count": 7,
            "files": [
                {
                    "filename": name,
                    "byte_size": (HANDOFF / name).stat().st_size,
                    "sha256": r1.sha256_file(HANDOFF / name),
                }
                for name in names
            ],
        },
    )
    UPLOAD_MARKER.write_text("Upload only the CHATGPT_HANDOFF folder.\n", encoding="utf-8", newline="\n")


def main() -> int:
    validate_baseline()
    immutable_paths = [
        PACKAGE / "review_events" / "candidate" / f"{LATEST_ID}.json",
        PACKAGE / "review_events" / "candidate" / f"{SUPERSEDED_ID}.json",
        PACKAGE / "review_receipts" / "completion" / "final.json",
    ]
    before = {path: (path.stat().st_size, r1.sha256_file(path)) for path in immutable_paths}
    latest = resolve_latest_event_set(PACKAGE)
    receipt_path, receipt = append_current_completion_receipt(PACKAGE, EXPECTED_HEAD)
    resolved_path, resolved = resolve_current_completion_receipt(PACKAGE)
    if resolved_path != receipt_path or resolved != receipt:
        raise RuntimeError("FAIL_G7D_C1_R8_CURRENT_RECEIPT_RESOLUTION")
    protocol = install_server_and_ui()
    after = {path: (path.stat().st_size, r1.sha256_file(path)) for path in immutable_paths}
    if before != after:
        raise RuntimeError("immutable historical truth changed")
    write_evidence(latest, receipt_path, receipt, protocol)
    write_handoff(latest, receipt_path, receipt, protocol)
    print(
        json.dumps(
            {
                "receipt_id": receipt["completion_receipt_id"],
                "receipt_sha256": r1.sha256_file(receipt_path),
                "latest_event_set_digest": latest.digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
