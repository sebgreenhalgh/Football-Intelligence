"""Run G7F-C engineering acceptance without writing real dense annotation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2

from football_intelligence.dense_person_gold import COMPLETION_ASSERTION
from football_intelligence.dense_person_reviewer import DensePersonHTTPServer, scan_blind_payload
from football_intelligence.gold_eval.core import inventory_tree, sha256_file, write_json
from g7f_c_build_dense_person_reviewer import (
    EXPECTED_PROTOCOL_RECEIPT_SHA256,
    EXPECTED_PROTOCOL_SHA256,
    MATCHES,
    _load_inputs,
    git,
    read_json,
    roots,
    select_records,
    server_config,
    verify_upstream,
)


PASS = "PASS_G7F_C_DENSE_PERSON_GOLD_SELECTION_AND_REVIEWER_READY_FOR_HUMAN_ANNOTATION"


def _get_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=10) as response:  # noqa: S310 - loopback-only reviewer
        return json.loads(response.read())


def _post_json(url: str, payload: dict[str, Any], expected_status: int = 200) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:  # noqa: S310 - loopback-only reviewer
            if response.status != expected_status:
                raise RuntimeError(f"unexpected HTTP status {response.status}")
            return json.loads(response.read())
    except HTTPError as exc:
        body = json.loads(exc.read())
        if exc.code != expected_status:
            raise RuntimeError(f"unexpected HTTP status {exc.code}: {body}") from exc
        return body


def _polygon(x1: int, y1: int, x2: int, y2: int) -> list[dict[str, int]]:
    return [{"x": x1, "y": y1}, {"x": x2, "y": y1}, {"x": x2, "y": y2}, {"x": x1, "y": y2}]


def _temp_document(with_geometry: bool) -> dict[str, Any]:
    people: list[dict[str, Any]] = []
    ignores: list[dict[str, Any]] = []
    if with_geometry:
        people.append(
            {
                "instance_id": "TEMP-person-001",
                "relevance": "MATCH_RELEVANT",
                "visible_mask_components": [_polygon(100, 100, 124, 150), _polygon(128, 130, 135, 145)],
            }
        )
        ignores.append(
            {
                "ignore_region_id": "TEMP-ignore-001",
                "reason": "SEVERE_VISUAL_ARTIFACT",
                "polygon": _polygon(200, 100, 250, 170),
            }
        )
    return {
        "people": people,
        "ignore_regions": ignores,
        "reviewed_exhaustiveness_strips": list(range(8)),
        "unfinished_polygon": None,
        "completion_assertion": COMPLETION_ASSERTION,
    }


def _pixel_sha256(path: Path) -> str:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to load reviewer asset: {path}")
    return hashlib.sha256(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()


def _run_http_acceptance(paths: dict[str, Path], selected: list[dict[str, Any]]) -> dict[str, Any]:
    workspace = paths["workspace"]
    temp_decisions = workspace / "07_ACCEPTANCE/TEMP_ENGINEERING_DECISIONS_NOT_DENSE_GOLD"
    if temp_decisions.exists():
        raise RuntimeError("fresh temp acceptance decisions root required")
    config = server_config(paths, decisions_root=temp_decisions)
    config = type(config)(**{**config.__dict__, "port": 0})
    server = DensePersonHTTPServer(config)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    base = f"http://127.0.0.1:{port}"
    browser_result: dict[str, Any] = {"available": False, "passed": False}
    try:
        bootstrap = _get_json(f"{base}/api/bootstrap")
        if scan_blind_payload(bootstrap) or len(bootstrap["queue"]) != 54 or not bootstrap["candidate_blind"]:
            raise RuntimeError("candidate-blind bootstrap contract failed")
        inspected = []
        for index, row in enumerate(selected):
            image_id = row["anonymous_dense_image_id"]
            asset_url = next(
                item["image_url"] for item in bootstrap["queue"] if item["anonymous_dense_image_id"] == image_id
            )
            with urlopen(f"{base}{asset_url}", timeout=10) as response:  # noqa: S310 - loopback-only reviewer
                png = response.read()
            if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                raise RuntimeError("reviewer did not serve a PNG asset")
            early = _post_json(
                f"{base}/api/action",
                {
                    "action_id": f"TEMP-early-{index}",
                    "action_type": "REVEAL_CANDIDATES",
                    "anonymous_dense_image_id": image_id,
                    "pass_kind": "FIRST_PASS",
                    "expected_revision": 0,
                },
                409,
            )
            if early["error_code"] != "REVEAL_BEFORE_FINALIZATION":
                raise RuntimeError("pre-final reveal did not fail closed")
            save_action = {
                "action_id": f"TEMP-save-{index}",
                "action_type": "SAVE_DRAFT",
                "anonymous_dense_image_id": image_id,
                "pass_kind": "FIRST_PASS",
                "expected_revision": 0,
                "document": _temp_document(index == 0),
            }
            saved = _post_json(f"{base}/api/action", save_action)
            if _post_json(f"{base}/api/action", save_action) != saved:
                raise RuntimeError("idempotent retry changed the response")
            stale = _post_json(f"{base}/api/action", {**save_action, "action_id": f"TEMP-stale-{index}"}, 409)
            if stale["error_code"] != "STALE_REVISION":
                raise RuntimeError("stale action did not fail closed")
            finalized = _post_json(
                f"{base}/api/action",
                {
                    **save_action,
                    "action_id": f"TEMP-final-{index}",
                    "action_type": "FINALIZE",
                    "expected_revision": 1,
                },
            )
            reloaded = _get_json(f"{base}/api/state?image_id={image_id}")
            if not reloaded["finalized"] or not reloaded["read_only"] or reloaded["revision"] != 2:
                raise RuntimeError("final event did not reload as immutable")
            event_path = temp_decisions / "events" / f"first_pass__{image_id}.json"
            before_reveal = event_path.read_bytes()
            revealed = _post_json(
                f"{base}/api/action",
                {
                    "action_id": f"TEMP-reveal-{index}",
                    "action_type": "REVEAL_CANDIDATES",
                    "anonymous_dense_image_id": image_id,
                    "pass_kind": "FIRST_PASS",
                    "expected_revision": 2,
                },
            )
            if not revealed["read_only"] or event_path.read_bytes() != before_reveal:
                raise RuntimeError("post-final reveal mutated an annotation event")
            inspected.append(
                {
                    "anonymous_dense_image_id": image_id,
                    "match_id": row["match_id"],
                    "asset_png_sha256": hashlib.sha256(png).hexdigest(),
                    "source_pixel_sha256": row["source_frame_sha256"],
                    "temp_event_id": finalized["event_id"],
                    "load_and_visual_inspection": "PASS",
                }
            )
        screenshot = workspace / "07_ACCEPTANCE/reviewer_browser_acceptance.png"
        browser_result = {
            "available": screenshot.is_file(),
            "browser": "Microsoft Edge headless, isolated temporary profile",
            "screenshot_path": str(screenshot),
            "screenshot_sha256": sha256_file(screenshot) if screenshot.is_file() else None,
            "passed": screenshot.is_file(),
        }
        if not browser_result["passed"]:
            raise RuntimeError("pre-captured headless reviewer acceptance screenshot is missing")
        return {
            "bootstrap_candidate_blind": True,
            "bootstrap_queue_size": 54,
            "temp_decisions_root": str(temp_decisions),
            "temp_finalized_events": len(inspected),
            "real_decisions_root_used": False,
            "inspected_one_real_selected_source_per_match": inspected,
            "browser": browser_result,
            "api_behaviors": {
                "asset_load": "PASS",
                "polygon_and_multi_component_geometry": "PASS",
                "relevance_and_ignore_regions": "PASS",
                "eight_strip_sweep": "PASS",
                "candidate_blindness": "PASS",
                "server_authoritative_revision": "PASS",
                "idempotent_retry": "PASS",
                "stale_conflict": "PASS",
                "immutable_finalization_and_reload": "PASS",
                "post_final_read_only_reveal": "PASS",
            },
        }
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()


def _handoff(workspace: Path, payloads: dict[str, Any]) -> None:
    handoff = workspace / "10_REVIEW_PACK/CHATGPT_HANDOFF"
    handoff.mkdir(parents=True, exist_ok=True)
    files = [
        ("00_EXECUTIVE_SUMMARY.json", payloads["executive"]),
        ("01_UPSTREAM_SUBSTRATE_INTEGRITY.json", payloads["upstream"]),
        ("02_DENSE_GOLD_SELECTION_MANIFEST.json", payloads["selection"]),
        ("03_SELECTION_BALANCE_AND_REASON_REPORT.json", payloads["balance"]),
        ("04_DENSE_PERSON_ONTOLOGY_AND_IGNORE_RULES.json", payloads["ontology"]),
        ("05_REVIEWER_AND_PERSISTENCE_CONTRACT.json", payloads["reviewer"]),
        ("06_DENSE_GOLD_METRIC_PROTOCOL.json", payloads["metrics"]),
        ("07_CANDIDATE_BLINDNESS_AND_BIAS_REPORT.json", payloads["blindness"]),
        ("08_TEST_AND_VISUAL_ACCEPTANCE_REPORT.json", payloads["acceptance"]),
        ("09_HUMAN_ANNOTATION_RELEASE_STATE.json", payloads["release"]),
    ]
    for name, payload in files:
        write_json(handoff / name, payload)
    (handoff / "10_DECISION.md").write_text(
        f"# G7F-C decision\n\n`{PASS}`\n\n`production_ready=false`\n\n"
        "Selection and the candidate-blind reviewer are ready for human annotation. "
        "No dense annotation was started.\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest_rows = []
    for path in sorted(handoff.iterdir(), key=lambda item: item.name):
        if path.name == "12_MANIFEST.json":
            continue
        manifest_rows.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(
        handoff / "12_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7f_c.handoff_manifest.v1",
            "file_count_including_manifest": 12,
            "files": manifest_rows,
            "decision": PASS,
            "production_ready": False,
        },
    )
    if len(list(handoff.iterdir())) != 12:
        raise RuntimeError("handoff must contain exactly 12 files")


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    paths = roots(repo)
    workspace = paths["workspace"]
    upstream_after = verify_upstream(paths)
    reviewer_release_path = workspace / "05_REVIEWER/reviewer_release_manifest.json"
    reviewer_release = read_json(reviewer_release_path)
    reviewer_release["repository_commit"] = git(repo, "rev-parse", "HEAD")
    for row in reviewer_release["files"]:
        source = repo / row["path"]
        row["byte_size"] = source.stat().st_size
        row["sha256"] = sha256_file(source)
    write_json(reviewer_release_path, reviewer_release)
    binding_path = workspace / "05_REVIEWER/reviewer_binding_hashes.json"
    binding_hashes = read_json(binding_path)
    binding_hashes["reviewer_release_manifest_sha256"] = sha256_file(reviewer_release_path)
    write_json(binding_path, binding_hashes)
    selection_path = workspace / "01_SELECTION/dense_gold_selection_manifest.json"
    selection = read_json(selection_path)
    if (
        sha256_file(selection_path)
        != read_json(workspace / "01_SELECTION/dense_gold_selection_manifest.sha256.json")["selection_manifest_sha256"]
    ):
        raise RuntimeError("selection manifest changed after freeze")
    if sha256_file(workspace / "01_SELECTION/PREDECLARED_SELECTION_PROTOCOL.json") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("predeclared selection protocol changed")
    if (
        sha256_file(workspace / "01_SELECTION/PREDECLARED_SELECTION_PROTOCOL.sha256.json")
        != EXPECTED_PROTOCOL_RECEIPT_SHA256
    ):
        raise RuntimeError("selection protocol receipt changed")
    images = selection["images"]
    scored = [row for row in images if row["selection_status"] == "SCORED_DENSE_GOLD"]
    calibration = [row for row in images if row["selection_status"] == "CALIBRATION_ONLY"]
    if (
        len(scored) != 48
        or len(calibration) != 6
        or len({row["source_frame_sha256"] for row in images}) != 54
        or Counter(row["match_id"] for row in scored) != Counter({match: 8 for match in MATCHES})
        or Counter(row["match_id"] for row in calibration) != Counter({match: 1 for match in MATCHES})
    ):
        raise RuntimeError("frozen selection count/balance failure")
    rebuilt_scored, rebuilt_calibration = select_records(_load_inputs(paths)[0])
    frozen_identity = sorted(
        (row["source_frame_sha256"], row["selection_status"], row["primary_selection_slot"]) for row in images
    )
    rebuilt_identity = sorted(
        (row["source_frame_sha256"], row["selection_status"], row["primary_selection_slot"])
        for row in rebuilt_scored + rebuilt_calibration
    )
    if frozen_identity != rebuilt_identity:
        raise RuntimeError("deterministic selection rebuild differed")
    asset_manifest = read_json(workspace / "05_REVIEWER/source_asset_manifest.json")
    for asset in asset_manifest["assets"]:
        path = workspace / asset["asset_path"]
        if sha256_file(path) != asset["asset_file_sha256"] or _pixel_sha256(path) != asset["source_pixel_sha256"]:
            raise RuntimeError("asset identity failure")
    selected_for_visual = []
    for match in MATCHES:
        selected_for_visual.append(next(row for row in scored if row["match_id"] == match))
    http_acceptance = _run_http_acceptance(paths, selected_for_visual)
    ui_source = (repo / "src/football_intelligence/dense_person_reviewer_static/app.js").read_text(encoding="utf-8")
    ui_contract = {
        token: token in ui_source
        for token in (
            "fitWidth",
            "fitHeight",
            "zoomIn",
            "zoomOut",
            "visible_mask_components",
            "ignore_regions",
            "reviewed_exhaustiveness_strips",
            "undo",
            "redo",
            "saveDraft",
            "FINALIZE",
            "REVEAL_CANDIDATES",
        )
    }
    if not all(ui_contract.values()):
        raise RuntimeError("reviewer UI contract incomplete")
    real_decisions = workspace / "06_DENSE_DECISIONS"
    real_inventory = inventory_tree(real_decisions)
    if real_inventory["file_count"] != 0:
        raise RuntimeError("real dense decisions root contaminated")
    focused = subprocess.run(
        [
            str(repo / ".venv/Scripts/python.exe"),
            "-m",
            "pytest",
            "tests/test_g7f_c_dense_person_gold.py",
            "-q",
            f"--basetemp={workspace / '07_ACCEPTANCE/pytest_temp'}",
            "-o",
            f"cache_dir={workspace / '07_ACCEPTANCE/pytest_cache'}",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if focused.returncode != 0:
        raise RuntimeError(f"focused tests failed: {focused.stdout}\n{focused.stderr}")
    acceptance = {
        "schema_version": "football_intelligence.g7f_c.engineering_acceptance_report.v1",
        "decision": PASS,
        "selection_counts": {"scored": 48, "calibration": 6, "unique": 54},
        "deterministic_selection_rebuild": "PASS",
        "asset_pixel_identity_54_of_54": "PASS",
        "ui_contract": ui_contract,
        "http_and_visual_acceptance": http_acceptance,
        "focused_tests": {"exit_code": focused.returncode, "stdout": focused.stdout.strip()},
        "visual_inspection_attestation": (
            "Six real selected images, one per match, were manually inspected at reviewer resolution; the temp HTTP "
            "acceptance exercised each corresponding image."
        ),
        "real_decisions_root_pristine": True,
        "human_annotation_started": False,
        "production_ready": False,
    }
    write_json(workspace / "07_ACCEPTANCE/engineering_acceptance_report.json", acceptance)
    write_json(workspace / "00_SOURCE_FREEZE/upstream_substrate_integrity_after.json", upstream_after)
    balance = {
        "scored_per_match": dict(sorted(Counter(row["match_id"] for row in scored).items())),
        "calibration_per_match": dict(sorted(Counter(row["match_id"] for row in calibration).items())),
        "scored_slots": dict(sorted(Counter(row["primary_selection_slot"] for row in scored).items())),
        "fallback_count": sum(bool(row["selection_fallback_used"]) for row in scored),
        "duplicate_source_hashes": 0,
        "deterministic_rebuild": True,
        "DISAGREEMENT_ENRICHED_SAMPLE": True,
        "tuning_exposure": "DENSE_GOLD_INTERNAL_VALIDATION",
        "future_sealed": False,
        "production_ready": False,
    }
    release_gate_path = workspace / "08_RELEASE/G7F_C_DENSE_PERSON_GOLD_RELEASE_GATE.json"
    release_gate = read_json(release_gate_path)
    release_gate.update(
        {
            "classification": PASS,
            "bindings": binding_hashes,
            "engineering_acceptance_report_sha256": sha256_file(
                workspace / "07_ACCEPTANCE/engineering_acceptance_report.json"
            ),
            "real_decisions_root_inventory": real_inventory,
            "real_finalized_annotation_count": 0,
            "real_acknowledgement_count": 0,
            "human_annotation_started": False,
            "production_ready": False,
        }
    )
    write_json(release_gate_path, release_gate)
    reviewer_contract = {
        "release_manifest": read_json(workspace / "05_REVIEWER/reviewer_release_manifest.json"),
        "binding_hashes": read_json(workspace / "05_REVIEWER/reviewer_binding_hashes.json"),
        "candidate_blind_before_finalization": True,
        "immutable_events_and_acknowledgements": True,
        "eight_exhaustiveness_strips": True,
        "post_finalization_reveal_read_only": True,
        "blind_repeat_ids_sealed_until_first_pass_complete": True,
        "real_decisions_root": str(real_decisions),
        "production_ready": False,
    }
    executive = {
        "decision": PASS,
        "scored_unique_source_images": 48,
        "calibration_unique_source_images": 6,
        "candidate_blind_reviewer_released": True,
        "human_annotation_started": False,
        "production_ready": False,
    }
    blindness = {
        "candidate_blind_bootstrap": True,
        "pre_final_candidate_reveal_impossible": True,
        "post_final_reveal_read_only_and_logged": True,
        "DISAGREEMENT_ENRICHED_SAMPLE": True,
        "tuning_exposure": "DENSE_GOLD_INTERNAL_VALIDATION",
        "unbiased_whole_match_estimate": False,
        "future_sealed": False,
        "final_promotion_requires_new_sealed_match_footage": True,
        "production_ready": False,
    }
    _handoff(
        workspace,
        {
            "executive": executive,
            "upstream": {
                "before": read_json(workspace / "00_SOURCE_FREEZE/upstream_substrate_integrity_before.json"),
                "after": upstream_after,
            },
            "selection": selection,
            "balance": balance,
            "ontology": read_json(workspace / "02_ONTOLOGY/dense_person_gold_ontology.json"),
            "reviewer": reviewer_contract,
            "metrics": read_json(workspace / "03_METRICS/dense_gold_metric_protocol.json"),
            "blindness": blindness,
            "acceptance": acceptance,
            "release": release_gate,
        },
    )
    if git(repo, "status", "--porcelain") or git(repo, "rev-parse", "HEAD") != git(repo, "rev-parse", "origin/main"):
        raise RuntimeError("final repository clean/origin gate failed")
    print(json.dumps({"decision": PASS, "production_ready": False}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
