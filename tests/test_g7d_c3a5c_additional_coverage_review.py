from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from football_intelligence.g7d_c3a5c_additional_coverage_review import ReviewStore


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
STAGE = (
    PROJECT / "experiments/football_observation_reasoner/part 7" / "G7D_C3A5C_ADDITIONAL_COVERAGE_REPLAY_AND_REVIEW_v1"
)
PACKAGE = STAGE / "04_ADDITIONAL_COVERAGE_REVIEW_PACKAGE"
MATCHES = {"117093", "118576", "118577"}
FRACTIONS = [0.08, 0.20, 0.32, 0.44, 0.56, 0.68, 0.80, 0.92]
CATEGORIES = {
    "ENDLINE_NEAREST_PROXY",
    "TOUCHLINE_OUTSIDE_PROXY",
    "HIGH_DENSITY_OVERLAP",
    "STABLE_CONTROL",
}


def load(relative: str) -> dict:
    return json.loads((STAGE / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_exact_frozen_frame_plan_and_corrected_117093_source() -> None:
    plan = load("01_FRAME_REPLAY/frame_plan.json")
    assert plan["frame_count"] == 48
    assert plan["fractions"] == FRACTIONS
    assert plan["adaptive_replacement"] is False
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for frame in plan["frames"]:
        grouped[(frame["match_id"], frame["half"])].append(frame["fraction"])
        assert frame["selection_rule"] == "FIXED_FRACTION_NEAREST_FRAME_ROUND_HALF_UP"
        assert sha256(Path(frame["frame_path"])) == frame["frame_sha256"]
        if frame["match_id"] == "117093" and frame["half"] == "FIRST_HALF":
            assert frame["source_video_relative_path"].endswith("117093_panorama_1st_half-008.mp4")
    assert set(grouped) == {(match, half) for match in MATCHES for half in ("FIRST_HALF", "SECOND_HALF")}
    assert all(values == FRACTIONS for values in grouped.values())


def test_proposal_runtime_was_bounded_and_used_required_gpu() -> None:
    report = load("01_FRAME_REPLAY/proposal_runtime_reuse_report.json")
    assert report["reused_exact_frozen_output_frames"] == 0
    assert report["ran_frozen_proposal_runtime_once_frames"] == 48
    assert report["proposal_inference_executed"] is True
    assert report["crop_features_executed"] is False
    assert report["semantic_folds_executed"] is False
    gpu = report["gpu_preflight"]
    assert gpu["device"] == "cuda:0"
    assert gpu["device_name"] == "NVIDIA GeForce RTX 5060 Laptop GPU"
    assert gpu["cpu_or_intel_fallback"] is False
    assert gpu["total_memory_bytes"] >= int(7.5 * 1024**3)


def test_gate_is_shadow_only_and_preserves_every_candidate() -> None:
    replay = load("01_FRAME_REPLAY/frame_and_candidate_manifest.json")
    candidates = replay["candidates"]
    gate_rows = [
        json.loads(line)
        for line in (STAGE / "02_GATE_RESULTS/gate_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(candidates) == len(gate_rows) == 3127
    assert [row["candidate_local_id"] for row in candidates] == [row["candidate_local_id"] for row in gate_rows]
    assert Counter(row["gate_decision"] for row in candidates) == {
        "KEEP": 1321,
        "BOUNDARY_REVIEW": 922,
        "SUPPRESS_SANDBOX": 870,
        "EXCEPTION_KEEP": 14,
    }
    assert all(row["production_ready"] is False for row in candidates)


def test_exact_blind_first_scene_and_target_quotas() -> None:
    scenes = load("03_SCENE_AND_TARGET_SELECTION/scene_shortlist.json")
    targets = load("03_SCENE_AND_TARGET_SELECTION/target_manifest.json")
    assert scenes["scene_count"] == 12
    assert scenes["human_labels_used"] is False
    assert targets["target_count"] == 60
    assert targets["human_labels_used"] is False
    by_match: dict[str, set[str]] = defaultdict(set)
    for scene in scenes["scenes"]:
        by_match[scene["match_id"]].add(scene["selection_category"])
    assert dict(by_match) == {match: CATEGORIES for match in MATCHES}
    by_scene = Counter(target["scene_id"] for target in targets["targets"])
    assert set(by_scene.values()) == {5}
    assert len({target["target_id"] for target in targets["targets"]}) == 60


def test_reviewer_assets_and_contract_are_exact() -> None:
    cases = json.loads((PACKAGE / "review_cases.json").read_text(encoding="utf-8"))
    assets = json.loads((PACKAGE / "review_asset_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((PACKAGE / "reviewer_contract.json").read_text(encoding="utf-8"))
    assert cases["blind_first"] is True
    assert cases["scene_count"] == 12 and cases["target_count"] == 60
    assert assets["asset_count"] == 12
    for asset in assets["assets"]:
        path = PROJECT / asset["project_relative_path"]
        assert path.is_file() and path.stat().st_size == asset["byte_size"]
        assert sha256(path) == asset["sha256"]
    packed = json.dumps(contract).lower()
    assert "team" not in packed
    assert contract["candidate_events_required"] == 60
    assert contract["scene_events_required"] == 12
    assert contract["latest_acknowledged_events_required"] == 72
    assert contract["completion_receipt_required"] == 1
    assert not (PACKAGE / "human_decisions").exists()


def test_append_only_acknowledgement_and_exact_completion(tmp_path: Path) -> None:
    store = ReviewStore(PACKAGE, tmp_path / "decisions")
    for scene in store.cases["scenes"]:
        for target in scene["targets"]:
            result = store.save_event(
                {"event_type": "candidate", "target_id": target["target_id"], "answers": {"validity": "NOT_SURE"}}
            )
            assert result["acknowledgement_receipt_id"].startswith("ack-")
        result = store.save_event(
            {
                "event_type": "scene",
                "scene_id": scene["scene_id"],
                "answers": {"missed_relevant_people": "NOT_SURE"},
                "full_frame_coverage_confirmed": True,
            }
        )
    assert result["all_cases_complete"] is True
    state = store.state()
    assert state["candidate_count"] == 60 and state["scene_count"] == 12
    assert state["all_cases_complete"] is True
    assert len(list((tmp_path / "decisions/events/candidate").glob("*.json"))) == 60
    assert len(list((tmp_path / "decisions/events/scene").glob("*.json"))) == 12
    assert len(list((tmp_path / "decisions/receipts/acknowledgements").glob("*.json"))) == 72
    assert len(list((tmp_path / "decisions/receipts/completion").glob("*.json"))) == 1


def test_live_acceptance_visuals_and_exact_handoff() -> None:
    acceptance = load("05_VISUAL_QA/live_edge_acceptance.json")
    assert acceptance["classification"] == "PASS_LIVE_EDGE_C3A5C_REVIEWER"
    assert acceptance["actual_local_server"] is True
    assert acceptance["browser"] == "INSTALLED_MICROSOFT_EDGE"
    assert acceptance["temporary_candidate_events"] == 60
    assert acceptance["temporary_scene_events"] == 12
    assert acceptance["uncaught_javascript_exceptions"] == 0
    assert len(list((STAGE / "05_VISUAL_QA").glob("*.png"))) == 2
    handoff = STAGE / "07_REVIEW_PACK/CHATGPT_HANDOFF"
    assert len(list(handoff.iterdir())) == 10
    manifest = json.loads((handoff / "10_MANIFEST.json").read_text(encoding="utf-8"))
    assert len(manifest["files"]) == 9
    assert "10_MANIFEST.json" not in {row["filename"] for row in manifest["files"]}
    for row in manifest["files"]:
        path = handoff / row["filename"]
        assert path.stat().st_size == row["byte_size"] and sha256(path) == row["sha256"]
