from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT / "SoccerTrack-v2"
STAGE = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B2C_RESUME_FROZEN_128058_BASELINE_v1"
B1 = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B1_PROPOSAL_CLOSURE_AND_FOLDWISE_RUNTIME_v1"
B2 = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B2_FROZEN_128058_BASELINE_RERUN_v1"
MATCH = PROJECT / "matches/128058"
EXPECTED_HEAD = "3ca6f6840a129f5c2ebd6b592a17fd1bccaf3239"
POLYGON_SHA = "24ad1e4d143527e5a3e92cded1b5d8b10526d67b5b0d1f8b02289a91e8c65307"
RUNTIME_SHA = "e310d7ef66940303fd6f1242f34b210f38a5d88a9d0b8fadf4ff7327b5b8464c"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_continuation_is_clean_and_inputs_are_hash_bound() -> None:
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() == EXPECTED_HEAD
    stop = read(B2 / "01_INPUT_CLOSURE/pitch_geometry_resolution.json")
    split = read(PROJECT / "datasets/soccertrack_v2/splits/split_v1/split_manifest.json")
    setup = read(MATCH / "calibration/match_setup.json")
    provenance = read(STAGE / "00_CONTINUATION_PROVENANCE/continuation_provenance.json")
    assert stop["status"] == "FAIL_G7D_B2_128058_PITCH_PROVENANCE" and not stop["sampling_or_inference_started"]
    assert split["frozen"] and split["status"] == "FROZEN_HUMAN_APPROVED"
    assert "128058" in split["membership"]["TRAIN_DEVELOPMENT"]
    assert setup["team_mapping"]["team_1_primary_colour"] == "BLUE"
    assert setup["team_mapping"]["team_2_primary_colour"] == "WHITE"
    assert sha256(MATCH / "calibration/pitch_polygon_v1/pitch_polygon.json") == POLYGON_SHA
    assert setup["pitch_calibration"]["polygon_sha256"] == POLYGON_SHA
    assert provenance["prior_b2_sampling_or_inference_started"] is False
    assert sha256(B1 / "02_FOLDWISE_RUNTIME/frozen_unseen_match_runtime_manifest.json") == RUNTIME_SHA


def test_exact_32_frame_preinference_sampling_manifest_is_hash_complete() -> None:
    manifest = read(STAGE / "02_BASELINE_INPUTS/ordered_sampling_manifest.json")
    receipt = read(STAGE / "05_BASELINE_FREEZE/pre_inference_freeze_receipt.json")
    frames = manifest["frames"]
    assert manifest["frame_count"] == 32 and manifest["frames_per_half"] == 16
    assert (
        manifest["quantile_range"] == [0.08, 0.92]
        and not manifest["adaptive_resampling"]
        and not manifest["inference_started"]
    )
    assert [frame["sequence_index"] for frame in frames] == list(range(32))
    assert Counter(frame["half"] for frame in frames) == {"FIRST_HALF": 16, "SECOND_HALF": 16}
    for half in ("FIRST_HALF", "SECOND_HALF"):
        selected = [frame for frame in frames if frame["half"] == half]
        assert selected[0]["quantile"] == 0.08 and selected[-1]["quantile"] == 0.92
        assert all(
            selected[index]["frame_index_zero_based"] < selected[index + 1]["frame_index_zero_based"]
            for index in range(15)
        )
    for frame in frames:
        path = PROJECT / frame["project_relative_path"]
        assert path.stat().st_size == frame["frame_byte_size"] and sha256(path) == frame["frame_sha256"]
    assert receipt["frame_count"] == 32 and receipt["inference_started_after_freeze_only"]


def test_every_successful_frame_has_exactly_one_pass_and_five_ordered_folds() -> None:
    frames = rows(STAGE / "04_BASELINE_REFERENCE/frame_execution_records.jsonl")
    candidates = rows(STAGE / "04_BASELINE_REFERENCE/foldwise_candidate_records.jsonl")
    execution = read(STAGE / "05_BASELINE_FREEZE/execution_receipt.json")
    assert len(frames) == 32 and all(
        frame["successful_pass_count"] == 1 and frame["all_views_exact"] for frame in frames
    )
    assert execution["successful_frame_count"] == 32 and execution["successful_passes_per_frame"] == 1
    assert execution["fold_order"] == [0, 1, 2, 3, 4] and execution["all_candidates_five_fold_complete"]
    assert all([fold["fold_id"] for fold in candidate["fold_outputs"]] == [0, 1, 2, 3, 4] for candidate in candidates)
    assert len({candidate["candidate_local_id"] for candidate in candidates}) == len(candidates)
    forbidden = {
        "aggregate_probability",
        "aggregate_label",
        "majority_vote",
        "consensus_label",
        "accepted",
        "suppressed",
        "selected",
        "final_observation",
    }
    assert all(not forbidden.intersection(candidate) for candidate in candidates)
    assert all(
        candidate["p2_status"].startswith("DISABLED")
        and candidate["p3_status"].startswith("DISABLED")
        and candidate["selector_status"] == "DISABLED"
        for candidate in candidates
    )


def test_reference_distributions_visual_and_no_source_mutation() -> None:
    distributions = read(STAGE / "04_BASELINE_REFERENCE/foldwise_reference_distributions.json")
    frames = read(STAGE / "02_BASELINE_INPUTS/ordered_sampling_manifest.json")["frames"]
    assert distributions["engineering_reference_only"] and not distributions["unbiased_accuracy_evaluation"]
    assert distributions["aggregation"] == "NONE" and distributions["rows"]
    assert {row["fold_id"] for row in distributions["rows"]} == {0, 1, 2, 3, 4}
    assert {row["perspective_band"] for row in distributions["rows"]} <= {"FAR", "MIDDLE", "NEAR"}
    assert {row["pitch_state"] for row in distributions["rows"]} <= {
        "ON_PITCH",
        "OFF_PITCH",
        "BOUNDARY_UNCERTAIN",
        "UNKNOWN_PITCH_STATE",
    }
    assert list((STAGE / "06_VISUAL_QA").glob("*.png")) == [
        STAGE / "06_VISUAL_QA/128058_frozen_baseline_contact_sheet.png"
    ]
    expected_sources = {frame["source_video_relative_path"]: frame["source_video_sha256"] for frame in frames}
    for relative_path, expected_hash in expected_sources.items():
        assert sha256(PROJECT / relative_path) == expected_hash


def test_frozen_contract_and_handoff_manifest_are_complete() -> None:
    contract = read(STAGE / "05_BASELINE_FREEZE/frozen_baseline_contract.json")
    handoff = STAGE / "07_REVIEW_PACK/CHATGPT_HANDOFF"
    expected = {
        "01_EXECUTIVE_SUMMARY.json",
        "02_CONTINUATION_INPUT_AND_SAMPLING_RESULTS.json",
        "03_BASELINE_RUNTIME_AND_REFERENCE_RESULTS.json",
        "04_DECISION.md",
        "05_RUNTIME_AND_BASELINE_CONTRACT.md",
        "06_TESTS_AND_SAFETY.json",
        "07_SOURCE_DIFF.patch",
        "08_BASELINE_CONTACT_SHEET.png",
        "09_MANIFEST.json",
    }
    assert contract["contract_id"] == "G7D_B2_FROZEN_128058_FOLDWISE_BASELINE_V1"
    assert contract["aggregation"] == "NONE" and contract["production_ready"] is False
    assert {path.name for path in handoff.iterdir()} == expected
    manifest = read(handoff / "09_MANIFEST.json")["files"]
    assert len(manifest) == 8 and {row["filename"] for row in manifest} == expected - {"09_MANIFEST.json"}
    for row in manifest:
        path = handoff / row["filename"]
        assert path.stat().st_size == row["byte_size"] and sha256(path) == row["sha256"]
    source = (REPO / "scripts/g7d_b2c_run_frozen_128058_baseline.py").read_text(encoding="utf-8").lower()
    assert "optimizer" not in source and ".backward(" not in source and "validation_model_selection" not in source
