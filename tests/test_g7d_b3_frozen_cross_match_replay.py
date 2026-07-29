from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT / "SoccerTrack-v2"
STAGE = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B3_FROZEN_CROSS_MATCH_REPLAY_v1"
EXPECTED_HEAD = "53fdb18f4363afe1db1ce62fcebc90bd9bb4d9d2"
MATCHES = ("118575", "117092")
POLYGONS = {
    "118575": "fbd7f3a473acc197b4c893d90bbaa4c5d484d1e883e8df1ac4601daf4396dec1",
    "117092": "92ca8040eedd3b0ec0bb685648691f0c314d8527f3fa8f2db1823b4461e4b338",
}


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


def test_preflight_provenance_and_train_only_scope_are_hash_valid() -> None:
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() == EXPECTED_HEAD
    validation = read(STAGE / "01_INPUT_CLOSURE/input_validation.json")
    split = read(PROJECT / "datasets/soccertrack_v2/splits/split_v1/split_manifest.json")
    assert validation["status"] == "PASS_G7D_B3_INPUTS_HASH_VALID"
    assert validation["runtime_manifest_sha256"] == "e310d7ef66940303fd6f1242f34b210f38a5d88a9d0b8fadf4ff7327b5b8464c"
    assert validation["baseline_contract"] == "G7D_B2_FROZEN_128058_FOLDWISE_BASELINE_V1"
    assert validation["total_frame_count"] == 64
    for match_id in MATCHES:
        assert match_id in split["membership"]["TRAIN_DEVELOPMENT"]
        polygon = PROJECT / f"matches/{match_id}/calibration/pitch_polygon_v1/pitch_polygon.json"
        assert sha256(polygon) == POLYGONS[match_id]


def test_exact_64_frame_preinference_sampling_is_hash_complete() -> None:
    all_frames = []
    source_hashes = {}
    for match_id in MATCHES:
        manifest = read(STAGE / "02_REPLAY_INPUTS" / match_id / "ordered_sampling_manifest.json")
        receipt = read(STAGE / "02_REPLAY_INPUTS" / match_id / "pre_inference_freeze_receipt.json")
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
            image = PROJECT / frame["project_relative_path"]
            assert image.stat().st_size == frame["frame_byte_size"] and sha256(image) == frame["frame_sha256"]
            source_hashes[frame["source_video_relative_path"]] = frame["source_video_sha256"]
        assert receipt["frame_count"] == 32 and receipt["inference_started_after_freeze_only"]
        all_frames.extend(frames)
    assert len(all_frames) == 64
    for relative_path, expected_hash in source_hashes.items():
        assert sha256(PROJECT / relative_path) == expected_hash


def test_every_successful_frame_has_one_pass_and_five_independent_folds() -> None:
    frames = rows(STAGE / "03_REPLAY_RUNTIME/foldwise_frame_records.jsonl")
    candidates = rows(STAGE / "03_REPLAY_RUNTIME/foldwise_candidate_records.jsonl")
    execution = read(STAGE / "03_REPLAY_RUNTIME/execution_receipt.json")
    assert len(frames) == 64 and Counter(frame["match_id"] for frame in frames) == {"118575": 32, "117092": 32}
    assert all(frame["successful_pass_count"] == 1 and frame["all_views_exact"] for frame in frames)
    assert execution["successful_frame_count"] == 64 and execution["successful_passes_per_frame"] == 1
    assert execution["fold_order"] == [0, 1, 2, 3, 4] and execution["all_candidates_five_fold_complete"]
    assert all([fold["fold_id"] for fold in candidate["fold_outputs"]] == [0, 1, 2, 3, 4] for candidate in candidates)
    assert len({candidate["candidate_local_id"] for candidate in candidates}) == len(candidates)
    forbidden = {
        "aggregate_probability",
        "aggregate_label",
        "majority_vote",
        "consensus_label",
        "primary_fold",
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


def test_transfer_shortlist_visuals_and_handoff_are_complete() -> None:
    comparison = read(STAGE / "04_TRANSFER_COMPARISON/foldwise_transfer_comparison.json")
    shortlist = read(STAGE / "05_RISK_SHORTLIST/diagnostic_shortlist.json")
    assert comparison["engineering_reference_only"] and comparison["aggregation"] == "NONE"
    assert {row["match_id"] for row in comparison["rows"]} == set(MATCHES)
    assert shortlist["total_scene_count"] == 24 and shortlist["per_match_count"] == {"117092": 12, "118575": 12}
    for match_id in MATCHES:
        selected = [row for row in shortlist["scenes"] if row["match_id"] == match_id]
        assert Counter(row["primary_quota"] for row in selected) == {
            "LOW_PROPOSAL_OR_CANDIDATE_SUPPLY": 2,
            "HIGH_PROPOSAL_OR_OFF_PITCH_BURDEN": 2,
            "HIGH_FOLD_LOCAL_UNCERTAINTY": 2,
            "HIGH_CROSS_FOLD_DISAGREEMENT": 2,
            "HIGH_SCALE_OR_PERSPECTIVE_RESIDUAL": 2,
            "STABLE_CONTROL": 2,
        }
        assert len({row["frame_id"] for row in selected}) == 12
        assert all(row["temporal_separation_relaxed"] is False for row in selected)
    assert {path.name for path in (STAGE / "06_VISUAL_QA").glob("*.png")} == {
        "118575_frozen_replay_diagnostic_contact_sheet.png",
        "117092_frozen_replay_diagnostic_contact_sheet.png",
    }
    handoff = STAGE / "08_REVIEW_PACK/CHATGPT_HANDOFF"
    expected = {
        "01_EXECUTIVE_SUMMARY.json",
        "02_INPUT_RUNTIME_AND_BASELINE_PROVENANCE.json",
        "03_CROSS_MATCH_TRANSFER_RESULTS.json",
        "04_RISK_SHORTLIST.json",
        "05_DECISION.md",
        "06_RUNTIME_AND_COMPARISON_CONTRACT.md",
        "07_TESTS_SAFETY_AND_SOURCE_CHANGES.json",
        "08_118575_CONTACT_SHEET.png",
        "09_117092_CONTACT_SHEET.png",
        "10_MANIFEST.json",
    }
    assert {path.name for path in handoff.iterdir()} == expected
    manifest = read(handoff / "10_MANIFEST.json")["files"]
    assert len(manifest) == 9 and {row["filename"] for row in manifest} == expected - {"10_MANIFEST.json"}
    for row in manifest:
        path = handoff / row["filename"]
        assert path.stat().st_size == row["byte_size"] and sha256(path) == row["sha256"]


def test_orchestrator_has_no_training_or_accuracy_claim_path() -> None:
    source = (REPO / "scripts/g7d_b3_run_frozen_cross_match_replay.py").read_text(encoding="utf-8").lower()
    assert "optimizer" not in source and ".backward(" not in source and "validation_model_selection" not in source
    assert '"accuracy"' not in source and '"recall"' not in source and '"precision"' not in source
