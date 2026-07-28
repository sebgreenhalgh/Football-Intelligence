import hashlib
import json
from pathlib import Path


ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
WORKSPACE = ROOT / r"experiments\football_observation_reasoner\part 5\G7C_DATASET_INVENTORY_AND_SPLIT_v1"
MATCHES = ["117092", "117093", "118575", "118576", "118577", "118578", "128057", "128058", "132831", "132877"]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_final_review_schema_and_approvals() -> None:
    review = load(WORKSPACE / r"05_CONDITION_REVIEW\HUMAN_CONDITION_REVIEW_FINAL.json")
    assert review["schema_version"] == "g7c.human_condition_review.v2"
    assert {row["match_id"] for row in review["matches"]} == set(MATCHES)
    assert all(
        row["representative_frame_approved"] and row["proposed_split_approved"] and row["team_mapping_confirmed"]
        for row in review["matches"]
    )
    assert all(row["team_1_primary_colour"] != "BLULE" for row in review["matches"])


def test_setup_semantic_equivalence_and_pitch_pending() -> None:
    review = {
        row["match_id"]: row
        for row in load(WORKSPACE / r"05_CONDITION_REVIEW\HUMAN_CONDITION_REVIEW_FINAL.json")["matches"]
    }
    for match in MATCHES:
        setup = load(ROOT / "matches" / match / "calibration" / "match_setup.json")
        assert setup["match_id"] == match
        assert setup["dataset_split"]["status"] == "FROZEN_HUMAN_APPROVED"
        assert setup["dataset_split"]["human_approved"] is True
        assert setup["dataset_split"]["frozen"] is True
        assert setup["dataset_split"]["proposed_assignment"] == review[match]["proposed_split"]
        assert setup["pitch_calibration"]["status"] == "HUMAN_REQUIRED"
        assert setup["human_review"]["representative_frame_approved"] is True


def test_frozen_split_manifest_and_artifacts() -> None:
    split_root = ROOT / r"datasets\soccertrack_v2\splits\split_v1"
    manifest = load(split_root / "split_manifest.json")
    assert manifest["status"] == "FROZEN_HUMAN_APPROVED" and manifest["frozen"] is True
    assert manifest["membership"]["TRAIN_DEVELOPMENT"] == ["117092", "117093", "118575", "118576", "118577", "128058"]
    assert manifest["membership"]["SEALED_HOLDOUT"] == ["132831", "132877"]
    assert (split_root / "split_manifest.sha256").read_text().split()[0] == hashlib.sha256(
        (split_root / "split_manifest.json").read_bytes()
    ).hexdigest()
    artifact = load(WORKSPACE / r"05_CONDITION_REVIEW\HUMAN_APPROVAL_ARTIFACT_MANIFEST.json")
    assert len(artifact["files"]) == 12
    for item in artifact["files"]:
        path = ROOT / Path(item["project_relative_path"])
        assert path.stat().st_size == item["byte_size"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_correction_and_source_mutation_evidence() -> None:
    event = load(ROOT / r"matches\117093\manifests\source_correction_events.json")[0]
    mutation = json.loads((WORKSPACE / r"07_TESTS_AND_LOGS\source_mutation_check.json").read_text(encoding="utf-8-sig"))
    assert event["event_type"] == "AUTHORIZED_PRE_FREEZE_SOURCE_CORRECTION"
    assert event["new_source_path"].endswith("117093_panorama_1st_half-008.mp4")
    assert mutation["changed"] == 0
