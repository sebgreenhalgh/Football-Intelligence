from __future__ import annotations

import hashlib
import json
import runpy
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from PIL import Image

from football_intelligence.temporal_burst_selection import (
    MATCHES,
    OFFSETS_SECONDS,
    ONTOLOGY,
    QUOTAS,
    frame_indices_for_centre,
    slot_plan,
    validate_burst_records,
    validate_ontology,
)

ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = ROOT / "SoccerTrack-v2"
STAGE = (
    ROOT
    / "experiments/football_observation_reasoner/part 7"
    / "G7E_A_TARGETED_TEMPORAL_BURST_SELECTION_AND_ANNOTATION_DESIGN_v1"
)
representative_bursts = runpy.run_path(str(REPO / "scripts/g7e_a_build_temporal_burst_design.py"))[
    "representative_bursts"
]


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_input_closure_is_exact_and_development_only() -> None:
    closure = _json(STAGE / "00_INPUT_CLOSURE/input_closure.json")
    assert closure["classification"] == "PASS_G7E_A_INPUT_PROVENANCE"
    assert closure["repository_head_before_changes"] == "f572fe8fb2ee819548eec0eb09bc57292b56aa81"
    assert closure["matches"] == list(MATCHES)
    assert closure["split"] == "TRAIN_DEVELOPMENT"
    assert closure["polygon_status"] == "HUMAN_CONFIRMED"
    assert closure["camera_policy"] == "MATCH_STABLE_CAMERA"
    assert closure["validation_or_holdout_access"] is False
    assert closure["frozen_closure"] == {
        "control_candidates": 9067,
        "frames": 144,
        "retained_candidates": 6509,
        "suppressed_candidates": 2558,
    }
    assert closure["human_evidence"] == {
        "candidate_labels": 252,
        "missed_person_marks": 25,
        "nested_ambiguous": 3,
        "nested_must_protect": 11,
        "nested_pair_reviews": 48,
        "nested_safe": 34,
        "scene_reviews": 36,
    }
    assert len(closure["source_videos"]) == 12
    corrected = next(
        row for row in closure["source_videos"] if row["match_id"] == "117093" and row["half"] == "FIRST_HALF"
    )
    assert corrected["relative_path"].endswith("117093_panorama_1st_half-008.mp4")


def test_exact_burst_quota_balance_and_companion_bounds() -> None:
    bursts = _jsonl(STAGE / "02_BURST_SELECTION/temporal_burst_manifest.jsonl")
    validation = validate_burst_records(bursts)
    assert validation == {"valid": True, "errors": [], "high_fallback_count": 3}
    assert len(bursts) == 120
    assert Counter(row["match_id"] for row in bursts) == Counter({match_id: 20 for match_id in MATCHES})
    assert Counter(row["primary_selection_class"] for row in bursts) == Counter(
        {selection_class: count * len(MATCHES) for selection_class, count in QUOTAS.items()}
    )
    by_match: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in bursts:
        by_match[str(row["match_id"])].append(row)
        assert row["selection_reason_is_human_truth"] is False
        assert row["blind_review_payload_contract"] == "NO_HUMAN_ANSWERS_OR_MODEL_CONCLUSIONS"
        assert row["production_ready"] is False
        assert row["h2_context"] == {
            "model": "H2_LOCAL_2D_WEIGHTED_MEDIAN",
            "temporal_truth": False,
            "used_as_context_only": True,
        }
    assert {match_id: sum(bool(row["companion"]) for row in rows) for match_id, rows in by_match.items()} == {
        "117092": 0,
        "117093": 4,
        "118575": 0,
        "118576": 4,
        "118577": 4,
        "128058": 0,
    }


def test_frame_resolution_and_pixel_provenance_are_frozen() -> None:
    bursts = _jsonl(STAGE / "02_BURST_SELECTION/temporal_burst_manifest.jsonl")
    frames = _jsonl(STAGE / "02_BURST_SELECTION/temporal_frame_manifest.jsonl")
    assert len(frames) == 1080
    assert len({row["frame_reference_id"] for row in frames}) == 1080
    assert len({(row["source_video_relative_path"], row["frame_index_zero_based"]) for row in frames}) == 1044
    by_burst = defaultdict(list)
    pixel_hash_by_source_frame: dict[tuple[str, int], str] = {}
    for row in frames:
        by_burst[row["burst_id"]].append(row)
        assert len(row["frame_pixel_sha256"]) == 64
        int(row["frame_pixel_sha256"], 16)
        assert row["frame_pixel_hash_contract"] == "SHA256_RGB24_C_CONTIGUOUS_SOURCE_DIMENSIONS"
        assert row["full_resolution_frame_persisted"] is False
        key = (row["source_video_relative_path"], row["frame_index_zero_based"])
        assert pixel_hash_by_source_frame.setdefault(key, row["frame_pixel_sha256"]) == row["frame_pixel_sha256"]
    for burst in bursts:
        rows = sorted(by_burst[burst["burst_id"]], key=lambda row: row["burst_frame_sequence"])
        assert [row["relative_offset_seconds"] for row in rows] == [float(value) for value in OFFSETS_SECONDS]
        assert [row["frame_index_zero_based"] for row in rows] == burst["frame_indices_zero_based"]
        assert tuple(burst["frame_indices_zero_based"]) == frame_indices_for_centre(
            burst["centre_frame_index_zero_based"], Decimal("25")
        )
        assert len(set(burst["frame_indices_zero_based"])) == 9


def test_selection_and_frame_manifest_hashes_validate() -> None:
    manifest = _json(STAGE / "02_BURST_SELECTION/burst_manifest_sha256.json")
    assert manifest["self_hashed"] is False
    assert len(manifest["files"]) == 4
    for row in manifest["files"]:
        path = Path(row["path"])
        assert path.stat().st_size == row["byte_size"]
        assert _sha256(path) == row["sha256"]


def test_representative_visual_selection_has_all_required_coverage() -> None:
    bursts = _jsonl(STAGE / "02_BURST_SELECTION/temporal_burst_manifest.jsonl")
    selected_ids = representative_bursts(bursts)
    assert selected_ids == representative_bursts(bursts)
    assert len(selected_ids) == len(set(selected_ids)) == 12
    selected = [row for row in bursts if row["burst_id"] in set(selected_ids)]
    assert {row["primary_selection_class"] for row in selected} == set(QUOTAS)
    assert {row["match_id"] for row in selected} == set(MATCHES)
    assert {row["perspective_band"] for row in selected} == {"FAR", "NEAR_MIDDLE"}
    assert any(row["match_id"] == "117092" for row in selected)
    assert any(row["match_id"] != "117092" for row in selected)
    tags = {tag for row in selected for tag in row["secondary_evidence_tags"]}
    assert {"NESTED_MUST_PROTECT", "HUMAN_SAFE_FRAGMENT", "MISSED_PERSON_MARK"} <= tags
    assert any(row["primary_selection_class"] == "STABLE_OPEN_PLAY_CONTROL" for row in selected)


def test_annotation_ontology_and_design_respect_identity_boundary() -> None:
    ontology = _json(STAGE / "03_ANNOTATION_PROTOCOL/temporal_annotation_ontology.json")
    assert validate_ontology(ontology["enumerations"])
    assert ontology["enumerations"] == {key: list(values) for key, values in ONTOLOGY.items()}
    assert ontology["identity_boundary"]["subject_tokens"] == ["SUBJECT_A", "SUBJECT_B", "SUBJECT_C"]
    assert ontology["identity_boundary"]["tokens_reset_each_burst"] is True
    assert ontology["identity_boundary"]["permanent_identity"] == "FORBIDDEN"
    assert ontology["identity_boundary"]["cross_burst_identity"] == "FORBIDDEN"
    assert ontology["identity_boundary"]["track_ids"] == "FORBIDDEN"
    assert ontology["team_classification"] == "INTENTIONALLY_EXCLUDED_FIRST_TEMPORAL_WAVE"
    schema = _json(STAGE / "03_ANNOTATION_PROTOCOL/temporal_event_schema_draft.json")
    assert schema["status"] == "DESIGN_ONLY_NOT_IMPLEMENTED"
    assert schema["atomic_unit"] == "ONE_FINAL_ANNOTATION_EVENT_PER_BURST"


def test_workload_and_stage_stop_are_bounded() -> None:
    workload = _json(STAGE / "03_ANNOTATION_PROTOCOL/human_workload_estimate.json")
    assert workload["burst_count"] == 120
    assert workload["frames_per_burst"] == 9
    assert workload["per_burst_minutes"] == {"high": 4.0, "low": 2.0, "median": 3.0}
    assert workload["team_classification_questions"] == 0
    assert workload["identity_questions"] == 0
    decision = _json(STAGE / "01_SELECTION_AND_ANNOTATION_CONTRACT/decision.json")
    assert decision["decision"] == "PASS_G7E_A_TEMPORAL_BURST_SELECTION_AND_ANNOTATION_DESIGN_FROZEN"
    assert decision["human_annotations_created"] is False
    assert decision["reviewer_implemented"] is False
    assert decision["inference_run"] is False
    assert decision["runtime_or_default_changed"] is False
    assert decision["stop_before"] == "G7E_B_TEMPORAL_REVIEWER_IMPLEMENTATION"


def test_exactly_two_visuals_and_no_full_resolution_frame_dump() -> None:
    visual_dir = STAGE / "04_VISUAL_QA"
    visuals = sorted(visual_dir.glob("*.png"))
    assert [path.name for path in visuals] == [
        "01_TEMPORAL_SELECTION_MATRIX.png",
        "02_REPRESENTATIVE_BURST_STRIPS.png",
    ]
    for path in visuals:
        with Image.open(path) as image:
            assert image.width >= 1800
            assert image.height >= 1000
    images = [path for path in STAGE.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    assert {path.name for path in images} == {
        "01_TEMPORAL_SELECTION_MATRIX.png",
        "02_REPRESENTATIVE_BURST_STRIPS.png",
        "08_SELECTION_MATRIX.png",
        "09_REPRESENTATIVE_BURSTS.png",
    }
    report = _json(STAGE / "05_TESTS_AND_LOGS/build_validation_report.json")
    assert report["full_resolution_images_saved"] == 0
    assert report["visual_count"] == 2


def test_handoff_has_exactly_ten_self_contained_files() -> None:
    handoff = STAGE / "06_REVIEW_PACK/CHATGPT_HANDOFF"
    files = sorted(path for path in handoff.iterdir() if path.is_file())
    assert len(files) == 10
    assert [path.name for path in files] == [
        "01_EXECUTIVE_SUMMARY.json",
        "02_INPUT_AND_SELECTION_CONTRACT.json",
        "03_BURST_SELECTION_RESULTS.json",
        "04_QUOTA_AND_PROVENANCE_RESULTS.json",
        "05_TEMPORAL_ANNOTATION_ONTOLOGY.json",
        "06_REVIEWER_WORKFLOW_AND_WORKLOAD.md",
        "07_DECISION.md",
        "08_SELECTION_MATRIX.png",
        "09_REPRESENTATIVE_BURSTS.png",
        "10_MANIFEST.json",
    ]
    manifest = _json(handoff / "10_MANIFEST.json")
    assert manifest["self_hashed"] is False
    assert len(manifest["files"]) == 9
    for row in manifest["files"]:
        path = handoff / row["filename"]
        assert path.stat().st_size == row["byte_size"]
        assert _sha256(path) == row["sha256"]


def test_frozen_slot_plan_is_deterministic() -> None:
    assert slot_plan() == slot_plan()
    assert len(slot_plan()) == 20
    assert Counter(row["selection_class"] for row in slot_plan()) == Counter(QUOTAS)
