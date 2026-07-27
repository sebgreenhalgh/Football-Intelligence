from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from PIL import Image


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPO = Path(__file__).resolve().parents[1]
BUILD = load_module(
    "m5_5g6g_build_test",
    REPO / "scripts" / "build_m5_5g6g_official_detector_bakeoff.py",
)
ADAPTER = load_module(
    "m5_5g6g_adapter_test",
    REPO / "scripts" / "m5_5g6g_detector_adapter.py",
)


def test_authorized_candidates_and_explicit_exclusions_are_exact() -> None:
    assert set(BUILD.CANDIDATES) == {"U26-S", "U26-M", "RF-S", "RF-M", "DF-S", "DF-M"}
    assert all("P2" not in row["checkpoint"] for row in BUILD.CANDIDATES.values())
    assert all("plus" not in row["checkpoint"].lower() for row in BUILD.CANDIDATES.values())
    assert {row["family"] for row in BUILD.CANDIDATES.values()} == {
        "ULTRALYTICS_YOLO26",
        "RF_DETR",
        "D_FINE",
    }
    source = (REPO / "scripts" / "build_m5_5g6g_official_detector_bakeoff.py").read_text(encoding="utf-8")
    assert "ARCHITECTURE_ONLY_NO_RELEASED_PRETRAINED_WEIGHTS" in source
    assert "PML_LICENCE_OUTSIDE_AUTHORIZED_SCOPE" in source


def test_official_domains_and_weight_hashes_are_frozen() -> None:
    allowed = (
        "https://github.com/ultralytics/",
        "https://storage.googleapis.com/rfdetr/",
        "https://github.com/Peterande/",
    )
    for row in BUILD.CANDIDATES.values():
        assert row["official_url"].startswith(allowed)
        assert len(row["repository_commit"]) == 40
        assert len(row["license_sha256"]) == 64
        assert len(row["checkpoint_sha256"]) == 64
        assert row["checkpoint_bytes"] > 0


def test_far_side_band_uses_only_pitch_polygon_and_fixed_constants() -> None:
    source = {
        "image_width": 2730,
        "image_height": 720,
        "pitch_polygon": [
            {"x": 100.0, "y": 100.0},
            {"x": 2600.0, "y": 200.0},
            {"x": 2500.0, "y": 500.0},
            {"x": 200.0, "y": 450.0},
        ],
        "human_target_box": {"x1": 1, "y1": 1, "x2": 2, "y2": 2},
    }
    assert BUILD.far_side_band(source) == {
        "x1": 36.0,
        "y1": 36.0,
        "x2": 2664.0,
        "y2": 344.0,
    }


def test_view_matrix_has_full_band_and_exact_s3_tiles(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "source.jpg"
    Image.new("RGB", (2730, 720), "green").save(image_path)
    source_hash = BUILD.sha256_file(image_path)
    monkeypatch.setitem(BUILD.DIRS, "tmp", tmp_path / "stage_tmp")
    source = {
        "source_frame_sha256": source_hash,
        "image_path": str(image_path),
        "image_width": 2730,
        "image_height": 720,
        "frame_sequence": 1,
        "timestamp_seconds": 0.04,
        "pitch_polygon_hash": "a" * 64,
        "pitch_polygon": [
            {"x": 72.0, "y": 100.0},
            {"x": 2627.0, "y": 367.0},
            {"x": 1094.0, "y": 480.0},
            {"x": 327.0, "y": 428.0},
        ],
    }
    views = BUILD.build_views({source_hash: source})
    assert len(views) == 6
    assert [row["view_type"] for row in views].count("V0_FULL_PANORAMA") == 1
    assert [row["view_type"] for row in views].count("V1_FAR_SIDE_PITCH_BAND") == 1
    tiles = [row for row in views if row["view_type"] == "V2_EXISTING_S3_TILES"]
    assert len(tiles) == 4
    assert [row["crop_bounds_panorama_pixels"]["x1"] for row in tiles] == [
        0.0,
        768.0,
        1536.0,
        1706.0,
    ]
    assert all(row["crop_bounds_panorama_pixels"]["y2"] == 720.0 for row in tiles)


def test_low_floor_and_operating_points_are_immutable() -> None:
    assert ADAPTER.LOW_FLOOR == BUILD.LOW_FLOOR == 0.001
    assert BUILD.OPERATING_POINTS == {"T0": 0.05, "T1": 0.15, "T2": 0.25}
    assert ADAPTER.SCHEMA_VERSION == "football_intelligence.m5_5g6g.detector_adapter.v1"


def test_cross_view_fusion_selects_real_member_without_averaging() -> None:
    rows = [
        {
            "diagnostic_uuid": "low",
            "score": 0.6,
            "bbox_panorama_pixels": {"x1": 10, "y1": 10, "x2": 30, "y2": 50},
        },
        {
            "diagnostic_uuid": "high",
            "score": 0.9,
            "bbox_panorama_pixels": {"x1": 11, "y1": 11, "x2": 31, "y2": 51},
        },
    ]
    fused = BUILD.fuse_rows(rows)
    assert len(fused) == 1
    assert fused[0]["bbox"] == rows[1]["bbox_panorama_pixels"]
    assert fused[0]["lineage"] == ["high", "low"]
    assert fused[0]["representative_selection"].endswith("NO_COORDINATE_AVERAGING")


def test_recovery_never_replaces_or_duplicates_clean_baseline() -> None:
    baseline = {
        "frame": [
            {
                "proposal_id": "baseline",
                "bbox": {"x1": 10, "y1": 10, "x2": 30, "y2": 50},
                "score": 0.8,
            }
        ]
    }
    family = {
        "frame": [
            {
                "proposal_id": "aligned",
                "bbox": {"x1": 10, "y1": 10, "x2": 30, "y2": 50},
                "score": 0.9,
            },
            {
                "proposal_id": "recovery",
                "bbox": {"x1": 100, "y1": 10, "x2": 120, "y2": 50},
                "score": 0.7,
            },
        ]
    }
    combined, proof = BUILD.recovery_map(baseline, family)
    assert [row["proposal_id"] for row in combined["frame"]] == ["baseline", "recovery"]
    assert proof["clean_baseline_observations_replaced"] == 0
    assert proof["candidate_aligned_to_baseline_not_counted_twice"] == 1
    assert proof["coordinate_averaging_performed"] is False


def test_finalist_selection_is_fixed_and_bounded() -> None:
    results = []
    for index, candidate_id in enumerate(BUILD.CANDIDATES):
        results.append(
            {
                "configuration_id": f"{candidate_id}:V0:T0",
                "candidate_id": candidate_id,
                "view_type": "V0_FULL_PANORAMA",
                "operating_point": "T0",
                "threshold": 0.05,
                "target": {
                    "independent_support": 9 - index,
                    "merged_as_clean_count": 0,
                    "duplicate_excess": 0,
                },
                "control": {"independent_support": 18 - index},
                "distinct_person_suppression": 0,
                "runtime": {"p95_seconds": 1.0, "peak_allocated_gib": 1.0},
                "deterministic": True,
                "coordinate_provenance_failures": 0,
            }
        )
    selection = BUILD.select_finalists({"results": results})
    assert selection["selection_frozen_before_phase_b"] is True
    assert selection["selected_count"] == 2
    assert selection["selected_count"] <= selection["maximum_finalists"] == 3
    assert len({row["candidate_id"] for row in selection["selected"]}) == 2


def test_adapter_contract_forbids_evaluator_payload(tmp_path) -> None:
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": ADAPTER.SCHEMA_VERSION,
                "evaluator_data_present": True,
                "candidate": {},
                "views": [],
            }
        ),
        encoding="utf-8",
    )
    try:
        ADAPTER.execute(request)
    except RuntimeError as error:
        assert "FAIL_GOLD_RUNTIME_LEAKAGE" in str(error)
    else:
        raise AssertionError("evaluator payload unexpectedly reached runtime adapter")


def test_safety_and_review_pack_limits_are_not_weakened() -> None:
    assert BUILD.SAFETY["training_performed"] is False
    assert BUILD.SAFETY["fine_tuning_performed"] is False
    assert BUILD.SAFETY["identity_tracking_performed"] is False
    assert BUILD.SAFETY["pitch_gate_settings_changed"] is False
    assert BUILD.SAFETY["project_defaults_changed"] is False
    contract = json.loads((BUILD.PROMPT / "08_REVIEW_PACK_CONTRACT.json").read_text(encoding="utf-8"))
    assert contract["maximum_file_count"] == 20
    assert contract["maximum_visual_files"] == 3
    assert contract["maximum_total_bytes"] == 50 * 1024 * 1024
