from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
WORKSPACE = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5D2B_CANONICAL_CANDIDATE_SOURCE_REBUILD_v1"
PACKAGE = WORKSPACE / "06_REBUILT_REVIEW_PACKAGE"
SOURCE_ROOT = ROOT / r"matches\128058\runs\step_m5\06f_balanced_role_then_continuity\continuity_v11\unseen_window"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


pytestmark = pytest.mark.skipif(not WORKSPACE.exists(), reason="M5.5D.2B generated workspace is not present")


def test_authoritative_canonical_source_is_native_and_unscaled() -> None:
    manifest = read_json(WORKSPACE / "02_CANONICAL_SOURCE_DISCOVERY" / "canonical_candidate_manifest.json")
    catalog = read_json(WORKSPACE / "02_CANONICAL_SOURCE_DISCOVERY" / "canonical_frame_catalog.json")
    hash_audit = read_json(WORKSPACE / "02_CANONICAL_SOURCE_DISCOVERY" / "canonical_source_hash_audit.json")

    assert manifest["direct_source_only"] is True
    assert manifest["coordinate_space"] == "ORIGINAL_PANORAMA_PIXELS"
    assert manifest["row_count"] == 12110
    assert hash_audit["model_sha256"] == "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
    assert catalog["dimensions"] == {"width": 2730, "height": 720}
    assert catalog["frame_count"] == 600
    assert read_json(WORKSPACE / "03_CANONICAL_FRAME_AND_ROW_VALIDATION" / "frame_hash_binding_results.json") == {
        "all_hashes_match": True,
        "frame_specific_only": True,
        "multi_frame_union": False,
        "validated_rows": 1641,
    }

    bindings = [
        json.loads(line)
        for line in (WORKSPACE / "03_CANONICAL_FRAME_AND_ROW_VALIDATION" / "row_binding_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(bindings) == 1641
    assert all(row["binding_valid"] and not row["scaling_applied"] for row in bindings)
    assert {row["coordinate_space"] for row in bindings} == {"ORIGINAL_PANORAMA_PIXELS"}


def test_review_package_keeps_layers_frame_bound_and_decisions_empty() -> None:
    status = read_json(PACKAGE / "package_status.json")
    decisions = read_json(PACKAGE / "decisions" / "review_decisions.json")
    layers = [
        json.loads(line)
        for line in (WORKSPACE / "04_TRACK_AND_RECOVERY_GEOMETRY_REBUILD" / "observed_segment_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    canonical_by_frame = read_json(PACKAGE / "reviewer_manifest.json")["cases"][0]["visible_metadata"][
        "safe_anonymous_candidates_by_frame"
    ]
    canonical = [row for rows in canonical_by_frame.values() for row in rows]

    assert status["case_count"] == 9
    assert status["decisions_root_empty"] is True
    assert status["reviewer_session_id"] == "m5_5d2b_canonical_source_human_reviewer"
    assert decisions["decisions"] == {}
    assert decisions["event_sequence"] == 0
    assert layers and all(row["coordinate_space"] == "ORIGINAL_PANORAMA_PIXELS" for row in layers)
    assert canonical and all(row["layer"] == "CANONICAL_DETECTIONS" for row in canonical)
    assert all("candidate_id" not in row for row in canonical)


def test_exact_frame_assets_match_authoritative_source_hashes() -> None:
    catalog = read_json(WORKSPACE / "02_CANONICAL_SOURCE_DISCOVERY" / "canonical_frame_catalog.json")["frames"]
    catalog_by_frame = {int(row["frame_sequence"]): row for row in catalog}
    manifest = read_json(PACKAGE / "reviewer_manifest.json")
    for case in manifest["cases"]:
        for asset in case["evidence_assets"]:
            if asset["asset_type"] != "image_sequence":
                continue
            frame = int(asset["frame_sequences"][0])
            path = PACKAGE / "evidence" / case["case_id"] / asset["relative_path"]
            assert path.exists()
            assert asset["sha256"] == catalog_by_frame[frame]["byte_sha256"]
            assert sha256(path) == catalog_by_frame[frame]["byte_sha256"]


def test_legacy_geometry_is_audit_only_and_not_the_canonical_source() -> None:
    source_manifest = read_json(WORKSPACE / "04_TRACK_AND_RECOVERY_GEOMETRY_REBUILD" / "layer_source_manifest.json")
    authorization = read_json(WORKSPACE / "01_AUTHORIZATION_AND_LEGACY_GEOMETRY_AUDIT" / "authorization_audit.json")
    legacy_rows = [
        json.loads(line)
        for line in (WORKSPACE / "01_AUTHORIZATION_AND_LEGACY_GEOMETRY_AUDIT" / "legacy_geometry_chain_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert source_manifest["review_manifest_geometry_used"] is False
    assert source_manifest["screenshots_used_as_geometry"] is False
    assert authorization["prior_packages_read_only"] is True
    assert {row["geometry_classification"] for row in legacy_rows} >= {
        "MATCHES_AUTHORITATIVE_CANONICAL_ROW",
        "WRONG_SOURCE_CANDIDATE",
    }
    assert any(
        row["legacy_declared_dimensions"]
        == {"basis": "prior port-8786 builder assumption", "height": 540, "width": 2048}
        for row in legacy_rows
    )
