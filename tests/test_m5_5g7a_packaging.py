from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from football_intelligence.football_observation_reasoner import packaging
from football_intelligence.football_observation_reasoner.packaging import (
    MAX_REVIEW_PACK_BYTES,
    MAX_REVIEW_PACK_FILES,
    MAX_REVIEW_PACK_VISUALS,
    REVIEW_PACK_MANIFEST_NAME,
    ReviewPackValidationError,
    assemble_review_pack,
    review_pack_validation_errors,
    sha256_file,
    stage_safety_summary,
    validate_review_pack,
)


def _source(root: Path, name: str, payload: bytes = b"evidence\n") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _required_diff(root: Path) -> Path:
    return _source(root, "04_SOURCE_DIFF.patch", b"diff --git a/source b/source\n")


def _visual(root: Path, name: str) -> Path:
    pixel = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    return _source(root, name, pixel)


def test_review_pack_is_flat_deterministic_and_preserves_source_names(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    sources = [
        _source(source_root, "01_EXECUTIVE_OUTCOME.md"),
        _required_diff(source_root),
        _visual(source_root, "17_PERSPECTIVE.png"),
    ]

    first_manifest = assemble_review_pack(reversed(sources), tmp_path / "first")
    second_manifest = assemble_review_pack(sources, tmp_path / "second")

    expected_payload_names = sorted(path.name for path in sources)
    expected_pack_names = sorted([*expected_payload_names, REVIEW_PACK_MANIFEST_NAME])
    assert sorted(path.name for path in (tmp_path / "first").iterdir()) == expected_pack_names
    assert [row["filename"] for row in first_manifest["files"]] == expected_payload_names
    assert all("source" not in row for row in first_manifest["files"])
    assert first_manifest == second_manifest
    assert (tmp_path / "first" / REVIEW_PACK_MANIFEST_NAME).read_bytes() == (
        tmp_path / "second" / REVIEW_PACK_MANIFEST_NAME
    ).read_bytes()
    audit = validate_review_pack(tmp_path / "first")
    assert audit["passed"] is True
    assert audit["file_count"] == 4
    assert audit["visual_file_count"] == 1


def test_manifest_is_non_recursive_hash_size_inventory_with_self_omitted(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    summary = _source(source_root, "02_SUMMARY.json", b'{"outcome":"development-only"}\n')
    source_diff = _required_diff(source_root)

    assemble_review_pack([summary, source_diff], tmp_path / "pack")

    manifest = json.loads((tmp_path / "pack" / REVIEW_PACK_MANIFEST_NAME).read_text(encoding="utf-8"))
    names = [row["filename"] for row in manifest["files"]]
    assert names == ["02_SUMMARY.json", "04_SOURCE_DIFF.patch"]
    assert REVIEW_PACK_MANIFEST_NAME not in names
    assert manifest["manifest_self_hash_omitted"] is True
    for row in manifest["files"]:
        path = tmp_path / "pack" / row["filename"]
        assert row == {
            "filename": path.name,
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
        }


@pytest.mark.parametrize(
    "name",
    [
        "football_reasoner_node_rows.parquet",
        "graph_reasoner.pt",
        "visual_embedding_cache.npy",
        "review_decisions.json",
        "credentials.json",
        "match.mp4",
    ],
)
def test_review_pack_rejects_prohibited_payloads_before_writing(tmp_path: Path, name: str) -> None:
    source_root = tmp_path / "source"
    sources = [_required_diff(source_root), _source(source_root, name)]

    with pytest.raises(ReviewPackValidationError):
        assemble_review_pack(sources, tmp_path / "pack")

    assert not (tmp_path / "pack").exists()


def test_review_pack_rejects_limits_and_duplicate_flat_names(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    too_many_visuals = [_required_diff(source_root)] + [
        _visual(source_root, f"visual_{index}.png") for index in range(MAX_REVIEW_PACK_VISUALS + 1)
    ]
    with pytest.raises(ReviewPackValidationError, match="visual count"):
        assemble_review_pack(too_many_visuals, tmp_path / "visual-pack")

    count_root = tmp_path / "count"
    too_many_files = [_required_diff(count_root)] + [
        _source(count_root, f"report_{index:02d}.md") for index in range(MAX_REVIEW_PACK_FILES - 1)
    ]
    with pytest.raises(ReviewPackValidationError, match="including generated manifest"):
        assemble_review_pack(too_many_files, tmp_path / "count-pack")

    first = _source(tmp_path / "a", "same.md")
    second = _source(tmp_path / "b", "same.md")
    with pytest.raises(ReviewPackValidationError, match="duplicate flat review-pack filename"):
        assemble_review_pack([_required_diff(tmp_path / "c"), first, second], tmp_path / "duplicate-pack")


def test_review_pack_enforces_total_bytes_including_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(packaging, "MAX_REVIEW_PACK_BYTES", 128)
    source_root = tmp_path / "source"

    with pytest.raises(ReviewPackValidationError, match="total bytes including generated manifest"):
        assemble_review_pack([_required_diff(source_root)], tmp_path / "pack")

    assert MAX_REVIEW_PACK_BYTES == 50 * 1024 * 1024
    assert not (tmp_path / "pack").exists()


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("renamed_weights.md", b"PK\x03\x04binary torch archive"),
        ("renamed_embeddings.json", b'{"embeddings":[[0.1,0.2]]}\n'),
        ("renamed_decisions.json", b'{"annotations":{"case-1":{"role":"GOALKEEPER"}}}\n'),
    ],
)
def test_review_pack_rejects_renamed_forbidden_payload_content(tmp_path: Path, name: str, payload: bytes) -> None:
    source_root = tmp_path / "source"
    with pytest.raises(ReviewPackValidationError):
        assemble_review_pack([_required_diff(source_root), _source(source_root, name, payload)], tmp_path / "pack")


def test_svg_and_pdf_count_toward_visual_cap(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    sources = [
        _required_diff(source_root),
        _visual(source_root, "one.png"),
        _source(source_root, "two.svg", b'<svg xmlns="http://www.w3.org/2000/svg"></svg>\n'),
        _source(source_root, "three.pdf", b"%PDF-1.4\n%%EOF\n"),
        _visual(source_root, "four.png"),
    ]
    with pytest.raises(ReviewPackValidationError, match="visual count"):
        assemble_review_pack(sources, tmp_path / "pack")


def test_validator_rejects_nested_entries_and_manifest_tampering(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    assemble_review_pack([_required_diff(source_root), _source(source_root, "summary.md")], tmp_path / "pack")
    nested = tmp_path / "pack" / "nested"
    nested.mkdir()
    _source(nested, "payload.md")
    assert any("must be flat" in error for error in review_pack_validation_errors(tmp_path / "pack"))
    nested.joinpath("payload.md").unlink()
    nested.rmdir()
    (tmp_path / "pack" / "summary.md").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ReviewPackValidationError, match="SHA-256 mismatch"):
        validate_review_pack(tmp_path / "pack")


def test_stage_safety_summary_is_non_overridable_and_complete() -> None:
    first = stage_safety_summary()
    first["production_ready"] = True
    safety = stage_safety_summary()

    assert safety["visual_only_warning"] == "VISUAL_ONLY_NOT_METRIC"
    assert safety["development_scope"] == "SINGLE_MATCH_GROUPED_DEVELOPMENT_ONLY"
    assert safety["sandbox_only"] is True
    assert safety["match_local_only"] is True
    assert safety["no_auto_promotion"] is True
    assert safety["production_ready"] is False
    false_guards = {
        "detector_changes_performed",
        "tracker_changes_performed",
        "detector_defaults_changed",
        "tracker_defaults_changed",
        "project_defaults_changed",
        "identity_predictions_performed",
        "identity_tracking_performed",
        "temporal_predictions_performed",
        "temporal_acceptance_predictions_performed",
        "exact_visible_person_count_forcing_performed",
        "exact_22_forcing_performed",
        "exact_two_visible_goalkeepers_forcing_performed",
        "exactly_one_goalkeeper_per_team_forcing_performed",
        "hard_goalkeeper_count_forcing_performed",
    }
    assert all(safety[key] is False for key in false_guards)
    assert MAX_REVIEW_PACK_FILES == 20
    assert MAX_REVIEW_PACK_VISUALS == 3
