from __future__ import annotations

from pathlib import Path

import pytest

from football_intelligence.research_handoff.review_pack import (
    REQUIRED_REVIEW_PACK_FILES,
    ReviewPackBuilder,
    ReviewPackItem,
    validate_review_pack_directory,
)


def _source_files(tmp_path: Path) -> dict[str, Path]:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    paths = {}
    for name in sorted(REQUIRED_REVIEW_PACK_FILES - {"REVIEW_PACK_MANIFEST.json"}):
        path = source_root / name
        path.write_text(f"{name}\n", encoding="utf-8")
        paths[name] = path
    return paths


def test_review_pack_builder_creates_flat_valid_pack(tmp_path: Path) -> None:
    builder = ReviewPackBuilder(
        root=tmp_path / "pack",
        stage_id="stage",
        repository_commit_before="before",
        repository_commit_after=None,
    )
    for name, path in _source_files(tmp_path).items():
        builder.add_file(ReviewPackItem(filename=name, source_path=path, purpose=f"Purpose for {name}"))

    builder.copy_items()
    builder.write_manifest()

    errors, warnings = validate_review_pack_directory(tmp_path / "pack")
    assert errors == []
    assert warnings == []


def test_review_pack_rejects_forbidden_suffix(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_text("not a real model", encoding="utf-8")
    builder = ReviewPackBuilder(
        root=tmp_path / "pack",
        stage_id="stage",
        repository_commit_before=None,
        repository_commit_after=None,
    )

    with pytest.raises(ValueError):
        builder.add_file(ReviewPackItem(filename="model.pt", source_path=model, purpose="forbidden"))


def test_review_pack_validator_flags_nested_files(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    (pack / "nested").mkdir(parents=True)
    (pack / "nested" / "file.txt").write_text("bad", encoding="utf-8")

    errors, _ = validate_review_pack_directory(pack)

    assert any("nested files" in error for error in errors)
