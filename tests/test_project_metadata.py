"""Offline canonical metadata guard; never loads private Gold objects."""

from pathlib import Path

from scripts.check_project_metadata import check


def test_canonical_metadata_and_links():
    result = check(Path(__file__).resolve().parents[1])
    assert result["valid"] and result["agents_words"] <= 1500
