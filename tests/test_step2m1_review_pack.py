from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.review_pack import (  # noqa: E402
    manifest_payload,
    validate_review_pack_entries,
)


def test_review_pack_manifest_stays_within_twenty_file_limit() -> None:
    entries = [{"name": f"{index:02d}.json", "kind": "json", "path": f"/tmp/{index}.json"} for index in range(20)]
    validate_review_pack_entries(entries)
    manifest = manifest_payload(entries, {"step2m1_visual_continuity_freeze_candidate_created": False})
    assert manifest["review_pack_file_count"] == 20
    assert manifest["review_pack_file_limit"] == 20
    with pytest.raises(RuntimeError):
        validate_review_pack_entries(entries + [{"name": "too_many.json"}])
