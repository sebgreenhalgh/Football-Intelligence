"""Focused, read-only checks of the operator-selected provisional freeze."""

from __future__ import annotations

import copy

import pytest

from scripts import check_g7f_f_provisional as frozen


@pytest.mark.skipif(not frozen.MANIFEST.is_file(), reason="external immutable detector manifest unavailable")
def test_external_provisional_freeze_integrity() -> None:
    result = frozen.check()
    assert result["valid"] is True
    assert result["candidate_c"] == "g7f_b_r1_recall_conf_012"
    assert result["blind_repeat_identities_accessed"] is False


@pytest.mark.skipif(not frozen.MANIFEST.is_file(), reason="external immutable detector manifest unavailable")
def test_changed_candidate_identity_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    original_read = frozen.read

    def altered_read(path):
        value = original_read(path)
        if path == frozen.STAGE / "candidate_identity_mapping.json":
            value = copy.deepcopy(value)
            value["candidates"]["Candidate C"]["role"] = "LOCAL_DEFAULT_RERUN"
        return value

    monkeypatch.setattr(frozen, "read", altered_read)
    with pytest.raises(ValueError, match="role mismatch"):
        frozen.check()


@pytest.mark.skipif(not frozen.MANIFEST.is_file(), reason="external immutable detector manifest unavailable")
def test_false_promotion_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    original_read = frozen.read

    def altered_read(path):
        value = original_read(path)
        if path == frozen.MANIFEST:
            value = copy.deepcopy(value)
            value["production_ready"] = True
        return value

    monkeypatch.setattr(frozen, "read", altered_read)
    with pytest.raises(ValueError, match="production promotion"):
        frozen.check()
