from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_step1c1_colour_features import base_row  # noqa: E402

from football_intelligence.step1_visual_reconstruction.colour_profile_sweep import (  # noqa: E402
    PROFILE_NAMES,
    build_colour_profile_sweep_payload,
    build_profile_sandbox_payloads,
)
from football_intelligence.step1_visual_reconstruction.io import STEP1C1_COLOUR_FEATURE_ROWS_PATH  # noqa: E402


def test_every_profile_produces_one_belief_row_per_b4_row_without_overwriting_c1() -> None:
    base_payload = {"rows": [base_row("a"), base_row("b")]}
    payload = build_colour_profile_sweep_payload(
        base_payload,
        labels_payload={"frames": []},
        frame_lookup={},
        prototype_strategies=["c1_top_chromatic"],
    )
    assert payload["profile_names"] == PROFILE_NAMES
    assert payload["canonical_c1_outputs_overwritten"] is False
    assert str(STEP1C1_COLOUR_FEATURE_ROWS_PATH).endswith("step1c1_colour_feature_rows.json")
    for profile in payload["profiles"]:
        summary = profile["prototype_strategy_summaries"][0]
        assert summary["feature_rows"] == 2
        assert summary["belief_rows"] == 2
    assert payload["summary"]["all_profiles_preserve_b4_row_count"] is True


def test_best_sandbox_payloads_are_sandbox_only_and_not_auto_promoted() -> None:
    base_payload = {"rows": [base_row("a")]}
    _features, _prototypes, belief_payload, unknown_payload = build_profile_sandbox_payloads(
        base_payload,
        "torso_wider",
        "c1_top_chromatic",
        frame_lookup={},
    )
    for payload in [belief_payload, unknown_payload]:
        assert payload["sandbox_only"] is True
        assert payload["auto_promoted"] is False
        assert payload["production_ready"] is False
        assert all(row["sandbox_only"] is True for row in payload["rows"])
        assert all(row["auto_promoted"] is False for row in payload["rows"])
