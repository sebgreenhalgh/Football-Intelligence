# ruff: noqa: E501

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import football_intelligence.step2_visual_continuity.sparse_handoff_package as m4  # noqa: E402
from football_intelligence.step2_visual_continuity.io import (  # noqa: E402
    STEP2M1_OUTPUT_DIR,
    STEP2M2_OUTPUT_DIR,
    STEP2M3_OUTPUT_DIR,
    STEP2M3R_OUTPUT_DIR,
    STEP2M3S_OUTPUT_DIR,
    STEP2M3T_OUTPUT_DIR,
    STEP2M4_OUTPUT_DIR,
)
from football_intelligence.step2_visual_continuity.schema import forbidden_keys_present  # noqa: E402


def sample_m3t_pathlet(**overrides) -> dict:
    row = {
        "pathlet_id": "step2m3t_pathlet_000001",
        "member_visible_person_base_ids": ["v1", "v2", "v3"],
        "member_frame_sequences": [10, 11, 12],
        "accepted_continuity_edge_ids": ["e1", "e2"],
        "min_frame_sequence": 10,
        "max_frame_sequence": 12,
        "frame_span": 2,
        "seconds_span": 0.2,
        "max_members_per_frame": 1,
        "max_in_degree": 1,
        "max_out_degree": 1,
        "branch_count": 0,
        "merge_count": 0,
        "from_branch_merge_heavy_region": True,
        "from_duplicate_frame_heavy_region": False,
        "from_role_context_mixed_region": True,
        "contains_m3s_handoff_seed_edge": False,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }
    row.update(overrides)
    return row


def sample_summary(**overrides) -> dict:
    row = {
        "m4_handoff_pathlet_count": 1,
        "m4_handoff_edge_count": 2,
        "overlay_gif_count": 1,
        "overlay_strip_count": 1,
        "overlay_asset_count": 4,
        "pathlets_over_cap": 0,
        "duplicate_frame_pathlets": 0,
        "branch_merge_pathlets": 0,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }
    row.update(overrides)
    return row


def test_step2m4_output_paths_are_isolated() -> None:
    m4_root = STEP2M4_OUTPUT_DIR.resolve()
    blocked = [
        STEP2M1_OUTPUT_DIR.resolve(),
        STEP2M2_OUTPUT_DIR.resolve(),
        STEP2M3_OUTPUT_DIR.resolve(),
        STEP2M3R_OUTPUT_DIR.resolve(),
        STEP2M3S_OUTPUT_DIR.resolve(),
        STEP2M3T_OUTPUT_DIR.resolve(),
    ]
    for path in m4.step2m4_output_paths().values():
        resolved = path.resolve()
        assert resolved == m4_root or m4_root in resolved.parents
        assert all(resolved != root and root not in resolved.parents for root in blocked)


def test_step2m4_loads_m3t_reviewed_decisions_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "step2m3t_reviewed_sparse_pathlet_decisions.json"
    path.write_text(json.dumps({"rows": [{"pathlet_id": "p1"}, {"continuity_edge_id": "e1"}]}), encoding="utf-8")
    rows = m4.load_m3t_reviewed_decision_rows(path)
    assert len(rows) == 2
    assert rows[0]["pathlet_id"] == "p1"


def test_step2m4_requires_m3t_future_handoff_ready() -> None:
    checks = m4.m4_freeze_gate_checks(
        m3t_handoff={"future_handoff_ready_candidate": False, "forbidden_keys_present": []},
        m3t_progress={"reviewed_candidates": 40, "sparse_pathlet_review_completed": True},
        m3t_validation={"forbidden_keys_present": []},
        summary=sample_summary(),
        forbidden=[],
        viewer_exists=True,
    )
    assert checks["m3t_future_handoff_ready_candidate"] is False


def test_step2m4_pathlet_row_is_cap_safe_branch_free_and_visual_only() -> None:
    decision = {"pathlet_id": "step2m3t_pathlet_000001", "human_review_decision": "accept_sparse_pathlet_for_visual_handoff"}
    row = m4.make_m4_handoff_pathlet_row(1, sample_m3t_pathlet(), decision)
    assert row["reviewed_by_m3t"] is True
    assert row["handoff_ready"] is True
    assert row["pathlet_not_identity"] is True
    assert row["pathlet_not_player_slot"] is True
    assert row["pathlet_not_goalkeeper_slot"] is True
    assert row["max_members_per_frame"] == 1
    assert row["branch_count"] == 0
    assert row["merge_count"] == 0
    assert row["production_ready"] is False
    assert row["no_auto_promotion"] is True
    assert row["human_approved"] is False
    assert forbidden_keys_present(row) == []


def test_step2m4_violation_counts_detect_unsafe_pathlets() -> None:
    rows = [
        m4.make_m4_handoff_pathlet_row(1, sample_m3t_pathlet()),
        m4.make_m4_handoff_pathlet_row(2, sample_m3t_pathlet(pathlet_id="p2", frame_span=31, seconds_span=3.1, max_members_per_frame=2, branch_count=1)),
    ]
    counts = m4.count_pathlet_violations(rows)
    assert counts["pathlets_over_cap"] == 1
    assert counts["duplicate_frame_pathlets"] == 1
    assert counts["branch_merge_pathlets"] == 1


def test_step2m4_viewer_html_exists(monkeypatch, tmp_path: Path) -> None:
    viewer_path = tmp_path / "step2m4_sparse_handoff_viewer.html"
    monkeypatch.setattr(m4, "STEP2M4_VIEWER_HTML_PATH", viewer_path)
    row = m4.make_m4_handoff_pathlet_row(1, sample_m3t_pathlet())
    m4.write_viewer([row], sample_summary())
    text = viewer_path.read_text(encoding="utf-8")
    assert viewer_path.exists()
    assert "visual-only sparse continuity" in text
    assert "Do not infer identity" in text


def test_step2m4_overlay_assets_gate_requires_generated_assets() -> None:
    passing = m4.m4_freeze_gate_checks(
        m3t_handoff={"future_handoff_ready_candidate": True, "forbidden_keys_present": []},
        m3t_progress={"reviewed_candidates": 40, "sparse_pathlet_review_completed": True},
        m3t_validation={"forbidden_keys_present": []},
        summary=sample_summary(overlay_gif_count=1, overlay_strip_count=1),
        forbidden=[],
        viewer_exists=True,
    )
    failing = m4.m4_freeze_gate_checks(
        m3t_handoff={"future_handoff_ready_candidate": True, "forbidden_keys_present": []},
        m3t_progress={"reviewed_candidates": 40, "sparse_pathlet_review_completed": True},
        m3t_validation={"forbidden_keys_present": []},
        summary=sample_summary(overlay_gif_count=0, overlay_strip_count=1),
        forbidden=[],
        viewer_exists=True,
    )
    assert passing["m4_overlay_assets_generated"] is True
    assert failing["m4_overlay_assets_generated"] is False


def test_step2m4_guardrail_defaults_are_not_promoted() -> None:
    fields = m4.m4_guardrail_fields()
    assert fields["match_local_only"] is True
    assert fields["safe_to_apply_globally"] is False
    assert fields["requires_future_match_validation"] is True
    assert fields["production_ready"] is False
    assert fields["no_auto_promotion"] is True
    assert fields["human_approved"] is False
    assert fields["no_identity_tracking_performed"] is True
    assert fields["no_player_slots_assigned"] is True
    assert fields["no_goalkeeper_slots_assigned"] is True
    assert fields["no_metric_event_tactical_or_physical_performance_analysis"] is True
