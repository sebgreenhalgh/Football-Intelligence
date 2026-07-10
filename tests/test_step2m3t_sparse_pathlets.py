# ruff: noqa: E501

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import football_intelligence.step2_visual_continuity.topology_sparse_pathlets as m3t  # noqa: E402
from football_intelligence.step2_visual_continuity.io import (  # noqa: E402
    STEP2M1_OUTPUT_DIR,
    STEP2M2_OUTPUT_DIR,
    STEP2M3_OUTPUT_DIR,
    STEP2M3R_OUTPUT_DIR,
    STEP2M3S_OUTPUT_DIR,
    STEP2M3T_OUTPUT_DIR,
)
from football_intelligence.step2_visual_continuity.schema import (  # noqa: E402
    forbidden_keys_present,
)


def candidate(
    edge_id: str,
    source: str,
    target: str,
    source_frame: int,
    target_frame: int,
    score: float,
    *,
    eligible: bool = True,
    seed: bool = False,
    reasons: list[str] | None = None,
) -> dict:
    return {
        "continuity_edge_id": edge_id,
        "source_visible_person_base_id": source,
        "target_visible_person_base_id": target,
        "source_frame_sequence": source_frame,
        "target_frame_sequence": target_frame,
        "frame_gap": max(1, target_frame - source_frame),
        "m3t_sparse_candidate_score": score,
        "m3t_sparse_candidate_eligible": eligible,
        "m3s_handoff_seed_edge": seed,
        "m3r_edge_level_accepted": False,
        "m3r_edge_level_rejected": False,
        "m3r_edge_level_unsure": False,
        "forced_quarantine_reasons": reasons or [],
        "visual_continuity_group_id": "g",
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
    }


def test_step2m3t_output_paths_are_isolated() -> None:
    m3t_root = STEP2M3T_OUTPUT_DIR.resolve()
    blocked = [
        STEP2M1_OUTPUT_DIR.resolve(),
        STEP2M2_OUTPUT_DIR.resolve(),
        STEP2M3_OUTPUT_DIR.resolve(),
        STEP2M3R_OUTPUT_DIR.resolve(),
        STEP2M3S_OUTPUT_DIR.resolve(),
    ]
    for path in m3t.step2m3t_output_paths().values():
        resolved = path.resolve()
        assert resolved == m3t_root or m3t_root in resolved.parents
        assert all(resolved != root and root not in resolved.parents for root in blocked)


def test_sparse_selection_enforces_one_outgoing_and_one_incoming() -> None:
    rows = [
        candidate("e1", "a", "b", 1, 2, 0.9),
        candidate("e2", "a", "c", 1, 2, 0.8),
        candidate("e3", "d", "b", 1, 2, 0.85),
        candidate("e4", "x", "y", 1, 2, 0.7),
    ]
    selected, quarantined = m3t.sparse_select_edges(rows)
    selected_ids = {row["continuity_edge_id"] for row in selected}
    quarantined_ids = {row["continuity_edge_id"] for row in quarantined}
    assert selected_ids == {"e1", "e4"}
    assert {"e2", "e3"} <= quarantined_ids
    assert all("one_to_one_matching_rejected" in row["m3t_topology_quarantine_reasons"] for row in quarantined if row["continuity_edge_id"] in {"e2", "e3"})


def test_m3s_seed_edges_are_retained_when_valid() -> None:
    rows = [
        candidate("seed", "a", "b", 1, 2, 0.3, seed=True),
        candidate("competing", "a", "c", 1, 2, 0.95),
    ]
    selected, quarantined = m3t.sparse_select_edges(rows)
    assert {row["continuity_edge_id"] for row in selected} == {"seed"}
    assert {row["continuity_edge_id"] for row in quarantined} == {"competing"}


def test_rejected_m3r_edges_are_never_selected() -> None:
    rows = [
        candidate("rejected", "a", "b", 1, 2, 1.0, eligible=False, reasons=["rejected_by_m3r"]),
        candidate("safe", "c", "d", 1, 2, 0.7),
    ]
    selected, quarantined = m3t.sparse_select_edges(rows)
    assert {row["continuity_edge_id"] for row in selected} == {"safe"}
    rejected_row = next(row for row in quarantined if row["continuity_edge_id"] == "rejected")
    assert "rejected_by_m3r" in rejected_row["m3t_topology_quarantine_reasons"]


def test_pathlets_are_topology_safe_for_simple_chains() -> None:
    selected = [
        candidate("e1", "a", "b", 1, 2, 0.9),
        candidate("e2", "b", "c", 2, 3, 0.8),
    ]
    pathlets, quarantined_pathlets, quarantined_edge_ids = m3t.build_pathlets_from_selected_edges(selected)
    assert len(pathlets) == 1
    assert quarantined_pathlets == []
    assert quarantined_edge_ids == set()
    pathlet = pathlets[0]
    assert pathlet["max_members_per_frame"] == 1
    assert pathlet["max_in_degree"] == 1
    assert pathlet["max_out_degree"] == 1
    assert pathlet["branch_count"] == 0
    assert pathlet["merge_count"] == 0
    assert pathlet["production_ready"] is False
    assert pathlet["no_auto_promotion"] is True
    assert pathlet["human_approved"] is False
    assert forbidden_keys_present(pathlet) == []


def test_unsafe_pathlet_emits_quarantine_reason() -> None:
    selected = [
        candidate("e1", "a", "b", 1, 2, 0.9),
        candidate("e2", "b", "c", 2, 2, 0.8),
    ]
    pathlets, quarantined_pathlets, quarantined_edge_ids = m3t.build_pathlets_from_selected_edges(selected)
    assert pathlets == []
    assert len(quarantined_pathlets) == 1
    assert quarantined_edge_ids == {"e1", "e2"}
    assert "duplicate_frame_member_conflict" in quarantined_pathlets[0]["m3t_topology_quarantine_reasons"]
    assert forbidden_keys_present(quarantined_pathlets) == []
