from __future__ import annotations

from pathlib import Path

from football_intelligence.replay.blind_hard_continuity import (
    F1_DIAGNOSTIC_CLASSIFICATION,
    PRINCIPAL_FEATURES,
    _balance_audits,
    _case_counts,
    _select_hard_cases,
    _selected_rows_for_audit,
    _write_blind_workbench,
    raw_feature_shortcut_audit,
)
from football_intelligence.review.schemas import CONTINUITY_NOT_APPLICABLE_DECISION


def _audit_row(label: str, iou: float, center: float, foot: float) -> dict[str, object]:
    return {
        "proposed_class": label,
        "bbox_iou": iou,
        "center_delta_px": center,
        "footpoint_delta_px": foot,
        "bbox_area_ratio": 1.0,
        "aspect_ratio_change": 0.02,
        "appearance_similarity": 0.7,
        "continuity_score": 0.62,
        "competing_candidate_margin": 0.08,
        "reciprocal_rank": 1,
        "intermediate_observed_support": 1.0,
        "crop_quality": 0.8,
        "occlusion": 0.0,
        "iou_band": "reviewable_iou",
        "delta_band": "bounded_delta",
        "score_band": "high_reviewable_score",
    }


def test_raw_numeric_audit_detects_disjoint_f1_geometry_and_blocks_perfect_threshold() -> None:
    rows = [
        _audit_row("likely_positive", 0.92, 0.1, 0.2),
        _audit_row("likely_positive", 0.99, 0.9, 0.9),
        _audit_row("likely_negative", 0.54, 4.0, 4.1),
        _audit_row("likely_negative", 0.64, 12.0, 13.0),
    ]
    audit = raw_feature_shortcut_audit(rows, artifact="test_raw_shortcut")

    assert {"bbox_iou", "center_delta_px", "footpoint_delta_px"} <= set(audit["disjoint_numeric_range_features"])
    assert {"bbox_iou", "center_delta_px", "footpoint_delta_px"} <= set(audit["perfect_univariate_threshold_features"])
    assert audit["features"]["bbox_iou"]["best_one_dimensional_threshold"]["balanced_accuracy"] == 1.0
    assert audit["feature_band_labels_hiding_raw_separation"] is True
    assert audit["passes_raw_feature_overlap_gates"] is False


def test_raw_overlap_audit_passes_when_principal_ranges_overlap_without_single_feature_perfection() -> None:
    rows = [
        _audit_row("likely_positive", 0.60, 3.0, 3.0),
        _audit_row("likely_positive", 0.82, 8.0, 8.0),
        _audit_row("likely_positive", 0.68, 6.0, 6.5),
        _audit_row("likely_negative", 0.62, 3.5, 3.2),
        _audit_row("likely_negative", 0.80, 7.0, 7.5),
        _audit_row("likely_negative", 0.70, 6.2, 5.8),
    ]
    audit = raw_feature_shortcut_audit(rows, artifact="test_raw_overlap")

    for feature in ("bbox_iou", "center_delta_px", "footpoint_delta_px"):
        assert audit["features"][feature]["ranges_overlap"] is True
        assert audit["features"][feature]["best_one_dimensional_threshold"]["balanced_accuracy"] < 1.0
    assert audit["passes_raw_feature_overlap_gates"] is True


def _hard_row(
    index: int,
    *,
    gap: int,
    team: str,
    iou: float,
    center: float,
    competing: int,
    occlusion: float,
    frame: int | None = None,
) -> dict[str, object]:
    frame = (frame if frame is not None else ((index % 4) * 150 + (index * 7) % 120)) + 5
    window_start = (frame // 30) * 30
    quartile = frame // 150
    quartile_start = quartile * 150
    quartile_end = min(599, (quartile + 1) * 150 - 1)
    return {
        "role_partitioned_continuity_candidate_id": f"cand_{index:04d}",
        "source_visible_person_base_id": f"source_{index:04d}",
        "target_visible_person_base_id": f"target_{index:04d}",
        "source_frame_sequence": frame,
        "target_frame_sequence": frame + gap,
        "frame_gap": gap,
        "bbox_iou": iou,
        "center_delta_px": center,
        "footpoint_delta_px": center + 0.25,
        "bbox_area_ratio": 1.0 + ((index % 5) - 2) * 0.015,
        "aspect_ratio_change": 0.01 * (index % 4),
        "appearance_similarity": 0.66 + (index % 5) * 0.015,
        "continuity_score": 0.58 + (index % 4) * 0.02,
        "competing_candidate_margin": 0.05 + (index % 4) * 0.01,
        "reciprocal_rank": 1 + (index % 2),
        "intermediate_observed_support": 1.0,
        "crop_quality": 0.72,
        "occlusion": occlusion,
        "competing_candidate_count": competing,
        "team_partition": team,
        "source_temporal_quartile": f"q{quartile + 1}_{quartile_start:03d}_{quartile_end:03d}",
        "source_thirty_frame_window": f"f{window_start:03d}_{window_start + 29:03d}",
        "source_spatial_region_bucket": f"region_{index % 9}",
        "bbox_size_bucket": ["small_bbox", "medium_bbox", "large_bbox"][index % 3],
        "effective_role_context": f"{team}_outfield_visual_context",
        "hard_positive_score": 3.0 + float(iou <= 0.8) + float(center >= 3.0) + float(competing > 0) * 0.4,
        "hard_negative_score": 3.0 + float(iou >= 0.75) + float(center <= 5.0) + float(competing > 0),
    }


def _synthetic_hard_pool() -> list[dict[str, object]]:
    rows = []
    for index in range(72):
        rows.append(
            _hard_row(
                index,
                gap=(index % 3) + 1,
                team="team_1" if index % 2 == 0 else "team_2",
                iou=[0.64, 0.76, 0.83, 0.69, 0.79, 0.58][index % 6],
                center=[3.0, 4.5, 6.5, 8.0, 5.5, 9.5][index % 6],
                competing=1 if index % 2 == 0 else 0,
                occlusion=1.0 if index % 5 == 0 else 0.0,
            )
        )
    return rows


def test_hard_case_selection_keeps_overlap_balance_clusters_and_endpoint_bounds() -> None:
    selection = _select_hard_cases(_synthetic_hard_pool(), limit_per_class=15)
    selected_rows = [*selection["likely_positive"], *selection["likely_negative"]]
    counts = _case_counts(selection)
    balance = _balance_audits(selection)
    audit = raw_feature_shortcut_audit(_selected_rows_for_audit(selection), artifact="test_selected_overlap")

    assert counts["proposed_positive_count"] == 15
    assert counts["proposed_negative_count"] == 15
    assert counts["lower_iou_positive_count"] >= 5
    assert counts["high_iou_negative_count"] >= 5
    assert counts["competing_candidate_count_by_proposed_class"]["likely_positive"] >= 5
    assert counts["competing_candidate_count_by_proposed_class"]["likely_negative"] >= 5
    assert balance["endpoint"]["endpoint_reuse_max"] <= 2
    assert min(balance["equivalence"]["independent_cluster_count_by_class"].values()) >= 10
    assert set(balance["temporal"]["frame_gap_distribution"]["likely_positive"]) == {1, 2, 3}
    assert set(balance["temporal"]["frame_gap_distribution"]["likely_negative"]) == {1, 2, 3}
    assert set(balance["team"]["team_distribution"]["likely_positive"]) == {"team_1", "team_2"}
    assert set(balance["team"]["team_distribution"]["likely_negative"]) == {"team_1", "team_2"}
    assert len({row["equivalence_cluster_id"] for row in selected_rows}) < len(selected_rows)
    for feature in ("center_delta_px", "footpoint_delta_px"):
        assert audit["features"][feature]["ranges_overlap"] is True


def test_blind_workbench_hides_model_info_by_default_and_records_reveal_state(tmp_path: Path) -> None:
    workbench_root = tmp_path / "workbench"
    _write_blind_workbench(workbench_root)
    index = (workbench_root / "index.html").read_text(encoding="utf-8")
    script = (workbench_root / "app.js").read_text(encoding="utf-8")

    assert "Reveal model information" in index
    assert '<section id="modelPanel" class="hidden"></section>' in index
    assert "blind_hidden_model_info" in script
    assert "model_info_revealed_before_decision=${!!revealed[c.review_case_id]}" in script
    assert "proposed_bucket" not in index


def test_n_decision_is_not_binary_training_label_and_safety_constants_remain_active() -> None:
    labels = ["accept_continuity", "reject_continuity", CONTINUITY_NOT_APPLICABLE_DECISION, "unresolved"]
    binary_labels = [label for label in labels if label in {"accept_continuity", "reject_continuity"}]

    assert binary_labels == ["accept_continuity", "reject_continuity"]
    assert CONTINUITY_NOT_APPLICABLE_DECISION not in binary_labels
    assert F1_DIAGNOSTIC_CLASSIFICATION == "M5_4F1_RAW_GEOMETRY_CONFOUNDED_CONTINUITY_REVIEW_DIAGNOSTIC_ONLY"
    assert "bbox_iou" in PRINCIPAL_FEATURES
