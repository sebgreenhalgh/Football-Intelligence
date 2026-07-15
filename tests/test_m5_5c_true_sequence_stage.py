from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from football_intelligence.replay.m5_5c_true_sequence_stage import (
    BASELINE_COMMIT,
    MANDATORY_REVIEW_PACK_FILES,
    _bounded_window_frames,
    _claim_evidence_audit,
    _joint_incoming_histories,
    _safe_case_results,
    _stage_final_classification,
    _write_temporal_gif,
    authorization_audit,
    detector_configurations,
    mine_blind_cases,
    select_matched_controls,
    validate_m5_5c_review_pack,
)
from football_intelligence.replay.short_window_candidate_graph import ImageBBox
from football_intelligence.replay.true_sequence_resolver import (
    FrameObservation,
    ResolverConfig,
    answer_independent_fingerprint,
    appearance_activation_gate,
    execute_ghost_intervals,
    fit_incoming_motion,
    observable_conflict_signals,
    resolve_joint_sequence,
)


def _observation(name: str, frame: int, x: float, *, height: float = 40.0) -> FrameObservation:
    return FrameObservation(
        observation_id=name,
        frame_sequence=frame,
        bbox=ImageBBox(x - 10, 100 - height, x + 10, 100),
        confidence=0.8,
    )


def _histories() -> dict[str, list[FrameObservation]]:
    return {
        "track_a": [_observation(f"a{frame}", frame, 100 + frame * 3) for frame in range(4)],
        "track_b": [_observation(f"b{frame}", frame, 180 - frame * 3) for frame in range(4)],
    }


def _frame_rows() -> dict[int, list[FrameObservation]]:
    return {
        frame: [
            _observation(f"a{frame}", frame, 100 + frame * 3),
            _observation(f"b{frame}", frame, 180 - frame * 3),
        ]
        for frame in range(4, 9)
    }


def _write_text(path: Path, value: str = "ok\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _write_review_pack(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for filename in sorted(MANDATORY_REVIEW_PACK_FILES):
        if filename == "REVIEW_PACK_MANIFEST.json":
            continue
        if filename.endswith(".json"):
            _write_text(root / filename, json.dumps({"safe": True}))
        elif filename.endswith(".jsonl"):
            _write_text(root / filename, json.dumps({"case_number": "008"}))
        elif filename.endswith(".jpg"):
            Image.new("RGB", (400, 240), (30, 90, 120)).save(root / filename, quality=92)
        elif filename.endswith(".gif"):
            first = Image.new("RGB", (400, 240), (30, 90, 120))
            second = Image.new("RGB", (400, 240), (80, 30, 120))
            first.save(root / filename, save_all=True, append_images=[second], duration=100, loop=0)
        else:
            _write_text(root / filename)
    manifest = {
        "stage_id": "test",
        "repository_commit_before": "a",
        "repository_commit_after": "b",
        "files": [],
    }
    _write_text(root / "REVIEW_PACK_MANIFEST.json", json.dumps(manifest))


def test_authorization_accepts_clean_descendant_and_baseline_constant(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    _write_text(repo / "first.txt")
    subprocess.run(["git", "add", "first.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "first"], cwd=repo, check=True, capture_output=True)
    baseline = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    _write_text(repo / "second.txt")
    subprocess.run(["git", "add", "second.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "second"], cwd=repo, check=True, capture_output=True)

    result = authorization_audit(repo, baseline_commit=baseline)

    assert result["authorization_gate_passed"] is True
    assert result["worktree_clean_at_audit"] is True
    assert BASELINE_COMMIT == "53c1a032336a59f3c3449478d27290da62fcc4fc"


def test_claim_audit_detects_endpoint_pseudosequence_and_copied_controls(tmp_path: Path) -> None:
    audit = _claim_evidence_audit(tmp_path / "m5b")
    by_claim = {row["claim"]: row for row in audit["rows"]}

    assert by_claim["actual_sequence_real_window"]["support_status"] == "unsupported"
    assert "duplicated" in by_claim["motion_fit_uses_real_observation"]["executed_evidence"]
    assert by_claim["protected_control_regressions=0"]["replacement_m5_5c_classification"] == (
        "NOT_EVALUATED_NO_ELIGIBLE_GATE"
    )
    assert by_claim["PASS_CORRECT_PATH_IN_TOPK_SAFE_REVIEW"]["support_status"] == "partially_supported"


def test_three_distinct_frame_motion_fit_and_duplicate_rejection() -> None:
    history = [_observation(f"o{frame}", frame, 10 + frame * 2) for frame in range(3)]
    fit = fit_incoming_motion(history)

    assert fit.status == "FIT_COMPLETE"
    assert fit.fitted_frame_sequences == (0, 1, 2)
    assert fit.state_mean["velocity_x"] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="duplicated observation frame"):
        fit_incoming_motion([history[0], _observation("other", 0, 20), history[2]])
    with pytest.raises(ValueError, match="duplicated observation object"):
        fit_incoming_motion([history[0], history[1], FrameObservation(**{**history[0].__dict__, "frame_sequence": 2})])


def test_insufficient_history_is_explicit() -> None:
    fit = fit_incoming_motion([_observation("one", 0, 10), _observation("two", 1, 12)])
    assert fit.status == "INSUFFICIENT_INCOMING_HISTORY"
    assert fit.usable is False


def test_conflict_uses_observable_evidence_only() -> None:
    histories = _histories()
    next_rows = {4: [_observation("shared", 4, 140, height=60)]}
    result = observable_conflict_signals(histories, next_rows)

    assert result["case_id_or_category_used"] is False
    assert "two_to_one_collapse" in result["triggers"]
    assert result["numeric_evidence"]["local_candidate_density"] == 1


def test_joint_graph_covers_every_frame_with_adjacent_edges_and_deterministic_kbest() -> None:
    first = resolve_joint_sequence(
        incoming_histories=_histories(),
        observations_by_frame=_frame_rows(),
        window_frames=list(range(9)),
        image_size=(500, 300),
    )
    second = resolve_joint_sequence(
        incoming_histories=_histories(),
        observations_by_frame=_frame_rows(),
        window_frames=list(range(9)),
        image_size=(500, 300),
    )

    assert first["classification"] == "RESOLVED_K_BEST"
    assert all(len(path) == 9 for hypothesis in first["hypotheses"] for path in hypothesis["paths"].values())
    assert all(edge["target_frame_sequence"] - edge["source_frame_sequence"] == 1 for edge in first["graph_edges"])
    assert [row["total_cost"] for row in first["hypotheses"]] == sorted(
        row["total_cost"] for row in first["hypotheses"]
    )
    assert answer_independent_fingerprint(first) == answer_independent_fingerprint(second)


def test_joint_assignment_enforces_exclusivity_and_real_merged_semantics() -> None:
    histories = _histories()
    observations = {4: [_observation("shared", 4, 140, height=80)]}
    result = resolve_joint_sequence(
        incoming_histories=histories,
        observations_by_frame=observations,
        window_frames=list(range(5)),
        image_size=(500, 300),
        config=ResolverConfig(k_best=4),
    )

    for hypothesis in result["hypotheses"]:
        nodes = [path[-1] for path in hypothesis["paths"].values()]
        detection_ids = [node["observation_id"] for node in nodes if node["node_type"] == "DETECTION"]
        assert len(detection_ids) == len(set(detection_ids))
    assert any(node["node_type"] == "MERGED_OBSERVATION" for node in result["graph_nodes"])


def test_null_propagation_and_frame_exit_boundary_gate() -> None:
    null_result = resolve_joint_sequence(
        incoming_histories={"track": _histories()["track_a"]},
        observations_by_frame={4: [], 5: []},
        window_frames=list(range(6)),
        image_size=(500, 300),
    )
    assert any(node["node_type"] == "OCCLUDED_NULL" for node in null_result["graph_nodes"])

    edge_history = {"track": [_observation(f"e{frame}", frame, 3 + frame * 0.2) for frame in range(4)]}
    exit_result = resolve_joint_sequence(
        incoming_histories=edge_history,
        observations_by_frame={4: []},
        window_frames=list(range(5)),
        image_size=(200, 150),
    )
    exit_nodes = [node for node in exit_result["graph_nodes"] if node["node_type"] == "FRAME_EXIT"]
    assert exit_nodes
    assert all(node["boundary_evidence"] for node in exit_nodes)


def test_ghost_covariance_growth_expiry_and_reentry_confirmation() -> None:
    hypothesis = {
        "paths": {
            "track": [
                {"frame_sequence": 0, "node_type": "DETECTION", "observation_id": "a"},
                {"frame_sequence": 1, "node_type": "OCCLUDED_NULL", "observation_id": None},
                {"frame_sequence": 2, "node_type": "MERGED_OBSERVATION", "observation_id": "m"},
                {"frame_sequence": 3, "node_type": "DETECTION", "observation_id": "b"},
                {"frame_sequence": 4, "node_type": "DETECTION", "observation_id": "c"},
            ]
        }
    }
    result = execute_ghost_intervals(case_id="case", hypothesis=hypothesis, max_hidden_frames=4)

    covariance = [row["covariance"] for row in result["ghost_state_rows"]]
    assert covariance == sorted(covariance)
    assert result["eligible_intervals"][0]["reentry_confirmed"] is True
    assert result["reentry_hypotheses"][0]["confirmed"] is False
    assert result["reentry_hypotheses"][1]["confirmed"] is True

    expired = execute_ghost_intervals(
        case_id="expired",
        hypothesis={
            "paths": {
                "track": [
                    {"frame_sequence": 0, "node_type": "DETECTION", "observation_id": "a"},
                    *[
                        {"frame_sequence": frame, "node_type": "OCCLUDED_NULL", "observation_id": None}
                        for frame in range(1, 5)
                    ],
                ]
            }
        },
        max_hidden_frames=2,
    )
    assert any(row["dynamic_expiry_executed"] for row in expired["ghost_state_rows"])
    assert expired["eligible_intervals"] == []
    assert expired["hidden_intervals"][0]["terminated"] is True

    leading_null = execute_ghost_intervals(
        case_id="leading-null",
        hypothesis={
            "paths": {
                "track": [
                    {"frame_sequence": 0, "node_type": "OCCLUDED_NULL", "observation_id": None},
                    {"frame_sequence": 1, "node_type": "DETECTION", "observation_id": "a"},
                    {"frame_sequence": 2, "node_type": "DETECTION", "observation_id": "b"},
                ]
            }
        },
        max_hidden_frames=2,
    )
    assert leading_null["hidden_intervals"] == []
    assert leading_null["eligible_intervals"] == []


def test_window_builder_guarantees_nine_to_fifteen_available_frames() -> None:
    frames = set(range(109, 140))
    boundary_window = _bounded_window_frames(frames, source_frame=112, target_frame=113)
    longer_gap_window = _bounded_window_frames(frames, source_frame=118, target_frame=124)

    assert len(boundary_window) == 9
    assert 112 in boundary_window and 113 in boundary_window
    assert 9 <= len(longer_gap_window) <= 15
    assert 118 in longer_gap_window and 124 in longer_gap_window


def test_joint_incoming_histories_do_not_share_observations() -> None:
    def row(frame: int, candidate_id: str, x: float) -> dict[str, object]:
        return {
            "frame_sequence": frame,
            "candidate_id": candidate_id,
            "bbox": {"x1": x - 5, "y1": 40, "x2": x + 5, "y2": 100},
            "confidence": 0.9,
        }

    candidates_by_frame = {
        frame: [row(frame, f"a-{frame}", 10 + frame), row(frame, f"b-{frame}", 14 + frame)] for frame in range(1, 5)
    }
    histories = _joint_incoming_histories(
        candidates_by_frame[4][0],
        candidates_by_frame[4][1],
        candidates_by_frame,
    )
    keys = [
        (observation.frame_sequence, observation.observation_id)
        for history in histories.values()
        for observation in history
    ]

    assert all(len(history) >= 3 for history in histories.values())
    assert len(keys) == len(set(keys))


def test_unreviewed_ghost_branch_blocks_an_otherwise_passing_stage() -> None:
    assert (
        _stage_final_classification("PASS_SAFE_ESCALATION_ONLY", "BLOCKED_NO_REVIEWED_GHOST_INTERVAL")
        == "BLOCKED_NO_REVIEWED_GHOST_INTERVAL"
    )
    assert (
        _stage_final_classification("FAIL_WRONG_CONFIDENT_RANKER", "BLOCKED_NO_REVIEWED_GHOST_INTERVAL")
        == "FAIL_WRONG_CONFIDENT_RANKER"
    )


def test_appearance_gate_requires_every_independent_condition() -> None:
    eligible = appearance_activation_gate(
        conflict_active=True,
        motion_compatible_candidate_count=2,
        geometry_margin=0.05,
        source_bbox_height=40,
        target_bbox_heights=[38, 41],
        source_contamination=0.1,
        target_contamination=0.2,
    )
    blocked = appearance_activation_gate(
        conflict_active=True,
        motion_compatible_candidate_count=2,
        geometry_margin=0.05,
        source_bbox_height=40,
        target_bbox_heights=[38, 41],
        source_contamination=0.1,
        target_contamination=0.8,
    )
    assert eligible["eligible"] is True
    assert blocked["eligible"] is False
    assert blocked["gates"]["contamination_below_threshold"] is False


def test_detector_configuration_contract_and_control_grouping(tmp_path: Path) -> None:
    configs = detector_configurations()
    assert len(configs) == 7
    assert {row["name"] for row in configs} >= {
        "canonical_baseline",
        "higher_resolution_2048",
        "native_crop_2_height",
        "native_crop_3_height",
    }
    frame_paths = {}
    candidates_by_frame = {}
    for frame in (343, 344, 347, 348, 214, 215, 217, 218, 446, 447, 449, 450):
        path = tmp_path / f"frame_{frame}.jpg"
        Image.new("RGB", (400, 200), (20, 80, 100)).save(path)
        frame_paths[frame] = path
        candidates_by_frame[frame] = [
            {
                "candidate_id": f"candidate_{frame}",
                "frame_sequence": frame,
                "bbox": {"x1": 90, "y1": 50, "x2": 110, "y2": 100},
            }
        ]
    localizations = [
        {
            "source_case_number": "004",
            "target_frame_sequence": 345,
            "reviewer_bbox": {"x1": 90, "y1": 50, "x2": 110, "y2": 100},
        },
        {
            "source_case_number": "016",
            "target_frame_sequence": 346,
            "reviewer_bbox": {"x1": 90, "y1": 50, "x2": 110, "y2": 100},
        },
        {
            "source_case_number": "009",
            "target_frame_sequence": 216,
            "reviewer_bbox": {"x1": 90, "y1": 50, "x2": 110, "y2": 100},
        },
        {
            "source_case_number": "011",
            "target_frame_sequence": 448,
            "reviewer_bbox": {"x1": 90, "y1": 50, "x2": 110, "y2": 100},
        },
    ]
    case_to_group = {
        "m5_4h1_cadence_matched_target_choice_case_004": "shared",
        "m5_4h1_cadence_matched_target_choice_case_016": "shared",
        "m5_4h1_cadence_matched_target_choice_case_009": "nine",
        "m5_4h1_cadence_matched_target_choice_case_011": "eleven",
    }
    controls = select_matched_controls(localizations, candidates_by_frame, frame_paths, case_to_group)
    assert len({row["trajectory_safe_region_id"] for row in controls}) == 3
    assert all(row["frame_sequence"] not in {345, 346, 216, 448} for row in controls)
    assert all(row["same_trajectory_group_as_affected"] is False for row in controls)


def test_type_safe_blind_mining_excludes_reviewed_endpoints_and_groups() -> None:
    candidate_rows = []
    by_frame: dict[int, list[dict[str, object]]] = {}
    for frame in range(0, 80):
        by_frame[frame] = []
        for index in range(3):
            row = {
                "candidate_id": f"new_{frame}_{index}",
                "frame_sequence": frame,
                "bbox": {"x1": 100 + index * 12, "y1": 50, "x2": 120 + index * 12, "y2": 100},
                "confidence": 0.8,
                "frame_file": f"frame_{frame}.jpg",
            }
            candidate_rows.append(row)
            by_frame[frame].append(row)
    reviewed = [
        {
            "endpoint_safe_group_id": "old_endpoint",
            "local_assignment_neighbourhood_id": "old_neighbourhood",
            "source_candidate_id": "old_source",
            "target_options": [{"target_candidate_id": "old_target"}],
        }
    ]
    groups = {"groups": {"old_group": {}}, "case_to_group": {}, "endpoint_to_group": {}}
    result = mine_blind_cases(candidate_rows, by_frame, reviewed, groups, maximum_cases=12)

    assert result["selected"]
    assert result["identifier_domains_compared_type_safely"] is True
    assert result["no_previous_reviewed_group_leakage"] is True
    assert result["no_previous_endpoint_candidate_leakage"] is True
    assert len({row["local_assignment_neighbourhood_id"] for row in result["selected"]}) == len(result["selected"])


def test_temporal_gif_contains_real_multiple_frames(tmp_path: Path) -> None:
    frame_paths = {}
    for frame in range(4):
        path = tmp_path / f"frame_{frame}.jpg"
        Image.new("RGB", (640, 240), (20 + frame * 20, 80, 100)).save(path)
        frame_paths[frame] = path
    target = _write_temporal_gif(
        tmp_path / "evidence.gif",
        frame_paths,
        [0, 1, 2, 3],
        source_frame=0,
        target_frame=3,
        source_bbox={"x1": 10, "y1": 20, "x2": 30, "y2": 60},
        target_options=[{"target_bbox": {"x1": 20, "y1": 20, "x2": 40, "y2": 60}}],
    )
    with Image.open(target) as image:
        assert image.n_frames == 4


def test_safe_case_rows_strip_answer_and_canonical_identifiers() -> None:
    rows = _safe_case_results(
        [
            {
                "case_number": "008",
                "human_target_observation_id": "m5_4h1_pc_secret",
                "human_decision": "PATH_A",
                "correct_path_in_top2": True,
            }
        ]
    )
    text = json.dumps(rows)
    assert "m5_4h1_pc" not in text
    assert "human_decision" not in text
    assert rows[0]["correct_path_in_top2"] is True


def test_review_pack_validator_enforces_flat_twenty_file_contract(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    _write_review_pack(pack)

    result = validate_m5_5c_review_pack(pack)

    assert result["passed"] is True, result["errors"]
    assert result["file_count"] == 20
    assert result["visual_file_count"] == 2
    _write_text(pack / "extra.txt")
    result = validate_m5_5c_review_pack(pack)
    assert result["passed"] is False
    assert any("exceeds 20" in error for error in result["errors"])
