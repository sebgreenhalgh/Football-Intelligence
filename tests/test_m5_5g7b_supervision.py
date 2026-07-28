from __future__ import annotations

from copy import deepcopy

import pytest

from football_intelligence.football_observation_reasoner.g7b_supervision import (
    authoritative_case_binding_sha256,
    candidate_propagation_eligibility,
    explicit_supervision_masks,
    inherited_fold_mapping_receipt,
    multiclass_head_metrics,
    nested_grouped_split_receipt,
    normalize_case_propagation_weights,
    validate_authoritative_case_binding,
    validate_k1_annotations,
)


def _annotation(
    role: str,
    team: str,
    kit: str,
    pitch: str,
    participation: str,
    *,
    seed: int,
) -> dict[str, str]:
    return {
        "schema_version": "football_intelligence.m5_5g7a.k1_annotation.v1",
        "role": role,
        "team_affiliation": team,
        "kit_state": kit,
        "pitch_state": pitch,
        "participation_state": participation,
        "certainty": "CERTAIN",
        "source_frame_sha256": f"{seed + 1:064x}",
        "target_crop_sha256": f"{seed + 1001:064x}",
        "target_binding_sha256": f"{seed + 2001:064x}",
    }


def _completed_k1_decisions() -> list[dict]:
    combinations = (
        (4, "GOALKEEPER", "TEAM_1", "MATCH_GOALKEEPER_KIT", "ON_PITCH", "ACTIVE_ON_PITCH"),
        (4, "GOALKEEPER", "TEAM_2", "MATCH_GOALKEEPER_KIT", "ON_PITCH", "ACTIVE_ON_PITCH"),
        (8, "OTHER_MATCH_OFFICIAL", "NO_TEAM", "OFFICIAL_KIT", "OFF_PITCH", "OFF_PITCH_NON_PLAYER"),
        (1, "OUTFIELD_PLAYER", "TEAM_2", "MATCH_OUTFIELD_KIT", "BOUNDARY_UNCERTAIN", "ACTIVE_ON_PITCH"),
        (1, "OUTFIELD_PLAYER", "TEAM_2", "MATCH_OUTFIELD_KIT", "OFF_PITCH", "ACTIVE_ON_PITCH"),
        (
            33,
            "OUTFIELD_PLAYER",
            "UNKNOWN_TEAM",
            "WARMUP_OR_BIB",
            "OFF_PITCH",
            "OFF_PITCH_SUBSTITUTE_OR_WARMING",
        ),
        (25, "OUTFIELD_PLAYER", "TEAM_1", "MATCH_OUTFIELD_KIT", "ON_PITCH", "ACTIVE_ON_PITCH"),
        (31, "OUTFIELD_PLAYER", "TEAM_2", "MATCH_OUTFIELD_KIT", "ON_PITCH", "ACTIVE_ON_PITCH"),
        (5, "REFEREE", "NO_TEAM", "OFFICIAL_KIT", "ON_PITCH", "ACTIVE_ON_PITCH"),
        (
            7,
            "STAFF_OR_SPECTATOR",
            "NO_TEAM",
            "STAFF_OR_SPECTATOR_CLOTHING",
            "OFF_PITCH",
            "OFF_PITCH_NON_PLAYER",
        ),
        (
            5,
            "STAFF_OR_SPECTATOR",
            "UNKNOWN_TEAM",
            "STAFF_OR_SPECTATOR_CLOTHING",
            "OFF_PITCH",
            "OFF_PITCH_NON_PLAYER",
        ),
        (1, "UNKNOWN_ROLE", "NO_TEAM", "UNKNOWN_KIT", "OFF_PITCH", "OFF_PITCH_NON_PLAYER"),
        (1, "UNKNOWN_ROLE", "UNKNOWN_TEAM", "UNKNOWN_KIT", "OFF_PITCH", "OFF_PITCH_NON_PLAYER"),
        (1, "UNKNOWN_ROLE", "UNKNOWN_TEAM", "UNKNOWN_KIT", "OFF_PITCH", "UNKNOWN_PARTICIPATION"),
        (1, "UNKNOWN_ROLE", "UNKNOWN_TEAM", "UNKNOWN_KIT", "ON_PITCH", "UNKNOWN_PARTICIPATION"),
    )
    decisions = []
    for count, role, team, kit, pitch, participation in combinations:
        for _ in range(count):
            index = len(decisions) + 1
            decisions.append(
                {
                    "case_id": f"k1_target_{index:03d}_{index:012x}",
                    "source_group_id": f"source_group_{index % 17:02d}",
                    "annotation": _annotation(role, team, kit, pitch, participation, seed=index),
                }
            )
    assert len(decisions) == 128
    return decisions


def test_exact_k1_schema_distributions_unknowns_and_warmups_are_immutable() -> None:
    decisions = _completed_k1_decisions()
    receipt = validate_k1_annotations(decisions, expected_case_ids=[row["case_id"] for row in decisions])
    assert receipt["passed"] is True
    assert receipt["accepted_decision_count"] == 128
    assert receipt["warmup_player_count"] == receipt["warmup_unknown_team_count"] == 33
    assert receipt["goalkeepers_by_team"] == {"TEAM_1": 4, "TEAM_2": 4}
    assert receipt["unknown_role_count"] == 4
    assert receipt["unknown_team_count"] == 41
    assert receipt["candidate_state_collected"] is False
    assert receipt["human_certainty_head_authorized"] is False

    inferred_team = deepcopy(decisions)
    warmup = next(row for row in inferred_team if row["annotation"]["kit_state"] == "WARMUP_OR_BIB")
    warmup["annotation"]["team_affiliation"] = "TEAM_1"
    with pytest.raises(ValueError, match="distributions|warmup"):
        validate_k1_annotations(inferred_team)

    fabricated_candidate_state = deepcopy(decisions)
    fabricated_candidate_state[0]["annotation"]["candidate_state"] = "CLEAN_INDEPENDENT_PERSON"
    with pytest.raises(ValueError, match="exactly"):
        validate_k1_annotations(fabricated_candidate_state)


def test_authoritative_binding_reproduces_manifest_hash_and_binds_annotation() -> None:
    case_id = "k1_target_001_000000000001"
    source_hash = "a" * 64
    crop_hash = "b" * 64
    box = {"x1": 10, "y1": 20, "x2": 30, "y2": 80}
    binding = authoritative_case_binding_sha256(
        case_id=case_id,
        source_frame_sha256=source_hash,
        target_crop_sha256=crop_hash,
        bbox_original_pixels=box,
    )
    case = {
        "case_id": case_id,
        "source_group_id": "source-group",
        "source_frame_sha256": source_hash,
        "target_crop_sha256": crop_hash,
        "target_binding_sha256": binding,
        "target": {"bbox_original_pixels": box, "binding_sha256": binding},
    }
    annotation = _annotation(
        "UNKNOWN_ROLE",
        "UNKNOWN_TEAM",
        "UNKNOWN_KIT",
        "ON_PITCH",
        "UNKNOWN_PARTICIPATION",
        seed=9,
    )
    annotation.update(
        {
            "source_frame_sha256": source_hash,
            "target_crop_sha256": crop_hash,
            "target_binding_sha256": binding,
        }
    )
    assert validate_authoritative_case_binding(case, annotation=annotation)["passed"] is True

    tampered = deepcopy(case)
    tampered["target"]["bbox_original_pixels"]["x2"] = 31
    with pytest.raises(ValueError, match="binding hash mismatch"):
        validate_authoritative_case_binding(tampered, annotation=annotation)


def test_conservative_propagation_masks_and_case_normalized_weights() -> None:
    states = {
        "clean": ("CLEAN_INDEPENDENT_PERSON", ("person",), True),
        "partial": ("PARTIAL_PERSON", ("person",), True),
        "duplicate": ("DUPLICATE_OF_PERSON", ("person",), True),
        "merged": ("MERGED_MULTIPLE_PEOPLE", ("person", "other"), False),
        "background": ("BACKGROUND", (), False),
        "multi": ("CLEAN_INDEPENDENT_PERSON", ("person", "other"), False),
    }
    receipts = {
        name: candidate_propagation_eligibility(
            candidate_state=state,
            target_person_id="person",
            contained_person_ids=people,
            aligned_to_target=True,
        )
        for name, (state, people, _) in states.items()
    }
    assert {name for name, receipt in receipts.items() if receipt["eligible"]} == {
        "clean",
        "partial",
        "duplicate",
    }
    assert receipts["duplicate"]["propagation_kind"] == "SAME_PERSON_DUPLICATE"
    assert receipts["merged"]["reason"] == "MERGED_CANDIDATE_REJECTED"
    assert receipts["multi"]["reason"] == "MULTI_PERSON_CONTENT_REJECTED"
    assert all(receipt["candidate_state_changed"] is False for receipt in receipts.values())
    with pytest.raises(ValueError, match="sequence"):
        candidate_propagation_eligibility(
            candidate_state="CLEAN_INDEPENDENT_PERSON",
            target_person_id="person",
            contained_person_ids="person",
            aligned_to_target=True,
        )

    unknown_annotation = _annotation(
        "UNKNOWN_ROLE",
        "UNKNOWN_TEAM",
        "UNKNOWN_KIT",
        "ON_PITCH",
        "UNKNOWN_PARTICIPATION",
        seed=5,
    )
    masks = explicit_supervision_masks(
        prior_candidate_state="DUPLICATE_OF_PERSON",
        annotation=unknown_annotation,
        propagation_eligible=True,
    )
    assert masks["masks"] == {
        "candidate_state": True,
        "role": True,
        "team": True,
        "kit": True,
        "pitch": True,
        "participation": True,
        "footpoint": False,
    }
    assert masks["targets"]["role"] == "UNKNOWN_ROLE"
    assert masks["targets"]["team"] == "UNKNOWN_TEAM"
    assert masks["candidate_state_inferred_from_k1"] is False
    assert masks["certainty_head_present"] is False
    assert "certainty" not in masks["masks"]

    rows = [
        {
            "case_id": "case-a",
            "candidate_uuid": name,
            "propagation_eligible": receipt["eligible"],
        }
        for name, receipt in receipts.items()
    ]
    rows.append({"case_id": "case-b", "candidate_uuid": "only", "propagation_eligible": True})
    weighted = normalize_case_propagation_weights(rows)
    eligible_a = [row for row in weighted["rows"] if row["case_id"] == "case-a" and row["propagation_eligible"]]
    assert [row["propagation_weight"] for row in eligible_a] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert weighted["cases"]["case-a"]["propagation_weight_sum"] == pytest.approx(1.0)
    assert weighted["cases"]["case-b"]["propagation_weight_sum"] == 1.0
    assert weighted["all_case_weight_sums_valid"] is True


def test_exact_fold_inheritance_leakage_checks_and_nested_grouped_calibration() -> None:
    inherited = {f"candidate-{index}": index % 5 for index in range(10)}
    rows = [
        {
            "candidate_uuid": f"candidate-{index}",
            "source_group_id": f"group-{index // 5}-{index % 5}",
            "source_frame_sha256": f"{index + 1:064x}",
            "proposal_lineage": (f"lineage-{index}",),
        }
        for index in range(10)
    ]
    rows[5].pop("proposal_lineage")
    rows[5]["lineage_ids"] = rows[0]["proposal_lineage"]
    edge = {
        "edge_uuid": "positive-0-5",
        "left_candidate_uuid": "candidate-0",
        "right_candidate_uuid": "candidate-5",
        "target_relation": "SAME_PERSON_DUPLICATE",
    }
    exact = inherited_fold_mapping_receipt(inherited, dict(inherited), candidate_rows=rows, positive_edges=[edge])
    assert exact["passed"] is True
    assert exact["checks"]["all_five_folds_present"] is True

    changed = dict(inherited)
    changed["candidate-5"] = 1
    leaking = inherited_fold_mapping_receipt(inherited, changed, candidate_rows=rows, positive_edges=[edge])
    assert leaking["passed"] is False
    assert leaking["changed_candidate_uuids"] == ["candidate-5"]
    assert leaking["positive_edge_cross_fold_edge_uuids"] == ["positive-0-5"]
    assert leaking["cofold_violations"]

    outer = {f"group-{index}": index % 5 for index in range(15)}
    inner = {}
    for outer_fold in range(5):
        training = sorted(group for group, fold in outer.items() if fold != outer_fold)
        inner[outer_fold] = {group: index % 3 for index, group in enumerate(training)}
    nested = nested_grouped_split_receipt(outer, inner)
    assert nested["passed"] is True
    assert nested["outer_labels_used_to_choose_thresholds"] is False
    assert all(not row["outer_holdout_groups_in_calibration"] for row in nested["outer_folds"])

    leaked_inner = deepcopy(inner)
    leaked_inner[0]["group-0"] = 0
    assert nested_grouped_split_receipt(outer, leaked_inner)["passed"] is False


def test_multiclass_metrics_include_macro_f1_calibration_selective_risk_and_unknowns() -> None:
    rows = [
        {
            "case_id": "a",
            "source_group_id": "g1",
            "team_target": "TEAM_1",
            "supervision_masks": {"team": True},
        },
        {
            "case_id": "b",
            "source_group_id": "g1",
            "team_target": "TEAM_2",
            "supervision_masks": {"team": True},
        },
        {
            "case_id": "c",
            "source_group_id": "g2",
            "team_target": "UNKNOWN_TEAM",
            "supervision_masks": {"team": True},
        },
        {
            "case_id": "d",
            "source_group_id": "g2",
            "team_target": "NO_TEAM",
            "supervision_masks": {"team": True},
        },
    ]
    classes = ("TEAM_1", "TEAM_2", "NO_TEAM", "UNKNOWN_TEAM")
    predictions = {"a": "TEAM_1", "b": "TEAM_1", "c": "UNKNOWN_TEAM", "d": "NO_TEAM"}
    probabilities = {
        "a": {"TEAM_1": 0.8, "TEAM_2": 0.1, "NO_TEAM": 0.05, "UNKNOWN_TEAM": 0.05},
        "b": {"TEAM_1": 0.6, "TEAM_2": 0.3, "NO_TEAM": 0.05, "UNKNOWN_TEAM": 0.05},
        "c": {"TEAM_1": 0.05, "TEAM_2": 0.05, "NO_TEAM": 0.1, "UNKNOWN_TEAM": 0.8},
        "d": {"TEAM_1": 0.05, "TEAM_2": 0.05, "NO_TEAM": 0.8, "UNKNOWN_TEAM": 0.1},
    }
    metrics = multiclass_head_metrics(
        rows,
        target_field="team_target",
        predictions=predictions,
        probabilities=probabilities,
        ordered_classes=classes,
        head_name="team",
        availability_mask_field="team",
    )
    assert metrics["denominator"] == 4
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0 + 1 + 1) / 4)
    assert metrics["per_class"]["UNKNOWN_TEAM"]["recall"] == 1.0
    assert metrics["calibration"]["multiclass_brier_score"] is not None
    assert metrics["selective_risk"]["points"][-1]["risk"] == 0.25
    assert metrics["source_group_normalized_accuracy"] == 0.75
    assert metrics["one_row_per_authoritative_case"] is True
