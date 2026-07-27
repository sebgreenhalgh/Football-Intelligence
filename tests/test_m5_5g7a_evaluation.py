from football_intelligence.football_observation_reasoner.evaluation import (
    categorical_head_metrics,
    candidate_confusion_audits,
    candidate_development_screen,
    candidate_outcomes,
    candidate_stratified_metrics,
    exhaustive_candidate_person_ledgers,
    expected_calibration_error,
    k1_pending_receipt,
    pair_relation_metrics,
    required_ablation_variants,
    scene_prior_safety,
    selective_risk_curve,
    zero_harm_receipt,
)


def _rows():
    return [
        {
            "example_uuid": "clean",
            "source_group_id": "g1",
            "candidate_state_target": "CLEAN_INDEPENDENT_PERSON",
            "gold_person_ids": ["p1"],
        },
        {
            "example_uuid": "dup",
            "source_group_id": "g1",
            "candidate_state_target": "DUPLICATE_OF_PERSON",
            "gold_person_ids": ["p1"],
        },
        {
            "example_uuid": "merged",
            "source_group_id": "g2",
            "candidate_state_target": "MERGED_MULTIPLE_PEOPLE",
            "gold_person_ids": ["p2", "p3"],
        },
    ]


def test_candidate_metrics_keep_explicit_denominators() -> None:
    metrics = candidate_outcomes(
        _rows(),
        {
            "clean": "CLEAN_INDEPENDENT_PERSON",
            "dup": "DUPLICATE_OF_PERSON",
            "merged": "AMBIGUOUS_UNRESOLVED",
        },
    )
    assert metrics["independent_person_supply"] == {"numerator": 1, "denominator": 3}
    assert metrics["duplicate_accepted_rate"]["rate"] == 0.0
    assert metrics["merged_as_clean_count"] == 0
    assert metrics["denominators"]["source_groups"] == 2


def test_full_evaluator_universe_retains_zero_proposal_people_in_denominators() -> None:
    metrics = candidate_outcomes(
        _rows(),
        {
            "clean": "CLEAN_INDEPENDENT_PERSON",
            "dup": "DUPLICATE_OF_PERSON",
            "merged": "AMBIGUOUS_UNRESOLVED",
        },
        evaluator_person_ids=("p1", "p2", "p3", "p4"),
    )
    assert metrics["evaluator_universe_mode"] == "EXPLICIT_FULL_EVALUATOR_UNIVERSE"
    assert metrics["denominators"]["all_evaluator_people"] == 4
    assert metrics["denominators"]["zero_linked_proposal_evaluator_people"] == 1
    assert metrics["exactly_one_observation"] == {"numerator": 1, "denominator": 4}


def test_calibration_and_selective_risk_are_deterministic() -> None:
    calibration = expected_calibration_error([0.9, 0.7, 0.2, 0.1], [1, 1, 0, 1], bin_count=5)
    curve = selective_risk_curve([0.9, 0.7, 0.2, 0.1], [1, 1, 1, 0], coverages=(0.5, 1.0))
    assert calibration["denominator"] == 4
    assert calibration["expected_calibration_error"] is not None
    assert curve["points"][0]["risk"] == 0.0
    assert curve["points"][1]["risk"] == 0.25


def test_scene_prior_is_warning_only_and_never_forces_cardinality() -> None:
    states = {"a": "CLEAN_INDEPENDENT_PERSON", "b": "AMBIGUOUS_UNRESOLVED"}
    safe = scene_prior_safety(states, dict(states))
    unsafe = scene_prior_safety(states, {"b": "CLEAN_INDEPENDENT_PERSON"})
    assert safe["passed"] is True
    assert safe["exact_22_forcing_performed"] is False
    assert safe["exactly_one_goalkeeper_per_team_forcing_performed"] is False
    assert unsafe["passed"] is False


def test_development_screen_cannot_be_weakened() -> None:
    predictions = {
        "clean": "CLEAN_INDEPENDENT_PERSON",
        "dup": "DUPLICATE_OF_PERSON",
        "merged": "AMBIGUOUS_UNRESOLVED",
    }
    metrics = candidate_outcomes(_rows(), predictions)
    screen = candidate_development_screen(
        metrics,
        metrics,
        selective_risk_improved=True,
        deterministic=True,
        provenance_complete=True,
    )
    assert screen["passed"] is True
    assert screen["thresholds_weakened"] is False
    assert len(required_ablation_variants()) == 7


def _evaluation_node(
    identifier: str,
    target: str | None,
    *,
    people: tuple[str, ...] = (),
    role: str | None = None,
    pitch: str | None = None,
    source_group: str = "g1",
    height: float = 60.0,
    runtime_leak: bool = False,
) -> dict:
    source_hash = "a" * 64
    return {
        "example_uuid": identifier,
        "candidate_uuid": f"candidate-{identifier}",
        "source_group_id": source_group,
        "source_frame_sha256": source_hash,
        "provenance_hash": "b" * 64,
        "source_artifact_hashes": {"source_frame": source_hash},
        "candidate_state_target": target,
        "role_target": role,
        "pitch_state_target": pitch,
        "gold_person_ids": list(people),
        "universe": "STATIC" if source_group == "g1" else "DENSE",
        "case_family": "SMALL_FAR_SIDE" if height <= 40 else "CORE",
        "visible_box": {"x1": 10.0, "y1": 100.0, "x2": 30.0, "y2": 100.0 + height},
        "source_coordinates": {"image_width": 2000.0, "image_height": 1000.0},
        "label_availability_mask": {
            "candidate_state": target is not None,
            "role": role is not None,
            "pitch": pitch is not None,
            "team": False,
            "kit": False,
            "participation": False,
        },
        "shape_features": {"role_target": "LEAK"} if runtime_leak else {"aspect_ratio": 0.3},
    }


def test_exhaustive_ledgers_named_confusions_and_required_strata() -> None:
    nodes = [
        _evaluation_node(
            "small-clean",
            "CLEAN_INDEPENDENT_PERSON",
            people=("p1",),
            role="GOALKEEPER",
            pitch="ON_PITCH",
            height=20.0,
        ),
        _evaluation_node("duplicate", "DUPLICATE_OF_PERSON", people=("p1",)),
        _evaluation_node(
            "merged",
            "MERGED_MULTIPLE_PEOPLE",
            people=("p2", "p3"),
            source_group="g2",
        ),
        _evaluation_node("partial", "PARTIAL_PERSON", people=("p2",), source_group="g2"),
        _evaluation_node(
            "background",
            "BACKGROUND",
            role="REFEREE",
            source_group="g2",
        ),
        _evaluation_node("runtime", None, source_group="g2", runtime_leak=True),
    ]
    predictions = {
        "small-clean": "BACKGROUND",
        "duplicate": "CLEAN_INDEPENDENT_PERSON",
        "merged": "CLEAN_INDEPENDENT_PERSON",
        "partial": "BACKGROUND",
        "background": "PARTIAL_PERSON",
        "runtime": "AMBIGUOUS_UNRESOLVED",
    }
    ledgers = exhaustive_candidate_person_ledgers(
        nodes,
        predictions,
        evaluator_person_ids=("p1", "p2", "p3", "p4"),
        predicted_roles={"small-clean": "REFEREE", "background": "GOALKEEPER"},
        predicted_pitch_states={"small-clean": "OFF_PITCH"},
    )
    assert ledgers["denominators"] == {
        "all_candidate_nodes": 6,
        "candidate_state_labelled_nodes": 5,
        "all_evaluator_people": 4,
        "zero_proposal_evaluator_people": 1,
    }
    p4 = next(row for row in ledgers["person_rows"] if row["evaluator_person_id"] == "p4")
    assert p4["zero_proposal"] is True
    runtime = next(row for row in ledgers["candidate_rows"] if row["example_uuid"] == "runtime")
    assert runtime["outcome_flags"]["provenance_or_leakage_defect"] is True
    assert runtime["provenance"]["runtime_target_leakage_paths"] == ["shape_features.role_target"]

    audits = candidate_confusion_audits(
        ledgers["candidate_rows"],
        person_ledger_rows=ledgers["person_rows"],
    )
    assert audits["audits"]["small_far_side_miss"]["numerator"] == 1
    assert audits["audits"]["duplicate_accepted"]["rate"] == 1.0
    assert audits["audits"]["merged_accepted"]["numerator"] == 1
    assert audits["audits"]["partial_background_confusion"]["numerator"] == 2
    assert audits["audits"]["goalkeeper_referee_confusion"]["numerator"] == 2
    assert audits["audits"]["pitch_state_mismatch"]["numerator"] == 1
    assert audits["audits"]["zero_proposal_evaluator_person"]["evaluator_person_ids"] == ["p4"]

    strata = candidate_stratified_metrics(ledgers["candidate_rows"])
    assert set(strata["dimensions"]) == {
        "candidate_class",
        "universe",
        "case_family",
        "small_far_proxy",
        "pitch_state",
        "role",
    }
    assert strata["dimensions"]["small_far_proxy"]["SMALL_FAR_PROXY"]["population_count"] == 1
    assert strata["dimensions"]["candidate_class"]["UNLABELLED_CANDIDATE_STATE"]["not_evaluable_count"] == 1


def test_pair_relation_metrics_are_per_relation_and_source_normalized() -> None:
    edges = [
        {
            "edge_uuid": "duplicate",
            "source_group_id": "g1",
            "target_relation": "SAME_PERSON_DUPLICATE",
            "target_available": True,
        },
        {
            "edge_uuid": "distinct",
            "source_group_id": "g1",
            "target_relation": "DISTINCT_PEOPLE",
            "target_available": True,
        },
        {
            "edge_uuid": "merged",
            "source_group_id": "g2",
            "target_relation": "MERGED_CONTAINS_BOTH",
            "target_available": True,
        },
        {
            "edge_uuid": "unlabelled",
            "source_group_id": "g2",
            "target_relation": None,
            "target_available": False,
        },
    ]
    metrics = pair_relation_metrics(
        edges,
        {
            "duplicate": "SAME_PERSON_DUPLICATE",
            "distinct": "SAME_PERSON_DUPLICATE",
            "merged": "MERGED_CONTAINS_BOTH",
        },
    )
    assert metrics["labelled_edge_denominator"] == 3
    assert metrics["per_relation"]["DISTINCT_PEOPLE"]["recall"] == 0.0
    assert metrics["confusion_matrix"]["DISTINCT_PEOPLE"] == {"SAME_PERSON_DUPLICATE": 1}
    assert metrics["source_group_normalized_accuracy"] == 0.75
    assert len(metrics["ledger"]) == 3


def test_k1_pending_and_zero_harm_receipts_are_explicit() -> None:
    nodes = [
        _evaluation_node(
            "goalkeeper",
            "CLEAN_INDEPENDENT_PERSON",
            role="GOALKEEPER",
        ),
        _evaluation_node("other", "CLEAN_INDEPENDENT_PERSON"),
    ]
    predictions_by_head = {
        "team": {"goalkeeper": "UNKNOWN_TEAM", "other": "UNKNOWN_TEAM"},
        "kit": {"goalkeeper": "UNKNOWN_KIT", "other": "UNKNOWN_KIT"},
        "participation": {
            "goalkeeper": "UNKNOWN_PARTICIPATION",
            "other": "UNKNOWN_PARTICIPATION",
        },
    }
    k1 = k1_pending_receipt(nodes, predictions_by_head)
    assert k1["passed"] is True
    assert k1["both_team_goalkeeper_classes_screen"] == "NOT_EVALUABLE_K1_PENDING"

    states = {
        "goalkeeper": "CLEAN_INDEPENDENT_PERSON",
        "other": "CLEAN_INDEPENDENT_PERSON",
    }
    safe = zero_harm_receipt(nodes, states, dict(states))
    harmful = zero_harm_receipt(
        nodes,
        states,
        {"other": "CLEAN_INDEPENDENT_PERSON"},
        hard_deleted_node_ids=("goalkeeper",),
    )
    assert safe["passed"] is True
    assert safe["hard_goalkeeper_prior_deletion_count"] == 0
    assert harmful["passed"] is False
    assert harmful["hard_goalkeeper_prior_deletion_example_uuids"] == ["goalkeeper"]


def test_categorical_head_metrics_cover_authorized_subset_calibration_and_source_groups() -> None:
    rows = [
        {
            "example_uuid": "a",
            "source_group_id": "g1",
            "role_target": "GOALKEEPER",
            "label_availability_mask": {"role": True},
        },
        {
            "example_uuid": "b",
            "source_group_id": "g1",
            "role_target": "REFEREE",
            "label_availability_mask": {"role": True},
        },
        {
            "example_uuid": "c",
            "source_group_id": "g2",
            "role_target": "GOALKEEPER",
            "label_availability_mask": {"role": True},
        },
        {
            "example_uuid": "d",
            "source_group_id": "g2",
            "role_target": "REFEREE",
            "label_availability_mask": {"role": True},
        },
    ]
    classes = ("GOALKEEPER", "REFEREE")
    predictions = {"a": "GOALKEEPER", "b": "GOALKEEPER", "c": "REFEREE", "d": "REFEREE"}
    matrix = ([0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8])
    metrics = categorical_head_metrics(rows, "role_target", predictions, matrix, classes)
    mapping = {
        row["example_uuid"]: dict(zip(classes, probabilities, strict=True))
        for row, probabilities in zip(rows, matrix, strict=True)
    }
    mapped_metrics = categorical_head_metrics(rows, "role_target", predictions, mapping, classes)
    assert metrics["metrics_hash"] == mapped_metrics["metrics_hash"]
    assert metrics["development_scope"] == "SINGLE_MATCH_GROUPED_DEVELOPMENT_ONLY"
    assert metrics["denominator"] == 4
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_recall"] == 0.5
    assert metrics["confusion_matrix"]["GOALKEEPER"] == {"GOALKEEPER": 1, "REFEREE": 1}
    assert metrics["per_class"]["REFEREE"]["support"] == 2
    assert metrics["per_class"]["REFEREE"]["recall"] == 0.5
    assert metrics["source_group_normalized_accuracy"] == 0.5
    assert metrics["top_class_confidence_calibration"]["denominator"] == 4
    assert metrics["selective_risk"]["points"][-1]["risk"] == 0.5


def test_categorical_head_metrics_leave_k1_dependent_empty_head_not_evaluable() -> None:
    rows = [
        {
            "example_uuid": "a",
            "source_group_id": "g1",
            "team_target": None,
            "label_availability_mask": {"team": False},
        },
        {
            "example_uuid": "b",
            "source_group_id": "g2",
            "team_target": None,
            "label_availability_mask": {"team": False},
        },
    ]
    metrics = categorical_head_metrics(
        rows,
        "team_target",
        {},
        {},
        ("TEAM_1", "TEAM_2", "NO_TEAM", "UNKNOWN_TEAM"),
    )
    assert metrics["evaluation_status"] == "NOT_EVALUABLE_NO_AUTHORIZED_LABELS"
    assert metrics["denominator"] == 0
    assert metrics["accuracy"] is None
    assert metrics["macro_recall"] is None
    assert metrics["top_class_confidence_calibration"]["expected_calibration_error"] is None
    assert metrics["k1_pending_compatible"] is True


def test_clean_control_preservation_uses_only_explicit_clean_control_case_family() -> None:
    rows = [
        {
            "example_uuid": "control",
            "source_group_id": "g1",
            "case_family": "clean_control",
            "candidate_state_target": "CLEAN_INDEPENDENT_PERSON",
            "gold_person_ids": ["p1"],
        },
        {
            "example_uuid": "ordinary",
            "source_group_id": "g2",
            "case_family": "CORE_STATIC",
            "candidate_state_target": "CLEAN_INDEPENDENT_PERSON",
            "gold_person_ids": ["p2"],
        },
        {
            "example_uuid": "non-clean-control",
            "source_group_id": "g3",
            "case_family": "clean_control",
            "candidate_state_target": "MERGED_MULTIPLE_PEOPLE",
            "gold_person_ids": ["p3", "p4"],
        },
    ]
    baseline = candidate_outcomes(
        rows,
        {
            "control": "CLEAN_INDEPENDENT_PERSON",
            "ordinary": "BACKGROUND",
            "non-clean-control": "MERGED_MULTIPLE_PEOPLE",
        },
    )
    candidate = candidate_outcomes(
        rows,
        {
            "control": "BACKGROUND",
            "ordinary": "CLEAN_INDEPENDENT_PERSON",
            "non-clean-control": "MERGED_MULTIPLE_PEOPLE",
        },
    )
    assert baseline["clean_control_preservation"] == {
        "numerator": 1,
        "preserved": 1,
        "errors": 0,
        "denominator": 1,
        "rate": 1.0,
        "case_family": "clean_control",
        "error_example_uuids": [],
    }
    assert candidate["clean_control_preservation"]["errors"] == 1
    assert candidate["distinct_person_suppression"] == baseline["distinct_person_suppression"] == 1
    screen = candidate_development_screen(
        candidate,
        baseline,
        selective_risk_improved=True,
        deterministic=True,
        provenance_complete=True,
    )
    assert screen["checks"]["zero_clean_control_regression"] is False
    assert screen["checks"]["distinct_person_suppression_no_worse_than_r0"] is True
