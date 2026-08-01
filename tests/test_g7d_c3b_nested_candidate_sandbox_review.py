import json
from pathlib import Path
from football_intelligence.nested_candidate_sandbox import POLICY_IDS, pair_geometry, policy_decisions

STAGE = (
    Path(r"C:\Users\sebgr\Documents\football-intelligence\experiments\football_observation_reasoner\part 7")
    / "G7D_C3B_NESTED_CANDIDATE_SUPPRESSION_SANDBOX_REVIEW_v1"
)


def test_input_and_geometry_closure():
    x = json.loads((STAGE / "01_INPUT_AND_PAIR_CLOSURE/candidate_input_manifest.json").read_text())
    assert (x["frames"], x["pre_gate_candidates"], x["retained_candidates"], x["pitch_gate_suppressions"]) == (
        144,
        9067,
        6509,
        2558,
    )
    g = json.loads((STAGE / "02_NESTED_PAIR_GEOMETRY/geometry_manifest.json").read_text())
    assert g["coordinate_space"] == "SOURCE_IMAGE" and g["pair_count"] > 0


def test_pair_math_and_policy_grid():
    a = {"source_box_xyxy": [2, 6, 4, 10], "approximate_footpoint_xy": [3, 10]}
    b = {"source_box_xyxy": [0, 0, 10, 10], "approximate_footpoint_xy": [5, 10]}
    g = pair_geometry(a, b, [a, b])
    assert g["inner_containment"] == 1 and g["inner_outer_area_ratio"] == 0.08
    p = policy_decisions(g)
    assert tuple(p) == POLICY_IDS and p["N5_HUMAN_ORACLE_NOT_IMPLEMENTABLE"] == "HUMAN_REVIEW_REQUIRED"


def test_human_closure_selection_and_blind_payload():
    h = json.loads((STAGE / "03_EXISTING_HUMAN_SAFETY/existing_safety_manifest.json").read_text())
    assert (h["candidate_labels"], h["scene_reviews"], h["missed_person_marks"]) == (252, 36, 25)
    cases = json.loads((STAGE / "06_NESTED_REVIEW_PACKAGE/cases.json").read_text())
    assert len(cases["cases"]) == 48 and cases["blind_first"]
    assert all("policies" not in c and "selection_quota" not in c for c in cases["cases"])
    counts = {m: sum(c["match_id"] == m for c in cases["cases"]) for m in {c["match_id"] for c in cases["cases"]}}
    assert set(counts.values()) == {8}


def test_live_protocol_visual_cap_and_handoff():
    live = json.loads((STAGE / "08_TESTS_AND_LOGS/live_edge_acceptance.json").read_text())
    assert live["temporary_completion"] and live["human_root_synthetic_events"] == 0
    assert len(list((STAGE / "07_VISUAL_QA").glob("*.png"))) == 2
    hand = STAGE / "09_REVIEW_PACK/CHATGPT_HANDOFF"
    assert len(list(hand.iterdir())) == 10
    assert json.loads((hand / "10_MANIFEST.json").read_text())["self_hash_omitted"]
