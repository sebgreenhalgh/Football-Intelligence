import json
from pathlib import Path

from football_intelligence.perspective_nested_policy import POLICIES
from football_intelligence.perspective_scale_surface import MODELS, predict

STAGE = (
    Path(r"C:\Users\sebgr\Documents\football-intelligence\experiments\football_observation_reasoner\part 7")
    / "G7D_C3B2_PERSPECTIVE_NORMALIZED_CANDIDATE_SCALE_SANDBOX_v1"
)


def test_frozen_closure_and_truth():
    x = json.loads((STAGE / "00_INPUT_CLOSURE/input_closure.json").read_text())
    assert (x["frames"], x["post_pitch_gate_candidates"], x["nested_pairs"]) == (144, 6509, 1630)
    assert x["pair_truth"] == {"safe": 34, "must_protect": 11, "ambiguous": 3}


def test_surface_models_are_leave_one_frame_out_and_cpu_geometry_only():
    refs = [
        {"frame_id": "a", "x_norm": 0.2, "y_norm": 0.2, "height": 30},
        {"frame_id": "b", "x_norm": 0.3, "y_norm": 0.3, "height": 32},
    ] * 12
    target = {"frame_id": "a", "x_norm": 0.25, "y_norm": 0.25, "height": 31}
    for model in MODELS:
        assert "prediction_available" in predict(model, target, refs)
    c = json.loads((STAGE / "02_REFERENCE_POOL/reference_pool_contract.json").read_text())
    assert c["leave_one_frame_out"] and not c["human_labels_used"]


def test_policy_contract_and_no_safe_selection():
    assert len(POLICIES) == 7 and POLICIES[-1] == "S5_HUMAN_ORACLE_NOT_IMPLEMENTABLE"
    d = json.loads((STAGE / "07_RULE_SELECTION/decision.json").read_text())
    assert d["decision"] == "NO_SAFE_PERSPECTIVE_NORMALIZED_RULE" and not d["runtime_activation"]
    assert not (STAGE / "07_RULE_SELECTION/selected_scale_policy.json").exists()


def test_outputs_visuals_and_handoff():
    lines = (STAGE / "03_EXPECTED_HEIGHT_SURFACES/candidate_scale_predictions.jsonl").read_text().splitlines()
    assert len(lines) == 6509
    assert len(list((STAGE / "08_VISUAL_QA").glob("*.png"))) == 3
    hand = STAGE / "10_REVIEW_PACK/CHATGPT_HANDOFF"
    assert len(list(hand.iterdir())) == 11
    assert json.loads((hand / "11_MANIFEST.json").read_text())["self_hash_omitted"]
