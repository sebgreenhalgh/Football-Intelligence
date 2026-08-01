import json
from pathlib import Path

ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence\experiments\football_observation_reasoner\part 7")
STAGE = ROOT / "G7D_C3B1_NESTED_REVIEW_FINALIZATION_AND_SAFE_RULE_SELECTION_v1"
C3B = ROOT / "G7D_C3B_NESTED_CANDIDATE_SUPPRESSION_SANDBOX_REVIEW_v1"


def test_continuation_and_event_chain():
    x = json.loads((STAGE / "00_INPUT_CLOSURE/input_closure.json").read_text())
    assert (x["frames"], x["post_pitch_gate_candidates"], x["overlapping_pairs"]) == (144, 6509, 1630)
    e = json.loads((STAGE / "01_EVENT_AND_RESTORATION_CLOSURE/human_event_chain_validation.json").read_text())
    assert (e["latest_events"], e["acknowledgements"], e["current_completion_receipts"]) == (48, 48, 1)
    assert (
        e["visible_last_event_id"] == "6b7a55ca-0da7-4af9-b36f-7376ad901dd1"
        and e["completion_receipt_id"] == "completion-37401efba568571a0f627ee5"
    )


def test_restoration_and_normalized_truth():
    r = json.loads((STAGE / "01_EVENT_AND_RESTORATION_CLOSURE/restoration_acceptance.json").read_text())
    assert r["actual_human_root_completed"] and not r["actual_human_files_mutated"] and r["refresh_created_events"] == 0
    lines = (STAGE / "02_NORMALIZED_PAIR_TRUTH/pair_human_labels.jsonl").read_text().splitlines()
    assert len(lines) == 48 and all(len(json.loads(x)["answers"]) == 6 for x in lines)


def test_frozen_policy_selection():
    p = json.loads((STAGE / "03_POLICY_SAFETY_EVALUATION/policy_evaluation_manifest.json").read_text())
    assert not p["thresholds_changed"] and not p["candidate_specific_human_decisions"]
    d = json.loads((STAGE / "05_RULE_SELECTION/decision.json").read_text())
    assert d["decision"] == "NO_SAFE_IMPLEMENTABLE_NESTED_RULE" and not d["nested_rule_activated"]
    assert not (STAGE / "05_RULE_SELECTION/selected_nested_policy.json").exists()


def test_visuals_handoff_and_reviewer_endpoint():
    assert len(list((STAGE / "06_VISUAL_QA").glob("*.png"))) == 2
    hand = STAGE / "08_REVIEW_PACK/CHATGPT_HANDOFF"
    assert len(list(hand.iterdir())) == 10
    server = (C3B / "06_NESTED_REVIEW_PACKAGE/review_server.py").read_text()
    client = (C3B / "06_NESTED_REVIEW_PACKAGE/review.js").read_text()
    assert (
        "/api/review-state" in server
        and "/api/review-state" in client
        and "G7D_C3B1_COMPLETION_RESTORATION_V1" in server
    )
