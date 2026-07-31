from pathlib import Path
import json

STAGE = Path(
    r"C:\Users\sebgr\Documents\football-intelligence\experiments\football_observation_reasoner\part 7"
) / "G7D_C3A6_DEVELOPMENT_ONLY_DEFAULT_ACTIVATION_v1"


def test_provenance_roles_and_hashes():
    p = json.loads((STAGE / "00_INPUT_AND_APPROVAL_CLOSURE/approval_provenance_hash_reconciliation.json").read_text())
    c = p["corrected_mapping"]
    assert c["development_default_policy_draft_sha256"].startswith("eb1200bf")
    assert c["development_default_policy_draft_sha256"].endswith("014a620e")
    assert c["active_sandbox_contract_sha256"].startswith("175b5987")
    assert c["active_sandbox_contract_sha256"].endswith("53c8d0ad")
    assert p["distinct_hashes"]


def test_regression_and_fail_closed():
    r = json.loads((STAGE / "02_SIX_MATCH_POLICY_REGRESSION/six_match_policy_regression.json").read_text())
    assert r["totals"] == {
        "frames": 144,
        "control_candidates": 9067,
        "retained_candidates": 6509,
        "suppressed_candidates": 2558,
    }
    assert r["exact_parity"]
    f = json.loads((STAGE / "03_FAIL_CLOSED_REGRESSION/fail_closed_matrix.json").read_text())
    assert f["active_resolutions"] == 0
    assert all(x["resolved_mode"] == "DISABLED" for x in f["fixtures"])


def test_outputs_and_handoff():
    assert len(list((STAGE / "06_VISUAL_QA").glob("*.png"))) == 2
    assert len(list((STAGE / "08_REVIEW_PACK/CHATGPT_HANDOFF").iterdir())) == 10
    assert json.loads((STAGE / "04_END_TO_END_SMOKE/end_to_end_smoke_parity.json").read_text())["zero_mismatches"]
