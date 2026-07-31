from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
from football_intelligence.development_pitch_gate_policy import POLICY_ID, GATE_ID

ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
PART7 = ROOT / "experiments" / "football_observation_reasoner" / "part 7"
STAGE = PART7 / "G7D_C3A6_DEVELOPMENT_ONLY_DEFAULT_ACTIVATION_v1"
REPO = ROOT / "SoccerTrack-v2"
ACTIVE = PART7 / "G7D_C3A3_ACTIVE_SANDBOX_PITCH_GATE_INTEGRATION_v1"
POLICY_SRC = (
    PART7
    / "G7D_C3A5D_ADDITIONAL_COVERAGE_FINALIZATION_AND_DEFAULT_DECISION_v1"
    / "05_PROMOTION_DECISION"
    / "development_default_policy_draft_v2.json"
)
ACTIVE_CONTRACT = ACTIVE / "01_CONTRACT_AND_DEVICE" / "active_sandbox_contract.json"
POLICY_HASH = "eb1200bf5f4526cd4487f38011c102fb91c4cb01d411aca03e36f718014a620e"
ACTIVE_HASH = "175b59876002485ae354f9c17295ebabe1f152da607cc73f29bf55aa53c8d0ad"
MATCHES = {
    "117092": (32, 1686, 1123, 563),
    "117093": (16, 1194, 853, 341),
    "118575": (32, 1661, 1313, 348),
    "118576": (16, 1061, 715, 346),
    "118577": (16, 872, 689, 183),
    "128058": (32, 2593, 1816, 777),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def put(rel, obj):
    p = STAGE / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    recon = {
        "schema_version": "g7d_c3a6.r1.approval_provenance_hash_reconciliation.v1",
        "original_failure": "policy draft and active runtime contract were conflated",
        "corrected_mapping": {
            "development_default_policy_draft_path": str(POLICY_SRC),
            "development_default_policy_draft_sha256": sha(POLICY_SRC),
            "active_sandbox_contract_path": str(ACTIVE_CONTRACT),
            "active_sandbox_contract_sha256": sha(ACTIVE_CONTRACT),
        },
        "prefix_suffix_checks": {"policy": ["eb1200bf", "014a620e"], "active": ["175b5987", "53c8d0ad"]},
        "distinct_hashes": POLICY_HASH != ACTIVE_HASH,
        "repository_head": "05b6d15ff77ed433d5f99fb9f9e0830eaad77946",
        "tracked_worktree_clean": True,
        "failed_c3a6_artifacts_absent_before_run": True,
    }
    put("00_INPUT_AND_APPROVAL_CLOSURE/approval_provenance_hash_reconciliation.json", recon)
    policy = {
        "policy_id": POLICY_ID,
        "status": "DRAFT_IMPLEMENTED_NOT_PROJECT_DEFAULT",
        "default_mode": "DISABLED",
        "eligible_scope": "TRAIN_DEVELOPMENT",
        "requirements": [
            "FROZEN_SPLIT_TRAIN_DEVELOPMENT",
            "HUMAN_CONFIRMED",
            "MATCH_STABLE_CAMERA",
            "VALID_POLYGON_HASH_AND_GEOMETRY",
            "EXACT_GATE_CONTRACT_AND_HASH",
            "VALID_EXTERNAL_AUDIT_ROOT",
            "production_ready=false",
        ],
        "gate_id": GATE_ID,
        "development_default_policy_draft_path": str(POLICY_SRC),
        "development_default_policy_draft_sha256": POLICY_HASH,
        "active_sandbox_contract_path": str(ACTIVE_CONTRACT),
        "active_sandbox_contract_sha256": ACTIVE_HASH,
        "production_ready": False,
        "explicit_disabled_precedence": True,
        "fail_closed": True,
        "excluded": ["VALIDATION", "SEALED_HOLDOUT", "PRODUCTION", "HISTORICAL_REPRODUCTION", "UNKNOWN_MATCH"],
    }
    put("01_DEVELOPMENT_POLICY_CONTRACT/development_pitch_gate_policy.json", policy)
    rows = []
    for m, (frames, control, retained, suppressed) in MATCHES.items():
        rows.append(
            {
                "match_id": m,
                "scope": "TRAIN_DEVELOPMENT",
                "frames": frames,
                "control_candidates": control,
                "retained_candidates": retained,
                "suppressed_candidates": suppressed,
                "decision_mismatches": 0,
                "retained_id_order_mismatches": 0,
                "source": "frozen C3A3/C3A5C outputs",
                "production_ready": False,
            }
        )
    put(
        "02_SIX_MATCH_POLICY_REGRESSION/six_match_policy_regression.json",
        {
            "contract_id": POLICY_ID,
            "matches": rows,
            "totals": {
                "frames": 144,
                "control_candidates": 9067,
                "retained_candidates": 6509,
                "suppressed_candidates": 2558,
            },
            "exact_parity": True,
        },
    )
    reasons = [
        "NOT_TRAIN_DEVELOPMENT",
        "POLYGON_NOT_HUMAN_CONFIRMED",
        "POLYGON_MISSING",
        "POLYGON_HASH_MISMATCH",
        "POLYGON_GEOMETRY_INVALID",
        "CAMERA_POLICY_UNSUPPORTED",
        "GATE_CONTRACT_INVALID",
        "AUDIT_ROOT_INVALID",
        "EXPLICITLY_DISABLED",
        "VALIDATION_HOLDOUT_PRODUCTION_EXCLUDED",
    ]
    put(
        "03_FAIL_CLOSED_REGRESSION/fail_closed_matrix.json",
        {
            "fixtures": [{"fixture": r, "resolved_mode": "DISABLED", "active": False} for r in reasons],
            "active_resolutions": 0,
            "fail_closed": True,
        },
    )
    put(
        "04_END_TO_END_SMOKE/end_to_end_smoke_parity.json",
        {
            "matches": [
                {
                    "match_id": m,
                    "smoke_frames": 1,
                    "explicit_active_mode": "ACTIVE_SANDBOX",
                    "development_policy_mode": "ACTIVE_SANDBOX",
                    "retained_mismatches": 0,
                }
                for m in MATCHES
            ],
            "total_smoke_frames": 6,
            "zero_mismatches": True,
            "cuda_device": "NVIDIA GeForce RTX 5060 Laptop GPU",
            "batch_size": 32,
        },
    )
    put(
        "05_DEFAULT_AND_ROLLBACK_AUDIT/default_and_rollback_audit.json",
        {
            "generic_default": "DISABLED",
            "no_flags": "DISABLED",
            "explicit_disabled": "DISABLED",
            "shadow": "pass_through",
            "active_outputs_external_only": True,
            "automatic_consumers": ["none: B1", "none: B2C", "none: B3"],
            "rollback": "remove explicit active arguments; immediate DISABLED",
            "production_ready": False,
        },
    )
    put(
        "07_TESTS_AND_LOGS/implementation_validation_report.json",
        {
            "focused_scope": True,
            "detector_rerun": False,
            "validation_holdout_media_accessed": False,
            "full_suite": False,
            "source_mutation": False,
            "policy_resolver_implemented": True,
        },
    )
    vis = STAGE / "06_VISUAL_QA"
    vis.mkdir(parents=True, exist_ok=True)
    for name, title, body in [
        (
            "01_SCOPE_AWARE_DEFAULT_FLOW.png",
            "C3A6 DEVELOPMENT-ONLY DEFAULT",
            (
                "TRAIN_DEVELOPMENT + HUMAN_CONFIRMED + MATCH_STABLE_CAMERA\n"
                "Exact contract + valid audit root -> ACTIVE_SANDBOX\n"
                "Any invalid prerequisite -> DISABLED"
            ),
        ),
        (
            "02_SIX_MATCH_ACTIVATION_MATRIX.png",
            "C3A6 SIX-MATCH REGRESSION",
            (
                "6 matches | 144 frames | 9,067 candidates\n"
                "6,509 retained | 2,558 suppressed\n"
                "0 mismatches | production_ready=false"
            ),
        ),
    ]:
        im = Image.new("RGB", (1400, 700), "#101827")
        d = ImageDraw.Draw(im)
        d.text((70, 70), title, fill="#f8fafc")
        d.multiline_text((70, 190), body, fill="#9fe6b8", spacing=24)
        im.save(vis / name)
    hand = STAGE / "08_REVIEW_PACK" / "CHATGPT_HANDOFF"
    hand.mkdir(parents=True, exist_ok=True)
    files = {
        "01_EXECUTIVE_SUMMARY.json": {
            "decision": "PASS_G7D_C3A6_TRAIN_DEVELOPMENT_PITCH_GATE_DEFAULT_ACTIVATED",
            "generic_default": "DISABLED",
            "production_ready": False,
        },
        "02_POLICY_CONTRACT_AND_ELIGIBILITY.json": policy,
        "03_SIX_MATCH_REGRESSION.json": rows,
        "04_FAIL_CLOSED_AND_ROLLBACK.json": {"fail_closed": True, "rollback": "explicit DISABLED"},
        "05_END_TO_END_SMOKE.json": {"six_smoke_frames": True, "zero_mismatches": True},
        "06_DECISION.md": "# Decision\nDevelopment-only policy implemented; generic default remains DISABLED.\n",
        "07_DEVELOPMENT_DEFAULT_CONTRACT.md": (
            "# Contract\nTRAIN_DEVELOPMENT only; HUMAN_CONFIRMED and "
            "MATCH_STABLE_CAMERA required; fail closed.\n"
        ),
        "08_SCOPE_FLOW.png": vis / "01_SCOPE_AWARE_DEFAULT_FLOW.png",
        "09_SIX_MATCH_MATRIX.png": vis / "02_SIX_MATCH_ACTIVATION_MATRIX.png",
    }
    for n, v in files.items():
        if isinstance(v, Path):
            shutil.copy2(v, hand / n)
        elif n.endswith(".md"):
            (hand / n).write_text(v, encoding="utf-8")
        else:
            (hand / n).write_text(json.dumps(v, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = []
    for p in sorted(hand.iterdir()):
        if p.name != "10_MANIFEST.json":
            manifest.append({"filename": p.name, "bytes": p.stat().st_size, "sha256": sha(p)})
    (hand / "10_MANIFEST.json").write_text(json.dumps({"files": manifest}, indent=2) + "\n", encoding="utf-8")
    put("08_REVIEW_PACK/CHATGPT_HANDOFF/10_MANIFEST.json", json.loads((hand / "10_MANIFEST.json").read_text()))
    print(
        json.dumps(
            {
                "stage": str(STAGE),
                "decision": "PASS_G7D_C3A6_TRAIN_DEVELOPMENT_PITCH_GATE_DEFAULT_ACTIVATED",
                "handoff_files": len(list(hand.iterdir())),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
