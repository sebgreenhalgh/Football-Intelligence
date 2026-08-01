from __future__ import annotations
import hashlib
import json
import shutil

# Compact handoff prose is intentionally kept as complete evidence strings.
# ruff: noqa: E501
from pathlib import Path

ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
STAGE = ROOT / "experiments/football_observation_reasoner/part 7/G7D_C3B_NESTED_CANDIDATE_SUPPRESSION_SANDBOX_REVIEW_v1"
HAND = STAGE / "09_REVIEW_PACK/CHATGPT_HANDOFF"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(name, obj):
    (HAND / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    HAND.mkdir(parents=True, exist_ok=True)
    geometry = json.loads((STAGE / "02_NESTED_PAIR_GEOMETRY/geometry_manifest.json").read_text())
    thresholds = json.loads((STAGE / "02_NESTED_PAIR_GEOMETRY/containment_threshold_summary.json").read_text())
    policies = json.loads((STAGE / "04_FULL_UNIVERSE_SANDBOX/full_universe_policy_comparison.json").read_text())
    safety = json.loads((STAGE / "03_EXISTING_HUMAN_SAFETY/existing_human_policy_comparison.json").read_text())
    selection = json.loads((STAGE / "05_REVIEW_SELECTION/selection_quota_report.json").read_text())
    edge = json.loads((STAGE / "08_TESTS_AND_LOGS/live_edge_acceptance.json").read_text())
    dump(
        "01_EXECUTIVE_SUMMARY.json",
        {
            "decision": "PASS_G7D_C3B_NESTED_CANDIDATE_REVIEW_READY_FOR_HUMAN_REVIEW",
            "suppression_approved": False,
            "production_ready": False,
            "pairs": geometry["pair_count"],
            "review_cases": 48,
        },
    )
    dump(
        "02_INPUT_AND_PAIR_GEOMETRY.json",
        {
            "input": {
                "frames": 144,
                "pre_gate": 9067,
                "retained": 6509,
                "pitch_gate_suppressed": 2558,
                "candidate_labels": 252,
                "scene_reviews": 36,
                "missed_person_marks": 25,
            },
            "geometry": geometry,
            "thresholds": thresholds,
        },
    )
    dump(
        "03_POLICY_AND_EXISTING_SAFETY.json",
        {
            "policies": policies,
            "targeted_existing_safety": safety,
            "human_labels_used_in_implementable_decisions": False,
        },
    )
    dump("04_FULL_UNIVERSE_NESTED_RESULTS.json", policies)
    dump(
        "05_REVIEW_SELECTION_AND_REVIEWER.json",
        {
            "selection": selection,
            "review_revision": "G7D_C3B_NESTED_CANDIDATE_REVIEW_V1",
            "port": 8817,
            "live_edge_acceptance": edge,
            "human_root_synthetic_events": 0,
        },
    )
    (HAND / "06_DECISION.md").write_text(
        "# Decision\nPASS_G7D_C3B_NESTED_CANDIDATE_REVIEW_READY_FOR_HUMAN_REVIEW\n\nNo nested rule is selected or activated.\n",
        encoding="utf-8",
    )
    (HAND / "07_NESTED_REVIEW_CONTRACT.md").write_text(
        "# Review contract\n48 blind-first pair cases, eight per development match. Yellow is inner; cyan is outer. Six plain-English questions; no team label. Immutable event, acknowledgement, completion.\n",
        encoding="utf-8",
    )
    shutil.copy2(STAGE / "07_VISUAL_QA/01_NESTED_PAIR_REVIEW_READY.png", HAND / "08_REVIEWER_READY.png")
    shutil.copy2(STAGE / "07_VISUAL_QA/02_LEGITIMATE_INNER_PROTECTION_PATH.png", HAND / "09_PROTECTION_PATH.png")
    files = []
    for p in sorted(HAND.iterdir()):
        if p.name != "10_MANIFEST.json":
            files.append({"filename": p.name, "bytes": p.stat().st_size, "sha256": sha(p)})
    dump("10_MANIFEST.json", {"files": files, "self_hash_omitted": True})
    (STAGE / "09_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only CHATGPT_HANDOFF.\n", encoding="utf-8"
    )
    print("PASS_G7D_C3B_CHATGPT_HANDOFF")


if __name__ == "__main__":
    main()
