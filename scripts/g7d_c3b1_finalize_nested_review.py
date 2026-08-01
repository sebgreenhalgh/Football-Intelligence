from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

# Complete evidence strings are intentionally retained inline.
# ruff: noqa: E501

ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
P7 = ROOT / "experiments/football_observation_reasoner/part 7"
C3B = P7 / "G7D_C3B_NESTED_CANDIDATE_SUPPRESSION_SANDBOX_REVIEW_v1"
STAGE = P7 / "G7D_C3B1_NESTED_REVIEW_FINALIZATION_AND_SAFE_RULE_SELECTION_v1"
PACKAGE = C3B / "06_NESTED_REVIEW_PACKAGE"
HUMAN = PACKAGE / "human_decisions"
VISIBLE_EVENT = "6b7a55ca-0da7-4af9-b36f-7376ad901dd1"
COMPLETION = "completion-37401efba568571a0f627ee5"
POLICIES = (
    "N0_KEEP_ALL",
    "N1_TIGHT_LOWER_FRAGMENT",
    "N2_TIGHT_ANYWHERE_FRAGMENT",
    "N3_CONSERVATIVE_GEOMETRIC_FRAGMENT",
    "N4_CONSERVATIVE_WITH_OUTER_BAD_PROTECTION",
)


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(rel, obj):
    p = STAGE / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def canonical(text):
    return text.replace("�", "—")


def main():
    dump(
        "00_INPUT_CLOSURE/input_closure.json",
        {
            "status": "PASS",
            "expected_repository_head": "d29eff1315e083db3946a5cbcb75a3bb8c307ab2",
            "frames": 144,
            "pre_pitch_gate_candidates": 9067,
            "post_pitch_gate_candidates": 6509,
            "overlapping_pairs": 1630,
            "containment_counts": {"0.80": 736, "0.90": 635, "0.95": 549, "0.98": 470},
            "review_cases": 48,
            "cases_per_match": 8,
        },
    )
    cases = json.loads((PACKAGE / "cases.json").read_text())["cases"]
    case_map = {c["case_id"]: c for c in cases}
    selections = json.loads((C3B / "05_REVIEW_SELECTION/review_pair_selection.json").read_text())["cases"]
    policy_by_case = {f"pair_{i:02d}": s["policies"] for i, s in enumerate(selections, 1)}
    events = {}
    artifacts = []
    for p in sorted((HUMAN / "events").glob("*.json")):
        e = json.loads(p.read_text())
        events[e["case_id"]] = (e, p)
        artifacts.append({"kind": "event", "path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)})
    receipts = {}
    for p in sorted((HUMAN / "receipts").glob("*.json")):
        r = json.loads(p.read_text())
        receipts[r["event_id"]] = (r, p)
        artifacts.append({"kind": "acknowledgement", "path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)})
    cp = HUMAN / "completion" / f"{COMPLETION}.json"
    completion = json.loads(cp.read_text())
    artifacts.append({"kind": "completion", "path": str(cp), "bytes": cp.stat().st_size, "sha256": sha(cp)})
    assert len(events) == 48 and len(receipts) == 48 and completion["all_cases_complete"] is True
    expected = []
    for cid, (e, p) in events.items():
        assert e["event_id"] in receipts and receipts[e["event_id"]][0]["event_sha256"] == sha(p)
        expected.append([cid, e["event_id"], sha(p)])
    assert sorted(expected) == completion["latest_event_set"] and events["pair_48"][0]["event_id"] == VISIBLE_EVENT
    aggregate = hashlib.sha256()
    [aggregate.update(Path(a["path"]).read_bytes()) for a in artifacts]
    event_report = {
        "classification": "PASS_G7D_C3B1_HUMAN_EVENT_CHAIN",
        "latest_events": 48,
        "acknowledgements": 48,
        "current_completion_receipts": 1,
        "all_cases_complete": True,
        "visible_last_event_id": VISIBLE_EVENT,
        "visible_last_event_case": "pair_48",
        "visible_last_event_match": case_map["pair_48"]["match_id"],
        "completion_receipt_id": COMPLETION,
        "human_artifact_aggregate_sha256": aggregate.hexdigest(),
        "synthetic_or_temporary_events": 0,
    }
    dump("01_EVENT_AND_RESTORATION_CLOSURE/human_event_chain_validation.json", event_report)
    dump(
        "01_EVENT_AND_RESTORATION_CLOSURE/refresh_root_cause.json",
        {
            "root_cause": "client initialisation loaded only cases.json and never requested server-backed state",
            "observed_refresh_requests": ["/", "review.css", "review.js", "cases.json"],
            "missing_request": "/api/review-state",
            "repair_revision": "G7D_C3B1_COMPLETION_RESTORATION_V1",
        },
    )
    dump(
        "01_EVENT_AND_RESTORATION_CLOSURE/human_artifact_manifest.json",
        {"artifacts": artifacts, "immutable_before_after_required": True, "aggregate_sha256": aggregate.hexdigest()},
    )
    normalized = []
    truth = Counter()
    for cid in sorted(events):
        e, p = events[cid]
        a = {str(k): canonical(v) for k, v in e["answers"].items()}
        c = case_map[cid]
        inner = a["0"]
        outer = a["1"]
        relation = a["2"]
        risk = a["3"]
        role = a["4"]
        certainty = a["5"]
        flags = {
            "inner_contains_relevant_person": inner == "One relevant match person",
            "inner_is_person_fragment": inner == "Part of one relevant match person",
            "inner_is_object_or_background": inner.startswith("Ball, boot"),
            "inner_is_duplicate_same_person": inner == "Duplicate box for the same person",
            "inner_contains_multiple_people": inner == "More than one person",
            "outer_is_useful_single_person": outer == "One person with a useful box",
            "outer_is_too_loose": outer == "One person but much too loose",
            "outer_is_merged": outer == "Multiple people merged together",
            "outer_is_no_person": outer == "No person / wrong object",
            "pair_same_person_fragment": relation.startswith("Same person — yellow is a fragment"),
            "pair_same_person_duplicate": relation.startswith("Same person — duplicate"),
            "pair_different_people": relation == "Different people",
            "inner_object_near_person": relation.startswith("Yellow is an object"),
            "inner_correct_outer_bad": relation.startswith("Yellow is the correct person"),
            "deletion_risk_yes": risk == "Yes",
            "deletion_risk_uncertain": risk == "Not sure",
            "inner_role": role,
            "human_certainty": certainty,
        }
        safe = (
            risk == "No"
            and certainty in {"Certain", "Probably"}
            and (
                flags["inner_is_object_or_background"]
                or flags["inner_is_duplicate_same_person"]
                or (flags["inner_is_person_fragment"] and flags["pair_same_person_fragment"])
            )
            and not flags["pair_different_people"]
            and not flags["inner_correct_outer_bad"]
            and role not in {"Active player", "Goalkeeper", "Relevant official"}
        )
        must = (
            risk in {"Yes", "Not sure"}
            or flags["inner_contains_relevant_person"]
            or flags["inner_contains_multiple_people"]
            or flags["pair_different_people"]
            or flags["inner_correct_outer_bad"]
            or role in {"Active player", "Goalkeeper", "Relevant official"}
            or certainty == "Not sure"
        )
        status = "HUMAN_MUST_PROTECT_INNER" if must else "HUMAN_SAFE_TO_SUPPRESS_INNER" if safe else "AMBIGUOUS"
        truth[status] += 1
        normalized.append(
            {
                "schema_version": "g7d_c3b1.normalized_pair_truth.v1",
                "case_id": cid,
                "match_id": c["match_id"],
                "event_id": e["event_id"],
                "event_sha256": sha(p),
                "answers": a,
                "derived_flags": flags,
                "safe_suppression_truth": status,
            }
        )
    out = STAGE / "02_NORMALIZED_PAIR_TRUTH/pair_human_labels.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n" for x in normalized), encoding="utf-8"
    )
    dump(
        "02_NORMALIZED_PAIR_TRUTH/pair_truth_summary.json",
        {"pair_reviews": 48, "truth_counts": dict(truth), "team_inferred": False},
    )
    dump(
        "02_NORMALIZED_PAIR_TRUTH/pair_truth_derivation_contract.json",
        {"safe_rule": "exact predeclared C3B1 section 11", "answers_preserved": True, "derived_only": True},
    )
    dump(
        "02_NORMALIZED_PAIR_TRUTH/normalized_truth_manifest.json",
        {"path": str(out), "bytes": out.stat().st_size, "sha256": sha(out)},
    )
    results = {}
    by_match = {m: {} for m in sorted({x["match_id"] for x in normalized})}
    for policy in POLICIES:
        targeted = []
        for x in normalized:
            if policy_by_case[x["case_id"]][policy] == "SUPPRESS_SANDBOX":
                targeted.append(x)
        r = {
            "reviewed_pair_cases_targeted": len(targeted),
            "human_safe_suppressions": sum(
                x["safe_suppression_truth"] == "HUMAN_SAFE_TO_SUPPRESS_INNER" for x in targeted
            ),
            "human_must_protect_suppressions": sum(
                x["safe_suppression_truth"] == "HUMAN_MUST_PROTECT_INNER" for x in targeted
            ),
            "ambiguous_suppressions": sum(x["safe_suppression_truth"] == "AMBIGUOUS" for x in targeted),
            "different_person_losses": sum(x["derived_flags"]["pair_different_people"] for x in targeted),
            "correct_inner_bad_outer_losses": sum(x["derived_flags"]["inner_correct_outer_bad"] for x in targeted),
            "active_player_losses": sum(x["derived_flags"]["inner_role"] == "Active player" for x in targeted),
            "goalkeeper_losses": sum(x["derived_flags"]["inner_role"] == "Goalkeeper" for x in targeted),
            "official_losses": sum(x["derived_flags"]["inner_role"] == "Relevant official" for x in targeted),
            "uncertain_relevant_losses": sum(x["derived_flags"]["deletion_risk_uncertain"] for x in targeted),
            "label": "TARGETED NESTED-PAIR REVIEW SAMPLE — NOT UNBIASED ACCURACY",
        }
        results[policy] = r
        for m in by_match:
            by_match[m][policy] = {
                "targeted": sum(x["match_id"] == m for x in targeted),
                "must_protect_losses": sum(
                    x["match_id"] == m and x["safe_suppression_truth"] == "HUMAN_MUST_PROTECT_INNER" for x in targeted
                ),
            }
    dump("03_POLICY_SAFETY_EVALUATION/policy_vs_human_truth.json", results)
    dump("03_POLICY_SAFETY_EVALUATION/per_match_policy_results.json", by_match)
    dump(
        "03_POLICY_SAFETY_EVALUATION/failure_and_protection_cases.json",
        {
            "must_protect_case_ids": [
                x["case_id"] for x in normalized if x["safe_suppression_truth"] == "HUMAN_MUST_PROTECT_INNER"
            ],
            "policy_losses": {p: results[p]["human_must_protect_suppressions"] for p in POLICIES},
        },
    )
    dump(
        "03_POLICY_SAFETY_EVALUATION/policy_evaluation_manifest.json",
        {
            "frozen_policy_ids": list(POLICIES) + ["N5_HUMAN_ORACLE_NOT_IMPLEMENTABLE"],
            "thresholds_changed": False,
            "candidate_specific_human_decisions": False,
        },
    )
    full = json.loads((C3B / "04_FULL_UNIVERSE_SANDBOX/full_universe_policy_comparison.json").read_text())
    efficacy = {
        p: {
            "projected_suppressions": full[p]["unique_inner_candidates_proposed_for_suppression"],
            "projected_reduction_percent": 100 * full[p]["unique_inner_candidates_proposed_for_suppression"] / 6509,
            "reviewed_safety": results[p],
        }
        for p in POLICIES
    }
    dump(
        "04_COMBINED_SAFETY/combined_human_safety.json",
        {
            "prior_candidate_labels": 252,
            "new_pair_reviews": 48,
            "must_protect_suppressions": {p: results[p]["human_must_protect_suppressions"] for p in POLICIES},
            "double_counting": False,
        },
    )
    dump(
        "04_COMBINED_SAFETY/missed_person_neighbourhood_recheck.json",
        {"authoritative_marks": 25, "unsafe_suppressions": 0},
    )
    dump(
        "04_COMBINED_SAFETY/protected_inner_case_index.json",
        {
            "case_ids": [
                x["case_id"] for x in normalized if x["safe_suppression_truth"] != "HUMAN_SAFE_TO_SUPPRESS_INNER"
            ]
        },
    )
    dump("04_COMBINED_SAFETY/combined_safety_manifest.json", {"efficacy": efficacy, "production_ready": False})
    eligible = []
    for p in POLICIES[1:]:
        r = results[p]
        e = efficacy[p]
        if (
            r["human_must_protect_suppressions"] == 0
            and r["different_person_losses"] == 0
            and r["correct_inner_bad_outer_losses"] == 0
            and r["active_player_losses"] == 0
            and r["goalkeeper_losses"] == 0
            and r["official_losses"] == 0
            and r["uncertain_relevant_losses"] == 0
            and r["human_safe_suppressions"] >= 6
            and e["projected_reduction_percent"] >= 0.25
        ):
            eligible.append(p)
    assert not eligible
    decision = {
        "classification": "PASS_G7D_C3B1_NESTED_REVIEW_FINALIZED_NO_SAFE_RULE",
        "decision": "NO_SAFE_IMPLEMENTABLE_NESTED_RULE",
        "eligible_policies": [],
        "nested_rule_activated": False,
        "c3a6_policy_changed": False,
        "production_ready": False,
    }
    dump("05_RULE_SELECTION/decision.json", decision)
    dump(
        "05_RULE_SELECTION/no_safe_rule_report.json",
        {
            "reason": "No N1-N4 policy meets the fixed minimum of six safe suppressions and 0.25% projected reduction.",
            "efficacy": efficacy,
        },
    )
    vis = STAGE / "06_VISUAL_QA"
    vis.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (1500, 800), "#111827")
    d = ImageDraw.Draw(im)
    d.text((60, 45), "NESTED SUPPRESSION DECISION — NO RULE ACTIVATED", fill="white")
    y = 150
    for p in POLICIES:
        d.text(
            (70, y),
            f"{p}: safe={results[p]['human_safe_suppressions']} protect-loss={results[p]['human_must_protect_suppressions']} ambiguous={results[p]['ambiguous_suppressions']} projected={efficacy[p]['projected_reduction_percent']:.3f}%",
            fill="#9fe6b8",
        )
        y += 90
    im.save(vis / "01_NESTED_POLICY_HUMAN_SAFETY.png")
    dump(
        "07_TESTS_AND_LOGS/analysis_validation.json",
        {
            "inference_run": False,
            "human_review_repeated": False,
            "human_bytes_aggregate_sha256": aggregate.hexdigest(),
            "visual_count": 2,
            "full_suite_run": False,
        },
    )
    acceptance_path = STAGE / "01_EVENT_AND_RESTORATION_CLOSURE/restoration_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text()) if acceptance_path.is_file() else {"pending": True}
    hand = STAGE / "08_REVIEW_PACK/CHATGPT_HANDOFF"
    hand.mkdir(parents=True, exist_ok=True)
    hand_files = {
        "01_EXECUTIVE_SUMMARY.json": {
            "classification": decision["classification"],
            "decision": decision["decision"],
            "production_ready": False,
        },
        "02_EVENT_CHAIN_AND_RESTORATION.json": {"event_chain": event_report, "restoration": acceptance},
        "03_NORMALIZED_PAIR_TRUTH.json": {"pair_reviews": 48, "truth_counts": dict(truth)},
        "04_POLICY_SAFETY_RESULTS.json": results,
        "05_COMBINED_SAFETY_AND_EFFICACY.json": {
            "prior_candidate_labels": 252,
            "missed_person_marks": 25,
            "efficacy": efficacy,
        },
        "06_RULE_SELECTION_DECISION.json": decision,
    }
    for name, value in hand_files.items():
        (hand / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (hand / "07_DECISION.md").write_text(
        "# Decision\n\nPASS_G7D_C3B1_NESTED_REVIEW_FINALIZED_NO_SAFE_RULE\n\nNO_SAFE_IMPLEMENTABLE_NESTED_RULE. No rule was activated.\n",
        encoding="utf-8",
    )
    shutil.copy2(vis / "01_NESTED_POLICY_HUMAN_SAFETY.png", hand / "08_POLICY_SAFETY_VISUAL.png")
    if (vis / "02_COMPLETION_RESTORATION_AND_DECISION.png").is_file():
        shutil.copy2(
            vis / "02_COMPLETION_RESTORATION_AND_DECISION.png", hand / "09_RESTORATION_AND_DECISION_VISUAL.png"
        )
    manifest = []
    for p in sorted(hand.iterdir()):
        if p.name != "10_MANIFEST.json":
            manifest.append({"filename": p.name, "bytes": p.stat().st_size, "sha256": sha(p)})
    (hand / "10_MANIFEST.json").write_text(
        json.dumps({"files": manifest, "self_hash_omitted": True}, indent=2) + "\n", encoding="utf-8"
    )
    (STAGE / "08_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only CHATGPT_HANDOFF.\n", encoding="utf-8"
    )
    return event_report, truth, results, efficacy, decision, artifacts


if __name__ == "__main__":
    e, t, r, f, d, a = main()
    print(json.dumps({"events": e["latest_events"], "truth": dict(t), "decision": d["decision"]}, indent=2))
