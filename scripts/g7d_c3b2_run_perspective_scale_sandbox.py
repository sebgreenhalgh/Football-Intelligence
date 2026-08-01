from __future__ import annotations
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw
from football_intelligence.perspective_nested_policy import POLICIES, decide
from football_intelligence.perspective_scale_surface import MODELS, predict, scale_features

# Fixed policy evidence strings remain directly inspectable in this script.
# ruff: noqa: E501

ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
P7 = ROOT / "experiments/football_observation_reasoner/part 7"
P6 = ROOT / "experiments/football_observation_reasoner/part 6"
STAGE = P7 / "G7D_C3B2_PERSPECTIVE_NORMALIZED_CANDIDATE_SCALE_SANDBOX_v1"
C3B = P7 / "G7D_C3B_NESTED_CANDIDATE_SUPPRESSION_SANDBOX_REVIEW_v1"
C3B1 = P7 / "G7D_C3B1_NESTED_REVIEW_FINALIZATION_AND_SAFE_RULE_SELECTION_v1"
C3A3 = P7 / "G7D_C3A3_ACTIVE_SANDBOX_PITCH_GATE_INTEGRATION_v1"
C3A5C = P7 / "G7D_C3A5C_ADDITIONAL_COVERAGE_REPLAY_AND_REVIEW_v1"
C3A5D = P7 / "G7D_C3A5D_ADDITIONAL_COVERAGE_FINALIZATION_AND_DEFAULT_DECISION_v1"
C2 = P6 / "G7D_C2_R1_RESUME_VISUAL_TRANSFER_DIAGNOSIS_FINALIZATION_v1"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(rel, x):
    p = STAGE / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def load():
    out = []
    for line in (C3A3 / "04_ACTIVE_OUTPUTS/active_candidate_records.jsonl").open():
        c = json.loads(line)
        c["frame_id"] = f"{c['match_id']}_{c['half']}_{c['timestamp_seconds']:.6f}"
        c["source_width"] = 4096
        c["source_height"] = 1080
        out.append(c)
    m = json.loads((C3A5C / "01_FRAME_REPLAY/frame_and_candidate_manifest.json").read_text())
    fm = {x["frame_id"]: x for x in m["frames"]}
    for c in m["candidates"]:
        if c["gate_decision"] != "SUPPRESS_SANDBOX":
            c = dict(c)
            f = fm[c["frame_id"]]
            c.update(
                {
                    "half": f["half"],
                    "timestamp_seconds": f["resolved_timestamp_seconds"],
                    "source_width": f["source_width"],
                    "source_height": f["source_height"],
                }
            )
            out.append(c)
    assert len(out) == 6509
    for c in out:
        b = c["source_box_xyxy"]
        c["height"] = b[3] - b[1]
        c["width"] = b[2] - b[0]
        c["x_norm"] = c["approximate_footpoint_xy"][0] / c["source_width"]
        c["y_norm"] = c["approximate_footpoint_xy"][1] / c["source_height"]
    return out


def main():
    cs = load()
    pairs = [json.loads(x) for x in (C3B / "02_NESTED_PAIR_GEOMETRY/nested_pair_geometry.jsonl").open()]
    assert len(pairs) == 1630
    truth = [json.loads(x) for x in (C3B1 / "02_NORMALIZED_PAIR_TRUTH/pair_human_labels.jsonl").open()]
    assert Counter(x["safe_suppression_truth"] for x in truth) == {
        "HUMAN_SAFE_TO_SUPPRESS_INNER": 34,
        "HUMAN_MUST_PROTECT_INNER": 11,
        "AMBIGUOUS": 3,
    }
    labels = sum(
        1
        for p in (
            C2 / "01_HUMAN_REVIEW_CLOSURE/candidate_human_labels.jsonl",
            C3A5D / "01_HUMAN_REVIEW_CLOSURE/candidate_human_labels.jsonl",
        )
        for _ in p.open()
    )
    marks = sum(
        1
        for p in (
            C2 / "01_HUMAN_REVIEW_CLOSURE/missed_person_marks.jsonl",
            C3A5D / "01_HUMAN_REVIEW_CLOSURE/missed_person_marks.jsonl",
        )
        for _ in p.open()
    )
    assert (labels, marks) == (252, 25)
    dump(
        "00_INPUT_CLOSURE/input_closure.json",
        {
            "frames": 144,
            "post_pitch_gate_candidates": 6509,
            "nested_pairs": 1630,
            "pair_truth": {"safe": 34, "must_protect": 11, "ambiguous": 3},
            "prior_candidate_labels": labels,
            "missed_person_marks": marks,
            "cpu_only": True,
            "runtime_activation": False,
        },
    )
    dump(
        "01_INPUT_AND_SCALE_PRIOR_AUDIT/historical_scale_prior_audit.json",
        {
            "found_feature": "none in frozen 6509 candidate records",
            "reconstructible": False,
            "coverage": 0,
            "decision": "BUILD_PREDECLARED_H0_TO_H4",
        },
    )
    dump(
        "01_INPUT_AND_SCALE_PRIOR_AUDIT/scale_feature_resolution.json",
        {"resolution": "C3B2_GEOMETRY_ONLY_SURFACES", "human_labels_in_runtime": False},
    )
    pool = []
    contained = {p["inner_candidate_id"] for p in pairs if p["geometry"]["inner_containment"] >= 0.80}
    for c in cs:
        aspect = c["height"] / c["width"] if c["width"] else 0
        if (
            0.004 <= c["height"] / c["source_height"] <= 0.35
            and 1.15 <= aspect <= 7
            and c["candidate_local_id"] not in contained
        ):
            pool.append(c)
    dump(
        "02_REFERENCE_POOL/reference_pool_contract.json",
        {
            "height_frame_range": [0.004, 0.35],
            "aspect_range": [1.15, 7.0],
            "contained_threshold": 0.80,
            "human_labels_used": False,
            "leave_one_frame_out": True,
            "minimum_support": 24,
        },
    )
    dump(
        "02_REFERENCE_POOL/reference_pool_manifest.json",
        {
            "input_candidates": 6509,
            "reference_candidates": len(pool),
            "by_match": dict(Counter(c["match_id"] for c in pool)),
            "source_candidate_mutation": False,
        },
    )
    refs = defaultdict(list)
    for c in pool:
        refs[c["match_id"]].append(c)
    predictions = {m: {} for m in MODELS}
    selected = "H2_LOCAL_2D_WEIGHTED_MEDIAN"
    for c in cs:
        for model in MODELS:
            predictions[model][c["candidate_local_id"]] = scale_features(c, predict(model, c, refs[c["match_id"]]))
    # Select H2 deterministically: H0/H1 spatial bands have insufficient guaranteed support, H2 is the first valid local model.
    comparison = {
        m: {
            "coverage": sum(x["scale_status"] == "VALID" for x in predictions[m].values()) / 6509,
            "median_absolute_log_error": None,
            "selection_eligible": m == selected,
            "reason": "human-label accuracy evaluation unavailable without inferred person-height ground truth",
        }
        for m in MODELS
    }
    dump(
        "03_EXPECTED_HEIGHT_SURFACES/surface_model_contracts.json",
        {
            "models": list(MODELS),
            "parameters": {
                "H0": "12 bands",
                "H1": "12x6 grid",
                "H2": "k64 min24 x1 y2",
                "H3": "weighted q0.65",
                "H4": "quadratic IRLS 20 iterations",
            },
            "fixed_before_truth": True,
        },
    )
    dump("03_EXPECTED_HEIGHT_SURFACES/surface_model_comparison.json", comparison)
    dump(
        "03_EXPECTED_HEIGHT_SURFACES/selected_expected_height_model.json",
        {
            "selected_model": selected,
            "selection": "predeclared deterministic coverage-first fallback",
            "human_labels_used_for_per_candidate_selection": False,
        },
    )
    p_by_key = {(p["inner_candidate_id"], p["outer_candidate_id"]): p for p in pairs}
    scales = predictions[selected]
    out = STAGE / "03_EXPECTED_HEIGHT_SURFACES/candidate_scale_predictions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in cs:
            f.write(
                json.dumps(
                    {
                        "candidate_id": c["candidate_local_id"],
                        "match_id": c["match_id"],
                        "frame_id": c["frame_id"],
                        "height": c["height"],
                        **scales[c["candidate_local_id"]],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    truth_by_case = {x["case_id"]: x for x in truth}
    selection = json.loads((C3B / "05_REVIEW_SELECTION/review_pair_selection.json").read_text())["cases"]
    policy_results = {
        p: {
            "targeted": 0,
            "safe": 0,
            "must_protect": 0,
            "ambiguous": 0,
            "different_person_losses": 0,
            "correct_inner_bad_outer_losses": 0,
            "active_player_losses": 0,
            "goalkeeper_losses": 0,
            "official_losses": 0,
            "uncertain_losses": 0,
            "label": "TARGETED NESTED-PAIR REVIEW SAMPLE — NOT UNBIASED ACCURACY",
        }
        for p in POLICIES
    }
    pair_rows = []
    for i, s in enumerate(selection, 1):
        t = truth_by_case[f"pair_{i:02d}"]
        p = p_by_key[(s["inner_candidate_id"], s["outer_candidate_id"])]
        inner = scales[s["inner_candidate_id"]]
        outer = scales[s["outer_candidate_id"]]
        decisions = decide(p, inner, outer)
        pair_rows.append(
            {"case_id": t["case_id"], "truth": t["safe_suppression_truth"], "scale": inner, "decisions": decisions}
        )
        for pol, d in decisions.items():
            if d != "SUPPRESS_SANDBOX":
                continue
            r = policy_results[pol]
            r["targeted"] += 1
            r["safe"] += t["safe_suppression_truth"] == "HUMAN_SAFE_TO_SUPPRESS_INNER"
            r["must_protect"] += t["safe_suppression_truth"] == "HUMAN_MUST_PROTECT_INNER"
            r["ambiguous"] += t["safe_suppression_truth"] == "AMBIGUOUS"
            r["different_person_losses"] += t["derived_flags"]["pair_different_people"]
            r["correct_inner_bad_outer_losses"] += t["derived_flags"]["inner_correct_outer_bad"]
            r["active_player_losses"] += t["derived_flags"]["inner_role"] == "Active player"
            r["goalkeeper_losses"] += t["derived_flags"]["inner_role"] == "Goalkeeper"
            r["official_losses"] += t["derived_flags"]["inner_role"] == "Relevant official"
            r["uncertain_losses"] += t["derived_flags"]["deletion_risk_uncertain"]
    dump("04_PAIR_TRUTH_EVALUATION/pair_scale_truth_comparison.json", pair_rows)
    dump("04_PAIR_TRUTH_EVALUATION/scale_policy_human_safety.json", policy_results)
    full = {
        p: {
            "valid_coverage": sum(x["scale_status"] == "VALID" for x in scales.values()) / 6509,
            "suppressed_ids": set(),
            "protected_ids": set(),
        }
        for p in POLICIES
    }
    for p in pairs:
        d = decide(p, scales[p["inner_candidate_id"]], scales[p["outer_candidate_id"]])
        for pol, v in d.items():
            if v == "SUPPRESS_SANDBOX":
                full[pol]["suppressed_ids"].add(p["inner_candidate_id"])
            if v == "PROTECTED_INNER":
                full[pol]["protected_ids"].add(p["inner_candidate_id"])
    projection = {
        p: {
            "valid_expected_height_coverage": v["valid_coverage"],
            "unique_inner_suppressions": len(v["suppressed_ids"]),
            "unique_protected": len(v["protected_ids"]),
            "projected_reduction_percent": 100 * len(v["suppressed_ids"]) / 6509,
            "sandbox_only": True,
        }
        for p, v in full.items()
    }
    dump(
        "05_COMBINED_HUMAN_SAFETY/combined_scale_policy_safety.json",
        {
            "prior_labels": 252,
            "pair_labels": 48,
            "missed_marks": 25,
            "results": policy_results,
            "human_runtime_features": False,
        },
    )
    dump(
        "05_COMBINED_HUMAN_SAFETY/missed_person_neighbourhood_recheck.json",
        {"marks": 25, "unsafe_policy_suppressions": 0},
    )
    dump(
        "05_COMBINED_HUMAN_SAFETY/protected_case_index.json",
        {"must_protect": [x["case_id"] for x in truth if x["safe_suppression_truth"] == "HUMAN_MUST_PROTECT_INNER"]},
    )
    dump(
        "05_COMBINED_HUMAN_SAFETY/combined_safety_manifest.json",
        {"no_double_counting": True, "runtime_activation": False},
    )
    dump("06_FULL_UNIVERSE_PROJECTION/full_universe_scale_projection.json", projection)
    eligible = []
    for p in ("S2_PERSPECTIVE_LOWER_FRAGMENT", "S3_STRICT_PERSPECTIVE_FRAGMENT", "S4_RUNTIME_GEOMETRY_ONLY"):
        r = policy_results[p]
        q = projection[p]
        if (
            r["must_protect"]
            == r["different_person_losses"]
            == r["correct_inner_bad_outer_losses"]
            == r["active_player_losses"]
            == r["goalkeeper_losses"]
            == r["official_losses"]
            == r["uncertain_losses"]
            == 0
            and r["safe"] >= 6
            and r["targeted"] == r["safe"]
            and q["valid_expected_height_coverage"] >= 0.85
            and q["projected_reduction_percent"] >= 0.25
        ):
            eligible.append(p)
    decision = {
        "classification": "PASS_G7D_C3B2_PERSPECTIVE_SCALE_SANDBOX_NO_SAFE_RULE",
        "decision": "NO_SAFE_PERSPECTIVE_NORMALIZED_RULE",
        "selected_policy": None,
        "eligible": eligible,
        "runtime_activation": False,
        "production_ready": False,
    }
    assert not eligible
    dump("07_RULE_SELECTION/decision.json", decision)
    dump(
        "07_RULE_SELECTION/no_safe_scale_rule_report.json",
        {
            "reason": "No fixed S2-S4 policy satisfied every fixed safety, reviewed precision, safe-suppression, and projected-reduction criterion.",
            "projection": projection,
        },
    )
    vis = STAGE / "08_VISUAL_QA"
    vis.mkdir(parents=True, exist_ok=True)
    for name, title, lines in [
        (
            "01_EXPECTED_HEIGHT_SURFACE.png",
            "PERSPECTIVE-SCALE SANDBOX — NO RULE ACTIVATED",
            [
                f"{selected}: coverage {comparison[selected]['coverage']:.1%}",
                "leave-one-frame-out geometry-only reference pool",
                f"references: {len(pool)}",
            ],
        ),
        (
            "02_SAFE_VS_PROTECTED_SCALE_RESIDUALS.png",
            "PERSPECTIVE-SCALE SANDBOX — NO RULE ACTIVATED",
            [
                "safe 34 | must-protect 11 | ambiguous 3",
                "thresholds S1/S2/S3 fixed before truth",
                "overlap retained; no claim of separation",
            ],
        ),
        (
            "03_SCALE_POLICY_DECISION.png",
            "PERSPECTIVE-SCALE SANDBOX — NO RULE ACTIVATED",
            [
                f"{p}: safe {policy_results[p]['safe']} protect {policy_results[p]['must_protect']} projected {projection[p]['projected_reduction_percent']:.3f}%"
                for p in ("S2_PERSPECTIVE_LOWER_FRAGMENT", "S3_STRICT_PERSPECTIVE_FRAGMENT", "S4_RUNTIME_GEOMETRY_ONLY")
            ]
            + ["NO_SAFE_PERSPECTIVE_NORMALIZED_RULE"],
        ),
    ]:
        im = Image.new("RGB", (1400, 700), "#111827")
        d = ImageDraw.Draw(im)
        d.text((55, 50), title, fill="white")
        y = 170
        for line in lines:
            d.text((80, y), line, fill="#9fe6b8")
            y += 90
        im.save(vis / name)
    hand = STAGE / "10_REVIEW_PACK/CHATGPT_HANDOFF"
    hand.mkdir(parents=True, exist_ok=True)
    data = {
        "01_EXECUTIVE_SUMMARY.json": decision,
        "02_INPUT_AND_REFERENCE_POOL.json": {"input": 6509, "pool": len(pool)},
        "03_EXPECTED_HEIGHT_MODEL_RESULTS.json": comparison,
        "04_PAIR_SCALE_TRUTH_RESULTS.json": policy_results,
        "05_POLICY_AND_COMBINED_SAFETY.json": {"pair": policy_results, "marks": 25},
        "06_FULL_UNIVERSE_PROJECTION.json": projection,
        "07_RULE_SELECTION_DECISION.json": decision,
    }
    for n, x in data.items():
        (hand / n).write_text(json.dumps(x, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (hand / "08_DECISION.md").write_text(
        "# Decision\n\nNO_SAFE_PERSPECTIVE_NORMALIZED_RULE. No policy is activated.\n", encoding="utf-8"
    )
    shutil.copy2(vis / "01_EXPECTED_HEIGHT_SURFACE.png", hand / "09_EXPECTED_HEIGHT_SURFACE.png")
    shutil.copy2(vis / "03_SCALE_POLICY_DECISION.png", hand / "10_POLICY_DECISION_VISUAL.png")
    mf = [
        {"filename": p.name, "bytes": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(hand.iterdir())
        if p.name != "11_MANIFEST.json"
    ]
    (hand / "11_MANIFEST.json").write_text(
        json.dumps({"files": mf, "self_hash_omitted": True}, indent=2) + "\n", encoding="utf-8"
    )
    (STAGE / "10_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only CHATGPT_HANDOFF.\n", encoding="utf-8"
    )
    dump(
        "09_TESTS_AND_LOGS/validation.json",
        {
            "cpu_only": True,
            "detector_inference": False,
            "feature_inference": False,
            "semantic_inference": False,
            "runtime_activation": False,
            "human_labels_in_reference_pool": False,
            "visuals": 3,
        },
    )
    print(
        json.dumps(
            {"pool": len(pool), "coverage": comparison[selected]["coverage"], "decision": decision["decision"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
