# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

from football_intelligence.paths import STAGE3C13_GOLD20_MANUAL_LABELS_PATH
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import (
    gold_visible_person_rows,
    load_completed_gold8_frames,
    strict_one_to_one_match,
)
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1C1_GOLD8_COLOUR_EVAL_SUMMARY_PATH,
    STEP1C1B_PROFILE_EVAL_SUMMARY_PATH,
    STEP1C1C_RECOMMENDED_NEXT_ACTION_PATH,
    STEP1C1C_SEED_VALIDATION_SUMMARY_PATH,
    STEP1C1C_SEEDED_COLOUR_BELIEF_ROWS_SANDBOX_PATH,
    STEP1C1C_SEEDED_COLOUR_EVAL_REPORT_PATH,
    STEP1C1C_SEEDED_COLOUR_EVAL_SUMMARY_PATH,
    read_json,
    write_json,
    write_text,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    safe_float,
)


GOLD_TEAM_COLOUR_PROXY_TYPES = {"team_1_player", "team_2_player"}
UNKNOWN_SEEDED_BELIEFS = {"unknown_ambiguous_colour", "ambiguous_outfield_colour", "crop_unusable"}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def dominant(counts: dict[str, int]) -> tuple[str, int, float]:
    if not counts:
        return "", 0, 0.0
    total = sum(counts.values())
    label, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return label, count, round(count / max(1, total), 4)


def seeded_gold_proxy_rows(
    belief_payload: dict[str, Any],
    *,
    labels_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    labels_payload = labels_payload or read_json(STAGE3C13_GOLD20_MANUAL_LABELS_PATH)
    frame_sequences = {int(safe_float(frame.get("frame_sequence"), -1)) for frame in load_completed_gold8_frames(labels_payload)}
    belief_rows = [
        row
        for row in belief_payload.get("rows", [])
        if int(safe_float(row.get("frame_sequence"), -1)) in frame_sequences
    ]
    matches, _missed, _extra = strict_one_to_one_match(gold_visible_person_rows(labels_payload), belief_rows)
    rows = []
    for match in matches:
        gold = match["gold"]
        candidate = match["candidate"]
        visible_type = str(gold.get("visible_person_type_gold", ""))
        if visible_type not in GOLD_TEAM_COLOUR_PROXY_TYPES:
            continue
        rows.append(
            {
                "frame_id": candidate.get("frame_id", ""),
                "frame_sequence": int(safe_float(candidate.get("frame_sequence"), -1)),
                "timestamp_seconds": safe_float(candidate.get("timestamp_seconds")),
                "visible_person_base_id": candidate.get("visible_person_base_id", ""),
                "detection_id": candidate.get("detection_id", ""),
                "source_detection_id": candidate.get("source_detection_id", ""),
                "gold_row_id": gold.get("gold_row_id", ""),
                "visible_person_type_gold": visible_type,
                "seed_team_colour_belief": candidate.get("seed_team_colour_belief", ""),
                "seed_team_colour_belief_state": candidate.get("seed_team_colour_belief_state", ""),
                "seed_team_colour_belief_confidence": safe_float(candidate.get("seed_team_colour_belief_confidence")),
                "nearest_seed_label_candidate": candidate.get("nearest_seed_label_candidate", ""),
                "bbox_iou": match.get("match_features", {}).get("bbox_iou", 0.0),
                "visual_gap_px": match.get("match_features", {}).get("visual_gap_px", 0.0),
                "visual_only_warning": VISUAL_ONLY_WARNING,
                "do_not_use_for_metrics": True,
                "production_ready": PRODUCTION_READY,
            }
        )
    return rows


def proxy_distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    distribution: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        distribution[str(row.get("visible_person_type_gold", ""))][str(row.get("seed_team_colour_belief", ""))] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(distribution.items())}


def separation_score(distribution: dict[str, dict[str, int]], unknown_count: int, dark_count: int) -> float:
    team_1_label, _team_1_count, team_1_purity = dominant(distribution.get("team_1_player", {}))
    team_2_label, _team_2_count, team_2_purity = dominant(distribution.get("team_2_player", {}))
    proxy_total = sum(sum(counts.values()) for counts in distribution.values())
    if not team_1_label or not team_2_label:
        return 0.0
    if team_1_label != team_2_label and team_1_label not in UNKNOWN_SEEDED_BELIEFS and team_2_label not in UNKNOWN_SEEDED_BELIEFS:
        raw = (team_1_purity + team_2_purity) / 2.0
    elif team_1_label != team_2_label:
        raw = (team_1_purity + team_2_purity) / 4.0
    else:
        raw = abs(team_1_purity - team_2_purity) * 0.15
    return round(max(0.0, raw - (unknown_count / max(1, proxy_total) * 0.25) - (dark_count / max(1, proxy_total) * 0.25)), 4)


def one_frame_contradictions(belief_payload: dict[str, Any], labels_payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    frame_sequences = {int(safe_float(frame.get("frame_sequence"), -1)) for frame in load_completed_gold8_frames(labels_payload)}
    counts_by_frame: dict[int, Counter[str]] = defaultdict(Counter)
    for row in belief_payload.get("rows", []):
        seq = int(safe_float(row.get("frame_sequence"), -1))
        if seq in frame_sequences:
            counts_by_frame[seq][str(row.get("seed_team_colour_belief", ""))] += 1
    diagnostics = []
    for seq, counts in sorted(counts_by_frame.items()):
        non_unknown = [key for key in counts if key not in UNKNOWN_SEEDED_BELIEFS]
        if len(non_unknown) > 3:
            diagnostics.append(
                {
                    "frame_sequence": seq,
                    "distinct_non_unknown_seeded_beliefs": len(non_unknown),
                    "cluster_counts": dict(sorted(counts.items())),
                }
            )
    return diagnostics


def safety_missing_reasons(
    validation_summary: dict[str, Any],
    distribution: dict[str, dict[str, int]],
    unknown_count: int,
    dark_count: int,
    context_forced_count: int,
    c1_baseline: dict[str, Any],
) -> list[str]:
    reasons = []
    if not validation_summary.get("reviewed_seed_labels_loaded"):
        reasons.append("reviewed_seed_labels_missing")
    if not validation_summary.get("reviewed_seed_labels_valid"):
        reasons.append("reviewed_seed_labels_invalid_or_absent")
    if int(validation_summary.get("human_confirmed_team_1_seed_count", 0)) < 8:
        reasons.append("need_at_least_8_human_confirmed_team_1_seeds")
    if int(validation_summary.get("human_confirmed_team_2_seed_count", 0)) < 8:
        reasons.append("need_at_least_8_human_confirmed_team_2_seeds")
    if int(validation_summary.get("human_confirmed_negative_seed_count", 0)) < 4:
        reasons.append("need_at_least_4_human_confirmed_ambiguous_context_dark_negative_seeds")
    team_1_label, _count_1, _purity_1 = dominant(distribution.get("team_1_player", {}))
    team_2_label, _count_2, _purity_2 = dominant(distribution.get("team_2_player", {}))
    if not team_1_label or not team_2_label or team_1_label == team_2_label:
        reasons.append("gold_proxy_team_1_team_2_dominant_seeded_beliefs_do_not_differ")
    if unknown_count > int(c1_baseline.get("unknown_on_gold_player_proxy_count", 40)):
        reasons.append("unknown_on_gold_player_proxy_exceeds_c1_baseline")
    if dark_count > int(c1_baseline.get("dark_context_on_gold_player_proxy_count", 2)):
        reasons.append("dark_context_on_gold_player_proxy_exceeds_c1_baseline")
    if context_forced_count != 0:
        reasons.append("context_or_offroi_rows_forced_to_team")
    return reasons


def build_seeded_eval_summary(
    validation_summary: dict[str, Any],
    belief_payload: dict[str, Any],
    *,
    labels_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    c1_eval_summary = read_json(STEP1C1_GOLD8_COLOUR_EVAL_SUMMARY_PATH)
    c1b_eval_summary = read_json(STEP1C1B_PROFILE_EVAL_SUMMARY_PATH)
    c1_baseline = {
        "unknown_on_gold_player_proxy_count": c1_eval_summary.get("gold8_colour_eval_summary", {}).get("gold8_colour_proxy_distribution", {}).get("team_1_player", {}).get("unknown_ambiguous_colour", 0)
        + c1_eval_summary.get("gold8_colour_eval_summary", {}).get("gold8_colour_proxy_distribution", {}).get("team_2_player", {}).get("unknown_ambiguous_colour", 0),
        "dark_context_on_gold_player_proxy_count": c1_eval_summary.get("gold8_colour_eval_summary", {}).get("gold8_colour_proxy_distribution", {}).get("team_1_player", {}).get("dark_context_colour_like", 0)
        + c1_eval_summary.get("gold8_colour_eval_summary", {}).get("gold8_colour_proxy_distribution", {}).get("team_2_player", {}).get("dark_context_colour_like", 0),
    }
    proxy_rows = seeded_gold_proxy_rows(belief_payload, labels_payload=labels_payload)
    distribution = proxy_distribution(proxy_rows)
    team_1_label, team_1_count, team_1_purity = dominant(distribution.get("team_1_player", {}))
    team_2_label, team_2_count, team_2_purity = dominant(distribution.get("team_2_player", {}))
    unknown_count = sum(1 for row in proxy_rows if row.get("seed_team_colour_belief") in UNKNOWN_SEEDED_BELIEFS)
    dark_count = sum(1 for row in proxy_rows if row.get("seed_team_colour_belief") == "dark_context_colour_like")
    context_forced = belief_payload.get("summary", {}).get("context_offroi_forced_to_team_count", 0)
    contradictions = one_frame_contradictions(belief_payload, labels_payload=labels_payload)
    missing = safety_missing_reasons(validation_summary, distribution, unknown_count, dark_count, context_forced, c1_baseline)
    safe = not missing
    return {
        "artifact": "step1c1c_seeded_colour_eval_summary",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_classification_performed": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "reviewed_seed_labels_loaded": validation_summary.get("reviewed_seed_labels_loaded", False),
        "reviewed_seed_labels_valid": validation_summary.get("reviewed_seed_labels_valid", False),
        "human_confirmed_seed_counts_by_manual_label": validation_summary.get("human_confirmed_seed_counts_by_manual_label", {}),
        "human_confirmed_team_1_seed_count": validation_summary.get("human_confirmed_team_1_seed_count", 0),
        "human_confirmed_team_2_seed_count": validation_summary.get("human_confirmed_team_2_seed_count", 0),
        "human_confirmed_negative_seed_count": validation_summary.get("human_confirmed_negative_seed_count", 0),
        "seeded_belief_row_count": len(belief_payload.get("rows", [])),
        "seeded_unknown_ambiguous_count": belief_payload.get("summary", {}).get("seeded_unknown_ambiguous_rows", 0),
        "seeded_team_1_team_2_gold_proxy_distribution": distribution,
        "seeded_team_1_proxy_dominant_belief": team_1_label,
        "seeded_team_2_proxy_dominant_belief": team_2_label,
        "seeded_team_1_proxy_dominant_count": team_1_count,
        "seeded_team_2_proxy_dominant_count": team_2_count,
        "seeded_team_1_proxy_purity": team_1_purity,
        "seeded_team_2_proxy_purity": team_2_purity,
        "seeded_team_1_team_2_separation_score": separation_score(distribution, unknown_count, dark_count),
        "unknown_on_gold_player_proxy_count": unknown_count,
        "dark_context_on_gold_player_proxy_count": dark_count,
        "context_offroi_forced_to_team_count": context_forced,
        "one_frame_colour_contradiction_diagnostics": contradictions[:20],
        "frames_needing_manual_followup": [item["frame_sequence"] for item in contradictions[:12]],
        "c1_baseline": c1_baseline,
        "c1b_baseline": {
            "c1b_best_profile_name": c1b_eval_summary.get("c1b_best_profile_name", ""),
            "c1b_safe_for_team_colour_separation_review": c1b_eval_summary.get("c1b_safe_for_team_colour_separation_review", False),
            "c1b_best_unknown_ambiguous_colour_rows": c1b_eval_summary.get("c1b_best_unknown_ambiguous_colour_rows", 0),
        },
        "c1c_safe_for_c2_smoothing_review": safe,
        "c1c_safety_missing_reasons": missing,
        "c1c_safety_message": "Step1.C1c seeded profile is safe for Step1.C2 short-burst colour stability sandbox review." if safe else "Step1.C1c is not safe for C2 smoothing yet; more manual seed review is required.",
        "gold_proxy_note": "Gold visible_person_type_gold is used only as visual colour QA proxy context.",
    }


def seeded_colour_eval_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Step1.C1c Seeded Colour Eval Report",
        "",
        f"- Warning: `{VISUAL_ONLY_WARNING}`.",
        "- Scope: manual visual seed review and seeded colour-belief validation only.",
        "- Gold visible_person_type_gold is used only as visual QA/proxy context.",
        "- No team mapping was auto-promoted and no GK/official/identity/slot/metric/football analysis was performed.",
        "",
        "## Why Manual Seeds Are Needed",
        "",
        "Step1.C1 and Step1.C1b produced useful colour evidence, but C1b did not find a safe automatic two-outfield-colour separation profile.",
        "",
        "## C1/C1b Baseline",
        "",
        "```json",
        json.dumps({"c1_baseline": summary.get("c1_baseline", {}), "c1b_baseline": summary.get("c1b_baseline", {})}, indent=2),
        "```",
        "",
        "## Candidate Seed Selection Policy",
        "",
        "- Build diverse visual seed candidates from B4/C1/C1b rows.",
        "- Keep Gold-derived suggestions prefill-only until human confirmation.",
        "- Include likely team seeds plus ambiguous, context, dark, other-distinct, and crop-quality review examples.",
        "",
        "## Reviewed Seed Labels",
        "",
        f"- Reviewed labels found: {summary.get('reviewed_seed_labels_loaded', False)}",
        f"- Reviewed labels valid: {summary.get('reviewed_seed_labels_valid', False)}",
        f"- Human-confirmed seed counts: {summary.get('human_confirmed_seed_counts_by_manual_label', {})}",
        "",
    ]
    if not summary.get("reviewed_seed_labels_loaded"):
        lines.extend(
            [
                "Manual review is required before seeded validation can proceed. Complete `step1c1c_manual_colour_seed_label_template.json` or the CSV version, then save the reviewed JSON as `step1c1c_reviewed_colour_seed_labels.json`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Seeded Prototype Results And Gold-8 Proxy Diagnostics",
            "",
            f"- Seeded belief rows: {summary.get('seeded_belief_row_count', 0)}",
            f"- Seeded unknown/ambiguous count: {summary.get('seeded_unknown_ambiguous_count', 0)}",
            f"- Gold proxy distribution: {summary.get('seeded_team_1_team_2_gold_proxy_distribution', {})}",
            f"- Separation score: {summary.get('seeded_team_1_team_2_separation_score', 0.0)}",
            f"- Context/off-ROI forced-to-team count: {summary.get('context_offroi_forced_to_team_count', 0)}",
            "",
            "## Recommendation",
            "",
            summary.get("c1c_safety_message", ""),
            "",
            "## Safety Missing Reasons",
            "",
            "```json",
            json.dumps(summary.get("c1c_safety_missing_reasons", []), indent=2),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def recommended_next_action(summary: dict[str, Any]) -> str:
    if not summary.get("reviewed_seed_labels_loaded"):
        action = "Complete the JSON or CSV manual seed label template and save reviewed labels as `step1c1c_reviewed_colour_seed_labels.json`. Do not proceed to C2."
    elif not summary.get("c1c_safe_for_c2_smoothing_review"):
        action = "Do not proceed to C2. Address the missing safety requirements listed below and add more manual seed review."
    else:
        action = "Proceed only to a Step1.C2 short-burst colour stability sandbox. Do not auto-promote any team-colour mapping."
    return "\n".join(
        [
            "# Step1.C1c Recommended Next Action",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            f"- Reviewed labels loaded: {summary.get('reviewed_seed_labels_loaded', False)}",
            f"- Reviewed labels valid: {summary.get('reviewed_seed_labels_valid', False)}",
            f"- Safe for C2 smoothing review: {summary.get('c1c_safe_for_c2_smoothing_review', False)}",
            "",
            "## Action",
            "",
            action,
            "",
            "## Missing Requirements",
            "",
            "```json",
            json.dumps(summary.get("c1c_safety_missing_reasons", []), indent=2),
            "```",
            "",
            "- No team mapping was auto-promoted.",
            "- production_ready remains false.",
        ]
    ) + "\n"


def build_and_write_seeded_colour_eval() -> dict[str, Any]:
    validation_summary = read_json(STEP1C1C_SEED_VALIDATION_SUMMARY_PATH)
    belief_payload = read_json(STEP1C1C_SEEDED_COLOUR_BELIEF_ROWS_SANDBOX_PATH)
    summary = build_seeded_eval_summary(validation_summary, belief_payload)
    write_json(STEP1C1C_SEEDED_COLOUR_EVAL_SUMMARY_PATH, summary)
    write_text(STEP1C1C_SEEDED_COLOUR_EVAL_REPORT_PATH, seeded_colour_eval_report(summary))
    write_text(STEP1C1C_RECOMMENDED_NEXT_ACTION_PATH, recommended_next_action(summary))
    return summary
