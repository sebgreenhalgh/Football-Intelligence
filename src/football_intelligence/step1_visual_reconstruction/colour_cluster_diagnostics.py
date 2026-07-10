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
    STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH,
    STEP1C1_GOLD8_COLOUR_EVAL_SUMMARY_PATH,
    STEP1C1B_BEST_SANDBOX_BELIEF_ROWS_PATH,
    STEP1C1B_BEST_SANDBOX_UNKNOWN_AMBIGUOUS_ROWS_PATH,
    STEP1C1B_COLOUR_PROFILE_SWEEP_PATH,
    STEP1C1B_GOLD8_CLUSTER_CONFUSION_ROWS_PATH,
    STEP1C1B_PROFILE_EVAL_REPORT_PATH,
    STEP1C1B_PROFILE_EVAL_SUMMARY_PATH,
    STEP1C1B_RECOMMENDED_PROFILE_PATH,
    STEP1C1B_REVIEW_DECISION_TEMPLATE_PATH,
    read_json,
    step1c1b_review_decision_template_payload,
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
UNKNOWN_OR_UNUSABLE_BELIEFS = {"unknown_ambiguous_colour", "crop_unusable"}
NON_UNKNOWN_BELIEFS = {
    "outfield_colour_cluster_a",
    "outfield_colour_cluster_b",
    "other_distinct_colour_like",
    "dark_context_colour_like",
    "team_1_colour_like",
    "team_2_colour_like",
}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def completed_gold_frame_sequences(labels_payload: dict[str, Any] | None = None) -> set[int]:
    return {int(safe_float(frame.get("frame_sequence"), -1)) for frame in load_completed_gold8_frames(labels_payload)}


def gold_proxy_confusion_rows(
    belief_payload: dict[str, Any],
    *,
    labels_payload: dict[str, Any] | None = None,
    profile_name: str = "",
    prototype_strategy: str = "",
) -> list[dict[str, Any]]:
    labels_payload = labels_payload or read_json(STAGE3C13_GOLD20_MANUAL_LABELS_PATH)
    frame_sequences = completed_gold_frame_sequences(labels_payload)
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
                "profile_name": profile_name or belief_payload.get("profile_name", ""),
                "prototype_strategy": prototype_strategy or belief_payload.get("prototype_strategy", ""),
                "frame_id": candidate.get("frame_id", ""),
                "frame_sequence": int(safe_float(candidate.get("frame_sequence"), -1)),
                "timestamp_seconds": safe_float(candidate.get("timestamp_seconds")),
                "visible_person_base_id": candidate.get("visible_person_base_id", ""),
                "detection_id": candidate.get("detection_id", ""),
                "source_detection_id": candidate.get("source_detection_id", ""),
                "gold_row_id": gold.get("gold_row_id", ""),
                "visible_person_type_gold": visible_type,
                "team_colour_belief": candidate.get("team_colour_belief", ""),
                "team_colour_belief_state": candidate.get("team_colour_belief_state", ""),
                "colour_cluster_candidate": candidate.get("colour_cluster_candidate", ""),
                "team_colour_belief_confidence": safe_float(candidate.get("team_colour_belief_confidence")),
                "crop_quality": candidate.get("crop_quality", ""),
                "crop_quality_reason": candidate.get("crop_quality_reason", ""),
                "bbox_iou": match.get("match_features", {}).get("bbox_iou", 0.0),
                "visual_gap_px": match.get("match_features", {}).get("visual_gap_px", 0.0),
                "sandbox_only": bool(belief_payload.get("sandbox_only", True)),
                "auto_promoted": False,
                "visual_only_warning": VISUAL_ONLY_WARNING,
                "do_not_use_for_metrics": True,
                "production_ready": PRODUCTION_READY,
            }
        )
    return rows


def proxy_distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    distribution: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        distribution[str(row.get("visible_person_type_gold", ""))][str(row.get("team_colour_belief", ""))] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(distribution.items())}


def dominant_cluster(counts: dict[str, int]) -> tuple[str, int, float]:
    if not counts:
        return "", 0, 0.0
    total = sum(counts.values())
    cluster, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return cluster, count, round(count / max(1, total), 4)


def frame_colour_contradictions(
    belief_payload: dict[str, Any],
    *,
    labels_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    frame_sequences = completed_gold_frame_sequences(labels_payload)
    counts_by_frame: dict[int, Counter[str]] = defaultdict(Counter)
    for row in belief_payload.get("rows", []):
        seq = int(safe_float(row.get("frame_sequence"), -1))
        if seq not in frame_sequences:
            continue
        counts_by_frame[seq][str(row.get("team_colour_belief", ""))] += 1
    diagnostics = []
    for seq, counts in sorted(counts_by_frame.items()):
        non_unknown = [key for key, count in counts.items() if count > 0 and key in NON_UNKNOWN_BELIEFS]
        if len(non_unknown) > 3:
            diagnostics.append(
                {
                    "frame_sequence": seq,
                    "distinct_non_unknown_colour_beliefs": len(non_unknown),
                    "cluster_counts": dict(sorted(counts.items())),
                }
            )
    return diagnostics


def separation_score(
    distribution: dict[str, dict[str, int]],
    *,
    unknown_on_gold_player_proxy_count: int,
    dark_context_on_gold_player_proxy_count: int,
) -> float:
    team_1_cluster, _team_1_count, team_1_purity = dominant_cluster(distribution.get("team_1_player", {}))
    team_2_cluster, _team_2_count, team_2_purity = dominant_cluster(distribution.get("team_2_player", {}))
    proxy_total = sum(sum(counts.values()) for counts in distribution.values())
    if not team_1_cluster or not team_2_cluster:
        return 0.0
    if team_1_cluster != team_2_cluster and team_1_cluster not in UNKNOWN_OR_UNUSABLE_BELIEFS and team_2_cluster not in UNKNOWN_OR_UNUSABLE_BELIEFS:
        raw = (team_1_purity + team_2_purity) / 2.0
    elif team_1_cluster != team_2_cluster:
        raw = (team_1_purity + team_2_purity) / 4.0
    else:
        raw = abs(team_1_purity - team_2_purity) * 0.15
    unknown_penalty = unknown_on_gold_player_proxy_count / max(1, proxy_total) * 0.25
    dark_penalty = dark_context_on_gold_player_proxy_count / max(1, proxy_total) * 0.25
    return round(max(0.0, raw - unknown_penalty - dark_penalty), 4)


def evaluate_gold8_proxy_clusters(
    belief_payload: dict[str, Any],
    *,
    labels_payload: dict[str, Any] | None = None,
    profile_name: str = "",
    prototype_strategy: str = "",
    high_confidence_background_contaminated_rows: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels_payload = labels_payload or read_json(STAGE3C13_GOLD20_MANUAL_LABELS_PATH)
    rows = gold_proxy_confusion_rows(
        belief_payload,
        labels_payload=labels_payload,
        profile_name=profile_name,
        prototype_strategy=prototype_strategy,
    )
    distribution = proxy_distribution(rows)
    team_1_cluster, team_1_count, team_1_purity = dominant_cluster(distribution.get("team_1_player", {}))
    team_2_cluster, team_2_count, team_2_purity = dominant_cluster(distribution.get("team_2_player", {}))
    unknown_count = sum(1 for row in rows if row.get("team_colour_belief") in UNKNOWN_OR_UNUSABLE_BELIEFS)
    dark_count = sum(1 for row in rows if row.get("team_colour_belief") == "dark_context_colour_like")
    contradictions = frame_colour_contradictions(belief_payload, labels_payload=labels_payload)
    score = separation_score(
        distribution,
        unknown_on_gold_player_proxy_count=unknown_count,
        dark_context_on_gold_player_proxy_count=dark_count,
    )
    same_cluster = bool(team_1_cluster and team_1_cluster == team_2_cluster)
    team_2_total = sum(distribution.get("team_2_player", {}).values())
    team_2_unknown = sum(
        count
        for belief, count in distribution.get("team_2_player", {}).items()
        if belief in UNKNOWN_OR_UNUSABLE_BELIEFS
    )
    promising = bool(
        team_1_cluster
        and team_2_cluster
        and team_1_cluster != team_2_cluster
        and team_1_cluster not in UNKNOWN_OR_UNUSABLE_BELIEFS
        and team_2_cluster not in UNKNOWN_OR_UNUSABLE_BELIEFS
    )
    summary = {
        "gold8_colour_eval_available": bool(rows),
        "gold8_colour_eval_fields_used": ["visible_person_type_gold"] if rows else [],
        "gold8_proxy_match_rows": len(rows),
        "gold8_proxy_distribution": distribution,
        "team_1_proxy_dominant_cluster": team_1_cluster,
        "team_2_proxy_dominant_cluster": team_2_cluster,
        "team_1_proxy_dominant_count": team_1_count,
        "team_2_proxy_dominant_count": team_2_count,
        "team_1_proxy_cluster_purity": team_1_purity,
        "team_2_proxy_cluster_purity": team_2_purity,
        "team_1_team_2_cluster_separation_score": score,
        "team_1_team_2_same_cluster_proxy": same_cluster,
        "one_cluster_dominates_both_teams": same_cluster,
        "team_2_mostly_unknown": bool(team_2_total and team_2_unknown / team_2_total >= 0.40),
        "unknown_on_gold_player_proxy_count": unknown_count,
        "dark_context_on_gold_player_proxy_count": dark_count,
        "one_frame_colour_contradiction_count": len(contradictions),
        "frames_needing_manual_followup": [item["frame_sequence"] for item in contradictions[:12]],
        "one_frame_colour_contradiction_diagnostics": contradictions[:20],
        "high_confidence_background_contaminated_rows": high_confidence_background_contaminated_rows,
        "profile_visually_promising": promising,
        "gold_proxy_note": "Gold visible_person_type_gold is used only as visual colour QA context, not role correctness.",
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "auto_promoted": False,
    }
    return summary, rows


def c1_baseline_proxy_counts(c1_eval_summary: dict[str, Any]) -> dict[str, int]:
    distribution = c1_eval_summary.get("gold8_colour_eval_summary", {}).get("gold8_colour_proxy_distribution", {})
    unknown = 0
    dark = 0
    for counts in distribution.values():
        unknown += sum(count for belief, count in counts.items() if belief in UNKNOWN_OR_UNUSABLE_BELIEFS)
        dark += int(counts.get("dark_context_colour_like", 0))
    return {"unknown_on_gold_player_proxy_count": unknown, "dark_context_on_gold_player_proxy_count": dark}


def iter_strategy_summaries(sweep_payload: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for profile in sweep_payload.get("profiles", []):
        for strategy_summary in profile.get("prototype_strategy_summaries", []):
            summary = dict(strategy_summary)
            summary["profile_name"] = profile.get("profile_name", summary.get("profile_name", ""))
            summaries.append(summary)
    return summaries


def select_best_strategy_summary(sweep_payload: dict[str, Any], c1_eval_summary: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    baseline = c1_baseline_proxy_counts(c1_eval_summary)
    b4_rows = int(safe_float(sweep_payload.get("b4_visible_person_base_rows"), 0))
    candidates = iter_strategy_summaries(sweep_payload)
    if not candidates:
        return {}, False

    def ranking(summary: dict[str, Any]) -> tuple[Any, ...]:
        stable = int(safe_float(summary.get("belief_rows"), -1)) == b4_rows
        unknown = int(safe_float(summary.get("unknown_on_gold_player_proxy_count"), 999999))
        dark = int(safe_float(summary.get("dark_context_on_gold_player_proxy_count"), 999999))
        bg = int(safe_float(summary.get("high_confidence_background_contaminated_rows"), 999999))
        safe_candidate = bool(
            summary.get("profile_visually_promising")
            and stable
            and unknown <= baseline["unknown_on_gold_player_proxy_count"]
            and dark <= baseline["dark_context_on_gold_player_proxy_count"]
            and bg == 0
        )
        return (
            safe_candidate,
            safe_float(summary.get("team_1_team_2_cluster_separation_score")),
            -unknown,
            -dark,
            -bg,
            stable,
            -int(safe_float(summary.get("unknown_ambiguous_rows"), 999999)),
            str(summary.get("profile_name", "")),
            str(summary.get("prototype_strategy", "")),
        )

    best = sorted(candidates, key=ranking, reverse=True)[0]
    safe_for_review = bool(ranking(best)[0])
    return best, safe_for_review


def profile_comparison_rows(sweep_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for summary in iter_strategy_summaries(sweep_payload):
        rows.append(
            {
                "profile_name": summary.get("profile_name", ""),
                "prototype_strategy": summary.get("prototype_strategy", ""),
                "belief_rows": summary.get("belief_rows", 0),
                "unknown_ambiguous_rows": summary.get("unknown_ambiguous_rows", 0),
                "high_confidence_visual_colour_rows": summary.get("high_confidence_visual_colour_rows", 0),
                "team_1_proxy_dominant_cluster": summary.get("team_1_proxy_dominant_cluster", ""),
                "team_2_proxy_dominant_cluster": summary.get("team_2_proxy_dominant_cluster", ""),
                "team_1_proxy_cluster_purity": summary.get("team_1_proxy_cluster_purity", 0.0),
                "team_2_proxy_cluster_purity": summary.get("team_2_proxy_cluster_purity", 0.0),
                "team_1_team_2_cluster_separation_score": summary.get("team_1_team_2_cluster_separation_score", 0.0),
                "unknown_on_gold_player_proxy_count": summary.get("unknown_on_gold_player_proxy_count", 0),
                "dark_context_on_gold_player_proxy_count": summary.get("dark_context_on_gold_player_proxy_count", 0),
                "profile_visually_promising": summary.get("profile_visually_promising", False),
            }
        )
    return rows


def build_profile_eval_summary(
    sweep_payload: dict[str, Any],
    c1_eval_summary: dict[str, Any],
    best_summary: dict[str, Any],
    *,
    safe_for_review: bool,
) -> dict[str, Any]:
    baseline_proxy = c1_baseline_proxy_counts(c1_eval_summary)
    manual_frames = sorted({int(safe_float(seq, -1)) for seq in best_summary.get("frames_needing_manual_followup", []) if int(safe_float(seq, -1)) >= 0})
    return {
        "artifact": "step1c1b_profile_eval_summary",
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
        "c1_baseline": {
            "b4_visible_person_base_rows": c1_eval_summary.get("b4_visible_person_base_rows", 0),
            "step1c1_colour_feature_rows": c1_eval_summary.get("step1c1_colour_feature_rows", 0),
            "step1c1_team_colour_belief_rows": c1_eval_summary.get("step1c1_team_colour_belief_rows", 0),
            "unknown_ambiguous_colour_rows": c1_eval_summary.get("unknown_ambiguous_colour_rows", 0),
            "cluster_counts": c1_eval_summary.get("cluster_counts", {}),
            "mapped_team_colour_counts": c1_eval_summary.get("mapped_team_colour_counts", {}),
            "gold8_proxy_distribution": c1_eval_summary.get("gold8_colour_eval_summary", {}).get("gold8_colour_proxy_distribution", {}),
            **baseline_proxy,
        },
        "profile_comparison_rows": profile_comparison_rows(sweep_payload),
        "c1_unknown_ambiguous_colour_rows": c1_eval_summary.get("unknown_ambiguous_colour_rows", 0),
        "c1b_best_profile_name": best_summary.get("profile_name", ""),
        "c1b_best_prototype_strategy": best_summary.get("prototype_strategy", ""),
        "c1b_best_unknown_ambiguous_colour_rows": best_summary.get("unknown_ambiguous_rows", 0),
        "c1b_team_1_team_2_separation_score": best_summary.get("team_1_team_2_cluster_separation_score", 0.0),
        "c1b_safe_for_team_colour_separation_review": safe_for_review,
        "c1b_safe_for_team_colour_separation_reason": "visual_proxy_clusters_separate_without_extra_unknown_dark_or_background_contamination" if safe_for_review else "No C1b profile is safe for team-colour separation yet; manual crop/label review is required.",
        "best_profile_summary": best_summary,
        "frames_needing_manual_followup": manual_frames,
        "gold8_cluster_confusion_rows": len(sweep_payload.get("gold8_cluster_confusion_rows", [])),
        "sandbox_only": True,
        "no_auto_promotion": True,
    }


def profile_eval_report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Step1.C1b Profile Eval Report",
        "",
        f"- Warning: `{VISUAL_ONLY_WARNING}`.",
        "- Scope: crop/prototype audit and visual team-colour belief diagnostics only.",
        "- Gold visible_person_type_gold is used only as visual QA/proxy context.",
        "- No team mapping, goalkeeper classification, official classification, identity, player slots, expected roles, physical metrics, tactical metrics, or football analysis was performed.",
        "- No profile was auto-promoted.",
        "",
        "## Why C1 Needed Audit",
        "",
        "Step1.C1 produced complete visual colour beliefs, but the Gold-8 proxy distribution showed team_1_player and team_2_player mostly collapsing into the same outfield colour cluster.",
        "",
        "## C1 Baseline",
        "",
        "```json",
        json.dumps(summary.get("c1_baseline", {}), indent=2),
        "```",
        "",
        "## Crop Profile Comparison",
        "",
        "| profile | strategy | rows | unknown | team_1 dominant | team_2 dominant | score | unknown proxy | dark proxy | promising |",
        "|---|---|---:|---:|---|---|---:|---:|---:|---|",
    ]
    for row in summary.get("profile_comparison_rows", []):
        lines.append(
            "| {profile_name} | {prototype_strategy} | {belief_rows} | {unknown_ambiguous_rows} | {team_1_proxy_dominant_cluster} | {team_2_proxy_dominant_cluster} | {team_1_team_2_cluster_separation_score} | {unknown_on_gold_player_proxy_count} | {dark_context_on_gold_player_proxy_count} | {profile_visually_promising} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Gold-8 Proxy Cluster Confusion",
            "",
            "The confusion rows are written separately in `step1c1b_gold8_cluster_confusion_rows.json`; only the Gold visible-person type field is used.",
            "",
            "## Best Sandbox Profile For Human Review",
            "",
            f"- Profile: `{summary.get('c1b_best_profile_name', '')}`",
            f"- Prototype strategy: `{summary.get('c1b_best_prototype_strategy', '')}`",
            f"- Unknown/ambiguous rows: {summary.get('c1b_best_unknown_ambiguous_colour_rows', 0)}",
            f"- Separation score: {summary.get('c1b_team_1_team_2_separation_score', 0.0)}",
            f"- Safe for team-colour separation review: {summary.get('c1b_safe_for_team_colour_separation_review', False)}",
            f"- Decision: {summary.get('c1b_safe_for_team_colour_separation_reason', '')}",
            "",
            "## Known Risks",
            "",
            "- Small far-side crops remain vulnerable to grass, compression, blur, and shadow.",
            "- Dark context colour remains colour evidence only and is not official/referee classification.",
            "- Other distinct colour remains colour evidence only and is not goalkeeper classification.",
            "- Unknown/context/off-ROI rows remain conservative and are not forced into team labels.",
        ]
    )
    return "\n".join(lines) + "\n"


def recommended_profile_markdown(summary: dict[str, Any]) -> str:
    safe = bool(summary.get("c1b_safe_for_team_colour_separation_review"))
    best = summary.get("best_profile_summary", {})
    lines = [
        "# Step1.C1b Recommended Profile For Human Review",
        "",
        f"- Warning: `{VISUAL_ONLY_WARNING}`.",
        f"- Chosen sandbox profile: `{summary.get('c1b_best_profile_name', '')}`" if safe else "- Chosen sandbox profile: `none safe`",
        f"- Diagnostic best profile: `{summary.get('c1b_best_profile_name', '')}`",
        f"- Prototype strategy: `{summary.get('c1b_best_prototype_strategy', '')}`",
        f"- No auto-promotion: {summary.get('no_auto_promotion', True)}",
        "",
        "## Reasons",
        "",
        f"- {summary.get('c1b_safe_for_team_colour_separation_reason', '')}",
        f"- Team 1 dominant cluster: {best.get('team_1_proxy_dominant_cluster', '')} purity={best.get('team_1_proxy_cluster_purity', 0.0)}",
        f"- Team 2 dominant cluster: {best.get('team_2_proxy_dominant_cluster', '')} purity={best.get('team_2_proxy_cluster_purity', 0.0)}",
        f"- Unknown on Gold player proxy: {best.get('unknown_on_gold_player_proxy_count', 0)}",
        f"- Dark context on Gold player proxy: {best.get('dark_context_on_gold_player_proxy_count', 0)}",
        "",
        "## Frames Needing Manual Follow-Up",
        "",
        json.dumps(summary.get("frames_needing_manual_followup", []), indent=2),
        "",
        "## Crop Examples Needing Manual Review",
        "",
        "See `step1c1b_crop_comparison_contact_sheet.jpg` and `step1c1b_cluster_crop_contact_sheet.jpg`.",
        "",
        "## Next Suggested Action",
        "",
        "Review the C1b contact sheets and decision template before allowing any colour profile into the next visual-only smoothing stage.",
    ]
    return "\n".join(lines) + "\n"


def build_and_write_profile_eval() -> dict[str, Any]:
    sweep_payload = read_json(STEP1C1B_COLOUR_PROFILE_SWEEP_PATH)
    c1_eval_summary = read_json(STEP1C1_GOLD8_COLOUR_EVAL_SUMMARY_PATH)
    best_summary, safe_for_review = select_best_strategy_summary(sweep_payload, c1_eval_summary)
    summary = build_profile_eval_summary(
        sweep_payload,
        c1_eval_summary,
        best_summary,
        safe_for_review=safe_for_review,
    )

    from football_intelligence.step1_visual_reconstruction.colour_profile_sweep import build_profile_sandbox_payloads

    base_payload = read_json(STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH)
    _features, _prototypes, belief_payload, unknown_payload = build_profile_sandbox_payloads(
        base_payload,
        str(best_summary.get("profile_name", "c1_current")),
        str(best_summary.get("prototype_strategy", "c1_top_chromatic")),
    )
    belief_payload["artifact"] = "step1c1b_profile_belief_rows_best_sandbox"
    belief_payload["best_sandbox_profile_for_human_review"] = True
    unknown_payload["artifact"] = "step1c1b_unknown_ambiguous_rows_best_sandbox"
    unknown_payload["best_sandbox_profile_for_human_review"] = True

    confusion_payload = {
        "artifact": "step1c1b_gold8_cluster_confusion_rows",
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
        "sandbox_only": True,
        "rows": sweep_payload.get("gold8_cluster_confusion_rows", []),
        "summary": {"gold8_cluster_confusion_rows": len(sweep_payload.get("gold8_cluster_confusion_rows", []))},
    }

    write_json(STEP1C1B_PROFILE_EVAL_SUMMARY_PATH, summary)
    write_text(STEP1C1B_PROFILE_EVAL_REPORT_PATH, profile_eval_report_markdown(summary))
    write_text(STEP1C1B_RECOMMENDED_PROFILE_PATH, recommended_profile_markdown(summary))
    write_json(STEP1C1B_BEST_SANDBOX_BELIEF_ROWS_PATH, belief_payload)
    write_json(STEP1C1B_BEST_SANDBOX_UNKNOWN_AMBIGUOUS_ROWS_PATH, unknown_payload)
    write_json(STEP1C1B_GOLD8_CLUSTER_CONFUSION_ROWS_PATH, confusion_payload)
    write_json(STEP1C1B_REVIEW_DECISION_TEMPLATE_PATH, step1c1b_review_decision_template_payload())
    return summary
