# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.paths import CLIP_ID, MATCH_ID, STAGE3C13_GOLD20_MANUAL_LABELS_PATH
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import (
    gold_visible_person_rows,
    load_completed_gold8_frames,
    strict_one_to_one_match,
)
from football_intelligence.step1_visual_reconstruction.io import (
    SOCCERTRACK_ROOT,
    STEP1C1C_SEEDED_COLOUR_EVAL_SUMMARY_PATH,
    STEP1C2_COLOUR_FLIP_AUDIT_ROWS_PATH,
    STEP1C2_COLOUR_STABILITY_REPORT_PATH,
    STEP1C2_COLOUR_STABILITY_ROWS_PATH,
    STEP1C2_GOLD8_COLOUR_STABILITY_EVAL_REPORT_PATH,
    STEP1C2_GOLD8_COLOUR_STABILITY_EVAL_SUMMARY_PATH,
    STEP1C2_GROUP_CROP_CONTACT_SHEET_PATH,
    STEP1C2_REVIEW_CONTACT_SHEET_PATH,
    STEP1C2_REVIEW_DECISION_TEMPLATE_PATH,
    STEP1C2_REVIEW_PACK_DIR,
    STEP1C2_REVIEW_PACK_MANIFEST_PATH,
    STEP1C2_SHORT_BURST_COLOUR_GROUP_ROWS_PATH,
    copy_binary_file,
    copy_text_file,
    read_json,
    write_json,
    write_text,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    FORBIDDEN_OUTPUT_KEYS,
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    safe_float,
)


GOLD_TEAM_COLOUR_PROXY_TYPES = {"team_1_player", "team_2_player"}
UNKNOWN_C2_BELIEFS = {"unknown_ambiguous_colour", "ambiguous_outfield_colour", "crop_unusable"}
TEAM_C2_BELIEFS = {"team_1_outfield_colour_like", "team_2_outfield_colour_like"}
C2_EXTRA_FORBIDDEN_KEYS = {"track_id", "persistent_player_id"}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def dominant(counts: dict[str, int]) -> tuple[str, int, float]:
    if not counts:
        return "", 0, 0.0
    total = sum(counts.values())
    label, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return label, count, round(count / max(1, total), 4)


def separation_score(distribution: dict[str, dict[str, int]], unknown_count: int, dark_count: int) -> float:
    team_1_label, _team_1_count, team_1_purity = dominant(distribution.get("team_1_player", {}))
    team_2_label, _team_2_count, team_2_purity = dominant(distribution.get("team_2_player", {}))
    proxy_total = sum(sum(counts.values()) for counts in distribution.values())
    if not team_1_label or not team_2_label:
        return 0.0
    if team_1_label != team_2_label and team_1_label not in UNKNOWN_C2_BELIEFS and team_2_label not in UNKNOWN_C2_BELIEFS:
        raw = (team_1_purity + team_2_purity) / 2.0
    elif team_1_label != team_2_label:
        raw = (team_1_purity + team_2_purity) / 4.0
    else:
        raw = abs(team_1_purity - team_2_purity) * 0.15
    return round(max(0.0, raw - (unknown_count / max(1, proxy_total) * 0.25) - (dark_count / max(1, proxy_total) * 0.25)), 4)


def c2_gold_proxy_rows(
    stability_payload: dict[str, Any],
    *,
    labels_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    labels_payload = labels_payload or read_json(STAGE3C13_GOLD20_MANUAL_LABELS_PATH)
    frame_sequences = {int(safe_float(frame.get("frame_sequence"), -1)) for frame in load_completed_gold8_frames(labels_payload)}
    stability_rows = [
        row
        for row in stability_payload.get("rows", [])
        if int(safe_float(row.get("frame_sequence"), -1)) in frame_sequences
    ]
    matches, _missed, _extra = strict_one_to_one_match(gold_visible_person_rows(labels_payload), stability_rows)
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
                "c1c_seed_team_colour_belief": candidate.get("c1c_seed_team_colour_belief", ""),
                "c2_stable_colour_belief": candidate.get("c2_stable_colour_belief", ""),
                "c2_stable_colour_belief_state": candidate.get("c2_stable_colour_belief_state", ""),
                "c2_stable_colour_belief_confidence": safe_float(candidate.get("c2_stable_colour_belief_confidence")),
                "c2_stability_action": candidate.get("c2_stability_action", ""),
                "short_burst_colour_group_id": candidate.get("short_burst_colour_group_id", ""),
                "bbox_iou": match.get("match_features", {}).get("bbox_iou", 0.0),
                "visual_gap_px": match.get("match_features", {}).get("visual_gap_px", 0.0),
                "visual_only_warning": VISUAL_ONLY_WARNING,
                "do_not_use_for_metrics": True,
                "production_ready": PRODUCTION_READY,
            }
        )
    return rows


def proxy_distribution(rows: list[dict[str, Any]], belief_key: str) -> dict[str, dict[str, int]]:
    distribution: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        distribution[str(row.get("visible_person_type_gold", ""))][str(row.get(belief_key, ""))] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(distribution.items())}


def one_frame_colour_diagnostics(stability_payload: dict[str, Any], labels_payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    frame_sequences = {int(safe_float(frame.get("frame_sequence"), -1)) for frame in load_completed_gold8_frames(labels_payload)}
    counts_by_frame: dict[int, Counter[str]] = defaultdict(Counter)
    review_by_frame: dict[int, int] = defaultdict(int)
    for row in stability_payload.get("rows", []):
        seq = int(safe_float(row.get("frame_sequence"), -1))
        if seq not in frame_sequences:
            continue
        counts_by_frame[seq][str(row.get("c2_stable_colour_belief", ""))] += 1
        if row.get("c2_review_required") is True:
            review_by_frame[seq] += 1
    diagnostics = []
    for seq, counts in sorted(counts_by_frame.items()):
        non_unknown = [key for key in counts if key not in UNKNOWN_C2_BELIEFS]
        if len(non_unknown) > 3 or review_by_frame.get(seq, 0) > 0:
            diagnostics.append(
                {
                    "frame_sequence": seq,
                    "distinct_non_unknown_c2_beliefs": len(non_unknown),
                    "c2_review_required_rows": review_by_frame.get(seq, 0),
                    "cluster_counts": dict(sorted(counts.items())),
                }
            )
    return diagnostics


def forbidden_keys_present(payload: dict[str, Any]) -> list[str]:
    banned = set(FORBIDDEN_OUTPUT_KEYS) | C2_EXTRA_FORBIDDEN_KEYS
    found: set[str] = set()
    for row in payload.get("rows", []):
        found.update(key for key in banned if key in row)
    return sorted(found)


def actual_team_flip_counts(stability_rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "team_1_to_team_2_actual_flip_count": sum(
            1
            for row in stability_rows
            if row.get("c1c_seed_team_colour_belief") == "team_1_outfield_colour_like"
            and row.get("c2_stable_colour_belief") == "team_2_outfield_colour_like"
        ),
        "team_2_to_team_1_actual_flip_count": sum(
            1
            for row in stability_rows
            if row.get("c1c_seed_team_colour_belief") == "team_2_outfield_colour_like"
            and row.get("c2_stable_colour_belief") == "team_1_outfield_colour_like"
        ),
    }


def c2_safety_missing_reasons(
    *,
    c1c_summary: dict[str, Any],
    stability_payload: dict[str, Any],
    group_payload: dict[str, Any],
    c2_separation: float,
    c2_unknown_count: int,
    c2_dark_count: int,
    context_forced_count: int,
    team_flips: dict[str, int],
    forbidden_keys: list[str],
) -> list[str]:
    reasons = []
    c1c_count = int(c1c_summary.get("seeded_belief_row_count", 0))
    c2_count = len(stability_payload.get("rows", []))
    if c1c_count != c2_count:
        reasons.append("c2_row_count_does_not_match_c1c_seeded_belief_row_count")
    if not stability_payload.get("summary", {}).get("one_row_per_c1c_seeded_belief_row", False):
        reasons.append("c2_not_one_row_per_c1c_seeded_belief_row")
    if context_forced_count != 0:
        reasons.append("context_or_offroi_rows_forced_to_team_colour")
    if forbidden_keys:
        reasons.append("forbidden_identity_slot_metric_keys_present")
    if team_flips.get("team_1_to_team_2_actual_flip_count", 0) != 0:
        reasons.append("team_1_to_team_2_flip_detected")
    if team_flips.get("team_2_to_team_1_actual_flip_count", 0) != 0:
        reasons.append("team_2_to_team_1_flip_detected")
    c1c_separation = safe_float(c1c_summary.get("seeded_team_1_team_2_separation_score"))
    if c2_separation < c1c_separation - 0.03:
        reasons.append("c2_gold_proxy_separation_materially_regressed")
    if c2_unknown_count > int(c1c_summary.get("unknown_on_gold_player_proxy_count", 0)):
        reasons.append("c2_unknown_on_gold_player_proxy_count_increased")
    if c2_dark_count > int(c1c_summary.get("dark_context_on_gold_player_proxy_count", 0)):
        reasons.append("c2_dark_context_on_gold_player_proxy_count_increased")
    if not group_payload.get("rows"):
        reasons.append("short_burst_colour_groups_absent")
    return reasons


def build_colour_stability_eval_summary(
    c1c_summary: dict[str, Any],
    group_payload: dict[str, Any],
    stability_payload: dict[str, Any],
    flip_payload: dict[str, Any],
    *,
    labels_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    proxy_rows = c2_gold_proxy_rows(stability_payload, labels_payload=labels_payload)
    c2_distribution = proxy_distribution(proxy_rows, "c2_stable_colour_belief")
    c2_unknown_count = sum(1 for row in proxy_rows if row.get("c2_stable_colour_belief") in UNKNOWN_C2_BELIEFS)
    c2_dark_count = sum(1 for row in proxy_rows if row.get("c2_stable_colour_belief") == "dark_context_colour_like")
    c2_separation = separation_score(c2_distribution, c2_unknown_count, c2_dark_count)
    context_forced_count = int(stability_payload.get("summary", {}).get("context_offroi_forced_to_team_count", 0))
    team_flips = actual_team_flip_counts(stability_payload.get("rows", []))
    forbidden_keys = forbidden_keys_present(stability_payload)
    diagnostics = one_frame_colour_diagnostics(stability_payload, labels_payload=labels_payload)
    missing = c2_safety_missing_reasons(
        c1c_summary=c1c_summary,
        stability_payload=stability_payload,
        group_payload=group_payload,
        c2_separation=c2_separation,
        c2_unknown_count=c2_unknown_count,
        c2_dark_count=c2_dark_count,
        context_forced_count=context_forced_count,
        team_flips=team_flips,
        forbidden_keys=forbidden_keys,
    )
    c2_safe = not missing
    team_1_label, team_1_count, team_1_purity = dominant(c2_distribution.get("team_1_player", {}))
    team_2_label, team_2_count, team_2_purity = dominant(c2_distribution.get("team_2_player", {}))
    followup_frames = sorted(set(c1c_summary.get("frames_needing_manual_followup", [])) | {item["frame_sequence"] for item in diagnostics[:12]})
    return {
        "artifact": "step1c2_gold8_colour_stability_eval_summary",
        "created_at": utc_iso(),
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
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
        "c1c_seeded_belief_row_count": int(c1c_summary.get("seeded_belief_row_count", 0)),
        "c2_stability_row_count": len(stability_payload.get("rows", [])),
        "one_row_per_c1c_seeded_belief_row": stability_payload.get("summary", {}).get("one_row_per_c1c_seeded_belief_row", False),
        "short_burst_colour_group_count": len(group_payload.get("rows", [])),
        "c1c_proxy_distribution": c1c_summary.get("seeded_team_1_team_2_gold_proxy_distribution", {}),
        "c2_proxy_distribution": c2_distribution,
        "c1c_separation_score": safe_float(c1c_summary.get("seeded_team_1_team_2_separation_score")),
        "c2_separation_score": c2_separation,
        "c2_team_1_proxy_dominant_belief": team_1_label,
        "c2_team_2_proxy_dominant_belief": team_2_label,
        "c2_team_1_proxy_dominant_count": team_1_count,
        "c2_team_2_proxy_dominant_count": team_2_count,
        "c2_team_1_proxy_purity": team_1_purity,
        "c2_team_2_proxy_purity": team_2_purity,
        "c1c_unknown_on_gold_player_proxy_count": int(c1c_summary.get("unknown_on_gold_player_proxy_count", 0)),
        "c2_unknown_on_gold_player_proxy_count": c2_unknown_count,
        "c1c_dark_context_on_gold_player_proxy_count": int(c1c_summary.get("dark_context_on_gold_player_proxy_count", 0)),
        "c2_dark_context_on_gold_player_proxy_count": c2_dark_count,
        "context_offroi_forced_to_team_count": context_forced_count,
        **team_flips,
        "flip_type_counts": flip_payload.get("summary", {}).get("flip_type_counts", {}),
        "c2_stability_action_counts": stability_payload.get("summary", {}).get("c2_stability_action_counts", {}),
        "c2_stable_belief_counts": stability_payload.get("summary", {}).get("c2_stable_belief_counts", {}),
        "one_frame_colour_contradiction_diagnostics": diagnostics[:20],
        "frames_needing_manual_followup": followup_frames,
        "c2_safe_for_human_review": c2_safe,
        "c2_safety_missing_reasons": missing,
        "c2_safety_message": "Step1.C2 colour stability sandbox is safe for human visual review." if c2_safe else "Step1.C2 colour stability sandbox needs conservative review before any next-stage use.",
        "gold_proxy_note": "Gold visible_person_type_gold is used only as visual colour QA proxy context.",
        "no_auto_promotion": True,
    }


def colour_stability_eval_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.C2 Gold-8 Colour Stability Eval Report",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            "- Scope: short-burst visual colour-stability QA only.",
            "- Gold visible_person_type_gold is used only as visual QA/proxy context.",
            "- No team mapping was auto-promoted.",
            "",
            "## Row Count And Safety",
            "",
            f"- C1c seeded rows: {summary.get('c1c_seeded_belief_row_count', 0)}",
            f"- C2 stability rows: {summary.get('c2_stability_row_count', 0)}",
            f"- One row per C1c row: {summary.get('one_row_per_c1c_seeded_belief_row', False)}",
            f"- Short-burst colour groups: {summary.get('short_burst_colour_group_count', 0)}",
            f"- Context/off-ROI forced to team: {summary.get('context_offroi_forced_to_team_count', 0)}",
            "",
            "## Gold-8 Proxy Comparison",
            "",
            f"- C1c proxy distribution: {summary.get('c1c_proxy_distribution', {})}",
            f"- C2 proxy distribution: {summary.get('c2_proxy_distribution', {})}",
            f"- C1c separation score: {summary.get('c1c_separation_score', 0.0)}",
            f"- C2 separation score: {summary.get('c2_separation_score', 0.0)}",
            f"- C1c unknown-on-gold player proxy: {summary.get('c1c_unknown_on_gold_player_proxy_count', 0)}",
            f"- C2 unknown-on-gold player proxy: {summary.get('c2_unknown_on_gold_player_proxy_count', 0)}",
            f"- C1c dark-context-on-gold player proxy: {summary.get('c1c_dark_context_on_gold_player_proxy_count', 0)}",
            f"- C2 dark-context-on-gold player proxy: {summary.get('c2_dark_context_on_gold_player_proxy_count', 0)}",
            "",
            "## Flip Audit",
            "",
            "```json",
            json.dumps(summary.get("flip_type_counts", {}), indent=2),
            "```",
            "",
            "## Manual Follow-Up Frames",
            "",
            "```json",
            json.dumps(summary.get("frames_needing_manual_followup", []), indent=2),
            "```",
            "",
            "## Recommendation",
            "",
            summary.get("c2_safety_message", ""),
            "",
            "## Safety Missing Reasons",
            "",
            "```json",
            json.dumps(summary.get("c2_safety_missing_reasons", []), indent=2),
            "```",
        ]
    ) + "\n"


def colour_stability_report(
    stability_payload: dict[str, Any],
    group_payload: dict[str, Any],
    flip_payload: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# Step1.C2 Colour Stability Sandbox Report",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            "- Short-burst colour groups are local visual review helpers, not identities.",
            "- No player slots, expected roles, goalkeeper classification, official/referee exclusion, tracking, or metrics are created.",
            "",
            "## Short-Burst Grouping",
            "",
            f"- Group rows: {len(group_payload.get('rows', []))}",
            f"- Max frame gap: {group_payload.get('max_frame_gap', 2)}",
            f"- Max group span frames: {group_payload.get('max_group_span_frames', 7)}",
            f"- Singleton groups: {group_payload.get('summary', {}).get('singleton_group_count', 0)}",
            f"- Multi-row groups: {group_payload.get('summary', {}).get('multi_row_group_count', 0)}",
            "",
            "## Stability Policy",
            "",
            f"- Stability rows: {len(stability_payload.get('rows', []))}",
            f"- Action counts: {stability_payload.get('summary', {}).get('c2_stability_action_counts', {})}",
            f"- Stable belief counts: {stability_payload.get('summary', {}).get('c2_stable_belief_counts', {})}",
            f"- Review-required rows: {stability_payload.get('summary', {}).get('c2_review_required_count', 0)}",
            "",
            "## Flip Audit",
            "",
            f"- Flip rows: {len(flip_payload.get('rows', []))}",
            f"- Flip type counts: {flip_payload.get('summary', {}).get('flip_type_counts', {})}",
            "",
            "## Human Review Safety",
            "",
            f"- Safe for human review: {summary.get('c2_safe_for_human_review', False)}",
            f"- Safety missing reasons: {summary.get('c2_safety_missing_reasons', [])}",
        ]
    ) + "\n"


def review_decision_template_payload() -> dict[str, Any]:
    return {
        "reviewer_name": "",
        "reviewed_at": "",
        "review_contact_sheet_reviewed": False,
        "group_crop_sheet_reviewed": False,
        "approve_c2_colour_stability_for_next_stage": False,
        "approve_any_team_colour_mapping": False,
        "known_issues": [],
        "frames_requiring_manual_followup": [],
        "notes": "",
        "production_ready": PRODUCTION_READY,
        "no_auto_promotion": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
    }


def build_and_write_colour_stability_eval() -> dict[str, Any]:
    c1c_summary = read_json(STEP1C1C_SEEDED_COLOUR_EVAL_SUMMARY_PATH)
    group_payload = read_json(STEP1C2_SHORT_BURST_COLOUR_GROUP_ROWS_PATH)
    stability_payload = read_json(STEP1C2_COLOUR_STABILITY_ROWS_PATH)
    flip_payload = read_json(STEP1C2_COLOUR_FLIP_AUDIT_ROWS_PATH)
    summary = build_colour_stability_eval_summary(c1c_summary, group_payload, stability_payload, flip_payload)
    write_json(STEP1C2_GOLD8_COLOUR_STABILITY_EVAL_SUMMARY_PATH, summary)
    write_text(STEP1C2_GOLD8_COLOUR_STABILITY_EVAL_REPORT_PATH, colour_stability_eval_report(summary))
    write_text(STEP1C2_COLOUR_STABILITY_REPORT_PATH, colour_stability_report(stability_payload, group_payload, flip_payload, summary))
    write_json(STEP1C2_REVIEW_DECISION_TEMPLATE_PATH, review_decision_template_payload())
    return summary


def sample_payload(path: Path, row_limit: int, artifact: str) -> dict[str, Any]:
    payload = read_json(path)
    rows = payload.get("rows", [])[:row_limit]
    return {
        "artifact": artifact,
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "sample_rows": len(rows),
        "total_rows": len(payload.get("rows", [])),
        "summary": payload.get("summary", {}),
        "rows": rows,
    }


def review_index_text(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.C2 Review Index",
            "",
            f"- C1c seeded rows: {summary.get('c1c_seeded_belief_row_count', 0)}",
            f"- C2 stability rows: {summary.get('c2_stability_row_count', 0)}",
            f"- Short-burst groups: {summary.get('short_burst_colour_group_count', 0)}",
            f"- C1c separation score: {summary.get('c1c_separation_score', 0.0)}",
            f"- C2 separation score: {summary.get('c2_separation_score', 0.0)}",
            f"- Safe for human review: {summary.get('c2_safe_for_human_review', False)}",
            f"- Visual flag: {VISUAL_ONLY_WARNING}",
            "- production_ready=false",
            "- no_auto_promotion=true",
        ]
    ) + "\n"


def scope_text() -> str:
    return "\n".join(
        [
            "# Step1.C2 Scope And Restrictions",
            "",
            "Step1.C2 is a short-burst seeded colour-stability sandbox.",
            "",
            "- Short-burst groups are local and temporary visual QA groups, not identities.",
            "- Do not use the groups for player tracking, player metrics, tactical conclusions, or role assignment.",
            "- No C2 output is canonical.",
            "- No team mapping is auto-promoted.",
            "- No goalkeeper classification or official/referee specialist exclusion is performed.",
            "- No identity tracking, player slots, or expected 22-role states are created.",
            "- No football, physical, tactical, speed, distance, fatigue, player-load, pass, dribble, or team-shape metrics are calculated.",
            "- Stage 3D registries and project-wide defaults remain unchanged.",
        ]
    ) + "\n"


def tests_added_text() -> str:
    return "\n".join(
        [
            "# Step1.C2 Tests Added",
            "",
            "- `tests/test_step1c2_colour_stability_groups.py` covers local visual grouping evidence, max span, temporary group ids, and non-identity flags.",
            "- `tests/test_step1c2_colour_stability_policy.py` covers one-row-per-input, no context/off-ROI force, retained context/other rows, and conflict review behavior.",
            "- `tests/test_step1c2_colour_stability_eval.py` covers Gold proxy usage, preserved row count, and emitted C1c/C2 comparison fields.",
            "- `tests/test_step1c2_restrictions.py` covers forbidden keys, promotion imports, registry/default flags, and production_ready=false.",
        ]
    ) + "\n"


def clear_review_pack_dir() -> None:
    STEP1C2_REVIEW_PACK_DIR.mkdir(parents=True, exist_ok=True)
    for path in STEP1C2_REVIEW_PACK_DIR.iterdir():
        if path.is_file():
            path.unlink()


def build_step1c2_review_pack() -> dict[str, Any]:
    clear_review_pack_dir()
    summary = read_json(STEP1C2_GOLD8_COLOUR_STABILITY_EVAL_SUMMARY_PATH)
    entries: list[dict[str, Any]] = []

    def add_entry(name: str, description: str, kind: str) -> Path:
        path = STEP1C2_REVIEW_PACK_DIR / name
        entries.append({"name": name, "kind": kind, "description": description, "path": str(path.resolve())})
        return path

    write_text(add_entry("00_REVIEW_INDEX.md", "C2 review starting point.", "markdown"), review_index_text(summary))
    write_text(add_entry("01_SCOPE_AND_RESTRICTIONS.md", "C2 scope guardrails.", "markdown"), scope_text())
    write_json(add_entry("02_C2_EVAL_SUMMARY.json", "C2 Gold-8 visual QA summary.", "json"), summary)
    copy_text_file(STEP1C2_GOLD8_COLOUR_STABILITY_EVAL_REPORT_PATH, add_entry("03_C2_EVAL_REPORT.md", "C2 eval report.", "markdown"))
    copy_text_file(STEP1C2_COLOUR_STABILITY_REPORT_PATH, add_entry("04_C2_STABILITY_REPORT.md", "C2 stability report.", "markdown"))
    write_json(add_entry("05_SHORT_BURST_GROUP_SAMPLE.json", "Sample of short-burst group rows.", "json"), sample_payload(STEP1C2_SHORT_BURST_COLOUR_GROUP_ROWS_PATH, 80, "step1c2_short_burst_group_sample"))
    write_json(add_entry("06_STABILITY_ROWS_SAMPLE.json", "Sample of C2 stability rows.", "json"), sample_payload(STEP1C2_COLOUR_STABILITY_ROWS_PATH, 80, "step1c2_stability_rows_sample"))
    write_json(add_entry("07_FLIP_AUDIT_SAMPLE.json", "Sample of C2 flip audit rows.", "json"), sample_payload(STEP1C2_COLOUR_FLIP_AUDIT_ROWS_PATH, 80, "step1c2_flip_audit_sample"))
    copy_binary_file(STEP1C2_REVIEW_CONTACT_SHEET_PATH, add_entry("08_REVIEW_CONTACT_SHEET.jpg", "C2 multi-panel review contact sheet.", "image"))
    copy_binary_file(STEP1C2_GROUP_CROP_CONTACT_SHEET_PATH, add_entry("09_GROUP_CROP_CONTACT_SHEET.jpg", "C2 grouped crop contact sheet.", "image"))
    write_json(add_entry("10_REVIEW_DECISION_TEMPLATE.json", "C2 review decision template.", "json"), read_json(STEP1C2_REVIEW_DECISION_TEMPLATE_PATH))
    code_files = [
        ("11_colour_stability_groups.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_stability_groups.py", "C2 short-burst grouping."),
        ("12_colour_stability_policy.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_stability_policy.py", "C2 stability policy."),
        ("13_colour_stability_eval.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_stability_eval.py", "C2 eval and review pack."),
        ("14_colour_stability_render.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_stability_render.py", "C2 visual renderers."),
        ("15_SCRIPT_BUILD_GROUPS.py", SOCCERTRACK_ROOT / "scripts" / "step1c2_build_short_burst_colour_groups.py", "C2 build groups script."),
        ("16_SCRIPT_APPLY_POLICY.py", SOCCERTRACK_ROOT / "scripts" / "step1c2_apply_colour_stability_policy.py", "C2 apply policy script."),
        ("17_SCRIPT_EVAL.py", SOCCERTRACK_ROOT / "scripts" / "step1c2_evaluate_colour_stability_gold8.py", "C2 eval script."),
    ]
    for name, source, description in code_files:
        copy_text_file(source, add_entry(name, description, "python"))
    write_text(add_entry("18_TESTS_ADDED.md", "Summary of C2 tests.", "markdown"), tests_added_text())
    manifest_path = add_entry("19_REVIEW_PACK_MANIFEST.json", "C2 review pack manifest.", "json")
    manifest = {
        "artifact": "step1c2_review_pack_manifest",
        "created_at": utc_iso(),
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
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
        "outputs": {
            "step1c2_short_burst_colour_group_rows_path": str(STEP1C2_SHORT_BURST_COLOUR_GROUP_ROWS_PATH.resolve()),
            "step1c2_colour_stability_rows_path": str(STEP1C2_COLOUR_STABILITY_ROWS_PATH.resolve()),
            "step1c2_colour_flip_audit_rows_path": str(STEP1C2_COLOUR_FLIP_AUDIT_ROWS_PATH.resolve()),
            "step1c2_gold8_colour_stability_eval_summary_path": str(STEP1C2_GOLD8_COLOUR_STABILITY_EVAL_SUMMARY_PATH.resolve()),
            "step1c2_colour_stability_report_path": str(STEP1C2_COLOUR_STABILITY_REPORT_PATH.resolve()),
            "step1c2_review_contact_sheet_path": str(STEP1C2_REVIEW_CONTACT_SHEET_PATH.resolve()),
            "step1c2_group_crop_contact_sheet_path": str(STEP1C2_GROUP_CROP_CONTACT_SHEET_PATH.resolve()),
            "step1c2_review_pack_manifest_path": str(STEP1C2_REVIEW_PACK_MANIFEST_PATH.resolve()),
        },
        "summary": summary,
        "review_pack_file_count": len(entries),
        "review_pack_file_limit": 20,
        "review_pack_entries": entries,
    }
    write_json(manifest_path, manifest)
    write_json(STEP1C2_REVIEW_PACK_MANIFEST_PATH, manifest)
    if len(entries) > 20:
        raise RuntimeError(f"Step1.C2 review pack contains {len(entries)} files; maximum is 20.")
    return manifest


def print_step1c2_final_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    summary = manifest["summary"]
    print(f"step1c2_short_burst_colour_group_rows_path: {outputs['step1c2_short_burst_colour_group_rows_path']}")
    print(f"step1c2_colour_stability_rows_path: {outputs['step1c2_colour_stability_rows_path']}")
    print(f"step1c2_colour_flip_audit_rows_path: {outputs['step1c2_colour_flip_audit_rows_path']}")
    print(f"step1c2_gold8_colour_stability_eval_summary_path: {outputs['step1c2_gold8_colour_stability_eval_summary_path']}")
    print(f"step1c2_colour_stability_report_path: {outputs['step1c2_colour_stability_report_path']}")
    print(f"step1c2_review_contact_sheet_path: {outputs['step1c2_review_contact_sheet_path']}")
    print(f"step1c2_group_crop_contact_sheet_path: {outputs['step1c2_group_crop_contact_sheet_path']}")
    print(f"step1c2_review_pack_manifest_path: {outputs['step1c2_review_pack_manifest_path']}")
    print(f"c1c_seeded_belief_row_count: {summary.get('c1c_seeded_belief_row_count', 0)}")
    print(f"c2_stability_row_count: {summary.get('c2_stability_row_count', 0)}")
    print(f"c1c_separation_score: {summary.get('c1c_separation_score', 0.0)}")
    print(f"c2_separation_score: {summary.get('c2_separation_score', 0.0)}")
    print(f"c1c_unknown_on_gold_player_proxy_count: {summary.get('c1c_unknown_on_gold_player_proxy_count', 0)}")
    print(f"c2_unknown_on_gold_player_proxy_count: {summary.get('c2_unknown_on_gold_player_proxy_count', 0)}")
    print(f"context_offroi_forced_to_team_count: {summary.get('context_offroi_forced_to_team_count', 0)}")
    print(f"short_burst_colour_group_count: {summary.get('short_burst_colour_group_count', 0)}")
    print(f"c2_safe_for_human_review={str(summary.get('c2_safe_for_human_review', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("goalkeeper_classification_performed=false")
    print("official_specialist_exclusion_performed=false")
