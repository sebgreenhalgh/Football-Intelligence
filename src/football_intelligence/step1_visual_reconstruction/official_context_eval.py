# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.paths import CLIP_ID, MATCH_ID, STAGE3C13_GOLD20_MANUAL_LABELS_PATH
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import (
    gold_excluded_nonperson_rows,
    gold_visible_person_rows,
    strict_one_to_one_match,
)
from football_intelligence.step1_visual_reconstruction.official_context_beliefs import OFFICIAL_LIKE_BELIEFS
from football_intelligence.step1_visual_reconstruction.schema import (
    FORBIDDEN_OUTPUT_KEYS,
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
)
from football_intelligence.step1_visual_reconstruction.io import (
    SOCCERTRACK_ROOT,
    STEP1C2C_GOLD8_HUMAN_CORRECTED_COLOUR_EVAL_SUMMARY_PATH,
    STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH,
    STEP1D1_CONTEXT_CROP_CONTACT_SHEET_PATH,
    STEP1D1_GOLD8_OFFICIAL_CONTEXT_EVAL_REPORT_PATH,
    STEP1D1_GOLD8_OFFICIAL_CONTEXT_EVAL_SUMMARY_PATH,
    STEP1D1_OFFICIAL_CONTEXT_BELIEF_REPORT_PATH,
    STEP1D1_OFFICIAL_CONTEXT_BELIEF_ROWS_PATH,
    STEP1D1_OFFICIAL_CONTEXT_FEATURE_ROWS_PATH,
    STEP1D1_OFFICIAL_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH,
    STEP1D1_REVIEW_CONTACT_SHEET_PATH,
    STEP1D1_REVIEW_DECISION_TEMPLATE_PATH,
    STEP1D1_REVIEW_PACK_DIR,
    STEP1D1_REVIEW_PACK_MANIFEST_PATH,
    copy_binary_file,
    copy_text_file,
    read_json,
    write_json,
    write_text,
)


D1_EXTRA_FORBIDDEN_KEYS = {
    "track_id",
    "persistent_player_id",
    "official_exclusion",
    "official_exclusion_reason",
    "exclude_from_player_review",
    "excluded_from_player_review",
    "excluded_from_player_team_review",
    "goalkeeper_classification",
    "goalkeeper_role",
}
D1_FORBIDDEN_KEYS = set(FORBIDDEN_OUTPUT_KEYS) | D1_EXTRA_FORBIDDEN_KEYS
NON_OFFICIAL_PLAYER_PROXY_TYPES = {"team_1_player", "team_2_player", "gk_team_1", "gk_team_2"}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def proxy_group(visible_type: str) -> str:
    if visible_type == "official_referee":
        return "official_proxy"
    if visible_type in NON_OFFICIAL_PLAYER_PROXY_TYPES:
        return "non_official_visible_player_proxy"
    if visible_type == "unknown_player":
        return "unknown_player_proxy"
    if visible_type == "off_pitch_person":
        return "off_pitch_context_proxy"
    return "other_visible_person_proxy"


def gold_proxy_matches(
    belief_payload: dict[str, Any],
    *,
    labels_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    labels_payload = labels_payload or read_json(STAGE3C13_GOLD20_MANUAL_LABELS_PATH)
    matches, _missed, _extra = strict_one_to_one_match(gold_visible_person_rows(labels_payload), belief_payload.get("rows", []))
    out = []
    for match in matches:
        gold = match["gold"]
        row = match["candidate"]
        visible_type = str(gold.get("visible_person_type_gold", ""))
        out.append(
            {
                "gold_row_id": gold.get("gold_row_id", ""),
                "visible_person_type_gold": visible_type,
                "proxy_group": proxy_group(visible_type),
                "visible_person_base_id": row.get("visible_person_base_id", ""),
                "frame_sequence": row.get("frame_sequence", -1),
                "official_context_belief": row.get("official_context_belief", ""),
                "official_context_review_required": row.get("official_context_review_required", False),
                "review_candidate": False,
                "visual_only_warning": VISUAL_ONLY_WARNING,
                "do_not_use_for_metrics": True,
                "production_ready": PRODUCTION_READY,
            }
        )
    return out


def bad_detection_proxy_matches(
    belief_payload: dict[str, Any],
    *,
    labels_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    labels_payload = labels_payload or read_json(STAGE3C13_GOLD20_MANUAL_LABELS_PATH)
    matches, _missed, _extra = strict_one_to_one_match(gold_excluded_nonperson_rows(labels_payload), belief_payload.get("rows", []))
    return [
        {
            "gold_row_id": match["gold"].get("gold_row_id", ""),
            "visible_person_type_gold": match["gold"].get("visible_person_type_gold", ""),
            "visible_person_base_id": match["candidate"].get("visible_person_base_id", ""),
            "official_context_belief": match["candidate"].get("official_context_belief", ""),
            "visual_only_warning": VISUAL_ONLY_WARNING,
            "do_not_use_for_metrics": True,
            "production_ready": PRODUCTION_READY,
        }
        for match in matches
    ]


def distribution_by_proxy(matches: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    distribution: dict[str, Counter[str]] = defaultdict(Counter)
    for match in matches:
        distribution[str(match.get("proxy_group", ""))][str(match.get("official_context_belief", ""))] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(distribution.items())}


def forbidden_keys_present(*payloads: dict[str, Any]) -> list[str]:
    found: set[str] = set()
    for payload in payloads:
        for row in payload.get("rows", []):
            found.update(key for key in D1_FORBIDDEN_KEYS if key in row)
    return sorted(found)


def safety_missing_reasons(
    *,
    c2c_payload: dict[str, Any],
    feature_payload: dict[str, Any],
    belief_payload: dict[str, Any],
    review_payload: dict[str, Any],
    c2c_summary: dict[str, Any],
    proxy_matches: list[dict[str, Any]],
    forbidden_keys: list[str],
) -> list[str]:
    reasons = []
    c2c_count = len(c2c_payload.get("rows", []))
    feature_count = len(feature_payload.get("rows", []))
    belief_count = len(belief_payload.get("rows", []))
    if c2c_count != 10418 or feature_count != 10418 or belief_count != 10418:
        reasons.append("d1_row_counts_not_10418")
    if c2c_count != feature_count:
        reasons.append("one_feature_row_per_c2c_row_not_preserved")
    if c2c_count != belief_count:
        reasons.append("one_belief_row_per_c2c_row_not_preserved")
    c2c_ids = sorted(str(row.get("visible_person_base_id", "")) for row in c2c_payload.get("rows", []))
    belief_ids = sorted(str(row.get("visible_person_base_id", "")) for row in belief_payload.get("rows", []))
    if c2c_ids != belief_ids:
        reasons.append("d1_visible_person_base_ids_do_not_match_c2c")
    if forbidden_keys:
        reasons.append("forbidden_identity_slot_metric_or_exclusion_keys_present")
    if any(row.get("retained_for_future_player_team_review") is not True for row in belief_payload.get("rows", [])):
        reasons.append("not_all_rows_retained_for_future_player_team_review")
    if belief_payload.get("production_ready") is not False or any(row.get("production_ready") is not False for row in belief_payload.get("rows", [])):
        reasons.append("production_ready_not_false")
    for flag in ["project_wide_defaults_changed", "stage3d_registries_changed", "identity_tracking_performed", "player_slots_assigned", "expected_22_role_states_created", "goalkeeper_classification_performed", "official_specialist_exclusion_performed"]:
        if belief_payload.get(flag) is not False:
            reasons.append(f"{flag}_not_false")
    c2c_context_override_count = int(c2c_summary.get("context_offroi_human_team_override_count", 0))
    d1_context_override_count = int(belief_payload.get("summary", {}).get("c2c_context_offroi_human_team_override_count", 0))
    if c2c_context_override_count != d1_context_override_count:
        reasons.append("c2c_context_offroi_human_team_override_count_not_preserved")
    review_ids = {str(row.get("visible_person_base_id", "")) for row in review_payload.get("rows", [])}
    official_matches = [match for match in proxy_matches if match.get("proxy_group") == "official_proxy"]
    surfaced = sum(1 for match in official_matches if str(match.get("visible_person_base_id", "")) in review_ids)
    if official_matches and surfaced / max(1, len(official_matches)) < 0.80:
        reasons.append("gold_official_proxy_rows_not_mostly_surfaced_as_review_candidates")
    non_official_player_matches = [match for match in proxy_matches if match.get("proxy_group") == "non_official_visible_player_proxy"]
    official_like_false_positive = sum(1 for match in non_official_player_matches if match.get("official_context_belief") in OFFICIAL_LIKE_BELIEFS)
    if non_official_player_matches and official_like_false_positive / max(1, len(non_official_player_matches)) > 0.20:
        reasons.append("normal_player_proxy_rows_broadly_mislabelled_official_like")
    return reasons


def build_official_context_eval_summary(
    c2c_payload: dict[str, Any],
    feature_payload: dict[str, Any],
    belief_payload: dict[str, Any],
    review_payload: dict[str, Any],
    c2c_summary: dict[str, Any],
    *,
    labels_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    labels_payload = labels_payload or read_json(STAGE3C13_GOLD20_MANUAL_LABELS_PATH)
    proxy_matches = gold_proxy_matches(belief_payload, labels_payload=labels_payload)
    bad_matches = bad_detection_proxy_matches(belief_payload, labels_payload=labels_payload)
    review_ids = {str(row.get("visible_person_base_id", "")) for row in review_payload.get("rows", [])}
    for match in proxy_matches:
        match["review_candidate"] = str(match.get("visible_person_base_id", "")) in review_ids
    proxy_dist = distribution_by_proxy(proxy_matches)
    official_matches = [match for match in proxy_matches if match.get("proxy_group") == "official_proxy"]
    non_official_player_matches = [match for match in proxy_matches if match.get("proxy_group") == "non_official_visible_player_proxy"]
    official_like_false_positive = sum(1 for match in non_official_player_matches if match.get("official_context_belief") in OFFICIAL_LIKE_BELIEFS)
    player_like_official_missed = sum(1 for match in official_matches if match.get("official_context_belief") == "player_like_not_official_context")
    bad_detection_proxy_count = sum(1 for match in bad_matches if match.get("official_context_belief") == "bad_detection_or_not_person")
    forbidden_keys = forbidden_keys_present(feature_payload, belief_payload, review_payload)
    missing = safety_missing_reasons(
        c2c_payload=c2c_payload,
        feature_payload=feature_payload,
        belief_payload=belief_payload,
        review_payload=review_payload,
        c2c_summary=c2c_summary,
        proxy_matches=proxy_matches,
        forbidden_keys=forbidden_keys,
    )
    safe = not missing
    return {
        "artifact": "step1d1_gold8_official_context_eval_summary",
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
        "total_c2c_rows": len(c2c_payload.get("rows", [])),
        "total_d1_feature_rows": len(feature_payload.get("rows", [])),
        "total_d1_belief_rows": len(belief_payload.get("rows", [])),
        "d1_review_candidate_count": len(review_payload.get("rows", [])),
        "official_context_belief_counts": belief_payload.get("summary", {}).get("official_context_belief_counts", {}),
        "review_required_count": belief_payload.get("summary", {}).get("review_required_count", 0),
        "source_official_candidate_count": belief_payload.get("summary", {}).get("source_official_candidate_count", 0),
        "c2c_context_offroi_human_team_override_count": belief_payload.get("summary", {}).get("c2c_context_offroi_human_team_override_count", 0),
        "gold8_official_proxy_rows": sum(1 for row in gold_visible_person_rows(labels_payload) if row.get("visible_person_type_gold") == "official_referee"),
        "gold8_official_proxy_matched_rows": len(official_matches),
        "official_proxy_d1_belief_distribution": proxy_dist.get("official_proxy", {}),
        "non_official_player_proxy_d1_belief_distribution": proxy_dist.get("non_official_visible_player_proxy", {}),
        "off_pitch_context_proxy_d1_belief_distribution": proxy_dist.get("off_pitch_context_proxy", {}),
        "unknown_player_proxy_d1_belief_distribution": proxy_dist.get("unknown_player_proxy", {}),
        "official_like_false_positive_proxy_count": official_like_false_positive,
        "player_like_official_missed_proxy_count": player_like_official_missed,
        "bad_detection_proxy_count": bad_detection_proxy_count,
        "bad_detection_proxy_distribution": dict(sorted(Counter(str(match.get("official_context_belief", "")) for match in bad_matches).items())),
        "gold_proxy_note": "Gold visible_person_type_gold is used only as visual official/context QA proxy context.",
        "forbidden_keys_present": forbidden_keys,
        "d1_safe_for_human_review_candidate": safe,
        "d1_safety_missing_reasons": missing,
        "d1_safety_message": "Step1.D1 official/context beliefs are safe for human visual review candidate use." if safe else "Step1.D1 official/context beliefs need review before next-stage candidate use.",
        "no_auto_promotion": True,
    }


def official_context_eval_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.D1 Gold-8 Official/Context Eval Report",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            "- Gold labels are used only as visual QA/proxy context.",
            "- D1 is not an official/referee exclusion stage.",
            "",
            "## Row Counts",
            "",
            f"- C2c rows: {summary.get('total_c2c_rows', 0)}",
            f"- D1 feature rows: {summary.get('total_d1_feature_rows', 0)}",
            f"- D1 belief rows: {summary.get('total_d1_belief_rows', 0)}",
            f"- Review candidates: {summary.get('d1_review_candidate_count', 0)}",
            "",
            "## Belief Counts",
            "",
            "```json",
            json.dumps(summary.get("official_context_belief_counts", {}), indent=2),
            "```",
            "",
            "## Gold Proxy Distributions",
            "",
            f"- Gold-8 official proxy rows: {summary.get('gold8_official_proxy_rows', 0)}",
            f"- Gold-8 official proxy matched rows: {summary.get('gold8_official_proxy_matched_rows', 0)}",
            f"- Official proxy D1 belief distribution: {summary.get('official_proxy_d1_belief_distribution', {})}",
            f"- Non-official player proxy D1 belief distribution: {summary.get('non_official_player_proxy_d1_belief_distribution', {})}",
            f"- Off-pitch/context proxy D1 belief distribution: {summary.get('off_pitch_context_proxy_d1_belief_distribution', {})}",
            f"- Official-like false-positive proxy count: {summary.get('official_like_false_positive_proxy_count', 0)}",
            f"- Player-like official-missed proxy count: {summary.get('player_like_official_missed_proxy_count', 0)}",
            f"- Bad detection proxy count: {summary.get('bad_detection_proxy_count', 0)}",
            "",
            "## Recommendation",
            "",
            summary.get("d1_safety_message", ""),
            "",
            "## Safety Missing Reasons",
            "",
            "```json",
            json.dumps(summary.get("d1_safety_missing_reasons", []), indent=2),
            "```",
        ]
    ) + "\n"


def review_decision_template_payload() -> dict[str, Any]:
    return {
        "reviewer_name": "",
        "reviewed_at": "",
        "review_contact_sheet_reviewed": False,
        "context_crop_sheet_reviewed": False,
        "approve_d1_official_context_for_next_stage_candidate": False,
        "approve_any_official_exclusion": False,
        "approve_any_player_slot_use": False,
        "known_issues": [],
        "notes": "",
        "production_ready": PRODUCTION_READY,
        "no_auto_promotion": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
    }


def build_and_write_official_context_eval() -> dict[str, Any]:
    c2c_payload = read_json(STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH)
    feature_payload = read_json(STEP1D1_OFFICIAL_CONTEXT_FEATURE_ROWS_PATH)
    belief_payload = read_json(STEP1D1_OFFICIAL_CONTEXT_BELIEF_ROWS_PATH)
    review_payload = read_json(STEP1D1_OFFICIAL_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH)
    c2c_summary = read_json(STEP1C2C_GOLD8_HUMAN_CORRECTED_COLOUR_EVAL_SUMMARY_PATH)
    summary = build_official_context_eval_summary(c2c_payload, feature_payload, belief_payload, review_payload, c2c_summary)
    write_json(STEP1D1_GOLD8_OFFICIAL_CONTEXT_EVAL_SUMMARY_PATH, summary)
    write_text(STEP1D1_GOLD8_OFFICIAL_CONTEXT_EVAL_REPORT_PATH, official_context_eval_report(summary))
    write_json(STEP1D1_REVIEW_DECISION_TEMPLATE_PATH, review_decision_template_payload())
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
            "# Step1.D1 Review Index",
            "",
            f"- C2c rows: {summary.get('total_c2c_rows', 0)}",
            f"- D1 belief rows: {summary.get('total_d1_belief_rows', 0)}",
            f"- Review candidates: {summary.get('d1_review_candidate_count', 0)}",
            f"- Gold official proxy matched rows: {summary.get('gold8_official_proxy_matched_rows', 0)}",
            f"- Safe for human review candidate: {summary.get('d1_safe_for_human_review_candidate', False)}",
            f"- Visual flag: {VISUAL_ONLY_WARNING}",
            "- production_ready=false",
            "- no_auto_promotion=true",
        ]
    ) + "\n"


def scope_text() -> str:
    return "\n".join(
        [
            "# Step1.D1 Scope And Restrictions",
            "",
            "Step1.D1 is a visual-only official/referee/context-person belief review sandbox on top of C2c.",
            "",
            "- It is not an official/referee exclusion stage.",
            "- It does not remove candidates from player/team review.",
            "- It does not create identities, player slots, expected roles, goalkeeper classifications, projected-pitch truth, tactical/physical/football metrics, project default changes, registry changes, or promotion.",
            "- Colour, provenance, and image-space cues are visual QA hints only.",
            "- production_ready remains false.",
        ]
    ) + "\n"


def combined_script_text(paths: list[Path]) -> str:
    chunks = []
    for path in paths:
        chunks.append(f"# ===== {path.name} =====\n")
        chunks.append(path.read_text(encoding="utf-8"))
        chunks.append("\n")
    return "\n".join(chunks)


def tests_added_text() -> str:
    return "\n".join(
        [
            "# Step1.D1 Tests Added",
            "",
            "- `tests/test_step1d1_official_context_features.py` covers feature row preservation, C2c colour/provenance fields, visual-only flags, and provenance-only source flags.",
            "- `tests/test_step1d1_official_context_beliefs.py` covers allowed beliefs, no exclusion, context override preservation, and bad-detection review requirements.",
            "- `tests/test_step1d1_official_context_eval.py` covers Gold proxy-only eval fields and safety-gate reporting.",
            "- `tests/test_step1d1_restrictions.py` covers forbidden identity/slot/metric/exclusion keys, registry/default flags, promotion strings, and production_ready=false.",
        ]
    ) + "\n"


def clear_review_pack_dir() -> None:
    STEP1D1_REVIEW_PACK_DIR.mkdir(parents=True, exist_ok=True)
    for path in STEP1D1_REVIEW_PACK_DIR.iterdir():
        if path.is_file():
            path.unlink()


def build_step1d1_review_pack() -> dict[str, Any]:
    clear_review_pack_dir()
    summary = read_json(STEP1D1_GOLD8_OFFICIAL_CONTEXT_EVAL_SUMMARY_PATH)
    entries: list[dict[str, Any]] = []

    def add_entry(name: str, description: str, kind: str) -> Path:
        path = STEP1D1_REVIEW_PACK_DIR / name
        entries.append({"name": name, "kind": kind, "description": description, "path": str(path.resolve())})
        return path

    write_text(add_entry("00_REVIEW_INDEX.md", "D1 review starting point.", "markdown"), review_index_text(summary))
    write_text(add_entry("01_SCOPE_AND_RESTRICTIONS.md", "D1 scope guardrails.", "markdown"), scope_text())
    write_json(add_entry("02_D1_EVAL_SUMMARY.json", "D1 Gold-8 visual QA summary.", "json"), summary)
    copy_text_file(STEP1D1_GOLD8_OFFICIAL_CONTEXT_EVAL_REPORT_PATH, add_entry("03_D1_EVAL_REPORT.md", "D1 eval report.", "markdown"))
    copy_text_file(STEP1D1_OFFICIAL_CONTEXT_BELIEF_REPORT_PATH, add_entry("04_D1_BELIEF_REPORT.md", "D1 belief report.", "markdown"))
    write_json(add_entry("05_OFFICIAL_CONTEXT_FEATURE_SAMPLE.json", "Sample of D1 feature rows.", "json"), sample_payload(STEP1D1_OFFICIAL_CONTEXT_FEATURE_ROWS_PATH, 80, "step1d1_official_context_feature_sample"))
    write_json(add_entry("06_OFFICIAL_CONTEXT_BELIEF_SAMPLE.json", "Sample of D1 belief rows.", "json"), sample_payload(STEP1D1_OFFICIAL_CONTEXT_BELIEF_ROWS_PATH, 80, "step1d1_official_context_belief_sample"))
    write_json(add_entry("07_REVIEW_CANDIDATE_SAMPLE.json", "Sample of D1 review candidates.", "json"), sample_payload(STEP1D1_OFFICIAL_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH, 80, "step1d1_review_candidate_sample"))
    copy_binary_file(STEP1D1_REVIEW_CONTACT_SHEET_PATH, add_entry("08_REVIEW_CONTACT_SHEET.jpg", "D1 multi-panel review contact sheet.", "image"))
    copy_binary_file(STEP1D1_CONTEXT_CROP_CONTACT_SHEET_PATH, add_entry("09_CONTEXT_CROP_CONTACT_SHEET.jpg", "D1 context crop contact sheet.", "image"))
    write_json(add_entry("10_REVIEW_DECISION_TEMPLATE.json", "D1 review decision template.", "json"), read_json(STEP1D1_REVIEW_DECISION_TEMPLATE_PATH))
    code_files = [
        ("11_official_context_features.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "official_context_features.py", "D1 official/context feature extraction."),
        ("12_official_context_beliefs.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "official_context_beliefs.py", "D1 official/context belief policy."),
        ("13_official_context_eval.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "official_context_eval.py", "D1 eval and review pack."),
        ("14_official_context_render.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "official_context_render.py", "D1 visual renderers."),
        ("15_SCRIPT_FEATURES.py", SOCCERTRACK_ROOT / "scripts" / "step1d1_extract_official_context_features.py", "D1 feature script."),
    ]
    for name, source, description in code_files:
        copy_text_file(source, add_entry(name, description, "python"))
    write_text(
        add_entry("16_SCRIPT_BELIEFS_EVAL_RENDER.py", "D1 belief, eval, render, and pack scripts.", "python"),
        combined_script_text(
            [
                SOCCERTRACK_ROOT / "scripts" / "step1d1_build_official_context_beliefs.py",
                SOCCERTRACK_ROOT / "scripts" / "step1d1_evaluate_official_context_gold8.py",
                SOCCERTRACK_ROOT / "scripts" / "step1d1_render_official_context_review.py",
                SOCCERTRACK_ROOT / "scripts" / "step1d1_build_review_pack.py",
            ]
        ),
    )
    write_text(add_entry("17_TESTS_ADDED.md", "Summary of D1 tests.", "markdown"), tests_added_text())
    manifest_path = add_entry("18_REVIEW_PACK_MANIFEST.json", "D1 review pack manifest.", "json")
    manifest = {
        "artifact": "step1d1_review_pack_manifest",
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
            "step1d1_official_context_feature_rows_path": str(STEP1D1_OFFICIAL_CONTEXT_FEATURE_ROWS_PATH.resolve()),
            "step1d1_official_context_belief_rows_path": str(STEP1D1_OFFICIAL_CONTEXT_BELIEF_ROWS_PATH.resolve()),
            "step1d1_official_context_review_candidate_rows_path": str(STEP1D1_OFFICIAL_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH.resolve()),
            "step1d1_gold8_official_context_eval_summary_path": str(STEP1D1_GOLD8_OFFICIAL_CONTEXT_EVAL_SUMMARY_PATH.resolve()),
            "step1d1_review_pack_manifest_path": str(STEP1D1_REVIEW_PACK_MANIFEST_PATH.resolve()),
        },
        "summary": summary,
        "review_pack_file_count": len(entries),
        "review_pack_file_limit": 20,
        "review_pack_entries": entries,
    }
    write_json(manifest_path, manifest)
    write_json(STEP1D1_REVIEW_PACK_MANIFEST_PATH, manifest)
    if len(entries) > 20:
        raise RuntimeError(f"Step1.D1 review pack contains {len(entries)} files; maximum is 20.")
    return manifest


def print_step1d1_final_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    summary = manifest["summary"]
    print(f"step1d1_official_context_feature_rows_path: {outputs['step1d1_official_context_feature_rows_path']}")
    print(f"step1d1_official_context_belief_rows_path: {outputs['step1d1_official_context_belief_rows_path']}")
    print(f"step1d1_official_context_review_candidate_rows_path: {outputs['step1d1_official_context_review_candidate_rows_path']}")
    print(f"step1d1_gold8_official_context_eval_summary_path: {outputs['step1d1_gold8_official_context_eval_summary_path']}")
    print(f"step1d1_review_pack_manifest_path: {outputs['step1d1_review_pack_manifest_path']}")
    print(f"c2c_row_count: {summary.get('total_c2c_rows', 0)}")
    print(f"d1_feature_row_count: {summary.get('total_d1_feature_rows', 0)}")
    print(f"d1_belief_row_count: {summary.get('total_d1_belief_rows', 0)}")
    print(f"d1_review_candidate_count: {summary.get('d1_review_candidate_count', 0)}")
    print(f"official_context_belief_counts: {summary.get('official_context_belief_counts', {})}")
    print(f"gold8_official_proxy_rows: {summary.get('gold8_official_proxy_rows', 0)}")
    print(f"gold8_official_proxy_matched_rows: {summary.get('gold8_official_proxy_matched_rows', 0)}")
    print(f"d1_safe_for_human_review_candidate={str(summary.get('d1_safe_for_human_review_candidate', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("goalkeeper_classification_performed=false")
    print("official_specialist_exclusion_performed=false")
