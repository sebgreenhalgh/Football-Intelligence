# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.paths import CLIP_ID, MATCH_ID, STAGE3C13_GOLD20_MANUAL_LABELS_PATH
from football_intelligence.step1_visual_reconstruction.goalkeeper_context_beliefs import (
    E1_FORBIDDEN_KEYS,
    GOALKEEPER_LIKE_BELIEFS,
)
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import gold_visible_person_rows, strict_one_to_one_match
from football_intelligence.step1_visual_reconstruction.io import (
    SOCCERTRACK_ROOT,
    STEP1D1C_GOLD8_HUMAN_CORRECTED_OFFICIAL_CONTEXT_EVAL_SUMMARY_PATH,
    STEP1E1_GOLD8_GOALKEEPER_CONTEXT_EVAL_REPORT_PATH,
    STEP1E1_GOLD8_GOALKEEPER_CONTEXT_EVAL_SUMMARY_PATH,
    STEP1E1_GOALKEEPER_CONTEXT_BELIEF_ROWS_PATH,
    STEP1E1_GOALKEEPER_CONTEXT_FEATURE_ROWS_PATH,
    STEP1E1_GOALKEEPER_CONTEXT_REPORT_PATH,
    STEP1E1_GOALKEEPER_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH,
    STEP1E1_GOALKEEPER_CROP_CONTACT_SHEET_PATH,
    STEP1E1_REVIEW_CONTACT_SHEET_PATH,
    STEP1E1_REVIEW_DECISION_TEMPLATE_PATH,
    STEP1E1_REVIEW_PACK_DIR,
    STEP1E1_REVIEW_PACK_MANIFEST_PATH,
    copy_binary_file,
    copy_text_file,
    read_json,
    write_json,
    write_text,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
)


GOALKEEPER_PROXY_TYPES = {"gk_team_1", "gk_team_2"}
NON_GOALKEEPER_VISIBLE_PLAYER_PROXY_TYPES = {"team_1_player", "team_2_player", "unknown_player"}
OFFICIAL_CONTEXT_PROXY_TYPES = {"official_referee", "off_pitch_person"}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def proxy_group(visible_type: str) -> str:
    if visible_type in GOALKEEPER_PROXY_TYPES:
        return "goalkeeper_proxy"
    if visible_type in NON_GOALKEEPER_VISIBLE_PLAYER_PROXY_TYPES:
        return "non_goalkeeper_visible_player_proxy"
    if visible_type in OFFICIAL_CONTEXT_PROXY_TYPES:
        return "official_context_proxy"
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
                "d1c_final_official_context_belief": row.get("d1c_final_official_context_belief", ""),
                "e1_goalkeeper_context_belief": row.get("e1_goalkeeper_context_belief", ""),
                "e1_goalkeeper_context_review_required": row.get("e1_goalkeeper_context_review_required", False),
                "bbox_iou": match.get("match_features", {}).get("bbox_iou", 0.0),
                "visual_gap_px": match.get("match_features", {}).get("visual_gap_px", 0.0),
                "visual_only_warning": VISUAL_ONLY_WARNING,
                "do_not_use_for_metrics": True,
                "production_ready": PRODUCTION_READY,
            }
        )
    return out


def distribution_by_proxy(rows: list[dict[str, Any]], belief_key: str) -> dict[str, dict[str, int]]:
    distribution: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        distribution[str(row.get("proxy_group", ""))][str(row.get(belief_key, ""))] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(distribution.items())}


def forbidden_keys_present(*payloads: dict[str, Any]) -> list[str]:
    found: set[str] = set()
    for payload in payloads:
        for row in payload.get("rows", []):
            found.update(key for key in E1_FORBIDDEN_KEYS if key in row)
    return sorted(found)


def goalkeeper_like_false_positive_proxy_count(rows: list[dict[str, Any]], proxy_name: str) -> int:
    return sum(
        1
        for row in rows
        if row.get("proxy_group") == proxy_name and row.get("e1_goalkeeper_context_belief") in GOALKEEPER_LIKE_BELIEFS
    )


def e1_safety_missing_reasons(
    *,
    d1c_summary: dict[str, Any],
    feature_payload: dict[str, Any],
    belief_payload: dict[str, Any],
    review_payload: dict[str, Any],
    proxy_rows: list[dict[str, Any]],
    forbidden_keys: list[str],
) -> list[str]:
    reasons = []
    d1c_count = int(d1c_summary.get("d1c_row_count", 0))
    feature_count = len(feature_payload.get("rows", []))
    belief_count = len(belief_payload.get("rows", []))
    if d1c_summary.get("d1c_safe_for_step1e_candidate") is not True:
        reasons.append("d1c_not_safe_for_step1e_candidate")
    if d1c_count != 10418 or feature_count != 10418 or belief_count != 10418:
        reasons.append("e1_row_counts_not_10418")
    if d1c_count != feature_count:
        reasons.append("one_feature_row_per_d1c_row_not_preserved")
    if d1c_count != belief_count:
        reasons.append("one_belief_row_per_d1c_row_not_preserved")
    feature_ids = sorted(str(row.get("visible_person_base_id", "")) for row in feature_payload.get("rows", []))
    belief_ids = sorted(str(row.get("visible_person_base_id", "")) for row in belief_payload.get("rows", []))
    if feature_ids != belief_ids:
        reasons.append("e1_feature_and_belief_visible_person_base_ids_do_not_match")
    if forbidden_keys:
        reasons.append("forbidden_identity_slot_metric_or_exclusion_keys_present")
    if any(row.get("retained_for_future_player_team_review") is not True for row in belief_payload.get("rows", [])):
        reasons.append("not_all_rows_retained_for_future_player_team_review")
    if any(row.get("eligible_for_identity_tracking") is not False for row in belief_payload.get("rows", [])):
        reasons.append("identity_tracking_eligibility_not_false")
    if any(row.get("eligible_for_player_slot_assignment") is not False for row in belief_payload.get("rows", [])):
        reasons.append("player_slot_assignment_eligibility_not_false")
    if belief_payload.get("production_ready") is not False or any(row.get("production_ready") is not False for row in belief_payload.get("rows", [])):
        reasons.append("production_ready_not_false")
    for flag in ["project_wide_defaults_changed", "stage3d_registries_changed", "identity_tracking_performed", "player_slots_assigned", "expected_22_role_states_created", "official_specialist_exclusion_performed"]:
        if belief_payload.get(flag) is not False:
            reasons.append(f"{flag}_not_false")
    if len(review_payload.get("rows", [])) == 0:
        reasons.append("e1_review_candidates_not_emitted")
    if not proxy_rows:
        reasons.append("gold_proxy_evaluation_not_emitted")
    if belief_payload.get("summary", {}).get("exact_two_goalkeeper_forcing_performed") is True:
        reasons.append("exact_two_goalkeeper_forcing_was_performed")
    return reasons


def build_goalkeeper_context_eval_summary(
    d1c_summary: dict[str, Any],
    feature_payload: dict[str, Any],
    belief_payload: dict[str, Any],
    review_payload: dict[str, Any],
    *,
    labels_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    labels_payload = labels_payload or read_json(STAGE3C13_GOLD20_MANUAL_LABELS_PATH)
    proxy_rows = gold_proxy_matches(belief_payload, labels_payload=labels_payload)
    proxy_dist = distribution_by_proxy(proxy_rows, "e1_goalkeeper_context_belief")
    d1c_dist = distribution_by_proxy(proxy_rows, "d1c_final_official_context_belief")
    goalkeeper_proxy_rows = [row for row in gold_visible_person_rows(labels_payload) if row.get("visible_person_type_gold") in GOALKEEPER_PROXY_TYPES]
    goalkeeper_matches = [row for row in proxy_rows if row.get("proxy_group") == "goalkeeper_proxy"]
    missed_goalkeeper = sum(1 for row in goalkeeper_matches if row.get("e1_goalkeeper_context_belief") not in GOALKEEPER_LIKE_BELIEFS)
    unknown_goalkeeper = sum(1 for row in goalkeeper_matches if row.get("e1_goalkeeper_context_belief") == "unknown_goalkeeper_context")
    bad_goalkeeper = sum(1 for row in goalkeeper_matches if row.get("e1_goalkeeper_context_belief") == "bad_detection_or_not_person")
    false_positive_players = goalkeeper_like_false_positive_proxy_count(proxy_rows, "non_goalkeeper_visible_player_proxy")
    false_positive_official_context = goalkeeper_like_false_positive_proxy_count(proxy_rows, "official_context_proxy")
    forbidden_keys = forbidden_keys_present(feature_payload, belief_payload, review_payload)
    missing = e1_safety_missing_reasons(
        d1c_summary=d1c_summary,
        feature_payload=feature_payload,
        belief_payload=belief_payload,
        review_payload=review_payload,
        proxy_rows=proxy_rows,
        forbidden_keys=forbidden_keys,
    )
    safe = not missing
    return {
        "artifact": "step1e1_gold8_goalkeeper_context_eval_summary",
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
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "d1c_row_count": int(d1c_summary.get("d1c_row_count", 0)),
        "e1_feature_row_count": len(feature_payload.get("rows", [])),
        "e1_belief_row_count": len(belief_payload.get("rows", [])),
        "e1_review_candidate_count": len(review_payload.get("rows", [])),
        "e1_goalkeeper_context_belief_counts": belief_payload.get("summary", {}).get("e1_goalkeeper_context_belief_counts", {}),
        "gold_goalkeeper_proxy_rows": len(goalkeeper_proxy_rows),
        "gold_goalkeeper_proxy_matched_rows": len(goalkeeper_matches),
        "d1c_context_distribution_on_goalkeeper_proxy_rows": d1c_dist.get("goalkeeper_proxy", {}),
        "e1_goalkeeper_proxy_belief_distribution": proxy_dist.get("goalkeeper_proxy", {}),
        "e1_missed_goalkeeper_proxy_count": missed_goalkeeper,
        "e1_goalkeeper_like_false_positive_proxy_count": false_positive_players,
        "e1_official_context_false_goalkeeper_like_proxy_count": false_positive_official_context,
        "unknown_goalkeeper_proxy_count": unknown_goalkeeper,
        "bad_detection_goalkeeper_proxy_count": bad_goalkeeper,
        "e1_non_goalkeeper_visible_player_proxy_distribution": proxy_dist.get("non_goalkeeper_visible_player_proxy", {}),
        "e1_official_context_proxy_distribution": proxy_dist.get("official_context_proxy", {}),
        "forbidden_keys_present": forbidden_keys,
        "d1c_safe_for_step1e_candidate": d1c_summary.get("d1c_safe_for_step1e_candidate", False),
        "e1_safe_for_human_review_candidate": safe,
        "e1_safety_missing_reasons": missing,
        "e1_safety_message": "Step1.E1 goalkeeper visual-context beliefs are safe for human visual review candidate use." if safe else "Step1.E1 goalkeeper visual-context beliefs need review before next-stage candidate use.",
        "gold_proxy_note": "Gold visible_person_type_gold is used only as visual goalkeeper/context QA proxy context.",
        "no_auto_promotion": True,
    }


def goalkeeper_context_eval_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.E1 Gold-8 Goalkeeper Context Eval Report",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            "- Gold visible_person_type_gold is used only as visual QA/proxy context.",
            "- E1 is not a player-slot, identity, expected-role, team-shape, or metric stage.",
            "",
            "## Row Counts",
            "",
            f"- D1c rows: {summary.get('d1c_row_count', 0)}",
            f"- E1 feature rows: {summary.get('e1_feature_row_count', 0)}",
            f"- E1 belief rows: {summary.get('e1_belief_row_count', 0)}",
            f"- E1 review candidates: {summary.get('e1_review_candidate_count', 0)}",
            "",
            "## Gold Proxy Summary",
            "",
            f"- Gold goalkeeper proxy rows: {summary.get('gold_goalkeeper_proxy_rows', 0)}",
            f"- Gold goalkeeper proxy matched rows: {summary.get('gold_goalkeeper_proxy_matched_rows', 0)}",
            f"- D1c context distribution on goalkeeper proxies: {summary.get('d1c_context_distribution_on_goalkeeper_proxy_rows', {})}",
            f"- E1 goalkeeper proxy belief distribution: {summary.get('e1_goalkeeper_proxy_belief_distribution', {})}",
            f"- E1 missed goalkeeper proxy count: {summary.get('e1_missed_goalkeeper_proxy_count', 0)}",
            f"- E1 goalkeeper-like false-positive proxy count: {summary.get('e1_goalkeeper_like_false_positive_proxy_count', 0)}",
            f"- E1 official/context false goalkeeper-like proxy count: {summary.get('e1_official_context_false_goalkeeper_like_proxy_count', 0)}",
            f"- Unknown goalkeeper proxy count: {summary.get('unknown_goalkeeper_proxy_count', 0)}",
            f"- Bad-detection goalkeeper proxy count: {summary.get('bad_detection_goalkeeper_proxy_count', 0)}",
            "",
            "## E1 Belief Counts",
            "",
            "```json",
            json.dumps(summary.get("e1_goalkeeper_context_belief_counts", {}), indent=2),
            "```",
            "",
            "## Recommendation",
            "",
            summary.get("e1_safety_message", ""),
            "",
            "## Safety Missing Reasons",
            "",
            "```json",
            json.dumps(summary.get("e1_safety_missing_reasons", []), indent=2),
            "```",
        ]
    ) + "\n"


def review_decision_template_payload() -> dict[str, Any]:
    return {
        "reviewer_name": "",
        "reviewed_at": "",
        "review_contact_sheet_reviewed": False,
        "goalkeeper_crop_sheet_reviewed": False,
        "approve_e1_goalkeeper_context_for_human_review": False,
        "approve_any_goalkeeper_slot_use": False,
        "approve_any_identity_tracking": False,
        "approve_any_metric_use": False,
        "known_issues": [],
        "notes": "",
        "production_ready": PRODUCTION_READY,
        "no_auto_promotion": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
    }


def build_and_write_goalkeeper_context_eval() -> dict[str, Any]:
    d1c_summary = read_json(STEP1D1C_GOLD8_HUMAN_CORRECTED_OFFICIAL_CONTEXT_EVAL_SUMMARY_PATH)
    feature_payload = read_json(STEP1E1_GOALKEEPER_CONTEXT_FEATURE_ROWS_PATH)
    belief_payload = read_json(STEP1E1_GOALKEEPER_CONTEXT_BELIEF_ROWS_PATH)
    review_payload = read_json(STEP1E1_GOALKEEPER_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH)
    summary = build_goalkeeper_context_eval_summary(d1c_summary, feature_payload, belief_payload, review_payload)
    write_json(STEP1E1_GOLD8_GOALKEEPER_CONTEXT_EVAL_SUMMARY_PATH, summary)
    write_text(STEP1E1_GOLD8_GOALKEEPER_CONTEXT_EVAL_REPORT_PATH, goalkeeper_context_eval_report(summary))
    write_json(STEP1E1_REVIEW_DECISION_TEMPLATE_PATH, review_decision_template_payload())
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
            "# Step1.E1 Review Index",
            "",
            f"- D1c rows: {summary.get('d1c_row_count', 0)}",
            f"- E1 belief rows: {summary.get('e1_belief_row_count', 0)}",
            f"- Review candidates: {summary.get('e1_review_candidate_count', 0)}",
            f"- Gold goalkeeper proxy matched rows: {summary.get('gold_goalkeeper_proxy_matched_rows', 0)}",
            f"- Safe for human review candidate: {summary.get('e1_safe_for_human_review_candidate', False)}",
            f"- Visual flag: {VISUAL_ONLY_WARNING}",
            "- production_ready=false",
            "- no_auto_promotion=true",
        ]
    ) + "\n"


def scope_text() -> str:
    return "\n".join(
        [
            "# Step1.E1 Scope And Restrictions",
            "",
            "Step1.E1 is a visual-only goalkeeper/context belief sandbox on top of D1c.",
            "",
            "- It does not delete rows or exclude officials/referees from future player/team review.",
            "- It does not assign goalkeeper slots, player slots, identities, expected 22-role states, team shape, tactics, performance, or metrics.",
            "- It does not force exactly two goalkeepers.",
            "- Image-space goal-area hints are visual context only and are not metric pitch truth.",
            "- Gold visible_person_type_gold is used only as visual QA/proxy context.",
            "- production_ready remains false.",
        ]
    ) + "\n"


def tests_added_text() -> str:
    return "\n".join(
        [
            "# Step1.E1 Tests Added",
            "",
            "- `tests/test_step1e1_goalkeeper_context_beliefs.py` covers row preservation, official/context negatives, bad-detection handling, visual-context-only goalkeeper-like beliefs, no exact-two forcing, and retention flags.",
            "- `tests/test_step1e1_goalkeeper_context_eval.py` covers Gold proxy visual-only reporting, goalkeeper proxy distributions, false-positive counts, and human-review safety requirements.",
            "- `tests/test_step1e1_restrictions.py` covers forbidden keys, no registry/default changes, no promotion strings, and production_ready=false.",
        ]
    ) + "\n"


def clear_review_pack_dir() -> None:
    STEP1E1_REVIEW_PACK_DIR.mkdir(parents=True, exist_ok=True)
    for path in STEP1E1_REVIEW_PACK_DIR.iterdir():
        if path.is_file():
            path.unlink()


def build_step1e1_review_pack() -> dict[str, Any]:
    clear_review_pack_dir()
    summary = read_json(STEP1E1_GOLD8_GOALKEEPER_CONTEXT_EVAL_SUMMARY_PATH)
    entries: list[dict[str, Any]] = []

    def add_entry(name: str, description: str, kind: str) -> Path:
        path = STEP1E1_REVIEW_PACK_DIR / name
        entries.append({"name": name, "kind": kind, "description": description, "path": str(path.resolve())})
        return path

    write_text(add_entry("00_REVIEW_INDEX.md", "E1 review starting point.", "markdown"), review_index_text(summary))
    write_text(add_entry("01_SCOPE_AND_RESTRICTIONS.md", "E1 scope guardrails.", "markdown"), scope_text())
    write_json(add_entry("02_E1_EVAL_SUMMARY.json", "E1 Gold-8 visual QA summary.", "json"), summary)
    copy_text_file(STEP1E1_GOLD8_GOALKEEPER_CONTEXT_EVAL_REPORT_PATH, add_entry("03_E1_EVAL_REPORT.md", "E1 eval report.", "markdown"))
    copy_text_file(STEP1E1_GOALKEEPER_CONTEXT_REPORT_PATH, add_entry("04_E1_GOALKEEPER_CONTEXT_REPORT.md", "E1 goalkeeper context report.", "markdown"))
    write_json(add_entry("05_REVIEW_CANDIDATE_SAMPLE.json", "Sample of E1 review candidates.", "json"), sample_payload(STEP1E1_GOALKEEPER_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH, 80, "step1e1_review_candidate_sample"))
    write_json(add_entry("06_E1_ROWS_SAMPLE.json", "Sample of E1 belief rows.", "json"), sample_payload(STEP1E1_GOALKEEPER_CONTEXT_BELIEF_ROWS_PATH, 80, "step1e1_rows_sample"))
    copy_binary_file(STEP1E1_REVIEW_CONTACT_SHEET_PATH, add_entry("07_REVIEW_CONTACT_SHEET.jpg", "E1 multi-panel review contact sheet.", "image"))
    copy_binary_file(STEP1E1_GOALKEEPER_CROP_CONTACT_SHEET_PATH, add_entry("08_GOALKEEPER_CROP_CONTACT_SHEET.jpg", "E1 goalkeeper crop contact sheet.", "image"))
    write_json(add_entry("09_REVIEW_DECISION_TEMPLATE.json", "E1 review decision template.", "json"), read_json(STEP1E1_REVIEW_DECISION_TEMPLATE_PATH))
    code_files = [
        ("10_goalkeeper_context_beliefs.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "goalkeeper_context_beliefs.py", "E1 goalkeeper context feature and belief policy."),
        ("11_goalkeeper_context_eval.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "goalkeeper_context_eval.py", "E1 eval and review pack."),
        ("12_goalkeeper_context_render.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "goalkeeper_context_render.py", "E1 visual renderers."),
        ("13_SCRIPT_BUILD_BELIEFS.py", SOCCERTRACK_ROOT / "scripts" / "step1e1_build_goalkeeper_context_beliefs.py", "E1 build beliefs script."),
        ("14_SCRIPT_EVAL.py", SOCCERTRACK_ROOT / "scripts" / "step1e1_evaluate_goalkeeper_context_beliefs.py", "E1 eval script."),
        ("15_SCRIPT_RENDER.py", SOCCERTRACK_ROOT / "scripts" / "step1e1_render_goalkeeper_context_review.py", "E1 render script."),
    ]
    for name, source, description in code_files:
        copy_text_file(source, add_entry(name, description, "python"))
    write_text(add_entry("16_TESTS_ADDED.md", "Summary of E1 tests.", "markdown"), tests_added_text())
    manifest_path = add_entry("17_REVIEW_PACK_MANIFEST.json", "E1 review pack manifest.", "json")
    manifest = {
        "artifact": "step1e1_review_pack_manifest",
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
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "outputs": {
            "step1e1_goalkeeper_context_feature_rows_path": str(STEP1E1_GOALKEEPER_CONTEXT_FEATURE_ROWS_PATH.resolve()),
            "step1e1_goalkeeper_context_belief_rows_path": str(STEP1E1_GOALKEEPER_CONTEXT_BELIEF_ROWS_PATH.resolve()),
            "step1e1_goalkeeper_context_review_candidate_rows_path": str(STEP1E1_GOALKEEPER_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH.resolve()),
            "step1e1_gold8_goalkeeper_context_eval_summary_path": str(STEP1E1_GOLD8_GOALKEEPER_CONTEXT_EVAL_SUMMARY_PATH.resolve()),
            "step1e1_gold8_goalkeeper_context_eval_report_path": str(STEP1E1_GOLD8_GOALKEEPER_CONTEXT_EVAL_REPORT_PATH.resolve()),
            "step1e1_goalkeeper_context_report_path": str(STEP1E1_GOALKEEPER_CONTEXT_REPORT_PATH.resolve()),
            "step1e1_review_pack_manifest_path": str(STEP1E1_REVIEW_PACK_MANIFEST_PATH.resolve()),
        },
        "summary": summary,
        "review_pack_file_count": len(entries),
        "review_pack_file_limit": 20,
        "review_pack_entries": entries,
    }
    write_json(manifest_path, manifest)
    write_json(STEP1E1_REVIEW_PACK_MANIFEST_PATH, manifest)
    if len(entries) > 20:
        raise RuntimeError(f"Step1.E1 review pack contains {len(entries)} files; maximum is 20.")
    return manifest


def print_step1e1_final_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    summary = manifest["summary"]
    print(f"step1e1_goalkeeper_context_feature_rows_path: {outputs['step1e1_goalkeeper_context_feature_rows_path']}")
    print(f"step1e1_goalkeeper_context_belief_rows_path: {outputs['step1e1_goalkeeper_context_belief_rows_path']}")
    print(f"step1e1_goalkeeper_context_review_candidate_rows_path: {outputs['step1e1_goalkeeper_context_review_candidate_rows_path']}")
    print(f"step1e1_gold8_goalkeeper_context_eval_summary_path: {outputs['step1e1_gold8_goalkeeper_context_eval_summary_path']}")
    print(f"step1e1_review_pack_manifest_path: {outputs['step1e1_review_pack_manifest_path']}")
    print(f"d1c_row_count: {summary.get('d1c_row_count', 0)}")
    print(f"e1_feature_row_count: {summary.get('e1_feature_row_count', 0)}")
    print(f"e1_belief_row_count: {summary.get('e1_belief_row_count', 0)}")
    print(f"e1_review_candidate_count: {summary.get('e1_review_candidate_count', 0)}")
    print(f"e1_goalkeeper_context_belief_counts: {summary.get('e1_goalkeeper_context_belief_counts', {})}")
    print(f"gold_goalkeeper_proxy_rows: {summary.get('gold_goalkeeper_proxy_rows', 0)}")
    print(f"gold_goalkeeper_proxy_matched_rows: {summary.get('gold_goalkeeper_proxy_matched_rows', 0)}")
    print(f"e1_goalkeeper_proxy_belief_distribution: {summary.get('e1_goalkeeper_proxy_belief_distribution', {})}")
    print(f"e1_goalkeeper_like_false_positive_proxy_count: {summary.get('e1_goalkeeper_like_false_positive_proxy_count', 0)}")
    print(f"e1_safe_for_human_review_candidate={str(summary.get('e1_safe_for_human_review_candidate', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("expected_22_role_states_created=false")
    print("official_specialist_exclusion_performed=false")
