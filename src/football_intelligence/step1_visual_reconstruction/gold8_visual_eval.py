# ruff: noqa: E501

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

from football_intelligence.paths import STAGE3C13_GOLD20_MANUAL_LABELS_PATH
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1B2_ERROR_ROWS_PATH,
    STEP1B2_GOLD8_EVAL_REPORT_PATH,
    STEP1B2_GOLD8_EVAL_SUMMARY_PATH,
    load_person_states,
    read_json,
    write_json,
    write_text,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    bbox_from_item,
    bbox_iou,
    footpoint_from_bbox,
    is_observed_visible_state,
    safe_float,
    visual_stamp,
)


INCLUDED_GOLD_VISIBLE_TYPES = {
    "team_1_player",
    "team_2_player",
    "gk_team_1",
    "gk_team_2",
    "official_referee",
    "unknown_player",
    "off_pitch_person",
}
EXCLUDED_GOLD_NONPERSON_TYPES = {"false_positive", "unknown_nonplayer"}
INCLUDED_OCCLUSION_STATES = {"observed_clear", "observed_partial"}
PLAYER_OR_GK_TYPES = {"team_1_player", "team_2_player", "gk_team_1", "gk_team_2", "unknown_player"}

FOOTPOINT_THRESHOLD_PX = 45.0
CENTRE_THRESHOLD_PX = 45.0
BBOX_IOU_FALLBACK = 0.12
DUPLICATE_FOOTPOINT_THRESHOLD_PX = 8.0
DUPLICATE_IOU_THRESHOLD = 0.85
NEAR_VISUAL_THRESHOLD_PX = 45.0


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def valid_bbox(item: dict[str, Any]) -> dict[str, float] | None:
    return bbox_from_item(item)


def point_gap(left: dict[str, Any] | None, right: dict[str, Any] | None) -> float:
    if not left or not right or left.get("x") is None or right.get("x") is None:
        return 1_000_000.0
    return float(((safe_float(left.get("x")) - safe_float(right.get("x"))) ** 2 + (safe_float(left.get("y")) - safe_float(right.get("y"))) ** 2) ** 0.5)


def bbox_center(bbox: dict[str, Any] | None) -> dict[str, Any] | None:
    if not bbox:
        return None
    return {
        "x": (safe_float(bbox.get("x1")) + safe_float(bbox.get("x2"))) / 2.0,
        "y": (safe_float(bbox.get("y1")) + safe_float(bbox.get("y2"))) / 2.0,
        "method": "bbox_center",
        "confidence": 0.75,
    }


def gold_footpoint(person: dict[str, Any], bbox: dict[str, Any]) -> dict[str, Any]:
    value = person.get("image_footpoint_xy_gold")
    if isinstance(value, list) and len(value) == 2:
        return {"x": safe_float(value[0]), "y": safe_float(value[1]), "method": "gold_image_footpoint", "confidence": 0.95}
    return footpoint_from_bbox(bbox)


def candidate_footpoint(row: dict[str, Any]) -> dict[str, Any]:
    footpoint = row.get("footpoint")
    if isinstance(footpoint, dict) and footpoint.get("x") is not None:
        return footpoint
    return footpoint_from_bbox(bbox_from_item(row))


def notes_excluded(person: dict[str, Any]) -> bool:
    notes = str(person.get("notes", "")).lower()
    return "accidental press" in notes


def load_completed_gold8_frames(labels_payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    payload = labels_payload or read_json(STAGE3C13_GOLD20_MANUAL_LABELS_PATH)
    return [frame for frame in payload.get("frames", []) if boolish(frame.get("labels_complete"))]


def gold_visible_person_rows(labels_payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in load_completed_gold8_frames(labels_payload):
        for person in frame.get("persons", []):
            visible_type = str(person.get("visible_person_type_gold", "")).strip()
            occlusion_state = str(person.get("occlusion_state_gold", "")).strip()
            bbox = valid_bbox(person)
            if not bbox:
                continue
            if notes_excluded(person) or visible_type in EXCLUDED_GOLD_NONPERSON_TYPES:
                continue
            if visible_type not in INCLUDED_GOLD_VISIBLE_TYPES:
                continue
            if occlusion_state not in INCLUDED_OCCLUSION_STATES:
                continue
            rows.append(
                visual_stamp(
                    {
                        "gold_row_id": str(person.get("candidate_row_id") or person.get("gold_person_id") or ""),
                        "gold_person_id": str(person.get("gold_person_id", "")),
                        "candidate_row_id": str(person.get("candidate_row_id", "")),
                        "frame_id": str(frame.get("frame_id", "")),
                        "frame_sequence": int(safe_float(frame.get("frame_sequence"), -1)),
                        "timestamp_seconds": safe_float(frame.get("timestamp_seconds")),
                        "bbox": {key: round(safe_float(bbox[key]), 3) for key in ["x1", "y1", "x2", "y2"]},
                        "footpoint": gold_footpoint(person, bbox),
                        "visible_person_type_gold": visible_type,
                        "occlusion_state_gold": occlusion_state,
                    }
                )
            )
    return rows


def gold_excluded_nonperson_rows(labels_payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in load_completed_gold8_frames(labels_payload):
        for person in frame.get("persons", []):
            visible_type = str(person.get("visible_person_type_gold", "")).strip()
            bbox = valid_bbox(person)
            if not bbox:
                continue
            if visible_type not in EXCLUDED_GOLD_NONPERSON_TYPES and not notes_excluded(person):
                continue
            rows.append(
                visual_stamp(
                    {
                        "gold_row_id": str(person.get("candidate_row_id") or person.get("gold_person_id") or ""),
                        "gold_person_id": str(person.get("gold_person_id", "")),
                        "candidate_row_id": str(person.get("candidate_row_id", "")),
                        "frame_id": str(frame.get("frame_id", "")),
                        "frame_sequence": int(safe_float(frame.get("frame_sequence"), -1)),
                        "timestamp_seconds": safe_float(frame.get("timestamp_seconds")),
                        "bbox": {key: round(safe_float(bbox[key]), 3) for key in ["x1", "y1", "x2", "y2"]},
                        "footpoint": gold_footpoint(person, bbox),
                        "visible_person_type_gold": visible_type,
                        "exclusion_reason": "accidental_press_note" if notes_excluded(person) else "excluded_nonperson_type",
                    }
                )
            )
    return rows


def observed_visible_rows(state_payload: dict[str, Any], frame_sequences: set[int] | None = None) -> list[dict[str, Any]]:
    rows = [
        row
        for row in state_payload.get("rows", [])
        if is_observed_visible_state(str(row.get("state"))) and (frame_sequences is None or int(safe_float(row.get("frame_sequence"), -1)) in frame_sequences)
    ]
    return rows


def unknown_state_rows(state_payload: dict[str, Any], frame_sequences: set[int] | None = None) -> list[dict[str, Any]]:
    return [
        row
        for row in state_payload.get("rows", [])
        if str(row.get("state")) == "unknown" and (frame_sequences is None or int(safe_float(row.get("frame_sequence"), -1)) in frame_sequences)
    ]


def candidate_match_features(gold: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    gold_bbox = bbox_from_item(gold)
    candidate_bbox = bbox_from_item(candidate)
    fp_gap = point_gap(gold.get("footpoint"), candidate_footpoint(candidate))
    centre_gap = point_gap(bbox_center(gold_bbox), bbox_center(candidate_bbox))
    iou = bbox_iou(gold_bbox, candidate_bbox)
    primary_gap = min(fp_gap, centre_gap)
    score = primary_gap - (iou * 60.0)
    return {
        "score": round(score, 4),
        "visual_gap_px": round(primary_gap, 3),
        "footpoint_gap_px": round(fp_gap, 3),
        "centre_gap_px": round(centre_gap, 3),
        "bbox_iou": round(iou, 4),
    }


def candidate_matches_gold(gold: dict[str, Any], candidate: dict[str, Any]) -> bool:
    features = candidate_match_features(gold, candidate)
    return (
        features["footpoint_gap_px"] <= FOOTPOINT_THRESHOLD_PX
        or features["centre_gap_px"] <= CENTRE_THRESHOLD_PX
        or features["bbox_iou"] >= BBOX_IOU_FALLBACK
    )


def strict_one_to_one_match(
    gold_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    edges = []
    for gold in gold_rows:
        for candidate in candidate_rows:
            if not candidate_matches_gold(gold, candidate):
                continue
            features = candidate_match_features(gold, candidate)
            edges.append((features["score"], -features["bbox_iou"], features["visual_gap_px"], gold, candidate, features))
    edges.sort(key=lambda item: (item[0], item[1], item[2], str(item[3].get("gold_row_id")), str(item[4].get("detection_id"))))
    matched_gold: set[str] = set()
    matched_candidate: set[str] = set()
    matches: list[dict[str, Any]] = []
    for _score, _neg_iou, _gap, gold, candidate, features in edges:
        gold_id = str(gold.get("gold_row_id"))
        candidate_id = str(candidate.get("detection_id"))
        if gold_id in matched_gold or candidate_id in matched_candidate:
            continue
        matched_gold.add(gold_id)
        matched_candidate.add(candidate_id)
        matches.append(
            {
                "gold": gold,
                "candidate": candidate,
                "match_features": features,
            }
        )
    missed = [gold for gold in gold_rows if str(gold.get("gold_row_id")) not in matched_gold]
    extra = [candidate for candidate in candidate_rows if str(candidate.get("detection_id")) not in matched_candidate]
    return matches, missed, extra


def duplicate_candidate_pairs(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_frame[int(safe_float(row.get("frame_sequence"), -1))].append(row)
    for seq, rows in by_frame.items():
        for index, left in enumerate(rows):
            for right in rows[index + 1 :]:
                fp_gap = point_gap(candidate_footpoint(left), candidate_footpoint(right))
                iou = bbox_iou(bbox_from_item(left), bbox_from_item(right))
                if fp_gap <= DUPLICATE_FOOTPOINT_THRESHOLD_PX or iou >= DUPLICATE_IOU_THRESHOLD:
                    pairs.append(
                        {
                            "frame_sequence": seq,
                            "frame_id": left.get("frame_id", ""),
                            "left_detection_id": left.get("detection_id", ""),
                            "right_detection_id": right.get("detection_id", ""),
                            "visual_gap_px": round(fp_gap, 3),
                            "bbox_iou": round(iou, 4),
                        }
                    )
    return pairs


def base_error(issue_type: str, frame_id: str, frame_sequence: int, details: dict[str, Any]) -> dict[str, Any]:
    return visual_stamp(
        {
            "issue_type": issue_type,
            "frame_id": frame_id,
            "frame_sequence": int(frame_sequence),
            **details,
        }
    )


def nearest_gold(candidate: dict[str, Any], gold_rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, float]]:
    best_gold = None
    best_features = {"visual_gap_px": 1_000_000.0, "bbox_iou": 0.0}
    for gold in gold_rows:
        if int(safe_float(gold.get("frame_sequence"), -1)) != int(safe_float(candidate.get("frame_sequence"), -2)):
            continue
        features = candidate_match_features(gold, candidate)
        if features["visual_gap_px"] < best_features["visual_gap_px"] or features["bbox_iou"] > best_features["bbox_iou"]:
            best_gold = gold
            best_features = features
    return best_gold, best_features


def evaluate_state_payload_against_gold8(
    state_payload: dict[str, Any],
    labels_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    completed_frames = load_completed_gold8_frames(labels_payload)
    gold_rows = gold_visible_person_rows(labels_payload)
    excluded_rows = gold_excluded_nonperson_rows(labels_payload)
    frame_sequences = {int(safe_float(frame.get("frame_sequence"), -1)) for frame in completed_frames}
    observed_rows = observed_visible_rows(state_payload, frame_sequences)
    unknown_rows = unknown_state_rows(state_payload, frame_sequences)
    matches, missed, extra = strict_one_to_one_match(gold_rows, observed_rows)
    duplicate_pairs = duplicate_candidate_pairs(observed_rows)
    matched_gold_ids = {str(match["gold"].get("gold_row_id")) for match in matches}
    matched_by_type = Counter(str(match["gold"].get("visible_person_type_gold")) for match in matches)
    gold_by_type = Counter(str(row.get("visible_person_type_gold")) for row in gold_rows)

    errors: list[dict[str, Any]] = []
    for gold in missed:
        errors.append(
            base_error(
                "missed_visible_person",
                str(gold.get("frame_id", "")),
                int(safe_float(gold.get("frame_sequence"), -1)),
                {
                    "gold_row_id": gold.get("gold_row_id", ""),
                    "gold_person_id": gold.get("gold_person_id", ""),
                    "visible_person_type_gold": gold.get("visible_person_type_gold", ""),
                    "occlusion_state_gold": gold.get("occlusion_state_gold", ""),
                },
            )
        )
        if gold.get("visible_person_type_gold") == "official_referee":
            errors.append(
                base_error(
                    "gold_official_unmatched",
                    str(gold.get("frame_id", "")),
                    int(safe_float(gold.get("frame_sequence"), -1)),
                    {"gold_row_id": gold.get("gold_row_id", ""), "visible_person_type_gold": "official_referee"},
                )
            )
        if gold.get("visible_person_type_gold") == "unknown_player":
            errors.append(
                base_error(
                    "gold_unknown_player_unmatched",
                    str(gold.get("frame_id", "")),
                    int(safe_float(gold.get("frame_sequence"), -1)),
                    {"gold_row_id": gold.get("gold_row_id", ""), "visible_person_type_gold": "unknown_player"},
                )
            )

    for candidate in extra:
        nearest, features = nearest_gold(candidate, gold_rows)
        details = {
            "candidate_detection_id": candidate.get("detection_id", ""),
            "candidate_type": candidate.get("candidate_type", ""),
            "state": candidate.get("state", ""),
            "bbox_quality_score": candidate.get("bbox_quality_score", 0.0),
            "roi_status": candidate.get("roi_status", ""),
            "nearest_gold_row_id": "" if nearest is None else nearest.get("gold_row_id", ""),
            "visual_gap_px": features.get("visual_gap_px", 1_000_000.0),
            "bbox_iou": features.get("bbox_iou", 0.0),
        }
        errors.append(base_error("extra_observed_candidate", str(candidate.get("frame_id", "")), int(safe_float(candidate.get("frame_sequence"), -1)), details))
        if safe_float(candidate.get("bbox_quality_score")) < 0.45:
            errors.append(base_error("low_quality_extra_candidate", str(candidate.get("frame_id", "")), int(safe_float(candidate.get("frame_sequence"), -1)), details))
        if str(candidate.get("roi_status")) == "outside_playing_roi":
            errors.append(base_error("outside_roi_extra_candidate", str(candidate.get("frame_id", "")), int(safe_float(candidate.get("frame_sequence"), -1)), details))

    for pair in duplicate_pairs:
        errors.append(base_error("duplicate_candidate_pair", str(pair["frame_id"]), int(pair["frame_sequence"]), pair))

    for unknown in unknown_rows:
        nearest, features = nearest_gold(unknown, gold_rows)
        if nearest is not None and (features["visual_gap_px"] <= NEAR_VISUAL_THRESHOLD_PX or features["bbox_iou"] >= BBOX_IOU_FALLBACK):
            errors.append(
                base_error(
                    "unknown_state_near_gold_visible_person",
                    str(unknown.get("frame_id", "")),
                    int(safe_float(unknown.get("frame_sequence"), -1)),
                    {
                        "candidate_detection_id": unknown.get("detection_id", ""),
                        "gold_row_id": nearest.get("gold_row_id", ""),
                        "visible_person_type_gold": nearest.get("visible_person_type_gold", ""),
                        "visual_gap_px": features["visual_gap_px"],
                        "bbox_iou": features["bbox_iou"],
                    },
                )
            )

    excluded_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in excluded_rows:
        excluded_by_frame[int(safe_float(row.get("frame_sequence"), -1))].append(row)
    for candidate in observed_rows:
        seq = int(safe_float(candidate.get("frame_sequence"), -1))
        for excluded in excluded_by_frame.get(seq, []):
            features = candidate_match_features(excluded, candidate)
            if features["visual_gap_px"] <= NEAR_VISUAL_THRESHOLD_PX or features["bbox_iou"] >= BBOX_IOU_FALLBACK:
                errors.append(
                    base_error(
                        "observed_candidate_near_excluded_gold_nonperson",
                        str(candidate.get("frame_id", "")),
                        seq,
                        {
                            "candidate_detection_id": candidate.get("detection_id", ""),
                            "excluded_gold_row_id": excluded.get("gold_row_id", ""),
                            "excluded_visible_person_type_gold": excluded.get("visible_person_type_gold", ""),
                            "visual_gap_px": features["visual_gap_px"],
                            "bbox_iou": features["bbox_iou"],
                        },
                    )
                )

    duplicate_candidate_ids = {
        str(pair["left_detection_id"]) for pair in duplicate_pairs
    } | {str(pair["right_detection_id"]) for pair in duplicate_pairs}
    player_or_gk_gold_rows = sum(gold_by_type.get(item, 0) for item in PLAYER_OR_GK_TYPES)
    player_or_gk_matched_rows = sum(matched_by_type.get(item, 0) for item in PLAYER_OR_GK_TYPES)
    summary = {
        "artifact": "step1b2_gold8_person_state_eval_summary",
        "created_at": utc_iso(),
        "gold8_frames_used": [
            {
                "frame_id": frame.get("frame_id", ""),
                "frame_sequence": int(safe_float(frame.get("frame_sequence"), -1)),
                "timestamp_seconds": safe_float(frame.get("timestamp_seconds")),
            }
            for frame in completed_frames
        ],
        "gold_visible_person_rows": len(gold_rows),
        "step1_observed_visible_rows": len(observed_rows),
        "matched_gold_visible_rows": len(matches),
        "missed_gold_visible_rows": len(missed),
        "extra_observed_candidate_rows": len(extra),
        "duplicate_candidate_rows": len(duplicate_candidate_ids),
        "unknown_state_rows": len(unknown_rows),
        "unknown_not_counted_as_observed_visible": len(unknown_rows),
        "official_referee_gold_rows": gold_by_type.get("official_referee", 0),
        "official_referee_matched_rows": matched_by_type.get("official_referee", 0),
        "unknown_player_gold_rows": gold_by_type.get("unknown_player", 0),
        "unknown_player_matched_rows": matched_by_type.get("unknown_player", 0),
        "player_or_gk_gold_rows": player_or_gk_gold_rows,
        "player_or_gk_matched_rows": player_or_gk_matched_rows,
        "issue_counts_by_type": dict(sorted(Counter(error["issue_type"] for error in errors).items())),
        "matched_gold_row_ids": sorted(matched_gold_ids),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "no_team_role_identity_or_slot_evaluation": True,
    }
    return summary, errors


def frame_issue_counts(error_rows: list[dict[str, Any]], issue_type: str) -> list[dict[str, Any]]:
    counts: dict[tuple[int, str], int] = defaultdict(int)
    for row in error_rows:
        if row.get("issue_type") == issue_type:
            counts[(int(safe_float(row.get("frame_sequence"), -1)), str(row.get("frame_id", "")))] += 1
    out = [
        {"frame_sequence": seq, "frame_id": frame_id, "count": count}
        for (seq, frame_id), count in counts.items()
    ]
    return sorted(out, key=lambda item: (-item["count"], item["frame_sequence"]))[:8]


def gold8_eval_report_markdown(summary: dict[str, Any], error_rows: list[dict[str, Any]]) -> str:
    issue_counts = summary.get("issue_counts_by_type", {})
    lines = [
        "# Step1.B2 Gold-8 Visual Person-State Eval Report",
        "",
        f"- Warning: `{VISUAL_ONLY_WARNING}`.",
        "- Scope: image-space visible-person reconstruction only.",
        "- No team correctness, role correctness, identity correctness, or player-slot correctness was evaluated.",
        "- No speed, distance, fatigue, player-load, team-shape, pass, dribble, tactical, physical-performance, or football-conclusion metrics were calculated.",
        "- production_ready false",
        "- project_wide_defaults_changed false",
        "- stage3d_registries_changed false",
        "",
        "## Gold-8 Frames Used",
        "",
        "| frame_sequence | frame_id | timestamp_seconds |",
        "|---:|---|---:|",
    ]
    for frame in summary.get("gold8_frames_used", []):
        lines.append(f"| {frame['frame_sequence']} | {frame['frame_id']} | {frame['timestamp_seconds']:.1f} |")
    lines.extend(
        [
            "",
            "## Visual-Only Matching Policy",
            "",
            "- Strict one-to-one visual matching.",
            f"- Footpoint threshold: {FOOTPOINT_THRESHOLD_PX:.0f} px.",
            f"- Centre fallback threshold: {CENTRE_THRESHOLD_PX:.0f} px.",
            f"- Bbox IoU fallback: {BBOX_IOU_FALLBACK:.2f}.",
            "- These are pixel QA thresholds only; they are not football or physical metrics.",
            "",
            "## Summary",
            "",
            "| field | value |",
            "|---|---:|",
        ]
    )
    for key in [
        "gold_visible_person_rows",
        "step1_observed_visible_rows",
        "matched_gold_visible_rows",
        "missed_gold_visible_rows",
        "extra_observed_candidate_rows",
        "duplicate_candidate_rows",
        "unknown_state_rows",
        "official_referee_gold_rows",
        "official_referee_matched_rows",
        "unknown_player_gold_rows",
        "unknown_player_matched_rows",
        "player_or_gk_gold_rows",
        "player_or_gk_matched_rows",
    ]:
        lines.append(f"| {key} | {summary.get(key, 0)} |")
    lines.extend(["", "## Issue Counts", "", "| issue_type | count |", "|---|---:|"])
    for key, value in sorted(issue_counts.items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Worst Frames By Missed Visible Person", "", "| frame_sequence | frame_id | missed |", "|---:|---|---:|"])
    for item in frame_issue_counts(error_rows, "missed_visible_person"):
        lines.append(f"| {item['frame_sequence']} | {item['frame_id']} | {item['count']} |")
    lines.extend(["", "## Worst Frames By Extra Observed Candidate", "", "| frame_sequence | frame_id | extra |", "|---:|---|---:|"])
    for item in frame_issue_counts(error_rows, "extra_observed_candidate"):
        lines.append(f"| {item['frame_sequence']} | {item['frame_id']} | {item['count']} |")
    extra = int(summary.get("extra_observed_candidate_rows", 0))
    missed = int(summary.get("missed_gold_visible_rows", 0))
    if extra > missed:
        threshold_note = "Current Step1.B appears visually loose on Gold-8 because extra observed candidates exceed missed visible persons."
    elif missed > extra:
        threshold_note = "Current Step1.B appears visually strict on Gold-8 because missed visible persons exceed extra observed candidates."
    else:
        threshold_note = "Current Step1.B has balanced missed and extra counts on Gold-8; review image panels before threshold changes."
    lines.extend(["", "## Threshold Read", "", f"- {threshold_note}", "- This report does not auto-promote any threshold change."])
    return "\n".join(lines) + "\n"


def build_and_write_gold8_eval(state_payload: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = state_payload or load_person_states()
    summary, error_rows = evaluate_state_payload_against_gold8(payload)
    write_json(STEP1B2_GOLD8_EVAL_SUMMARY_PATH, summary)
    write_json(
        STEP1B2_ERROR_ROWS_PATH,
        {
            "artifact": "step1b2_gold8_error_rows",
            "visual_only_warning": VISUAL_ONLY_WARNING,
            "do_not_use_for_metrics": True,
            "production_ready": PRODUCTION_READY,
            "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
            "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
            "rows": error_rows,
            "summary": {"issue_counts_by_type": summary.get("issue_counts_by_type", {}), "row_count": len(error_rows)},
        },
    )
    write_text(STEP1B2_GOLD8_EVAL_REPORT_PATH, gold8_eval_report_markdown(summary, error_rows))
    return summary, error_rows
