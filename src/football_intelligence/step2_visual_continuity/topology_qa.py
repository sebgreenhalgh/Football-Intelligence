# ruff: noqa: E501

from __future__ import annotations

import gzip
import hashlib
import json
import mimetypes
import shutil
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

try:  # pragma: no cover - exercised when OpenCV is available in the local environment
    import cv2
    import numpy as np
except Exception:  # pragma: no cover - fallback still writes a small placeholder file
    cv2 = None
    np = None

try:  # pragma: no cover - exercised when imageio is available in the local environment
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - static strips still make missing animation explicit
    imageio = None

from football_intelligence.paths import CLIP_ID, MATCH_ID, STAGE3C_FRAMES_DIR, ensure_dir
from football_intelligence.step2_visual_continuity.adaptation_safe_output import (
    MAX_M3_GROUP_SPAN_FRAMES,
    MAX_M3_GROUP_SPAN_SECONDS,
    iter_jsonl_gz_rows,
)
from football_intelligence.step2_visual_continuity.io import (
    STEP2M1_OUTPUT_DIR,
    STEP2M1_NODE_ROWS_PATH,
    STEP2M2_OUTPUT_DIR,
    STEP2M3_ACCEPTED_EDGES_JSONL_GZ_PATH,
    STEP2M3_FREEZE_CANDIDATE_MANIFEST_PATH,
    STEP2M3_GROUP_ROWS_PATH,
    STEP2M3_GROUP_SUMMARY_PATH,
    STEP2M3_OUTPUT_DIR,
    STEP2M3_VALIDATION_SUMMARY_PATH,
    STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_ROWS_JSONL_GZ_PATH,
    STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_SAMPLE_PATH,
    STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_SUMMARY_PATH,
    STEP2M3R_EDGE_BURST_ANIMATIONS_DIR,
    STEP2M3R_EDGE_BURST_STRIPS_DIR,
    STEP2M3R_FREEZE_CANDIDATE_MANIFEST_PATH,
    STEP2M3R_GROUP_TIMELINE_ANIMATIONS_DIR,
    STEP2M3R_GROUP_TIMELINE_STRIPS_DIR,
    STEP2M3R_GROUP_TOPOLOGY_AUDIT_ROWS_PATH,
    STEP2M3R_GROUP_TOPOLOGY_AUDIT_SAMPLE_PATH,
    STEP2M3R_GROUP_TOPOLOGY_AUDIT_SUMMARY_PATH,
    STEP2M3R_HANDOFF_READINESS_SUMMARY_PATH,
    STEP2M3R_ISSUE_REGISTER_PATH,
    STEP2M3R_OUTPUT_DIR,
    STEP2M3R_REVIEW_DECISION_SUMMARY_PATH,
    STEP2M3R_REVIEW_PACK_DIR,
    STEP2M3R_REVIEW_PACK_MANIFEST_PATH,
    STEP2M3R_REVIEW_PROGRESS_SUMMARY_PATH,
    STEP2M3R_REVIEWED_TOPOLOGY_DECISIONS_PATH,
    STEP2M3R_SAFETY_GUARDRAIL_AUDIT_PATH,
    STEP2M3R_TOPOLOGY_REVIEW_CANDIDATE_ROWS_PATH,
    STEP2M3R_TOPOLOGY_REVIEW_CONTACT_SHEET_PATH,
    STEP2M3R_TOPOLOGY_REVIEW_UI_HTML_PATH,
    STEP2M3R_VALIDATION_SUMMARY_PATH,
    STEP2_VISUAL_CONTINUITY_DIR,
    read_json,
    write_json,
    write_text,
)
from football_intelligence.step2_visual_continuity.schema import (
    NO_AUTO_PROMOTION,
    PRODUCTION_READY,
    UNSURE_DECISION,
    VISUAL_ONLY_WARNING,
    assert_no_forbidden_keys,
    forbidden_keys_present,
    guardrail_stamp,
    rows_from_payload,
    safe_float,
    safe_int,
    utc_iso,
    visual_stamp,
)


M3R_TARGET_REVIEW_CARDS = 40
M3R_HARD_MAX_REVIEW_CARDS = 60
M3R_CURRENT_REVIEW_VERSION = "step2m3r_topology_qa_v1"
M3R_CURRENT_VISUAL_EVIDENCE_VERSION = "step2m3r_visual_evidence_v2_animation"
M3R_ACCEPT_DECISION = "accept_m3_handoff_visual_continuity"
M3R_REJECT_DECISION = "reject_or_quarantine_m3_handoff_visual_continuity"
M3R_UNSURE_DECISION = UNSURE_DECISION
M3R_RISKY_BUCKETS = {
    "merged_or_ambiguous",
    "high_uncertainty_low_margin",
    "role_state_mismatch",
    "team_colour_ambiguity",
    "official_context_warning",
    "goalkeeper_context_warning",
    "bad_detection_proxy_adjacent",
}


def m3r_guardrail_fields() -> dict[str, Any]:
    return {
        "match_local_only": True,
        "safe_to_apply_globally": False,
        "requires_future_match_validation": True,
        "production_ready": PRODUCTION_READY,
        "no_auto_promotion": NO_AUTO_PROMOTION,
        "human_approved": False,
        "no_identity_tracking_performed": True,
        "no_player_slots_assigned": True,
        "no_goalkeeper_slots_assigned": True,
        "no_expected_22_role_states": True,
        "no_exact_count_forcing": True,
        "no_metric_event_tactical_or_physical_performance_analysis": True,
        "official_referee_exclusion_performed": False,
        "bad_detection_rows_deleted": False,
    }


def step2m3r_output_paths() -> dict[str, Path]:
    return {
        "step2m3r_output_dir": STEP2M3R_OUTPUT_DIR,
        "group_topology_audit_rows": STEP2M3R_GROUP_TOPOLOGY_AUDIT_ROWS_PATH,
        "group_topology_audit_summary": STEP2M3R_GROUP_TOPOLOGY_AUDIT_SUMMARY_PATH,
        "group_topology_audit_sample": STEP2M3R_GROUP_TOPOLOGY_AUDIT_SAMPLE_PATH,
        "accepted_edge_topology_audit_rows": STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_ROWS_JSONL_GZ_PATH,
        "accepted_edge_topology_audit_summary": STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_SUMMARY_PATH,
        "accepted_edge_topology_audit_sample": STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_SAMPLE_PATH,
        "topology_review_candidates": STEP2M3R_TOPOLOGY_REVIEW_CANDIDATE_ROWS_PATH,
        "topology_review_ui": STEP2M3R_TOPOLOGY_REVIEW_UI_HTML_PATH,
        "topology_review_contact_sheet": STEP2M3R_TOPOLOGY_REVIEW_CONTACT_SHEET_PATH,
        "group_timeline_strips": STEP2M3R_GROUP_TIMELINE_STRIPS_DIR,
        "edge_burst_strips": STEP2M3R_EDGE_BURST_STRIPS_DIR,
        "group_timeline_animations": STEP2M3R_GROUP_TIMELINE_ANIMATIONS_DIR,
        "edge_burst_animations": STEP2M3R_EDGE_BURST_ANIMATIONS_DIR,
        "reviewed_topology_decisions": STEP2M3R_REVIEWED_TOPOLOGY_DECISIONS_PATH,
        "review_progress": STEP2M3R_REVIEW_PROGRESS_SUMMARY_PATH,
        "review_decision": STEP2M3R_REVIEW_DECISION_SUMMARY_PATH,
        "handoff_readiness": STEP2M3R_HANDOFF_READINESS_SUMMARY_PATH,
        "validation_summary": STEP2M3R_VALIDATION_SUMMARY_PATH,
        "safety_guardrail_audit": STEP2M3R_SAFETY_GUARDRAIL_AUDIT_PATH,
        "issue_register": STEP2M3R_ISSUE_REGISTER_PATH,
        "freeze_candidate_manifest": STEP2M3R_FREEZE_CANDIDATE_MANIFEST_PATH,
        "review_pack_manifest": STEP2M3R_REVIEW_PACK_MANIFEST_PATH,
    }


def assert_m3r_output_path_isolation() -> None:
    m3r_root = STEP2M3R_OUTPUT_DIR.resolve()
    blocked_roots = [STEP2M1_OUTPUT_DIR.resolve(), STEP2M2_OUTPUT_DIR.resolve(), STEP2M3_OUTPUT_DIR.resolve()]
    for path in step2m3r_output_paths().values():
        resolved = path.resolve()
        if resolved != m3r_root and m3r_root not in resolved.parents:
            raise ValueError(f"Step2.M3R output path is outside the M3R root: {resolved}")
        if any(resolved == root or root in resolved.parents for root in blocked_roots):
            raise ValueError(f"Step2.M3R output path points inside an earlier Step2 folder: {resolved}")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    temp_path = path.with_name(f"{path.stem}.tmp.{digest}{path.suffix}")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def write_jsonl_gz_rows(path: Path, row_iterable: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in row_iterable:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
            handle.write("\n")


def decision_rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    raw_rows = payload.get("rows", payload.get("decisions", []))
    return [dict(row) for row in raw_rows if isinstance(row, dict)] if isinstance(raw_rows, list) else []


def accepted_edge_by_id(accepted_rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("continuity_edge_id", "")): row for row in accepted_rows if row.get("continuity_edge_id")}


def safe_asset_stem(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:80]
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned}_{digest}" if cleaned else digest


def m3r_rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(STEP2M3R_OUTPUT_DIR.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def m3r_abs_asset_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else STEP2M3R_OUTPUT_DIR / path


def node_lookup_from_payload(node_payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[int, list[dict[str, Any]]], dict[int, str]]:
    nodes_by_id: dict[str, dict[str, Any]] = {}
    nodes_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    frame_lookup: dict[int, str] = {}
    for row in rows_from_payload(node_payload):
        visible_id = str(row.get("visible_person_base_id", ""))
        frame_sequence = safe_int(row.get("frame_sequence"), -1)
        if visible_id:
            nodes_by_id[visible_id] = row
        if frame_sequence >= 0:
            nodes_by_frame[frame_sequence].append(row)
            frame_id = str(row.get("frame_id", ""))
            frame_path = STAGE3C_FRAMES_DIR / f"{frame_id}.jpg" if frame_id else Path()
            if frame_id and frame_path.exists():
                frame_lookup[frame_sequence] = str(frame_path)
    return nodes_by_id, nodes_by_frame, frame_lookup


def sample_frame_sequences(start: int, end: int, required_frames: Iterable[int] = (), *, max_frames: int = 8) -> list[int]:
    if start > end:
        start, end = end, start
    candidates = {frame for frame in required_frames if start <= frame <= end}
    candidates.update({start, end})
    span = max(0, end - start)
    if span <= max_frames - 1:
        candidates.update(range(start, end + 1))
    else:
        for index in range(max_frames):
            candidates.add(round(start + (span * index / max(1, max_frames - 1))))
    return sorted(candidates)[:max_frames]


def group_issue_frames(group: dict[str, Any], edges: list[dict[str, Any]], nodes_by_id: dict[str, dict[str, Any]]) -> dict[str, set[int]]:
    frame_members: dict[int, set[str]] = defaultdict(set)
    for frame, member in group_member_pairs(group):
        if frame >= 0 and member:
            frame_members[frame].add(member)
    out_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    in_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        source = str(edge.get("source_visible_person_base_id", ""))
        target = str(edge.get("target_visible_person_base_id", ""))
        if source:
            out_edges[source].append(edge)
        if target:
            in_edges[target].append(edge)
    role_frames: set[int] = set()
    for frame, members in frame_members.items():
        roles = {
            str(nodes_by_id.get(member, {}).get("step1f3_final_visual_role_state", ""))
            for member in members
            if nodes_by_id.get(member, {}).get("step1f3_final_visual_role_state")
        }
        if len(roles) > 1:
            role_frames.add(frame)
    return {
        "duplicate": {frame for frame, members in frame_members.items() if len(members) > 1},
        "branch": {
            safe_int(edge.get("source_frame_sequence"), -1)
            for source_edges in out_edges.values()
            if len(source_edges) > 1
            for edge in source_edges
        },
        "merge": {
            safe_int(edge.get("target_frame_sequence"), -1)
            for target_edges in in_edges.values()
            if len(target_edges) > 1
            for edge in target_edges
        },
        "role": role_frames,
    }


def tile_from_frame(frame_sequence: int, frame_lookup: dict[int, str], *, tile_w: int = 420, tile_h: int = 280) -> tuple[Any | None, dict[str, float]]:
    if cv2 is None or np is None:
        return None, {}
    image_path = frame_lookup.get(frame_sequence, "")
    source_image = cv2.imread(image_path) if image_path and Path(image_path).exists() else None
    canvas = np.full((tile_h, tile_w, 3), 28, dtype=np.uint8)
    image_h = tile_h - 36
    if source_image is None:
        cv2.putText(canvas, f"missing frame {frame_sequence}", (20, tile_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
        return canvas, {"scale_x": 1.0, "scale_y": 1.0, "offset_y": 0.0, "tile_w": float(tile_w), "tile_h": float(tile_h)}
    original_h, original_w = source_image.shape[:2]
    resized = cv2.resize(source_image, (tile_w, image_h), interpolation=cv2.INTER_AREA)
    canvas[32 : 32 + image_h, 0:tile_w] = resized
    metadata = {
        "scale_x": tile_w / max(1, original_w),
        "scale_y": image_h / max(1, original_h),
        "offset_y": 32.0,
        "tile_w": float(tile_w),
        "tile_h": float(tile_h),
    }
    return canvas, metadata


def draw_scaled_box(
    image: Any,
    bbox: dict[str, Any],
    metadata: dict[str, float],
    *,
    label: str,
    colour: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    if cv2 is None or not bbox or not metadata:
        return
    x1 = int(max(0, min(metadata["tile_w"] - 1, safe_float(bbox.get("x1")) * metadata["scale_x"])))
    x2 = int(max(0, min(metadata["tile_w"] - 1, safe_float(bbox.get("x2")) * metadata["scale_x"])))
    y1 = int(max(0, min(metadata["tile_h"] - 1, safe_float(bbox.get("y1")) * metadata["scale_y"] + metadata["offset_y"])))
    y2 = int(max(0, min(metadata["tile_h"] - 1, safe_float(bbox.get("y2")) * metadata["scale_y"] + metadata["offset_y"])))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)
    cv2.putText(image, label[:32], (x1, max(48, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)


def write_image(path: Path, image: Any) -> bool:
    if cv2 is None or image is None:
        return False
    ensure_dir(path.parent)
    return bool(cv2.imwrite(str(path), image))


def write_animation_gif(path: Path, frames: list[Any], *, duration_seconds: float = 0.38) -> bool:
    if cv2 is None or np is None or imageio is None or not frames:
        return False
    ensure_dir(path.parent)
    try:
        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames if frame is not None]
        if not rgb_frames:
            return False
        imageio.mimsave(str(path), rgb_frames, duration=duration_seconds, loop=0)
    except Exception:
        return False
    return path.exists() and path.stat().st_size > 0


def horizontal_image_strip(images: list[Any]) -> Any | None:
    if cv2 is None or np is None or not images:
        return None
    min_h = min(image.shape[0] for image in images)
    resized = []
    for image in images:
        if image.shape[0] == min_h:
            resized.append(image)
            continue
        scale = min_h / max(1, image.shape[0])
        resized.append(cv2.resize(image, (max(1, int(image.shape[1] * scale)), min_h), interpolation=cv2.INTER_AREA))
    return np.hstack(resized)


def draw_visual_only_topology_watermark(image: Any) -> None:
    if cv2 is None or image is None:
        return
    cv2.putText(
        image,
        "not identity / visual-only topology QA",
        (10, image.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (120, 230, 240),
        1,
        cv2.LINE_AA,
    )


def group_member_pairs(group: dict[str, Any]) -> list[tuple[int, str]]:
    frames = list(group.get("member_frame_sequences", []))
    members = list(group.get("member_visible_person_base_ids", []))
    pairs: list[tuple[int, str]] = []
    for index, member in enumerate(members):
        frame = frames[index] if index < len(frames) else None
        pairs.append((safe_int(frame, -1), str(member)))
    return pairs


def edge_endpoint_pairs(edge: dict[str, Any]) -> list[tuple[int, str]]:
    return [
        (safe_int(edge.get("source_frame_sequence"), -1), str(edge.get("source_visible_person_base_id", ""))),
        (safe_int(edge.get("target_frame_sequence"), -1), str(edge.get("target_visible_person_base_id", ""))),
    ]


def normalized_role_count(role_counts: Any) -> dict[str, int]:
    if not isinstance(role_counts, dict):
        return {}
    return {str(key): safe_int(value, 0) for key, value in role_counts.items()}


def group_topology_metrics(group: dict[str, Any], edges: list[dict[str, Any]]) -> dict[str, Any]:
    member_pairs = group_member_pairs(group)
    frame_to_members: dict[int, set[str]] = defaultdict(set)
    for frame, member in member_pairs:
        if frame >= 0 and member:
            frame_to_members[frame].add(member)
    for edge in edges:
        for frame, member in edge_endpoint_pairs(edge):
            if frame >= 0 and member:
                frame_to_members[frame].add(member)
    member_ids = {member for _frame, member in member_pairs if member}
    for edge in edges:
        member_ids.update(member for _frame, member in edge_endpoint_pairs(edge) if member)
    unique_frame_count = len(frame_to_members)
    max_members_per_frame = max((len(members) for members in frame_to_members.values()), default=0)
    frames_with_multiple_members_count = sum(1 for members in frame_to_members.values() if len(members) > 1)
    duplicate_frame_member_ratio = round(frames_with_multiple_members_count / max(1, unique_frame_count), 4)
    out_degree: Counter[str] = Counter()
    in_degree: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    review_source_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    for edge in edges:
        source = str(edge.get("source_visible_person_base_id", ""))
        target = str(edge.get("target_visible_person_base_id", ""))
        if source and target:
            out_degree[source] += 1
            in_degree[target] += 1
        bucket_counts[str(edge.get("source_review_bucket", ""))] += 1
        reason_counts[str(edge.get("m3_acceptance_reason", ""))] += 1
        review_source_counts[str(edge.get("human_review_decision_source", ""))] += 1
        decision_counts[str(edge.get("human_review_decision", ""))] += 1
    max_out_degree = max(out_degree.values(), default=0)
    max_in_degree = max(in_degree.values(), default=0)
    branch_count = sum(1 for degree in out_degree.values() if degree > 1)
    merge_count = sum(1 for degree in in_degree.values() if degree > 1)
    role_counts = normalized_role_count(group.get("role_state_counts_visual_context_only", {}))
    nonzero_role_count = sum(1 for value in role_counts.values() if value > 0)
    has_role_context_mixing = nonzero_role_count > 1
    edge_density = round(len(edges) / max(1, len(member_ids) * max(1, unique_frame_count)), 4)
    return {
        "member_count": len(member_ids),
        "accepted_edge_count": len(edges),
        "unique_frame_count": unique_frame_count,
        "max_members_per_frame": max_members_per_frame,
        "frames_with_multiple_members_count": frames_with_multiple_members_count,
        "duplicate_frame_member_ratio": duplicate_frame_member_ratio,
        "max_out_degree": max_out_degree,
        "max_in_degree": max_in_degree,
        "branch_count": branch_count,
        "merge_count": merge_count,
        "has_branching": branch_count > 0,
        "has_merging": merge_count > 0,
        "role_state_counts_visual_context_only": role_counts,
        "role_context_mixing_count": nonzero_role_count,
        "role_context_count_visual_only": nonzero_role_count,
        "has_role_context_mixing": has_role_context_mixing,
        "edge_density_within_short_window": edge_density,
        "edge_density_visual_graph": edge_density,
        "accepted_bucket_counts": dict(sorted(bucket_counts.items())),
        "accepted_reason_counts": dict(sorted(reason_counts.items())),
        "accepted_human_review_source_counts": dict(sorted(review_source_counts.items())),
        "accepted_human_decision_counts": dict(sorted(decision_counts.items())),
        "frame_to_member_counts_sample": {
            str(frame): len(members) for frame, members in list(sorted(frame_to_members.items()))[:12]
        },
    }


def topology_risk(metrics: dict[str, Any]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0
    if safe_int(metrics.get("frames_with_multiple_members_count"), 0) > 0:
        reasons.append("duplicate_frame_members_visual_topology")
        score += min(0.3, safe_float(metrics.get("duplicate_frame_member_ratio"), 0.0) * 0.5 + 0.12)
    if metrics.get("has_branching") is True:
        reasons.append("branching_visual_topology")
        score += 0.2
    if metrics.get("has_merging") is True:
        reasons.append("merging_visual_topology")
        score += 0.2
    if metrics.get("has_role_context_mixing") is True:
        reasons.append("role_context_mixing_visual_only")
        score += 0.2
    if safe_float(metrics.get("edge_density_visual_graph"), 0.0) > 0.08:
        reasons.append("dense_short_window_visual_graph")
        score += 0.12
    if safe_int(metrics.get("member_count"), 0) >= 26:
        reasons.append("large_visual_continuity_group")
        score += 0.08
    return round(min(score, 1.0), 4), reasons


def make_group_topology_row(group: dict[str, Any], edges: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = group_topology_metrics(group, edges)
    score, reasons = topology_risk(metrics)
    row = {
        "visual_continuity_group_id": str(group.get("visual_continuity_group_id", "")),
        "min_frame_sequence": safe_int(group.get("min_frame_sequence"), -1),
        "max_frame_sequence": safe_int(group.get("max_frame_sequence"), -1),
        "frame_span": safe_int(group.get("frame_span"), 0),
        "seconds_span": safe_float(group.get("seconds_span"), 0.0),
        "max_group_span_frames_allowed": MAX_M3_GROUP_SPAN_FRAMES,
        "max_group_span_seconds_allowed": MAX_M3_GROUP_SPAN_SECONDS,
        "group_over_span_cap": safe_int(group.get("frame_span"), 0) > MAX_M3_GROUP_SPAN_FRAMES
        or safe_float(group.get("seconds_span"), 0.0) > MAX_M3_GROUP_SPAN_SECONDS,
        "topology_risk_score": score,
        "topology_risk_reasons": reasons,
        "high_topology_risk": score >= 0.45 or bool(reasons),
        "topology_review_recommended": bool(reasons),
        "group_not_identity": True,
        "group_not_player_slot": True,
        "group_not_goalkeeper_slot": True,
        "short_window_visual_continuity_only": True,
        **metrics,
        **m3r_guardrail_fields(),
    }
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def build_group_topology_audit(
    group_payload: dict[str, Any],
    accepted_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    edge_lookup = accepted_edge_by_id(accepted_rows)
    group_rows: list[dict[str, Any]] = []
    edges_by_group: dict[str, list[dict[str, Any]]] = {}
    for group in rows_from_payload(group_payload):
        group_id = str(group.get("visual_continuity_group_id", ""))
        edge_ids = [str(edge_id) for edge_id in group.get("accepted_continuity_edge_ids", [])]
        edges = [edge_lookup[edge_id] for edge_id in edge_ids if edge_id in edge_lookup]
        edges_by_group[group_id] = edges
        group_rows.append(make_group_topology_row(group, edges))
    summary = guardrail_stamp(
        {
            "artifact": "step2m3r_group_topology_audit_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "total_groups_audited": len(group_rows),
            "high_topology_risk_group_count": sum(1 for row in group_rows if row.get("high_topology_risk") is True),
            "branch_merge_group_count": sum(1 for row in group_rows if row.get("has_branching") is True or row.get("has_merging") is True),
            "duplicate_frame_member_group_count": sum(1 for row in group_rows if safe_int(row.get("frames_with_multiple_members_count"), 0) > 0),
            "role_context_mixed_group_count": sum(1 for row in group_rows if row.get("has_role_context_mixing") is True),
            "groups_over_cap_count": sum(1 for row in group_rows if row.get("group_over_span_cap") is True),
            "max_topology_risk_score": max((safe_float(row.get("topology_risk_score"), 0.0) for row in group_rows), default=0.0),
            "max_members_per_frame_observed": max((safe_int(row.get("max_members_per_frame"), 0) for row in group_rows), default=0),
            "max_branch_count_observed": max((safe_int(row.get("branch_count"), 0) for row in group_rows), default=0),
            "max_merge_count_observed": max((safe_int(row.get("merge_count"), 0) for row in group_rows), default=0),
            "sample_json_path": str(STEP2M3R_GROUP_TOPOLOGY_AUDIT_SAMPLE_PATH.resolve()),
            **m3r_guardrail_fields(),
        }
    )
    payload = guardrail_stamp(
        {
            "artifact": "step2m3r_group_topology_audit_rows",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "rows": sorted(group_rows, key=lambda row: (-safe_float(row.get("topology_risk_score"), 0.0), str(row.get("visual_continuity_group_id", "")))),
            **m3r_guardrail_fields(),
        }
    )
    sample = guardrail_stamp(
        {
            "artifact": "step2m3r_group_topology_audit_sample",
            "created_at": utc_iso(),
            "sample_rows": min(80, len(group_rows)),
            "total_rows": len(group_rows),
            "rows": rows_from_payload(payload)[:80],
            **m3r_guardrail_fields(),
        }
    )
    for item in [payload, summary, sample]:
        assert_no_forbidden_keys(item)
    return payload, summary, sample, edges_by_group


def edge_is_branch_merge_point(edge: dict[str, Any], group_row: dict[str, Any]) -> bool:
    if group_row.get("has_branching") is True or group_row.get("has_merging") is True:
        return True
    return False


def make_edge_topology_row(
    edge: dict[str, Any],
    *,
    group_id: str,
    group_row: dict[str, Any],
) -> dict[str, Any]:
    source_frame = safe_int(edge.get("source_frame_sequence"), -1)
    target_frame = safe_int(edge.get("target_frame_sequence"), -1)
    frame_gap = safe_int(edge.get("frame_gap"), abs(target_frame - source_frame))
    reasons = list(group_row.get("topology_risk_reasons", []))
    bucket = str(edge.get("source_review_bucket", ""))
    if bucket in M3R_RISKY_BUCKETS and "risky_source_review_bucket" not in reasons:
        reasons.append("risky_source_review_bucket")
    row = {
        "continuity_edge_id": str(edge.get("continuity_edge_id", "")),
        "visual_continuity_group_id": group_id,
        "source_visible_person_base_id": str(edge.get("source_visible_person_base_id", "")),
        "target_visible_person_base_id": str(edge.get("target_visible_person_base_id", "")),
        "source_frame_sequence": source_frame,
        "target_frame_sequence": target_frame,
        "frame_gap": frame_gap,
        "source_review_bucket": bucket,
        "m3_acceptance_reason": str(edge.get("m3_acceptance_reason", "")),
        "human_review_decision_source": str(edge.get("human_review_decision_source", "")),
        "human_review_decision": str(edge.get("human_review_decision", "")),
        "topology_group_high_risk": group_row.get("high_topology_risk") is True,
        "topology_group_risk_score": safe_float(group_row.get("topology_risk_score"), 0.0),
        "branch_merge_point_visual_only": edge_is_branch_merge_point(edge, group_row),
        "duplicate_frame_group_segment_visual_only": safe_int(group_row.get("frames_with_multiple_members_count"), 0) > 0,
        "role_context_mixed_group_visual_only": group_row.get("has_role_context_mixing") is True,
        "source_target_role_visual_context_differs": group_row.get("has_role_context_mixing") is True,
        "topology_review_reasons": reasons,
        "short_window_visual_continuity_only": True,
        **m3r_guardrail_fields(),
    }
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def build_edge_topology_audit(
    group_audit_rows: list[dict[str, Any]],
    edges_by_group: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    group_lookup = {str(row.get("visual_continuity_group_id", "")): row for row in group_audit_rows}
    edge_rows: list[dict[str, Any]] = []
    for group_id, edges in edges_by_group.items():
        group_row = group_lookup.get(group_id, {})
        for edge in edges:
            edge_rows.append(make_edge_topology_row(edge, group_id=group_id, group_row=group_row))
    reason_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    for row in edge_rows:
        bucket_counts[str(row.get("source_review_bucket", ""))] += 1
        reason_counts.update(str(reason) for reason in row.get("topology_review_reasons", []))
    summary = guardrail_stamp(
        {
            "artifact": "step2m3r_accepted_edge_topology_audit_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "accepted_edges_audited": len(edge_rows),
            "high_topology_risk_edge_count": sum(1 for row in edge_rows if row.get("topology_group_high_risk") is True),
            "branch_merge_point_edge_count": sum(1 for row in edge_rows if row.get("branch_merge_point_visual_only") is True),
            "duplicate_frame_group_segment_edge_count": sum(1 for row in edge_rows if row.get("duplicate_frame_group_segment_visual_only") is True),
            "role_context_mixed_group_edge_count": sum(1 for row in edge_rows if row.get("role_context_mixed_group_visual_only") is True),
            "source_review_bucket_counts": dict(sorted(bucket_counts.items())),
            "topology_review_reason_counts": dict(sorted(reason_counts.items())),
            "full_rows_jsonl_gz_path": str(STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_ROWS_JSONL_GZ_PATH.resolve()),
            "sample_json_path": str(STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_SAMPLE_PATH.resolve()),
            **m3r_guardrail_fields(),
        }
    )
    sample = guardrail_stamp(
        {
            "artifact": "step2m3r_accepted_edge_topology_audit_sample",
            "created_at": utc_iso(),
            "sample_rows": min(80, len(edge_rows)),
            "total_rows": len(edge_rows),
            "rows": sorted(edge_rows, key=lambda row: (-safe_float(row.get("topology_group_risk_score"), 0.0), str(row.get("continuity_edge_id", ""))))[:80],
            **m3r_guardrail_fields(),
        }
    )
    for item in [summary, sample]:
        assert_no_forbidden_keys(item)
    return edge_rows, summary, sample


def review_candidate_from_group(row: dict[str, Any], category: str, index: int) -> dict[str, Any]:
    candidate = {
        "step2m3r_topology_review_candidate_id": f"step2m3r_topology_card_{index:03d}",
        "step2m3r_review_category": category,
        "review_subject_type": "visual_continuity_group",
        "visual_continuity_group_id": str(row.get("visual_continuity_group_id", "")),
        "min_frame_sequence": safe_int(row.get("min_frame_sequence"), -1),
        "max_frame_sequence": safe_int(row.get("max_frame_sequence"), -1),
        "frame_span": safe_int(row.get("frame_span"), 0),
        "seconds_span": safe_float(row.get("seconds_span"), 0.0),
        "member_count": safe_int(row.get("member_count"), 0),
        "accepted_edge_count": safe_int(row.get("accepted_edge_count"), 0),
        "topology_risk_score": safe_float(row.get("topology_risk_score"), 0.0),
        "topology_risk_reasons": list(row.get("topology_risk_reasons", [])),
        "review_instruction": "Review whether the visual-only group topology is suitable for handoff, or whether it should be quarantined for later visual inspection.",
        "review_question": "Is this M3 visual-continuity group/edge visually coherent enough for short-window continuity handoff, without treating it as identity?",
        "allowed_keyboard_decisions": {
            "A": M3R_ACCEPT_DECISION,
            "X": M3R_REJECT_DECISION,
            "U": M3R_UNSURE_DECISION,
        },
        "short_window_visual_continuity_only": True,
        **m3r_guardrail_fields(),
    }
    visual_stamp(candidate)
    assert_no_forbidden_keys(candidate)
    return candidate


def review_candidate_from_edge(row: dict[str, Any], category: str, index: int) -> dict[str, Any]:
    candidate = {
        "step2m3r_topology_review_candidate_id": f"step2m3r_topology_card_{index:03d}",
        "step2m3r_review_category": category,
        "review_subject_type": "accepted_visual_continuity_edge",
        "visual_continuity_group_id": str(row.get("visual_continuity_group_id", "")),
        "continuity_edge_id": str(row.get("continuity_edge_id", "")),
        "source_visible_person_base_id": str(row.get("source_visible_person_base_id", "")),
        "target_visible_person_base_id": str(row.get("target_visible_person_base_id", "")),
        "source_frame_sequence": safe_int(row.get("source_frame_sequence"), -1),
        "target_frame_sequence": safe_int(row.get("target_frame_sequence"), -1),
        "frame_gap": safe_int(row.get("frame_gap"), 0),
        "source_review_bucket": str(row.get("source_review_bucket", "")),
        "topology_group_risk_score": safe_float(row.get("topology_group_risk_score"), 0.0),
        "topology_review_reasons": list(row.get("topology_review_reasons", [])),
        "review_instruction": "Review whether this accepted visual-only edge is topology-safe for handoff, or whether it should be quarantined for later visual inspection.",
        "review_question": "Is this M3 visual-continuity group/edge visually coherent enough for short-window continuity handoff, without treating it as identity?",
        "allowed_keyboard_decisions": {
            "A": M3R_ACCEPT_DECISION,
            "X": M3R_REJECT_DECISION,
            "U": M3R_UNSURE_DECISION,
        },
        "short_window_visual_continuity_only": True,
        **m3r_guardrail_fields(),
    }
    visual_stamp(candidate)
    assert_no_forbidden_keys(candidate)
    return candidate


def pick_unique(items: Iterable[dict[str, Any]], key_name: str, used: set[str], limit: int) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    for item in items:
        key = str(item.get(key_name, ""))
        if not key or key in used:
            continue
        used.add(key)
        picked.append(item)
        if len(picked) >= limit:
            break
    return picked


def build_topology_review_queue(
    group_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    used_groups: set[str] = set()
    used_edges: set[str] = set()
    raw_candidates: list[tuple[str, dict[str, Any]]] = []
    branch_merge = sorted(
        [row for row in group_rows if row.get("has_branching") is True or row.get("has_merging") is True],
        key=lambda row: (-safe_float(row.get("topology_risk_score"), 0.0), -safe_int(row.get("accepted_edge_count"), 0)),
    )
    duplicate_frames = sorted(
        [row for row in group_rows if safe_int(row.get("frames_with_multiple_members_count"), 0) > 0],
        key=lambda row: (-safe_int(row.get("frames_with_multiple_members_count"), 0), -safe_float(row.get("topology_risk_score"), 0.0)),
    )
    role_mixing = sorted(
        [row for row in group_rows if row.get("has_role_context_mixing") is True],
        key=lambda row: (-safe_float(row.get("topology_risk_score"), 0.0), -safe_int(row.get("role_context_count_visual_only"), 0)),
    )
    risky_edges = sorted(
        [row for row in edge_rows if str(row.get("source_review_bucket", "")) in M3R_RISKY_BUCKETS],
        key=lambda row: (-safe_float(row.get("topology_group_risk_score"), 0.0), str(row.get("continuity_edge_id", ""))),
    )
    clean_controls = sorted(
        [
            row
            for row in group_rows
            if row.get("high_topology_risk") is not True
            and safe_int(row.get("frames_with_multiple_members_count"), 0) == 0
            and row.get("has_role_context_mixing") is not True
        ],
        key=lambda row: (-safe_int(row.get("accepted_edge_count"), 0), str(row.get("visual_continuity_group_id", ""))),
    )
    for row in pick_unique(branch_merge, "visual_continuity_group_id", used_groups, 8):
        raw_candidates.append(("branch_merge_topology_group", row))
    for row in pick_unique(duplicate_frames, "visual_continuity_group_id", used_groups, 8):
        raw_candidates.append(("duplicate_frame_member_group", row))
    for row in pick_unique(role_mixing, "visual_continuity_group_id", used_groups, 8):
        raw_candidates.append(("role_context_mixed_group", row))
    for row in pick_unique(risky_edges, "continuity_edge_id", used_edges, 8):
        raw_candidates.append(("risky_accepted_edge", row))
    for row in pick_unique(clean_controls, "visual_continuity_group_id", used_groups, 8):
        raw_candidates.append(("clean_topology_control_group", row))
    if len(raw_candidates) < M3R_TARGET_REVIEW_CARDS:
        fallback_groups = sorted(group_rows, key=lambda row: (-safe_float(row.get("topology_risk_score"), 0.0), str(row.get("visual_continuity_group_id", ""))))
        for row in pick_unique(fallback_groups, "visual_continuity_group_id", used_groups, M3R_TARGET_REVIEW_CARDS - len(raw_candidates)):
            raw_candidates.append(("topology_risk_fallback_group", row))
    candidates: list[dict[str, Any]] = []
    for index, (category, row) in enumerate(raw_candidates[:M3R_HARD_MAX_REVIEW_CARDS], start=1):
        if "continuity_edge_id" in row:
            candidates.append(review_candidate_from_edge(row, category, index))
        else:
            candidates.append(review_candidate_from_group(row, category, index))
    category_counts = Counter(str(row.get("step2m3r_review_category", "")) for row in candidates)
    payload = guardrail_stamp(
        {
            "artifact": "step2m3r_topology_review_candidate_rows",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "target_review_cards": M3R_TARGET_REVIEW_CARDS,
            "hard_max_review_cards": M3R_HARD_MAX_REVIEW_CARDS,
            "summary": {
                "review_queue_size": len(candidates),
                "review_queue_hard_max": M3R_HARD_MAX_REVIEW_CARDS,
                "review_category_counts": dict(sorted(category_counts.items())),
                "review_version": M3R_CURRENT_REVIEW_VERSION,
            },
            "rows": candidates,
            **m3r_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(payload)
    return payload


def render_group_timeline_strip(
    candidate: dict[str, Any],
    group: dict[str, Any],
    edges: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
    frame_lookup: dict[int, str],
) -> dict[str, Any]:
    group_id = str(candidate.get("visual_continuity_group_id", ""))
    stem = safe_asset_stem(str(candidate.get("step2m3r_topology_review_candidate_id", group_id)))
    strip_path = STEP2M3R_GROUP_TIMELINE_STRIPS_DIR / f"{stem}_group_timeline.jpg"
    animation_path = STEP2M3R_GROUP_TIMELINE_ANIMATIONS_DIR / f"{stem}_group_timeline.gif"
    issue_frames = group_issue_frames(group, edges, nodes_by_id)
    required = set().union(*issue_frames.values()) if issue_frames else set()
    start = safe_int(group.get("min_frame_sequence", candidate.get("min_frame_sequence")), -1)
    end = safe_int(group.get("max_frame_sequence", candidate.get("max_frame_sequence")), -1)
    sampled_frames = sample_frame_sequences(start, end, required, max_frames=16)
    group_members_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame, member_id in group_member_pairs(group):
        node = nodes_by_id.get(member_id)
        if node:
            group_members_by_frame[frame].append(node)
    rendered: list[Any] = []
    for frame in sampled_frames:
        tile, metadata = tile_from_frame(frame, frame_lookup)
        if tile is None:
            continue
        flags = [name for name, values in issue_frames.items() if frame in values]
        title = f"group {group_id[-12:]} frame {frame}"
        cv2.putText(tile, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
        if flags:
            cv2.putText(tile, ",".join(flags)[:42], (210, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 180, 255), 1, cv2.LINE_AA)
        for node in group_members_by_frame.get(frame, []):
            colour = (80, 240, 255)
            if frame in issue_frames["duplicate"]:
                colour = (40, 40, 255)
            elif frame in issue_frames["branch"] or frame in issue_frames["merge"]:
                colour = (0, 170, 255)
            elif frame in issue_frames["role"]:
                colour = (255, 80, 190)
            label = f"{node.get('visible_person_base_id', '')[-8:]} {node.get('step1f3_final_visual_role_state', '')[:12]}"
            draw_scaled_box(tile, node.get("bbox", {}), metadata, label=label, colour=colour, thickness=2)
        draw_visual_only_topology_watermark(tile)
        rendered.append(tile)
    strip = horizontal_image_strip(rendered)
    static_available = bool(strip is not None and write_image(strip_path, strip))
    animation_available = write_animation_gif(animation_path, rendered)
    available = animation_available and static_available
    return {
        "evidence_available": available,
        "evidence_type": "group_timeline_animation",
        "evidence_animation_gif_path": m3r_rel_path(animation_path) if animation_available else "",
        "evidence_animation_mp4_path": "",
        "evidence_static_strip_path": m3r_rel_path(strip_path) if static_available else "",
        "evidence_image_path": m3r_rel_path(strip_path) if static_available else "",
        "comparison_image_path": "",
        "sampled_frame_sequences": sampled_frames,
        "overlay_version": M3R_CURRENT_VISUAL_EVIDENCE_VERSION,
        "current_visual_evidence_version": M3R_CURRENT_VISUAL_EVIDENCE_VERSION,
        "animation_evidence_available": animation_available,
        "static_strip_fallback_available": static_available,
        "visual_evidence_missing_reason": "" if available else "group_timeline_animation_or_strip_not_rendered",
    }


def render_edge_burst_strip(
    candidate: dict[str, Any],
    edge: dict[str, Any],
    group: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
    frame_lookup: dict[int, str],
) -> dict[str, Any]:
    edge_id = str(candidate.get("continuity_edge_id", edge.get("continuity_edge_id", "")))
    stem = safe_asset_stem(str(candidate.get("step2m3r_topology_review_candidate_id", edge_id)))
    strip_path = STEP2M3R_EDGE_BURST_STRIPS_DIR / f"{stem}_edge_burst.jpg"
    animation_path = STEP2M3R_EDGE_BURST_ANIMATIONS_DIR / f"{stem}_edge_burst.gif"
    source_frame = safe_int(edge.get("source_frame_sequence", candidate.get("source_frame_sequence")), -1)
    target_frame = safe_int(edge.get("target_frame_sequence", candidate.get("target_frame_sequence")), -1)
    start = min(source_frame, target_frame)
    end = max(source_frame, target_frame)
    group_members_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame, member_id in group_member_pairs(group):
        node = nodes_by_id.get(member_id)
        if node:
            group_members_by_frame[frame].append(node)
    source_id = str(edge.get("source_visible_person_base_id", ""))
    target_id = str(edge.get("target_visible_person_base_id", ""))
    for frame, member_id in [(source_frame, source_id), (target_frame, target_id)]:
        node = nodes_by_id.get(member_id)
        if node and all(str(existing.get("visible_person_base_id", "")) != member_id for existing in group_members_by_frame[frame]):
            group_members_by_frame[frame].append(node)
    sampled_frames = [frame for frame in range(start, end + 1) if frame in frame_lookup or group_members_by_frame.get(frame)]
    if not sampled_frames:
        sampled_frames = list(range(start, end + 1))
    if len(sampled_frames) > 16:
        sampled_frames = sample_frame_sequences(start, end, {source_frame, target_frame}, max_frames=16)
    rendered: list[Any] = []
    for frame in sampled_frames:
        tile, metadata = tile_from_frame(frame, frame_lookup)
        if tile is None:
            continue
        cv2.putText(tile, f"edge {edge_id[-14:]} frame {frame}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
        for node in group_members_by_frame.get(frame, []):
            node_id = str(node.get("visible_person_base_id", ""))
            colour = (180, 190, 200)
            label = f"group {node_id[-8:]}"
            thickness = 1
            if frame == source_frame and node_id == source_id:
                colour = (0, 215, 255)
                label = f"source {node_id[-8:]}"
                thickness = 3
            elif frame == target_frame and node_id == target_id:
                colour = (90, 245, 120)
                label = f"target {node_id[-8:]}"
                thickness = 3
            draw_scaled_box(tile, node.get("bbox", {}), metadata, label=label, colour=colour, thickness=thickness)
        draw_visual_only_topology_watermark(tile)
        rendered.append(tile)
    strip = horizontal_image_strip(rendered)
    static_available = bool(strip is not None and write_image(strip_path, strip))
    animation_available = write_animation_gif(animation_path, rendered)
    available = animation_available and static_available
    return {
        "evidence_available": available,
        "evidence_type": "edge_burst_animation",
        "evidence_animation_gif_path": m3r_rel_path(animation_path) if animation_available else "",
        "evidence_animation_mp4_path": "",
        "evidence_static_strip_path": m3r_rel_path(strip_path) if static_available else "",
        "evidence_image_path": m3r_rel_path(strip_path) if static_available else "",
        "comparison_image_path": "",
        "sampled_frame_sequences": sampled_frames,
        "overlay_version": M3R_CURRENT_VISUAL_EVIDENCE_VERSION,
        "current_visual_evidence_version": M3R_CURRENT_VISUAL_EVIDENCE_VERSION,
        "animation_evidence_available": animation_available,
        "static_strip_fallback_available": static_available,
        "visual_evidence_missing_reason": "" if available else "edge_burst_animation_or_strip_not_rendered",
    }


def visual_evidence_summary(review_payload: dict[str, Any]) -> dict[str, Any]:
    rows = rows_from_payload(review_payload)
    animation_available = sum(1 for row in rows if row.get("animation_evidence_available") is True and row.get("evidence_animation_gif_path"))
    animation_missing = len(rows) - animation_available
    animation_missing_rate = round(animation_missing / max(1, len(rows)), 4)
    static_available = sum(1 for row in rows if row.get("static_strip_fallback_available") is True and row.get("evidence_static_strip_path"))
    available = sum(
        1
        for row in rows
        if row.get("evidence_available") is True
        and row.get("animation_evidence_available") is True
        and row.get("evidence_animation_gif_path")
    )
    missing = len(rows) - available
    missing_rate = round(missing / max(1, len(rows)), 4)
    return {
        "visual_evidence_available_count": available,
        "visual_evidence_missing_count": missing,
        "visual_evidence_missing_rate": missing_rate,
        "visual_evidence_safe_for_review": animation_missing == 0 and len(rows) > 0,
        "animation_evidence_available_count": animation_available,
        "animation_evidence_missing_count": animation_missing,
        "animation_evidence_missing_rate": animation_missing_rate,
        "animation_evidence_safe_for_review": animation_missing == 0 and len(rows) > 0,
        "static_strip_fallback_available_count": static_available,
        "current_visual_evidence_version": M3R_CURRENT_VISUAL_EVIDENCE_VERSION,
    }


def add_visual_evidence_to_review_payload(
    review_payload: dict[str, Any],
    *,
    source_groups_by_id: dict[str, dict[str, Any]],
    edges_by_group: dict[str, list[dict[str, Any]]],
    accepted_edges_by_id: dict[str, dict[str, Any]],
    node_payload: dict[str, Any],
) -> dict[str, Any]:
    nodes_by_id, _nodes_by_frame, frame_lookup = node_lookup_from_payload(node_payload)
    ensure_dir(STEP2M3R_GROUP_TIMELINE_STRIPS_DIR)
    ensure_dir(STEP2M3R_EDGE_BURST_STRIPS_DIR)
    ensure_dir(STEP2M3R_GROUP_TIMELINE_ANIMATIONS_DIR)
    ensure_dir(STEP2M3R_EDGE_BURST_ANIMATIONS_DIR)
    enriched_rows: list[dict[str, Any]] = []
    for row in rows_from_payload(review_payload):
        group_id = str(row.get("visual_continuity_group_id", ""))
        subject_type = str(row.get("review_subject_type", ""))
        if subject_type == "accepted_visual_continuity_edge":
            edge = accepted_edges_by_id.get(str(row.get("continuity_edge_id", "")), {})
            evidence = render_edge_burst_strip(row, edge, source_groups_by_id.get(group_id, {}), nodes_by_id, frame_lookup)
        else:
            evidence = render_group_timeline_strip(row, source_groups_by_id.get(group_id, {}), edges_by_group.get(group_id, []), nodes_by_id, frame_lookup)
        enriched_rows.append({**row, **evidence})
    enriched_payload = dict(review_payload)
    enriched_payload["rows"] = enriched_rows
    summary = dict(enriched_payload.get("summary", {}))
    summary.update(visual_evidence_summary(enriched_payload))
    enriched_payload["summary"] = summary
    assert_no_forbidden_keys(enriched_payload)
    return enriched_payload


def render_topology_contact_sheet(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    ensure_dir(STEP2M3R_TOPOLOGY_REVIEW_CONTACT_SHEET_PATH.parent)
    if cv2 is None or np is None:
        STEP2M3R_TOPOLOGY_REVIEW_CONTACT_SHEET_PATH.write_bytes(b"STEP2M3R_TOPOLOGY_CONTACT_SHEET_FALLBACK")
        return {
            "topology_review_contact_sheet_path": str(STEP2M3R_TOPOLOGY_REVIEW_CONTACT_SHEET_PATH.resolve()),
            "fallback_image": True,
            "contains_visual_evidence_thumbnails": False,
            "animation_evidence_label_count": 0,
        }
    cell_w, cell_h = 520, 250
    thumb_w, thumb_h = 500, 190
    rows_to_render = max(1, min(len(candidates), 40))
    sheet = np.full((cell_h * rows_to_render, cell_w, 3), 245, dtype=np.uint8)
    thumbnail_count = 0
    animation_label_count = 0
    for idx, candidate in enumerate(candidates[:rows_to_render]):
        y = idx * cell_h
        cv2.rectangle(sheet, (0, y), (cell_w - 1, y + cell_h - 1), (190, 190, 190), 1)
        static_path_value = str(candidate.get("evidence_static_strip_path") or candidate.get("evidence_image_path", ""))
        evidence_path = m3r_abs_asset_path(static_path_value)
        evidence = cv2.imread(str(evidence_path)) if evidence_path.exists() else None
        if evidence is not None:
            thumbnail_count += 1
            thumbnail = cv2.resize(evidence, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            sheet[y + 4 : y + 4 + thumb_h, 10 : 10 + thumb_w] = thumbnail
        else:
            cv2.putText(sheet, "missing evidence", (20, y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 180), 2, cv2.LINE_AA)
        title = f"{idx + 1:02d} {candidate.get('step2m3r_review_category', '')}"
        subject = f"group={candidate.get('visual_continuity_group_id', '')[-16:]} edge={candidate.get('continuity_edge_id', '')[-16:]}"
        frames = ",".join(str(frame) for frame in candidate.get("sampled_frame_sequences", [])[:8])
        animation_label = "animation: yes" if candidate.get("animation_evidence_available") is True else "animation: no"
        if candidate.get("animation_evidence_available") is True:
            animation_label_count += 1
        cv2.putText(sheet, title, (12, y + 214), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(sheet, subject, (12, y + 232), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(sheet, animation_label, (260, y + 214), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 110, 50), 1, cv2.LINE_AA)
        cv2.putText(sheet, f"frames {frames}", (260, y + 232), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 70), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(STEP2M3R_TOPOLOGY_REVIEW_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"Failed to write M3R topology contact sheet: {STEP2M3R_TOPOLOGY_REVIEW_CONTACT_SHEET_PATH}")
    return {
        "topology_review_contact_sheet_path": str(STEP2M3R_TOPOLOGY_REVIEW_CONTACT_SHEET_PATH.resolve()),
        "fallback_image": False,
        "contains_visual_evidence_thumbnails": thumbnail_count == rows_to_render,
        "visual_evidence_thumbnail_count": thumbnail_count,
        "animation_evidence_label_count": animation_label_count,
    }


def topology_review_ui_html(review_payload: dict[str, Any]) -> str:
    payload_json = json.dumps(review_payload, indent=2).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Step2.M3R Topology QA Review</title>
<style>
:root{{font-family:Arial,sans-serif;color:#18202a;background:#f4f5f7;}}
body{{margin:0;}}
header{{position:sticky;top:0;background:#18202a;color:#fff;padding:12px 18px;z-index:2;display:flex;gap:16px;align-items:center;justify-content:space-between;}}
main{{max-width:1160px;margin:0 auto;padding:18px;}}
button{{border:0;border-radius:6px;padding:10px 14px;font-weight:700;cursor:pointer;}}
.accept{{background:#157347;color:#fff;}} .reject{{background:#b02a37;color:#fff;}} .unsure{{background:#8a6d1d;color:#fff;}}
.nav{{background:#d8dee6;color:#18202a;}}
.card{{background:#fff;border:1px solid #d8dee6;border-radius:8px;padding:18px;margin-bottom:14px;box-shadow:0 1px 2px rgba(0,0,0,.05);}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;}}
.field{{background:#f7f8fa;border:1px solid #e1e5ea;border-radius:6px;padding:10px;min-height:52px;}}
.label{{font-size:12px;color:#5b6675;text-transform:uppercase;letter-spacing:.02em;}}
.value{{font-size:15px;word-break:break-word;margin-top:4px;}}
.risk{{font-size:22px;font-weight:800;}}
.warning{{color:#8a2d12;font-weight:700;}}
.evidence{{width:100%;max-height:64vh;object-fit:contain;background:#0f1720;border:1px solid #c8d0da;border-radius:6px;}}
.evidence-static{{width:100%;max-height:24vh;object-fit:contain;background:#0f1720;border:1px solid #d8dee6;border-radius:6px;margin-top:8px;}}
.evidence-label{{font-size:12px;color:#5b6675;text-transform:uppercase;margin-top:12px;font-weight:700;}}
.card-actions{{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0;}}
.media-actions{{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0;}}
#status{{font-weight:700;}} #status.error{{color:#ffb3b3;}}
pre{{white-space:pre-wrap;background:#111827;color:#f9fafb;border-radius:6px;padding:12px;overflow:auto;}}
</style>
</head>
<body>
<header>
  <div><strong>Step2.M3R Topology QA</strong> <span id="counter"></span></div>
  <div id="status">ready</div>
</header>
<main>
  <section class="card">
    <p class="warning">Visual-only topology QA. Do not infer identity. Do not use for metrics, events, tactics, or physical-performance analysis.</p>
    <p>Is this M3 visual-continuity group/edge visually coherent enough for short-window continuity handoff, without treating it as identity?</p>
    <p>Decision rule: accept only if the visual-only topology looks suitable for handoff. Reject if the group or edge shows a visible branch, merge, duplicate-frame person conflict, or role-context mix that should be quarantined. Use unsure when the evidence is ambiguous.</p>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <button class="accept" onclick="decide('accept_m3_handoff_visual_continuity')">A Accept</button>
      <button class="reject" onclick="decide('reject_or_quarantine_m3_handoff_visual_continuity')">X Reject</button>
      <button class="unsure" onclick="decide('unsure_needs_later_review')">U Unsure</button>
      <button class="nav" onclick="prevCard()">Left</button>
      <button class="nav" onclick="nextCard()">Right</button>
      <button class="nav" onclick="exportDecisions()">Export Backup</button>
    </div>
  </section>
  <section id="card" class="card"></section>
</main>
<script>
const state={payload_json};
let index=Number(localStorage.getItem('step2m3r_topology_index')||0);
const decisions=JSON.parse(localStorage.getItem('step2m3r_topology_decisions')||'{{}}');
function esc(v){{return String(v??'').replace(/[&<>"]/g,s=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[s]));}}
function setStatus(text,isError=false){{const el=document.getElementById('status'); el.textContent=text; el.className=isError?'error':'';}}
function field(label,value){{return `<div class="field"><div class="label">${{esc(label)}}</div><div class="value">${{esc(value)}}</div></div>`;}}
function evidenceHtml(row){{
  const mp4=row.evidence_animation_mp4_path||'';
  const gif=row.evidence_animation_gif_path||'';
  const strip=row.evidence_static_strip_path||row.evidence_image_path||'';
  let primary='';
  if(mp4){{
    primary=`<video id="evidenceVideo" class="evidence" src="${{esc(mp4)}}" controls loop muted playsinline></video><div class="media-actions"><button class="nav" onclick="togglePlayback()">Play / Pause</button></div>`;
  }}else if(gif){{
    primary=`<img class="evidence" src="${{esc(gif)}}" alt="animated visual evidence for review card">`;
  }}else{{
    primary='<div class="field warning">Animated evidence missing. Do not complete this card.</div>';
  }}
  const fallback=strip ? `<div class="evidence-label">Static Strip Fallback</div><img class="evidence-static" src="${{esc(strip)}}" alt="static visual evidence strip">` : '<div class="field warning">Static strip fallback missing.</div>';
  return `${{primary}}${{fallback}}`;
}}
function togglePlayback(){{
  const video=document.getElementById('evidenceVideo');
  if(!video)return;
  if(video.paused){{video.play().catch(()=>{{}});}}else{{video.pause();}}
}}
function show(){{
  const rows=state.rows||[];
  if(!rows.length){{document.getElementById('card').innerHTML='<p>No topology review candidates.</p>'; return;}}
  index=Math.max(0,Math.min(index,rows.length-1));
  localStorage.setItem('step2m3r_topology_index',String(index));
  const row=rows[index];
  document.getElementById('counter').textContent=`${{index+1}} / ${{rows.length}}`;
  const evidence=evidenceHtml(row);
  document.getElementById('card').innerHTML=`
    <h2>${{esc(row.step2m3r_review_category)}} <span class="risk">${{esc(row.topology_risk_score ?? row.topology_group_risk_score ?? 0)}}</span></h2>
    <p class="warning">Do not infer identity. Judge only visual handoff coherence.</p>
    ${{evidence}}
    <div class="card-actions">
      <button class="accept" onclick="decide('accept_m3_handoff_visual_continuity')">A Accept</button>
      <button class="reject" onclick="decide('reject_or_quarantine_m3_handoff_visual_continuity')">X Reject</button>
      <button class="unsure" onclick="decide('unsure_needs_later_review')">U Unsure</button>
    </div>
    <div class="grid">
      ${{field('candidate',row.step2m3r_topology_review_candidate_id)}}
      ${{field('subject',row.review_subject_type)}}
      ${{field('evidence',`${{row.evidence_type || ''}} / ${{row.current_visual_evidence_version || row.overlay_version || ''}}`)}}
      ${{field('animation evidence',row.animation_evidence_available === true ? 'available' : 'missing')}}
      ${{field('static fallback',row.static_strip_fallback_available === true ? 'available' : 'missing')}}
      ${{field('sampled frames',(row.sampled_frame_sequences || []).join(', '))}}
      ${{field('group',row.visual_continuity_group_id)}}
      ${{field('edge',row.continuity_edge_id||'')}}
      ${{field('frames',`${{row.source_frame_sequence ?? row.min_frame_sequence ?? ''}} -> ${{row.target_frame_sequence ?? row.max_frame_sequence ?? ''}}`)}}
      ${{field('bucket',row.source_review_bucket||'')}}
      ${{field('members',row.member_count||'')}}
      ${{field('accepted edges',row.accepted_edge_count||'')}}
    </div>
    <h3>Risk Reasons</h3>
    <pre>${{esc(JSON.stringify(row.topology_risk_reasons || row.topology_review_reasons || [], null, 2))}}</pre>
    <h3>Visual-Only Warning</h3>
    <p>${{esc(row.visual_only_warning)}} | production_ready=${{esc(row.production_ready)}} | no_auto_promotion=${{esc(row.no_auto_promotion)}}</p>
  `;
  const saved=decisions[row.step2m3r_topology_review_candidate_id];
  setStatus(saved ? `saved: ${{saved.human_review_decision}}` : 'ready');
}}
async function decide(decision){{
  const row=(state.rows||[])[index];
  if(!row)return;
  const payload={{
    step2m3r_topology_review_candidate_id: row.step2m3r_topology_review_candidate_id,
    human_review_decision: decision,
    reviewer_name: localStorage.getItem('step2m3r_reviewer_name') || '',
    notes: '',
  }};
  try{{
    setStatus('saving...');
    const response=await fetch('/api/step2m3r/topology-review-decision',{{method:'POST',headers:{{'content-type':'application/json'}},body:JSON.stringify(payload)}});
    const body=await response.json();
    if(!response.ok || !body.success)throw new Error(body.error||'save failed');
    decisions[row.step2m3r_topology_review_candidate_id]={{...payload,reviewed_at:new Date().toISOString()}};
    localStorage.setItem('step2m3r_topology_decisions',JSON.stringify(decisions));
    setStatus(`saved to disk (${{body.reviewed_count}} / ${{body.total_review_candidates}})`);
    if(index < (state.rows||[]).length-1){{index++; show();}}
  }}catch(error){{
    setStatus(`disk autosave failed: ${{error.message}}`, true);
  }}
}}
function nextCard(){{index=Math.min((state.rows||[]).length-1,index+1);show();}}
function prevCard(){{index=Math.max(0,index-1);show();}}
function exportDecisions(){{
  const blob=new Blob([JSON.stringify({{artifact:'step2m3r_topology_decision_export_backup',created_at:new Date().toISOString(),rows:Object.values(decisions)}},null,2)],{{type:'application/json'}});
  const url=URL.createObjectURL(blob);
  const a=document.createElement('a'); a.href=url; a.download='step2m3r_topology_decisions_export.json'; a.click(); URL.revokeObjectURL(url);
}}
document.addEventListener('keydown',ev=>{{if(ev.key==='a'||ev.key==='A')decide('accept_m3_handoff_visual_continuity'); if(ev.key==='x'||ev.key==='X')decide('reject_or_quarantine_m3_handoff_visual_continuity'); if(ev.key==='u'||ev.key==='U')decide('unsure_needs_later_review'); if(ev.key===' '){{ev.preventDefault(); togglePlayback();}} if(ev.key==='ArrowRight')nextCard(); if(ev.key==='ArrowLeft')prevCard();}});
show();
</script>
</body>
</html>"""


def read_m3r_reviewed_decisions() -> dict[str, Any]:
    if not STEP2M3R_REVIEWED_TOPOLOGY_DECISIONS_PATH.exists():
        return guardrail_stamp(
            {
                "artifact": "step2m3r_reviewed_topology_decisions",
                "created_at": utc_iso(),
                "source_match_id": MATCH_ID,
                "source_clip_id": CLIP_ID,
                "current_review_version": M3R_CURRENT_REVIEW_VERSION,
                "current_visual_evidence_version": M3R_CURRENT_VISUAL_EVIDENCE_VERSION,
                "reviewed_decision_rows": 0,
                "rows": [],
                **m3r_guardrail_fields(),
            }
        )
    payload = read_json(STEP2M3R_REVIEWED_TOPOLOGY_DECISIONS_PATH)
    decision_rows = decision_rows_from_payload(payload)
    normalized = dict(payload) if isinstance(payload, dict) else {}
    normalized.pop("decisions", None)
    normalized.update(
        {
            "artifact": "step2m3r_reviewed_topology_decisions",
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "current_review_version": normalized.get("current_review_version", M3R_CURRENT_REVIEW_VERSION),
            "current_visual_evidence_version": normalized.get("current_visual_evidence_version", M3R_CURRENT_VISUAL_EVIDENCE_VERSION),
            "reviewed_decision_rows": len(decision_rows),
            "rows": decision_rows,
            **m3r_guardrail_fields(),
        }
    )
    return guardrail_stamp(normalized)


def m3r_candidate_by_review_id(review_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("step2m3r_topology_review_candidate_id", "")): row
        for row in rows_from_payload(review_payload)
        if row.get("step2m3r_topology_review_candidate_id")
    }


def normalize_decision(decision: str) -> str:
    if decision == "unsure":
        return M3R_UNSURE_DECISION
    return str(decision)


def m3r_review_decision_row(candidate: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    decision = normalize_decision(payload.get("human_review_decision", ""))
    if decision not in {M3R_ACCEPT_DECISION, M3R_REJECT_DECISION, M3R_UNSURE_DECISION}:
        raise ValueError(f"Step2.M3R topology review decision is not allowed: {decision}")
    row = {
        "step2m3r_topology_review_candidate_id": str(candidate.get("step2m3r_topology_review_candidate_id", "")),
        "review_subject_type": str(candidate.get("review_subject_type", "")),
        "step2m3r_review_category": str(candidate.get("step2m3r_review_category", "")),
        "visual_continuity_group_id": str(candidate.get("visual_continuity_group_id", "")),
        "continuity_edge_id": str(candidate.get("continuity_edge_id", "")),
        "source_visible_person_base_id": str(candidate.get("source_visible_person_base_id", "")),
        "target_visible_person_base_id": str(candidate.get("target_visible_person_base_id", "")),
        "source_frame_sequence": safe_int(candidate.get("source_frame_sequence", candidate.get("min_frame_sequence")), -1),
        "target_frame_sequence": safe_int(candidate.get("target_frame_sequence", candidate.get("max_frame_sequence")), -1),
        "source_review_bucket": str(candidate.get("source_review_bucket", "")),
        "human_review_decision": decision,
        "reviewer_name": str(payload.get("reviewer_name", "")),
        "notes": str(payload.get("notes", "")),
        "reviewed_at": utc_iso(),
        "human_confirmed": True,
        "current_review_version": M3R_CURRENT_REVIEW_VERSION,
        "review_decisions_collected_with_review_version": M3R_CURRENT_REVIEW_VERSION,
        "current_visual_evidence_version": M3R_CURRENT_VISUAL_EVIDENCE_VERSION,
        "review_decisions_collected_with_visual_evidence_version": M3R_CURRENT_VISUAL_EVIDENCE_VERSION,
        "review_decisions_visual_evidence_version_matches_current": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
        "approve_any_identity_tracking": False,
        "approve_any_player_slot_use": False,
        "approve_any_goalkeeper_slot_use": False,
        "approve_any_metric_use": False,
        "approve_event_or_tactical_analysis": False,
        "approve_exact_22_or_exact_two_goalkeeper_forcing": False,
        "approve_official_referee_exclusion": False,
        "approve_bad_detection_deletion": False,
        "approve_production_promotion": False,
        **m3r_guardrail_fields(),
    }
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def write_m3r_reviewed_decisions_payload(decision_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(decision_rows, key=lambda row: str(row.get("step2m3r_topology_review_candidate_id", "")))
    payload = guardrail_stamp(
        {
            "artifact": "step2m3r_reviewed_topology_decisions",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "current_review_version": M3R_CURRENT_REVIEW_VERSION,
            "current_visual_evidence_version": M3R_CURRENT_VISUAL_EVIDENCE_VERSION,
            "reviewed_decision_rows": len(ordered),
            "rows": ordered,
            **m3r_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(payload)
    write_json_atomic(STEP2M3R_REVIEWED_TOPOLOGY_DECISIONS_PATH, payload)
    return payload


def validate_m3r_reviewed_rows(review_payload: dict[str, Any], reviewed_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = m3r_candidate_by_review_id(review_payload)
    usable_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in decision_rows_from_payload(reviewed_payload):
        candidate_id = str(row.get("step2m3r_topology_review_candidate_id", ""))
        if not candidate_id or candidate_id not in candidates:
            errors.append({"issue_code": "unknown_step2m3r_topology_review_candidate", "candidate_id": candidate_id})
            continue
        if candidate_id in seen:
            errors.append({"issue_code": "duplicate_step2m3r_topology_review_candidate", "candidate_id": candidate_id})
            continue
        decision = str(row.get("human_review_decision", ""))
        if decision not in {M3R_ACCEPT_DECISION, M3R_REJECT_DECISION, M3R_UNSURE_DECISION}:
            errors.append({"issue_code": "invalid_step2m3r_topology_review_decision", "candidate_id": candidate_id, "decision": decision})
            continue
        seen.add(candidate_id)
        usable_rows.append(row)
    return usable_rows, errors


def m3r_review_progress_payload(review_payload: dict[str, Any], reviewed_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    review_rows = rows_from_payload(review_payload)
    reviewed_payload = reviewed_payload or {"rows": []}
    usable_rows, validation_errors = validate_m3r_reviewed_rows(review_payload, reviewed_payload)
    decision_counts = Counter(str(row.get("human_review_decision", "")) for row in usable_rows)
    category_counts = Counter(str(row.get("step2m3r_review_category", "")) for row in review_rows)
    candidates = m3r_candidate_by_review_id(review_payload)
    reviewed_by_category = Counter(
        str(candidates.get(str(row.get("step2m3r_topology_review_candidate_id", "")), {}).get("step2m3r_review_category", ""))
        for row in usable_rows
    )
    category_progress = {
        category: {"total": category_counts.get(category, 0), "reviewed": reviewed_by_category.get(category, 0)}
        for category in sorted(category_counts)
    }
    collected_versions = {
        str(row.get("review_decisions_collected_with_review_version", ""))
        for row in usable_rows
        if row.get("review_decisions_collected_with_review_version")
    }
    collected_evidence_versions = {
        str(row.get("review_decisions_collected_with_visual_evidence_version", ""))
        for row in usable_rows
        if row.get("review_decisions_collected_with_visual_evidence_version")
    }
    version_matches = bool(usable_rows) and collected_versions == {M3R_CURRENT_REVIEW_VERSION}
    evidence_version_matches = not usable_rows or collected_evidence_versions == {M3R_CURRENT_VISUAL_EVIDENCE_VERSION}
    topology_review_completed = len(usable_rows) == len(review_rows) and not validation_errors
    forbidden = sorted(set(forbidden_keys_present(review_payload)) | set(forbidden_keys_present(reviewed_payload)))
    evidence_summary = visual_evidence_summary(review_payload)
    return guardrail_stamp(
        {
            "artifact": "step2m3r_review_progress_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "total_review_candidates": len(review_rows),
            "reviewed_candidates": len(usable_rows),
            "accepted_count": decision_counts.get(M3R_ACCEPT_DECISION, 0),
            "rejected_count": decision_counts.get(M3R_REJECT_DECISION, 0),
            "unsure_count": decision_counts.get(M3R_UNSURE_DECISION, 0),
            "topology_review_completed": topology_review_completed,
            "review_category_counts": dict(sorted(category_counts.items())),
            "review_category_progress": category_progress,
            "review_queue_hard_max": M3R_HARD_MAX_REVIEW_CARDS,
            "current_review_version": M3R_CURRENT_REVIEW_VERSION,
            "review_decisions_collected_with_review_version": sorted(collected_versions),
            "review_decisions_version_matches_current": version_matches,
            "current_visual_evidence_version": M3R_CURRENT_VISUAL_EVIDENCE_VERSION,
            "review_decisions_collected_with_visual_evidence_version": sorted(collected_evidence_versions),
            "review_decisions_visual_evidence_version_matches_current": evidence_version_matches,
            **evidence_summary,
            "validation_errors": validation_errors,
            "forbidden_keys_present": forbidden,
            **m3r_guardrail_fields(),
        }
    )


def m3r_review_decision_summary_payload(review_payload: dict[str, Any], reviewed_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    progress = m3r_review_progress_payload(review_payload, reviewed_payload)
    usable_rows, _errors = validate_m3r_reviewed_rows(review_payload, reviewed_payload or {"rows": []})
    return guardrail_stamp(
        {
            "artifact": "step2m3r_review_decision_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "total_review_candidates": progress["total_review_candidates"],
            "reviewed_candidates": progress["reviewed_candidates"],
            "accepted_count": progress["accepted_count"],
            "rejected_count": progress["rejected_count"],
            "unsure_count": progress["unsure_count"],
            "human_review_decision_counts": dict(sorted(Counter(str(row.get("human_review_decision", "")) for row in usable_rows).items())),
            "review_category_progress": progress["review_category_progress"],
            "topology_review_completed": progress["topology_review_completed"],
            "current_review_version": M3R_CURRENT_REVIEW_VERSION,
            "review_decisions_collected_with_review_version": progress["review_decisions_collected_with_review_version"],
            "review_decisions_version_matches_current": progress["review_decisions_version_matches_current"],
            "current_visual_evidence_version": M3R_CURRENT_VISUAL_EVIDENCE_VERSION,
            "review_decisions_collected_with_visual_evidence_version": progress["review_decisions_collected_with_visual_evidence_version"],
            "review_decisions_visual_evidence_version_matches_current": progress["review_decisions_visual_evidence_version_matches_current"],
            "visual_evidence_available_count": progress["visual_evidence_available_count"],
            "visual_evidence_missing_count": progress["visual_evidence_missing_count"],
            "visual_evidence_missing_rate": progress["visual_evidence_missing_rate"],
            "visual_evidence_safe_for_review": progress["visual_evidence_safe_for_review"],
            "animation_evidence_available_count": progress["animation_evidence_available_count"],
            "animation_evidence_missing_count": progress["animation_evidence_missing_count"],
            "animation_evidence_missing_rate": progress["animation_evidence_missing_rate"],
            "animation_evidence_safe_for_review": progress["animation_evidence_safe_for_review"],
            "static_strip_fallback_available_count": progress["static_strip_fallback_available_count"],
            "forbidden_keys_present": progress["forbidden_keys_present"],
            **m3r_guardrail_fields(),
        }
    )


def refresh_m3r_review_summaries(
    review_payload: dict[str, Any],
    reviewed_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    progress = m3r_review_progress_payload(review_payload, reviewed_payload)
    decision_summary = m3r_review_decision_summary_payload(review_payload, reviewed_payload)
    write_json_atomic(STEP2M3R_REVIEW_PROGRESS_SUMMARY_PATH, progress)
    write_json_atomic(STEP2M3R_REVIEW_DECISION_SUMMARY_PATH, decision_summary)
    return progress, decision_summary


def save_m3r_topology_review_decision(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    review_payload = read_json(STEP2M3R_TOPOLOGY_REVIEW_CANDIDATE_ROWS_PATH)
    candidates = m3r_candidate_by_review_id(review_payload)
    candidate_id = str(payload.get("step2m3r_topology_review_candidate_id", ""))
    candidate = candidates.get(candidate_id)
    if not candidate:
        raise ValueError("unknown_step2m3r_topology_review_candidate")
    decision = m3r_review_decision_row(candidate, payload)
    reviewed_payload = read_m3r_reviewed_decisions()
    by_id = {
        str(row.get("step2m3r_topology_review_candidate_id", "")): row
        for row in decision_rows_from_payload(reviewed_payload)
        if row.get("step2m3r_topology_review_candidate_id")
    }
    by_id[str(decision["step2m3r_topology_review_candidate_id"])] = decision
    updated_payload = write_m3r_reviewed_decisions_payload(list(by_id.values()))
    progress, _decision_summary = refresh_m3r_review_summaries(review_payload, updated_payload)
    return decision, updated_payload, progress


def build_handoff_readiness_summary(
    group_summary: dict[str, Any],
    edge_summary: dict[str, Any],
    review_payload: dict[str, Any],
    progress: dict[str, Any],
) -> dict[str, Any]:
    unsafe_reasons: list[str] = []
    if safe_int(group_summary.get("high_topology_risk_group_count"), 0) > 0 and progress.get("topology_review_completed") is not True:
        unsafe_reasons.append("topology_review_not_completed_for_high_risk_groups")
    if safe_int(group_summary.get("groups_over_cap_count"), 0) != 0:
        unsafe_reasons.append("m3r_group_span_cap_violation")
    if len(rows_from_payload(review_payload)) > M3R_HARD_MAX_REVIEW_CARDS:
        unsafe_reasons.append("m3r_review_queue_over_hard_max")
    evidence_summary = visual_evidence_summary(review_payload)
    if evidence_summary.get("visual_evidence_safe_for_review") is not True:
        unsafe_reasons.append("visual_evidence_not_safe_for_review")
    if safe_float(evidence_summary.get("visual_evidence_missing_rate"), 1.0) > 0:
        unsafe_reasons.append("visual_evidence_missing_for_review_cards")
    if evidence_summary.get("animation_evidence_safe_for_review") is not True:
        unsafe_reasons.append("animation_evidence_not_safe_for_review")
    if safe_int(evidence_summary.get("animation_evidence_missing_count"), 1) > 0:
        unsafe_reasons.append("animation_evidence_missing_for_review_cards")
    if safe_int(progress.get("reviewed_candidates"), 0) > 0 and progress.get("review_decisions_visual_evidence_version_matches_current") is not True:
        unsafe_reasons.append("review_decisions_visual_evidence_version_mismatch")
    if group_summary.get("forbidden_keys_present", []) or edge_summary.get("forbidden_keys_present", []):
        unsafe_reasons.append("m3r_forbidden_keys_present")
    summary = guardrail_stamp(
        {
            "artifact": "step2m3r_handoff_readiness_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "safe_for_visual_continuity_handoff_candidate": not unsafe_reasons,
            "safe_for_visual_continuity_handoff_reasons": [
                "m3r_topology_audit_rows_generated",
                "m3r_review_queue_within_hard_max",
                "m3r_visual_only_guardrails_present",
            ],
            "unsafe_for_visual_continuity_handoff_reasons": unsafe_reasons,
            "total_groups_audited": group_summary.get("total_groups_audited", 0),
            "accepted_edges_audited": edge_summary.get("accepted_edges_audited", 0),
            "high_topology_risk_group_count": group_summary.get("high_topology_risk_group_count", 0),
            "branch_merge_group_count": group_summary.get("branch_merge_group_count", 0),
            "duplicate_frame_member_group_count": group_summary.get("duplicate_frame_member_group_count", 0),
            "role_context_mixed_group_count": group_summary.get("role_context_mixed_group_count", 0),
            "review_queue_size": len(rows_from_payload(review_payload)),
            "topology_review_completed": progress.get("topology_review_completed", False),
            "reviewed_candidates": progress.get("reviewed_candidates", 0),
            "accepted_count": progress.get("accepted_count", 0),
            "rejected_count": progress.get("rejected_count", 0),
            "unsure_count": progress.get("unsure_count", 0),
            **evidence_summary,
            **m3r_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(summary)
    return summary


def build_step2m3r_validation_outputs(
    *,
    group_summary: dict[str, Any],
    edge_summary: dict[str, Any],
    review_payload: dict[str, Any],
    handoff: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    m3_freeze = read_json(STEP2M3_FREEZE_CANDIDATE_MANIFEST_PATH)
    m3_validation = read_json(STEP2M3_VALIDATION_SUMMARY_PATH)
    m3_group_summary = read_json(STEP2M3_GROUP_SUMMARY_PATH)
    reviewed_payload = read_m3r_reviewed_decisions()
    progress, decision_summary = refresh_m3r_review_summaries(review_payload, reviewed_payload)
    review_queue_size = len(rows_from_payload(review_payload))
    forbidden = sorted(
        set(forbidden_keys_present(group_summary))
        | set(forbidden_keys_present(edge_summary))
        | set(forbidden_keys_present(review_payload))
        | set(forbidden_keys_present(handoff))
        | set(forbidden_keys_present(reviewed_payload))
    )
    issues: list[dict[str, Any]] = []

    def require(condition: bool, code: str, message: str) -> None:
        if not condition:
            issues.append({"severity": "blocking", "issue_code": code, "message": message})

    require(m3_freeze.get("step2m3_freeze_candidate_created") is True, "m3_freeze_candidate_missing", "Step2.M3 freeze candidate must exist before M3R.")
    require(m3_freeze.get("forbidden_keys_present") == [], "m3_forbidden_keys_present", "Step2.M3 must have no forbidden keys.")
    require(m3_validation.get("forbidden_keys_present") == [], "m3_validation_forbidden_keys_present", "Step2.M3 validation must have no forbidden keys.")
    require(safe_int(m3_group_summary.get("groups_over_cap_count"), 1) == 0, "m3_groups_over_cap", "Step2.M3 groups must be cap-safe.")
    require(review_queue_size <= M3R_HARD_MAX_REVIEW_CARDS, "m3r_review_queue_over_hard_max", "Step2.M3R review queue must be at or below hard max.")
    require(safe_int(group_summary.get("total_groups_audited"), 0) > 0, "m3r_group_topology_rows_missing", "Step2.M3R group topology audit rows must be generated.")
    require(safe_int(edge_summary.get("accepted_edges_audited"), 0) > 0, "m3r_edge_topology_rows_missing", "Step2.M3R accepted-edge topology rows must be generated.")
    require(forbidden == [], "m3r_forbidden_keys_present", "Step2.M3R artifacts must have no forbidden keys.")
    require(progress.get("visual_evidence_safe_for_review") is True, "m3r_visual_evidence_not_safe_for_review", "Step2.M3R review cards must have visual evidence.")
    require(safe_float(progress.get("visual_evidence_missing_rate"), 1.0) == 0.0, "m3r_visual_evidence_missing", "Step2.M3R visual evidence missing rate must be zero.")
    require(
        progress.get("animation_evidence_safe_for_review") is True,
        "m3r_animation_evidence_not_safe_for_review",
        "Step2.M3R review cards must have animated visual evidence.",
    )
    require(
        safe_int(progress.get("animation_evidence_missing_count"), 1) == 0,
        "m3r_animation_evidence_missing",
        "Step2.M3R animation evidence missing count must be zero.",
    )
    if safe_int(progress.get("reviewed_candidates"), 0) > 0:
        require(
            progress.get("review_decisions_visual_evidence_version_matches_current") is True,
            "m3r_review_decisions_visual_evidence_version_mismatch",
            "Step2.M3R review decisions must be collected with the current visual evidence version.",
        )
    require(PRODUCTION_READY is False, "production_ready_not_false", "Step2.M3R production_ready must remain false.")
    require(NO_AUTO_PROMOTION is True, "no_auto_promotion_not_true", "Step2.M3R no_auto_promotion must remain true.")
    if progress.get("validation_errors"):
        issues.append(
            {
                "severity": "blocking",
                "issue_code": "m3r_persisted_review_decision_validation_errors",
                "message": "Step2.M3R persisted topology review decisions failed schema or candidate validation.",
                "validation_errors": progress.get("validation_errors", []),
            }
        )
    gate_checks = {
        "m3_freeze_candidate_created": m3_freeze.get("step2m3_freeze_candidate_created") is True,
        "m3_forbidden_keys_absent": m3_freeze.get("forbidden_keys_present") == [] and m3_validation.get("forbidden_keys_present") == [],
        "m3_groups_over_cap_zero": safe_int(m3_group_summary.get("groups_over_cap_count"), 1) == 0,
        "m3r_review_queue_within_hard_max": review_queue_size <= M3R_HARD_MAX_REVIEW_CARDS,
        "m3r_topology_audit_rows_generated": safe_int(group_summary.get("total_groups_audited"), 0) > 0 and safe_int(edge_summary.get("accepted_edges_audited"), 0) > 0,
        "m3r_visual_evidence_safe_for_review": progress.get("visual_evidence_safe_for_review") is True,
        "m3r_visual_evidence_missing_rate_zero": safe_float(progress.get("visual_evidence_missing_rate"), 1.0) == 0.0,
        "m3r_animation_evidence_safe_for_review": progress.get("animation_evidence_safe_for_review") is True,
        "m3r_animation_evidence_missing_count_zero": safe_int(progress.get("animation_evidence_missing_count"), 1) == 0,
        "m3r_review_decisions_visual_evidence_version_current": safe_int(progress.get("reviewed_candidates"), 0) == 0
        or progress.get("review_decisions_visual_evidence_version_matches_current") is True,
        "m3r_forbidden_keys_absent": forbidden == [],
        "production_ready_false": PRODUCTION_READY is False,
        "no_auto_promotion_true": NO_AUTO_PROMOTION is True,
    }
    freeze_candidate = all(gate_checks.values()) and not issues
    validation = guardrail_stamp(
        {
            "artifact": "step2m3r_validation_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "total_groups_audited": group_summary.get("total_groups_audited", 0),
            "accepted_edges_audited": edge_summary.get("accepted_edges_audited", 0),
            "high_topology_risk_group_count": group_summary.get("high_topology_risk_group_count", 0),
            "branch_merge_group_count": group_summary.get("branch_merge_group_count", 0),
            "duplicate_frame_member_group_count": group_summary.get("duplicate_frame_member_group_count", 0),
            "role_context_mixed_group_count": group_summary.get("role_context_mixed_group_count", 0),
            "review_queue_size": review_queue_size,
            "reviewed_candidates": progress.get("reviewed_candidates", 0),
            "topology_review_completed": progress.get("topology_review_completed", False),
            "safe_for_visual_continuity_handoff_candidate": handoff.get("safe_for_visual_continuity_handoff_candidate", False),
            "visual_evidence_available_count": progress.get("visual_evidence_available_count", 0),
            "visual_evidence_missing_count": progress.get("visual_evidence_missing_count", 0),
            "visual_evidence_missing_rate": progress.get("visual_evidence_missing_rate", 1.0),
            "visual_evidence_safe_for_review": progress.get("visual_evidence_safe_for_review", False),
            "animation_evidence_available_count": progress.get("animation_evidence_available_count", 0),
            "animation_evidence_missing_count": progress.get("animation_evidence_missing_count", 0),
            "animation_evidence_missing_rate": progress.get("animation_evidence_missing_rate", 1.0),
            "animation_evidence_safe_for_review": progress.get("animation_evidence_safe_for_review", False),
            "static_strip_fallback_available_count": progress.get("static_strip_fallback_available_count", 0),
            "current_visual_evidence_version": M3R_CURRENT_VISUAL_EVIDENCE_VERSION,
            "review_decisions_visual_evidence_version_matches_current": progress.get("review_decisions_visual_evidence_version_matches_current", False),
            "step2m3r_freeze_candidate_created": freeze_candidate,
            "forbidden_keys_present": forbidden,
            "gate_checks": gate_checks,
            **m3r_guardrail_fields(),
        }
    )
    audit = guardrail_stamp(
        {
            "artifact": "step2m3r_safety_guardrail_audit",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "forbidden_keys_present": forbidden,
            "step2_visual_continuity_root": str(STEP2_VISUAL_CONTINUITY_DIR.resolve()),
            "m1_read_root": str(STEP2M1_OUTPUT_DIR.resolve()),
            "m2_read_root": str(STEP2M2_OUTPUT_DIR.resolve()),
            "m3_read_root": str(STEP2M3_OUTPUT_DIR.resolve()),
            "m3r_write_root": str(STEP2M3R_OUTPUT_DIR.resolve()),
            "no_m3r_writes_to_m1_m2_m3": True,
            "visual_evidence_available_count": progress.get("visual_evidence_available_count", 0),
            "visual_evidence_missing_count": progress.get("visual_evidence_missing_count", 0),
            "visual_evidence_safe_for_review": progress.get("visual_evidence_safe_for_review", False),
            "animation_evidence_available_count": progress.get("animation_evidence_available_count", 0),
            "animation_evidence_missing_count": progress.get("animation_evidence_missing_count", 0),
            "animation_evidence_safe_for_review": progress.get("animation_evidence_safe_for_review", False),
            "static_strip_fallback_available_count": progress.get("static_strip_fallback_available_count", 0),
            "current_visual_evidence_version": M3R_CURRENT_VISUAL_EVIDENCE_VERSION,
            "review_decisions_visual_evidence_version_matches_current": progress.get("review_decisions_visual_evidence_version_matches_current", False),
            **m3r_guardrail_fields(),
        }
    )
    issue_register = guardrail_stamp(
        {
            "artifact": "step2m3r_issue_register",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "blocking_issue_count": sum(1 for issue in issues if issue.get("severity") == "blocking"),
            "rows": issues,
            **m3r_guardrail_fields(),
        }
    )
    freeze_manifest = guardrail_stamp(
        {
            "artifact": "step2m3r_freeze_candidate_manifest",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "step2m3r_freeze_candidate_created": freeze_candidate,
            "human_approved": False,
            "review_required_before_any_future_promotion": True,
            "safe_to_apply_globally": False,
            "match_local_only": True,
            "validation_summary_path": str(STEP2M3R_VALIDATION_SUMMARY_PATH.resolve()),
            "handoff_readiness_summary_path": str(STEP2M3R_HANDOFF_READINESS_SUMMARY_PATH.resolve()),
            "review_queue_size": review_queue_size,
            "reviewed_candidates": progress.get("reviewed_candidates", 0),
            "topology_review_completed": progress.get("topology_review_completed", False),
            "visual_evidence_available_count": progress.get("visual_evidence_available_count", 0),
            "visual_evidence_missing_count": progress.get("visual_evidence_missing_count", 0),
            "visual_evidence_safe_for_review": progress.get("visual_evidence_safe_for_review", False),
            "animation_evidence_available_count": progress.get("animation_evidence_available_count", 0),
            "animation_evidence_missing_count": progress.get("animation_evidence_missing_count", 0),
            "animation_evidence_safe_for_review": progress.get("animation_evidence_safe_for_review", False),
            "static_strip_fallback_available_count": progress.get("static_strip_fallback_available_count", 0),
            "current_visual_evidence_version": M3R_CURRENT_VISUAL_EVIDENCE_VERSION,
            "review_decisions_visual_evidence_version_matches_current": progress.get("review_decisions_visual_evidence_version_matches_current", False),
            "forbidden_keys_present": forbidden,
            "gate_checks": gate_checks,
            **m3r_guardrail_fields(),
        }
    )
    for payload in [validation, audit, issue_register, freeze_manifest, decision_summary]:
        assert_no_forbidden_keys(payload)
    return validation, audit, issue_register, freeze_manifest


def build_step2m3r_topology_qa() -> dict[str, Any]:
    assert_m3r_output_path_isolation()
    ensure_dir(STEP2M3R_OUTPUT_DIR)
    group_payload = read_json(STEP2M3_GROUP_ROWS_PATH)
    accepted_rows = list(iter_jsonl_gz_rows(STEP2M3_ACCEPTED_EDGES_JSONL_GZ_PATH))
    group_audit_payload, group_summary, group_sample, edges_by_group = build_group_topology_audit(group_payload, accepted_rows)
    group_audit_rows = rows_from_payload(group_audit_payload)
    edge_rows, edge_summary, edge_sample = build_edge_topology_audit(group_audit_rows, edges_by_group)
    review_payload = build_topology_review_queue(group_audit_rows, edge_rows)
    source_groups_by_id = {
        str(row.get("visual_continuity_group_id", "")): row
        for row in rows_from_payload(group_payload)
        if row.get("visual_continuity_group_id")
    }
    review_payload = add_visual_evidence_to_review_payload(
        review_payload,
        source_groups_by_id=source_groups_by_id,
        edges_by_group=edges_by_group,
        accepted_edges_by_id=accepted_edge_by_id(accepted_rows),
        node_payload=read_json(STEP2M1_NODE_ROWS_PATH),
    )
    render_result = render_topology_contact_sheet(rows_from_payload(review_payload))
    write_text(STEP2M3R_TOPOLOGY_REVIEW_UI_HTML_PATH, topology_review_ui_html(review_payload))
    reviewed_payload = read_m3r_reviewed_decisions()
    progress = m3r_review_progress_payload(review_payload, reviewed_payload)
    decision_summary = m3r_review_decision_summary_payload(review_payload, reviewed_payload)
    handoff = build_handoff_readiness_summary(group_summary, edge_summary, review_payload, progress)
    validation, audit, issue_register, freeze_manifest = build_step2m3r_validation_outputs(
        group_summary=group_summary,
        edge_summary=edge_summary,
        review_payload=review_payload,
        handoff=handoff,
    )
    for payload, path in [
        (group_audit_payload, STEP2M3R_GROUP_TOPOLOGY_AUDIT_ROWS_PATH),
        (group_summary, STEP2M3R_GROUP_TOPOLOGY_AUDIT_SUMMARY_PATH),
        (group_sample, STEP2M3R_GROUP_TOPOLOGY_AUDIT_SAMPLE_PATH),
        (edge_summary, STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_SUMMARY_PATH),
        (edge_sample, STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_SAMPLE_PATH),
        (review_payload, STEP2M3R_TOPOLOGY_REVIEW_CANDIDATE_ROWS_PATH),
        (progress, STEP2M3R_REVIEW_PROGRESS_SUMMARY_PATH),
        (decision_summary, STEP2M3R_REVIEW_DECISION_SUMMARY_PATH),
        (handoff, STEP2M3R_HANDOFF_READINESS_SUMMARY_PATH),
        (validation, STEP2M3R_VALIDATION_SUMMARY_PATH),
        (audit, STEP2M3R_SAFETY_GUARDRAIL_AUDIT_PATH),
        (issue_register, STEP2M3R_ISSUE_REGISTER_PATH),
        (freeze_manifest, STEP2M3R_FREEZE_CANDIDATE_MANIFEST_PATH),
    ]:
        assert_no_forbidden_keys(payload)
        write_json(path, payload)
    write_jsonl_gz_rows(STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_ROWS_JSONL_GZ_PATH, edge_rows)
    return {
        "group_topology_audit": group_audit_payload,
        "group_topology_summary": group_summary,
        "group_topology_sample": group_sample,
        "accepted_edge_topology_rows": edge_rows,
        "accepted_edge_topology_summary": edge_summary,
        "accepted_edge_topology_sample": edge_sample,
        "topology_review_candidates": review_payload,
        "contact_sheet": render_result,
        "review_progress": progress,
        "review_decision": decision_summary,
        "handoff_readiness_summary": handoff,
        "validation_summary": validation,
        "safety_guardrail_audit": audit,
        "issue_register": issue_register,
        "freeze_candidate_manifest": freeze_manifest,
    }


def validate_step2m3r_topology_qa() -> dict[str, Any]:
    assert_m3r_output_path_isolation()
    group_summary = read_json(STEP2M3R_GROUP_TOPOLOGY_AUDIT_SUMMARY_PATH)
    edge_summary = read_json(STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_SUMMARY_PATH)
    review_payload = read_json(STEP2M3R_TOPOLOGY_REVIEW_CANDIDATE_ROWS_PATH)
    reviewed_payload = read_m3r_reviewed_decisions()
    progress, decision_summary = refresh_m3r_review_summaries(review_payload, reviewed_payload)
    handoff = build_handoff_readiness_summary(group_summary, edge_summary, review_payload, progress)
    validation, audit, issue_register, freeze_manifest = build_step2m3r_validation_outputs(
        group_summary=group_summary,
        edge_summary=edge_summary,
        review_payload=review_payload,
        handoff=handoff,
    )
    for payload, path in [
        (handoff, STEP2M3R_HANDOFF_READINESS_SUMMARY_PATH),
        (validation, STEP2M3R_VALIDATION_SUMMARY_PATH),
        (audit, STEP2M3R_SAFETY_GUARDRAIL_AUDIT_PATH),
        (issue_register, STEP2M3R_ISSUE_REGISTER_PATH),
        (freeze_manifest, STEP2M3R_FREEZE_CANDIDATE_MANIFEST_PATH),
        (decision_summary, STEP2M3R_REVIEW_DECISION_SUMMARY_PATH),
        (progress, STEP2M3R_REVIEW_PROGRESS_SUMMARY_PATH),
    ]:
        assert_no_forbidden_keys(payload)
        write_json(path, payload)
    return {
        "review_progress": progress,
        "review_decision": decision_summary,
        "handoff_readiness_summary": handoff,
        "validation_summary": validation,
        "safety_guardrail_audit": audit,
        "issue_register": issue_register,
        "freeze_candidate_manifest": freeze_manifest,
    }


def write_step2m3r_review_pack() -> dict[str, Any]:
    ensure_dir(STEP2M3R_REVIEW_PACK_DIR)
    files = [
        STEP2M3R_GROUP_TOPOLOGY_AUDIT_SUMMARY_PATH,
        STEP2M3R_GROUP_TOPOLOGY_AUDIT_SAMPLE_PATH,
        STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_SUMMARY_PATH,
        STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_SAMPLE_PATH,
        STEP2M3R_TOPOLOGY_REVIEW_CANDIDATE_ROWS_PATH,
        STEP2M3R_TOPOLOGY_REVIEW_UI_HTML_PATH,
        STEP2M3R_TOPOLOGY_REVIEW_CONTACT_SHEET_PATH,
        STEP2M3R_REVIEWED_TOPOLOGY_DECISIONS_PATH,
        STEP2M3R_REVIEW_PROGRESS_SUMMARY_PATH,
        STEP2M3R_REVIEW_DECISION_SUMMARY_PATH,
        STEP2M3R_HANDOFF_READINESS_SUMMARY_PATH,
        STEP2M3R_VALIDATION_SUMMARY_PATH,
        STEP2M3R_SAFETY_GUARDRAIL_AUDIT_PATH,
        STEP2M3R_ISSUE_REGISTER_PATH,
        STEP2M3R_FREEZE_CANDIDATE_MANIFEST_PATH,
    ]
    copied: list[str] = []
    for path in files:
        if not path.exists():
            continue
        destination = STEP2M3R_REVIEW_PACK_DIR / path.name
        shutil.copyfile(path, destination)
        copied.append(str(destination.resolve()))
    copied_evidence: list[str] = []
    for source_root in [
        STEP2M3R_GROUP_TIMELINE_STRIPS_DIR,
        STEP2M3R_EDGE_BURST_STRIPS_DIR,
        STEP2M3R_GROUP_TIMELINE_ANIMATIONS_DIR,
        STEP2M3R_EDGE_BURST_ANIMATIONS_DIR,
    ]:
        if not source_root.exists():
            continue
        relative_root = source_root.relative_to(STEP2M3R_OUTPUT_DIR)
        for extension in ("*.jpg", "*.gif", "*.mp4"):
            for source_path in source_root.glob(extension):
                destination = STEP2M3R_REVIEW_PACK_DIR / relative_root / source_path.name
                ensure_dir(destination.parent)
                shutil.copyfile(source_path, destination)
                copied_evidence.append(str(destination.resolve()))
    manifest = guardrail_stamp(
        {
            "artifact": "step2m3r_review_pack_manifest",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "review_pack_dir": str(STEP2M3R_REVIEW_PACK_DIR.resolve()),
            "copied_files": copied,
            "copied_visual_evidence_files": copied_evidence,
            "group_topology_audit_rows_path": str(STEP2M3R_GROUP_TOPOLOGY_AUDIT_ROWS_PATH.resolve()),
            "accepted_edge_topology_audit_rows_jsonl_gz_path": str(STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_ROWS_JSONL_GZ_PATH.resolve()),
            "autosave_endpoint": "/api/step2m3r/topology-review-decision",
            **m3r_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(manifest)
    write_json(STEP2M3R_REVIEW_PACK_MANIFEST_PATH, manifest)
    return manifest


def prepare_step2m3r_review_ui(host: str = "127.0.0.1", port: int = 8786) -> dict[str, Any]:
    assert_m3r_output_path_isolation()
    review_payload = read_json(STEP2M3R_TOPOLOGY_REVIEW_CANDIDATE_ROWS_PATH)
    write_text(STEP2M3R_TOPOLOGY_REVIEW_UI_HTML_PATH, topology_review_ui_html(review_payload))
    reviewed_payload = read_m3r_reviewed_decisions()
    progress, decision_summary = refresh_m3r_review_summaries(review_payload, reviewed_payload)
    manifest = guardrail_stamp(
        {
            "artifact": "step2m3r_review_ui_manifest",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "url": f"http://{host}:{port}/",
            "topology_review_ui_html_path": str(STEP2M3R_TOPOLOGY_REVIEW_UI_HTML_PATH.resolve()),
            "topology_review_candidate_rows_path": str(STEP2M3R_TOPOLOGY_REVIEW_CANDIDATE_ROWS_PATH.resolve()),
            "reviewed_topology_decisions_path": str(STEP2M3R_REVIEWED_TOPOLOGY_DECISIONS_PATH.resolve()),
            "review_progress_summary_path": str(STEP2M3R_REVIEW_PROGRESS_SUMMARY_PATH.resolve()),
            "review_decision_summary_path": str(STEP2M3R_REVIEW_DECISION_SUMMARY_PATH.resolve()),
            "total_review_candidates": progress.get("total_review_candidates", len(rows_from_payload(review_payload))),
            "reviewed_candidates": progress.get("reviewed_candidates", 0),
            "topology_review_completed": decision_summary.get("topology_review_completed", False),
            "autosave_endpoint": "/api/step2m3r/topology-review-decision",
            **m3r_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(manifest)
    return manifest


class Step2M3RTopologyReviewHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        raw_path = unquote(urlparse(self.path).path.lstrip("/"))
        file_path = STEP2M3R_TOPOLOGY_REVIEW_UI_HTML_PATH if not raw_path else (STEP2M3R_OUTPUT_DIR / raw_path).resolve()
        root = STEP2M3R_OUTPUT_DIR.resolve()
        if not str(file_path).startswith(str(root)):
            self.send_error(403)
            return
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_review_decision(self) -> None:
        if urlparse(self.path).path != "/api/step2m3r/topology-review-decision":
            self._send_json(404, {"success": False, "error": "unknown_endpoint"})
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            _decision, reviewed_payload, progress = save_m3r_topology_review_decision(payload)
            self._send_json(
                200,
                {
                    "success": True,
                    "reviewed_count": progress.get("reviewed_candidates", len(decision_rows_from_payload(reviewed_payload))),
                    "total_review_candidates": progress.get("total_review_candidates", 0),
                    "accepted_count": progress.get("accepted_count", 0),
                    "rejected_count": progress.get("rejected_count", 0),
                    "unsure_count": progress.get("unsure_count", 0),
                },
            )
        except Exception as exc:  # pragma: no cover - surfaced to browser and smoke tests
            self._send_json(400, {"success": False, "error": str(exc)})

    def do_POST(self) -> None:
        self._handle_review_decision()

    def do_PUT(self) -> None:
        self._handle_review_decision()


def serve_step2m3r_topology_review_ui(host: str = "127.0.0.1", port: int = 8786) -> None:
    prepare_step2m3r_review_ui(host=host, port=port)
    server = ThreadingHTTPServer((host, port), Step2M3RTopologyReviewHandler)
    print(f"Serving Step2.M3R topology review UI at http://{host}:{port}/")
    print(f"Autosave endpoint: http://{host}:{port}/api/step2m3r/topology-review-decision")
    server.serve_forever()


def print_step2m3r_console(outputs: dict[str, Any]) -> None:
    group_summary = outputs["group_topology_summary"]
    edge_summary = outputs["accepted_edge_topology_summary"]
    review_payload = outputs["topology_review_candidates"]
    validation = outputs["validation_summary"]
    freeze = outputs["freeze_candidate_manifest"]
    print(f"step2m3r_output_dir: {STEP2M3R_OUTPUT_DIR.resolve()}")
    print(f"total_groups_audited: {group_summary.get('total_groups_audited', 0)}")
    print(f"accepted_edges_audited: {edge_summary.get('accepted_edges_audited', 0)}")
    print(f"high_topology_risk_group_count: {group_summary.get('high_topology_risk_group_count', 0)}")
    print(f"branch_merge_group_count: {group_summary.get('branch_merge_group_count', 0)}")
    print(f"duplicate_frame_member_group_count: {group_summary.get('duplicate_frame_member_group_count', 0)}")
    print(f"role_context_mixed_group_count: {group_summary.get('role_context_mixed_group_count', 0)}")
    print(f"review_queue_size: {len(rows_from_payload(review_payload))}")
    print(f"forbidden_keys_present: {validation.get('forbidden_keys_present', [])}")
    print(f"production_ready={str(validation.get('production_ready', False)).lower()}")
    print(f"no_auto_promotion={str(validation.get('no_auto_promotion', True)).lower()}")
    print(f"human_approved={str(validation.get('human_approved', False)).lower()}")
    print(f"step2m3r_freeze_candidate_created={str(freeze.get('step2m3r_freeze_candidate_created', False)).lower()}")


def print_step2m3r_validation_console(outputs: dict[str, Any]) -> None:
    validation = outputs["validation_summary"]
    freeze = outputs["freeze_candidate_manifest"]
    print(f"step2m3r_validation_summary_path: {STEP2M3R_VALIDATION_SUMMARY_PATH.resolve()}")
    print(f"step2m3r_freeze_candidate_manifest_path: {STEP2M3R_FREEZE_CANDIDATE_MANIFEST_PATH.resolve()}")
    print(f"blocking_issue_count: {outputs['issue_register'].get('blocking_issue_count', 0)}")
    print(f"forbidden_keys_present: {validation.get('forbidden_keys_present', [])}")
    print(f"step2m3r_freeze_candidate_created={str(freeze.get('step2m3r_freeze_candidate_created', False)).lower()}")


def print_step2m3r_review_pack_console(manifest: dict[str, Any]) -> None:
    print(f"step2m3r_review_pack_manifest_path: {STEP2M3R_REVIEW_PACK_MANIFEST_PATH.resolve()}")
    print(f"step2m3r_review_pack_dir: {manifest.get('review_pack_dir')}")
    print(f"copied_files: {len(manifest.get('copied_files', []))}")
    print("production_ready=false")
    print("no_auto_promotion=true")


def print_step2m3r_review_ui_console(manifest: dict[str, Any]) -> None:
    print(f"step2m3r_topology_review_ui_html_path: {manifest['topology_review_ui_html_path']}")
    print(f"step2m3r_topology_review_candidate_rows_path: {manifest['topology_review_candidate_rows_path']}")
    print(f"step2m3r_reviewed_topology_decisions_path: {manifest['reviewed_topology_decisions_path']}")
    print(f"step2m3r_review_progress_summary_path: {manifest['review_progress_summary_path']}")
    print(f"total_review_candidates: {manifest.get('total_review_candidates', 0)}")
    print(f"reviewed_candidates: {manifest.get('reviewed_candidates', 0)}")
    print(f"autosave_endpoint: {manifest.get('autosave_endpoint')}")
    print("visual_only_warning=VISUAL_ONLY_NOT_METRIC")
    print("production_ready=false")
    print("no_auto_promotion=true")
