# ruff: noqa: E501

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from football_intelligence.paths import CLIP_ID, MATCH_ID, ensure_dir
from football_intelligence.step2_visual_continuity.io import (
    STEP2M1_NODE_ROWS_PATH,
    STEP2M1_OUTPUT_DIR,
    STEP2M2_OUTPUT_DIR,
    STEP2M3_OUTPUT_DIR,
    STEP2M3R_OUTPUT_DIR,
    STEP2M3S_OUTPUT_DIR,
    STEP2M3T_HANDOFF_MANIFEST_PATH,
    STEP2M3T_OUTPUT_DIR,
    STEP2M3T_REVIEW_PROGRESS_SUMMARY_PATH,
    STEP2M3T_REVIEWED_SPARSE_PATHLET_DECISIONS_PATH,
    STEP2M3T_SELECTED_SPARSE_EDGES_JSONL_GZ_PATH,
    STEP2M3T_SPARSE_PATHLETS_PATH,
    STEP2M3T_TOPOLOGY_QUARANTINED_EDGES_JSONL_GZ_PATH,
    STEP2M3T_VALIDATION_SUMMARY_PATH,
    STEP2M4_FREEZE_CANDIDATE_MANIFEST_PATH,
    STEP2M4_HANDOFF_EDGES_JSONL_GZ_PATH,
    STEP2M4_HANDOFF_MANIFEST_PATH,
    STEP2M4_HANDOFF_PATHLETS_PATH,
    STEP2M4_HANDOFF_SUMMARY_PATH,
    STEP2M4_ISSUE_REGISTER_PATH,
    STEP2M4_OUTPUT_DIR,
    STEP2M4_PATHLET_OVERLAY_FRAMES_DIR,
    STEP2M4_PATHLET_OVERLAY_GIFS_DIR,
    STEP2M4_PATHLET_OVERLAY_STRIPS_DIR,
    STEP2M4_REVIEW_PACK_DIR,
    STEP2M4_REVIEW_PACK_MANIFEST_PATH,
    STEP2M4_SAFETY_GUARDRAIL_AUDIT_PATH,
    STEP2M4_VALIDATION_SUMMARY_PATH,
    STEP2M4_VIEWER_HTML_PATH,
    STEP2_VISUAL_CONTINUITY_DIR,
    read_json,
    read_jsonl_gz_rows,
    write_json,
    write_jsonl_gz,
    write_text,
)
from football_intelligence.step2_visual_continuity.schema import (
    DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_FRAMES,
    DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_SECONDS,
    NO_AUTO_PROMOTION,
    PRODUCTION_READY,
    assert_no_forbidden_keys,
    forbidden_keys_present,
    guardrail_stamp,
    rows_from_payload,
    safe_float,
    safe_int,
    utc_iso,
    visual_stamp,
)
from football_intelligence.step2_visual_continuity.topology_qa import (
    cv2,
    decision_rows_from_payload,
    draw_scaled_box,
    horizontal_image_strip,
    node_lookup_from_payload,
    sample_frame_sequences,
    safe_asset_stem,
    tile_from_frame,
    write_animation_gif,
    write_image,
)


M4_CURRENT_VISUAL_EVIDENCE_VERSION = "step2m4_sparse_handoff_overlay_v1_animation"
M4_RENDER_TARGET_PATHLETS = 50
M4_MAX_OVERLAY_FRAMES_PER_PATHLET = 16
M4_MAX_PATHLET_SPAN_FRAMES = DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_FRAMES
M4_MAX_PATHLET_SPAN_SECONDS = DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_SECONDS


def m4_guardrail_fields() -> dict[str, Any]:
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


def step2m4_output_paths() -> dict[str, Path]:
    return {
        "step2m4_output_dir": STEP2M4_OUTPUT_DIR,
        "sparse_handoff_pathlets": STEP2M4_HANDOFF_PATHLETS_PATH,
        "sparse_handoff_edges": STEP2M4_HANDOFF_EDGES_JSONL_GZ_PATH,
        "sparse_handoff_summary": STEP2M4_HANDOFF_SUMMARY_PATH,
        "pathlet_overlay_frames_dir": STEP2M4_PATHLET_OVERLAY_FRAMES_DIR,
        "pathlet_overlay_gifs_dir": STEP2M4_PATHLET_OVERLAY_GIFS_DIR,
        "pathlet_overlay_strips_dir": STEP2M4_PATHLET_OVERLAY_STRIPS_DIR,
        "viewer_html": STEP2M4_VIEWER_HTML_PATH,
        "handoff_manifest": STEP2M4_HANDOFF_MANIFEST_PATH,
        "validation_summary": STEP2M4_VALIDATION_SUMMARY_PATH,
        "safety_guardrail_audit": STEP2M4_SAFETY_GUARDRAIL_AUDIT_PATH,
        "issue_register": STEP2M4_ISSUE_REGISTER_PATH,
        "freeze_candidate_manifest": STEP2M4_FREEZE_CANDIDATE_MANIFEST_PATH,
        "review_pack_manifest": STEP2M4_REVIEW_PACK_MANIFEST_PATH,
    }


def assert_m4_output_path_isolation() -> None:
    m4_root = STEP2M4_OUTPUT_DIR.resolve()
    blocked_roots = [
        STEP2M1_OUTPUT_DIR.resolve(),
        STEP2M2_OUTPUT_DIR.resolve(),
        STEP2M3_OUTPUT_DIR.resolve(),
        STEP2M3R_OUTPUT_DIR.resolve(),
        STEP2M3S_OUTPUT_DIR.resolve(),
        STEP2M3T_OUTPUT_DIR.resolve(),
    ]
    for path in step2m4_output_paths().values():
        resolved = path.resolve()
        if resolved != m4_root and m4_root not in resolved.parents:
            raise ValueError(f"Step2.M4 output path is outside the M4 root: {resolved}")
        if any(resolved == root or root in resolved.parents for root in blocked_roots):
            raise ValueError(f"Step2.M4 output path points inside an earlier Step2 folder: {resolved}")


def m4_rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(STEP2M4_OUTPUT_DIR.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_m3t_reviewed_decision_rows(path: Path = STEP2M3T_REVIEWED_SPARSE_PATHLET_DECISIONS_PATH) -> list[dict[str, Any]]:
    payload = read_json(path)
    return decision_rows_from_payload(payload)


def m3t_review_maps(decision_rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    pathlet_rows: dict[str, dict[str, Any]] = {}
    edge_rows: dict[str, dict[str, Any]] = {}
    for row in decision_rows:
        pathlet_id = str(row.get("pathlet_id", ""))
        edge_id = str(row.get("continuity_edge_id", ""))
        if pathlet_id:
            pathlet_rows[pathlet_id] = row
        if edge_id:
            edge_rows[edge_id] = row
    return {"pathlets": pathlet_rows, "edges": edge_rows}


def risk_regions_from_pathlet(pathlet: dict[str, Any]) -> list[str]:
    regions: list[str] = []
    if pathlet.get("from_branch_merge_heavy_region") is True:
        regions.append("branch_merge_heavy_region")
    if pathlet.get("from_duplicate_frame_heavy_region") is True:
        regions.append("duplicate_frame_heavy_region")
    if pathlet.get("from_role_context_mixed_region") is True:
        regions.append("role_context_mixed_region")
    if pathlet.get("contains_m3s_handoff_seed_edge") is True:
        regions.append("m3s_seed_edge")
    return regions


def make_m4_handoff_pathlet_row(index: int, pathlet: dict[str, Any], review_row: dict[str, Any] | None = None) -> dict[str, Any]:
    reviewed = review_row is not None
    row = guardrail_stamp(
        {
            "m4_handoff_pathlet_id": f"step2m4_handoff_pathlet_{index:06d}",
            "source_m3t_pathlet_id": str(pathlet.get("pathlet_id", "")),
            "member_visible_person_base_ids": list(pathlet.get("member_visible_person_base_ids", []) or []),
            "member_frame_sequences": list(pathlet.get("member_frame_sequences", []) or []),
            "accepted_continuity_edge_ids": list(pathlet.get("accepted_continuity_edge_ids", []) or []),
            "min_frame_sequence": safe_int(pathlet.get("min_frame_sequence"), -1),
            "max_frame_sequence": safe_int(pathlet.get("max_frame_sequence"), -1),
            "frame_span": safe_int(pathlet.get("frame_span"), 0),
            "seconds_span": safe_float(pathlet.get("seconds_span"), 0.0),
            "max_members_per_frame": safe_int(pathlet.get("max_members_per_frame"), 0),
            "max_in_degree": safe_int(pathlet.get("max_in_degree"), 0),
            "max_out_degree": safe_int(pathlet.get("max_out_degree"), 0),
            "branch_count": safe_int(pathlet.get("branch_count"), 0),
            "merge_count": safe_int(pathlet.get("merge_count"), 0),
            "reviewed_by_m3t": reviewed,
            "m3t_review_decision": str((review_row or {}).get("human_review_decision", "")),
            "m3t_review_candidate_id": str((review_row or {}).get("step2m3t_review_candidate_id", "")),
            "source_risk_regions": risk_regions_from_pathlet(pathlet),
            "branch_merge_heavy_region": pathlet.get("from_branch_merge_heavy_region") is True,
            "duplicate_frame_heavy_region": pathlet.get("from_duplicate_frame_heavy_region") is True,
            "role_context_mixed_region": pathlet.get("from_role_context_mixed_region") is True,
            "m3s_seed_edge": pathlet.get("contains_m3s_handoff_seed_edge") is True,
            "handoff_ready": True,
            "pathlet_not_identity": True,
            "pathlet_not_player_slot": True,
            "pathlet_not_goalkeeper_slot": True,
            "current_visual_evidence_version": M4_CURRENT_VISUAL_EVIDENCE_VERSION,
            **m4_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(row)
    return row


def make_m4_handoff_edge_row(edge: dict[str, Any], pathlet_id_map: dict[str, str], edge_review_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    continuity_edge_id = str(edge.get("continuity_edge_id", ""))
    source_pathlet_id = str(edge.get("pathlet_id", ""))
    row = dict(edge)
    row.update(
        {
            "m4_sparse_handoff_edge": True,
            "m4_handoff_pathlet_id": pathlet_id_map.get(source_pathlet_id, ""),
            "source_m3t_pathlet_id": source_pathlet_id,
            "reviewed_by_m3t": continuity_edge_id in edge_review_rows,
            "m3t_review_decision": str(edge_review_rows.get(continuity_edge_id, {}).get("human_review_decision", "")),
            "handoff_ready": True,
            "pathlet_not_identity": True,
            "pathlet_not_player_slot": True,
            "pathlet_not_goalkeeper_slot": True,
            **m4_guardrail_fields(),
        }
    )
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def build_m4_handoff_rows(
    m3t_pathlets: list[dict[str, Any]],
    selected_edges: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    maps = m3t_review_maps(decision_rows)
    pathlet_rows = [
        make_m4_handoff_pathlet_row(index, pathlet, maps["pathlets"].get(str(pathlet.get("pathlet_id", ""))))
        for index, pathlet in enumerate(sorted(m3t_pathlets, key=lambda row: str(row.get("pathlet_id", ""))), start=1)
    ]
    pathlet_id_map = {str(row.get("source_m3t_pathlet_id", "")): str(row.get("m4_handoff_pathlet_id", "")) for row in pathlet_rows}
    edge_rows = [
        make_m4_handoff_edge_row(edge, pathlet_id_map, maps["edges"])
        for edge in selected_edges
        if str(edge.get("pathlet_id", "")) in pathlet_id_map
    ]
    return pathlet_rows, edge_rows


def overlay_selection(pathlets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reviewed = [row for row in pathlets if row.get("reviewed_by_m3t") is True]
    selected_ids = {str(row.get("source_m3t_pathlet_id", "")) for row in reviewed}
    target_count = max(M4_RENDER_TARGET_PATHLETS, len(reviewed))
    candidates = sorted(
        [row for row in pathlets if str(row.get("source_m3t_pathlet_id", "")) not in selected_ids],
        key=lambda row: (
            row.get("branch_merge_heavy_region") is True,
            row.get("duplicate_frame_heavy_region") is True,
            row.get("role_context_mixed_region") is True,
            row.get("m3s_seed_edge") is True,
            safe_int(row.get("frame_span"), 0),
            str(row.get("source_m3t_pathlet_id", "")),
        ),
        reverse=True,
    )
    return reviewed + candidates[: max(0, target_count - len(reviewed))]


def render_m4_pathlet_overlay(pathlet: dict[str, Any], nodes_by_id: dict[str, dict[str, Any]], frame_lookup: dict[int, str]) -> dict[str, Any]:
    ensure_dir(STEP2M4_PATHLET_OVERLAY_FRAMES_DIR)
    ensure_dir(STEP2M4_PATHLET_OVERLAY_GIFS_DIR)
    ensure_dir(STEP2M4_PATHLET_OVERLAY_STRIPS_DIR)
    stem = safe_asset_stem(str(pathlet.get("m4_handoff_pathlet_id", pathlet.get("source_m3t_pathlet_id", ""))))
    gif_path = STEP2M4_PATHLET_OVERLAY_GIFS_DIR / f"{stem}.gif"
    strip_path = STEP2M4_PATHLET_OVERLAY_STRIPS_DIR / f"{stem}.jpg"
    member_frames = [safe_int(frame, -1) for frame in pathlet.get("member_frame_sequences", [])]
    frames = sample_frame_sequences(
        safe_int(pathlet.get("min_frame_sequence"), -1),
        safe_int(pathlet.get("max_frame_sequence"), -1),
        member_frames,
        max_frames=M4_MAX_OVERLAY_FRAMES_PER_PATHLET,
    )
    members_by_frame: dict[int, list[str]] = defaultdict(list)
    for frame, member in zip(pathlet.get("member_frame_sequences", []), pathlet.get("member_visible_person_base_ids", []), strict=False):
        members_by_frame[safe_int(frame, -1)].append(str(member))
    rendered: list[Any] = []
    frame_paths: list[str] = []
    for frame in frames:
        tile, metadata = tile_from_frame(frame, frame_lookup, tile_w=560, tile_h=360)
        if tile is None:
            continue
        cv2.putText(tile, f"{pathlet.get('m4_handoff_pathlet_id', '')} frame {frame}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(tile, "visual-only sparse continuity; not identity", (10, tile.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 235, 245), 1, cv2.LINE_AA)
        for _member_id in members_by_frame.get(frame, []):
            node = nodes_by_id.get(_member_id, {})
            draw_scaled_box(tile, node.get("bbox", {}), metadata, label=str(pathlet.get("m4_handoff_pathlet_id", ""))[-14:], colour=(0, 215, 255), thickness=2)
        frame_path = STEP2M4_PATHLET_OVERLAY_FRAMES_DIR / f"{stem}_f{frame:06d}.jpg"
        if write_image(frame_path, tile):
            frame_paths.append(m4_rel_path(frame_path))
        rendered.append(tile)
    strip = horizontal_image_strip(rendered)
    strip_available = bool(strip is not None and write_image(strip_path, strip))
    gif_available = write_animation_gif(gif_path, rendered, duration_seconds=0.32)
    return {
        "m4_overlay_frame_paths": frame_paths,
        "m4_overlay_gif_path": m4_rel_path(gif_path) if gif_available else "",
        "m4_overlay_strip_path": m4_rel_path(strip_path) if strip_available else "",
        "m4_overlay_sampled_frame_sequences": frames,
        "m4_overlay_evidence_available": gif_available and strip_available and bool(frame_paths),
    }


def render_m4_overlay_assets(pathlets: list[dict[str, Any]]) -> dict[str, Any]:
    selected = overlay_selection(pathlets)
    node_payload = read_json(STEP2M1_NODE_ROWS_PATH)
    nodes_by_id, _nodes_by_frame, frame_lookup = node_lookup_from_payload(node_payload)
    rendered_count = 0
    for pathlet in selected:
        evidence = render_m4_pathlet_overlay(pathlet, nodes_by_id, frame_lookup)
        pathlet.update(evidence)
        rendered_count += 1 if evidence.get("m4_overlay_evidence_available") is True else 0
    gif_count = len(list(STEP2M4_PATHLET_OVERLAY_GIFS_DIR.glob("*.gif"))) if STEP2M4_PATHLET_OVERLAY_GIFS_DIR.exists() else 0
    strip_count = len(list(STEP2M4_PATHLET_OVERLAY_STRIPS_DIR.glob("*.jpg"))) if STEP2M4_PATHLET_OVERLAY_STRIPS_DIR.exists() else 0
    frame_count = len(list(STEP2M4_PATHLET_OVERLAY_FRAMES_DIR.glob("*.jpg"))) if STEP2M4_PATHLET_OVERLAY_FRAMES_DIR.exists() else 0
    return {
        "overlay_requested_pathlet_count": len(selected),
        "overlay_evidence_pathlet_count": rendered_count,
        "overlay_gif_count": gif_count,
        "overlay_strip_count": strip_count,
        "overlay_frame_count": frame_count,
        "overlay_asset_count": gif_count + strip_count + frame_count,
        "current_visual_evidence_version": M4_CURRENT_VISUAL_EVIDENCE_VERSION,
    }


def count_pathlet_violations(pathlets: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "pathlets_over_cap": sum(
            1
            for row in pathlets
            if safe_int(row.get("frame_span"), 0) > M4_MAX_PATHLET_SPAN_FRAMES or safe_float(row.get("seconds_span"), 0.0) > M4_MAX_PATHLET_SPAN_SECONDS
        ),
        "duplicate_frame_pathlets": sum(1 for row in pathlets if safe_int(row.get("max_members_per_frame"), 0) > 1),
        "branch_merge_pathlets": sum(
            1
            for row in pathlets
            if safe_int(row.get("branch_count"), 0) > 0
            or safe_int(row.get("merge_count"), 0) > 0
            or safe_int(row.get("max_in_degree"), 0) > 1
            or safe_int(row.get("max_out_degree"), 0) > 1
        ),
    }


def build_m4_handoff_summary(
    pathlets: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    quarantined_edge_count: int,
    decision_rows: list[dict[str, Any]],
    m3t_manifest: dict[str, Any],
    overlay_summary: dict[str, Any],
) -> dict[str, Any]:
    violations = count_pathlet_violations(pathlets)
    risk_counts = Counter(region for pathlet in pathlets for region in pathlet.get("source_risk_regions", []))
    summary = guardrail_stamp(
        {
            "artifact": "step2m4_sparse_handoff_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "source_m3t_folder": str(STEP2M3T_OUTPUT_DIR.resolve()),
            "source_m3t_reviewed_decisions_count": len(decision_rows),
            "source_sparse_pathlet_count": m3t_manifest.get("sparse_pathlet_count", len(pathlets)),
            "source_sparse_selected_edge_count": m3t_manifest.get("sparse_selected_edge_count", len(edges)),
            "source_topology_quarantined_edge_count": quarantined_edge_count,
            "m4_handoff_pathlet_count": len(pathlets),
            "m4_handoff_edge_count": len(edges),
            "reviewed_by_m3t_pathlet_count": sum(1 for row in pathlets if row.get("reviewed_by_m3t") is True),
            "source_risk_region_counts": dict(sorted(risk_counts.items())),
            "max_pathlet_span_frames_observed": max((safe_int(row.get("frame_span"), 0) for row in pathlets), default=0),
            "max_pathlet_span_seconds_observed": max((safe_float(row.get("seconds_span"), 0.0) for row in pathlets), default=0.0),
            **violations,
            **overlay_summary,
            "viewer_path": str(STEP2M4_VIEWER_HTML_PATH.resolve()),
            "qa_sample_implemented": False,
            **m4_guardrail_fields(),
        }
    )
    summary["forbidden_keys_present"] = sorted(set(forbidden_keys_present(summary)) | set(forbidden_keys_present(pathlets[:100])) | set(forbidden_keys_present(edges[:100])))
    assert_no_forbidden_keys(summary)
    return summary


def viewer_html(pathlets: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    viewer_rows = [
        {
            "m4_handoff_pathlet_id": row.get("m4_handoff_pathlet_id"),
            "source_m3t_pathlet_id": row.get("source_m3t_pathlet_id"),
            "min_frame_sequence": row.get("min_frame_sequence"),
            "max_frame_sequence": row.get("max_frame_sequence"),
            "frame_span": row.get("frame_span"),
            "seconds_span": row.get("seconds_span"),
            "reviewed_by_m3t": row.get("reviewed_by_m3t"),
            "m3t_review_decision": row.get("m3t_review_decision"),
            "source_risk_regions": row.get("source_risk_regions", []),
            "m4_overlay_gif_path": row.get("m4_overlay_gif_path", ""),
            "m4_overlay_strip_path": row.get("m4_overlay_strip_path", ""),
            "m4_overlay_evidence_available": row.get("m4_overlay_evidence_available", False),
            "member_count": len(row.get("member_visible_person_base_ids", []) or []),
            "accepted_edge_count": len(row.get("accepted_continuity_edge_ids", []) or []),
            "max_members_per_frame": row.get("max_members_per_frame"),
            "max_in_degree": row.get("max_in_degree"),
            "max_out_degree": row.get("max_out_degree"),
            "branch_count": row.get("branch_count"),
            "merge_count": row.get("merge_count"),
            "handoff_ready": row.get("handoff_ready"),
            "visual_only_warning": row.get("visual_only_warning"),
        }
        for row in pathlets
    ]
    payload_json = json.dumps({"rows": viewer_rows, "summary": summary}, indent=2).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Step2.M4 Sparse Visual Continuity Handoff Viewer</title>
<style>
:root{{font-family:Arial,sans-serif;color:#17212d;background:#f4f6f8;}}
body{{margin:0;}}
header{{position:sticky;top:0;background:#17212d;color:#fff;padding:12px 18px;z-index:2;display:flex;justify-content:space-between;gap:12px;align-items:center;}}
main{{display:grid;grid-template-columns:360px minmax(0,1fr);gap:14px;max-width:1320px;margin:0 auto;padding:14px;}}
.panel{{background:#fff;border:1px solid #d8dee6;border-radius:8px;padding:12px;}}
.warning{{font-weight:700;color:#8a2d12;}}
.filters{{display:grid;gap:8px;}}
label{{font-size:12px;color:#536170;font-weight:700;}}
input,select{{width:100%;box-sizing:border-box;border:1px solid #c8d0da;border-radius:6px;padding:8px;background:#fff;}}
button{{border:0;border-radius:6px;padding:8px 10px;font-weight:700;background:#dbe3ec;color:#17212d;cursor:pointer;}}
.list{{display:grid;gap:8px;max-height:64vh;overflow:auto;margin-top:10px;}}
.item{{border:1px solid #e1e6ed;border-radius:6px;padding:8px;background:#fafbfc;cursor:pointer;}}
.item.active{{border-color:#1264a3;background:#eef7ff;}}
.meta{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:10px;}}
.field{{background:#f7f8fa;border:1px solid #e1e6ed;border-radius:6px;padding:8px;min-height:46px;}}
.label{{font-size:11px;text-transform:uppercase;color:#647386;}}
.value{{margin-top:3px;word-break:break-word;}}
.evidence{{width:100%;max-height:60vh;object-fit:contain;background:#101820;border:1px solid #c8d0da;border-radius:6px;}}
.strip{{width:100%;max-height:22vh;object-fit:contain;background:#101820;border:1px solid #c8d0da;border-radius:6px;margin-top:8px;}}
.chips{{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;}}
.chip{{font-size:12px;background:#e8edf3;border-radius:999px;padding:4px 8px;}}
@media(max-width:900px){{main{{grid-template-columns:1fr;}}.meta{{grid-template-columns:1fr 1fr;}}}}
</style>
</head>
<body>
<header><strong>Step2.M4 Sparse Handoff Viewer</strong><span id="count"></span></header>
<main>
<aside class="panel">
<p class="warning">This is visual-only sparse continuity. Do not infer identity, player slots, events, tactics, or metrics.</p>
<div class="filters">
<label>Frame From<input id="from" type="number"></label>
<label>Frame To<input id="to" type="number"></label>
<label>Reviewed Filter<select id="reviewed"><option value="all">All</option><option value="reviewed">Reviewed by M3T</option><option value="unreviewed">Unreviewed</option></select></label>
<label>Risk Region<select id="risk"><option value="all">All</option><option value="branch_merge_heavy_region">branch_merge_heavy_region</option><option value="duplicate_frame_heavy_region">duplicate_frame_heavy_region</option><option value="role_context_mixed_region">role_context_mixed_region</option><option value="m3s_seed_edge">m3s_seed_edge</option></select></label>
<button onclick="applyFilters()">Apply</button>
</div>
<div id="list" class="list"></div>
</aside>
<section class="panel">
<div id="detail"></div>
</section>
</main>
<script>
const data={payload_json};
let rows=data.rows||[];
let filtered=rows.slice();
let active=0;
function esc(v){{return String(v??'').replace(/[&<>"]/g,s=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[s]));}}
function field(label,value){{return `<div class="field"><div class="label">${{esc(label)}}</div><div class="value">${{esc(value)}}</div></div>`;}}
function applyFilters(){{
 const from=Number(document.getElementById('from').value||'-1');
 const to=Number(document.getElementById('to').value||'999999');
 const reviewed=document.getElementById('reviewed').value;
 const risk=document.getElementById('risk').value;
 filtered=rows.filter(row=>{{
   if(Number(row.max_frame_sequence)<from || Number(row.min_frame_sequence)>to)return false;
   if(reviewed==='reviewed' && row.reviewed_by_m3t!==true)return false;
   if(reviewed==='unreviewed' && row.reviewed_by_m3t===true)return false;
   if(risk!=='all' && !(row.source_risk_regions||[]).includes(risk))return false;
   return true;
 }});
 active=0; renderList(); renderDetail();
}}
function renderList(){{
 document.getElementById('count').textContent=`${{filtered.length}} / ${{rows.length}} pathlets`;
 document.getElementById('list').innerHTML=filtered.map((row,i)=>`<div class="item ${{i===active?'active':''}}" onclick="active=${{i}};renderList();renderDetail();"><strong>${{esc(row.m4_handoff_pathlet_id)}}</strong><br>frames ${{esc(row.min_frame_sequence)}}-${{esc(row.max_frame_sequence)}} · span ${{esc(row.frame_span)}}<div class="chips">${{(row.source_risk_regions||[]).map(r=>`<span class="chip">${{esc(r)}}</span>`).join('')}}</div></div>`).join('');
}}
function renderDetail(){{
 const row=filtered[active];
 if(!row){{document.getElementById('detail').innerHTML='<p>No pathlets match the filters.</p>';return;}}
 const gif=row.m4_overlay_gif_path?`<img class="evidence" src="${{esc(row.m4_overlay_gif_path)}}" alt="M4 pathlet overlay GIF">`:'<p class="warning">GIF evidence not rendered for this pathlet.</p>';
 const strip=row.m4_overlay_strip_path?`<img class="strip" src="${{esc(row.m4_overlay_strip_path)}}" alt="M4 pathlet overlay strip">`:'';
 document.getElementById('detail').innerHTML=`<h2>${{esc(row.m4_handoff_pathlet_id)}}</h2><p class="warning">visual-only sparse continuity; not identity</p>${{gif}}${{strip}}<div class="meta">${{field('source M3T pathlet',row.source_m3t_pathlet_id)}}${{field('frames',`${{row.min_frame_sequence}} -> ${{row.max_frame_sequence}}`)}}${{field('reviewed by M3T',row.reviewed_by_m3t)}}${{field('M3T decision',row.m3t_review_decision||'')}}${{field('members',row.member_count)}}${{field('accepted edges',row.accepted_edge_count)}}${{field('max members/frame',row.max_members_per_frame)}}${{field('branch/merge',`${{row.branch_count}} / ${{row.merge_count}}`)}}${{field('warning',row.visual_only_warning)}}</div>`;
}}
renderList(); renderDetail();
</script>
</body>
</html>"""


def write_viewer(pathlets: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    write_text(STEP2M4_VIEWER_HTML_PATH, viewer_html(pathlets, summary))


def m4_freeze_gate_checks(
    *,
    m3t_handoff: dict[str, Any],
    m3t_progress: dict[str, Any],
    m3t_validation: dict[str, Any],
    summary: dict[str, Any],
    forbidden: list[str],
    viewer_exists: bool,
) -> dict[str, bool]:
    return {
        "m3t_future_handoff_ready_candidate": m3t_handoff.get("future_handoff_ready_candidate") is True,
        "m3t_reviewed_candidates_40": safe_int(m3t_progress.get("reviewed_candidates"), 0) == 40,
        "m3t_sparse_pathlet_review_completed": m3t_progress.get("sparse_pathlet_review_completed") is True,
        "m3t_forbidden_keys_absent": m3t_handoff.get("forbidden_keys_present", []) == [] and m3t_validation.get("forbidden_keys_present", []) == [],
        "m4_pathlet_count_positive": safe_int(summary.get("m4_handoff_pathlet_count"), 0) > 0,
        "m4_overlay_assets_generated": safe_int(summary.get("overlay_gif_count"), 0) > 0 and safe_int(summary.get("overlay_strip_count"), 0) > 0 and viewer_exists,
        "m4_pathlets_cap_safe": safe_int(summary.get("pathlets_over_cap"), 1) == 0,
        "m4_pathlets_max_one_member_per_frame": safe_int(summary.get("duplicate_frame_pathlets"), 1) == 0,
        "m4_pathlets_branch_merge_free": safe_int(summary.get("branch_merge_pathlets"), 1) == 0,
        "m4_forbidden_keys_absent": forbidden == [],
        "production_ready_false": PRODUCTION_READY is False,
        "no_auto_promotion_true": NO_AUTO_PROMOTION is True,
    }


def build_m4_validation_outputs(
    *,
    summary: dict[str, Any],
    pathlets: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    m3t_handoff: dict[str, Any],
    m3t_progress: dict[str, Any],
    m3t_validation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    forbidden = sorted(
        set(forbidden_keys_present(summary))
        | set(forbidden_keys_present(pathlets[:100]))
        | set(forbidden_keys_present(edges[:100]))
        | set(forbidden_keys_present(m3t_handoff))
        | set(forbidden_keys_present(m3t_progress))
        | set(forbidden_keys_present(m3t_validation))
    )
    gate_checks = m4_freeze_gate_checks(
        m3t_handoff=m3t_handoff,
        m3t_progress=m3t_progress,
        m3t_validation=m3t_validation,
        summary=summary,
        forbidden=forbidden,
        viewer_exists=STEP2M4_VIEWER_HTML_PATH.exists(),
    )
    issues = [
        {"severity": "blocking", "issue_code": code, "message": f"Step2.M4 gate failed: {code}"}
        for code, passed in gate_checks.items()
        if not passed
    ]
    freeze_candidate = all(gate_checks.values()) and not issues
    manifest = guardrail_stamp(
        {
            "artifact": "step2m4_handoff_manifest",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "source_m3t_folder": str(STEP2M3T_OUTPUT_DIR.resolve()),
            "source_m3t_reviewed_decisions_count": summary.get("source_m3t_reviewed_decisions_count", 0),
            "source_sparse_pathlet_count": summary.get("source_sparse_pathlet_count", 0),
            "source_sparse_selected_edge_count": summary.get("source_sparse_selected_edge_count", 0),
            "m4_handoff_pathlet_count": summary.get("m4_handoff_pathlet_count", 0),
            "m4_handoff_edge_count": summary.get("m4_handoff_edge_count", 0),
            "overlay_asset_count": summary.get("overlay_asset_count", 0),
            "viewer_path": str(STEP2M4_VIEWER_HTML_PATH.resolve()),
            "max_pathlet_span_frames_observed": summary.get("max_pathlet_span_frames_observed", 0),
            "max_pathlet_span_seconds_observed": summary.get("max_pathlet_span_seconds_observed", 0.0),
            "pathlets_over_cap": summary.get("pathlets_over_cap", 0),
            "pathlets_with_duplicate_frame_members": summary.get("duplicate_frame_pathlets", 0),
            "pathlets_with_branch_merge": summary.get("branch_merge_pathlets", 0),
            "handoff_safe_candidate": freeze_candidate,
            "forbidden_keys_present": forbidden,
            "gate_checks": gate_checks,
            **m4_guardrail_fields(),
        }
    )
    validation = guardrail_stamp(
        {
            "artifact": "step2m4_validation_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "m4_handoff_pathlet_count": summary.get("m4_handoff_pathlet_count", 0),
            "m4_handoff_edge_count": summary.get("m4_handoff_edge_count", 0),
            "overlay_asset_count": summary.get("overlay_asset_count", 0),
            "viewer_exists": STEP2M4_VIEWER_HTML_PATH.exists(),
            "pathlets_over_cap": summary.get("pathlets_over_cap", 0),
            "duplicate_frame_pathlets": summary.get("duplicate_frame_pathlets", 0),
            "branch_merge_pathlets": summary.get("branch_merge_pathlets", 0),
            "step2m4_freeze_candidate_created": freeze_candidate,
            "forbidden_keys_present": forbidden,
            "gate_checks": gate_checks,
            **m4_guardrail_fields(),
        }
    )
    audit = guardrail_stamp(
        {
            "artifact": "step2m4_safety_guardrail_audit",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "step2_visual_continuity_root": str(STEP2_VISUAL_CONTINUITY_DIR.resolve()),
            "m3t_read_root": str(STEP2M3T_OUTPUT_DIR.resolve()),
            "m4_write_root": str(STEP2M4_OUTPUT_DIR.resolve()),
            "no_m4_writes_to_m1_m2_m3_m3r_m3s_m3t": True,
            "qa_sample_implemented": False,
            "forbidden_keys_present": forbidden,
            **m4_guardrail_fields(),
        }
    )
    issue_register = guardrail_stamp(
        {
            "artifact": "step2m4_issue_register",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "blocking_issue_count": sum(1 for issue in issues if issue.get("severity") == "blocking"),
            "rows": issues,
            **m4_guardrail_fields(),
        }
    )
    freeze_manifest = guardrail_stamp(
        {
            "artifact": "step2m4_freeze_candidate_manifest",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "step2m4_freeze_candidate_created": freeze_candidate,
            "human_approved": False,
            "safe_to_apply_globally": False,
            "production_ready": PRODUCTION_READY,
            "no_auto_promotion": NO_AUTO_PROMOTION,
            "validation_summary_path": str(STEP2M4_VALIDATION_SUMMARY_PATH.resolve()),
            "handoff_manifest_path": str(STEP2M4_HANDOFF_MANIFEST_PATH.resolve()),
            "viewer_path": str(STEP2M4_VIEWER_HTML_PATH.resolve()),
            "forbidden_keys_present": forbidden,
            "gate_checks": gate_checks,
            **m4_guardrail_fields(),
        }
    )
    for payload in [manifest, validation, audit, issue_register, freeze_manifest]:
        assert_no_forbidden_keys(payload)
    return manifest, validation, audit, issue_register, freeze_manifest


def build_step2m4_sparse_handoff_package() -> dict[str, Any]:
    assert_m4_output_path_isolation()
    ensure_dir(STEP2M4_OUTPUT_DIR)
    m3t_handoff = read_json(STEP2M3T_HANDOFF_MANIFEST_PATH)
    m3t_progress = read_json(STEP2M3T_REVIEW_PROGRESS_SUMMARY_PATH)
    m3t_validation = read_json(STEP2M3T_VALIDATION_SUMMARY_PATH)
    decision_rows = load_m3t_reviewed_decision_rows()
    m3t_pathlets = rows_from_payload(read_json(STEP2M3T_SPARSE_PATHLETS_PATH))
    selected_edges = read_jsonl_gz_rows(STEP2M3T_SELECTED_SPARSE_EDGES_JSONL_GZ_PATH)
    quarantined_edges = read_jsonl_gz_rows(STEP2M3T_TOPOLOGY_QUARANTINED_EDGES_JSONL_GZ_PATH)
    handoff_pathlets, handoff_edges = build_m4_handoff_rows(m3t_pathlets, selected_edges, decision_rows)
    overlay_summary = render_m4_overlay_assets(handoff_pathlets)
    summary = build_m4_handoff_summary(
        handoff_pathlets,
        handoff_edges,
        len(quarantined_edges),
        decision_rows,
        m3t_handoff,
        overlay_summary,
    )
    write_viewer(handoff_pathlets, summary)
    manifest, validation, audit, issue_register, freeze_manifest = build_m4_validation_outputs(
        summary=summary,
        pathlets=handoff_pathlets,
        edges=handoff_edges,
        m3t_handoff=m3t_handoff,
        m3t_progress=m3t_progress,
        m3t_validation=m3t_validation,
    )
    for payload, path in [
        ({"artifact": "step2m4_sparse_handoff_pathlets", "created_at": utc_iso(), "source_match_id": MATCH_ID, "source_clip_id": CLIP_ID, "rows": handoff_pathlets, **m4_guardrail_fields()}, STEP2M4_HANDOFF_PATHLETS_PATH),
        (summary, STEP2M4_HANDOFF_SUMMARY_PATH),
        (manifest, STEP2M4_HANDOFF_MANIFEST_PATH),
        (validation, STEP2M4_VALIDATION_SUMMARY_PATH),
        (audit, STEP2M4_SAFETY_GUARDRAIL_AUDIT_PATH),
        (issue_register, STEP2M4_ISSUE_REGISTER_PATH),
        (freeze_manifest, STEP2M4_FREEZE_CANDIDATE_MANIFEST_PATH),
    ]:
        assert_no_forbidden_keys(payload)
        write_json(path, payload)
    write_jsonl_gz(STEP2M4_HANDOFF_EDGES_JSONL_GZ_PATH, handoff_edges)
    return {
        "handoff_pathlets": handoff_pathlets,
        "handoff_edges": handoff_edges,
        "handoff_summary": summary,
        "handoff_manifest": manifest,
        "validation_summary": validation,
        "safety_guardrail_audit": audit,
        "issue_register": issue_register,
        "freeze_candidate_manifest": freeze_manifest,
    }


def validate_step2m4_sparse_handoff_package() -> dict[str, Any]:
    assert_m4_output_path_isolation()
    pathlets = rows_from_payload(read_json(STEP2M4_HANDOFF_PATHLETS_PATH))
    edges = read_jsonl_gz_rows(STEP2M4_HANDOFF_EDGES_JSONL_GZ_PATH)
    summary = read_json(STEP2M4_HANDOFF_SUMMARY_PATH)
    m3t_handoff = read_json(STEP2M3T_HANDOFF_MANIFEST_PATH)
    m3t_progress = read_json(STEP2M3T_REVIEW_PROGRESS_SUMMARY_PATH)
    m3t_validation = read_json(STEP2M3T_VALIDATION_SUMMARY_PATH)
    manifest, validation, audit, issue_register, freeze_manifest = build_m4_validation_outputs(
        summary=summary,
        pathlets=pathlets,
        edges=edges,
        m3t_handoff=m3t_handoff,
        m3t_progress=m3t_progress,
        m3t_validation=m3t_validation,
    )
    for payload, path in [
        (manifest, STEP2M4_HANDOFF_MANIFEST_PATH),
        (validation, STEP2M4_VALIDATION_SUMMARY_PATH),
        (audit, STEP2M4_SAFETY_GUARDRAIL_AUDIT_PATH),
        (issue_register, STEP2M4_ISSUE_REGISTER_PATH),
        (freeze_manifest, STEP2M4_FREEZE_CANDIDATE_MANIFEST_PATH),
    ]:
        assert_no_forbidden_keys(payload)
        write_json(path, payload)
    return {
        "handoff_manifest": manifest,
        "validation_summary": validation,
        "safety_guardrail_audit": audit,
        "issue_register": issue_register,
        "freeze_candidate_manifest": freeze_manifest,
    }


def write_step2m4_review_pack() -> dict[str, Any]:
    assert_m4_output_path_isolation()
    ensure_dir(STEP2M4_REVIEW_PACK_DIR)
    files = [
        STEP2M4_HANDOFF_PATHLETS_PATH,
        STEP2M4_HANDOFF_EDGES_JSONL_GZ_PATH,
        STEP2M4_HANDOFF_SUMMARY_PATH,
        STEP2M4_VIEWER_HTML_PATH,
        STEP2M4_HANDOFF_MANIFEST_PATH,
        STEP2M4_VALIDATION_SUMMARY_PATH,
        STEP2M4_SAFETY_GUARDRAIL_AUDIT_PATH,
        STEP2M4_ISSUE_REGISTER_PATH,
        STEP2M4_FREEZE_CANDIDATE_MANIFEST_PATH,
    ]
    copied: list[str] = []
    for path in files:
        if not path.exists():
            continue
        destination = STEP2M4_REVIEW_PACK_DIR / path.name
        shutil.copyfile(path, destination)
        copied.append(str(destination.resolve()))
    copied_visual_assets: list[str] = []
    for source_root in [STEP2M4_PATHLET_OVERLAY_GIFS_DIR, STEP2M4_PATHLET_OVERLAY_STRIPS_DIR, STEP2M4_PATHLET_OVERLAY_FRAMES_DIR]:
        if not source_root.exists():
            continue
        relative_root = source_root.relative_to(STEP2M4_OUTPUT_DIR)
        for extension in ("*.gif", "*.jpg"):
            for source_path in source_root.glob(extension):
                destination = STEP2M4_REVIEW_PACK_DIR / relative_root / source_path.name
                ensure_dir(destination.parent)
                shutil.copyfile(source_path, destination)
                copied_visual_assets.append(str(destination.resolve()))
    validation = read_json(STEP2M4_VALIDATION_SUMMARY_PATH)
    manifest = guardrail_stamp(
        {
            "artifact": "step2m4_review_pack_manifest",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "review_pack_dir": str(STEP2M4_REVIEW_PACK_DIR.resolve()),
            "copied_files": copied,
            "copied_visual_assets": copied_visual_assets,
            "viewer_path": str(STEP2M4_VIEWER_HTML_PATH.resolve()),
            "step2m4_freeze_candidate_created": validation.get("step2m4_freeze_candidate_created", False),
            **m4_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(manifest)
    write_json(STEP2M4_REVIEW_PACK_MANIFEST_PATH, manifest)
    return manifest


def print_step2m4_console(outputs: dict[str, Any]) -> None:
    summary = outputs["handoff_summary"]
    validation = outputs["validation_summary"]
    freeze = outputs["freeze_candidate_manifest"]
    print(f"step2m4_output_dir: {STEP2M4_OUTPUT_DIR.resolve()}")
    print(f"m3t_reviewed_decisions_loaded: {summary.get('source_m3t_reviewed_decisions_count', 0)}")
    print(f"m3t_future_handoff_ready_candidate={str(validation.get('gate_checks', {}).get('m3t_future_handoff_ready_candidate', False)).lower()}")
    print(f"m4_handoff_pathlet_count: {summary.get('m4_handoff_pathlet_count', 0)}")
    print(f"m4_handoff_edge_count: {summary.get('m4_handoff_edge_count', 0)}")
    print(f"overlay_asset_count: {summary.get('overlay_asset_count', 0)}")
    print(f"viewer_path: {STEP2M4_VIEWER_HTML_PATH.resolve()}")
    print(f"pathlets_over_cap: {summary.get('pathlets_over_cap', 0)}")
    print(f"duplicate_frame_pathlets: {summary.get('duplicate_frame_pathlets', 0)}")
    print(f"branch_merge_pathlets: {summary.get('branch_merge_pathlets', 0)}")
    print(f"forbidden_keys_present: {json.dumps(validation.get('forbidden_keys_present', []))}")
    print(f"production_ready={str(validation.get('production_ready')).lower()}")
    print(f"no_auto_promotion={str(validation.get('no_auto_promotion')).lower()}")
    print(f"human_approved={str(validation.get('human_approved')).lower()}")
    print(f"step2m4_freeze_candidate_created={str(freeze.get('step2m4_freeze_candidate_created')).lower()}")


def print_step2m4_validation_console(outputs: dict[str, Any]) -> None:
    validation = outputs["validation_summary"]
    freeze = outputs["freeze_candidate_manifest"]
    issues = outputs["issue_register"]
    print(f"step2m4_validation_summary_path: {STEP2M4_VALIDATION_SUMMARY_PATH.resolve()}")
    print(f"step2m4_freeze_candidate_manifest_path: {STEP2M4_FREEZE_CANDIDATE_MANIFEST_PATH.resolve()}")
    print(f"blocking_issue_count: {issues.get('blocking_issue_count', 0)}")
    print(f"forbidden_keys_present: {json.dumps(validation.get('forbidden_keys_present', []))}")
    print(f"step2m4_freeze_candidate_created={str(freeze.get('step2m4_freeze_candidate_created')).lower()}")


def print_step2m4_review_pack_console(manifest: dict[str, Any]) -> None:
    print(f"step2m4_review_pack_manifest_path: {STEP2M4_REVIEW_PACK_MANIFEST_PATH.resolve()}")
    print(f"step2m4_review_pack_dir: {STEP2M4_REVIEW_PACK_DIR.resolve()}")
    print(f"copied_files: {len(manifest.get('copied_files', []))}")
    print(f"copied_visual_assets: {len(manifest.get('copied_visual_assets', []))}")
    print(f"production_ready={str(manifest.get('production_ready')).lower()}")
    print(f"no_auto_promotion={str(manifest.get('no_auto_promotion')).lower()}")
