from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from football_intelligence.step2_visual_continuity.sparse_handoff_package import (
    M4_CURRENT_VISUAL_EVIDENCE_VERSION,
    M4_MAX_OVERLAY_FRAMES_PER_PATHLET,
    overlay_selection,
)
from football_intelligence.step2_visual_continuity.topology_qa import (
    cv2,
    draw_scaled_box,
    horizontal_image_strip,
    sample_frame_sequences,
    safe_asset_stem,
    tile_from_frame,
    write_animation_gif,
    write_image,
)


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def render_true_m4_pathlet_overlay(
    *,
    pathlet: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
    frame_lookup: dict[int, str],
    frame_records: dict[int, dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    frames_dir = output_root / "step2m4_pathlet_overlay_frames"
    gifs_dir = output_root / "step2m4_pathlet_overlay_gifs"
    strips_dir = output_root / "step2m4_pathlet_overlay_strips"
    frames_dir.mkdir(parents=True, exist_ok=True)
    gifs_dir.mkdir(parents=True, exist_ok=True)
    strips_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_asset_stem(str(pathlet.get("m4_handoff_pathlet_id", pathlet.get("source_m3t_pathlet_id", ""))))
    gif_path = gifs_dir / f"{stem}.gif"
    strip_path = strips_dir / f"{stem}.jpg"
    member_frames = [_safe_int(frame) for frame in pathlet.get("member_frame_sequences", [])]
    frames = sample_frame_sequences(
        _safe_int(pathlet.get("min_frame_sequence")),
        _safe_int(pathlet.get("max_frame_sequence")),
        member_frames,
        max_frames=M4_MAX_OVERLAY_FRAMES_PER_PATHLET,
    )
    members_by_frame: dict[int, list[str]] = defaultdict(list)
    for frame, member in zip(
        pathlet.get("member_frame_sequences", []),
        pathlet.get("member_visible_person_base_ids", []),
        strict=False,
    ):
        members_by_frame[_safe_int(frame)].append(str(member))
    rendered: list[Any] = []
    frame_paths: list[str] = []
    source_frame_hashes: list[dict[str, Any]] = []
    for frame in frames:
        tile, metadata = tile_from_frame(frame, frame_lookup, tile_w=560, tile_h=360)
        if tile is None:
            continue
        cv2.putText(
            tile,
            f"{pathlet.get('m4_handoff_pathlet_id', '')} frame {frame}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            "visual-only sparse continuity; not identity",
            (10, tile.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (120, 235, 245),
            1,
            cv2.LINE_AA,
        )
        for member_id in members_by_frame.get(frame, []):
            node = nodes_by_id.get(member_id, {})
            draw_scaled_box(
                tile,
                node.get("bbox", {}),
                metadata,
                label=str(pathlet.get("m4_handoff_pathlet_id", ""))[-14:],
                colour=(0, 215, 255),
                thickness=2,
            )
        frame_path = frames_dir / f"{stem}_f{frame:06d}.jpg"
        if write_image(frame_path, tile):
            frame_paths.append(_rel(frame_path, output_root))
        rendered.append(tile)
        source_frame_hashes.append(
            {
                "frame_sequence": frame,
                "source_frame_uri": frame_records.get(frame, {}).get("root_relative_source_frame_uri", ""),
                "source_frame_byte_hash": frame_records.get(frame, {}).get("byte_hash"),
            }
        )
    strip = horizontal_image_strip(rendered)
    strip_available = bool(strip is not None and write_image(strip_path, strip))
    gif_available = write_animation_gif(gif_path, rendered, duration_seconds=0.32)
    return {
        "m4_overlay_frame_paths": frame_paths,
        "m4_overlay_gif_path": _rel(gif_path, output_root) if gif_available else "",
        "m4_overlay_strip_path": _rel(strip_path, output_root) if strip_available else "",
        "m4_overlay_sampled_frame_sequences": frames,
        "asset_source_frame_hashes": source_frame_hashes,
        "m4_overlay_evidence_available": gif_available and strip_available and bool(frame_paths),
    }


def render_true_m4_overlay_assets(
    *,
    pathlets: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
    frame_lookup: dict[int, str],
    frame_lookup_payload: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    frame_records = {
        int(record["frame_sequence"]): record
        for record in frame_lookup_payload.get("records", [])
        if isinstance(record, dict)
    }
    selected = overlay_selection(pathlets)
    rendered_count = 0
    provenance: list[dict[str, Any]] = []
    for pathlet in selected:
        evidence = render_true_m4_pathlet_overlay(
            pathlet=pathlet,
            nodes_by_id=nodes_by_id,
            frame_lookup=frame_lookup,
            frame_records=frame_records,
            output_root=output_root,
        )
        pathlet.update({key: value for key, value in evidence.items() if key.startswith("m4_overlay_")})
        rendered_count += 1 if evidence.get("m4_overlay_evidence_available") is True else 0
        provenance.append(
            {
                "m4_handoff_pathlet_id": pathlet.get("m4_handoff_pathlet_id", ""),
                "source_m3t_pathlet_id": pathlet.get("source_m3t_pathlet_id", ""),
                "overlay_frame_paths": evidence.get("m4_overlay_frame_paths", []),
                "overlay_gif_path": evidence.get("m4_overlay_gif_path", ""),
                "overlay_strip_path": evidence.get("m4_overlay_strip_path", ""),
                "source_frame_hashes": evidence.get("asset_source_frame_hashes", []),
            }
        )
    gif_count = len(list((output_root / "step2m4_pathlet_overlay_gifs").glob("*.gif")))
    strip_count = len(list((output_root / "step2m4_pathlet_overlay_strips").glob("*.jpg")))
    frame_count = len(list((output_root / "step2m4_pathlet_overlay_frames").glob("*.jpg")))
    return {
        "overlay_requested_pathlet_count": len(selected),
        "overlay_evidence_pathlet_count": rendered_count,
        "overlay_gif_count": gif_count,
        "overlay_strip_count": strip_count,
        "overlay_frame_count": frame_count,
        "overlay_asset_count": gif_count + strip_count + frame_count,
        "current_visual_evidence_version": M4_CURRENT_VISUAL_EVIDENCE_VERSION,
        "asset_source_frame_records": provenance,
        "no_m4_asset_was_read": True,
    }
