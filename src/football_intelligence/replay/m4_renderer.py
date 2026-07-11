from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from football_intelligence.core.fingerprints import media_type_for_path, semantic_hash, sha256_file
from football_intelligence.replay.contracts import M4_EVIDENCE_DIRS


def _decoded_image_hash(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        from PIL import Image, ImageSequence

        with Image.open(path) as image:
            frames = []
            dimensions = []
            for frame in ImageSequence.Iterator(image):
                rgba = frame.convert("RGBA")
                frames.append(hashlib.sha256(rgba.tobytes()).hexdigest())
                dimensions.append({"width": rgba.width, "height": rgba.height})
            payload = {
                "frame_count": len(frames),
                "frame_hashes": frames,
                "dimensions": dimensions[:1],
            }
            return semantic_hash(payload), payload
    except Exception as exc:
        payload = {"frame_count": None, "frame_hashes": [], "dimensions": [], "diagnostic": str(exc)}
        return sha256_file(path), payload


def evidence_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for directory in M4_EVIDENCE_DIRS:
        base = root / directory
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            decoded_hash, decoded = _decoded_image_hash(path)
            records.append(
                {
                    "relative_uri": path.relative_to(root).as_posix(),
                    "byte_size": path.stat().st_size,
                    "content_hash": sha256_file(path),
                    "decoded_content_hash": decoded_hash,
                    "media_type": media_type_for_path(path),
                    "dimensions": decoded.get("dimensions", []),
                    "decoded_frame_count": decoded.get("frame_count"),
                    "decoded_frame_hashes": decoded.get("frame_hashes", []),
                }
            )
    return records


def evidence_inventory(root: Path) -> dict[str, Any]:
    records = evidence_records(root)
    return {
        "schema_version": "m5.replay.evidence_inventory.v1",
        "asset_count": len(records),
        "records": records,
        "evidence_inventory_hash": semantic_hash(
            [
                {
                    "relative_uri": record["relative_uri"],
                    "decoded_content_hash": record["decoded_content_hash"],
                    "media_type": record["media_type"],
                }
                for record in records
            ]
        ),
    }
