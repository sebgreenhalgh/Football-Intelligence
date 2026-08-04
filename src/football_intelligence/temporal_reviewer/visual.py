"""Deterministic, geometry-preserving review-only image enhancement."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

VISUAL_CONTRACT = "G7E_B_R6_1_REVIEW_ONLY_PHOTOMETRIC_ENHANCEMENT_V1"
JPEG_QUALITY = 95


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _metrics(image: np.ndarray) -> dict[str, float | int | bool]:
    luminance = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
    measured = luminance[luminance > 10]
    if measured.size == 0:
        measured = luminance.reshape(-1)
    median = float(np.median(measured))
    p10 = float(np.percentile(measured, 10))
    p90 = float(np.percentile(measured, 90))
    low_light = median < 96.0 or (median < 116.0 and p10 < 42.0)
    return {
        "luminance_mean": float(np.mean(measured)),
        "luminance_median": median,
        "luminance_p10": p10,
        "luminance_p90": p90,
        "measured_pixel_count": int(measured.size),
        "low_light": low_light,
    }


def enhance_review_image(image: np.ndarray, metrics: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Enhance only LAB luminance; never resize, warp, or alter chroma channels."""

    median = float(metrics["luminance_median"])
    strength = float(np.clip((132.0 - median) / 90.0, 0.12, 0.68))
    gamma = 1.0 - 0.28 * strength
    clip_limit = 1.0 + 1.8 * strength
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    original_l = lab[:, :, 0]
    table = np.array([round(255.0 * ((value / 255.0) ** gamma)) for value in range(256)], dtype=np.uint8)
    gamma_l = cv2.LUT(original_l, table)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    contrast_l = clahe.apply(gamma_l)
    blend = 0.35 + 0.45 * strength
    enhanced_l = cv2.addWeighted(gamma_l, 1.0 - blend, contrast_l, blend, 0)
    result_lab = lab.copy()
    result_lab[:, :, 0] = enhanced_l
    result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    parameters = {
        "algorithm": "LAB_LUMINANCE_GAMMA_PLUS_CLAHE",
        "geometry_operation": "NONE",
        "chroma_channels_changed": False,
        "gamma": round(gamma, 6),
        "clahe_clip_limit": round(clip_limit, 6),
        "clahe_tile_grid": [8, 8],
        "clahe_blend": round(blend, 6),
        "jpeg_quality": JPEG_QUALITY,
        "input_width": int(image.shape[1]),
        "input_height": int(image.shape[0]),
        "output_width": int(result.shape[1]),
        "output_height": int(result.shape[0]),
    }
    return result, parameters


def _load_verified(path: Path, expected_sha256: str) -> tuple[bytes, np.ndarray]:
    data = path.read_bytes()
    if _sha256(data) != expected_sha256:
        raise ValueError(f"visual source hash mismatch: {path}")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3:
        raise ValueError(f"visual source could not be decoded: {path}")
    return data, image


def build_visual_modes(
    document: dict[str, Any], asset_root: Path, output_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Attach hash-bound ORIGINAL/ENHANCED/AUTO choices to every frame."""

    asset_root = asset_root.resolve()
    output_root = output_root.resolve()
    updated = copy.deepcopy(document)
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    manifest: list[dict[str, Any]] = []
    for case in updated["cases"]:
        for frame in case["frames"]:
            frame_rows: dict[str, dict[str, Any]] = {}
            panorama_metrics: dict[str, Any] | None = None
            for kind in ("panorama", "focus"):
                url = str(frame[f"{kind}_url"])
                expected = str(frame[f"{kind}_sha256"])
                if not url.startswith("/assets/"):
                    raise ValueError(f"unsupported original visual URL: {url}")
                relative = Path(url.removeprefix("/assets/"))
                source = (asset_root / relative).resolve()
                if asset_root not in source.parents or not source.is_file():
                    raise ValueError(f"visual source escapes the approved asset root: {url}")
                key = (kind, expected)
                if key not in cache:
                    original_bytes, image = _load_verified(source, expected)
                    metrics = _metrics(image)
                    enhanced, parameters = enhance_review_image(image, metrics)
                    ok, encoded = cv2.imencode(".jpg", enhanced, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    if not ok:
                        raise ValueError(f"enhanced visual could not be encoded: {url}")
                    enhanced_bytes = encoded.tobytes()
                    enhanced_sha = _sha256(enhanced_bytes)
                    destination = output_root / "enhanced" / kind / f"{enhanced_sha}.jpg"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists() and destination.read_bytes() != enhanced_bytes:
                        raise ValueError("hash-addressed enhanced derivative collision")
                    if not destination.exists():
                        destination.write_bytes(enhanced_bytes)
                    if image.shape[:2] != enhanced.shape[:2]:
                        raise ValueError("review enhancement changed image geometry")
                    cache[key] = {
                        "kind": kind,
                        "original_url": url,
                        "original_sha256": expected,
                        "original_byte_size": len(original_bytes),
                        "enhanced_url": f"/review-assets/enhanced/{kind}/{enhanced_sha}.jpg",
                        "enhanced_sha256": enhanced_sha,
                        "enhanced_byte_size": len(enhanced_bytes),
                        "decoded_width": int(image.shape[1]),
                        "decoded_height": int(image.shape[0]),
                        "metrics": metrics,
                        "parameters": parameters,
                    }
                    manifest.append(cache[key])
                row = cache[key]
                frame_rows[kind] = row
                if kind == "panorama":
                    panorama_metrics = row["metrics"]
            auto_mode = "ENHANCED" if panorama_metrics and panorama_metrics["low_light"] else "ORIGINAL"
            frame["visual_modes"] = {
                "ORIGINAL": {
                    "panorama_url": frame_rows["panorama"]["original_url"],
                    "panorama_sha256": frame_rows["panorama"]["original_sha256"],
                    "focus_url": frame_rows["focus"]["original_url"],
                    "focus_sha256": frame_rows["focus"]["original_sha256"],
                },
                "ENHANCED": {
                    "panorama_url": frame_rows["panorama"]["enhanced_url"],
                    "panorama_sha256": frame_rows["panorama"]["enhanced_sha256"],
                    "focus_url": frame_rows["focus"]["enhanced_url"],
                    "focus_sha256": frame_rows["focus"]["enhanced_sha256"],
                },
            }
            frame["auto_visual_mode"] = auto_mode
            frame["visual_contract"] = VISUAL_CONTRACT
    return updated, sorted(manifest, key=lambda row: (row["kind"], row["original_sha256"]))


def write_visual_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    document = {
        "schema_version": "football_intelligence.g7e_b_r6_1.visual_asset_manifest.v1",
        "visual_contract": VISUAL_CONTRACT,
        "asset_count": len(rows),
        "source_truth_changed": False,
        "geometry_changed": False,
        "assets": rows,
        "production_ready": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
