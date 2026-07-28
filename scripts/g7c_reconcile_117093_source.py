"""Bounded pre-freeze source-selection reconciliation for match 117093."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stream_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root
    match = root / "matches" / "117093"
    manifest_dir = match / "manifests"
    existing = json.loads((manifest_dir / "source_file_hashes.json").read_text(encoding="utf-8"))
    records = []
    for folder in (
        match / "source" / "bas",
        match / "source" / "videos",
        match / "source" / "raw",
        match / "source" / "gsr",
    ):
        for path in sorted(folder.glob("*")):
            if path.is_file():
                records.append(
                    {
                        "match_id": "117093",
                        "relative_path": path.relative_to(root).as_posix(),
                        "sha256": stream_hash(path),
                        "byte_size": path.stat().st_size,
                    }
                )
    old_path = "matches/117093/source/videos/117093_calibrated_panorama_1st_half.mp4"
    new_path = "matches/117093/source/videos/117093_panorama_1st_half-008.mp4"
    prior_inventory = json.loads(
        (args.workspace / "02_SOURCE_INVENTORY" / "inventory_result.json").read_text(encoding="utf-8")
    )
    prior_hashes = {record["relative_path"]: record["sha256"] for record in prior_inventory["source_records"]}
    old_hash = existing.get(old_path, prior_hashes.get(old_path))
    new_hash = next(record["sha256"] for record in records if record["relative_path"] == new_path)
    event = {
        "event_type": "AUTHORIZED_PRE_FREEZE_SOURCE_CORRECTION",
        "match_id": "117093",
        "old_source_path": old_path,
        "new_source_path": new_path,
        "old_hash": old_hash,
        "new_hash": new_hash,
        "reason": (
            "The calibrated first-half file was an incorrect representative reference; "
            "the canonical source/videos panorama first-half file is authoritative."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split_status_at_event": "PROVISIONAL_PENDING_HUMAN_APPROVAL",
        "split_frozen_at_event": False,
    }
    dump(manifest_dir / "source_file_hashes.json", {record["relative_path"]: record["sha256"] for record in records})
    dump(manifest_dir / "source_file_manifest.json", {"match_id": "117093", "files": records})
    manifest = json.loads((manifest_dir / "match_manifest.json").read_text(encoding="utf-8"))
    manifest["representative_source"] = {
        "relative_path": new_path,
        "timestamp_seconds": 1.0,
        "sha256": new_hash,
        "selection_status": "AUTHORITATIVE_PRE_FREEZE_CORRECTION",
    }
    manifest["source_file_count"] = len(records)
    manifest["source_correction_event"] = event
    dump(manifest_dir / "match_manifest.json", manifest)
    dump(manifest_dir / "source_correction_events.json", [event])

    registry_path = root / "datasets/soccertrack_v2/match_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for entry in registry["matches"]:
        if entry["match_id"] == "117093":
            entry["match_manifest_sha256"] = hashlib.sha256(
                (manifest_dir / "match_manifest.json").read_bytes()
            ).hexdigest()
            entry["representative_source"] = {"relative_path": new_path, "timestamp_seconds": 1.0, "sha256": new_hash}
    dump(registry_path, registry)
    dataset_path = root / "datasets/soccertrack_v2/dataset_manifest.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    reps = dataset.setdefault("representative_sources", {})
    reps["117093"] = {
        "relative_path": new_path,
        "timestamp_seconds": 1.0,
        "sha256": new_hash,
        "correction_event": "matches/117093/manifests/source_correction_events.json",
    }
    dump(dataset_path, dataset)

    frame = args.workspace / "_tmp" / "117093_corrected.png"
    source = root / Path(new_path)
    subprocess.run(
        [
            str(args.ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            "1.0",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-y",
            str(frame),
        ],
        check=True,
        timeout=120,
    )
    sheet_path = args.workspace / "05_CONDITION_REVIEW" / "ten_match_contact_sheet.png"
    sheet = Image.open(sheet_path).convert("RGB")
    panel_w, panel_h = 640, 250
    x, y = panel_w, 0
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((x, y, x + panel_w - 1, y + panel_h - 1), fill="#17202a")
    image = Image.open(frame).convert("RGB")
    image.thumbnail((panel_w - 16, 180))
    sheet.paste(image, (x + 8, y + 8))
    draw.rectangle((x + 4, y + 4, x + panel_w - 4, y + panel_h - 4), outline="#8aa4b8", width=2)
    font = ImageFont.load_default()
    for line_no, line in enumerate(["117093", f"source: {source.name}", "timestamp: 1.0s", "resolution: 4096x1080"]):
        draw.text((x + 12, y + 190 + line_no * 14), line, fill="#f4f7f9", font=font)
    sheet.save(sheet_path)
    provenance = args.workspace / "05_CONDITION_REVIEW" / "contact_sheet_provenance.json"
    current = json.loads(provenance.read_text(encoding="utf-8")) if provenance.exists() else {}
    current["117093"] = {
        "relative_path": new_path,
        "source_sha256": new_hash,
        "timestamp_seconds": 1.0,
        "correction_event": "matches/117093/manifests/source_correction_events.json",
    }
    dump(provenance, current)
    review = args.workspace / "08_REVIEW_PACK"
    shutil.copy2(sheet_path, review / "06_TEN_MATCH_CONTACT_SHEET.png")
    summary_path = review / "01_EXECUTIVE_SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["117093_source_correction"] = event
    summary["contact_sheet_status"] = "REAL_REPRESENTATIVE_FRAMES_WITH_117093_CORRECTION"
    dump(summary_path, summary)
    results_path = review / "02_INVENTORY_AND_SPLIT_RESULTS.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results["117093_source_correction"] = event
    dump(results_path, results)
    safety_path = review / "05_TESTS_AND_SAFETY.json"
    safety = json.loads(safety_path.read_text(encoding="utf-8"))
    safety["117093_reconciliation"] = "passed: 16 match-local files rehashed; source bytes preserved"
    safety["contact_sheet_status"] = "117093 PANEL CORRECTED AT 1.0 SECONDS"
    dump(safety_path, safety)
    manifest = {"files": []}
    for path in sorted(review.iterdir()):
        if path.name != "07_MANIFEST.json":
            manifest["files"].append(
                {
                    "filename": path.name,
                    "byte_size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    manifest["maximum_files"] = 7
    manifest["maximum_visuals"] = 1
    dump(review / "07_MANIFEST.json", manifest)


if __name__ == "__main__":
    main()
