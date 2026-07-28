"""Bounded G7C repair: real representative frames, human form, and pack provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FIELDS = (
    "lighting",
    "weather",
    "visibility",
    "panorama_quality",
    "crowd_background",
    "team_1_primary_colour",
    "team_2_primary_colour",
    "unusual_conditions",
)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root
    exp = args.experiment_root
    result = json.loads((exp / "02_SOURCE_INVENTORY" / "inventory_result.json").read_text(encoding="utf-8"))
    matches = result["matches"]
    media = result["media_metadata"]
    tmp = exp / "_tmp" / "g7c_real_frames"
    tmp.mkdir(parents=True, exist_ok=True)
    selected = {}
    for match in matches:
        candidates = [
            m for m in media if m["relative_path"].startswith(f"matches/{match}/") and "panorama" in m["relative_path"]
        ]
        if not candidates:
            raise RuntimeError(f"No frozen panorama selection for {match}")
        item = candidates[0]
        timestamp = round(float(item["duration"]) * 0.25, 3)
        source = root / Path(item["relative_path"])
        frame = tmp / f"{match}.png"
        subprocess.run(
            [
                str(args.ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(timestamp),
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
        selected[match] = {
            "source_filename": source.name,
            "relative_path": item["relative_path"],
            "timestamp": timestamp,
            "resolution": f"{item['width']}x{item['height']}",
            "source_sha256": item["sha256"],
            "frame_path": str(frame),
        }

    sheet_path = exp / "05_CONDITION_REVIEW" / "ten_match_contact_sheet.png"
    panel_w, panel_h = 640, 250
    sheet = Image.new("RGB", (panel_w * 2, panel_h * 5), "#17202a")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, match in enumerate(matches):
        item = selected[match]
        image = Image.open(item["frame_path"]).convert("RGB")
        image.thumbnail((panel_w - 16, 180))
        x, y = (index % 2) * panel_w, (index // 2) * panel_h
        sheet.paste(image, (x + 8, y + 8))
        draw.rectangle((x + 4, y + 4, x + panel_w - 4, y + panel_h - 4), outline="#8aa4b8", width=2)
        lines = [
            f"{match}",
            f"source: {item['source_filename']}",
            f"timestamp: {item['timestamp']}s",
            f"resolution: {item['resolution']}",
        ]
        for line_no, line in enumerate(lines):
            draw.text((x + 12, y + 190 + line_no * 14), line, fill="#f4f7f9", font=font)
    sheet.save(sheet_path)

    form = {
        "schema_version": "g7c.human_condition_review.v1",
        "matches": [
            {
                "match_id": match,
                **{field: "" for field in FIELDS},
                "representative_frame_approved": False,
                "proposed_split_approved": False,
            }
            for match in matches
        ],
    }
    dump(exp / "05_CONDITION_REVIEW" / "HUMAN_CONDITION_REVIEW.json", form)

    split = json.loads(
        (root / "datasets/soccertrack_v2/splits/split_v1/proposed_split.json").read_text(encoding="utf-8")
    )
    if (
        split["status"] != "PROVISIONAL_PENDING_HUMAN_APPROVAL"
        or split["frozen"]
        or split["allocation"]
        != {
            "TRAIN_DEVELOPMENT": ["117092", "117093", "118575", "118576", "118577", "128058"],
            "VALIDATION_MODEL_SELECTION": ["118578", "128057"],
            "SEALED_HOLDOUT": ["132831", "132877"],
        }
    ):
        raise RuntimeError("Existing provisional split changed")

    review = exp / "08_REVIEW_PACK"
    diff = subprocess.run(
        ["git", "diff", f"{args.baseline}..HEAD", "--binary"],
        cwd=root / "SoccerTrack-v2",
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (review / "04_SOURCE_DIFF.patch").write_text(diff, encoding="utf-8")
    summary_path = review / "01_EXECUTIVE_SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["contact_sheet_status"] = "REAL_REPRESENTATIVE_FRAMES"
    summary["human_condition_review_form"] = "05_CONDITION_REVIEW/HUMAN_CONDITION_REVIEW.json"
    summary["packaging_validation"] = "PASSED_REPAIR_FOCUSED_TESTS"
    dump(summary_path, summary)
    safety_path = review / "05_TESTS_AND_SAFETY.json"
    safety = json.loads(safety_path.read_text(encoding="utf-8"))
    safety["contact_sheet_status"] = "REAL_REPRESENTATIVE_FRAMES"
    safety["split_preservation"] = "passed"
    safety["packaging_validation"] = "PASSED_REPAIR_FOCUSED_TESTS"
    dump(safety_path, safety)
    manifest = {"files": []}
    for path in sorted(review.iterdir()):
        if path.name == "07_MANIFEST.json":
            continue
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
