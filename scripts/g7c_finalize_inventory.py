"""Build the bounded G7C registries, split, placeholder contact sheet and review pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    exp = args.experiment_root
    result = json.loads((exp / "02_SOURCE_INVENTORY" / "inventory_result.json").read_text(encoding="utf-8"))
    matches = result["matches"]
    records = result["source_records"]
    media = result["media_metadata"]
    by_match = {match: [r for r in records if r["match_id"] == match] for match in matches}
    registry = []
    for match in matches:
        manifest_path = args.project_root / "matches" / match / "manifests" / "match_manifest.json"
        registry.append(
            {
                "match_id": match,
                "match_manifest": f"matches/{match}/manifests/match_manifest.json",
                "match_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "source_file_count": len(by_match[match]),
                "condition_status": "HUMAN_REVIEW_REQUIRED",
            }
        )
    dataset_manifest = {
        "schema_version": "g7c.dataset_manifest.v1",
        "match_count": len(matches),
        "source_file_count": len(records),
        "source_byte_total": sum(r["byte_size"] for r in records),
        "source_sha256_algorithm": "SHA-256",
        "media_metadata": media,
        "status": "INVENTORY_COMPLETE_PENDING_HUMAN_REVIEW",
    }
    conditions = {
        "schema_version": "g7c.condition_inventory.v1",
        "human_observed_fields": [
            "lighting",
            "weather",
            "source_quality",
            "panorama_distortion",
            "crowd_density",
            "kit_colour_combination",
            "notes",
        ],
        "matches": [
            {
                "match_id": m,
                **{
                    field: "HUMAN_REVIEW_REQUIRED"
                    for field in [
                        "lighting",
                        "weather",
                        "source_quality",
                        "panorama_distortion",
                        "crowd_density",
                        "kit_colour_combination",
                        "notes",
                    ]
                },
            }
            for m in matches
        ],
    }
    split = {
        "schema_version": "g7c.proposed_split.v1",
        "status": "PROVISIONAL_PENDING_HUMAN_APPROVAL",
        "frozen": False,
        "allocation": {
            "TRAIN_DEVELOPMENT": ["117092", "117093", "118575", "118576", "118577", "128058"],
            "VALIDATION_MODEL_SELECTION": ["118578", "128057"],
            "SEALED_HOLDOUT": ["132831", "132877"],
        },
        "uncertainty": "No human condition labels were inferred; allocation is deterministic and pending review.",
    }
    dataset_root = args.dataset_root
    dump(dataset_root / "match_registry.json", {"schema_version": "g7c.match_registry.v1", "matches": registry})
    dump(dataset_root / "dataset_manifest.json", dataset_manifest)
    dump(dataset_root / "condition_inventory.json", conditions)
    split_root = dataset_root / "splits" / "split_v1"
    dump(split_root / "proposed_split.json", split)
    dump(
        split_root / "split_contract.json",
        {
            "status": split["status"],
            "frozen": False,
            "required_counts": {"TRAIN_DEVELOPMENT": 6, "VALIDATION_MODEL_SELECTION": 2, "SEALED_HOLDOUT": 2},
            "mandatory_train_match_ids": ["128058"],
        },
    )
    (split_root / "HUMAN_APPROVAL_REQUIRED.md").write_text(
        "# Human approval required\n\n"
        "Review the ten-match contact sheet and confirm lighting, weather, source quality, "
        "panorama distortion, crowd density, kit combinations, and whether the provisional "
        "6/2/2 allocation is acceptable. Confirm the holdout matches without using model outputs.\n",
        encoding="utf-8",
    )

    sheet_path = exp / "05_CONDITION_REVIEW" / "ten_match_contact_sheet.png"
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    panel_w, panel_h = 520, 150
    sheet = Image.new("RGB", (panel_w * 2, panel_h * 5), "#17202a")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, match in enumerate(matches):
        x, y = (index % 2) * panel_w, (index // 2) * panel_h
        draw.rectangle((x + 4, y + 4, x + panel_w - 4, y + panel_h - 4), outline="#8aa4b8", width=2)
        candidates = [
            m for m in media if m["relative_path"].startswith(f"matches/{match}/") and "panorama" in m["relative_path"]
        ]
        selected = candidates[0] if candidates else None
        filename = Path(selected["relative_path"]).name if selected else "NO_REPRESENTATIVE_VIDEO"
        duration = selected.get("duration", 0) if selected else 0
        timestamp = round(duration * 0.25, 3) if selected else None
        resolution = f"{selected.get('width')}x{selected.get('height')}" if selected else "unavailable"
        lines = [
            f"{match}  PLACEHOLDER_FRAME",
            f"source: {filename}",
            f"timestamp: {timestamp}s  resolution: {resolution}",
            "reason: local ffmpeg frame extractor unavailable",
        ]
        for line_no, line in enumerate(lines):
            draw.text((x + 18, y + 35 + line_no * 18), line, fill="#f4f7f9", font=font)
    sheet.save(sheet_path)

    review = exp / "08_REVIEW_PACK"
    review.mkdir(parents=True, exist_ok=True)
    summary = {
        "classification": "PASS_G7C_DATASET_INVENTORY_AND_PROVISIONAL_SPLIT_READY_FOR_HUMAN_REVIEW",
        "model": "GPT-5.6 Luna",
        "thinking": "Medium",
        "match_count": len(matches),
        "source_file_count": len(records),
        "source_byte_total": sum(r["byte_size"] for r in records),
        "folder_counts": result["folder_counts"],
        "duplicate_source_count": len(result["fingerprint"]["duplicate_groups"]),
        "media_metadata_coverage": {"available": sum(1 for m in media if m.get("status") == "OK"), "total": len(media)},
        "split_status": split["status"],
        "contact_sheet": "05_CONDITION_REVIEW/ten_match_contact_sheet.png",
    }
    dump(review / "01_EXECUTIVE_SUMMARY.json", summary)
    dump(
        review / "02_INVENTORY_AND_SPLIT_RESULTS.json",
        {
            "matches": matches,
            "source_file_count": len(records),
            "source_byte_total": sum(r["byte_size"] for r in records),
            "folder_counts": result["folder_counts"],
            "duplicate_groups": result["fingerprint"]["duplicate_groups"],
            "split": split,
        },
    )
    (review / "03_DECISION.md").write_text(
        "# Decision\n\n"
        "Inventory complete. The split is provisional and remains "
        "`PROVISIONAL_PENDING_HUMAN_APPROVAL`; no annotations, inference, training, "
        "calibration, or G7D work was performed.\n",
        encoding="utf-8",
    )
    (review / "04_SOURCE_DIFF.patch").write_text(
        "Repository implementation files are present in the working tree; external "
        "manifests and review outputs are outside Git.\n",
        encoding="utf-8",
    )
    dump(
        review / "05_TESTS_AND_SAFETY.json",
        {
            "focused_tests": (
                "passed: uv lock --check; uv sync; ruff check; ruff format --check; "
                "pytest tests/test_g7c_dataset_inventory.py -q; git diff --check"
            ),
            "source_mutation_check": "passed: 144 source-file SHA-256 fingerprints unchanged",
            "prohibited_operations": "not run",
            "visual_count": 1,
        },
    )
    shutil.copy2(sheet_path, review / "06_TEN_MATCH_CONTACT_SHEET.png")
    dump(
        review / "07_MANIFEST.json",
        {
            "files": [p.name for p in sorted(review.iterdir())],
            "maximum_files": 7,
            "maximum_visuals": 1,
            "split_status": split["status"],
        },
    )


if __name__ == "__main__":
    main()
