"""Run live Edge acceptance against temporary adjudication decisions only."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from football_intelligence.calibration_adjudication import CALIBRATION_IDS, sha256_file
from football_intelligence.calibration_adjudication_reviewer import (
    CalibrationAdjudicationHTTPServer,
    CalibrationAdjudicationReviewerConfig,
)
from g7f_c_run_calibration_adjudication_reviewer import EXPECTED_HASHES, REVIEWER_RELEASE, roots


def inventory(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    rows = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    payload = "\n".join(f"{row['relative_path']}\t{row['byte_size']}\t{row['sha256']}" for row in rows).encode()
    return {
        "file_count": len(rows),
        "byte_size": sum(row["byte_size"] for row in rows),
        "ordered_inventory_sha256": hashlib.sha256(payload).hexdigest(),
        "files": rows,
    }


def run(repo: Path, output: Path) -> dict[str, Any]:
    paths = roots(repo)
    original_root = paths["source"] / "06_DENSE_DECISIONS"
    before = inventory(original_root)
    frozen = {}
    for image_id in CALIBRATION_IDS:
        frozen[image_id] = {
            "event_file_sha256": sha256_file(original_root / "events" / f"first_pass__{image_id}.json"),
            "acknowledgement_file_sha256": sha256_file(
                original_root / "acknowledgements" / f"first_pass__{image_id}.json"
            ),
        }
    audit_path = paths["audit"] / "03_MANUAL_VISUAL_ASSESSMENT.json"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="g7f_c_adjudication_", dir=output) as temp_name:
        temp_root = Path(temp_name)
        server = CalibrationAdjudicationHTTPServer(
            CalibrationAdjudicationReviewerConfig(
                selection_manifest_path=paths["source"] / "01_SELECTION/dense_gold_selection_manifest.json",
                assets_root=paths["source"] / "05_REVIEWER/assets",
                original_decisions_root=original_root,
                adjudication_root=temp_root / "decisions/calibration_adjudication",
                binding_hashes={key.replace("/", "_"): value for key, value in EXPECTED_HASHES.items()},
                reviewer_release=REVIEWER_RELEASE,
                audit_assessment_path=audit_path,
                audit_checklist_sha256=sha256_file(audit_path),
                frozen_parent_file_hashes=frozen,
                port=0,
            )
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        screenshot = output / "live_edge_adjudication_acceptance.png"
        try:
            result = subprocess.run(
                [
                    "node",
                    str(repo / "scripts/g7f_c_calibration_adjudication_edge_acceptance.js"),
                    f"http://127.0.0.1:{server.server_address[1]}/",
                    str(screenshot),
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        if result.returncode:
            raise RuntimeError(f"live Edge adjudication acceptance failed:\n{result.stdout}\n{result.stderr}")
        browser = json.loads(result.stdout)
        temp_events = sorted((temp_root / "decisions/calibration_adjudication/events").glob("*.json"))
        temp_acks = sorted((temp_root / "decisions/calibration_adjudication/acknowledgements").glob("*.json"))
        if (len(temp_events), len(temp_acks)) != (1, 1):
            raise RuntimeError("live Edge did not produce one exact temporary event/ack pair")
    after = inventory(original_root)
    if before != after:
        raise RuntimeError("live Edge acceptance changed real decisions")
    payload = {
        "browser": browser,
        "real_decisions_before": {key: before[key] for key in ("file_count", "byte_size", "ordered_inventory_sha256")},
        "real_decisions_after": {key: after[key] for key in ("file_count", "byte_size", "ordered_inventory_sha256")},
        "real_decisions_byte_identical": True,
        "temporary_event_ack_pairs": 1,
        "scored_annotation_authorized": False,
        "production_ready": False,
    }
    (output / "live_edge_acceptance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.repo.resolve(), args.output.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
