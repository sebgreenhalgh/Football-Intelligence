from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_crop_audit import build_and_write_crop_audit  # noqa: E402
from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1C1B_CROP_AUDIT_ROWS_PATH,
    STEP1C1B_CROP_AUDIT_SUMMARY_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    audit_payload, summary_payload = build_and_write_crop_audit()
    print(f"step1c1b_crop_audit_rows_path: {STEP1C1B_CROP_AUDIT_ROWS_PATH.resolve()}")
    print(f"step1c1b_crop_audit_summary_path: {STEP1C1B_CROP_AUDIT_SUMMARY_PATH.resolve()}")
    print(f"step1c1b_crop_audit_rows: {len(audit_payload.get('rows', []))}")
    print(f"audit_issue_flag_counts: {summary_payload.get('audit_issue_flag_counts', {})}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")


if __name__ == "__main__":
    main()
