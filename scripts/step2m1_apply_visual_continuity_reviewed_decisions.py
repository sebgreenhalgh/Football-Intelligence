# ruff: noqa: E501

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.human_corrections import (  # noqa: E402
    apply_and_write_reviewed_decisions,
)
from football_intelligence.step2_visual_continuity.io import (  # noqa: E402
    STEP2M1_CORRECTION_AUDIT_ROWS_PATH,
    STEP2M1_GROUP_ROWS_SANDBOX_PATH,
    STEP2M1_HUMAN_CORRECTED_EDGE_ROWS_JSONL_GZ_PATH,
    STEP2M1_HUMAN_CORRECTED_EDGE_ROWS_PATH,
    STEP2M1_HUMAN_CORRECTED_EDGE_SAMPLE_PATH,
    STEP2M1_HUMAN_CORRECTED_EDGE_SUMMARY_PATH,
    STEP2M1_TRAINING_EXAMPLES_PATH,
)
from football_intelligence.step2_visual_continuity.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    corrected, audit, training_rows, group_payload, validation_outputs = apply_and_write_reviewed_decisions()
    validation_summary = validation_outputs.get("validation_summary", {})
    print(f"step2m1_human_corrected_visual_continuity_edge_rows_manifest_path: {STEP2M1_HUMAN_CORRECTED_EDGE_ROWS_PATH.resolve()}")
    print(f"step2m1_human_corrected_visual_continuity_edge_rows_jsonl_gz_path: {STEP2M1_HUMAN_CORRECTED_EDGE_ROWS_JSONL_GZ_PATH.resolve()}")
    print(f"step2m1_human_corrected_visual_continuity_edge_summary_path: {STEP2M1_HUMAN_CORRECTED_EDGE_SUMMARY_PATH.resolve()}")
    print(f"step2m1_human_corrected_visual_continuity_edge_sample_path: {STEP2M1_HUMAN_CORRECTED_EDGE_SAMPLE_PATH.resolve()}")
    print(f"step2m1_visual_continuity_correction_audit_rows_path: {STEP2M1_CORRECTION_AUDIT_ROWS_PATH.resolve()}")
    print(f"step2m1_visual_continuity_training_examples_path: {STEP2M1_TRAINING_EXAMPLES_PATH.resolve()}")
    print(f"step2m1_visual_continuity_group_rows_sandbox_path: {STEP2M1_GROUP_ROWS_SANDBOX_PATH.resolve()}")
    print(f"corrected_edge_rows: {len(corrected.get('rows', []))}")
    print(f"correction_audit_rows: {len(audit.get('rows', []))}")
    print(f"training_examples: {len(training_rows)}")
    print(f"visual_continuity_group_rows: {len(group_payload.get('rows', []))}")
    print(f"post_review_validation_refreshed={str(validation_summary.get('post_review_validation_refreshed', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")


if __name__ == "__main__":
    main()
