from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_human_correction_eval import (  # noqa: E402
    build_and_write_f3_eval,
)
from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_EVAL_SUMMARY_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    summary = build_and_write_f3_eval()
    print(f"step1f3_eval_summary_path: {STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_EVAL_SUMMARY_PATH.resolve()}")
    print(f"f1_row_count: {summary.get('f1_row_count', 0)}")
    print(f"f3_row_count: {summary.get('f3_row_count', 0)}")
    print(f"f2_reviewed_decision_count: {summary.get('f2_reviewed_decision_count', 0)}")
    print(f"f1_missed_goalkeeper_proxy_count: {summary.get('f1_missed_goalkeeper_proxy_count', 0)}")
    print(f"f3_missed_goalkeeper_proxy_count: {summary.get('f3_missed_goalkeeper_proxy_count', 0)}")
    safe = str(summary.get("f3_safe_for_step1g_validation_candidate", False)).lower()
    print(f"f3_safe_for_step1g_validation_candidate={safe}")
    print(f"missing_reasons: {summary.get('f3_safety_missing_reasons', [])}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")


if __name__ == "__main__":
    main()
