from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1D1_GOLD8_OFFICIAL_CONTEXT_EVAL_REPORT_PATH,
    STEP1D1_GOLD8_OFFICIAL_CONTEXT_EVAL_SUMMARY_PATH,
)
from football_intelligence.step1_visual_reconstruction.official_context_eval import (  # noqa: E402
    build_and_write_official_context_eval,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    summary = build_and_write_official_context_eval()
    print(
        "step1d1_gold8_official_context_eval_summary_path: "
        f"{STEP1D1_GOLD8_OFFICIAL_CONTEXT_EVAL_SUMMARY_PATH.resolve()}"
    )
    print(
        "step1d1_gold8_official_context_eval_report_path: "
        f"{STEP1D1_GOLD8_OFFICIAL_CONTEXT_EVAL_REPORT_PATH.resolve()}"
    )
    print(f"gold8_official_proxy_rows: {summary.get('gold8_official_proxy_rows', 0)}")
    print(f"gold8_official_proxy_matched_rows: {summary.get('gold8_official_proxy_matched_rows', 0)}")
    print(f"d1_safe_for_human_review_candidate={str(summary.get('d1_safe_for_human_review_candidate', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")


if __name__ == "__main__":
    main()
