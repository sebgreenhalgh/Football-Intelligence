from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_seed_candidates import (  # noqa: E402
    build_and_write_colour_seed_candidates,
)
from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1C1C_COLOUR_SEED_CANDIDATE_ROWS_PATH,
    STEP1C1C_COLOUR_SEED_CANDIDATE_SUMMARY_PATH,
    STEP1C1C_MANUAL_COLOUR_SEED_LABEL_TEMPLATE_CSV_PATH,
    STEP1C1C_MANUAL_COLOUR_SEED_LABEL_TEMPLATE_JSON_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    candidate_payload, summary_payload, _template_payload = build_and_write_colour_seed_candidates()
    print(f"step1c1c_colour_seed_candidate_rows_path: {STEP1C1C_COLOUR_SEED_CANDIDATE_ROWS_PATH.resolve()}")
    print(f"step1c1c_colour_seed_candidate_summary_path: {STEP1C1C_COLOUR_SEED_CANDIDATE_SUMMARY_PATH.resolve()}")
    print(
        "step1c1c_manual_colour_seed_label_template_json_path: "
        f"{STEP1C1C_MANUAL_COLOUR_SEED_LABEL_TEMPLATE_JSON_PATH.resolve()}"
    )
    print(
        "step1c1c_manual_colour_seed_label_template_csv_path: "
        f"{STEP1C1C_MANUAL_COLOUR_SEED_LABEL_TEMPLATE_CSV_PATH.resolve()}"
    )
    print(f"step1c1c_colour_seed_candidate_rows: {len(candidate_payload.get('rows', []))}")
    category_counts = summary_payload.get("summary", {}).get("seed_candidate_category_counts", {})
    print(f"seed_candidate_category_counts: {category_counts}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")


if __name__ == "__main__":
    main()
