from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.io import STEP1C1C_REVIEWED_COLOUR_SEED_LABELS_PATH  # noqa: E402
from football_intelligence.step1_visual_reconstruction.manual_seed_review_export import (  # noqa: E402
    export_existing_reviewed_seed_labels,
)


def main() -> None:
    payload = export_existing_reviewed_seed_labels()
    summary = payload.get("summary", {})
    print(f"reviewed labels path: {STEP1C1C_REVIEWED_COLOUR_SEED_LABELS_PATH.resolve()}")
    print(f"number reviewed: {summary.get('reviewed_rows', 0)}")
    print(f"team 1 seed count: {summary.get('human_confirmed_team_1_seed_count', 0)}")
    print(f"team 2 seed count: {summary.get('human_confirmed_team_2_seed_count', 0)}")
    print(f"negative/context/dark/ambiguous count: {summary.get('human_confirmed_negative_seed_count', 0)}")
    print(f"minimum seed counts satisfied: {str(summary.get('minimum_seed_counts_satisfied', False)).lower()}")
    print("rerun C1c validation / seeded prototype evaluation:")
    print("  .\\.venv\\Scripts\\python.exe scripts\\step1c1c_validate_reviewed_colour_seeds.py")
    print("  .\\.venv\\Scripts\\python.exe scripts\\step1c1c_build_seeded_colour_beliefs_sandbox.py")
    print("  .\\.venv\\Scripts\\python.exe scripts\\step1c1c_evaluate_seeded_colour_beliefs.py")


if __name__ == "__main__":
    main()
