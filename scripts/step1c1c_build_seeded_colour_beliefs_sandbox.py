from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1C1C_SEEDED_COLOUR_BELIEF_ROWS_SANDBOX_PATH,
    STEP1C1C_SEEDED_COLOUR_PROTOTYPES_SANDBOX_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402
from football_intelligence.step1_visual_reconstruction.seeded_colour_prototypes import (  # noqa: E402
    build_seeded_colour_sandbox_payloads,
)


def main() -> None:
    validation_summary, prototypes, beliefs = build_seeded_colour_sandbox_payloads()
    print(f"step1c1c_seeded_colour_prototypes_sandbox_path: {STEP1C1C_SEEDED_COLOUR_PROTOTYPES_SANDBOX_PATH.resolve()}")
    print(
        "step1c1c_seeded_colour_belief_rows_sandbox_path: "
        f"{STEP1C1C_SEEDED_COLOUR_BELIEF_ROWS_SANDBOX_PATH.resolve()}"
    )
    print(f"reviewed_seed_labels_loaded={str(validation_summary.get('reviewed_seed_labels_loaded', False)).lower()}")
    print(f"reviewed_seed_labels_valid={str(validation_summary.get('reviewed_seed_labels_valid', False)).lower()}")
    print(f"prototype_count: {prototypes.get('summary', {}).get('prototype_count', 0)}")
    print(f"seeded_colour_belief_rows: {len(beliefs.get('rows', []))}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")


if __name__ == "__main__":
    main()
