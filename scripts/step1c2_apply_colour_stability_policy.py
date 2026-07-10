from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_stability_policy import (  # noqa: E402
    build_and_write_colour_stability_policy,
)
from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1C2_COLOUR_FLIP_AUDIT_ROWS_PATH,
    STEP1C2_COLOUR_STABILITY_ROWS_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    stability_payload, flip_payload = build_and_write_colour_stability_policy()
    print(f"step1c2_colour_stability_rows_path: {STEP1C2_COLOUR_STABILITY_ROWS_PATH.resolve()}")
    print(f"step1c2_colour_flip_audit_rows_path: {STEP1C2_COLOUR_FLIP_AUDIT_ROWS_PATH.resolve()}")
    print(f"c2_stability_row_count: {stability_payload.get('summary', {}).get('c2_stability_row_count', 0)}")
    print(f"flip_audit_row_count: {flip_payload.get('summary', {}).get('flip_audit_row_count', 0)}")
    context_forced = stability_payload.get("summary", {}).get("context_offroi_forced_to_team_count", 0)
    print(f"context_offroi_forced_to_team_count: {context_forced}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")


if __name__ == "__main__":
    main()
