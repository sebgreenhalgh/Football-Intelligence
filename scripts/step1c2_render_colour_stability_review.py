from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_stability_render import (  # noqa: E402
    render_all_colour_stability_review_sheets,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    outputs = render_all_colour_stability_review_sheets()
    for key, value in outputs.items():
        print(f"{key}: {value}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")


if __name__ == "__main__":
    main()
