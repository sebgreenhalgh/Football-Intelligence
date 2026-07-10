from __future__ import annotations

from stage1_bootstrap import bootstrap

bootstrap()

from football_intelligence.step1_visual_reconstruction.colour_features import build_and_write_colour_features  # noqa: E402
from football_intelligence.step1_visual_reconstruction.io import STEP1C1_COLOUR_FEATURE_ROWS_PATH  # noqa: E402
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    payload = build_and_write_colour_features()
    print(f"step1c1_colour_feature_rows_path: {STEP1C1_COLOUR_FEATURE_ROWS_PATH.resolve()}")
    print(f"step1c1_colour_feature_rows: {payload['summary']['step1c1_colour_feature_rows']}")
    print(f"crop_quality_counts: {payload['summary']['crop_quality_counts']}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")


if __name__ == "__main__":
    main()
