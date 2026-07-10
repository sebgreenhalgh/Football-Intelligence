from __future__ import annotations

from stage1_bootstrap import bootstrap

bootstrap()

from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1C1_COLOUR_PROTOTYPES_PATH,
    STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH,
    STEP1C1_UNKNOWN_AMBIGUOUS_COLOUR_ROWS_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402
from football_intelligence.step1_visual_reconstruction.team_colour_beliefs import build_and_write_team_colour_beliefs  # noqa: E402


def main() -> None:
    prototypes, beliefs, unknown = build_and_write_team_colour_beliefs()
    print(f"step1c1_colour_prototypes_path: {STEP1C1_COLOUR_PROTOTYPES_PATH.resolve()}")
    print(f"step1c1_team_colour_belief_rows_path: {STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH.resolve()}")
    print(f"step1c1_unknown_ambiguous_colour_rows_path: {STEP1C1_UNKNOWN_AMBIGUOUS_COLOUR_ROWS_PATH.resolve()}")
    print(f"step1c1_team_colour_belief_rows: {beliefs['summary']['step1c1_team_colour_belief_rows']}")
    print(f"unknown_ambiguous_colour_rows: {unknown['summary']['unknown_ambiguous_colour_rows']}")
    print(f"prototype_sandbox_only={str(prototypes['prototype_sandbox_only']).lower()}")
    print(f"auto_promoted={str(prototypes['auto_promoted']).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("goalkeeper_classification_performed=false")
    print("official_specialist_exclusion_performed=false")


if __name__ == "__main__":
    main()
