from __future__ import annotations

from stage1_bootstrap import bootstrap

bootstrap()

from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1_PERSON_STATES_PATH,
    STEP1_STATE_REPORT_PATH,
    build_and_write_person_states,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    payload = build_and_write_person_states()
    print(f"step1_person_states_path: {STEP1_PERSON_STATES_PATH.resolve()}")
    print(f"person_state_report_path: {STEP1_STATE_REPORT_PATH.resolve()}")
    print(f"state_counts: {payload['summary']['state_counts']}")
    print(f"visual_only_warning: {VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")


if __name__ == "__main__":
    main()
