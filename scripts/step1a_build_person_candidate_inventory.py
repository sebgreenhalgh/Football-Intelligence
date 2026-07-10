from __future__ import annotations

from stage1_bootstrap import bootstrap

bootstrap()

from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1_CANDIDATE_REPORT_PATH,
    STEP1_PERSON_CANDIDATES_PATH,
    build_and_write_candidate_inventory,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    payload = build_and_write_candidate_inventory()
    print(f"step1_person_candidates_path: {STEP1_PERSON_CANDIDATES_PATH.resolve()}")
    print(f"candidate_inventory_report_path: {STEP1_CANDIDATE_REPORT_PATH.resolve()}")
    print(f"candidate_rows: {payload['summary']['total_rows']}")
    print(f"visual_only_warning: {VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")


if __name__ == "__main__":
    main()
