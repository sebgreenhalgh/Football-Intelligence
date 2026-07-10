from __future__ import annotations

from stage1_bootstrap import bootstrap

bootstrap()

from football_intelligence.step1_visual_reconstruction.input_contracts import build_and_write_step1c_input_contracts  # noqa: E402
from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1B4_RETAINED_CANDIDATE_PROVENANCE_ROWS_PATH,
    STEP1B4_REVIEW_DECISION_TEMPLATE_PATH,
    STEP1B4_STEP1C_INPUT_CONTRACT_PATH,
    STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402
from football_intelligence.step1_visual_reconstruction.visible_person_base import build_and_write_visible_person_base  # noqa: E402


def main() -> None:
    base_payload, provenance_payload = build_and_write_visible_person_base()
    build_and_write_step1c_input_contracts()
    print(f"step1b4_visible_person_base_rows_path: {STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH.resolve()}")
    print(
        "step1b4_retained_candidate_provenance_rows_path: "
        f"{STEP1B4_RETAINED_CANDIDATE_PROVENANCE_ROWS_PATH.resolve()}"
    )
    print(f"step1b4_step1c_input_contract_path: {STEP1B4_STEP1C_INPUT_CONTRACT_PATH.resolve()}")
    print(f"step1b4_review_decision_template_path: {STEP1B4_REVIEW_DECISION_TEMPLATE_PATH.resolve()}")
    print(f"visible_person_base_rows: {base_payload['summary']['visible_person_base_rows']}")
    print(f"retained_candidate_provenance_rows: {provenance_payload['summary']['provenance_rows']}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")


if __name__ == "__main__":
    main()
