from __future__ import annotations

from stage1_bootstrap import bootstrap

bootstrap()

from football_intelligence.step1_visual_reconstruction.count_policy import build_and_write_count_policy_rows  # noqa: E402
from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1B3_COUNT_POLICY_ROWS_PATH,
    STEP1B3_RECONCILIATION_ROWS_PATH,
)
from football_intelligence.step1_visual_reconstruction.reconciliation import build_and_write_reconciliation_sandbox  # noqa: E402
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    reconciliation_payload = build_and_write_reconciliation_sandbox()
    count_policy_payload = build_and_write_count_policy_rows(reconciliation_payload)
    print(f"step1b3_reconciliation_rows_path: {STEP1B3_RECONCILIATION_ROWS_PATH.resolve()}")
    print(f"step1b3_count_policy_rows_path: {STEP1B3_COUNT_POLICY_ROWS_PATH.resolve()}")
    print(f"reconciliation_action_counts: {reconciliation_payload['summary']['reconciliation_action_counts']}")
    print(f"counted_observed_visible_rows: {count_policy_payload['summary']['counted_observed_visible_rows']}")
    print(f"not_counted_rows: {count_policy_payload['summary']['not_counted_rows']}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")


if __name__ == "__main__":
    main()
