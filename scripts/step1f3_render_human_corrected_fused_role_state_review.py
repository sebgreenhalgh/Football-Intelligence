from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_human_correction_render import (  # noqa: E402
    render_all_human_corrected_fused_role_state_review_sheets,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    manifest = render_all_human_corrected_fused_role_state_review_sheets()
    print(f"step1f3_review_contact_sheet_path: {manifest.get('step1f3_review_contact_sheet_path', '')}")
    print(f"step1f3_role_crop_contact_sheet_path: {manifest.get('step1f3_role_crop_contact_sheet_path', '')}")
    print(f"review_contact_sheet_panels: {manifest.get('panels', 0)}")
    print(f"role_crop_groups_rendered: {manifest.get('groups_rendered', 0)}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")


if __name__ == "__main__":
    main()
