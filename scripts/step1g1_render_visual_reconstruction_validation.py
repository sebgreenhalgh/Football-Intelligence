from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402
from football_intelligence.step1_visual_reconstruction.step1g_visual_reconstruction_render import (  # noqa: E402
    render_all_step1g1_validation_sheets,
)


def main() -> None:
    manifest = render_all_step1g1_validation_sheets()
    print(f"step1g1_validation_contact_sheet_path: {manifest.get('step1g1_validation_contact_sheet_path', '')}")
    crop_path = manifest.get("step1g1_final_role_crop_contact_sheet_path", "")
    print(f"step1g1_final_role_crop_contact_sheet_path: {crop_path}")
    print(f"validation_contact_sheet_panels: {manifest.get('panels', 0)}")
    print(f"final_role_crop_groups_rendered: {manifest.get('groups_rendered', 0)}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("no_auto_promotion=true")


if __name__ == "__main__":
    main()
