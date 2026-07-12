from __future__ import annotations

from pathlib import Path

from PIL import Image

from football_intelligence.replay.frame_set_validator import (
    derived_classification,
    inspect_derived_asset_exclusions,
)


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 10), (12, 34, 56)).save(path)


def test_derived_assets_are_rejected_as_true_replay_sources(tmp_path: Path) -> None:
    artifact_root = tmp_path
    preserved = (
        artifact_root / "matches/128058/calibration/step2_visual_continuity/"
        "step2m4_sparse_handoff_package/step2m4_pathlet_overlay_frames"
    )
    static_freeze = artifact_root / "matches/128058/overlays/goal_window_stage3d_static_freeze_clean"
    contact_sheet = (
        artifact_root / "matches/128058/calibration/step2_visual_continuity/"
        "step2m3t_sparse_pathlets/step2m3t_review_contact_sheet.jpg"
    )
    write_image(preserved / "step2m4_handoff_pathlet_000001_f000001.jpg")
    write_image(static_freeze / "0001_128058_h1_1882_2062_stage3c_hq_f000625_stage3d4c_static_freeze.jpg")
    write_image(contact_sheet)

    report = inspect_derived_asset_exclusions(artifact_root)

    assert report["passed"] is True
    assert report["proves_no_derived_asset_promoted"] is True
    assert all(record["allowed_for_true_replay"] is False for record in report["records"])
    assert derived_classification(static_freeze) == "derived_annotated_frames_not_eligible"
    assert derived_classification(preserved) == "preserved_m4_evidence_not_eligible"
