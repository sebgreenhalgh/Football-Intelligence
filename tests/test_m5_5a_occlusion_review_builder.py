from __future__ import annotations

import json
from pathlib import Path

from football_intelligence.replay.occlusion_review_builder import build_occlusion_human_review_package


def test_occlusion_review_package_keeps_answer_key_out_of_served_manifest(tmp_path: Path) -> None:
    result = build_occlusion_human_review_package(
        output_root=tmp_path / "review",
        unresolved_cases=[{"case_id": "case_008", "source_frame_sequence": 1, "target_frame_sequence": 3}],
    )
    manifest = json.loads(Path(result["reviewer_manifest"]).read_text(encoding="utf-8"))
    ui_config = json.loads(Path(result["ui_config"]).read_text(encoding="utf-8"))

    assert manifest["predecision_answer_key_delivered_to_client"] is False
    assert ui_config["predecision_answer_key_delivered_to_client"] is False
    assert "sealed" not in json.dumps(manifest).lower()
    assert Path(result["sealed_mapping"]["path"]).exists()
    assert result["sealed_mapping"]["accessible_through_static_route"] is False
