from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from football_intelligence.replay.portable_context import build_portable_context
from football_intelligence.replay.portable_pipeline import compare_portable_runs, no_tuning_audit
from football_intelligence.replay.portable_step1 import run_portable_step1
from football_intelligence.replay.portable_step2 import run_portable_step2


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_context(tmp_path: Path, *, missing_model: bool = False):
    repo_root = tmp_path / "repo"
    artifact_root = tmp_path / "artifacts"
    frame_root = artifact_root / "matches/128058/runs/step_m5/05_blind_second_window/frames/extraction_a"
    frame_root.mkdir(parents=True)
    frames = []
    for seq in range(3):
        image = np.zeros((120, 240, 3), dtype=np.uint8)
        image[:, :] = (40, 120, 40)
        cv2.rectangle(image, (50 + seq, 20), (86 + seq, 100), (30, 30, 210), -1)
        filename = f"frame_{seq:06d}.jpg"
        path = frame_root / filename
        assert cv2.imwrite(str(path), image)
        frames.append(
            {
                "sequence": seq,
                "relative_uri": filename,
                "filename": filename,
                "width": 240,
                "height": 120,
                "byte_sha256": "",
                "decoded_pixel_sha256": "",
            }
        )
    write_json(
        frame_root / "frame_manifest.json",
        {"actual_frame_count": len(frames), "expected_frame_count": len(frames), "frames": frames},
    )
    write_json(
        artifact_root / "matches/128058/runs/step_m5/05_blind_second_window/source/source_video_manifest.json",
        {"artifact": "source_video_manifest", "source_video_sha256": "abc"},
    )
    write_json(
        artifact_root / "matches/128058/runs/step_m5/05_blind_second_window/source/artifact_retention_contract.json",
        {"artifact": "artifact_retention_contract", "backup_status": "local_primary_only_backup_not_confirmed"},
    )
    write_json(
        artifact_root / "matches/128058/runs/step_m5/05_blind_second_window/selection/blind_window_selection_seal.json",
        {"artifact": "blind_window_selection_seal", "selection_hash": "seal"},
    )
    repo_root.mkdir()
    (repo_root / "uv.lock").write_text("lock", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (repo_root / "src/football_intelligence/replay").mkdir(parents=True)
    for filename in [
        "portable_context.py",
        "portable_step1.py",
        "portable_step1_validation.py",
        "portable_step2.py",
        "portable_step2_validation.py",
    ]:
        (repo_root / "src/football_intelligence/replay" / filename).write_text("# test\n", encoding="utf-8")
    (repo_root / "src/football_intelligence/step1_visual_reconstruction").mkdir(parents=True)
    (repo_root / "src/football_intelligence/step2_visual_continuity").mkdir(parents=True)
    for relative in [
        "src/football_intelligence/step1_visual_reconstruction/person_candidates.py",
        "src/football_intelligence/step1_visual_reconstruction/state_model.py",
        "src/football_intelligence/step2_visual_continuity/nodes.py",
        "src/football_intelligence/step2_visual_continuity/edge_candidates.py",
        "src/football_intelligence/step2_visual_continuity/grouping.py",
    ]:
        (repo_root / relative).write_text("# test\n", encoding="utf-8")
    if not missing_model:
        (repo_root / "models").mkdir()
        (repo_root / "models/model=yolov8m-imgsz=2048.pt").write_text("weights", encoding="utf-8")
    config_path = repo_root / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "match_id: '128058'",
                "blind_window_id: test_blind_window",
                "selected_interval:",
                "  start_seconds: 780",
                "  end_seconds: 840",
                "  duration_seconds: 60",
                "source_video_manifest: "
                "matches/128058/runs/step_m5/05_blind_second_window/source/source_video_manifest.json",
                "canonical_frame_manifest: "
                "matches/128058/runs/step_m5/05_blind_second_window/frames/extraction_a/frame_manifest.json",
                "canonical_frame_root: matches/128058/runs/step_m5/05_blind_second_window/frames/extraction_a",
                "source_retention_contract: "
                "matches/128058/runs/step_m5/05_blind_second_window/source/artifact_retention_contract.json",
                "blind_selection_seal: "
                "matches/128058/runs/step_m5/05_blind_second_window/selection/blind_window_selection_seal.json",
                "output_stage: matches/128058/runs/step_m5/06_portable_blind_pipeline",
                "run_parent: matches/128058/runs/step_m5/06_portable_blind_pipeline/runs",
                "model_weight_path: models/model=yolov8m-imgsz=2048.pt",
                "step1_detection_source_name: player",
            ]
        ),
        encoding="utf-8",
    )
    context = build_portable_context(
        repo_root=repo_root,
        artifact_root=artifact_root,
        config_path=config_path,
        stage_root=artifact_root / "matches/128058/runs/step_m5/06_portable_blind_pipeline",
        run_root=artifact_root / "matches/128058/runs/step_m5/06_portable_blind_pipeline/runs/run_a",
    )
    return context


def synthetic_detection_payload() -> dict:
    return {
        "frames": [
            {
                "frame_sequence": seq,
                "detections": [
                    {
                        "detection_id": f"det_{seq}",
                        "source_detection_id": f"src_{seq}",
                        "bbox": {"x1": 50 + seq, "y1": 20, "x2": 86 + seq, "y2": 100},
                        "confidence": 0.91,
                        "role_label": "player",
                    }
                ],
            }
            for seq in range(3)
        ]
    }


def test_portable_step1_and_step2_synthetic_flow(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    step1 = run_portable_step1(context, detection_payload=synthetic_detection_payload())
    assert step1.completed
    assert step1.counts["f3_row_count"] == 3
    step2 = run_portable_step2(context)
    assert step2.completed
    assert step2.counts["node_count"] == 3
    assert step2.counts["candidate_edge_count"] > 0
    assert step2.counts["review_candidate_count"] <= 32
    validation = json.loads(context.run_path("validation/step2_portable_validation.json").read_text(encoding="utf-8"))
    assert validation["historical_decisions_used"] is False
    assert validation["preserved_m4_used_as_input"] is False


def test_missing_detection_dependency_blocks_without_placeholder_success(tmp_path: Path) -> None:
    context = make_context(tmp_path, missing_model=True)
    result = run_portable_step1(context)
    assert result.completion_status == "blocked_missing_model_or_configuration_dependency"
    assert not context.run_path("step1/step1f3_human_corrected_fused_visual_role_state_rows.json").exists()
    step2 = run_portable_step2(context)
    assert step2.completion_status == "blocked_step1_required_output_missing"


def test_source_ledger_rejects_preserved_m4_input(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    m4_path = (
        context.artifact_root
        / "matches/128058/calibration/step2_visual_continuity/step2m4_sparse_handoff_package/x.json"
    )
    write_json(m4_path, {"artifact": "m4"})
    try:
        context.source_ledger.record_binary_read(m4_path, stage="test", purpose="forbidden preserved M4")
    except ValueError as exc:
        assert "preserved_m4" in str(exc)
    else:
        raise AssertionError("preserved M4 input was not rejected")


def test_no_tuning_and_blocked_parity_not_repeatability(tmp_path: Path) -> None:
    context = make_context(tmp_path, missing_model=True)
    audit = no_tuning_audit(context)
    assert audit["passed"] is True
    run_portable_step1(context)
    context.write_json(
        "run_summary.json",
        {"completion_status": "blocked_missing_model_or_configuration_dependency"},
    )
    run_b = context.stage_root / "runs/run_b"
    run_b.mkdir(parents=True)
    write_json(run_b / "run_summary.json", {"completion_status": "blocked_missing_model_or_configuration_dependency"})
    comparison = compare_portable_runs(stage_root=context.stage_root, run_a=context.run_root, run_b=run_b)
    assert comparison["blocked_status_parity_is_not_pipeline_repeatability"] is True
    assert comparison["row_level_repeatability_passed"] is False
