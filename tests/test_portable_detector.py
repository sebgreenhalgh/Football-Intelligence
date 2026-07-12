from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from football_intelligence.replay.portable_context import build_portable_context
from football_intelligence.replay.portable_step1 import run_portable_step1


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class FakeBoxes:
    def __init__(self, *, rows: list[list[float]] | None = None, classes: list[int] | None = None):
        self.xyxy = rows if rows is not None else [[10.0, 10.0, 40.0, 80.0]]
        self.conf = [0.91 for _ in self.xyxy]
        self.cls = classes if classes is not None else [0 for _ in self.xyxy]


class FakeResult:
    def __init__(self, *, rows: list[list[float]] | None = None, classes: list[int] | None = None):
        self.boxes = FakeBoxes(rows=rows, classes=classes)


class FakeYolo:
    task = "detect"
    names = {0: "person", 1: "sports ball"}

    def __init__(self, *, zero: bool = False, task: str = "detect", names: dict[int, str] | None = None):
        self.zero = zero
        self.task = task
        self.names = names or {0: "person", 1: "sports ball"}
        self.predict_calls = 0

    def predict(self, *, source, **kwargs):  # noqa: ANN001
        self.predict_calls += 1
        rows = [] if self.zero else [[10.0, 10.0, 40.0, 80.0]]
        return [FakeResult(rows=rows) for _ in source]


def make_context(
    tmp_path: Path,
    *,
    detector_mode: str = "model",
    model_bytes: bytes = b"fake-yolov8m",
    model_sha: str | None = None,
    detection_manifest: bool = False,
):
    repo_root = tmp_path / "repo"
    artifact_root = tmp_path / "artifacts"
    repo_root.mkdir()
    frame_root = artifact_root / "matches/128058/runs/step_m5/05_blind_second_window/frames/extraction_a"
    frame_root.mkdir(parents=True)
    frames = []
    for seq in range(3):
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        image[:, :] = (40, 110, 40)
        filename = f"frame_{seq:06d}.jpg"
        path = frame_root / filename
        assert cv2.imwrite(str(path), image)
        frames.append({"sequence": seq, "relative_uri": filename, "filename": filename, "width": 120, "height": 80})
    write_json(frame_root / "frame_manifest.json", {"frames": frames})
    write_json(
        artifact_root / "matches/128058/runs/step_m5/05_blind_second_window/source/source_video_manifest.json",
        {"artifact": "source_video_manifest"},
    )
    write_json(
        artifact_root / "matches/128058/runs/step_m5/05_blind_second_window/source/artifact_retention_contract.json",
        {"artifact": "artifact_retention_contract", "backup_status": "local_primary_only_backup_not_confirmed"},
    )
    write_json(
        artifact_root / "matches/128058/runs/step_m5/05_blind_second_window/selection/blind_window_selection_seal.json",
        {"artifact": "blind_window_selection_seal"},
    )
    (repo_root / "uv.lock").write_text("lock", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    model_path = repo_root / "models/model=yolov8m-imgsz=2048.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(model_bytes)
    actual_sha = hashlib.sha256(model_bytes).hexdigest()
    expected_sha = model_sha or actual_sha
    (repo_root / "models/model=yolov8m-imgsz=2048.pt.sha256").write_text(f"{expected_sha}  x\n", encoding="utf-8")
    write_json(
        repo_root / "models/model=yolov8m-imgsz=2048.provenance.json",
        {
            "detector_recovery_classification": "OFFICIAL_YOLOV8M_REFERENCE_IDENTIFIED_WITHOUT_HISTORICAL_HASH",
            "detector_source_classification": "NEW_OFFICIAL_PRETRAINED_BASELINE_NOT_HISTORICAL_WEIGHT_RECOVERY",
            "sha256": expected_sha,
        },
    )
    source_line = ""
    if detection_manifest:
        source_rel = "matches/128058/runs/step_m5/06a_detector_dependency_recovery/declared_detection_source.json"
        source_line = f"step1_detection_source_manifest: {source_rel}"
        write_json(
            artifact_root / source_rel,
            {
                "rows": [
                    {
                        "frame_sequence": 0,
                        "detection_id": "manifest_det",
                        "confidence": 0.8,
                        "x1": 10,
                        "y1": 10,
                        "x2": 40,
                        "y2": 70,
                    }
                ]
            },
        )
    config_path = repo_root / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "match_id: '128058'",
                "blind_window_id: test_blind_window",
                "selected_interval: {start_seconds: 780, end_seconds: 840, duration_seconds: 60}",
                "source_video_manifest: "
                "matches/128058/runs/step_m5/05_blind_second_window/source/source_video_manifest.json",
                "canonical_frame_manifest: "
                "matches/128058/runs/step_m5/05_blind_second_window/frames/extraction_a/frame_manifest.json",
                "canonical_frame_root: matches/128058/runs/step_m5/05_blind_second_window/frames/extraction_a",
                "source_retention_contract: "
                "matches/128058/runs/step_m5/05_blind_second_window/source/artifact_retention_contract.json",
                "blind_selection_seal: "
                "matches/128058/runs/step_m5/05_blind_second_window/selection/blind_window_selection_seal.json",
                "output_stage: matches/128058/runs/step_m5/06a_detector_dependency_recovery",
                "run_parent: matches/128058/runs/step_m5/06a_detector_dependency_recovery/runs",
                f"detector_input_mode: {detector_mode}",
                "model_weight_path: models/model=yolov8m-imgsz=2048.pt",
                "model_sha256_path: models/model=yolov8m-imgsz=2048.pt.sha256",
                "model_provenance_path: models/model=yolov8m-imgsz=2048.provenance.json",
                source_line,
                "step1_detection_source_name: player",
                "detector:",
                "  weight_path: models/model=yolov8m-imgsz=2048.pt",
                f"  model_sha256: {expected_sha}",
                "  model_provenance_classification: NEW_OFFICIAL_PRETRAINED_BASELINE_NOT_HISTORICAL_WEIGHT_RECOVERY",
                "  detector_recovery_classification: OFFICIAL_YOLOV8M_REFERENCE_IDENTIFIED_WITHOUT_HISTORICAL_HASH",
                "  task: detect",
                "  person_class_id: 0",
                "  expected_class_count: 2",
                "  imgsz: 64",
                "  confidence_threshold: 0.25",
                "  iou_threshold: 0.7",
                "  max_detections: 300",
                "  device: cpu",
                "  half_precision: false",
                "  batch_size: 1",
                "  deterministic: true",
                "  augmentation: false",
                "  agnostic_nms: false",
                "  retina_masks: false",
                "  save: false",
                "  stream: false",
            ]
        ),
        encoding="utf-8",
    )
    return build_portable_context(
        repo_root=repo_root,
        artifact_root=artifact_root,
        config_path=config_path,
        stage_root=artifact_root / "matches/128058/runs/step_m5/06a_detector_dependency_recovery",
        run_root=artifact_root / "matches/128058/runs/step_m5/06a_detector_dependency_recovery/runs/run_a",
    )


def test_model_mode_invokes_detector_and_writes_real_inference_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("football_intelligence.replay.portable_detector.EXPECTED_BASELINE_SHA256", None)
    context = make_context(tmp_path)
    fake = FakeYolo()
    result = run_portable_step1(context, detector_model_factory=lambda _path: fake)
    rows = json.loads(context.run_path("step1/detector/detection_rows.json").read_text(encoding="utf-8"))["rows"]
    assert result.completed
    assert fake.predict_calls > 0
    assert rows
    assert rows[0]["source_type"] == "real_model_inference"


def test_manifest_mode_does_not_invoke_model(tmp_path: Path) -> None:
    context = make_context(tmp_path, detector_mode="manifest", detection_manifest=True)

    def fail_factory(_path):  # noqa: ANN001
        raise AssertionError("model should not be loaded in manifest mode")

    result = run_portable_step1(context, detector_model_factory=fail_factory)
    assert result.completed
    assert not context.run_path("step1/detector/detection_rows.json").exists()


def test_model_hash_mismatch_blocks(tmp_path: Path) -> None:
    context = make_context(tmp_path, model_sha="0" * 64)
    result = run_portable_step1(context, detector_model_factory=lambda _path: FakeYolo())
    assert result.completion_status == "blocked_model_load_or_task_validation"


def test_non_detection_task_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("football_intelligence.replay.portable_detector.EXPECTED_BASELINE_SHA256", None)
    context = make_context(tmp_path)
    result = run_portable_step1(context, detector_model_factory=lambda _path: FakeYolo(task="classify"))
    assert result.completion_status == "blocked_model_load_or_task_validation"


def test_missing_person_class_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("football_intelligence.replay.portable_detector.EXPECTED_BASELINE_SHA256", None)
    context = make_context(tmp_path)
    result = run_portable_step1(context, detector_model_factory=lambda _path: FakeYolo(names={1: "sports ball"}))
    assert result.completion_status == "blocked_model_load_or_task_validation"


def test_zero_detections_are_not_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("football_intelligence.replay.portable_detector.EXPECTED_BASELINE_SHA256", None)
    context = make_context(tmp_path)
    result = run_portable_step1(context, detector_model_factory=lambda _path: FakeYolo(zero=True))
    assert result.completion_status == "detector_executed_zero_detections"
    assert not context.run_path("step2/step2m1_visual_continuity_node_rows.json").exists()


def test_invalid_checkpoint_load_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("football_intelligence.replay.portable_detector.EXPECTED_BASELINE_SHA256", None)
    context = make_context(tmp_path, model_bytes=b"not a checkpoint")

    def raise_load(_path):  # noqa: ANN001
        raise RuntimeError("invalid checkpoint")

    result = run_portable_step1(context, detector_model_factory=raise_load)
    assert result.completion_status == "blocked_model_load_or_task_validation"
