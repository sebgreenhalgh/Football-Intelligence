from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from football_intelligence.replay.portable_context import (
    PortableVisualRunContext,
    guardrail_payload,
    semantic_hash,
    sha256_file,
    utc_now,
    write_json_file,
)


EXPECTED_BASELINE_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"


class DetectorValidationError(RuntimeError):
    """Raised when a declared detector cannot satisfy the portable runtime contract."""


@dataclass(frozen=True)
class PortableDetectorConfig:
    weight_path: Path
    model_sha256: str
    model_provenance_classification: str
    detector_recovery_classification: str
    task: str
    person_class_id: int
    expected_class_count: int
    imgsz: int
    confidence_threshold: float
    iou_threshold: float
    max_detections: int
    device: str
    half_precision: bool
    batch_size: int
    deterministic: bool
    augmentation: bool
    agnostic_nms: bool
    retina_masks: bool
    save: bool
    stream: bool
    runtime_values_source: str

    def contract_payload(self) -> dict[str, Any]:
        payload = guardrail_payload(
            {
                "artifact": "detector_runtime_contract",
                "created_at": utc_now(),
                "weight_path": str(self.weight_path),
                "model_sha256": self.model_sha256,
                "model_provenance_classification": self.model_provenance_classification,
                "detector_recovery_classification": self.detector_recovery_classification,
                "task": self.task,
                "person_class_id": self.person_class_id,
                "expected_class_count": self.expected_class_count,
                "imgsz": self.imgsz,
                "confidence_threshold": self.confidence_threshold,
                "iou_threshold": self.iou_threshold,
                "max_detections": self.max_detections,
                "device": self.device,
                "half_precision": self.half_precision,
                "batch_size": self.batch_size,
                "deterministic": self.deterministic,
                "augmentation": self.augmentation,
                "agnostic_nms": self.agnostic_nms,
                "retina_masks": self.retina_masks,
                "save": self.save,
                "stream": self.stream,
                "runtime_values_source": self.runtime_values_source,
                "historical_equivalence": False,
                "blind_output_tuning_performed": False,
            }
        )
        payload["inference_configuration_hash"] = semantic_hash(
            {
                "model_sha256": self.model_sha256,
                "imgsz": self.imgsz,
                "confidence_threshold": self.confidence_threshold,
                "iou_threshold": self.iou_threshold,
                "max_detections": self.max_detections,
                "device": self.device,
                "half_precision": self.half_precision,
                "batch_size": self.batch_size,
                "deterministic": self.deterministic,
                "augmentation": self.augmentation,
                "agnostic_nms": self.agnostic_nms,
                "retina_masks": self.retina_masks,
                "save": self.save,
                "stream": self.stream,
            }
        )
        return payload


def detector_config_from_context(context: PortableVisualRunContext) -> PortableDetectorConfig:
    cfg = dict(context.config.get("detector", {}) or {})
    weight_value = str(cfg.get("weight_path") or context.config.get("model_weight_path") or "")
    if not weight_value:
        raise DetectorValidationError("detector weight_path is not declared")
    weight_path = Path(weight_value)
    if not weight_path.is_absolute():
        weight_path = context.repo_root / weight_path
    model_sha = str(cfg.get("model_sha256") or "").lower()
    if not model_sha:
        raise DetectorValidationError("detector model_sha256 is not declared")
    return PortableDetectorConfig(
        weight_path=weight_path.resolve(),
        model_sha256=model_sha,
        model_provenance_classification=str(
            cfg.get("model_provenance_classification")
            or "NEW_OFFICIAL_PRETRAINED_BASELINE_NOT_HISTORICAL_WEIGHT_RECOVERY"
        ),
        detector_recovery_classification=str(
            cfg.get("detector_recovery_classification")
            or "OFFICIAL_YOLOV8M_REFERENCE_IDENTIFIED_WITHOUT_HISTORICAL_HASH"
        ),
        task=str(cfg.get("task") or "detect"),
        person_class_id=int(cfg.get("person_class_id", 0)),
        expected_class_count=int(cfg.get("expected_class_count", 80)),
        imgsz=int(cfg.get("imgsz", 2048)),
        confidence_threshold=float(cfg.get("confidence_threshold", 0.25)),
        iou_threshold=float(cfg.get("iou_threshold", 0.7)),
        max_detections=int(cfg.get("max_detections", 300)),
        device=str(cfg.get("device") or "cpu"),
        half_precision=bool(cfg.get("half_precision", False)),
        batch_size=max(1, int(cfg.get("batch_size", 1))),
        deterministic=bool(cfg.get("deterministic", True)),
        augmentation=bool(cfg.get("augmentation", False)),
        agnostic_nms=bool(cfg.get("agnostic_nms", False)),
        retina_masks=bool(cfg.get("retina_masks", False)),
        save=bool(cfg.get("save", False)),
        stream=bool(cfg.get("stream", False)),
        runtime_values_source=str(
            cfg.get("runtime_values_source") or "explicit_new_official_baseline_defaults_without_blind_tuning"
        ),
    )


def _plain_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _class_name(names: Any, class_id: int) -> str | None:
    if isinstance(names, dict):
        value = names.get(class_id, names.get(str(class_id)))
        return str(value) if value is not None else None
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return None


def _class_count(names: Any) -> int:
    if isinstance(names, dict | list):
        return len(names)
    return 0


def _load_yolo_model(weight_path: Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(weight_path))


def _torch_environment() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - defensive environment capture
        return {"torch_import_error": str(exc)}
    cuda_available = bool(torch.cuda.is_available())
    return {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
    }


def detector_environment_payload(config: PortableDetectorConfig) -> dict[str, Any]:
    try:
        import ultralytics

        ultralytics_version = getattr(ultralytics, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - defensive environment capture
        ultralytics_version = f"import_error:{exc}"
    payload = guardrail_payload(
        {
            "artifact": "detector_environment",
            "created_at": utc_now(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "ultralytics_version": ultralytics_version,
            "torch": _torch_environment(),
            "requested_device": config.device,
        }
    )
    return payload


def validate_detector_model(
    context: PortableVisualRunContext,
    config: PortableDetectorConfig,
    *,
    model_factory: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    if not config.weight_path.exists():
        raise DetectorValidationError(f"detector checkpoint is missing: {config.weight_path}")
    if context.source_ledger is not None:
        context.source_ledger.record_binary_read(
            config.weight_path,
            stage="detector",
            purpose="declared YOLO detector checkpoint",
        )
    actual_sha = sha256_file(config.weight_path)
    if actual_sha.lower() != config.model_sha256.lower():
        raise DetectorValidationError(f"detector checkpoint hash mismatch: {actual_sha}")
    if EXPECTED_BASELINE_SHA256 and config.model_sha256.lower() != EXPECTED_BASELINE_SHA256:
        raise DetectorValidationError("detector checkpoint is not the sealed official YOLOv8m baseline")

    factory = model_factory or _load_yolo_model
    try:
        model = factory(config.weight_path)
    except TypeError:
        model = factory(str(config.weight_path))
    except Exception as exc:
        raise DetectorValidationError(f"detector model load failed: {exc}") from exc

    task = str(getattr(model, "task", "") or getattr(getattr(model, "model", None), "task", "") or "")
    if task != config.task:
        raise DetectorValidationError(f"detector task mismatch: {task!r}")
    names = getattr(model, "names", None)
    class_name = _class_name(names, config.person_class_id)
    if class_name != "person":
        raise DetectorValidationError(f"person class id {config.person_class_id} is not mapped to 'person'")
    class_count = _class_count(names)
    if class_count != config.expected_class_count:
        raise DetectorValidationError(f"detector class count mismatch: {class_count}")

    validation = guardrail_payload(
        {
            "artifact": "detector_validation",
            "created_at": utc_now(),
            "passed": True,
            "model_load_successful": True,
            "weight_path": str(config.weight_path),
            "model_sha256": actual_sha,
            "task": task,
            "class_count": class_count,
            "person_class_id": config.person_class_id,
            "person_class_name": class_name,
            "historical_equivalence": False,
            "source_classification": config.model_provenance_classification,
        }
    )
    return model, validation


def _predict_kwargs(config: PortableDetectorConfig) -> dict[str, Any]:
    return {
        "imgsz": config.imgsz,
        "conf": config.confidence_threshold,
        "iou": config.iou_threshold,
        "max_det": config.max_detections,
        "classes": [config.person_class_id],
        "device": config.device,
        "half": config.half_precision,
        "augment": config.augmentation,
        "agnostic_nms": config.agnostic_nms,
        "save": config.save,
        "stream": config.stream,
        "verbose": False,
        "batch": config.batch_size,
    }


def _rows_from_results(
    *,
    results: list[Any],
    frames: list[dict[str, Any]],
    config: PortableDetectorConfig,
    inference_hash: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result, frame in zip(results, frames, strict=True):
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy_rows = _plain_list(getattr(boxes, "xyxy", []))
        confidences = _plain_list(getattr(boxes, "conf", []))
        classes = _plain_list(getattr(boxes, "cls", []))
        for local_index, xyxy in enumerate(xyxy_rows):
            class_id = int(float(classes[local_index])) if local_index < len(classes) else config.person_class_id
            if class_id != config.person_class_id:
                continue
            confidence = float(confidences[local_index]) if local_index < len(confidences) else 0.0
            if confidence < config.confidence_threshold:
                continue
            coords = [float(value) for value in xyxy]
            seq = int(frame["frame_sequence"])
            rows.append(
                {
                    "frame_sequence": seq,
                    "source_frame_index": frame.get("source_frame_index"),
                    "frame_filename": Path(str(frame["frame_file"])).name,
                    "frame_file": frame["frame_file"],
                    "detection_id": f"yolov8m_person_f{seq:06d}_{local_index:03d}",
                    "source_detection_id": f"yolov8m_person_f{seq:06d}_{local_index:03d}",
                    "class_id": class_id,
                    "class_name": "person",
                    "confidence": round(confidence, 6),
                    "x1": round(coords[0], 3),
                    "y1": round(coords[1], 3),
                    "x2": round(coords[2], 3),
                    "y2": round(coords[3], 3),
                    "model_sha256": config.model_sha256,
                    "inference_configuration_hash": inference_hash,
                    "source_type": "real_model_inference",
                    "classification_reason": "real_yolov8m_person_detection",
                    "object_type": "player_candidate",
                    "role_label": "player",
                }
            )
    rows.sort(key=lambda row: (int(row["frame_sequence"]), float(row["y1"]), float(row["x1"]), row["detection_id"]))
    for index, row in enumerate(rows):
        seq = int(row["frame_sequence"])
        row["detection_id"] = f"yolov8m_person_f{seq:06d}_{index:05d}"
        row["source_detection_id"] = row["detection_id"]
    return rows


def _frame_summary_payload(frames: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[int, int] = {int(frame["frame_sequence"]): 0 for frame in frames}
    for row in rows:
        counts[int(row["frame_sequence"])] = counts.get(int(row["frame_sequence"]), 0) + 1
    return guardrail_payload(
        {
            "artifact": "detection_frame_summary",
            "created_at": utc_now(),
            "frame_count": len(frames),
            "total_person_detections": len(rows),
            "frames_with_person_detections": sum(1 for value in counts.values() if value > 0),
            "frames_without_person_detections": sum(1 for value in counts.values() if value == 0),
            "per_frame_counts": [
                {"frame_sequence": seq, "person_detection_count": count} for seq, count in sorted(counts.items())
            ],
        }
    )


def run_detector_smoke_test(
    context: PortableVisualRunContext,
    *,
    model_factory: Any | None = None,
) -> dict[str, Any]:
    config = detector_config_from_context(context)
    contract = config.contract_payload()
    context.write_json("step1/detector/detector_runtime_contract.json", contract)
    context.write_json("step1/detector/detector_environment.json", detector_environment_payload(config))
    try:
        model, validation = validate_detector_model(context, config, model_factory=model_factory)
        context.write_json("step1/detector/detector_validation.json", validation)
        frames = context.canonical_frames()
        samples = [frames[index] for index in (0, len(frames) // 2, len(frames) - 1) if frames]
        for frame in samples:
            if context.source_ledger is not None:
                context.source_ledger.record_binary_read(
                    Path(frame["frame_file"]),
                    stage="detector_smoke",
                    purpose="smoke-test input frame",
                )
        results = list(model.predict(source=[frame["frame_file"] for frame in samples], **_predict_kwargs(config)))
        rows = _rows_from_results(
            results=results,
            frames=samples,
            config=config,
            inference_hash=contract["inference_configuration_hash"],
        )
        payload = guardrail_payload(
            {
                "artifact": "detector_smoke_test",
                "created_at": utc_now(),
                "passed": True,
                "sample_frame_sequences": [int(frame["frame_sequence"]) for frame in samples],
                "result_object_count": len(results),
                "person_detection_count": len(rows),
                "zero_detections_observed": len(rows) == 0,
                "configuration_changed_after_smoke": False,
            }
        )
    except Exception as exc:
        payload = guardrail_payload(
            {
                "artifact": "detector_smoke_test",
                "created_at": utc_now(),
                "passed": False,
                "blocking_reason": str(exc),
                "configuration_changed_after_smoke": False,
            }
        )
    write_json_file(context.stage_path("validation/detector_smoke_test.json"), payload)
    return payload


def run_detector_inference(
    context: PortableVisualRunContext,
    *,
    model_factory: Any | None = None,
) -> dict[str, Any]:
    config = detector_config_from_context(context)
    contract = config.contract_payload()
    context.write_json("step1/detector/detector_runtime_contract.json", contract)
    context.write_json("step1/detector/detector_environment.json", detector_environment_payload(config))
    model, validation = validate_detector_model(context, config, model_factory=model_factory)
    context.write_json("step1/detector/detector_validation.json", validation)

    frames = context.canonical_frames()
    all_rows: list[dict[str, Any]] = []
    kwargs = _predict_kwargs(config)
    for start in range(0, len(frames), config.batch_size):
        batch_frames = frames[start : start + config.batch_size]
        for frame in batch_frames:
            if context.source_ledger is not None:
                context.source_ledger.record_binary_read(
                    Path(frame["frame_file"]),
                    stage="detector",
                    purpose="YOLO inference input frame",
                )
        results = list(model.predict(source=[frame["frame_file"] for frame in batch_frames], **kwargs))
        all_rows.extend(
            _rows_from_results(
                results=results,
                frames=batch_frames,
                config=config,
                inference_hash=contract["inference_configuration_hash"],
            )
        )

    frame_summary = _frame_summary_payload(frames, all_rows)
    manifest = guardrail_payload(
        {
            "artifact": "detector_source_manifest",
            "created_at": utc_now(),
            "source_type": "real_model_inference",
            "model_sha256": config.model_sha256,
            "inference_configuration_hash": contract["inference_configuration_hash"],
            "frame_count": len(frames),
            "person_detection_count": len(all_rows),
            "rows": all_rows,
            "frames": [
                {
                    **frame,
                    "detections": [
                        row for row in all_rows if int(row["frame_sequence"]) == int(frame["frame_sequence"])
                    ],
                }
                for frame in frames
            ],
        }
    )
    context.write_json("step1/detector/detection_rows.json", {"artifact": "detection_rows", "rows": all_rows})
    context.write_json("step1/detector/detection_frame_summary.json", frame_summary)
    context.write_json("step1/detector/detector_source_manifest.json", manifest)
    return manifest
