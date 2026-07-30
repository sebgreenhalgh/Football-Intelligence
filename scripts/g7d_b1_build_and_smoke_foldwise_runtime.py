"""Build and smoke-test the frozen G7D-B1 five-fold runtime exactly once."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from football_intelligence.detection_gold.consolidation import consolidate_proposals
from football_intelligence.football_observation_reasoner.features import (
    FrozenTorchvisionEncoder,
    RobustPerspectivePrior,
    crop_tensor_from_box,
    deterministic_candidate_crop_boxes,
    extract_candidate_feature_families,
)
from football_intelligence.football_observation_reasoner.g7b_stage import (
    k1_crop_features,
    node_tabular_features,
)
from football_intelligence.g7d_b1_foldwise_runtime import (
    FOLD_ORDER,
    FoldArtifact,
    FrozenFoldwiseRuntime,
    PARENT_CONTRACT_ID,
    RUNTIME_CONTRACT_ID,
    frame_local_candidate_id,
    proposal_view_plan,
    sha256_file,
    validate_candidate_record,
)
from football_intelligence.proposal_gate_hook import apply_shadow_hook
from football_intelligence.review_chassis.hashing import stable_hash


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
STAGE = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B1_PROPOSAL_CLOSURE_AND_FOLDWISE_RUNTIME_v1"
PACK = (
    PROJECT
    / "experiments/football_observation_reasoner/part 6/G7D_B1_Proposal_Closure_And_Foldwise_Runtime_RevB_Codex_Pack"
)
PART3 = PROJECT / "matches/128058/runs/step_m5/part 3"
PART4 = PROJECT / "matches/128058/runs/step_m5/part 4"
G6E = PART3 / "M5_5G6E_C0_PROPOSAL_REINTEGRATION_AND_PLAYER_OBSERVATION_V1_FULL_UNIVERSE_VALIDATION_v1"
G7A = PART4 / "M5_5G7A_FOOTBALL_OBSERVATION_REASONER_V0_ARCHITECTURE_DATASET_AND_BASELINES_v1"
G7B = PART4 / "M5_5G7B_K1_SUPERVISED_MULTITASK_AND_HIERARCHICAL_OBSERVATION_SELECTION_v1"
G7D_A = PROJECT / "experiments/football_observation_reasoner/part 5/G7D_A_TWO_MATCH_SETUP_AND_PITCH_POLYGON_REVIEW_v1"
BASELINE = "c6f221fe2e9790e7128f3dc8079354825556121c"
DETECTOR = REPO / "models/model=yolov8m-imgsz=2048.pt"
DETECTOR_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
CLASSIFICATION = "PASS_G7D_B1_PROPOSAL_CLOSURE_AND_FOLDWISE_RUNTIME_READY_FOR_BASELINE_RERUN"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")


def artifact(path: Path, logical_name: str, source_stage: str, purpose: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing required artifact: {path}")
    return {
        "logical_name": logical_name,
        "project_relative_path": path.resolve().relative_to(PROJECT.resolve()).as_posix(),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
        "source_stage": source_stage,
        "purpose": purpose,
        "required": True,
    }


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def validate_baseline() -> None:
    status = git("status", "--porcelain").splitlines()
    allowed = {
        "?? scripts/g7d_b1_build_and_smoke_foldwise_runtime.py",
        "?? src/football_intelligence/g7d_b1_foldwise_runtime.py",
        "?? tests/test_g7d_b1_foldwise_runtime.py",
    }
    if git("rev-parse", "HEAD") != BASELINE or any(row not in allowed for row in status):
        raise RuntimeError("FAIL_BASELINE_OR_WORKTREE")
    split = read_json(PROJECT / "datasets/soccertrack_v2/splits/split_v1/split_manifest.json")
    if split.get("status") != "FROZEN_HUMAN_APPROVED" or split.get("frozen") is not True:
        raise RuntimeError("FAIL_FROZEN_SPLIT_OR_POLYGONS")
    expected = {
        "118575": "fbd7f3a473acc197b4c893d90bbaa4c5d484d1e883e8df1ac4601daf4396dec1",
        "117092": "92ca8040eedd3b0ec0bb685648691f0c314d8527f3fa8f2db1823b4461e4b338",
    }
    assignments = split.get("assignments") or split.get("match_assignments") or {}
    for match_id, polygon_hash in expected.items():
        setup = read_json(PROJECT / f"matches/{match_id}/calibration/match_setup.json")
        polygon_path = PROJECT / f"matches/{match_id}/calibration/pitch_polygon_v1/pitch_polygon.json"
        polygon = read_json(polygon_path)
        assignment = assignments.get(match_id)
        if isinstance(assignment, Mapping):
            assignment = assignment.get("split") or assignment.get("assignment")
        setup_assignment = setup["dataset_split"]["proposed_assignment"]
        if assignment not in {None, "TRAIN_DEVELOPMENT"} or setup_assignment != "TRAIN_DEVELOPMENT":
            raise RuntimeError("FAIL_FROZEN_SPLIT_OR_POLYGONS")
        if sha256_file(polygon_path) != polygon_hash:
            raise RuntimeError("FAIL_FROZEN_SPLIT_OR_POLYGONS")
        if not (
            polygon["status"] == "HUMAN_CONFIRMED"
            and polygon["second_half_alignment_answer"] == "YES"
            and len(polygon["camera_segments"]) == 1
            and polygon["camera_segments"][0]["segment_id"] == "MATCH_STABLE_CAMERA"
        ):
            raise RuntimeError("FAIL_FROZEN_SPLIT_OR_POLYGONS")


def validate_pack() -> None:
    manifest = read_json(PACK / "06_PACK_MANIFEST.json")
    for row in manifest["files"]:
        path = PACK / row["path"]
        if not path.is_file() or path.stat().st_size != row["byte_size"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"prompt pack mismatch: {row['path']}")


def proposal_closure() -> dict[str, Any]:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(STAGE / "_tmp/ultralytics_config"))
    anchors = [
        artifact(
            G6E / "03_FULL_UNIVERSE_C0_REPLAY/full_universe_contract.json",
            "g6e_full_universe_contract",
            "G6E",
            "frozen proposal contract",
        ),
        artifact(
            G6E / "03_FULL_UNIVERSE_C0_REPLAY/c0_full_universe_replay_manifest.json",
            "g6e_replay_manifest",
            "G6E",
            "historical exact replay evidence",
        ),
        artifact(
            REPO / "scripts/build_m5_5g6e_c0_reintegration.py",
            "g6e_generating_code",
            "G6E",
            "unseen missing-source branch and consolidation",
        ),
        artifact(
            REPO / "scripts/build_m5_5g0_detection_forensics.py",
            "g0_diagnostic_runner",
            "G0",
            "exact detector/NMS/coordinate runner",
        ),
        artifact(
            REPO / "src/football_intelligence/detection_forensics.py",
            "detector_runtime_helpers",
            "G0",
            "threshold, NMS, transform, lineage helpers",
        ),
        artifact(
            REPO / "src/football_intelligence/step1_visual_reconstruction/tiled_detection.py",
            "tile_grid_code",
            "G6E",
            "deterministic S3 tile grid",
        ),
        artifact(
            REPO / "src/football_intelligence/detection_gold/consolidation.py",
            "proposal_consolidation_code",
            "G6E",
            "IOU_CONNECTED_COMPONENT_055",
        ),
        artifact(
            REPO / "src/football_intelligence/football_observation_reasoner/features.py",
            "candidate_feature_code",
            "G7A",
            "crop, visual, geometry, perspective features",
        ),
        artifact(
            REPO / "src/football_intelligence/football_observation_reasoner/g7b_stage.py",
            "g7b_tabular_feature_code",
            "G7B",
            "fixed 32-value feature ordering",
        ),
        artifact(
            REPO / "src/football_intelligence/football_observation_reasoner/models.py",
            "n3_model_code",
            "G7B",
            "N3 architecture and class order",
        ),
        artifact(DETECTOR, "historical_yolov8m_checkpoint", "G0/G6E", "person proposal detector"),
        artifact(
            G7A / "04_FROZEN_PRETRAINED_ENCODER_PROVENANCE/frozen_visual_encoder_provenance.json",
            "frozen_resnet18_provenance",
            "G7A",
            "visual encoder and preprocessing",
        ),
        artifact(
            G7A / "06_PERSPECTIVE_AND_SCALE_PRIOR/global_descriptive_perspective_prior.json",
            "global_perspective_prior",
            "G7A",
            "frozen expected-scale evidence",
        ),
        artifact(
            G7A / "07_VISUAL_AND_GEOMETRY_FEATURES/feature_specification.json",
            "feature_specification",
            "G7A",
            "feature schema",
        ),
        artifact(
            G7A / "07_VISUAL_AND_GEOMETRY_FEATURES/feature_cache_manifest.json",
            "feature_cache_manifest",
            "G7A",
            "historical feature provenance",
        ),
    ]
    expected_hashes = {
        "g6e_full_universe_contract": "ae2f9a277b20a027313541393db5bde6c4236d44c045667c4662f9815feb1ceb",
        "g6e_replay_manifest": "062ddff152de5b7ce37a4b5cc81e79670793802383dcaf59828424690599bb90",
        "g6e_generating_code": "9cf4ef41d5272a92a78c8be104c860e99f8d11efa64446cdb4610059f1d719af",
        "historical_yolov8m_checkpoint": DETECTOR_SHA256,
        "feature_specification": "2661cf9eda3632eeefdf915243a1570f2aa2472e2dc55298bb15c26ffda3dad6",
        "feature_cache_manifest": "a3824ac5dc1c859952e6887c5e62b3196dbd4933e5ff68009c4ad29f53278f15",
    }
    by_name = {row["logical_name"]: row for row in anchors}
    for name, expected in expected_hashes.items():
        if by_name[name]["sha256"] != expected:
            raise RuntimeError(f"FAIL_G7D_B1_PROPOSAL_DEPENDENCY_CLOSURE: {name}")
    import PIL
    import torchvision
    import ultralytics

    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": PIL.__version__,
        "cuda_required": True,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda:0",
        "dtype": "fp16 detector; float32 features and semantic heads",
        "deterministic_algorithms": True,
    }
    if not torch.cuda.is_available() or ultralytics.__version__ != "8.3.49":
        raise RuntimeError("FAIL_G7D_B1_PROPOSAL_DEPENDENCY_CLOSURE: environment")
    contract = {
        "schema_version": "football_intelligence.g7d_b1.proposal_runtime_contract.v1",
        "runtime": {
            "architecture_family": "Ultralytics YOLOv8m COCO",
            "class_mapping": {"person": 0},
            "checkpoint_sha256": DETECTOR_SHA256,
            "views": [
                {"family": "S0_FULL_PANORAMA_1280", "imgsz": 1280, "count": 1},
                {
                    "family": "S3_OVERLAPPING_HIGH_RESOLUTION_TILES",
                    "imgsz": 1536,
                    "tile": [1024, 720],
                    "overlap": [256, 0],
                    "padding": 0,
                },
            ],
            "confidence": 0.22,
            "iou": 0.70,
            "max_detections_per_view": 80,
            "augment": False,
            "agnostic_nms": False,
            "batch": 1,
            "view_order": "S0 then S3 tile_index ascending",
            "consolidation": "IOU_CONNECTED_COMPONENT_055",
            "representative": "HIGHEST_SCORE_REAL_MEMBER with stable proposal UUID tie-break",
            "merged_gate": True,
            "source_files_are_full_resolution": True,
        },
        "environment": environment,
        "no_substitution": True,
        "human_geometry_used_by_proposal_detector": False,
    }
    coordinate = {
        "schema_version": "football_intelligence.g7d_b1.proposal_coordinate_transform_contract.v1",
        "crop_bounds": "integer rounded and source clipped",
        "letterbox": "Ultralytics 8.3.49 predictor preprocessing",
        "model_to_crop": "ultralytics.utils.ops.scale_boxes float32",
        "crop_to_source": "add integer crop origin",
        "box_clipping": "source image bounds",
        "coordinate_roundtrip_max_error_pixels": 1.0,
        "tile_generation": {"width": 1024, "height": 720, "overlap_x": 256, "overlap_y": 0, "padding": 0},
        "candidate_order": "observation_uuid lexical",
        "candidate_id_scope": "FRAME_LOCAL_ONLY",
    }
    graph = {
        "nodes": [
            "source_frame",
            "S0_and_S3_views",
            "raw_detector_tensor",
            "confidence_and_NMS_rows",
            "source_coordinate_proposals",
            "IOU_CC_055_consolidated_candidates",
            "context_crops",
            "shared_544_raw_features",
        ],
        "edges": [
            ["source_frame", "S0_and_S3_views"],
            ["S0_and_S3_views", "raw_detector_tensor"],
            ["raw_detector_tensor", "confidence_and_NMS_rows"],
            ["confidence_and_NMS_rows", "source_coordinate_proposals"],
            ["source_coordinate_proposals", "IOU_CC_055_consolidated_candidates"],
            ["IOU_CC_055_consolidated_candidates", "context_crops"],
            ["context_crops", "shared_544_raw_features"],
        ],
    }
    write_json(
        STAGE / "01_PROPOSAL_CLOSURE/proposal_dependency_registry.json",
        {
            "schema_version": "football_intelligence.g7d_b1.proposal_dependency_registry.v1",
            "artifacts": anchors,
            "environment": environment,
        },
    )
    write_json(STAGE / "01_PROPOSAL_CLOSURE/proposal_runtime_contract.json", contract)
    write_json(STAGE / "01_PROPOSAL_CLOSURE/proposal_coordinate_transform_contract.json", coordinate)
    report = {
        "schema_version": "football_intelligence.g7d_b1.proposal_closure_report.v1",
        "passed": True,
        "unseen_entry_point": "G6E exact missing-source DiagnosticRunner branch",
        "dependency_graph": graph,
        "artifact_count": len(anchors),
        "no_substitution": True,
    }
    write_json(STAGE / "01_PROPOSAL_CLOSURE/proposal_closure_report.json", report)
    return report


def materialize_folds() -> list[FoldArtifact]:
    audit = read_json(PACK / "07_B0_CHATGPT_HANDOFF/02_SEMANTIC_FOLD_AUDIT.json")
    expected_by_fold = {row["outer_fold"]: row for row in audit["folds"]}
    nodes_path = G7A / "05_FOOTBALL_REASONER_DATASET/football_reasoner_node_rows.parquet"
    import pyarrow.parquet as pq

    nodes = pq.read_table(nodes_path).to_pylist()
    split = read_json(G7A / "05_FOOTBALL_REASONER_DATASET/grouped_split_manifest.json")
    node_cache = torch.load(
        G7A / "_tmp/embeddings/official_resnet18_candidate_embeddings.pt", map_location="cpu", weights_only=True
    )["embeddings"]
    k1_rows = read_jsonl(G7B / "02_K1_TARGET_BINDING_AND_DATA_JOIN/authoritative_k1_person_rows.jsonl")
    k1_cache = torch.load(
        G7B / "_tmp/embeddings/k1_target_official_resnet18_embeddings.pt", map_location="cpu", weights_only=True
    )["embeddings"]
    vectors: list[torch.Tensor] = []
    folds: list[int] = []
    for node in nodes:
        example = str(node["example_uuid"])
        vectors.append(torch.cat((node_cache[example].float(), torch.from_numpy(node_tabular_features(node)))))
        folds.append(int(split["assignment_by_example_uuid"][example]))
    for row in k1_rows:
        vectors.append(
            torch.cat(
                (
                    k1_cache[row["example_uuid"]].float(),
                    torch.from_numpy(k1_crop_features(Path(row["crop_path"]), row["target_bbox_source_pixels"])),
                )
            )
        )
        folds.append(int(row["fold"]))
    features = torch.stack(vectors).detach().float()
    if features.shape != (2940, 544):
        raise RuntimeError("FAIL_G7D_B1_FOLD_ARTIFACT_CHAIN: feature matrix")
    calibration_path = G7B / "10_CALIBRATION_AND_SELECTIVE_ROUTING/nested_calibration_receipts.json"
    if sha256_file(calibration_path) != "40512278904edbeec8e42fff58141d3fdf5cf5cc94bd0ec7f840d8b7bc8d13ed":
        raise RuntimeError("FAIL_G7D_B1_FOLD_ARTIFACT_CHAIN: calibration receipt")
    receipt = read_json(calibration_path)
    n3_receipts = {row["outer_fold"]: row for row in receipt["receipts"]["N3"]}
    artifacts: list[FoldArtifact] = []
    registry_rows = []
    for fold in FOLD_ORDER:
        expected = expected_by_fold[fold]
        index = torch.tensor([i for i, value in enumerate(folds) if value != fold], dtype=torch.long)
        training = features.index_select(0, index)
        mean = training.mean(dim=0)
        std = training.std(dim=0, unbiased=False).clamp_min(1e-5)
        mean_hash, std_hash = stable_hash(mean.tolist()), stable_hash(std.tolist())
        if mean_hash != expected["normalization"]["mean_hash"] or std_hash != expected["normalization"]["std_hash"]:
            raise RuntimeError(f"FAIL_G7D_B1_FOLD_ARTIFACT_CHAIN: fold {fold} normalizer")
        scaler_path = STAGE / f"02_FOLDWISE_RUNTIME/fold_{fold}_normalizer.json"
        write_json(
            scaler_path,
            {
                "schema_version": "football_intelligence.g7d_b1.fold_normalizer.v1",
                "fold_id": fold,
                "fit_row_count": len(index),
                "mean_hash": mean_hash,
                "std_hash": std_hash,
                "mean": mean.tolist(),
                "std": std.tolist(),
            },
        )
        temperatures = {head: float(values["temperature"]) for head, values in n3_receipts[fold]["heads"].items()}
        temperature_path = STAGE / f"02_FOLDWISE_RUNTIME/fold_{fold}_temperatures.json"
        write_json(
            temperature_path,
            {
                "schema_version": "football_intelligence.g7d_b1.fold_temperatures.v1",
                "fold_id": fold,
                "source_receipt_sha256": sha256_file(calibration_path),
                "temperatures": temperatures,
                "abstention_thresholds_included": False,
            },
        )
        checkpoint = Path(expected["checkpoint"]["path"])
        if sha256_file(checkpoint) != expected["checkpoint"]["sha256"]:
            raise RuntimeError(f"FAIL_G7D_B1_FOLD_ARTIFACT_CHAIN: fold {fold} checkpoint")
        item = FoldArtifact(
            fold,
            checkpoint,
            expected["checkpoint"]["sha256"],
            scaler_path,
            sha256_file(scaler_path),
            temperature_path,
            sha256_file(temperature_path),
            tuple(expected["training_groups"]),
            tuple(expected["excluded_outer_groups"]),
        )
        artifacts.append(item)
        registry_rows.append(
            {
                "fold_id": fold,
                "checkpoint": artifact(checkpoint, f"n3_fold_{fold}_checkpoint", "G7B", "frozen N3 state"),
                "scaler": artifact(
                    scaler_path, f"fold_{fold}_normalizer", "G7D-B1", "exact materialized outer-training normalizer"
                ),
                "temperature": artifact(
                    temperature_path, f"fold_{fold}_temperatures", "G7D-B1", "exact retained N4 temperatures"
                ),
                "mean_hash": mean_hash,
                "std_hash": std_hash,
                "training_groups": list(item.training_groups),
                "excluded_outer_groups": list(item.excluded_outer_groups),
            }
        )
    registry = {
        "schema_version": "football_intelligence.g7d_b1.fold_artifact_registry.v1",
        "fold_order": list(FOLD_ORDER),
        "rows": registry_rows,
        "source_calibration_receipt": artifact(
            calibration_path, "nested_calibration_receipt", "G7B", "N4 temperatures"
        ),
    }
    write_json(STAGE / "02_FOLDWISE_RUNTIME/fold_artifact_registry.json", registry)
    compatibility = {
        "schema_version": "football_intelligence.g7d_b1.fold_compatibility_report.v1",
        "passed": True,
        "architecture": "HierarchicalSoftConditioningNodeModel",
        "input_dimension": 544,
        "hidden_dimension": 64,
        "head_class_order": audit["folds"][0]["heads"],
        "fold_order": list(FOLD_ORDER),
        "checkpoint_shapes_compatible": True,
        "fold_specific_scalers_expected": True,
        "fold_specific_temperatures_expected": True,
    }
    write_json(STAGE / "02_FOLDWISE_RUNTIME/fold_compatibility_report.json", compatibility)
    contract = {
        "schema_version": "football_intelligence.g7d_b1.foldwise_runtime_contract.v1",
        "contract_id": RUNTIME_CONTRACT_ID,
        "parent_contract_id": PARENT_CONTRACT_ID,
        "fold_order": list(FOLD_ORDER),
        "aggregation": "NONE",
        "primary_fold": None,
        "calibration": "MATCHING_FOLD_HEAD_TEMPERATURE_ONLY",
        "abstention_thresholds": "EXCLUDED",
        "p2": "DISABLED_REQUIRES_AUTHORIZED_STATE_REBUILD",
        "p3": "DISABLED_REQUIRES_AUTHORIZED_STATE_REBUILD",
        "h0_h3": "DISABLED",
        "selector": "DISABLED",
        "production_ready": False,
    }
    write_json(STAGE / "02_FOLDWISE_RUNTIME/foldwise_runtime_contract.json", contract)
    core_manifest = {
        "schema_version": "football_intelligence.g7d_b1.runtime_core_manifest.v1",
        "contract_id": RUNTIME_CONTRACT_ID,
        "parent_contract_id": PARENT_CONTRACT_ID,
        "classification": CLASSIFICATION,
        "proposal_dependency_registry_sha256": sha256_file(
            STAGE / "01_PROPOSAL_CLOSURE/proposal_dependency_registry.json"
        ),
        "proposal_runtime_contract_sha256": sha256_file(STAGE / "01_PROPOSAL_CLOSURE/proposal_runtime_contract.json"),
        "fold_artifact_registry_sha256": sha256_file(STAGE / "02_FOLDWISE_RUNTIME/fold_artifact_registry.json"),
        "fold_compatibility_report_sha256": sha256_file(STAGE / "02_FOLDWISE_RUNTIME/fold_compatibility_report.json"),
        "foldwise_runtime_contract_sha256": sha256_file(STAGE / "02_FOLDWISE_RUNTIME/foldwise_runtime_contract.json"),
        "runtime_code_sha256": sha256_file(REPO / "src/football_intelligence/g7d_b1_foldwise_runtime.py"),
        "orchestrator_code_sha256": sha256_file(REPO / "scripts/g7d_b1_build_and_smoke_foldwise_runtime.py"),
        "fold_order": list(FOLD_ORDER),
        "aggregation": "NONE",
        "p2": "DISABLED_REQUIRES_AUTHORIZED_STATE_REBUILD",
        "p3": "DISABLED_REQUIRES_AUTHORIZED_STATE_REBUILD",
        "h0_h3": "DISABLED",
        "production_ready": False,
    }
    write_json(
        STAGE / "02_FOLDWISE_RUNTIME/frozen_unseen_match_runtime_core_manifest.json",
        core_manifest,
    )
    return artifacts


def load_historical_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prior_from_payload(payload: Mapping[str, Any]) -> RobustPerspectivePrior:
    residual = payload["residual_scales"]
    quantiles = payload["residual_quantiles_10_90"]
    return RobustPerspectivePrior(
        image_width=int(payload["image_width"]),
        image_height=int(payload["image_height"]),
        height_coefficients=tuple(payload["height_coefficients"]),
        width_coefficients=tuple(payload["width_coefficients"]),
        aspect_coefficients=tuple(payload["aspect_coefficients"]),
        residual_scales=tuple(float(residual[name]) for name in ("height", "width", "aspect")),
        residual_quantiles=tuple(tuple(float(v) for v in quantiles[name]) for name in ("height", "width", "aspect")),
        view_offsets=tuple(
            sorted((str(key), tuple(float(v) for v in values)) for key, values in payload["view_offsets"].items())
        ),
        reliable_training_row_count=int(payload["reliable_training_row_count"]),
        rejected_training_row_count=int(payload["rejected_training_row_count"]),
        training_row_hash=str(payload["training_row_hash"]),
        ridge=float(payload["ridge"]),
        huber_delta=float(payload["huber_delta"]),
    )


def smoke_frames() -> list[dict[str, Any]]:
    rows = []
    expected = {
        "118575": (718.504, "8277afd8c85092ce878cab356d391890a1bc8f98c0b51c27297105acfa6c7878"),
        "117092": (673.751, "d1cc9b1e8b89129f370b28f31cdd9a075d0f4f6fbfea2089785f53859cd81eef"),
    }
    for match_id, (timestamp, frame_hash) in expected.items():
        path = G7D_A / f"06_PITCH_POLYGON_REVIEW_PACKAGE/_frames/{match_id}_first.png"
        if not path.is_file() or sha256_file(path) != frame_hash:
            raise RuntimeError("FAIL_G7D_B1_SMOKE_FRAME_PROVENANCE")
        with Image.open(path) as image:
            width, height = image.size
        rows.append(
            {
                "match_id": match_id,
                "half": 1,
                "timestamp_seconds": timestamp,
                "path": str(path),
                "byte_size": path.stat().st_size,
                "width": width,
                "height": height,
                "sha256": frame_hash,
            }
        )
    return rows


def smoke(artifacts: list[FoldArtifact]) -> None:
    receipt_path = STAGE / "03_SMOKE_RUNTIME/smoke_execution_receipt.json"
    if receipt_path.exists():
        raise RuntimeError("successful smoke already exists; rerun forbidden")
    frames = smoke_frames()
    runtime_manifest_sha256 = sha256_file(STAGE / "02_FOLDWISE_RUNTIME/frozen_unseen_match_runtime_core_manifest.json")
    write_json(
        STAGE / "03_SMOKE_RUNTIME/smoke_frame_manifest.json",
        {
            "schema_version": "football_intelligence.g7d_b1.smoke_frame_manifest.v1",
            "frame_count": 2,
            "frames": frames,
            "adaptive_resampling": False,
        },
    )
    os.environ.setdefault("YOLO_CONFIG_DIR", str(STAGE / "_tmp/ultralytics_config"))
    torch.use_deterministic_algorithms(True, warn_only=True)
    g0 = load_historical_module("g7d_b1_g0", REPO / "scripts/build_m5_5g0_detection_forensics.py")
    g6e = load_historical_module("g7d_b1_g6e", REPO / "scripts/build_m5_5g6e_c0_reintegration.py")
    runtime = FrozenFoldwiseRuntime(artifacts, device=torch.device("cuda:0"))
    encoder = (
        FrozenTorchvisionEncoder.from_official_weights(
            "resnet18", weights_identifier="IMAGENET1K_V1", progress=False, l2_normalize=True
        )
        .to(torch.device("cuda:0"))
        .eval()
    )
    prior = prior_from_payload(
        read_json(G7A / "06_PERSPECTIVE_AND_SCALE_PRIOR/global_descriptive_perspective_prior.json")
    )
    frame_records, candidate_records, visual_rows = [], [], []
    for frame in frames:
        started = time.perf_counter()
        match_id, frame_hash = frame["match_id"], frame["sha256"]
        temp = STAGE / f"_tmp/proposal/{match_id}"
        temp.mkdir(parents=True, exist_ok=True)
        runner = g0.DiagnosticRunner(temp / "raw.jsonl", temp / "post.jsonl", temp / "nms.jsonl")
        try:
            plan = proposal_view_plan(frame["width"], frame["height"])
            for view in plan:
                runner.run_view(
                    {
                        "image_path": Path(frame["path"]),
                        "image_sha256": frame_hash,
                        "frame_sequence": 0,
                        "timestamp_seconds": frame["timestamp_seconds"],
                    },
                    view_type=view["view_type"],
                    view_suffix=view["view_suffix"],
                    imgsz=view["imgsz"],
                    crop_bounds=view["crop_bounds_panorama_pixels"],
                )
        finally:
            runner.close()
        if not all(
            row.get("status") == "PASS" and row.get("nms_replay_exact") and row.get("coordinate_roundtrip_passed")
            for row in runner.views
        ):
            raise RuntimeError("FAIL_G7D_B1_SMOKE_RUNTIME: proposal view")
        post_rows = read_jsonl(temp / "post.jsonl")
        runtime_by_view = {}
        for row in runner.views:
            normalized = dict(row)
            normalized["c0_family"] = row["inference_view_type"]
            normalized["cache_provider"] = "G7D_B1_UNSEEN_EXACT"
            runtime_by_view[row["inference_view_id"]] = normalized
        normalized_post = []
        for row in post_rows:
            if row["inference_view_type"] not in {"S0_FULL_PANORAMA_1280", "S3_OVERLAPPING_HIGH_RESOLUTION_TILES"}:
                continue
            normalized = dict(row)
            normalized["c0_family"] = row["inference_view_type"]
            normalized["cache_provider"] = "G7D_B1_UNSEEN_EXACT"
            normalized_post.append(normalized)
        proposal_nodes = g6e.proposal_nodes({frame_hash: normalized_post}, runtime_by_view)[frame_hash]
        consolidated = consolidate_proposals(proposal_nodes, "IOU_CONNECTED_COMPONENT_055", apply_merged_gate=True)
        observations = sorted(consolidated["observations"], key=lambda row: row["observation_uuid"])
        # Versioned hook boundary: consolidation -> disabled pitch gate -> features.
        observations, _, _ = apply_shadow_hook(observations)
        with Image.open(frame["path"]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        source_tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        polygon_payload = read_json(PROJECT / f"matches/{match_id}/calibration/pitch_polygon_v1/pitch_polygon.json")
        polygon = [{"x": float(x), "y": float(y)} for x, y in polygon_payload["vertices_source_xy"]]
        candidates = []
        for observation in observations:
            views = tuple(str(value) for value in observation.get("all_source_view_ids", ()))
            candidates.append(
                {
                    "candidate_uuid": observation["observation_uuid"],
                    "visible_box": observation["box_panorama_pixels"],
                    "score": observation["score"],
                    "proposal_family": "G6E_C0_FROZEN_OBSERVATION",
                    "proposal_stage": "C0_" + observation["output_state"],
                    "source_view": views[0] if views else "UNKNOWN",
                    "source_view_ids": views,
                    "proposal_lineage": tuple(observation["cluster_member_proposal_uuids"]),
                    "duplicate_cluster_size": len(observation["cluster_member_proposal_uuids"]),
                    "cross_view_corroboration_count": len(views),
                }
            )
        for ordinal, (observation, candidate) in enumerate(zip(observations, candidates, strict=True)):
            bundle = extract_candidate_feature_families(
                candidate,
                source_rgb=rgb,
                frame_width=frame["width"],
                frame_height=frame["height"],
                pitch_polygon=polygon,
                neighbours=candidates,
                perspective_prior=prior,
            )
            crop_spec = deterministic_candidate_crop_boxes(
                candidate["visible_box"], image_width=frame["width"], image_height=frame["height"]
            )
            crop = (
                crop_tensor_from_box(source_tensor, crop_spec["crops"]["context"], output_size=(224, 224))
                .unsqueeze(0)
                .to("cuda:0")
                .float()
                .div_(255.0)
            )
            with torch.inference_mode():
                embedding = encoder(crop)[0].detach().cpu().float()
            box = candidate["visible_box"]
            centre_x, centre_y = (box["x1"] + box["x2"]) / 2.0, (box["y1"] + box["y2"]) / 2.0
            node = {
                "source_coordinates": {
                    "image_width": frame["width"],
                    "image_height": frame["height"],
                    "centre_x_normalized": centre_x / frame["width"],
                    "centre_y_normalized": centre_y / frame["height"],
                },
                "visible_box": box,
                "score": candidate["score"],
                "proposal_provenance_features": bundle["proposal_provenance_features"],
                "expected_scale_features": bundle["expected_scale_features"],
                "shape_features": bundle["shape_features"],
                "colour_kit_features": bundle["colour_kit_features"],
            }
            raw_features = torch.cat((embedding, torch.from_numpy(node_tabular_features(node)))).float()
            fold_outputs = runtime.run_all_folds(raw_features)
            local_id = frame_local_candidate_id(frame_hash, ordinal)
            pitch = bundle["pitch_context_features"]
            record = {
                "schema_version": "football_intelligence.g7d_b1.smoke_candidate.v1",
                "runtime_contract_id": RUNTIME_CONTRACT_ID,
                "runtime_manifest_sha256": runtime_manifest_sha256,
                "match_id": match_id,
                "half": 1,
                "timestamp_seconds": frame["timestamp_seconds"],
                "frame_sha256": frame_hash,
                "candidate_local_id": local_id,
                "source_box_xyxy": [box[key] for key in ("x1", "y1", "x2", "y2")],
                "approximate_footpoint_xy": [
                    observation["footpoint_proxy_panorama_pixels"]["x"],
                    observation["footpoint_proxy_panorama_pixels"]["y"],
                ],
                "footpoint_method": "CANDIDATE_BOX_BOTTOM_CENTRE_PROXY",
                "pitch_state": pitch["pitch_relation"],
                "proposal_provenance": {
                    "observation_uuid": observation["observation_uuid"],
                    "score": observation["score"],
                    "output_state": observation["output_state"],
                    "cluster_member_count": len(observation["cluster_member_proposal_uuids"]),
                    "source_views": list(views),
                    "provenance_hash": observation["provenance_hash"],
                },
                "shared_feature_provenance": {
                    "encoder_provenance_hash": encoder.provenance["provenance_hash"],
                    "crop_transform_hash": crop_spec["crop_transform_hash"],
                    "raw_feature_hash": stable_hash(raw_features.tolist()),
                    "perspective_prior_hash": read_json(
                        G7A / "06_PERSPECTIVE_AND_SCALE_PRIOR/global_descriptive_perspective_prior.json"
                    )["prior_hash"],
                },
                "fold_outputs": fold_outputs,
                "p2_status": "DISABLED_REQUIRES_AUTHORIZED_STATE_REBUILD",
                "p3_status": "DISABLED_REQUIRES_AUTHORIZED_STATE_REBUILD",
                "selector_status": "DISABLED",
                "production_ready": False,
            }
            validate_candidate_record(record)
            candidate_records.append(record)
        frame_records.append(
            {
                "schema_version": "football_intelligence.g7d_b1.smoke_frame.v1",
                "match_id": match_id,
                "half": 1,
                "timestamp_seconds": frame["timestamp_seconds"],
                "frame_sha256": frame_hash,
                "proposal_view_count": len(runner.views),
                "raw_consolidation_input_count": len(proposal_nodes),
                "candidate_count": len(observations),
                "five_fold_complete_candidate_count": len(observations),
                "runtime_seconds": round(time.perf_counter() - started, 6),
                "peak_allocated_vram_mib": max(float(row["peak_allocated_vram_mib"]) for row in runner.views),
                "all_views_exact": True,
            }
        )
        visual_rows.append((frame, observations, polygon))
    write_jsonl(STAGE / "03_SMOKE_RUNTIME/smoke_frame_records.jsonl", frame_records)
    write_jsonl(STAGE / "03_SMOKE_RUNTIME/smoke_candidate_records.jsonl", candidate_records)
    summary = {
        "schema_version": "football_intelligence.g7d_b1.smoke_summary.v1",
        "frame_count": 2,
        "inference_passes_per_frame": 1,
        "proposal_counts": {row["match_id"]: row["raw_consolidation_input_count"] for row in frame_records},
        "candidate_counts": {row["match_id"]: row["candidate_count"] for row in frame_records},
        "candidate_count": len(candidate_records),
        "all_candidates_have_five_folds": all(len(row["fold_outputs"]) == 5 for row in candidate_records),
        "aggregation_performed": False,
        "adaptive_resampling": False,
        "accuracy_claimed": False,
        "passed": True,
    }
    write_json(STAGE / "03_SMOKE_RUNTIME/smoke_runtime_summary.json", summary)
    write_json(
        receipt_path,
        {
            "frame_hashes": [row["sha256"] for row in frames],
            "real_frame_count": 2,
            "passes_per_frame": 1,
            "completed": True,
        },
    )
    draw_visual(visual_rows)


def draw_visual(rows: list[tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, float]]]]) -> None:
    panels = []
    for frame, observations, polygon in rows:
        with Image.open(frame["path"]) as image:
            panel = image.convert("RGB")
        scale = 1200 / panel.width
        panel = panel.resize((1200, round(panel.height * scale)))
        draw = ImageDraw.Draw(panel)
        points = [(point["x"] * scale, point["y"] * scale) for point in polygon]
        draw.line(points + [points[0]], fill="#00ff88", width=3)
        for ordinal, observation in enumerate(observations):
            box = observation["box_panorama_pixels"]
            xy = tuple(round(box[key] * scale) for key in ("x1", "y1", "x2", "y2"))
            draw.rectangle(xy, outline="#ffd54a", width=2)
            foot = observation["footpoint_proxy_panorama_pixels"]
            x, y = round(foot["x"] * scale), round(foot["y"] * scale)
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#ff5c8a")
            draw.text((xy[0], max(0, xy[1] - 12)), f"C{ordinal:03d}", fill="white")
        draw.rectangle((0, 0, panel.width, 44), fill="#111111")
        draw.text(
            (12, 5), f"{frame['match_id']}  ENGINEERING SMOKE - NOT GROUND TRUTH  |  folds 0-4 complete", fill="white"
        )
        panels.append(panel)
    width, height = max(p.width for p in panels), sum(p.height for p in panels)
    output = Image.new("RGB", (width, height), "black")
    y = 0
    for panel in panels:
        output.paste(panel, (0, y))
        y += panel.height
    path = STAGE / "04_VISUAL_QA/two_frame_foldwise_runtime_smoke.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path, optimize=True)


def freeze_manifest(repository_commit: str) -> None:
    proposal_registry = STAGE / "01_PROPOSAL_CLOSURE/proposal_dependency_registry.json"
    proposal_contract = STAGE / "01_PROPOSAL_CLOSURE/proposal_runtime_contract.json"
    fold_registry = read_json(STAGE / "02_FOLDWISE_RUNTIME/fold_artifact_registry.json")
    smoke_paths = [
        STAGE / "03_SMOKE_RUNTIME/smoke_frame_manifest.json",
        STAGE / "03_SMOKE_RUNTIME/smoke_frame_records.jsonl",
        STAGE / "03_SMOKE_RUNTIME/smoke_candidate_records.jsonl",
        STAGE / "03_SMOKE_RUNTIME/smoke_runtime_summary.json",
    ]
    manifest = {
        "schema_version": "football_intelligence.g7d_b1.frozen_runtime_manifest.v1",
        "contract_id": RUNTIME_CONTRACT_ID,
        "parent_contract_id": PARENT_CONTRACT_ID,
        "classification": CLASSIFICATION,
        "proposal_dependency_registry_sha256": sha256_file(proposal_registry),
        "proposal_runtime_contract_sha256": sha256_file(proposal_contract),
        "runtime_core_manifest_sha256": sha256_file(
            STAGE / "02_FOLDWISE_RUNTIME/frozen_unseen_match_runtime_core_manifest.json"
        ),
        "detector_sha256": DETECTOR_SHA256,
        "fold_order": list(FOLD_ORDER),
        "fold_artifacts": [
            {
                "fold_id": row["fold_id"],
                "checkpoint_sha256": row["checkpoint"]["sha256"],
                "scaler_sha256": row["scaler"]["sha256"],
                "temperature_sha256": row["temperature"]["sha256"],
            }
            for row in fold_registry["rows"]
        ],
        "ontology_class_order_hash": stable_hash(
            read_json(STAGE / "02_FOLDWISE_RUNTIME/fold_compatibility_report.json")["head_class_order"]
        ),
        "runtime_code": artifact(
            REPO / "src/football_intelligence/g7d_b1_foldwise_runtime.py",
            "foldwise_runtime_code",
            "G7D-B1",
            "reusable frozen runtime",
        ),
        "orchestrator_code": artifact(
            REPO / "scripts/g7d_b1_build_and_smoke_foldwise_runtime.py",
            "b1_orchestrator",
            "G7D-B1",
            "closure, materialization, smoke and freeze",
        ),
        "smoke_artifacts": [artifact(path, path.stem, "G7D-B1", "bounded smoke evidence") for path in smoke_paths],
        "smoke_frame_hashes": [row["sha256"] for row in smoke_frames()],
        "excluded_components": [
            "P2",
            "P3",
            "H0",
            "H1",
            "H2",
            "H3",
            "FAILED_HIERARCHICAL_SELECTOR",
            "SUPPRESSION_AFTER_PROPOSAL_CONSOLIDATION",
            "FINAL_OBSERVATION_ACCEPTANCE",
            "IDENTITY",
            "TRACKING",
        ],
        "aggregation": "NONE",
        "repository_baseline": BASELINE,
        "repository_commit": repository_commit,
        "production_ready": False,
        "next_authorized_stage": "G7D_B2_FROZEN_128058_BASELINE_RERUN",
    }
    write_json(STAGE / "02_FOLDWISE_RUNTIME/frozen_unseen_match_runtime_manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "smoke", "freeze"))
    parser.add_argument("--repository-commit", default=BASELINE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "prepare":
        validate_baseline()
        validate_pack()
        proposal_closure()
        materialize_folds()
    elif args.mode == "smoke":
        if git("rev-parse", "HEAD") != BASELINE:
            raise RuntimeError("FAIL_BASELINE_OR_WORKTREE")
        smoke(materialize_folds())
        freeze_manifest(BASELINE)
    else:
        freeze_manifest(args.repository_commit)
    print(json.dumps({"mode": args.mode, "passed": True, "stage": str(STAGE)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
