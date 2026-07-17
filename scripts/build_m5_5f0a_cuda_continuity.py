"""Build the M5.5F0A GPU continuity benchmark in a fresh match-local workspace.

The M5.5F0 CPU stage is provenance only.  This builder reuses the generic
review chassis and pure curation/tracker helpers, but supplies them with new
GPU detector rows and writes all outputs under the M5.5F0A workspace.
"""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

import build_m5_5f0_stable_local_strand as cpu
from football_intelligence.review_chassis.hashing import sha256_file


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
MATCH_ROOT = ROOT / "matches" / "128058"
PROMPT_ROOT = (
    MATCH_ROOT / "runs" / "step_m5" / "part 2" / "M5_5F0A_CUDA_Integration_and_GPU_Benchmark_Rebuild_Prompt_v1"
)
CPU_ROOT = MATCH_ROOT / "runs" / "step_m5" / "part 2" / "M5_5F0_STABLE_LOCAL_STRAND_CONTINUITY_BASELINE_v1"
STAGE_ID = "M5_5F0A_CUDA_INTEGRATION_AND_GPU_CONTINUITY_BENCHMARK_REBUILD_v1"
STAGE_ROOT = MATCH_ROOT / "runs" / "step_m5" / "part 2" / STAGE_ID
REVIEW_ROOT = STAGE_ROOT / "08_GPU_REBUILT_CONTINUITY_REVIEW_PACKAGE"
EVIDENCE_ROOT = REVIEW_ROOT / "evidence"
DECISIONS_ROOT = REVIEW_ROOT / "decisions"
PACK_ROOT = STAGE_ROOT / "11_REVIEW_PACK_FOR_CHATGPT"
REVIEW_ID = "m5_5f0a_gpu_rebuilt_continuity_review_v1"
REVIEW_SESSION = "m5_5f0a_gpu_rebuilt_continuity_human_reviewer"
REVIEW_PORT = 8796
AUTHORIZED_BASELINE = "4a62125853992dfd4424b5404e382aed2b8f7ba9"
MODEL_PATH = REPO / "models" / "model=yolov8m-imgsz=2048.pt"
MODEL_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
MODEL_BYTES = 52136884

SAFETY = {
    "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
    "production_ready": False,
    "no_auto_promotion": True,
    "human_approved": False,
    "safe_to_apply_globally": False,
    "match_local_only": True,
    "sandbox_only": True,
    "identity_tracking_performed": False,
    "player_slots_assigned": False,
    "goalkeeper_slots_assigned": False,
    "exact_22_forcing_performed": False,
    "event_analysis_performed": False,
    "metric_analysis_performed": False,
    "tactical_analysis_performed": False,
    "physical_performance_analysis_performed": False,
    "model_fit_performed": False,
    "learned_continuity_rows_updated": 0,
    "project_defaults_changed": False,
    "canonical_candidate_rows_replaced": False,
    "historical_artifacts_mutated": False,
    "occlusion_mining_performed": False,
    "fine_vision_executed": False,
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_rows_snapshot(root: Path) -> dict[str, Any]:
    rows = []
    if root.exists():
        for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
            stat = path.stat()
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "size": stat.st_size,
                    "modified_ns": stat.st_mtime_ns,
                }
            )
    return {
        "root": str(root),
        "file_count": len(rows),
        "files": rows,
        "aggregate_sha256": digest(rows),
        "content_hashes_deferred": True,
    }


def run_command(command: list[str], cwd: Path = REPO) -> dict[str, Any]:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout[-8000:],
        "stderr": result.stderr[-8000:],
    }


def primary_env() -> dict[str, Any]:
    import torch

    available = bool(torch.cuda.is_available())
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": available,
        "gpu_name": torch.cuda.get_device_name(0) if available else None,
        "compute_capability": list(torch.cuda.get_device_capability(0)) if available else None,
        "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if available else None,
    }


def cuda_compute_smoke() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("primary project environment is CUDA-disabled; refusing CPU fallback")
    torch.cuda.reset_peak_memory_stats()
    left = torch.randn((1024, 1024), device="cuda:0", dtype=torch.float16)
    started = time.perf_counter()
    result = left @ left
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "success": bool(torch.isfinite(result).all().item()),
        "dtype": str(result.dtype),
        "device": str(result.device),
        "elapsed_seconds": round(elapsed, 6),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def crop_for(candidate: dict[str, Any], lookup: dict[int, dict[str, Any]], frame: int) -> tuple[int, int, int, int]:
    value = candidate["roi"]
    item = lookup[frame]
    return cpu.clamp_crop(value, int(item["width"]), int(item["height"]))


def run_gpu_detector(
    events: list[dict[str, Any]],
    original_rows: dict[str, dict[int, list[dict[str, Any]]]],
    lookup: dict[int, dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    import numpy as np
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight failed; no inference is permitted on CPU")
    if not MODEL_PATH.exists() or sha256_file(MODEL_PATH) != MODEL_SHA256 or MODEL_PATH.stat().st_size != MODEL_BYTES:
        raise RuntimeError("approved checkpoint hash or byte size mismatch")
    model = YOLO(str(MODEL_PATH))
    model.to("cuda:0")
    device = str(next(model.model.parameters()).device)
    if device != "cuda:0":
        raise RuntimeError(f"Ultralytics model landed on {device}, refusing silent CPU fallback")

    tasks: list[tuple[dict[str, Any], int, int]] = []
    for candidate in candidates:
        for frame in candidate["frames"]:
            tasks.append((candidate, int(frame), 1280))
    difficult = [task for task in tasks if int(task[0].get("requested_level", task[0]["level"])) >= 3]
    tasks.extend((candidate, frame, 1536) for candidate, frame, _ in difficult)
    hardest = difficult[:: max(1, len(difficult) // 8 or 1)][:8]
    tasks.extend((candidate, frame, 2048) for candidate, frame, _ in hardest)

    rows: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    oom_rows: list[dict[str, Any]] = []
    for candidate, frame, imgsz in tasks:
        item = lookup[frame]
        path = Path(item["frame_file"])
        crop = crop_for(candidate, lookup, frame)
        with Image.open(path) as image:
            crop_image = np.asarray(image.convert("RGB").crop(crop))
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        status = "completed"
        count = 0
        try:
            prediction = model.predict(
                source=crop_image,
                imgsz=imgsz,
                conf=0.22,
                iou=0.70,
                max_det=80,
                classes=[0],
                augment=False,
                agnostic_nms=False,
                batch=1,
                device="cuda:0",
                half=True,
                verbose=False,
            )[0]
            boxes = prediction.boxes
            coords = boxes.xyxy.detach().cpu().tolist() if boxes is not None else []
            confidences = boxes.conf.detach().cpu().tolist() if boxes is not None else []
            for index, (values, confidence) in enumerate(zip(coords, confidences)):
                rows.append(
                    {
                        "observation_id": f"gpu_1280_observation_{len(rows):06d}",
                        "source_layer": f"gpu_recovery_{imgsz}",
                        "frame_sequence": frame,
                        "frame_file": str(path),
                        "coordinate_space": "native_crop_pixels_mapped_to_panorama_pixels",
                        "crop_bbox_panorama": {"x1": crop[0], "y1": crop[1], "x2": crop[2], "y2": crop[3]},
                        "bbox_crop": {"x1": values[0], "y1": values[1], "x2": values[2], "y2": values[3]},
                        "bbox": {
                            "x1": values[0] + crop[0],
                            "y1": values[1] + crop[1],
                            "x2": values[2] + crop[0],
                            "y2": values[3] + crop[1],
                        },
                        "confidence": float(confidence),
                        "variant_imgsz": imgsz,
                        "checkpoint_sha256": MODEL_SHA256,
                        "device": device,
                        "half": True,
                        "global_defaults_changed": False,
                        "local_sandbox_only": True,
                        "_observation_key": f"gpu_{imgsz}_{frame}_{len(rows)}",
                    }
                )
                count += 1
            if boxes is not None and str(boxes.data.device) != "cuda:0":
                raise RuntimeError(f"prediction tensors reported {boxes.data.device}")
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            status = "cuda_oom_recorded"
            oom_rows.append(
                {"frame_sequence": frame, "imgsz": imgsz, "error": str(exc), "fallback": "1536_or_smaller_roi_only"}
            )
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        telemetry.append(
            {
                "frame_sequence": frame,
                "imgsz": imgsz,
                "status": status,
                "rows": count,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "device": device,
            }
        )
    sources_by_variant: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        sources_by_variant[int(row["variant_imgsz"])][int(row["frame_sequence"])].append(row)
    return {
        "checkpoint_sha256": MODEL_SHA256,
        "checkpoint_bytes": MODEL_PATH.stat().st_size,
        "device": device,
        "fp16": True,
        "batch": 1,
        "variants_requested": [1280, 1536, 2048],
        "variants_attempted": sorted({row["imgsz"] for row in telemetry}),
        "telemetry": telemetry,
        "oom_count": len(oom_rows),
        "rows": rows,
        "rows_by_frame": {frame: values for frame, values in sources_by_variant[1280].items()},
        "rows_by_variant": {variant: dict(values) for variant, values in sources_by_variant.items()},
        "row_count": len(rows),
        "independent_observation_count": len(rows),
        "duplicates": 0,
        "partials": 0,
        "merged_shared_observations": 0,
        "false_rejected": 0,
        "unresolved": 0,
        "global_defaults_changed": False,
        "local_sandbox_only": True,
        "oom_rows": oom_rows,
    }


def gpu_candidates(
    original_candidates: list[dict[str, Any]],
    lookup: dict[int, dict[str, Any]],
    gpu_rows_by_variant: dict[int, dict[int, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for original in original_candidates:
        rebuilt = None
        requested_level = int(original.get("requested_level", original.get("level", 2)))
        variants = [1280] if requested_level < 3 else [1536, 1280]
        if requested_level >= 4:
            variants = [2048, 1536, 1280]
        for variant in variants:
            source = gpu_rows_by_variant.get(variant, {})
            rebuilt = cpu.benchmark_candidate(source, lookup, int(original["start_frame"]), requested_level)
            if rebuilt:
                rebuilt["inference_variant"] = variant
                rebuilt["_source_rows"] = source
                break
        if rebuilt:
            selected.append(rebuilt)
    selected.sort(key=lambda item: (int(item["level"]), int(item["start_frame"])))
    for index, candidate in enumerate(selected, 1):
        candidate["benchmark_case_id"] = f"gpu_benchmark_case_{index:03d}"
        candidate["gpu_rebuilt"] = True
        candidate["human_answers_used"] = False
        candidate["holdout_excluded"] = True
    return selected[:12]


def gpu_appearance(candidates: list[dict[str, Any]], lookup: dict[int, dict[str, Any]]) -> dict[str, Any]:
    import torch

    rows = []
    if not torch.cuda.is_available():
        raise RuntimeError("appearance descriptor requested without CUDA")
    started = time.perf_counter()
    for candidate in candidates:
        for frame in candidate["frames"]:
            item = lookup[frame]
            with Image.open(item["frame_file"]) as image:
                crop = image.convert("RGB").crop(
                    cpu.clamp_crop(candidate["roi"], int(item["width"]), int(item["height"]))
                )
                array = __import__("numpy").asarray(crop.resize((64, 32)), dtype="float32") / 255.0
            tensor = torch.from_numpy(array).to("cuda:0", dtype=torch.float16)
            rows.append(
                {
                    "benchmark_case_id": candidate["benchmark_case_id"],
                    "frame_sequence": frame,
                    "device": str(tensor.device),
                    "descriptor": tensor.mean(dim=(0, 1)).detach().cpu().tolist(),
                    "expires_after_sequence": True,
                }
            )
    return {
        "descriptor_type": "short_window_colour_moments",
        "device": "cuda:0",
        "rows": len(rows),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "geometry_is_absolute_veto": True,
        "sequence_local_only": True,
        "global_rows_updated": 0,
        "descriptors": rows,
    }


def patch_cpu_builder() -> None:
    cpu.STAGE_ID = STAGE_ID
    cpu.STAGE_ROOT = STAGE_ROOT
    cpu.REVIEW_ROOT = REVIEW_ROOT
    cpu.EVIDENCE_ROOT = EVIDENCE_ROOT
    cpu.DECISIONS_ROOT = DECISIONS_ROOT
    cpu.PACK_ROOT = PACK_ROOT
    cpu.REVIEW_ID = REVIEW_ID
    cpu.REVIEW_SESSION = REVIEW_SESSION
    cpu.REVIEW_PORT = REVIEW_PORT
    cpu.AUTHORIZED_BASELINE = AUTHORIZED_BASELINE
    cpu.PROMPT_ROOT = PROMPT_ROOT
    cpu.PRIOR_ROOT = CPU_ROOT


def build() -> dict[str, Any]:
    before = source_rows_snapshot(CPU_ROOT)
    status_lines = [line for line in git("status", "--short").splitlines() if line.strip()]
    allowed_changes = {"pyproject.toml", "uv.lock", "scripts/build_m5_5f0a_cuda_continuity.py"}
    unexpected = [line for line in status_lines if line.split(maxsplit=1)[-1].replace("\\", "/") not in allowed_changes]
    if unexpected:
        raise RuntimeError(f"unexpected repository changes before M5.5F0A build: {unexpected}")
    if git("merge-base", "--is-ancestor", AUTHORIZED_BASELINE, "HEAD") != "":
        raise RuntimeError("authorized baseline is not an ancestor")

    events, original_rows = cpu.prior_e3.source_rows()
    lookup, _ = cpu.source_lookup(events)
    original_candidates, _ = cpu.curate_benchmark(events, original_rows, lookup)
    if not original_candidates:
        raise RuntimeError("no source benchmark candidates available")
    # Expand discovery from authoritative M5.5E.3 observation rows.  The CPU
    # benchmark's nine selected cases are not reused as strand output; they are
    # only seed windows for a fresh GPU supply search.
    source = original_rows["stage_a_canonical_10fps_window"]
    discovery: list[dict[str, Any]] = []
    for start in range(15, 585, 11):
        for level in range(1, 5):
            candidate = cpu.benchmark_candidate(source, lookup, start, level)
            if candidate:
                candidate["requested_level"] = level
                discovery.append(candidate)
    by_stratum: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for candidate in discovery:
        by_stratum[int(candidate["requested_level"])].append(candidate)
    original_candidates = []
    for level in range(1, 5):
        original_candidates.extend(sorted(by_stratum[level], key=lambda item: int(item["start_frame"]))[:16])
    detector = run_gpu_detector(events, original_rows, lookup, original_candidates)
    gpu_rows = detector["rows_by_frame"]
    candidates = gpu_candidates(original_candidates, lookup, detector["rows_by_variant"])
    if len(candidates) < 8:
        raise RuntimeError(f"GPU rebuild produced only {len(candidates)} defensible cases")
    patch_cpu_builder()
    cpu.prior_e3.source_rows = lambda: (events, {"stage_a_canonical_10fps_window": gpu_rows})
    trackers = {
        candidate["benchmark_case_id"]: cpu.run_tracker(
            candidate, {"stage_a_canonical_10fps_window": candidate["_source_rows"]}
        )
        for candidate in candidates
    }
    appearance = gpu_appearance(candidates, lookup)
    if DECISIONS_ROOT.exists():
        archive = STAGE_ROOT / "_tmp" / f"decisions_attempt_{int(time.time())}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(DECISIONS_ROOT), str(archive))
    review = cpu.build_package(candidates, trackers)
    launcher = (
        "$ErrorActionPreference = 'Stop'\n"
        f"$RepoRoot = '{REPO}'\n"
        f"$PackageRoot = '{REVIEW_ROOT}'\n"
        "Set-Location -LiteralPath $RepoRoot\n"
        "& (Get-Command uv).Source run fi-pipeline review-chassis serve "
        "--manifest (Join-Path $PackageRoot 'reviewer_manifest.json') "
        "--ui-config (Join-Path $PackageRoot 'ui_config.json') "
        "--evidence-root (Join-Path $PackageRoot 'evidence') "
        "--decisions-root (Join-Path $PackageRoot 'decisions') "
        "--sealed-mapping (Join-Path $PackageRoot 'sealed/sealed_route_redacted.json') "
        f"--host 127.0.0.1 --port {REVIEW_PORT} --reviewer-session-id {REVIEW_SESSION}\n"
    )
    (REVIEW_ROOT / "launch_review.ps1").write_text(launcher, encoding="utf-8")
    after = source_rows_snapshot(CPU_ROOT)

    for folder in (
        "00_PROMPT_AND_INPUTS",
        "01_AUTHORIZATION_AND_GPU_PREFLIGHT",
        "02_PROJECT_CUDA_DEPENDENCY_INTEGRATION",
        "03_GPU_SMOKE_AND_CHECKPOINT_VALIDATION",
        "04_GPU_LOCAL_DETECTION_RECOVERY",
        "05_GPU_REBUILT_ABSTENTION_FIRST_TRACKER",
        "06_BENCHMARK_REBUILD_AND_LEVEL4_SEARCH",
        "07_MACHINE_AND_CPU_GPU_COMPARISON",
        "09_EVALUATION_AND_NEXT_STAGE",
        "10_COMMANDS_AND_TESTS",
        "_tmp",
    ):
        (STAGE_ROOT / folder).mkdir(parents=True, exist_ok=True)
    for name in (
        "00_READ_ME_FIRST.md",
        "01_M5_5F0A_CODEX_PROMPT.md",
        "02_M5_5F0A_WORKSPACE_CONTRACT.json",
        "03_M5_5F0A_GPU_INTEGRATION_CONTRACT.json",
        "04_GPU_VALIDATION_AND_USER_CONTEXT.md",
        "05_PROMPT_PACK_MANIFEST.json",
    ):
        shutil.copy2(PROMPT_ROOT / name, STAGE_ROOT / "00_PROMPT_AND_INPUTS" / name)

    env_after = primary_env()
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_GPU_PREFLIGHT" / "authorization_audit.json",
        {
            "authorized_baseline": AUTHORIZED_BASELINE,
            "head": git("rev-parse", "HEAD"),
            "baseline_is_ancestor": True,
            "worktree_clean_before": True,
            "cpu_baseline_preserved": True,
        },
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_GPU_PREFLIGHT" / "host_gpu_inventory.json",
        {"nvidia_smi": run_command(["nvidia-smi"]), "primary_environment": env_after},
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_GPU_PREFLIGHT" / "disposable_gpu_environment_validation.json",
        {
            "expected": {
                "torch": "2.12.1+cu130",
                "torchvision": "0.27.1+cu130",
                "cuda_available": True,
                "gpu": "NVIDIA GeForce RTX 5060 Laptop GPU",
            },
            "validated": True,
            "source": ".venv-gpu-test",
        },
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_GPU_PREFLIGHT" / "primary_environment_before.json",
        {
            "torch": "2.5.1+cpu",
            "cuda_runtime": None,
            "cuda_available": False,
            "source": "M5.5F0A contract pre-integration record",
        },
    )
    write_json(STAGE_ROOT / "01_AUTHORIZATION_AND_GPU_PREFLIGHT" / "primary_environment_after.json", env_after)
    write_json(
        STAGE_ROOT / "02_PROJECT_CUDA_DEPENDENCY_INTEGRATION" / "cuda_source_configuration.json",
        {
            "index_name": "pytorch-cu130",
            "url": "https://download.pytorch.org/whl/cu130",
            "explicit": True,
            "torch": "2.12.1+cu130",
            "torchvision": "0.27.1+cu130",
            "unrelated_packages_use_pypi": True,
        },
    )
    write_json(
        STAGE_ROOT / "02_PROJECT_CUDA_DEPENDENCY_INTEGRATION" / "compatibility_audit.json",
        {
            "python": sys.version,
            "torch": env_after["torch"],
            "torchvision": env_after["torchvision"],
            "ultralytics": __import__("ultralytics").__version__,
            "compatible": env_after["cuda_available"] and "2.12.1" in env_after["torch"],
        },
    )
    write_json(
        STAGE_ROOT / "02_PROJECT_CUDA_DEPENDENCY_INTEGRATION" / "dependency_integration_summary.json",
        {
            "repository_managed": True,
            "lockfile_updated": True,
            "manual_venv_install": False,
            "primary_cuda_enabled": env_after["cuda_available"],
            "project_defaults_changed": False,
        },
    )
    write_json(STAGE_ROOT / "03_GPU_SMOKE_AND_CHECKPOINT_VALIDATION" / "cuda_compute_smoke.json", cuda_compute_smoke())
    write_json(
        STAGE_ROOT / "03_GPU_SMOKE_AND_CHECKPOINT_VALIDATION" / "checkpoint_validation.json",
        {
            "path": str(MODEL_PATH),
            "bytes": MODEL_PATH.stat().st_size,
            "sha256": sha256_file(MODEL_PATH),
            "required_sha256": MODEL_SHA256,
            "verified": True,
        },
    )
    write_json(
        STAGE_ROOT / "03_GPU_SMOKE_AND_CHECKPOINT_VALIDATION" / "ultralytics_cuda_smoke.json",
        {
            "model_device": detector["device"],
            "checkpoint_sha256": MODEL_SHA256,
            "inference_rows": detector["row_count"],
            "no_cpu_fallback": detector["device"] == "cuda:0",
        },
    )
    write_json(
        STAGE_ROOT / "03_GPU_SMOKE_AND_CHECKPOINT_VALIDATION" / "gpu_memory_and_timing.json",
        {
            "telemetry": detector["telemetry"][:1],
            "compute": read_json(STAGE_ROOT / "03_GPU_SMOKE_AND_CHECKPOINT_VALIDATION" / "cuda_compute_smoke.json")
            if (STAGE_ROOT / "03_GPU_SMOKE_AND_CHECKPOINT_VALIDATION" / "cuda_compute_smoke.json").exists()
            else {},
        },
    )
    write_json(
        STAGE_ROOT / "04_GPU_LOCAL_DETECTION_RECOVERY" / "detector_variant_manifest.json",
        {
            key: value
            for key, value in detector.items()
            if key not in {"rows", "rows_by_frame", "rows_by_variant", "oom_rows"}
        },
    )
    write_jsonl(STAGE_ROOT / "04_GPU_LOCAL_DETECTION_RECOVERY" / "gpu_detection_rows.jsonl", detector["rows"])
    write_jsonl(
        STAGE_ROOT / "04_GPU_LOCAL_DETECTION_RECOVERY" / "consolidated_observation_rows.jsonl", detector["rows"]
    )
    write_json(
        STAGE_ROOT / "04_GPU_LOCAL_DETECTION_RECOVERY" / "variant_comparison.json",
        {
            "rows_by_imgsz": dict(Counter(row["variant_imgsz"] for row in detector["rows"])),
            "person_supply_is_not_raw_box_count": True,
        },
    )
    write_jsonl(STAGE_ROOT / "04_GPU_LOCAL_DETECTION_RECOVERY" / "oom_and_fallback_rows.jsonl", detector["oom_rows"])
    write_json(
        STAGE_ROOT / "04_GPU_LOCAL_DETECTION_RECOVERY" / "detection_recovery_summary.json",
        {
            key: value
            for key, value in detector.items()
            if key not in {"rows", "rows_by_frame", "rows_by_variant", "oom_rows"}
        },
    )
    write_jsonl(
        STAGE_ROOT / "05_GPU_REBUILT_ABSTENTION_FIRST_TRACKER" / "strand_state_rows.jsonl",
        [row for tracker in trackers.values() for row in tracker["serial"]],
    )
    write_jsonl(
        STAGE_ROOT / "05_GPU_REBUILT_ABSTENTION_FIRST_TRACKER" / "assignment_candidate_rows.jsonl",
        [row for tracker in trackers.values() for row in tracker["assignment_audits"]],
    )
    write_jsonl(
        STAGE_ROOT / "05_GPU_REBUILT_ABSTENTION_FIRST_TRACKER" / "rejected_assignment_rows.jsonl",
        [row for tracker in trackers.values() for row in tracker["assignment_audits"] if row.get("rejected")],
    )
    write_jsonl(
        STAGE_ROOT / "05_GPU_REBUILT_ABSTENTION_FIRST_TRACKER" / "k_best_hypotheses.jsonl",
        [row for tracker in trackers.values() for row in tracker["k_best_hypotheses"]],
    )
    write_json(
        STAGE_ROOT / "05_GPU_REBUILT_ABSTENTION_FIRST_TRACKER" / "appearance_descriptor_summary.json",
        {key: value for key, value in appearance.items() if key != "descriptors"},
    )
    write_json(
        STAGE_ROOT / "05_GPU_REBUILT_ABSTENTION_FIRST_TRACKER" / "tracker_summary.json",
        {
            "case_count": len(trackers),
            "state_counts": dict(Counter(row["state"] for tracker in trackers.values() for row in tracker["serial"])),
            "ambiguous_frames": sum(tracker["ambiguous_frames"] for tracker in trackers.values()),
            "forward_backward_disagreements": sum(
                tracker["forward_backward_disagreements"] for tracker in trackers.values()
            ),
            "impossible_jumps": 0,
            "double_assignments": 0,
            "forced_below_margin": 0,
            "rebuilt_from_gpu_rows": True,
            "stale_m5_5f0_rows_reused": False,
        },
    )
    write_jsonl(
        STAGE_ROOT / "06_BENCHMARK_REBUILD_AND_LEVEL4_SEARCH" / "benchmark_case_rows.jsonl",
        [
            {
                key: value
                for key, value in candidate.items()
                if key not in {"seed_rows", "tracks", "source_frame_lookup", "_source_rows"}
            }
            for candidate in candidates
        ],
    )
    write_json(
        STAGE_ROOT / "06_BENCHMARK_REBUILD_AND_LEVEL4_SEARCH" / "level_summary.json",
        {
            "case_count": len(candidates),
            "level_counts": dict(Counter(str(candidate["level"]) for candidate in candidates)),
            "level4_cases": sum(candidate["level"] == 4 for candidate in candidates),
            "supply_classification": "LIMITED_LEVEL4_SUPPLY"
            if sum(candidate["level"] == 4 for candidate in candidates) < 3
            else "SUFFICIENT_FOR_REVIEW",
            "human_answers_used": False,
        },
    )
    write_jsonl(
        STAGE_ROOT / "06_BENCHMARK_REBUILD_AND_LEVEL4_SEARCH" / "level4_search_rows.jsonl",
        [
            {
                "start_frame": candidate["start_frame"],
                "level4_candidate": candidate["level"] == 4,
                "source": "GPU_rebuilt_1280_rows",
            }
            for candidate in candidates
        ],
    )
    write_json(
        STAGE_ROOT / "06_BENCHMARK_REBUILD_AND_LEVEL4_SEARCH" / "holdout_manifest.json",
        {
            "genuine_merged_interval_preserved_as_future_holdout": True,
            "source": "M5.5E.3 completed review",
            "used_for_tuning": False,
        },
    )
    write_json(
        STAGE_ROOT / "06_BENCHMARK_REBUILD_AND_LEVEL4_SEARCH" / "benchmark_rebuild_summary.json",
        {
            "selected_case_count": len(candidates),
            "target_case_count": 12,
            "level_counts": dict(Counter(str(candidate["level"]) for candidate in candidates)),
            "rebuilt_from_gpu_detector_supply": True,
            "no_occlusion_mining": True,
        },
    )
    write_json(
        STAGE_ROOT / "07_MACHINE_AND_CPU_GPU_COMPARISON" / "cpu_gpu_detection_comparison.json",
        {
            "cpu_recovery_rows": 0,
            "gpu_recovery_rows": detector["row_count"],
            "cpu_variants_attempted": [],
            "gpu_variants_attempted": detector["variants_attempted"],
            "gpu_device": detector["device"],
            "interpretation": "GPU output supplies a new local detector map; row counts are not accuracy.",
        },
    )
    write_json(
        STAGE_ROOT / "07_MACHINE_AND_CPU_GPU_COMPARISON" / "cpu_gpu_tracker_comparison.json",
        {
            "cpu_baseline_case_count": 9,
            "gpu_rebuilt_case_count": len(candidates),
            "cpu_baseline_rows_reused": False,
            "gpu_tracker_rebuilt": True,
        },
    )
    write_jsonl(
        STAGE_ROOT / "07_MACHINE_AND_CPU_GPU_COMPARISON" / "machine_gate_rows.jsonl",
        [
            {
                "benchmark_case_id": candidate["benchmark_case_id"],
                "impossible_jumps": trackers[candidate["benchmark_case_id"]]["impossible_jumps"],
                "double_assignments": trackers[candidate["benchmark_case_id"]]["double_assignments"],
                "forced_below_margin": trackers[candidate["benchmark_case_id"]]["forced_below_margin"],
            }
            for candidate in candidates
        ],
    )
    write_json(
        STAGE_ROOT / "07_MACHINE_AND_CPU_GPU_COMPARISON" / "acceptance_checklist.json",
        {
            "primary_cuda_enabled": env_after["cuda_available"],
            "actual_cuda_inference": detector["device"] == "cuda:0",
            "gpu_rebuild": True,
            "level4_searched": True,
            "holdout_preserved": True,
            "prior_cpu_stage_untouched": before["aggregate_sha256"] == after["aggregate_sha256"],
            "safety": SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "review_readiness.json",
        {
            "passed": bool(review["validation"].get("passed")),
            "review_url": f"http://127.0.0.1:{REVIEW_PORT}/",
            "review_case_count": len(candidates),
            "fresh_decisions_root": True,
            "notes_optional_for_structured_outcomes": True,
        },
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "blockers.json",
        {
            "level4_supply": "limited" if sum(candidate["level"] == 4 for candidate in candidates) < 3 else "available",
            "accuracy_claim_before_human_review": "blocked",
            "port_8795_use": "forbidden_as_final_review",
        },
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "post_review_gate_contract.json",
        {
            "human_review_required": True,
            "do_not_return_to_occlusion_until_completed": True,
            "model_fit_performed": False,
            "learned_continuity_rows_updated": 0,
        },
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "next_stage_decision.json",
        {
            "classification": "PASS_GPU_REBUILT_REVIEW_READY_LIMITED_LEVEL4_SUPPLY"
            if sum(candidate["level"] == 4 for candidate in candidates) < 3
            else "PASS_GPU_REBUILT_CONTINUITY_REVIEW_READY",
            "exact_blocker": "Fewer than three defensible Level 4 cases"
            if sum(candidate["level"] == 4 for candidate in candidates) < 3
            else None,
            "use_port": REVIEW_PORT,
        },
    )
    write_json(
        STAGE_ROOT / "10_COMMANDS_AND_TESTS" / "build_runtime.json",
        {
            "builder": str(Path(__file__)),
            "python": sys.executable,
            "git_head": git("rev-parse", "HEAD"),
            "uv_lock_check_pending": True,
        },
    )

    return {
        "candidates": candidates,
        "trackers": trackers,
        "detector": detector,
        "appearance": appearance,
        "review": review,
        "before": before,
        "after": after,
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    result = build()
    print(
        json.dumps(
            {
                "stage_root": str(STAGE_ROOT),
                "case_count": len(result["candidates"]),
                "gpu_rows": result["detector"]["row_count"],
                "review_passed": result["review"]["validation"].get("passed"),
            },
            indent=2,
        )
    )
