"""Finalize the two bounded M5.5G.0 detector-forensics context packs."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageStat

from football_intelligence.detection_forensics import sha256_file, tree_digest, validate_flat_context_pack


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART2 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 2"
STAGE = PART2 / "M5_5G0_PLAYER_BALL_DETECTION_FORENSIC_PROVENANCE_AND_PRO_RESEARCH_HANDOFF_v1"
PRO_PACK = STAGE / "13_PRO_CONTEXT_PACK_FOR_CHATGPT_PRO"
REVIEW_PACK = STAGE / "14_REVIEW_PACK_FOR_CHATGPT"
ORIGINAL_DECISIONS = (
    PART2
    / "M5_5F1A4_SERVER_PERSISTENCE_CRASH_SAFE_GOLD_ANNOTATION_AND_REANNOTATION_ACCELERATION_v1"
    / "07_CRASH_SAFE_GOLD_ANNOTATION_PACKAGE"
    / "decisions"
)
FRESH_DECISIONS = (
    PART2
    / "M5_5F1E_SPENT_HOLDOUT_FORENSICS_ORACLE_REACHABILITY_INVARIANTS_AND_FRESH_CHALLENGE_GOLD_ACQUISITION_v1"
    / "10_FRESH_CHALLENGE_GOLD_ANNOTATION_PACKAGE"
    / "decisions"
)
BASELINE = "a508ef27fd399f824411bc80f51a56ae00c2633d"
ORIGIN = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
EXPECTED_NAMES = (
    "REVIEW_PACK_MANIFEST.json",
    "01_EXECUTIVE_BRIEFING.md",
    "02_REPO_RUNTIME_AND_SAFETY_CONTEXT.json",
    "03_CURRENT_DETECTOR_AND_BALL_PIPELINE.md",
    "04_SOURCE_DIFF.patch",
    "05_CHECKPOINT_ENVIRONMENT_AND_LICENSE_MANIFEST.json",
    "06_RAW_OUTPUT_INSTRUMENTATION_AND_SCHEMA.md",
    "07_PRE_NMS_TO_RENDERER_LINEAGE.json",
    "08_FAILURE_CASE_ROWS.jsonl",
    "09_DUPLICATE_CONSOLIDATION_AUDIT.json",
    "10_MERGED_INSTANCE_AUDIT.json",
    "11_MISSED_PERSON_AUDIT.json",
    "12_OFF_PITCH_AND_BOUNDARY_GATE_AUDIT.json",
    "13_FOOTBALL_BALL_CANDIDATE_AUDIT.json",
    "14_SCALE_TILE_TRANSFORM_AND_RUNTIME_AUDIT.json",
    "15_GOLD_DATASET_AND_CASE_SUPPLY_CONTEXT.json",
    "16_CHATGPT_PRO_RESEARCH_PROMPT.md",
    "17_PLAYER_FAILURE_ATLAS.jpg",
    "18_PRE_POST_NMS_AND_SCALE_ATLAS.jpg",
    "19_BALL_AND_OFF_PITCH_ATLAS.jpg",
)
SAFETY_KEYS = (
    "visual_only_warning",
    "production_ready",
    "no_auto_promotion",
    "human_approved",
    "safe_to_apply_globally",
    "match_local_only",
    "sandbox_only",
    "model_fit_performed",
    "learned_continuity_rows_updated",
    "project_defaults_changed",
    "historical_artifacts_mutated",
    "detector_promoted",
    "tracker_promoted",
)


def run(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=REPO,
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def git(*args: str) -> str:
    return run(["git", *args]).stdout.strip()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON object rows: {path}")
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n")


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def reset_pack(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for child in root.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def count(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key)) for row in rows))


def public_forensic_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "diagnostic_case_uuid": row["case_id"],
        "evidence_level": row["evidence_level"],
        "failure_type": row["failure_type"],
        "human_supported_visible_person_count": row.get("human_supported_visible_person_count"),
        "raw_proposal_count": row["raw_proposal_count"],
        "pre_nms_count": row["pre_nms_count"],
        "post_nms_count": row["post_nms_count"],
        "cross_view_cluster_count": row["cross_view_cluster_count"],
        "pitch_gate_result": row["pitch_gate_result"],
        "final_rendered_count": row["final_rendered_count"],
        "earliest_failure_stage": row["earliest_failure_stage"],
        "diagnosis_confidence": row["diagnosis_confidence"],
        "supporting_asset_paths": row["supporting_asset_paths"],
    }


def protected_snapshot_validation() -> dict[str, Any]:
    snapshot_root = STAGE / "01_AUTHORIZATION_AND_PRIOR_ARTIFACT_HASHES"
    before = read_json(snapshot_root / "prior_artifact_hash_before.json")
    after = read_json(snapshot_root / "prior_artifact_hash_after.json")
    files = []
    for name, expected in before["files"].items():
        safe = str(expected["path"])
        if safe.startswith("<REPOSITORY>/"):
            path = REPO / safe.removeprefix("<REPOSITORY>/")
        elif safe.startswith("<FOOTBALL_INTELLIGENCE_ROOT>/"):
            path = ROOT / safe.removeprefix("<FOOTBALL_INTELLIGENCE_ROOT>/")
        else:
            raise RuntimeError(f"unrecognized protected path token: {safe}")
        actual = {
            "name": name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        actual["matches_snapshot"] = (
            actual["size_bytes"] == expected["size_bytes"] and actual["sha256"] == expected["sha256"]
        )
        files.append(actual)
    tree_roots = {
        "original_completed_decisions": ORIGINAL_DECISIONS,
        "fresh_completed_decisions": FRESH_DECISIONS,
    }
    trees = []
    for name, path in tree_roots.items():
        actual = tree_digest(path)
        expected = before["trees"][name]
        trees.append(
            {
                "name": name,
                "file_count": actual["file_count"],
                "size_bytes": actual["size_bytes"],
                "tree_sha256": actual["tree_sha256"],
                "matches_snapshot": actual == expected,
            }
        )
    passed = all(
        (
            before["snapshot_hash"] == after["snapshot_hash"],
            after["matches_before_snapshot"],
            all(row["matches_snapshot"] for row in files),
            all(row["matches_snapshot"] for row in trees),
        )
    )
    result = {
        "passed": passed,
        "recorded_before_after_match": before["snapshot_hash"] == after["snapshot_hash"],
        "files": files,
        "trees": trees,
    }
    if not passed:
        raise RuntimeError(f"protected snapshot validation failed: {result}")
    return result


def compact_case(row: Mapping[str, Any], *, classification_key: str | None = None) -> dict[str, Any]:
    output = {
        "diagnostic_case_uuid": row["case_id"],
        "evidence_level": row["evidence_level"],
        "source_frame_sha256": row["source_frame_sha256"],
        "frame_sequence": row["frame_sequence"],
        "focal_bbox_original_pixels": row["focal_bbox_original_pixels"],
        "raw_proposal_count": row["raw_proposal_count"],
        "pre_nms_confidence_survivor_count": row["pre_nms_confidence_survivor_count"],
        "post_nms_production_scale_count": row["post_nms_production_scale_count"],
        "post_nms_high_resolution_or_crop_count": row["post_nms_high_resolution_or_crop_count"],
        "cross_view_cluster_count": row["cross_view_cluster_count"],
        "earliest_failure_stage": row["earliest_failure_stage"],
        "diagnosis_confidence": row["diagnosis_confidence"],
    }
    if classification_key:
        output[classification_key] = row.get(classification_key)
    return output


def compact_ball(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "diagnostic_case_uuid": row["case_id"],
        "source_frame_sha256": row["source_frame_sha256"],
        "frame_sequence": row["frame_sequence"],
        "stratum": row["stratum"],
        "stratum_interpretation": row["stratum_interpretation"],
        "classification": row["classification"],
        "top_raw_ball_score": row["top_raw_ball_score"],
        "top_raw_ball_bbox_panorama_pixels": row["top_raw_ball_bbox_panorama_pixels"],
        "post_nms_ball_candidate_count": row["post_nms_ball_candidate_count"],
        "minimum_person_footpoint_distance_pixels": row["minimum_person_footpoint_distance_pixels"],
        "tiny_candidate": row["tiny_candidate"],
        "temporal_context": [
            {key: value for key, value in context.items() if key != "source_asset_path"}
            for context in row["temporal_context"]
        ],
        "human_ball_gold_available": False,
        "visual_ground_truth_required": True,
    }


def validate_repository() -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    origin = git("remote", "get-url", "origin")
    status = git("status", "--porcelain")
    ancestor = run(["git", "merge-base", "--is-ancestor", BASELINE, head], check=False).returncode == 0
    origin_head = git("rev-parse", "origin/main")
    result = {
        "authorized_baseline": BASELINE,
        "implementation_commit": head,
        "branch": branch,
        "origin": origin,
        "origin_main": origin_head,
        "working_tree_clean": not status,
        "baseline_is_ancestor": ancestor,
        "commit_pushed": head == origin_head,
        "protected_artifacts": protected_snapshot_validation(),
    }
    if not all(
        (
            branch == "main",
            origin == ORIGIN,
            not status,
            ancestor,
            head != BASELINE,
            head == origin_head,
        )
    ):
        raise RuntimeError(f"repository finalization gate failed: {result}")
    return result


def executive_briefing(build: Mapping[str, Any], origin: Mapping[str, Any]) -> str:
    ball = build["ball_forensics"]
    return f"""# M5.5G.0 detector forensic briefing

## Outcome

The bounded stock-YOLOv8m forensic stage passed. It instruments the installed
model before confidence filtering, reproduces official NMS exactly, preserves
raw-to-render lineage, and leaves production detector and tracker behavior
unchanged.

## Evidence supply

- 152 player cases across duplicate, merged, missed, off-pitch, small-player,
  partial/occluded and clean-control strata.
- 24 original gold sequences and 32 fresh challenge sequences bound to verified
  2730x720 panorama sources.
- Ball proposal strata: {ball['stratum_counts']}.
- Human ball gold is absent; ball precision and recall were not computed.

## Root-cause signal

The aggregate diagnostic origins are {origin['counts_by_failure_origin']}.
Unresolved rows are intentionally retained because machine-mined rectangles do
not become human truth. The evidence supports an architecture decision, not a
production threshold change.

## Decision requested

Use `16_CHATGPT_PRO_RESEARCH_PROMPT.md` with primary research sources to select
the next bounded architecture and gold-label stage. Do not promote a detector
or tracker from this pack.
"""


def pipeline_document(architecture: str) -> str:
    return (
        architecture
        + """

## Ball branch boundary

The production person invocation remains class-filtered to the runtime-resolved
person class. The sports-ball inspection is a separate low-threshold diagnostic
replay of the same immutable tensor. It creates proposal evidence only and does
not infer possession, passes, shots or events.
"""
    )


def raw_schema_document(schema: Mapping[str, Any], nms: Mapping[str, Any]) -> str:
    example = schema["installed_model_examples"][0]
    return f"""# Raw output instrumentation and schema

The installed Ultralytics/model pair was inspected at runtime; no tensor layout
was assumed in advance.

- Decoded tensor shape: `{example['decoded_tensor_shape']}`
- Raw candidate count: `{example['raw_candidate_count']}`
- Class count: `{example['class_count']}`
- Independent objectness channel: `{example['independent_objectness_present']}`
- Feature-map shapes: `{example['feature_map_shapes']}`
- Resolved person class: `{schema['person_class_id']}`
- Resolved sports-ball class: `{schema['sports_ball_class_id']}`
- Retained raw rows per class/view: `{schema['raw_top_k_per_class']}`

The replay uses best-class confidence filtering and index-preserving
`torchvision.ops.nms` semantics equivalent to Ultralytics 8.3.49. Across
{nms['view_count']} views, exact replay was `{nms['all_views_exact']}` and the
maximum absolute difference was `{nms['maximum_absolute_difference']}`.
"""


def research_prompt() -> str:
    return """# ChatGPT Pro Extended research task

You are the research and architecture lead for the Football Intelligence
Infrastructure player and football detection subsystem. Use every uploaded
M5.5G.0 file and search primary research sources online. Do not write repository
implementation code yet.

Diagnose duplicate boxes, merged-person boxes, visible missed people, far-side
small players, partial/occluded players, off-pitch human supply and football
proposal false positives separately. Locate each failure at raw proposal,
localization, confidence, NMS, scale/tile fusion, pitch gating, temporal
recovery or renderer provenance.

Answer these ten questions explicitly:

1. Should person detection remain box-based, become visible-body/full-body
   dual-box prediction, or add instance segmentation only in dense regions?
2. Which crowd-aware regression or NMS method best addresses duplicate and
   merged boxes without hiding distinct nearby people?
3. Which multi-scale, tiled or zoom-crop design should recover small far-side
   and partial people while controlling duplicate fusion?
4. Should dense regions use short-burst promptable masks, and under what
   eligibility gate?
5. Which temporal image-evidence method should recover a visible missed person
   without becoming identity tracking?
6. Should temporal refinement be observation recovery or tracking?
7. How should off-pitch and boundary gating be separated from human detection?
8. Should football use a dedicated detector, candidate classifier or temporal
   heatmap branch?
9. What human gold labels and split design are required before training or
   precision/recall claims?
10. What exact staged CUDA bakeoff should Codex run next, including ablations,
    runtime/license risks and rejection criteria?

Return a root-cause diagnosis, primary architecture, fallback architecture,
experiments before training, label schema, benchmark design, ablation matrix,
GPU/runtime/licensing risks and a staged Codex roadmap. Do not select a model
only from generic COCO mAP. Preserve one visible on-pitch person -> one valid
independent observation, exact source provenance, and explicit uncertainty.
"""


def lineage_payload() -> dict[str, Any]:
    lineage_path = STAGE / "04_POST_NMS_FUSION_GATE_AND_RENDERER_LINEAGE" / "candidate_lineage_rows.jsonl"
    samples: list[dict[str, Any]] = []
    row_count = 0
    with lineage_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row_count += 1
            if len(samples) >= 24:
                continue
            row = json.loads(line)
            samples.append(
                {
                    key: row.get(key)
                    for key in (
                        "diagnostic_uuid",
                        "source_frame_sha256",
                        "inference_view_id",
                        "raw_candidate_index",
                        "class_name",
                        "score",
                        "bbox_panorama_pixels",
                        "confidence_filter_state",
                        "nms_state",
                        "cross_scale_cluster_id",
                        "pitch_gate_state",
                        "temporal_or_recovery_origin",
                        "final_renderer_row",
                        "canonical_row_hash",
                        "renderer_row_hash",
                    )
                }
            )
    return {
        "schema_version": "football_intelligence.m5_5g0.public_lineage_summary.v1",
        "candidate_lineage_row_count": row_count,
        "candidate_lineage_sha256": sha256_file(lineage_path),
        "raw_pre_nms_sha256": sha256_file(STAGE / "03_RAW_PRE_NMS_INSTRUMENTATION" / "pre_nms_candidate_rows.jsonl"),
        "nms_replay_validation": read_json(STAGE / "03_RAW_PRE_NMS_INSTRUMENTATION" / "nms_replay_validation.json"),
        "coordinate_validation": read_json(
            STAGE / "04_POST_NMS_FUSION_GATE_AND_RENDERER_LINEAGE" / "coordinate_transform_validation.json"
        ),
        "renderer_validation": read_json(
            STAGE / "04_POST_NMS_FUSION_GATE_AND_RENDERER_LINEAGE" / "renderer_binding_validation.json"
        ),
        "representative_rows": samples,
        "canonical_source_identifiers_excluded": True,
    }


def build_common_files(root: Path, repository: Mapping[str, Any]) -> None:
    build = read_json(STAGE / "12_COMMANDS_AND_TESTS" / "build_summary.json")
    runtime = read_json(STAGE / "02_CURRENT_DETECTOR_ARCHITECTURE_AND_RUNTIME" / "checkpoint_runtime_manifest.json")
    raw_schema = read_json(STAGE / "03_RAW_PRE_NMS_INSTRUMENTATION" / "raw_tensor_schema.json")
    nms = read_json(STAGE / "03_RAW_PRE_NMS_INSTRUMENTATION" / "nms_replay_validation.json")
    origin = read_json(STAGE / "05_PLAYER_FAILURE_CASE_MINING" / "failure_origin_summary.json")
    failure_rows = read_jsonl(STAGE / "05_PLAYER_FAILURE_CASE_MINING" / "failure_origin_matrix.jsonl")
    duplicate = read_jsonl(STAGE / "06_DUPLICATE_AND_MERGED_INSTANCE_FORENSICS" / "duplicate_forensic_rows.jsonl")
    merged = read_jsonl(STAGE / "06_DUPLICATE_AND_MERGED_INSTANCE_FORENSICS" / "merged_instance_forensic_rows.jsonl")
    missed = read_jsonl(STAGE / "07_MISSED_PLAYER_AND_SCALE_FORENSICS" / "missed_player_forensic_rows.jsonl")
    offpitch = read_jsonl(STAGE / "08_OFF_PITCH_AND_BOUNDARY_GATE_FORENSICS" / "off_pitch_boundary_forensic_rows.jsonl")
    ball = read_jsonl(STAGE / "09_FOOTBALL_BALL_RAW_CANDIDATE_FORENSICS" / "ball_candidate_rows.jsonl")
    ball_summary = read_json(STAGE / "09_FOOTBALL_BALL_RAW_CANDIDATE_FORENSICS" / "ball_candidate_summary.json")
    gpu = read_json(STAGE / "10_GPU_RUNTIME_TRANSFORM_AND_CACHE_AUDIT" / "gpu_runtime_and_memory.json")
    architecture = (STAGE / "02_CURRENT_DETECTOR_ARCHITECTURE_AND_RUNTIME" / "detector_architecture_map.md").read_text(
        encoding="utf-8"
    )

    write_text(root / "01_EXECUTIVE_BRIEFING.md", executive_briefing(build, origin))
    write_json(
        root / "02_REPO_RUNTIME_AND_SAFETY_CONTEXT.json",
        {
            "schema_version": "football_intelligence.m5_5g0.repo_safety_context.v1",
            "repository": repository,
            "safety": {key: build.get(key, runtime.get(key)) for key in SAFETY_KEYS},
            "classification": build["classification"],
            "prior_artifacts_preserved": build["prior_artifacts_preserved"],
        },
    )
    write_text(root / "03_CURRENT_DETECTOR_AND_BALL_PIPELINE.md", pipeline_document(architecture))
    diff = run(["git", "diff", "--binary", f"{BASELINE}..{repository['implementation_commit']}"]).stdout
    if not diff.strip():
        raise RuntimeError("final source diff is empty")
    write_text(root / "04_SOURCE_DIFF.patch", diff)
    write_json(
        root / "05_CHECKPOINT_ENVIRONMENT_AND_LICENSE_MANIFEST.json",
        {
            key: runtime[key]
            for key in (
                "checkpoint_sha256",
                "checkpoint_hash_required",
                "checkpoint_hash_matches",
                "checkpoint_size_bytes",
                "model_task",
                "class_count",
                "resolved_class_indices",
                "python",
                "torch",
                "torchvision",
                "torch_cuda_runtime",
                "ultralytics",
                "ultralytics_license_metadata",
                "cuda_available",
                "gpu_name",
                "gpu_compute_capability",
                "gpu_memory_total_mib",
                "gpu_driver",
                "silent_cpu_fallback",
                "new_weights_downloaded",
                "training_or_finetuning_performed",
                "production_defaults_changed",
            )
        },
    )
    write_text(root / "06_RAW_OUTPUT_INSTRUMENTATION_AND_SCHEMA.md", raw_schema_document(raw_schema, nms))
    write_json(root / "07_PRE_NMS_TO_RENDERER_LINEAGE.json", lineage_payload())
    write_jsonl(root / "08_FAILURE_CASE_ROWS.jsonl", (public_forensic_row(row) for row in failure_rows))
    write_json(
        root / "09_DUPLICATE_CONSOLIDATION_AUDIT.json",
        {
            "case_count": len(duplicate),
            "classification_counts": count(duplicate, "duplicate_origin_classification"),
            "failure_origin_counts": count(duplicate, "earliest_failure_stage"),
            "rows": [compact_case(row, classification_key="duplicate_origin_classification") for row in duplicate],
            "machine_mined_rows_are_not_human_truth": True,
        },
    )
    write_json(
        root / "10_MERGED_INSTANCE_AUDIT.json",
        {
            "case_count": len(merged),
            "classification_counts": count(merged, "merged_instance_classification"),
            "failure_origin_counts": count(merged, "earliest_failure_stage"),
            "rows": [compact_case(row, classification_key="merged_instance_classification") for row in merged],
            "machine_mined_rows_are_not_human_truth": True,
        },
    )
    write_json(
        root / "11_MISSED_PERSON_AUDIT.json",
        {
            "case_count": len(missed),
            "evidence_level_counts": count(missed, "evidence_level"),
            "failure_origin_counts": count(missed, "earliest_failure_stage"),
            "rows": [compact_case(row) for row in missed],
        },
    )
    write_json(
        root / "12_OFF_PITCH_AND_BOUNDARY_GATE_AUDIT.json",
        {
            "case_count": len(offpitch),
            "forensic_state_counts": count(offpitch, "forensic_pitch_state"),
            "failure_origin_counts": count(offpitch, "earliest_failure_stage"),
            "rows": [
                {
                    **compact_case(row),
                    "forensic_pitch_state": row["forensic_pitch_state"],
                    "pitch_gate_result": row["pitch_gate_result"],
                }
                for row in offpitch
            ],
        },
    )
    write_json(
        root / "13_FOOTBALL_BALL_CANDIDATE_AUDIT.json",
        {"summary": ball_summary, "rows": [compact_ball(row) for row in ball]},
    )
    write_json(
        root / "14_SCALE_TILE_TRANSFORM_AND_RUNTIME_AUDIT.json",
        {
            "diagnostic_summary": gpu["diagnostic_summary"],
            "total_runtime_seconds": gpu["total_runtime_seconds"],
            "maximum_peak_allocated_vram_mib": gpu["maximum_peak_allocated_vram_mib"],
            "maximum_peak_reserved_vram_mib": gpu["maximum_peak_reserved_vram_mib"],
            "silent_cpu_fallback": gpu["silent_cpu_fallback"],
            "views": [
                {key: value for key, value in view.items() if key != "source_asset_path"} for view in gpu["views"]
            ],
        },
    )
    write_json(
        root / "15_GOLD_DATASET_AND_CASE_SUPPLY_CONTEXT.json",
        {
            "gold_corpora": build["gold_corpus"],
            "player_case_supply": build["player_case_supply"],
            "player_case_targets_met": build["player_case_targets_met"],
            "ball_case_supply": ball_summary["stratum_counts"],
            "ball_case_targets_met": ball_summary["all_stratum_targets_met"],
            "original_gold_sequence_count": 24,
            "fresh_challenge_sequence_count": 32,
            "visible_no_valid_detection_state_count": 117,
            "off_pitch_person_rejection_count": 6,
            "user_screenshot_count": 5,
            "human_ball_gold_available": False,
        },
    )
    write_text(root / "16_CHATGPT_PRO_RESEARCH_PROMPT.md", research_prompt())
    for source_name, target_name in (
        ("player_failure_atlas.jpg", "17_PLAYER_FAILURE_ATLAS.jpg"),
        ("pre_post_nms_scale_atlas.jpg", "18_PRE_POST_NMS_AND_SCALE_ATLAS.jpg"),
        ("ball_off_pitch_atlas.jpg", "19_BALL_AND_OFF_PITCH_ATLAS.jpg"),
    ):
        shutil.copy2(STAGE / "11_FAILURE_ATLASES" / source_name, root / target_name)


def write_manifest(root: Path, *, pack_type: str, repository: Mapping[str, Any]) -> None:
    rows = [
        {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "REVIEW_PACK_MANIFEST.json"
    ]
    write_json(
        root / "REVIEW_PACK_MANIFEST.json",
        {
            "schema_version": "football_intelligence.m5_5g0.context_pack_manifest.v1",
            "pack_type": pack_type,
            "generated_at": datetime.now(UTC).isoformat(),
            "source_commit": repository["implementation_commit"],
            "authorized_baseline": BASELINE,
            "expected_file_count": 20,
            "actual_file_count": 20,
            "flat": True,
            "source_diff_present": True,
            "visual_evidence_file_count": 3,
            "raw_video_included": False,
            "model_weights_included": False,
            "private_mapping_payload_included": False,
            "canonical_source_identifiers_included": False,
            "credentials_included": False,
            "personal_paths_included": False,
            "manifest_self_hash_omitted_to_avoid_recursive_hashing": True,
            "files_except_manifest": rows,
        },
    )


def validate_public_content(root: Path) -> dict[str, Any]:
    research = (root / "16_CHATGPT_PRO_RESEARCH_PROMPT.md").read_text(encoding="utf-8")
    question_count = sum(
        line.startswith(tuple(f"{index}." for index in range(1, 11))) for line in research.splitlines()
    )
    public_data_files = [
        path for path in root.iterdir() if path.is_file() and path.suffix.casefold() in {".json", ".jsonl", ".md"}
    ]
    forbidden_public_keys = (b'"candidate_id"', b'"source_asset_path"', b'"source_observation_id"')
    forbidden_hits = []
    for path in public_data_files:
        payload = path.read_bytes().lower()
        for key in forbidden_public_keys:
            if key.lower() in payload:
                forbidden_hits.append({"file": path.name, "key": key.decode("ascii")})
    visual_rows = []
    for name in EXPECTED_NAMES[-3:]:
        with Image.open(root / name) as image:
            image.load()
            extrema = ImageStat.Stat(image.convert("L")).extrema[0]
            visual_rows.append(
                {
                    "name": name,
                    "width": image.width,
                    "height": image.height,
                    "nonblank": extrema[1] > extrema[0],
                }
            )
    ball = read_json(root / "13_FOOTBALL_BALL_CANDIDATE_AUDIT.json")
    ball_targets_met = ball["summary"]["all_stratum_targets_met"]
    passed = all(
        (
            len(EXPECTED_NAMES) == 20,
            len(set(EXPECTED_NAMES)) == 20,
            question_count == 10,
            not forbidden_hits,
            all(row["nonblank"] and row["width"] >= 2000 for row in visual_rows),
            ball_targets_met,
        )
    )
    result = {
        "passed": passed,
        "research_question_count": question_count,
        "forbidden_public_key_hits": forbidden_hits,
        "visual_evidence": visual_rows,
        "ball_stratum_targets_met": ball_targets_met,
    }
    if not passed:
        raise RuntimeError(f"public content validation failed: {result}")
    return result


def build_pack(root: Path, *, pack_type: str, repository: Mapping[str, Any]) -> dict[str, Any]:
    reset_pack(root)
    build_common_files(root, repository)
    write_manifest(root, pack_type=pack_type, repository=repository)
    validation = validate_flat_context_pack(
        root,
        expected_names=EXPECTED_NAMES,
        exact_file_count=20,
        maximum_file_count=20,
    )
    validation["public_content_validation"] = validate_public_content(root)
    validation.update(
        {
            "pack_type": pack_type,
            "source_commit": repository["implementation_commit"],
            "forensic_classification": "PASS_DETECTION_FORENSIC_PRO_CONTEXT_READY",
        }
    )
    return validation


def main() -> None:
    repository = validate_repository()
    pro_validation = build_pack(PRO_PACK, pack_type="CHATGPT_PRO_EXTENDED_CONTEXT", repository=repository)
    review_validation = build_pack(REVIEW_PACK, pack_type="STANDARD_CHATGPT_REVIEW", repository=repository)
    command_root = STAGE / "12_COMMANDS_AND_TESTS"
    write_json(command_root / "pro_context_pack_validation.json", pro_validation)
    write_json(command_root / "review_pack_validation.json", review_validation)
    final = {
        "classification": "PASS_DETECTION_FORENSIC_PRO_CONTEXT_READY",
        "repository": repository,
        "pro_context_pack": pro_validation,
        "review_pack": review_validation,
        "git_clean_after_pack_generation": not git("status", "--porcelain"),
    }
    write_json(command_root / "final_acceptance.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
