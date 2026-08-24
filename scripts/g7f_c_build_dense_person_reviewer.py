"""Build the frozen G7F-C dense-person selection and reviewer release."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import cv2

from football_intelligence.dense_person_gold import (
    ACK_SCHEMA,
    COMPLETION_ASSERTION,
    EVENT_SCHEMA,
    EXHAUSTIVENESS_STRIPS,
    IGNORE_REASONS,
    MIN_VISIBLE_BOX_HEIGHT_PX,
    MIN_VISIBLE_MASK_AREA_PX,
    RELEVANCE_CLASSES,
)
from football_intelligence.dense_person_reviewer import DensePersonReviewerConfig, run_server
from football_intelligence.gold_eval.core import inventory_tree, sha256_file, write_json


EXPECTED_HEAD = "62f6e4880b0f5f6d97a0125714367bc491bf494e"
EXPECTED_PROTOCOL_SHA256 = "a391200c6ccc4cc88877bb44435e03ba41a2c34fd96a8736e8ae114aef85635a"
EXPECTED_PROTOCOL_RECEIPT_SHA256 = "374a737d7b7cb48bd501efeea67579632796f3ec30a4235326af6f5f36f9431b"
EXPECTED = {
    "gold_manifest": "0a8e712487249801caa36509fce5833b4dbb3941f45d98fd2bb408371c170fdb",
    "frame_registry": "2b8831b79ba04248c6fd05f488f6321e78b6f24b1339fd295b67c6802ad13bb9",
    "source_registry": "dce19812fd6f09059b7dded19e14f01005f6c030fb117870712d01b9dd539154",
    "candidate_run_v2_schema": "f13c01b4c6b4e1061a42b66dd7239aa41ec1cf72a4617f8540363200c8d456c4",
    "human_source": "5fb20e72f35ba7bce75876d4aa584cf87db1604248ec5c3ea476b778db719672",
}
MATCHES = ("117092", "117093", "118575", "118576", "118577", "128058")
ROLE_FILES = {
    "FROZEN_G7E_CANDIDATE_REFERENCE": "frozen_historical_candidate_run_v2.json",
    "LOCAL_DEFAULT_RERUN": "g7f_b_r1_local_default_rerun.json",
    "RECALL_ORIENTED_VARIANT": "g7f_b_r1_recall_conf_012.json",
    "MULTIPLICITY_REDUCTION_VARIANT": "g7f_b_r1_multiplicity_nms_iou_050.json",
}
EXPECTED_CANDIDATE_ROWS = {
    "FROZEN_G7E_CANDIDATE_REFERENCE": 49803,
    "LOCAL_DEFAULT_RERUN": 45534,
    "RECALL_ORIENTED_VARIANT": 55419,
    "MULTIPLICITY_REDUCTION_VARIANT": 45659,
}
SLOTS = (
    "MISSED_MARK_DISCRIMINATION_RECALL",
    "MISSED_MARK_DISCRIMINATION_MULTIPLICITY",
    "MULTIPLICITY_DISAGREEMENT_INCREASE",
    "MULTIPLICITY_DISAGREEMENT_REDUCTION",
    "SUBJECT_SUPPORT_DISAGREEMENT",
    "MERGE_FRAGMENT_AMBIGUITY",
    "CANDIDATE_DENSITY_DIVERGENCE",
    "CLEAN_CONTROL",
)
RAW_COMPOSITE_COMPONENTS = (
    "recall_vs_default_recovered_missed_mark_count",
    "multiplicity_vs_default_recovered_missed_mark_count",
    "subject_support_disagreement_count",
    "subject_multiplicity_spread",
    "candidate_count_spread",
    "historical_merge_fragment_burden",
    "proposal_miss_burden",
)
WEIGHTS = {
    "recall_vs_default_recovered_missed_mark_count": 3.0,
    "multiplicity_vs_default_recovered_missed_mark_count": 2.0,
    "subject_support_disagreement_count": 3.0,
    "subject_multiplicity_spread": 2.0,
    "candidate_count_spread": 1.0,
    "historical_merge_fragment_burden": 2.0,
    "proposal_miss_burden": 2.0,
}


def roots(repo: Path) -> dict[str, Path]:
    project = repo.parent
    part9 = project / "experiments/football_observation_reasoner/part 9"
    return {
        "repo": repo,
        "project": project,
        "a": part9 / "G7F_A_R1_FULL_SOURCE_FRAME_REGISTRY_AND_ADAPTER_COVERAGE_REPAIR_v1",
        "a_original": part9 / "G7F_A_GOLD_CORPUS_V1_AND_EVALUATION_HARNESS_v1",
        "b": part9 / "G7F_B_R1_DETECTION_CANDIDATE_BAKEOFF_v1",
        "workspace": part9 / "G7F_C_DENSE_PERSON_GOLD_DISCRIMINATION_SET_v1",
        "frame_manifest": project
        / "experiments/football_observation_reasoner/part 7"
        / "G7E_A_TARGETED_TEMPORAL_BURST_SELECTION_AND_ANNOTATION_DESIGN_v1"
        / "02_BURST_SELECTION/temporal_frame_manifest.jsonl",
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _candidate_paths(paths: Mapping[str, Path]) -> dict[str, Path]:
    return {
        "FROZEN_G7E_CANDIDATE_REFERENCE": paths["a"] / "05_FROZEN_BASELINE/frozen_historical_candidate_run_v2.json",
        **{
            role: paths["b"] / "02_CANDIDATE_RUNS" / filename
            for role, filename in ROLE_FILES.items()
            if role != "FROZEN_G7E_CANDIDATE_REFERENCE"
        },
    }


def verify_upstream(paths: Mapping[str, Path]) -> dict[str, Any]:
    head, origin, status = (
        git(paths["repo"], "rev-parse", "HEAD"),
        git(paths["repo"], "rev-parse", "origin/main"),
        git(paths["repo"], "status", "--porcelain"),
    )
    if (
        status
        or head != origin
        or subprocess.call(["git", "merge-base", "--is-ancestor", EXPECTED_HEAD, head], cwd=paths["repo"])
    ):
        raise RuntimeError(f"FAIL_REPOSITORY_GATE: HEAD={head} origin={origin} dirty={bool(status)}")
    a_files = {
        "gold_manifest": paths["a"] / "02_NORMALIZED_GOLD/gold_normalization_manifest.json",
        "frame_registry": paths["a"] / "02_NORMALIZED_GOLD/gold_frame_instances.jsonl",
        "source_registry": paths["a"] / "02_NORMALIZED_GOLD/gold_source_frame_registry.jsonl",
        "candidate_run_v2_schema": paths["a"] / "01_GOLD_SCHEMAS/future_candidate_run_v2.schema.json",
    }
    hashes = {key: sha256_file(path) for key, path in a_files.items()}
    if any(hashes[key] != EXPECTED[key] for key in hashes):
        raise RuntimeError(f"FAIL_G7F_A_BINDINGS: {hashes}")
    candidate_paths = _candidate_paths(paths)
    candidate_rows, run_hashes = {}, {}
    for role, path in candidate_paths.items():
        payload = read_json(path)
        candidate_rows[role] = len(payload["candidates"])
        run_hashes[role] = sha256_file(path)
        if (
            candidate_rows[role] != EXPECTED_CANDIDATE_ROWS[role]
            or len(payload["processed_frame_instances"]) != 1080
            or len({row["source_frame_sha256"] for row in payload["processed_frame_instances"]}) != 1044
        ):
            raise RuntimeError(f"FAIL_G7F_B_RUN: {role}")
    b_acceptance = read_json(paths["b"] / "09_ACCEPTANCE/acceptance_report.json")
    if b_acceptance["decision"] != "PASS_G7F_B_R1_DETECTION_CANDIDATE_BAKEOFF_READY_FOR_DENSE_GOLD_VALIDATION":
        raise RuntimeError("FAIL_G7F_B_ACCEPTANCE")
    original_integrity = read_json(
        paths["a"] / "10_REVIEW_PACK/CHATGPT_HANDOFF/01_SOURCE_AND_ORIGINAL_GOLD_IMMUTABILITY.json"
    )
    human = inventory_tree(Path(original_integrity["human_source_inventory"]["root"]))
    original = inventory_tree(paths["a_original"])
    if human["ordered_inventory_sha256"] != EXPECTED["human_source"]:
        raise RuntimeError("FAIL_HUMAN_SOURCE_INVENTORY")
    if (
        original["ordered_inventory_sha256"]
        != original_integrity["original_workspace_after"]["ordered_inventory_sha256"]
    ):
        raise RuntimeError("FAIL_ORIGINAL_G7F_A_INVENTORY")
    return {
        "repository": {"head": head, "origin_main": origin, "clean": True, "required_head_is_ancestor": True},
        "g7f_a_hashes": hashes,
        "g7f_b_run_hashes": run_hashes,
        "candidate_rows": candidate_rows,
        "processed_instances": 1080,
        "unique_source_images": 1044,
        "human_source_inventory": human,
        "original_g7f_a_inventory": original,
        "passed": True,
        "production_ready": False,
    }


def _load_inputs(paths: Mapping[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_registry = read_jsonl(paths["a"] / "02_NORMALIZED_GOLD/gold_source_frame_registry.jsonl")
    frame_registry = read_jsonl(paths["a"] / "02_NORMALIZED_GOLD/gold_frame_instances.jsonl")
    frames_by_key = {(row["burst_id"], row["frame_sequence"]): row for row in frame_registry}
    candidates = {role: read_json(path) for role, path in _candidate_paths(paths).items()}
    counts = {
        role: Counter(row["source_frame_sha256"] for row in payload["candidates"])
        for role, payload in candidates.items()
    }
    subjects = read_jsonl(paths["b"] / "04_PAIRED_ERROR_ANALYSIS/subject_frame_delta_ledger.jsonl")
    missed = read_jsonl(paths["b"] / "04_PAIRED_ERROR_ANALYSIS/missed_mark_delta_ledger.jsonl")
    subjects_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missed_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subjects:
        subjects_by_hash[row["source_frame_sha256"]].append(row)
    for row in missed:
        missed_by_hash[row["source_frame_sha256"]].append(row)
    records = []
    merge_supplies = {"MERGED_WITH_OTHER_PEOPLE", "FRAGMENT_ONLY"}
    merge_relations = {"SAME_PERSON_FRAGMENTS", "CORRECT_INNER_BAD_OUTER"}
    for source in source_registry:
        source_hash = source["source_frame_sha256"]
        lineage = []
        matches = set()
        for member in source["frame_instances"]:
            row = frames_by_key[(member["burst_id"], member["frame_sequence"])]
            matches.add(row["match_id"])
            lineage.append(
                {
                    "burst_id": row["burst_id"],
                    "frame_sequence": row["frame_sequence"],
                    "frame_reference_id": row["frame_reference_id"],
                    "match_id": row["match_id"],
                    "half": row["half"],
                    "source_event_id": row["source_event_id"],
                }
            )
        if len(matches) != 1:
            raise RuntimeError("source hash crosses match identity")
        subject_rows, missed_rows = subjects_by_hash[source_hash], missed_by_hash[source_hash]
        raw = {
            "recall_vs_default_recovered_missed_mark_count": sum(
                row["candidate_counts"]["LOCAL_DEFAULT_RERUN"] == 0
                and row["candidate_counts"]["RECALL_ORIENTED_VARIANT"] > 0
                for row in missed_rows
            ),
            "multiplicity_vs_default_recovered_missed_mark_count": sum(
                row["candidate_counts"]["LOCAL_DEFAULT_RERUN"] == 0
                and row["candidate_counts"]["MULTIPLICITY_REDUCTION_VARIANT"] > 0
                for row in missed_rows
            ),
            "subject_support_disagreement_count": sum(
                max(
                    int(value > 0)
                    for role, value in row["candidate_counts"].items()
                    if role != "FROZEN_G7E_CANDIDATE_REFERENCE"
                )
                - min(
                    int(value > 0)
                    for role, value in row["candidate_counts"].items()
                    if role != "FROZEN_G7E_CANDIDATE_REFERENCE"
                )
                for row in subject_rows
            ),
            "subject_multiplicity_spread": sum(
                max(
                    value for role, value in row["candidate_counts"].items() if role != "FROZEN_G7E_CANDIDATE_REFERENCE"
                )
                - min(
                    value for role, value in row["candidate_counts"].items() if role != "FROZEN_G7E_CANDIDATE_REFERENCE"
                )
                for row in subject_rows
            ),
            "subject_multiplicity_increase_count": sum(
                max(
                    row["candidate_counts"]["RECALL_ORIENTED_VARIANT"],
                    row["candidate_counts"]["MULTIPLICITY_REDUCTION_VARIANT"],
                )
                > row["candidate_counts"]["LOCAL_DEFAULT_RERUN"]
                for row in subject_rows
            ),
            "subject_multiplicity_reduction_count": sum(
                0
                < min(
                    row["candidate_counts"]["RECALL_ORIENTED_VARIANT"],
                    row["candidate_counts"]["MULTIPLICITY_REDUCTION_VARIANT"],
                )
                < row["candidate_counts"]["LOCAL_DEFAULT_RERUN"]
                for row in subject_rows
            ),
            "candidate_count_spread": max(
                counts[role][source_hash] for role in ROLE_FILES if role != "FROZEN_G7E_CANDIDATE_REFERENCE"
            )
            - min(counts[role][source_hash] for role in ROLE_FILES if role != "FROZEN_G7E_CANDIDATE_REFERENCE"),
            "historical_merge_fragment_burden": sum(
                row.get("observation_supply") in merge_supplies or row.get("candidate_relationship") in merge_relations
                for row in subject_rows
            ),
            "proposal_miss_burden": sum(row.get("observation_supply") == "NO_CANDIDATE" for row in subject_rows)
            + sum(row["candidate_counts"]["LOCAL_DEFAULT_RERUN"] == 0 for row in missed_rows),
        }
        raw["candidate_count_spread_normalized_by_frozen_count"] = round(
            raw["candidate_count_spread"] / max(1, counts["FROZEN_G7E_CANDIDATE_REFERENCE"][source_hash]), 8
        )
        records.append(
            {
                "source_frame_sha256": source_hash,
                "source_width": source["source_width"],
                "source_height": source["source_height"],
                "match_id": next(iter(matches)),
                "all_frame_instance_lineage": sorted(lineage, key=lambda row: (row["burst_id"], row["frame_sequence"])),
                "raw_score_components": raw,
                "candidate_counts": {role: counts[role][source_hash] for role in ROLE_FILES},
                "source_registry_revision": source["registry_revision"],
            }
        )
    metadata = {"candidate_payloads": candidates, "candidate_counts": counts}
    return records, metadata


def _with_composites(records: list[dict[str, Any]]) -> None:
    for match in MATCHES:
        rows = [row for row in records if row["match_id"] == match]
        maxima = {
            component: max(float(row["raw_score_components"][component]) for row in rows)
            for component in RAW_COMPOSITE_COMPONENTS
        }
        for row in rows:
            normalized = {
                component: round(float(row["raw_score_components"][component]) / max(1.0, maxima[component]), 8)
                for component in RAW_COMPOSITE_COMPONENTS
            }
            row["normalized_score_components_within_match"] = normalized
            row["composite_disagreement_score"] = round(
                sum(normalized[component] * WEIGHTS[component] for component in RAW_COMPOSITE_COMPONENTS), 8
            )


def _slot_category(row: Mapping[str, Any], slot: str) -> tuple[float, ...]:
    raw = row["raw_score_components"]
    if slot == "MISSED_MARK_DISCRIMINATION_RECALL":
        return (raw["recall_vs_default_recovered_missed_mark_count"],)
    if slot == "MISSED_MARK_DISCRIMINATION_MULTIPLICITY":
        return (
            raw["multiplicity_vs_default_recovered_missed_mark_count"],
            raw["recall_vs_default_recovered_missed_mark_count"],
        )
    if slot == "MULTIPLICITY_DISAGREEMENT_INCREASE":
        return (raw["subject_multiplicity_increase_count"], raw["subject_multiplicity_spread"])
    if slot == "MULTIPLICITY_DISAGREEMENT_REDUCTION":
        return (raw["subject_multiplicity_reduction_count"], raw["subject_multiplicity_spread"])
    if slot == "SUBJECT_SUPPORT_DISAGREEMENT":
        return (raw["subject_support_disagreement_count"],)
    if slot == "MERGE_FRAGMENT_AMBIGUITY":
        return (raw["historical_merge_fragment_burden"],)
    if slot == "CANDIDATE_DENSITY_DIVERGENCE":
        return (raw["candidate_count_spread_normalized_by_frozen_count"],)
    return (-float(row["composite_disagreement_score"]), -raw["candidate_count_spread_normalized_by_frozen_count"])


def select_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = json.loads(json.dumps(records))
    _with_composites(records)
    scored, calibration = [], []
    for match in MATCHES:
        remaining = {row["source_frame_sha256"]: row for row in records if row["match_id"] == match}
        for slot in SLOTS:
            rows = list(remaining.values())
            if slot == "CLEAN_CONTROL":
                chosen = min(
                    rows,
                    key=lambda row: (
                        row["composite_disagreement_score"],
                        row["raw_score_components"]["candidate_count_spread_normalized_by_frozen_count"],
                        row["source_frame_sha256"],
                    ),
                )
                fallback = False
            else:
                available = [row for row in rows if _slot_category(row, slot)[0] > 0]
                fallback = not available
                pool = available or rows
                chosen = min(
                    pool,
                    key=lambda row: (
                        tuple(-value for value in _slot_category(row, slot)),
                        -row["composite_disagreement_score"],
                        row["source_frame_sha256"],
                    ),
                )
            selected = {
                **chosen,
                "selection_status": "SCORED_DENSE_GOLD",
                "primary_selection_slot": slot,
                "selection_fallback_used": fallback,
                "selection_fallback_reason": ("CATEGORY_UNAVAILABLE_HIGHEST_REMAINING_COMPOSITE" if fallback else None),
                "selection_reasons": [slot, "DISAGREEMENT_ENRICHED_INTERNAL_VALIDATION"],
            }
            scored.append(selected)
            del remaining[chosen["source_frame_sha256"]]
        eligible = [row for row in remaining.values() if row["candidate_counts"]["LOCAL_DEFAULT_RERUN"] >= 2]
        eligible.sort(key=lambda row: (row["composite_disagreement_score"], row["source_frame_sha256"]))
        bottom = eligible[: max(1, math.ceil(len(eligible) * 0.40))]
        chosen = min(
            bottom,
            key=lambda row: (
                row["composite_disagreement_score"],
                row["raw_score_components"]["candidate_count_spread_normalized_by_frozen_count"],
                row["source_frame_sha256"],
            ),
        )
        calibration.append(
            {
                **chosen,
                "selection_status": "CALIBRATION_ONLY",
                "primary_selection_slot": "CALIBRATION_LOW_DISAGREEMENT",
                "selection_fallback_used": False,
                "selection_fallback_reason": None,
                "selection_reasons": ["LOW_DISAGREEMENT", "MULTIPLE_LOCAL_DEFAULT_CANDIDATES_PROXY"],
            }
        )
    return scored, calibration


def _sha256_rgb(frame_bgr: Any) -> str:
    return hashlib.sha256(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()


def _decode_selected(paths: Mapping[str, Path], images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame_rows = read_jsonl(paths["frame_manifest"])
    by_hash = {row["frame_pixel_sha256"]: row for row in frame_rows}
    selected_rows = [by_hash[row["source_frame_sha256"]] for row in images]
    by_video: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    image_by_hash = {row["source_frame_sha256"]: row for row in images}
    for source in selected_rows:
        by_video[source["source_video_relative_path"]].append((source, image_by_hash[source["frame_pixel_sha256"]]))
    assets = paths["workspace"] / "05_REVIEWER/assets"
    assets.mkdir(parents=True, exist_ok=True)
    output = []
    for video_relative, pairs in sorted(by_video.items()):
        capture = cv2.VideoCapture(str(paths["project"] / video_relative))
        if not capture.isOpened():
            raise RuntimeError(f"unable to open {video_relative}")
        try:
            for source, image in sorted(pairs, key=lambda pair: pair[0]["frame_index_zero_based"]):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(source["frame_index_zero_based"]))
                ok, frame = capture.read()
                if not ok or _sha256_rgb(frame) != source["frame_pixel_sha256"]:
                    raise RuntimeError("FAIL_EXACT_SOURCE_DECODE")
                target = assets / f"{image['anonymous_dense_image_id']}.png"
                if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                    raise RuntimeError("FAIL_PNG_WRITE")
                rebuilt = cv2.imread(str(target), cv2.IMREAD_COLOR)
                if rebuilt is None or _sha256_rgb(rebuilt) != source["frame_pixel_sha256"]:
                    raise RuntimeError("FAIL_PNG_PIXEL_IDENTITY")
                output.append(
                    {
                        "anonymous_dense_image_id": image["anonymous_dense_image_id"],
                        "asset_path": f"05_REVIEWER/assets/{target.name}",
                        "asset_file_sha256": sha256_file(target),
                        "source_pixel_sha256": source["frame_pixel_sha256"],
                        "pixel_identity_verified": True,
                        "source_video_relative_path": video_relative,
                        "frame_index_zero_based": source["frame_index_zero_based"],
                    }
                )
        finally:
            capture.release()
    return sorted(output, key=lambda row: row["anonymous_dense_image_id"])


def ontology() -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.g7f_c.dense_person_gold_ontology.v1",
        "frozen_before_annotation": True,
        "primary_target": "ALL_INDIVIDUALLY_EVALUABLE_VISIBLE_HUMANS_IN_VALID_EVALUATION_REGION",
        "instance_class": "EVALUABLE_PERSON_INSTANCE",
        "relevance_classes": sorted(RELEVANCE_CLASSES),
        "primary_metric_relevance": ["MATCH_RELEVANT", "NON_MATCH_RELEVANT"],
        "ignored_relevance": ["RELEVANCE_UNCERTAIN"],
        "visible_mask_only": True,
        "amodal_completion_forbidden": True,
        "multiple_disconnected_components_allowed": True,
        "human_draws_gt_box": False,
        "derived_visible_box": "tight half-open XYXY bounds of visible-mask union",
        "evaluable_threshold": {
            "minimum_derived_visible_box_height_source_px": MIN_VISIBLE_BOX_HEIGHT_PX,
            "minimum_visible_mask_area_source_px": MIN_VISIBLE_MASK_AREA_PX,
            "individually_distinguishable_required": True,
        },
        "ignore_reasons": sorted(IGNORE_REASONS),
        "ignore_regions_human_drawn_only": True,
        "hard_individually_evaluable_people_must_not_be_ignored": True,
        "source_coordinate_convention": "zero-origin continuous source pixels bounded to [0,width-1] x [0,height-1]",
        "polygon_canonicalization": (
            "round 6 decimals; remove repeated closure; lexicographically minimum rotation across both directions"
        ),
        "multi_component_order": "ascending stable hash of canonical polygon",
        "holes": "FORBIDDEN",
        "rasterizer": {
            "library": "opencv-python",
            "version": cv2.__version__,
            "vertex_quantization": "numpy.rint_ties_to_even",
            "pixel_inclusion": "cv2.fillPoly LINE_8 boundary included",
            "mask_bytes": "uint8 C-contiguous height-width",
            "rle": "COCO-compatible uncompressed counts in Fortran column-major order",
        },
        "exhaustiveness": {
            "vertical_strip_count": EXHAUSTIVENESS_STRIPS,
            "required_state": "REVIEWED_FOR_EXHAUSTIVENESS",
            "zero_person_strip_valid": True,
        },
        "completion_assertion": COMPLETION_ASSERTION,
        "production_ready": False,
    }


def metric_protocol() -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.g7f_c.dense_gold_metric_protocol.v1",
        "frozen_before_annotation": True,
        "DISAGREEMENT_ENRICHED_SAMPLE": True,
        "tuning_exposure": "DENSE_GOLD_INTERNAL_VALIDATION",
        "future_sealed": False,
        "scored_sample_size_images": 48,
        "calibration_images_excluded": 6,
        "selection_unit": "UNIQUE_SOURCE_FRAME_SHA256",
        "primary_target": "ALL_EVALUABLE_PERSON",
        "gt_boxes": "DERIVED_FROM_VISIBLE_MASK_UNION",
        "metrics": ["AP@[IoU=.50:.95]", "AP50", "AP75", "recall@[IoU=.50:.95]", "recall@.50", "recall@.75"],
        "ap_interpolation": "COCO_101_POINT",
        "iou_thresholds": [round(0.50 + index * 0.05, 2) for index in range(10)],
        "candidate_confidence": "ORIGINAL_UNCHANGED",
        "dense_gold_confidence_recalibration_forbidden": True,
        "matching": "confidence-descending deterministic one-to-one per source image; candidate ID tie-break",
        "fixed_ignore_rule": {
            "excluded_from_fp_accounting_when": [
                "candidate center is inside a human-drawn ignore region",
                "intersection(candidate box, ignore mask) / candidate box area >= 0.50",
            ],
            "candidate_derived_ignore_forbidden": True,
        },
        "duplicate_diagnostic": {
            "name": "DUPLICATE_CANDIDATE_AT_IOU50",
            "rule": "additional candidate with IoU >= 0.50 to an already matched GT under IoU-0.50 matching",
        },
        "merge_diagnostic": {
            "name": "MULTI_PERSON_CANDIDATE_MASK_COVERAGE_030",
            "rule": "candidate covers at least 30 percent of each of at least two distinct visible person masks",
        },
        "candidate_burden": [
            "candidate_rows_per_image",
            "matched_candidates",
            "unmatched_candidates",
            "duplicates",
            "candidate_to_gt_ratio",
        ],
        "candidate_volume_is_false_positive_rate": False,
        "bias_statement": (
            "Metrics discriminate the shortlist on disagreement-enriched internal validation and are not unbiased "
            "whole-match estimates."
        ),
        "final_promotion_requires_new_sealed_match_footage": True,
        "production_ready": False,
    }


def _json_schema(kind: str) -> dict[str, Any]:
    schema = EVENT_SCHEMA if kind == "event" else ACK_SCHEMA
    required = (
        [
            "schema_version",
            "event_id",
            "event_sha256",
            "anonymous_dense_image_id",
            "pass_kind",
            "selection_status",
            "source_frame_sha256",
            "source_width",
            "source_height",
            "all_frame_instance_lineage",
            "binding_hashes",
            "reviewer_release",
            "annotation",
            "server_validation",
            "immutable",
        ]
        if kind == "event"
        else [
            "schema_version",
            "acknowledgement_id",
            "acknowledgement_sha256",
            "event_id",
            "event_sha256",
            "anonymous_dense_image_id",
            "pass_kind",
            "status",
        ]
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"G7F-C dense person {kind}",
        "type": "object",
        "properties": {"schema_version": {"const": schema}},
        "required": required,
        "additionalProperties": True,
    }


def build(paths: Mapping[str, Path]) -> None:
    workspace = paths["workspace"]
    protocol = workspace / "01_SELECTION/PREDECLARED_SELECTION_PROTOCOL.json"
    receipt = workspace / "01_SELECTION/PREDECLARED_SELECTION_PROTOCOL.sha256.json"
    if (
        sha256_file(protocol) != EXPECTED_PROTOCOL_SHA256
        or sha256_file(receipt) != EXPECTED_PROTOCOL_RECEIPT_SHA256
        or read_json(receipt)["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256
    ):
        raise RuntimeError("FAIL_SELECTION_PROTOCOL_FREEZE")
    if (workspace / "01_SELECTION/dense_gold_selection_manifest.json").exists():
        raise RuntimeError("selection manifest already exists; rebuild in a fresh workspace")
    upstream = verify_upstream(paths)
    write_json(workspace / "00_SOURCE_FREEZE/upstream_substrate_integrity_before.json", upstream)
    records, metadata = _load_inputs(paths)
    scored, calibration = select_records(records)
    if len(scored) != 48 or len(calibration) != 6:
        raise RuntimeError("FAIL_SELECTION_COUNTS")
    selected_hashes = [row["source_frame_sha256"] for row in scored + calibration]
    if len(set(selected_hashes)) != 54:
        raise RuntimeError("FAIL_SELECTION_DUPLICATE_HASH")
    scored_counts = Counter(row["match_id"] for row in scored)
    calibration_counts = Counter(row["match_id"] for row in calibration)
    if scored_counts != Counter({match: 8 for match in MATCHES}) or calibration_counts != Counter(
        {match: 1 for match in MATCHES}
    ):
        raise RuntimeError("FAIL_SELECTION_MATCH_BALANCE")
    ordered_calibration = sorted(
        calibration, key=lambda row: hashlib.sha256(row["source_frame_sha256"].encode()).hexdigest()
    )
    ordered_scored = sorted(scored, key=lambda row: hashlib.sha256(row["source_frame_sha256"].encode()).hexdigest())
    images = []
    for position, row in enumerate(ordered_calibration + ordered_scored, start=1):
        images.append(
            {
                **row,
                "anonymous_dense_image_id": f"DG-{position:03d}",
                "review_queue_position": position,
                "upstream_artifact_hashes": {
                    "gold_manifest": EXPECTED["gold_manifest"],
                    "frame_registry": EXPECTED["frame_registry"],
                    "source_registry": EXPECTED["source_registry"],
                    "selection_protocol": EXPECTED_PROTOCOL_SHA256,
                    **{f"candidate_run_{role.lower()}": value for role, value in upstream["g7f_b_run_hashes"].items()},
                },
                "frozen_source_lineage": {
                    "source_registry_revision": row["source_registry_revision"],
                    "all_frame_instance_lineage": row["all_frame_instance_lineage"],
                },
            }
        )
    selected_by_hash = {row["source_frame_sha256"]: row for row in images}
    repeat_rows = []
    for repeat_index, match in enumerate(MATCHES, start=1):
        chosen = min(
            [row for row in scored if row["match_id"] == match],
            key=lambda row: (-row["composite_disagreement_score"], row["source_frame_sha256"]),
        )
        repeat_rows.append(
            {
                "sealed_repeat_id": f"BR-{repeat_index:03d}",
                "anonymous_dense_image_id": selected_by_hash[chosen["source_frame_sha256"]]["anonymous_dense_image_id"],
                "source_frame_sha256": chosen["source_frame_sha256"],
                "match_id": match,
            }
        )
    selection_manifest = {
        "schema_version": "football_intelligence.g7f_c.dense_gold_selection_manifest.v1",
        "selection_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "selection_frozen_before_annotation": True,
        "selection_unit": "UNIQUE_SOURCE_FRAME_SHA256",
        "scored_unique_source_images": 48,
        "calibration_unique_source_images": 6,
        "reviewer_queue_images": 54,
        "scored_per_match": dict(sorted(scored_counts.items())),
        "calibration_per_match": dict(sorted(calibration_counts.items())),
        "duplicate_source_hash_count": 0,
        "DISAGREEMENT_ENRICHED_SAMPLE": True,
        "tuning_exposure": "DENSE_GOLD_INTERNAL_VALIDATION",
        "future_sealed": False,
        "images": images,
        "annotation_started": False,
        "production_ready": False,
    }
    selection_path = workspace / "01_SELECTION/dense_gold_selection_manifest.json"
    write_json(selection_path, selection_manifest)
    selection_sha = sha256_file(selection_path)
    write_json(
        workspace / "01_SELECTION/dense_gold_selection_manifest.sha256.json",
        {"selection_manifest_sha256": selection_sha, "frozen_before_annotation": True, "mutation_allowed": False},
    )
    assets = _decode_selected(paths, images)
    write_json(workspace / "05_REVIEWER/source_asset_manifest.json", {"assets": assets, "asset_count": 54})
    ontology_path = workspace / "02_ONTOLOGY/dense_person_gold_ontology.json"
    metric_path = workspace / "03_METRICS/dense_gold_metric_protocol.json"
    write_json(ontology_path, ontology())
    write_json(metric_path, metric_protocol())
    ontology_sha, metric_sha = sha256_file(ontology_path), sha256_file(metric_path)
    write_json(
        workspace / "02_ONTOLOGY/dense_person_gold_ontology.sha256.json",
        {"ontology_sha256": ontology_sha, "frozen_before_annotation": True},
    )
    write_json(
        workspace / "03_METRICS/dense_gold_metric_protocol.sha256.json",
        {"metric_protocol_sha256": metric_sha, "frozen_before_annotation": True},
    )
    write_json(workspace / "04_SCHEMAS/dense_person_frame_annotation.schema.json", _json_schema("event"))
    write_json(workspace / "04_SCHEMAS/dense_person_frame_acknowledgement.schema.json", _json_schema("ack"))
    reveal_payloads = {}
    candidate_rows_by_role: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for role, payload in metadata["candidate_payloads"].items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in payload["candidates"]:
            if candidate["source_frame_sha256"] in selected_by_hash:
                grouped[candidate["source_frame_sha256"]].append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "box_xyxy": candidate["box_xyxy"],
                        "confidence": candidate["confidence"],
                    }
                )
        candidate_rows_by_role[role] = dict(grouped)
    for image in images:
        source_hash = image["source_frame_sha256"]
        reveal_payloads[image["anonymous_dense_image_id"]] = {
            "read_only": True,
            "annotation_mutation_allowed": False,
            "source_frame_sha256": source_hash,
            "runs": {
                role: sorted(candidate_rows_by_role[role].get(source_hash, []), key=lambda row: row["candidate_id"])
                for role in ROLE_FILES
            },
        }
    reveal_path = workspace / "05_REVIEWER/sealed_candidate_reveal_payloads.json"
    write_json(
        reveal_path,
        {
            "schema_version": "football_intelligence.g7f_c.post_finalization_candidate_reveal.v1",
            "pre_finalization_access_forbidden": True,
            "reveal_payloads": reveal_payloads,
        },
    )
    repeat_path = workspace / "05_REVIEWER/sealed_blind_repeat_qa_selection.json"
    write_json(
        repeat_path,
        {
            "schema_version": "football_intelligence.g7f_c.sealed_blind_repeat_selection.v1",
            "hidden_during_first_pass": True,
            "one_per_match": True,
            "rows": repeat_rows,
        },
    )
    reviewer_files = [
        paths["repo"] / "src/football_intelligence/dense_person_gold.py",
        paths["repo"] / "src/football_intelligence/dense_person_reviewer.py",
        paths["repo"] / "src/football_intelligence/dense_person_reviewer_static/index.html",
        paths["repo"] / "src/football_intelligence/dense_person_reviewer_static/app.js",
        paths["repo"] / "src/football_intelligence/dense_person_reviewer_static/styles.css",
    ]
    reviewer_release = "G7F_C_DENSE_PERSON_REVIEWER_V1"
    release_manifest = {
        "reviewer_release": reviewer_release,
        "repository_commit": git(paths["repo"], "rev-parse", "HEAD"),
        "files": [
            {
                "path": path.resolve().relative_to(paths["repo"].resolve()).as_posix(),
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in reviewer_files
        ],
    }
    write_json(workspace / "05_REVIEWER/reviewer_release_manifest.json", release_manifest)
    decisions_root = workspace / "06_DENSE_DECISIONS"
    decisions_root.mkdir(parents=True, exist_ok=True)
    launch = workspace / "05_REVIEWER/launch_reviewer.ps1"
    launch.write_text(
        "$repo = 'C:\\Users\\sebgr\\Documents\\football-intelligence\\SoccerTrack-v2'\n"
        "Set-Location -LiteralPath $repo\n"
        ".\\.venv\\Scripts\\python.exe scripts\\g7f_c_build_dense_person_reviewer.py serve\n",
        encoding="utf-8",
        newline="\n",
    )
    repeat_launch = workspace / "05_REVIEWER/launch_blind_repeat_reviewer.ps1"
    repeat_launch.write_text(
        "$repo = 'C:\\Users\\sebgr\\Documents\\football-intelligence\\SoccerTrack-v2'\n"
        "Set-Location -LiteralPath $repo\n"
        ".\\.venv\\Scripts\\python.exe scripts\\g7f_c_build_dense_person_reviewer.py serve-repeat\n",
        encoding="utf-8",
        newline="\n",
    )
    bindings = {
        "selection_manifest_sha256": selection_sha,
        "ontology_sha256": ontology_sha,
        "metric_protocol_sha256": metric_sha,
        "reviewer_release_manifest_sha256": sha256_file(workspace / "05_REVIEWER/reviewer_release_manifest.json"),
        "event_schema_sha256": sha256_file(workspace / "04_SCHEMAS/dense_person_frame_annotation.schema.json"),
        "ack_schema_sha256": sha256_file(workspace / "04_SCHEMAS/dense_person_frame_acknowledgement.schema.json"),
        "sealed_reveal_payload_sha256": sha256_file(reveal_path),
        "sealed_repeat_qa_sha256": sha256_file(repeat_path),
    }
    write_json(workspace / "05_REVIEWER/reviewer_binding_hashes.json", bindings)
    release_gate = {
        "schema_version": "football_intelligence.g7f_c.dense_person_gold_release_gate.v1",
        "classification": "PENDING_ENGINEERING_ACCEPTANCE",
        "reviewer_release": reviewer_release,
        "bindings": bindings,
        "queue": {"scored": 48, "calibration": 6, "total": 54, "duplicates": 0},
        "real_decisions_root": str(decisions_root),
        "real_decisions_root_inventory": inventory_tree(decisions_root),
        "real_finalized_annotation_count": 0,
        "real_acknowledgement_count": 0,
        "human_annotation_started": False,
        "engineering_acceptance_uses_temp_decisions_only": True,
        "production_ready": False,
    }
    write_json(workspace / "08_RELEASE/G7F_C_DENSE_PERSON_GOLD_RELEASE_GATE.json", release_gate)
    write_json(
        workspace / "01_SELECTION/selection_execution_receipt.json",
        {
            "schema_version": "football_intelligence.g7f_c.selection_execution_receipt.v1",
            "predeclared_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "selection_manifest_sha256": selection_sha,
            "protocol_receipt_sha256": sha256_file(receipt),
            "protocol_receipt_remained_byte_immutable": sha256_file(receipt) == EXPECTED_PROTOCOL_RECEIPT_SHA256,
            "annotation_started": False,
            "production_ready": False,
        },
    )
    print(json.dumps({"status": "BUILT", "selection_sha256": selection_sha, "scored": 48, "calibration": 6}))


def server_config(
    paths: Mapping[str, Path], *, decisions_root: Path | None = None, blind_repeat: bool = False
) -> DensePersonReviewerConfig:
    workspace = paths["workspace"]
    return DensePersonReviewerConfig(
        selection_manifest_path=workspace / "01_SELECTION/dense_gold_selection_manifest.json",
        assets_root=workspace / "05_REVIEWER/assets",
        decisions_root=decisions_root or workspace / "06_DENSE_DECISIONS",
        binding_hashes=read_json(workspace / "05_REVIEWER/reviewer_binding_hashes.json"),
        reviewer_release="G7F_C_DENSE_PERSON_REVIEWER_V1",
        reveal_payload_path=workspace / "05_REVIEWER/sealed_candidate_reveal_payloads.json",
        pass_kind="BLIND_REPEAT" if blind_repeat else "FIRST_PASS",
        repeat_manifest_path=workspace / "05_REVIEWER/sealed_blind_repeat_qa_selection.json",
        require_completed_first_pass=blind_repeat,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("build", "serve", "serve-repeat"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--decisions-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = roots(args.repo.resolve())
    if args.phase == "build":
        build(paths)
    else:
        run_server(
            server_config(
                paths,
                decisions_root=args.decisions_root,
                blind_repeat=args.phase == "serve-repeat",
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
