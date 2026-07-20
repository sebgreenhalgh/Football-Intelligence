"""Immutable gold ingestion and sealed-split controls for sports-MOT research."""

from __future__ import annotations

import copy
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.gold_persistence import CrashSafeGoldPersistence
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest
from football_intelligence.review_chassis.persistence import atomic_write_json


EXPECTED_POLYGON_HASH = "8c9ae3e39229b8a8f35e6bfc69c9e8c83e32e02e3da5a1f8bbf90199ee82b055"
VISIBLE_STATES = {"OBSERVED_EXISTING_DETECTION", "OBSERVED_MANUAL_BBOX"}
NO_BOX_STATES = {
    "MISSING_VISIBLE_NO_VALID_DETECTION",
    "VISIBLE_NO_VALID_DETECTION",
    "NOT_VISIBLE",
    "NOT_VISIBLE_IN_PANORAMA",
    "AMBIGUOUS",
    "OUTSIDE_ROI",
    "OUTSIDE_DYNAMIC_VIEW_BUT_VISIBLE_IN_PANORAMA",
}
ALLOWED_GOLD_STATES = VISIBLE_STATES | NO_BOX_STATES
ARCHITECTURE_STAGE_NAME = "M5_5F1A_ON_PITCH_GOLD_STRAND_BENCHMARK_AND_SPORTS_MOT_ARCHITECTURE_RESET_v1"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def _event_hash(event: dict[str, Any]) -> str:
    return stable_hash({key: value for key, value in event.items() if key not in {"event_hash", "ack"}})


def validate_gold_ledger(events: list[dict[str, Any]]) -> dict[str, Any]:
    sequences = [int(event.get("event_sequence", -1)) for event in events]
    hash_failures = [
        index for index, event in enumerate(events, start=1) if event.get("event_hash") != _event_hash(event)
    ]
    duplicate_sequences = sorted(value for value, count in Counter(sequences).items() if count > 1)
    client_ids = [str(event.get("client_event_id")) for event in events]
    idempotency_keys = [str(event.get("idempotency_key")) for event in events]
    completion_events = [event for event in events if event.get("event_type") == "REVIEW_COMPLETED"]
    sequence_saved = {str(event.get("sequence_id")) for event in events if event.get("event_type") == "SEQUENCE_SAVED"}
    passed = (
        bool(events)
        and sequences == sorted(sequences)
        and all(sequence > 0 for sequence in sequences)
        and not hash_failures
        and len(client_ids) == len(set(client_ids))
        and len(idempotency_keys) == len(set(idempotency_keys))
        and len(completion_events) == 1
        and len(sequence_saved) == 24
        and max(sequences) == 1240
    )
    return {
        "passed": passed,
        "event_count": len(events),
        "highest_event_sequence": max(sequences, default=0),
        "append_order_nondecreasing": sequences == sorted(sequences),
        "duplicate_event_sequences": duplicate_sequences,
        "event_hash_failure_indices": hash_failures,
        "unique_client_event_ids": len(client_ids) == len(set(client_ids)),
        "unique_idempotency_keys": len(idempotency_keys) == len(set(idempotency_keys)),
        "review_completed_event_count": len(completion_events),
        "sequence_saved_distinct_count": len(sequence_saved),
    }


def validate_completed_gold(package_root: Path) -> dict[str, Any]:
    decisions_root = package_root / "decisions"
    bundle = validate_completion_bundle(decisions_root)
    completed = read_json(decisions_root / "completed_review.json")
    summary = read_json(decisions_root / "completed_review_summary.json")
    events = read_jsonl(decisions_root / "review_decision_events.jsonl")
    ledger = validate_gold_ledger(events)
    materialized = completed.get("state", {}).get("gold_materialized", {})
    sequences = materialized.get("sequences", {})
    if not isinstance(sequences, dict):
        sequences = {}
    seed_count = sum(sequence.get("seed_confirmation") is not None for sequence in sequences.values())
    finalized_count = sum(bool(sequence.get("finalized")) for sequence in sequences.values())
    frame_state_count = sum(
        value is not None
        for sequence in sequences.values()
        for frame in sequence.get("frames", {}).values()
        for value in (frame.get("A"), frame.get("B"))
    )
    polygon_hash = summary.get("approved_polygon_hash")
    checks = {
        "completion_bundle": bundle.get("passed") is True,
        "ledger": ledger["passed"],
        "completed": summary.get("completed") is True,
        "reviewed_sequences": summary.get("reviewed_sequences") == 24,
        "finalized_sequences": finalized_count == 24 and summary.get("finalized_sequences") == 24,
        "seed_confirmations": seed_count == 24 and summary.get("seed_confirmations") == 24,
        "strand_frame_states": frame_state_count == 624 and summary.get("strand_frame_states") == 624,
        "pending_outbox_events": summary.get("pending_outbox_events") == 0,
        "final_event_sequence": summary.get("final_server_event_sequence") == 1240,
        "approved_polygon": polygon_hash == EXPECTED_POLYGON_HASH,
        "materialized_hash": stable_hash(materialized) == summary.get("final_materialized_state_hash"),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "bundle_validation": bundle,
        "ledger_validation": ledger,
        "review_id": summary.get("review_id"),
        "reviewed_sequences": len(sequences),
        "finalized_sequences": finalized_count,
        "seed_confirmations": seed_count,
        "strand_frame_states": frame_state_count,
        "pending_outbox_events": summary.get("pending_outbox_events"),
        "completion_event_count": ledger["review_completed_event_count"],
        "final_server_event_sequence": summary.get("final_server_event_sequence"),
        "approved_polygon_hash": polygon_hash,
        "final_materialized_state_hash": stable_hash(materialized),
        "historical_summary": {
            "total_cases": summary.get("total_cases"),
            "reviewed": summary.get("reviewed"),
            "remaining": summary.get("remaining"),
        },
        "source_hashes": {
            name: sha256_file(decisions_root / name)
            for name in (
                "completed_review.json",
                "completed_review_events.jsonl",
                "completed_review_manifest.json",
                "completed_review_summary.json",
                "review_decision_events.jsonl",
            )
        },
    }


def replay_completed_gold(package_root: Path, replay_root: Path) -> dict[str, Any]:
    if replay_root.exists():
        raise FileExistsError(f"refusing to reuse replay root: {replay_root}")
    replay_root.mkdir(parents=True)
    source_events = package_root / "decisions" / "review_decision_events.jsonl"
    shutil.copy2(source_events, replay_root / "review_decision_events.jsonl")
    manifest = load_manifest(package_root / "reviewer_manifest.json")
    ui_config = load_ui_config(package_root / "ui_config.json")
    persistence = CrashSafeGoldPersistence(
        manifest=manifest,
        ui_config=ui_config,
        decisions_root=replay_root,
        reviewer_session_id="m5_5f1b_immutable_gold_replay",
        polygon_store=None,
    )
    events = read_jsonl(replay_root / "review_decision_events.jsonl")
    ledger = validate_gold_ledger(events)
    replayed = persistence._materialize_events(events)  # noqa: SLF001 - intentional ledger replay audit
    completed = read_json(package_root / "decisions" / "completed_review.json")
    source = completed["state"]["gold_materialized"]
    replay_hash = stable_hash(replayed)
    source_hash = stable_hash(source)
    atomic_write_json(
        replay_root / "review_decisions.json",
        {
            "schema_version": "football_intelligence.m5_5f1b.gold_replay_state.v1",
            "source_review_id": manifest.review_id,
            "gold_materialized": replayed,
            "server_state_hash": replay_hash,
            "event_sequence": ledger["highest_event_sequence"],
            "completed": replayed.get("review_completed") is True,
        },
    )
    return {
        "passed": ledger["passed"] and replayed == source and replay_hash == source_hash,
        "ledger_validation": ledger,
        "source_materialized_state_hash": source_hash,
        "replayed_materialized_state_hash": replay_hash,
        "materialized_states_equal": replayed == source,
        "source_event_ledger_sha256": sha256_file(source_events),
        "replay_event_ledger_sha256": sha256_file(replay_root / "review_decision_events.jsonl"),
        "scientific_events_added": 0,
        "temporary_decisions_root": str(replay_root),
    }


def _bbox(value: dict[str, Any]) -> dict[str, float] | None:
    candidate = value.get("bbox_original_pixels") or value.get("bbox")
    if not isinstance(candidate, dict):
        return None
    try:
        return {key: float(candidate[key]) for key in ("x1", "y1", "x2", "y2")}
    except (KeyError, TypeError, ValueError):
        return None


def _validate_bbox(box: dict[str, float], width: int, height: int) -> bool:
    return (
        0 <= box["x1"] < box["x2"] <= width
        and 0 <= box["y1"] < box["y2"] <= height
        and all(value == value and abs(value) != float("inf") for value in box.values())
    )


def _resolve_gold_state(
    value: dict[str, Any],
    *,
    visible_frame: dict[str, Any],
    sealed_frame: dict[str, Any],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    state = str(value.get("state"))
    if state not in ALLOWED_GOLD_STATES:
        raise ValueError(f"unsupported gold state: {state}")
    output = {
        "state": state,
        "visibility": "VISIBLE" if state in VISIBLE_STATES | {"MISSING_VISIBLE_NO_VALID_DETECTION"} else state,
        "bbox": None,
        "provenance_type": None,
        "anonymous_detection_id": value.get("anonymous_detection_id"),
        "source_observation_id": None,
        "source_row_hash": None,
        "source_layer": None,
    }
    if state == "OBSERVED_EXISTING_DETECTION":
        anonymous_id = str(value.get("anonymous_detection_id"))
        visible = next(
            (
                row
                for row in visible_frame.get("anonymous_detections", [])
                if str(row.get("anonymous_detection_id")) == anonymous_id
            ),
            None,
        )
        sealed = next(
            (
                row
                for row in sealed_frame.get("detections", [])
                if str(row.get("anonymous_detection_id")) == anonymous_id
            ),
            None,
        )
        if visible is None or sealed is None:
            raise ValueError(f"could not bind anonymous detection {anonymous_id}")
        box = _bbox(visible)
        if box is None or not _validate_bbox(box, image_width, image_height):
            raise ValueError(f"invalid detector bbox for {anonymous_id}")
        output.update(
            {
                "bbox": box,
                "provenance_type": "EXACT_SOURCE_DETECTION_ROW",
                "source_observation_id": sealed.get("source_observation_id"),
                "source_row_hash": sealed.get("source_row_hash"),
                "source_layer": sealed.get("source_layer"),
            }
        )
    elif state == "OBSERVED_MANUAL_BBOX":
        box = _bbox(value)
        if box is None or not _validate_bbox(box, image_width, image_height):
            raise ValueError("invalid manual gold bbox")
        output.update(
            {
                "bbox": box,
                "provenance_type": "HUMAN_MANUAL_BBOX",
                "source_row_hash": stable_hash(
                    {"bbox": box, "frame_sequence": sealed_frame["frame_sequence"], "source": "human_manual_bbox"}
                ),
                "source_layer": "human_manual_bbox",
            }
        )
    else:
        output["provenance_type"] = "HUMAN_NO_BOX_STATE"
    return output


@dataclass(frozen=True)
class GoldDataset:
    rows: tuple[dict[str, Any], ...]
    sequences: tuple[dict[str, Any], ...]
    approved_polygon: dict[str, Any]
    dataset_hash: str

    def rows_for_split(self, split: str) -> list[dict[str, Any]]:
        return [copy.deepcopy(row) for row in self.rows if row["split"] == split]


def _authoritative_case_bindings(package_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    architecture_root = package_root.parents[1] / ARCHITECTURE_STAGE_NAME
    mapping_path = architecture_root / "10_GOLD_STRAND_ANNOTATION_PACKAGE" / "sealed" / "server_mapping.json"
    curation_path = architecture_root / "05_GOLD_BENCHMARK_CURATION" / "selected_gold_sequences.jsonl"
    mapping = read_json(mapping_path)
    curation_rows = {str(row["sequence_id"]): row for row in read_jsonl(curation_path)}
    scientific_mappings = {
        case_id: row
        for case_id, row in mapping.get("cases", {}).items()
        if str(case_id).startswith("m5_5f1a_gold_sequence_")
    }
    if len(scientific_mappings) != 24 or len(curation_rows) != 24:
        raise ValueError("authoritative gold case-binding count mismatch")
    mapping = {**mapping, "cases": scientific_mappings}
    return mapping, curation_rows


def ingest_gold_dataset(package_root: Path) -> GoldDataset:
    manifest = read_json(package_root / "reviewer_manifest.json")
    completed = read_json(package_root / "decisions" / "completed_review.json")
    sealed_mapping, curation_rows = _authoritative_case_bindings(package_root)
    approved_polygon = read_json(package_root / "decisions" / "polygon" / "approved_polygon.json")
    if approved_polygon.get("approved_polygon_hash") != EXPECTED_POLYGON_HASH:
        raise ValueError("approved polygon hash mismatch")
    dimensions = approved_polygon["source_dimensions"]
    width, height = int(dimensions["width"]), int(dimensions["height"])
    cases = {str(case["case_id"]): case for case in manifest["cases"]}
    mappings = sealed_mapping.get("cases", {})
    materialized = completed["state"]["gold_materialized"]["sequences"]
    rows: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []
    for case_id in sorted(materialized):
        case = cases.get(case_id)
        mapping = mappings.get(case_id)
        sequence_state = materialized[case_id]
        if case is None or mapping is None:
            raise ValueError(f"missing manifest or sealed binding for {case_id}")
        internal_id = str(mapping["internal_sequence_id"])
        split = str(mapping["split"])
        curation = curation_rows.get(internal_id)
        if curation is None:
            raise ValueError(f"missing immutable curation binding for {internal_id}")
        if curation.get("split") != split or curation.get("temporal_event_cluster_id") != mapping.get(
            "temporal_event_cluster_id"
        ):
            raise ValueError(f"curation binding mismatch for {case_id}")
        visible_frames = {int(frame["frame_sequence"]): frame for frame in case["visible_metadata"]["frame_records"]}
        sealed_frames = {int(frame["frame_sequence"]): frame for frame in mapping["frames"]}
        ordered_frames = sorted(visible_frames)
        if ordered_frames != sorted(sealed_frames) or len(ordered_frames) != 13:
            raise ValueError(f"frame binding mismatch for {case_id}")
        if ordered_frames != [int(value) for value in curation["frames"]]:
            raise ValueError(f"curation frame mismatch for {case_id}")
        sequence_rows = []
        for frame_sequence in ordered_frames:
            frame_state = sequence_state["frames"].get(str(frame_sequence))
            if not isinstance(frame_state, dict):
                raise ValueError(f"missing materialized frame {case_id}:{frame_sequence}")
            visible_frame = visible_frames[frame_sequence]
            sealed_frame = sealed_frames[frame_sequence]
            if sealed_frame.get("source_frame_sha256") is None:
                raise ValueError(f"missing source frame hash for {case_id}:{frame_sequence}")
            frame_path = Path(str(sealed_frame["source_frame_path"]))
            if not frame_path.is_file() or sha256_file(frame_path) != sealed_frame["source_frame_sha256"]:
                raise ValueError(f"source frame missing or hash mismatch for {case_id}:{frame_sequence}")
            strands = {
                strand: _resolve_gold_state(
                    frame_state[strand],
                    visible_frame=visible_frame,
                    sealed_frame=sealed_frame,
                    image_width=width,
                    image_height=height,
                )
                for strand in ("A", "B")
            }
            if (
                strands["A"]["source_row_hash"] is not None
                and strands["A"]["source_row_hash"] == strands["B"]["source_row_hash"]
            ):
                raise ValueError(f"A/B reuse the same independent source row at {case_id}:{frame_sequence}")
            row = {
                "case_id": case_id,
                "sequence_id": internal_id,
                "split": split,
                "temporal_event_cluster_id": mapping["temporal_event_cluster_id"],
                "frame_sequence": frame_sequence,
                "timestamp_seconds": float(visible_frame["timestamp_seconds"]),
                "source_frame_path": str(frame_path),
                "source_frame_sha256": sealed_frame["source_frame_sha256"],
                "image_width": width,
                "image_height": height,
                "roi": copy.deepcopy(visible_frame["roi"]),
                "approved_polygon_hash": approved_polygon["approved_polygon_hash"],
                "approved_polygon_manifest_hash": completed["polygon_binding"]["approved_polygon_manifest_hash"],
                "seed_frame": frame_sequence == int(sequence_state["seed_confirmation"]["source_frame_sequence"]),
                "A": strands["A"],
                "B": strands["B"],
                "temporary_anonymous_strands_only": True,
                "persistent_identity_created": False,
            }
            row["gold_row_hash"] = stable_hash(row)
            rows.append(row)
            sequence_rows.append(row)
        sequences.append(
            {
                "case_id": case_id,
                "sequence_id": internal_id,
                "split": split,
                "temporal_event_cluster_id": mapping["temporal_event_cluster_id"],
                "frame_sequences": ordered_frames,
                "source_window": [ordered_frames[0], ordered_frames[-1]],
                "frame_hashes": [row["source_frame_sha256"] for row in sequence_rows],
                "A_source_row_hashes": [row["A"]["source_row_hash"] for row in sequence_rows],
                "B_source_row_hashes": [row["B"]["source_row_hash"] for row in sequence_rows],
                "roi_hash": stable_hash([row["roi"] for row in sequence_rows]),
                "seed_confirmation": copy.deepcopy(sequence_state["seed_confirmation"]),
                "finalized": bool(sequence_state["finalized"]),
                "decision": sequence_state["decision"],
            }
        )
    if len(sequences) != 24 or len(rows) != 312:
        raise ValueError(f"gold dataset count mismatch: {len(sequences)} sequences, {len(rows)} frames")
    strand_state_count = sum(1 for row in rows for _ in (row["A"], row["B"]))
    if strand_state_count != 624:
        raise ValueError(f"gold strand state count mismatch: {strand_state_count}")
    dataset_hash = stable_hash({"rows": rows, "sequences": sequences})
    return GoldDataset(tuple(rows), tuple(sequences), approved_polygon, dataset_hash)


def split_leakage_audit(dataset: GoldDataset) -> dict[str, Any]:
    split_names = ("diagnostic", "development", "sealed_holdout")
    groups = {split: [row for row in dataset.sequences if row["split"] == split] for split in split_names}
    counts = {split: len(rows) for split, rows in groups.items()}
    comparisons = []
    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1 :]:
            left, right = groups[left_name], groups[right_name]
            left_frames = {frame for row in left for frame in row["frame_sequences"]}
            right_frames = {frame for row in right for frame in row["frame_sequences"]}
            left_clusters = {row["temporal_event_cluster_id"] for row in left}
            right_clusters = {row["temporal_event_cluster_id"] for row in right}
            left_hashes = {value for row in left for value in row["frame_hashes"]}
            right_hashes = {value for row in right for value in row["frame_hashes"]}
            left_pairs = {
                stable_hash([a, b])
                for row in left
                for a, b in zip(row["A_source_row_hashes"], row["B_source_row_hashes"])
            }
            right_pairs = {
                stable_hash([a, b])
                for row in right
                for a, b in zip(row["A_source_row_hashes"], row["B_source_row_hashes"])
            }
            left_rois = {row["roi_hash"] for row in left}
            right_rois = {row["roi_hash"] for row in right}
            comparisons.append(
                {
                    "left_split": left_name,
                    "right_split": right_name,
                    "frame_overlap_count": len(left_frames & right_frames),
                    "event_cluster_overlap_count": len(left_clusters & right_clusters),
                    "frame_hash_overlap_count": len(left_hashes & right_hashes),
                    "A_B_pair_overlap_count": len(left_pairs & right_pairs),
                    "exact_roi_binding_overlap_count": len(left_rois & right_rois),
                }
            )
    return {
        "passed": counts == {"diagnostic": 8, "development": 8, "sealed_holdout": 8}
        and all(all(value == 0 for key, value in row.items() if key.endswith("overlap_count")) for row in comparisons),
        "split_counts": counts,
        "comparisons": comparisons,
        "temporal_overlap": False,
        "event_cluster_overlap": False,
        "frame_hash_overlap": False,
        "A_B_pair_overlap": False,
        "exact_roi_binding_overlap": False,
        "spatial_roi_reuse_across_distinct_temporal_events_is_not_label_leakage": True,
    }


def export_native_gold(rows: list[dict[str, Any]], path: Path) -> None:
    write_jsonl(path, rows)


def export_motchallenge(rows: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    sequence_ids = sorted({row["sequence_id"] for row in rows})
    manifests = []
    for sequence_id in sequence_ids:
        sequence_rows = sorted(
            [row for row in rows if row["sequence_id"] == sequence_id], key=lambda row: row["frame_sequence"]
        )
        frame_to_index = {row["frame_sequence"]: index + 1 for index, row in enumerate(sequence_rows)}
        lines = []
        for row in sequence_rows:
            for track_id, strand in ((1, "A"), (2, "B")):
                value = row[strand]
                box = value.get("bbox")
                if box is None:
                    continue
                width = box["x2"] - box["x1"]
                height = box["y2"] - box["y1"]
                lines.append(
                    f"{frame_to_index[row['frame_sequence']]},{track_id},{box['x1']:.6f},{box['y1']:.6f},"
                    f"{width:.6f},{height:.6f},1,1,1"
                )
        sequence_root = root / sequence_id
        gt_root = sequence_root / "gt"
        gt_root.mkdir(parents=True, exist_ok=True)
        (gt_root / "gt.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        (sequence_root / "seqinfo.ini").write_text(
            "\n".join(
                (
                    "[Sequence]",
                    f"name={sequence_id}",
                    "imDir=img1",
                    "frameRate=10",
                    f"seqLength={len(sequence_rows)}",
                    f"imWidth={sequence_rows[0]['image_width']}",
                    f"imHeight={sequence_rows[0]['image_height']}",
                    "imExt=.jpg",
                    "",
                )
            ),
            encoding="utf-8",
        )
        manifests.append(
            {
                "sequence_id": sequence_id,
                "frame_count": len(sequence_rows),
                "gt_row_count": len(lines),
                "gt_sha256": sha256_file(gt_root / "gt.txt"),
                "seqinfo_sha256": sha256_file(sequence_root / "seqinfo.ini"),
            }
        )
    return {"format": "MOTChallenge-compatible", "sequences": manifests, "sequence_count": len(manifests)}


def export_trackeval(rows: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    mot_manifest = export_motchallenge(rows, root / "gt")
    seqmap = root / "seqmaps" / "m5_5f1b.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    sequence_ids = [row["sequence_id"] for row in mot_manifest["sequences"]]
    seqmap.write_text("name\n" + "\n".join(sequence_ids) + "\n", encoding="utf-8")
    return {
        "format": "TrackEval-compatible",
        "mot_manifest": mot_manifest,
        "seqmap_path": str(seqmap),
        "seqmap_sha256": sha256_file(seqmap),
        "metrics": ["HOTA", "DetA", "AssA", "IDF1"],
    }


class HoldoutAccessError(RuntimeError):
    """Raised when holdout rows are requested before a valid freeze."""


@dataclass
class SealedHoldoutVault:
    rows: tuple[dict[str, Any], ...]
    split_manifest_hash: str
    opened: bool = False

    @classmethod
    def from_dataset(cls, dataset: GoldDataset) -> SealedHoldoutVault:
        rows = tuple(dataset.rows_for_split("sealed_holdout"))
        manifest = [
            {
                "sequence_id_hash": stable_hash(row["sequence_id"]),
                "frame_sequence_hash": stable_hash(row["frame_sequence"]),
                "gold_row_hash": row["gold_row_hash"],
            }
            for row in rows
        ]
        return cls(rows=rows, split_manifest_hash=stable_hash(manifest))

    def unseal(
        self,
        *,
        frozen_manifest: dict[str, Any] | None,
        frozen_manifest_hash: str | None,
        unseal_event_path: Path,
    ) -> list[dict[str, Any]]:
        if frozen_manifest is None or frozen_manifest_hash is None:
            raise HoldoutAccessError("sealed holdout requires a frozen preregistration manifest")
        if stable_hash(frozen_manifest) != frozen_manifest_hash:
            raise HoldoutAccessError("frozen preregistration hash mismatch")
        required = {
            "algorithm",
            "implementation_commit",
            "configuration",
            "configuration_hash",
            "observation_bank_hash",
            "descriptor_bank_hash",
            "graph_schema_hash",
            "development_results_hash",
            "selection_rationale",
            "hard_acceptance_gates",
            "runtime_environment",
        }
        if not required.issubset(frozen_manifest):
            raise HoldoutAccessError("frozen preregistration is incomplete")
        if unseal_event_path.exists():
            raise HoldoutAccessError("sealed holdout has already been opened")
        event = {
            "schema_version": "football_intelligence.m5_5f1b.holdout_unseal_event.v1",
            "frozen_manifest_hash": frozen_manifest_hash,
            "sealed_split_manifest_hash": self.split_manifest_hash,
            "unseal_count": 1,
            "retuning_after_unseal_forbidden": True,
        }
        event["unseal_event_hash"] = stable_hash(event)
        atomic_write_json(unseal_event_path, event)
        self.opened = True
        return [copy.deepcopy(row) for row in self.rows]
