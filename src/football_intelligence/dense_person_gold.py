"""Candidate-blind dense-person annotation and discrimination primitives.

The module is deliberately independent from detector inference. Candidate data
enters only through the post-finalization reveal and metric functions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from football_intelligence.review_chassis.hashing import stable_hash


EVENT_SCHEMA = "football_intelligence.g7f_c.dense_person_frame_annotation.v1"
ACK_SCHEMA = "football_intelligence.g7f_c.dense_person_frame_acknowledgement.v1"
RELEVANCE_CLASSES = {"MATCH_RELEVANT", "NON_MATCH_RELEVANT", "RELEVANCE_UNCERTAIN"}
PRIMARY_RELEVANCE_CLASSES = {"MATCH_RELEVANT", "NON_MATCH_RELEVANT"}
IGNORE_REASONS = {
    "DENSE_CROWD_UNRESOLVABLE",
    "BENCH_CLUSTER_UNRESOLVABLE",
    "TINY_AMBIGUOUS_PERSON_CLUSTER",
    "SEVERE_VISUAL_ARTIFACT",
    "OTHER_NON_EVALUABLE_REGION",
}
COMPLETION_ASSERTION = "I have reviewed the full image and annotated every individually evaluable visible human."
MIN_VISIBLE_BOX_HEIGHT_PX = 6
MIN_VISIBLE_MASK_AREA_PX = 12
EXHAUSTIVENESS_STRIPS = 8
PASS_KINDS = {"FIRST_PASS", "BLIND_REPEAT"}


class DensePersonValidationError(ValueError):
    pass


class DensePersonConflictError(RuntimeError):
    def __init__(self, code: str, message: str, *, current_revision: int | None = None):
        super().__init__(message)
        self.code = code
        self.current_revision = current_revision


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, newline="\n") as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _point(value: Mapping[str, Any], width: int, height: int) -> dict[str, float]:
    try:
        x, y = float(value["x"]), float(value["y"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise DensePersonValidationError("polygon vertices require finite x/y coordinates") from exc
    if not math.isfinite(x) or not math.isfinite(y):
        raise DensePersonValidationError("polygon vertices require finite x/y coordinates")
    if not 0 <= x <= width - 1 or not 0 <= y <= height - 1:
        raise DensePersonValidationError("polygon vertex outside source image")
    return {"x": round(x, 6), "y": round(y, 6)}


def _signed_area(points: Sequence[Mapping[str, float]]) -> float:
    return (
        sum(
            float(point["x"]) * float(points[(index + 1) % len(points)]["y"])
            - float(points[(index + 1) % len(points)]["x"]) * float(point["y"])
            for index, point in enumerate(points)
        )
        / 2
    )


def _orientation(a: Mapping[str, float], b: Mapping[str, float], c: Mapping[str, float]) -> float:
    return (b["x"] - a["x"]) * (c["y"] - a["y"]) - (b["y"] - a["y"]) * (c["x"] - a["x"])


def _segments_cross(
    a: Mapping[str, float], b: Mapping[str, float], c: Mapping[str, float], d: Mapping[str, float]
) -> bool:
    epsilon = 1e-9
    ab_c, ab_d = _orientation(a, b, c), _orientation(a, b, d)
    cd_a, cd_b = _orientation(c, d, a), _orientation(c, d, b)
    return ((ab_c > epsilon and ab_d < -epsilon) or (ab_c < -epsilon and ab_d > epsilon)) and (
        (cd_a > epsilon and cd_b < -epsilon) or (cd_a < -epsilon and cd_b > epsilon)
    )


def _self_intersects(points: Sequence[Mapping[str, float]]) -> bool:
    count = len(points)
    for left in range(count):
        left_next = (left + 1) % count
        for right in range(left + 1, count):
            right_next = (right + 1) % count
            if left in {right, right_next} or left_next in {right, right_next}:
                continue
            if _segments_cross(points[left], points[left_next], points[right], points[right_next]):
                return True
    return False


def canonicalize_polygon(points: Sequence[Mapping[str, Any]], *, width: int, height: int) -> list[dict[str, float]]:
    normalized = [_point(point, width, height) for point in points]
    if len(normalized) > 1 and normalized[0] == normalized[-1]:
        normalized.pop()
    if len(normalized) < 3 or len({(row["x"], row["y"]) for row in normalized}) < 3:
        raise DensePersonValidationError("polygon requires at least three distinct vertices")
    if abs(_signed_area(normalized)) < 1e-9:
        raise DensePersonValidationError("polygon has zero continuous area")
    if _self_intersects(normalized):
        raise DensePersonValidationError("self-intersecting polygon")
    sequences = []
    for ordered in (normalized, list(reversed(normalized))):
        for index in range(len(ordered)):
            sequences.append(ordered[index:] + ordered[:index])
    return min(sequences, key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")))


def _rasterize(polygons: Sequence[Sequence[Mapping[str, float]]], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        vertices = np.rint([[point["x"], point["y"]] for point in polygon]).astype(np.int32)
        cv2.fillPoly(mask, [vertices], color=1, lineType=cv2.LINE_8)
    return mask


def coco_uncompressed_rle(mask: np.ndarray) -> dict[str, Any]:
    flattened = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F") > 0
    transitions = np.flatnonzero(flattened[1:] != flattened[:-1]) + 1
    boundaries = np.concatenate(([0], transitions, [flattened.size]))
    counts = np.diff(boundaries).astype(int).tolist()
    if flattened.size and bool(flattened[0]):
        counts.insert(0, 0)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def canonical_mask_geometry(
    components: Sequence[Sequence[Mapping[str, Any]]], *, width: int, height: int
) -> dict[str, Any]:
    canonical = [canonicalize_polygon(component, width=width, height=height) for component in components]
    canonical.sort(key=stable_hash)
    mask = _rasterize(canonical, width, height)
    ys, xs = np.nonzero(mask)
    area = int(mask.sum())
    if area == 0:
        raise DensePersonValidationError("visible mask rasterizes to zero pixels")
    box = {"x1": int(xs.min()), "y1": int(ys.min()), "x2": int(xs.max()) + 1, "y2": int(ys.max()) + 1}
    rle = coco_uncompressed_rle(mask)
    return {
        "canonical_components": canonical,
        "component_count": len(canonical),
        "binary_mask_sha256": sha256_bytes(mask.tobytes(order="C")),
        "coco_uncompressed_rle": rle,
        "rle_sha256": sha256_bytes(canonical_json_bytes(rle)),
        "derived_visible_box_xyxy": box,
        "visible_mask_area_px": area,
        "derived_visible_box_height_px": box["y2"] - box["y1"],
        "rasterizer": {
            "library": "opencv-python",
            "version": cv2.__version__,
            "vertex_quantization": "numpy.rint_ties_to_even",
            "pixel_inclusion": "cv2.fillPoly_LINE_8_boundary_included",
            "holes": "FORBIDDEN",
            "mask_bytes": "UINT8_C_CONTIGUOUS_HEIGHT_WIDTH",
            "rle_order": "COCO_UNCOMPRESSED_FORTRAN_COLUMN_MAJOR",
        },
    }


def canonicalize_person(person: Mapping[str, Any], *, width: int, height: int) -> dict[str, Any]:
    instance_id = str(person.get("instance_id", "")).strip()
    relevance = str(person.get("relevance", ""))
    components = person.get("visible_mask_components")
    if not instance_id:
        raise DensePersonValidationError("person instance_id required")
    if relevance not in RELEVANCE_CLASSES:
        raise DensePersonValidationError("person relevance class required")
    if not isinstance(components, list) or not components:
        raise DensePersonValidationError("person requires one or more visible mask components")
    geometry = canonical_mask_geometry(components, width=width, height=height)
    if geometry["visible_mask_area_px"] < MIN_VISIBLE_MASK_AREA_PX:
        raise DensePersonValidationError("person visible mask area below 12 source pixels")
    if geometry["derived_visible_box_height_px"] < MIN_VISIBLE_BOX_HEIGHT_PX:
        raise DensePersonValidationError("person visible box height below 6 source pixels")
    return {
        "instance_id": instance_id,
        "relevance": relevance,
        **geometry,
    }


def canonicalize_ignore_region(region: Mapping[str, Any], *, width: int, height: int) -> dict[str, Any]:
    region_id = str(region.get("ignore_region_id", "")).strip()
    reason = str(region.get("reason", ""))
    polygon = region.get("polygon")
    if not region_id:
        raise DensePersonValidationError("ignore region ID required")
    if reason not in IGNORE_REASONS:
        raise DensePersonValidationError("unknown ignore reason")
    if not isinstance(polygon, list):
        raise DensePersonValidationError("ignore region polygon required")
    geometry = canonical_mask_geometry([polygon], width=width, height=height)
    return {"ignore_region_id": region_id, "reason": reason, **geometry}


def validate_and_canonicalize_document(document: Mapping[str, Any], frame: Mapping[str, Any]) -> dict[str, Any]:
    width, height = int(frame["source_width"]), int(frame["source_height"])
    if document.get("unfinished_polygon") not in {None, False}:
        raise DensePersonValidationError("unfinished polygon blocks finalization")
    strips = document.get("reviewed_exhaustiveness_strips")
    if not isinstance(strips, list) or sorted(set(strips)) != list(range(EXHAUSTIVENESS_STRIPS)):
        raise DensePersonValidationError("all exactly eight exhaustiveness strips must be reviewed")
    if document.get("completion_assertion") != COMPLETION_ASSERTION:
        raise DensePersonValidationError("exact human completion assertion required")
    people_source = document.get("people", [])
    ignores_source = document.get("ignore_regions", [])
    if not isinstance(people_source, list) or not isinstance(ignores_source, list):
        raise DensePersonValidationError("people and ignore regions must be arrays")
    people = [canonicalize_person(person, width=width, height=height) for person in people_source]
    ignores = [canonicalize_ignore_region(region, width=width, height=height) for region in ignores_source]
    person_ids = [row["instance_id"] for row in people]
    ignore_ids = [row["ignore_region_id"] for row in ignores]
    if len(set(person_ids)) != len(person_ids) or len(set(ignore_ids)) != len(ignore_ids):
        raise DensePersonValidationError("instance and ignore IDs must be unique")
    people.sort(key=lambda row: row["instance_id"])
    ignores.sort(key=lambda row: row["ignore_region_id"])
    return {
        "people": people,
        "ignore_regions": ignores,
        "reviewed_exhaustiveness_strips": list(range(EXHAUSTIVENESS_STRIPS)),
        "strip_state": [
            {
                "strip_index": index,
                "source_x1": width * index / EXHAUSTIVENESS_STRIPS,
                "source_x2": width * (index + 1) / EXHAUSTIVENESS_STRIPS,
                "state": "REVIEWED_FOR_EXHAUSTIVENESS",
            }
            for index in range(EXHAUSTIVENESS_STRIPS)
        ],
        "completion_assertion": COMPLETION_ASSERTION,
        "zero_person_frame_allowed": True,
    }


def build_final_event(
    frame: Mapping[str, Any],
    document: Mapping[str, Any],
    *,
    binding_hashes: Mapping[str, str],
    reviewer_release: str,
    pass_kind: str,
    final_revision: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if pass_kind not in PASS_KINDS:
        raise DensePersonValidationError("unknown annotation pass kind")
    canonical = validate_and_canonicalize_document(document, frame)
    payload = {
        "schema_version": EVENT_SCHEMA,
        "anonymous_dense_image_id": frame["anonymous_dense_image_id"],
        "pass_kind": pass_kind,
        "selection_status": frame["selection_status"],
        "source_frame_sha256": frame["source_frame_sha256"],
        "source_width": frame["source_width"],
        "source_height": frame["source_height"],
        "all_frame_instance_lineage": frame["all_frame_instance_lineage"],
        "binding_hashes": dict(sorted(binding_hashes.items())),
        "reviewer_release": reviewer_release,
        "final_revision": final_revision,
        "annotation": canonical,
        "server_validation": {
            "candidate_data_used": False,
            "source_hash_bound": True,
            "source_coordinates_validated": True,
            "thresholds_enforced": True,
            "all_eight_strips_reviewed": True,
        },
        "immutable": True,
        "production_ready": False,
    }
    event_hash = sha256_bytes(canonical_json_bytes(payload))
    event = {**payload, "event_id": f"dense-person-{event_hash[:24]}", "event_sha256": event_hash}
    ack_payload = {
        "schema_version": ACK_SCHEMA,
        "event_id": event["event_id"],
        "event_sha256": event_hash,
        "anonymous_dense_image_id": frame["anonymous_dense_image_id"],
        "pass_kind": pass_kind,
        "status": "IMMUTABLY_FINALIZED",
    }
    acknowledgement = {
        **ack_payload,
        "acknowledgement_id": f"ack-{stable_hash(ack_payload)[:24]}",
        "acknowledgement_sha256": sha256_bytes(canonical_json_bytes(ack_payload)),
    }
    return event, acknowledgement


class DensePersonDecisionStore:
    """Revisioned, idempotent and immutable frame decision persistence."""

    def __init__(
        self,
        root: Path,
        *,
        frames: Mapping[str, Mapping[str, Any]],
        binding_hashes: Mapping[str, str],
        reviewer_release: str,
        reveal_payloads: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.frames = {str(key): dict(value) for key, value in frames.items()}
        self.binding_hashes = dict(binding_hashes)
        self.reviewer_release = reviewer_release
        self.reveal_payloads = copy.deepcopy(dict(reveal_payloads or {}))
        self._lock = threading.RLock()

    def _key(self, image_id: str, pass_kind: str) -> str:
        if image_id not in self.frames or pass_kind not in PASS_KINDS:
            raise DensePersonValidationError("unknown image or pass kind")
        return f"{pass_kind.lower()}__{image_id}"

    def _draft_path(self, key: str) -> Path:
        return self.root / "drafts" / f"{key}.json"

    def _event_path(self, key: str) -> Path:
        return self.root / "events" / f"{key}.json"

    def _ack_path(self, key: str) -> Path:
        return self.root / "acknowledgements" / f"{key}.json"

    def _action_path(self, action_id: str) -> Path:
        return self.root / "action_receipts" / f"{action_id}.json"

    def state(self, image_id: str, pass_kind: str = "FIRST_PASS") -> dict[str, Any]:
        key = self._key(image_id, pass_kind)
        event_path, draft_path = self._event_path(key), self._draft_path(key)
        if event_path.is_file():
            event = json.loads(event_path.read_text(encoding="utf-8"))
            return {
                "anonymous_dense_image_id": image_id,
                "pass_kind": pass_kind,
                "revision": int(event["final_revision"]),
                "finalized": True,
                "read_only": True,
                "document": event["annotation"],
                "candidate_reveal_available": image_id in self.reveal_payloads,
            }
        if draft_path.is_file():
            draft = json.loads(draft_path.read_text(encoding="utf-8"))
            return {**draft, "finalized": False, "read_only": False}
        return {
            "anonymous_dense_image_id": image_id,
            "pass_kind": pass_kind,
            "revision": 0,
            "finalized": False,
            "read_only": False,
            "document": {
                "people": [],
                "ignore_regions": [],
                "reviewed_exhaustiveness_strips": [],
                "unfinished_polygon": None,
                "completion_assertion": None,
            },
        }

    def apply_action(self, action: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            action_id = str(action.get("action_id", "")).strip()
            image_id = str(action.get("anonymous_dense_image_id", ""))
            pass_kind = str(action.get("pass_kind", "FIRST_PASS"))
            action_type = str(action.get("action_type", ""))
            if not action_id:
                raise DensePersonValidationError("action_id required")
            request_hash = sha256_bytes(canonical_json_bytes(dict(action)))
            receipt_path = self._action_path(action_id)
            if receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if receipt["request_sha256"] != request_hash:
                    raise DensePersonConflictError("ACTION_ID_REUSE", "action ID reused with different payload")
                return receipt["response"]
            key = self._key(image_id, pass_kind)
            current = self.state(image_id, pass_kind)
            expected_revision = action.get("expected_revision")
            if not isinstance(expected_revision, int) or expected_revision != current["revision"]:
                raise DensePersonConflictError(
                    "STALE_REVISION",
                    "expected revision differs from server state",
                    current_revision=current["revision"],
                )
            if action_type == "REVEAL_CANDIDATES":
                if not current["finalized"]:
                    raise DensePersonConflictError(
                        "REVEAL_BEFORE_FINALIZATION", "candidate reveal requires finalization"
                    )
                response = {
                    "action_id": action_id,
                    "revision": current["revision"],
                    "read_only": True,
                    "candidate_comparison": copy.deepcopy(self.reveal_payloads.get(image_id, {})),
                }
                _atomic_json(
                    self.root / "reveal_logs" / f"{key}__{action_id}.json",
                    {
                        "schema_version": "football_intelligence.g7f_c.candidate_reveal_log.v1",
                        "image_id": image_id,
                        "pass_kind": pass_kind,
                        "annotation_event_unchanged": True,
                        "action_id": action_id,
                    },
                )
            elif action_type in {"SAVE_DRAFT", "FINALIZE"}:
                if current["finalized"]:
                    raise DensePersonConflictError("FINALIZED_READ_ONLY", "finalized annotation is immutable")
                document = action.get("document")
                if not isinstance(document, Mapping):
                    raise DensePersonValidationError("annotation document required")
                new_revision = current["revision"] + 1
                if action_type == "SAVE_DRAFT":
                    response = {
                        "anonymous_dense_image_id": image_id,
                        "pass_kind": pass_kind,
                        "revision": new_revision,
                        "finalized": False,
                        "read_only": False,
                        "document": copy.deepcopy(dict(document)),
                    }
                    _atomic_json(self._draft_path(key), response)
                else:
                    event, ack = build_final_event(
                        self.frames[image_id],
                        document,
                        binding_hashes=self.binding_hashes,
                        reviewer_release=self.reviewer_release,
                        pass_kind=pass_kind,
                        final_revision=new_revision,
                    )
                    event_path = self._event_path(key)
                    if event_path.exists():
                        raise DensePersonConflictError("EVENT_ALREADY_EXISTS", "immutable event already exists")
                    _atomic_json(event_path, event)
                    _atomic_json(self._ack_path(key), ack)
                    response = {
                        "anonymous_dense_image_id": image_id,
                        "pass_kind": pass_kind,
                        "revision": new_revision,
                        "finalized": True,
                        "read_only": True,
                        "event_id": event["event_id"],
                        "event_sha256": event["event_sha256"],
                        "acknowledgement": ack,
                    }
            else:
                raise DensePersonValidationError("unknown action type")
            _atomic_json(
                receipt_path,
                {"request_sha256": request_hash, "response": response, "schema_version": "g7f_c.action_receipt.v1"},
            )
            return response


def _xyxy(box: Sequence[float] | Mapping[str, float]) -> tuple[float, float, float, float]:
    if isinstance(box, Mapping):
        return float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
    return tuple(float(value) for value in box)  # type: ignore[return-value]


def box_iou(left: Sequence[float] | Mapping[str, float], right: Sequence[float] | Mapping[str, float]) -> float:
    left, right = _xyxy(left), _xyxy(right)
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def candidate_ignored(candidate_box: Sequence[float], ignore_mask: np.ndarray) -> bool:
    centre_x = min(max(int(math.floor((candidate_box[0] + candidate_box[2]) / 2)), 0), ignore_mask.shape[1] - 1)
    centre_y = min(max(int(math.floor((candidate_box[1] + candidate_box[3]) / 2)), 0), ignore_mask.shape[0] - 1)
    if bool(ignore_mask[centre_y, centre_x]):
        return True
    x1, y1 = max(0, math.floor(candidate_box[0])), max(0, math.floor(candidate_box[1]))
    x2, y2 = (
        min(ignore_mask.shape[1], math.ceil(candidate_box[2])),
        min(ignore_mask.shape[0], math.ceil(candidate_box[3])),
    )
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    return bool(box_area and int(ignore_mask[y1:y2, x1:x2].sum()) / box_area >= 0.50)


def decode_coco_uncompressed_rle(rle: Mapping[str, Any]) -> np.ndarray:
    height, width = (int(value) for value in rle["size"])
    values: list[int] = []
    bit = 0
    for count in rle["counts"]:
        count = int(count)
        if count < 0:
            raise DensePersonValidationError("RLE counts must be nonnegative")
        values.extend([bit] * count)
        bit = 1 - bit
    if len(values) != height * width:
        raise DensePersonValidationError("RLE size/count mismatch")
    return np.asarray(values, dtype=np.uint8).reshape((height, width), order="F")


def _person_mask(person: Mapping[str, Any]) -> np.ndarray:
    if "mask" in person:
        return np.asarray(person["mask"], dtype=np.uint8)
    return decode_coco_uncompressed_rle(person["coco_uncompressed_rle"])


def _frame_ignore_mask(frame: Mapping[str, Any]) -> np.ndarray:
    if "ignore_mask" in frame:
        return np.asarray(frame["ignore_mask"], dtype=np.uint8)
    height, width = int(frame["source_height"]), int(frame["source_width"])
    output = np.zeros((height, width), dtype=np.uint8)
    for region in frame.get("ignore_regions", []):
        output |= decode_coco_uncompressed_rle(region["coco_uncompressed_rle"])
    return output


def _ap_101(recalls: Sequence[float], precisions: Sequence[float]) -> float:
    return (
        sum(
            max((p for r, p in zip(recalls, precisions, strict=True) if r >= threshold), default=0.0)
            for threshold in np.linspace(0, 1, 101)
        )
        / 101
    )


def evaluate_dense_boxes(
    frames: Sequence[Mapping[str, Any]],
    *,
    iou_thresholds: Sequence[float] = tuple(round(0.50 + index * 0.05, 2) for index in range(10)),
) -> dict[str, Any]:
    """Evaluate scored frames with COCO-style 101-point AP and fixed diagnostics."""

    scored = [frame for frame in frames if frame.get("selection_status") == "SCORED_DENSE_GOLD"]
    if len({frame["source_frame_sha256"] for frame in scored}) != len(scored):
        raise DensePersonValidationError("dense metrics require unique source images")
    threshold_reports = {}
    total_gt = sum(
        person["relevance"] in PRIMARY_RELEVANCE_CLASSES for frame in scored for person in frame.get("people", [])
    )
    for threshold in iou_thresholds:
        detections = []
        matched_by_frame: dict[str, set[int]] = {frame["source_frame_sha256"]: set() for frame in scored}
        for frame in scored:
            for candidate in frame.get("candidates", []):
                detections.append((float(candidate["confidence"]), str(candidate["candidate_id"]), frame, candidate))
        detections.sort(key=lambda row: (-row[0], row[1]))
        tp, fp, duplicates, ignored = [], [], 0, 0
        for _, _, frame, candidate in detections:
            candidate_box = candidate["box_xyxy"]
            ignore_mask = _frame_ignore_mask(frame)
            if candidate_ignored(candidate_box, ignore_mask):
                ignored += 1
                continue
            eligible = [
                (index, person)
                for index, person in enumerate(frame.get("people", []))
                if person["relevance"] in PRIMARY_RELEVANCE_CLASSES
            ]
            overlaps = sorted(
                ((box_iou(candidate_box, person["derived_visible_box_xyxy"]), index) for index, person in eligible),
                key=lambda row: (-row[0], row[1]),
            )
            key = frame["source_frame_sha256"]
            available = [
                (iou, index) for iou, index in overlaps if iou >= threshold and index not in matched_by_frame[key]
            ]
            if available:
                _, best_index = available[0]
                matched_by_frame[key].add(best_index)
                tp.append(1)
                fp.append(0)
            else:
                if threshold == 0.50 and any(iou >= 0.50 and index in matched_by_frame[key] for iou, index in overlaps):
                    duplicates += 1
                tp.append(0)
                fp.append(1)
        cumulative_tp = np.cumsum(tp).tolist()
        cumulative_fp = np.cumsum(fp).tolist()
        recalls = [value / total_gt if total_gt else 0.0 for value in cumulative_tp]
        precisions = [
            true / (true + false) if true + false else 0.0
            for true, false in zip(cumulative_tp, cumulative_fp, strict=True)
        ]
        threshold_reports[f"{threshold:.2f}"] = {
            "ap_101_point": round(_ap_101(recalls, precisions), 6),
            "recall": round(recalls[-1] if recalls else 0.0, 6),
            "matched_candidates": int(cumulative_tp[-1] if cumulative_tp else 0),
            "unmatched_candidates": int(cumulative_fp[-1] if cumulative_fp else 0),
            "ignored_candidates": ignored,
            "duplicate_candidates": duplicates if threshold == 0.50 else None,
        }
    ap_values = [row["ap_101_point"] for row in threshold_reports.values()]
    recall_values = [row["recall"] for row in threshold_reports.values()]
    merge_count = 0
    for frame in scored:
        person_masks = [
            _person_mask(person)
            for person in frame.get("people", [])
            if person["relevance"] in PRIMARY_RELEVANCE_CLASSES
        ]
        for candidate in frame.get("candidates", []):
            x1, y1, x2, y2 = [int(round(value)) for value in candidate["box_xyxy"]]
            qualifying = 0
            for mask in person_masks:
                intersection = int(mask[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)].sum())
                if mask.sum() and intersection / int(mask.sum()) >= 0.30:
                    qualifying += 1
            merge_count += qualifying >= 2
    total_candidates = sum(len(frame.get("candidates", [])) for frame in scored)
    at_50 = threshold_reports["0.50"]
    return {
        "schema_version": "football_intelligence.g7f_c.dense_gold_metric_result.v1",
        "DISAGREEMENT_ENRICHED_SAMPLE": True,
        "tuning_exposure": "DENSE_GOLD_INTERNAL_VALIDATION",
        "sample_size_images": len(scored),
        "evaluable_person_denominator": total_gt,
        "AP_50_95": round(sum(ap_values) / len(ap_values), 6),
        "AP50": threshold_reports["0.50"]["ap_101_point"],
        "AP75": threshold_reports["0.75"]["ap_101_point"],
        "recall_50_95": round(sum(recall_values) / len(recall_values), 6),
        "recall_50": threshold_reports["0.50"]["recall"],
        "recall_75": threshold_reports["0.75"]["recall"],
        "DUPLICATE_CANDIDATE_AT_IOU50": at_50["duplicate_candidates"],
        "MULTI_PERSON_CANDIDATE_MASK_COVERAGE_030": int(merge_count),
        "candidate_burden": {
            "candidate_rows": total_candidates,
            "candidate_rows_per_image": round(total_candidates / len(scored), 6) if scored else None,
            "matched_candidates_iou50": at_50["matched_candidates"],
            "unmatched_candidates_iou50": at_50["unmatched_candidates"],
            "duplicates_iou50": at_50["duplicate_candidates"],
            "candidate_to_gt_ratio": round(total_candidates / total_gt, 6) if total_gt else None,
        },
        "by_iou_threshold": threshold_reports,
        "whole_match_generalization_valid": False,
        "production_ready": False,
    }
