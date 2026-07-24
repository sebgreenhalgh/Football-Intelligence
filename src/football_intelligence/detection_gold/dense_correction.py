from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.completion import (
    validate_completion_bundle,
    write_completion_transaction,
)
from football_intelligence.review_chassis.hashing import stable_hash
from football_intelligence.review_chassis.persistence import (
    GenericReviewPersistence,
    atomic_write_json,
    canonical_decision_state,
    synchronized,
    utc_now,
)


CORRECTION_SCHEMA = "C1_DENSE_MASK_GEOMETRY_CORRECTION_V1"
CORRECTION_TRANCHE_ID = "C1R_DENSE_MASK_GEOMETRY_REPAIR"
CORRECTED_DECISION = "CORRECTED_OUTLINE"
UNRELIABLE_DECISION = "UNRELIABLE_OUTLINE"
ALLOWED_COVERAGE = (0.0, 0.25, 0.5, 0.75, 1.0)
ALLOWED_QUALITY = {"PRECISE", "COARSE"}
ALLOWED_UNRELIABLE_QUALITY = {"UNCERTAIN", "IGNORE"}
ALLOWED_OCCLUSION_STATUS = {"ORDER_PRESERVED", "ORDER_CHANGED", "UNRESOLVED"}


def _point(value: Mapping[str, Any]) -> dict[str, float]:
    return {"x": float(value["x"]), "y": float(value["y"])}


def _same_point(left: Mapping[str, Any], right: Mapping[str, Any], *, epsilon: float = 1e-9) -> bool:
    return abs(float(left["x"]) - float(right["x"])) <= epsilon and abs(float(left["y"]) - float(right["y"])) <= epsilon


def _orientation(a: Mapping[str, Any], b: Mapping[str, Any], c: Mapping[str, Any]) -> float:
    return (float(b["x"]) - float(a["x"])) * (float(c["y"]) - float(a["y"])) - (float(b["y"]) - float(a["y"])) * (
        float(c["x"]) - float(a["x"])
    )


def _on_segment(a: Mapping[str, Any], b: Mapping[str, Any], c: Mapping[str, Any]) -> bool:
    epsilon = 1e-9
    return (
        min(float(a["x"]), float(b["x"])) - epsilon <= float(c["x"]) <= max(float(a["x"]), float(b["x"])) + epsilon
        and min(float(a["y"]), float(b["y"])) - epsilon <= float(c["y"]) <= max(float(a["y"]), float(b["y"])) + epsilon
        and abs(_orientation(a, b, c)) <= epsilon
    )


def segments_intersect(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    c: Mapping[str, Any],
    d: Mapping[str, Any],
) -> bool:
    """Return true for proper crossings, endpoint touches, and collinear overlap."""

    epsilon = 1e-9
    orientations = (_orientation(a, b, c), _orientation(a, b, d), _orientation(c, d, a), _orientation(c, d, b))
    if (
        (orientations[0] > epsilon and orientations[1] < -epsilon)
        or (orientations[0] < -epsilon and orientations[1] > epsilon)
    ) and (
        (orientations[2] > epsilon and orientations[3] < -epsilon)
        or (orientations[2] < -epsilon and orientations[3] > epsilon)
    ):
        return True
    return any(
        (
            abs(orientation) <= epsilon,
            _on_segment(left, right, point),
        )
        == (True, True)
        for orientation, left, right, point in (
            (orientations[0], a, b, c),
            (orientations[1], a, b, d),
            (orientations[2], c, d, a),
            (orientations[3], c, d, b),
        )
    )


def polygon_area(points: Sequence[Mapping[str, Any]]) -> float:
    if len(points) < 3:
        return 0.0
    return (
        abs(
            sum(
                float(point["x"]) * float(points[(index + 1) % len(points)]["y"])
                - float(points[(index + 1) % len(points)]["x"]) * float(point["y"])
                for index, point in enumerate(points)
            )
        )
        / 2.0
    )


def polygon_self_intersection_pairs(points: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    count = len(points)
    pairs: list[tuple[int, int]] = []
    if count < 4:
        return pairs
    for left in range(count):
        left_next = (left + 1) % count
        for right in range(left + 1, count):
            right_next = (right + 1) % count
            if left == right or left_next == right or right_next == left:
                continue
            if segments_intersect(points[left], points[left_next], points[right], points[right_next]):
                pairs.append((left, right))
    return pairs


def candidate_segment_crossings(
    points: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    *,
    close_polygon: bool = False,
) -> list[int]:
    """Identify existing edge indices crossed by one proposed editor segment."""

    if not points:
        return []
    start = points[-1] if not close_polygon else points[-1]
    end = points[0] if close_polygon else candidate
    maximum = len(points) - 1
    crossings = []
    for index in range(maximum):
        if index == maximum - 1:
            continue
        if close_polygon and index == 0:
            continue
        if segments_intersect(start, end, points[index], points[index + 1]):
            crossings.append(index)
    return crossings


def canonicalize_polygon(points: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    normalized = [{"x": round(float(point["x"]), 6), "y": round(float(point["y"]), 6)} for point in points]
    if len(normalized) > 1 and _same_point(normalized[0], normalized[-1]):
        normalized.pop()
    if not normalized:
        return []
    sequences: list[list[dict[str, float]]] = []
    for ordered in (normalized, list(reversed(normalized))):
        for index in range(len(ordered)):
            sequences.append(ordered[index:] + ordered[:index])
    return min(sequences, key=lambda row: json.dumps(row, separators=(",", ":"), sort_keys=True))


def polygon_hash(points: Sequence[Mapping[str, Any]]) -> str:
    return stable_hash({"coordinate_space": "canonical_panorama_pixels", "vertices": canonicalize_polygon(points)})


def tight_box(points: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        "x1": min(float(point["x"]) for point in points),
        "y1": min(float(point["y"]) for point in points),
        "x2": max(float(point["x"]) for point in points),
        "y2": max(float(point["y"]) for point in points),
    }


def validate_polygon_safe(
    points: Sequence[Mapping[str, Any]],
    *,
    focal_roi: Mapping[str, Any],
    image_width: int,
    image_height: int,
    minimum_area: float = 4.0,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        normalized = [_point(point) for point in points]
    except (KeyError, TypeError, ValueError, OverflowError):
        normalized = []
        errors.append("NON_NUMERIC_VERTEX")
    if normalized and not all(math.isfinite(point[axis]) for point in normalized for axis in ("x", "y")):
        errors.append("NON_FINITE_VERTEX")
    if len(normalized) < 3 or len({(point["x"], point["y"]) for point in normalized}) < 3:
        errors.append("FEWER_THAN_THREE_DISTINCT_VERTICES")
    if any(
        _same_point(normalized[index], normalized[(index + 1) % len(normalized)]) for index in range(len(normalized))
    ):
        errors.append("REPEATED_ADJACENT_VERTEX")
    if any(
        point["x"] < 0 or point["x"] > image_width or point["y"] < 0 or point["y"] > image_height
        for point in normalized
    ):
        errors.append("OUTSIDE_SOURCE_IMAGE")
    if any(
        point["x"] < float(focal_roi["x1"])
        or point["x"] > float(focal_roi["x2"])
        or point["y"] < float(focal_roi["y1"])
        or point["y"] > float(focal_roi["y2"])
        for point in normalized
    ):
        errors.append("OUTSIDE_FOCAL_ROI")
    intersections = polygon_self_intersection_pairs(normalized)
    if intersections:
        errors.append("SELF_INTERSECTION")
    area = polygon_area(normalized)
    if area < minimum_area:
        errors.append("INSUFFICIENT_AREA")
    canonical = canonicalize_polygon(normalized) if normalized else []
    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "self_intersection_edge_pairs": [
            {"left_edge_index": left, "right_edge_index": right} for left, right in intersections
        ],
        "area_pixels_squared": round(area, 6),
        "canonical_polygon_original_pixels": canonical,
        "polygon_hash": polygon_hash(canonical) if canonical else None,
        "tight_visible_box": tight_box(canonical) if canonical else None,
        "coordinate_space": "canonical_panorama_pixels",
        "silent_geometry_repair_performed": False,
    }


def _point_in_polygon(point: Mapping[str, Any], polygon: Sequence[Mapping[str, Any]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, vertex in enumerate(polygon):
        yi = float(vertex["y"])
        yj = float(polygon[j]["y"])
        if (yi > float(point["y"])) != (yj > float(point["y"])):
            crossing_x = (float(polygon[j]["x"]) - float(vertex["x"])) * (float(point["y"]) - yi) / (yj - yi) + float(
                vertex["x"]
            )
            if float(point["x"]) < crossing_x:
                inside = not inside
        j = i
    return inside


def polygons_overlap(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> bool:
    if not left or not right:
        return False
    for left_index, left_point in enumerate(left):
        left_next = left[(left_index + 1) % len(left)]
        for right_index, right_point in enumerate(right):
            if segments_intersect(left_point, left_next, right_point, right[(right_index + 1) % len(right)]):
                return True
    return _point_in_polygon(left[0], right) or _point_in_polygon(right[0], left)


def material_occlusion_dependencies(
    item: Mapping[str, Any], corrected_polygon: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    original = item["original_polygon_original_pixels"]
    rows = []
    for dependency in item.get("occlusion_dependencies", []):
        context_polygon = dependency["other_polygon_original_pixels"]
        original_overlap = polygons_overlap(original, context_polygon)
        corrected_overlap = polygons_overlap(corrected_polygon, context_polygon)
        inconsistent = bool(dependency.get("original_graph_inconsistent"))
        if original_overlap != corrected_overlap or inconsistent:
            rows.append(
                {
                    "other_mask_uuid": dependency["other_mask_uuid"],
                    "anonymous_label": dependency["anonymous_label"],
                    "original_overlap": original_overlap,
                    "corrected_overlap": corrected_overlap,
                    "original_graph_inconsistent": inconsistent,
                }
            )
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class DenseMaskCorrectionPersistence(GenericReviewPersistence):
    """Append-only persistence for the isolated C1 dense-mask correction overlay."""

    def state(self) -> dict[str, Any]:
        """Return the authoritative state with its optimistic-concurrency token."""

        materialized = self.state_path.exists()
        state = self.ensure_state() if materialized else self.empty_state()
        response = self._response(state)
        response["state_materialized"] = materialized
        if not materialized:
            response["server_state_hash"] = None
        return response

    def empty_state(self) -> dict[str, Any]:
        state = super().empty_state()
        state.update(
            {
                "schema_version": "football_intelligence.m5_5g4_r1.correction_decisions.v1",
                "correction_schema": CORRECTION_SCHEMA,
                "tranche_id": CORRECTION_TRANCHE_ID,
                "corrections": {},
                "original_c1_mutated": False,
            }
        )
        return state

    def _item(self, case_id: str, mask_uuid: str) -> tuple[Any, dict[str, Any]]:
        case = self.case_map().get(case_id)
        if case is None or case.task_type != "dense_mask_geometry_correction":
            raise ValueError("unknown dense-mask correction case")
        item = next(
            (
                row
                for row in case.visible_metadata.get("repair_items", [])
                if row.get("original_mask_uuid") == mask_uuid
            ),
            None,
        )
        if not isinstance(item, dict):
            raise ValueError("mask is not in the immutable flagged repair set")
        return case, item

    def _existing_event(self, idempotency_key: str) -> dict[str, Any] | None:
        return next(
            (event for event in _read_jsonl(self.events_path) if event.get("idempotency_key") == idempotency_key),
            None,
        )

    @staticmethod
    def _coverage_reviews(
        item: Mapping[str, Any], payload: Mapping[str, Any], *, unreliable: bool
    ) -> list[dict[str, Any]]:
        expected = {str(row["candidate_uuid"]): row for row in item.get("affected_candidates", [])}
        provided_rows = payload.get("candidate_coverage_reviews", [])
        if not isinstance(provided_rows, list):
            raise ValueError("candidate coverage reviews must be a list")
        provided = {str(row.get("candidate_uuid")): row for row in provided_rows if isinstance(row, dict)}
        if set(provided) != set(expected):
            raise ValueError("candidate coverage review set does not match the corrected mask dependencies")
        results = []
        for candidate_uuid in sorted(expected):
            prior = expected[candidate_uuid]
            review = provided[candidate_uuid]
            if unreliable:
                if review.get("review_status") != "EVIDENCE_UNRESOLVED":
                    raise ValueError("unreliable masks require unresolved candidate coverage")
                coverage = None
            else:
                coverage = float(review.get("candidate_visible_mask_coverage"))
                if coverage not in ALLOWED_COVERAGE:
                    raise ValueError("candidate coverage must use the frozen plain-language scale")
            results.append(
                {
                    "candidate_uuid": candidate_uuid,
                    "anonymous_label": prior["anonymous_label"],
                    "relation": prior["relation"],
                    "annotation_uuids": prior["annotation_uuids"],
                    "prior_candidate_visible_mask_coverage": prior.get("prior_candidate_visible_mask_coverage"),
                    "candidate_visible_mask_coverage": coverage,
                    "review_status": "REVALIDATED" if not unreliable else "EVIDENCE_UNRESOLVED",
                    "relation_preserved": True,
                }
            )
        return results

    @staticmethod
    def _occlusion_reviews(
        item: Mapping[str, Any],
        corrected_polygon: Sequence[Mapping[str, Any]],
        payload: Mapping[str, Any],
        *,
        unreliable: bool,
    ) -> list[dict[str, Any]]:
        required = (
            material_occlusion_dependencies(item, corrected_polygon)
            if not unreliable
            else [
                {
                    "other_mask_uuid": row["other_mask_uuid"],
                    "anonymous_label": row["anonymous_label"],
                    "original_overlap": None,
                    "corrected_overlap": None,
                    "original_graph_inconsistent": bool(row.get("original_graph_inconsistent")),
                }
                for row in item.get("occlusion_dependencies", [])
            ]
        )
        supplied_rows = payload.get("occlusion_reviews", [])
        if not isinstance(supplied_rows, list):
            raise ValueError("occlusion reviews must be a list")
        supplied = {str(row.get("other_mask_uuid")): row for row in supplied_rows if isinstance(row, dict)}
        if set(supplied) != {str(row["other_mask_uuid"]) for row in required}:
            raise ValueError("occlusion review set does not match material geometry dependencies")
        results = []
        for dependency in required:
            review = supplied[str(dependency["other_mask_uuid"])]
            status = "UNRESOLVED" if unreliable else str(review.get("status"))
            if status not in ALLOWED_OCCLUSION_STATUS:
                raise ValueError("invalid occlusion revalidation status")
            results.append({**dependency, "status": status, "reviewed_only_because_material": not unreliable})
        return results

    def _validate_correction(self, payload: Mapping[str, Any]) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        case_id = str(payload.get("case_id") or "")
        mask_uuid = str(payload.get("original_mask_uuid") or "")
        case, item = self._item(case_id, mask_uuid)
        decision = str(payload.get("decision") or "")
        if decision not in {CORRECTED_DECISION, UNRELIABLE_DECISION}:
            raise ValueError("invalid dense-mask correction decision")
        binding = case.visible_metadata["source_binding"]
        for key, expected in (
            ("source_frame_sha256", binding["source_frame_sha256"]),
            ("focal_transform_hash", binding["focal_transform_hash"]),
            ("original_polygon_hash", item["original_polygon_hash"]),
        ):
            if payload.get(key) != expected:
                raise ValueError(f"correction binding mismatch: {key}")

        unreliable = decision == UNRELIABLE_DECISION
        if unreliable:
            corrected_polygon: list[dict[str, float]] = []
            validation = {
                "valid": True,
                "errors": [],
                "review_state": "UNRELIABLE_EXCLUDED_FROM_MASK_IOU",
                "silent_geometry_repair_performed": False,
            }
            quality = str(payload.get("mask_quality") or "UNCERTAIN")
            if quality not in ALLOWED_UNRELIABLE_QUALITY:
                raise ValueError("unreliable masks must map to UNCERTAIN or IGNORE")
            if not str(payload.get("unreliable_reason") or "").strip():
                raise ValueError("unreliable masks require a structured reason")
            corrected_hash = None
            corrected_box = None
        else:
            raw_polygon = payload.get("corrected_polygon_original_pixels")
            if not isinstance(raw_polygon, list):
                raise ValueError("corrected polygon is required")
            validation = validate_polygon_safe(
                raw_polygon,
                focal_roi=binding["focal_roi_original_pixels"],
                image_width=int(binding["image_width"]),
                image_height=int(binding["image_height"]),
            )
            if not validation["valid"]:
                raise ValueError(f"corrected polygon is invalid: {validation['errors']}")
            corrected_polygon = validation["canonical_polygon_original_pixels"]
            corrected_hash = validation["polygon_hash"]
            corrected_box = validation["tight_visible_box"]
            quality = str(payload.get("mask_quality") or item["original_mask_quality"])
            if quality not in ALLOWED_QUALITY:
                raise ValueError("corrected masks must be PRECISE or COARSE")

        coverage = self._coverage_reviews(item, payload, unreliable=unreliable)
        occlusion = self._occlusion_reviews(item, corrected_polygon, payload, unreliable=unreliable)
        record = {
            "schema_version": "football_intelligence.m5_5g4_r1.correction_record.v1",
            "correction_schema": CORRECTION_SCHEMA,
            "correction_uuid": f"correction_{stable_hash([self.manifest.review_id, mask_uuid])[:24]}",
            "case_id": case_id,
            "dense_region_uuid": case.visible_metadata["dense_region_uuid"],
            "original_mask_uuid": mask_uuid,
            "source_frame_sha256": binding["source_frame_sha256"],
            "focal_roi_original_pixels": binding["focal_roi_original_pixels"],
            "focal_transform_hash": binding["focal_transform_hash"],
            "original_polygon_hash": item["original_polygon_hash"],
            "corrected_polygon_original_pixels": corrected_polygon or None,
            "corrected_polygon_hash": corrected_hash,
            "reviewer_session_id": self.reviewer_session_id,
            "decision": decision,
            "correction_reason": str(payload.get("correction_reason") or "SELF_INTERSECTION_REPAIR"),
            "unreliable_reason": payload.get("unreliable_reason") if unreliable else None,
            "validation": validation,
            "original_tight_visible_box": item["original_tight_visible_box"],
            "corrected_tight_visible_box": corrected_box,
            "original_mask_quality": item["original_mask_quality"],
            "mask_quality": quality,
            "affected_candidate_uuids": [row["candidate_uuid"] for row in coverage],
            "candidate_coverage_reviews": coverage,
            "occlusion_reviews": occlusion,
            "coverage_review_status": "UNRESOLVED" if unreliable else "COMPLETE",
            "occlusion_review_status": (
                "UNRESOLVED" if any(row["status"] == "UNRESOLVED" for row in occlusion) else "COMPLETE"
            ),
            "person_count_preserved": True,
            "excluded_from_mask_iou": unreliable,
            "original_mask_mutated": False,
        }
        return case, item, record

    def _response(self, state: dict[str, Any], *, duplicate: bool = False) -> dict[str, Any]:
        response = copy.deepcopy(state)
        response["counts"] = self.counts(state)
        response["resume_case_id"] = self.resume_case_id(state)
        response["server_state_hash"] = stable_hash(canonical_decision_state(state))
        response["server_event_sequence"] = int(state.get("event_sequence", 0))
        response["duplicate_event"] = duplicate
        return response

    @synchronized
    def save_correction(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = self.ensure_state()
        if state.get("completed") is True:
            raise ValueError("completed correction reviews are immutable")
        client_event_id = str(payload.get("client_event_id") or "")
        idempotency_key = str(payload.get("idempotency_key") or "")
        if not client_event_id or not idempotency_key:
            raise ValueError("client_event_id and idempotency_key are required")
        prior_event = self._existing_event(idempotency_key)
        if prior_event is not None:
            return self._response(state, duplicate=True)
        expected_hash = payload.get("expected_server_state_hash")
        current_hash = stable_hash(canonical_decision_state(state))
        if expected_hash not in (None, "", current_hash):
            raise ValueError("server state divergence; recover before saving")
        case, item, record = self._validate_correction(payload)
        mask_uuid = item["original_mask_uuid"]
        prior = state.setdefault("corrections", {}).get(mask_uuid)
        state["corrections"][mask_uuid] = record
        expected_case_masks = {str(row["original_mask_uuid"]) for row in case.visible_metadata.get("repair_items", [])}
        saved_case_masks = expected_case_masks & set(state["corrections"])
        if saved_case_masks == expected_case_masks:
            state.setdefault("decisions", {})[case.case_id] = "CASE_REPAIR_COMPLETE"
        else:
            state.setdefault("decisions", {}).pop(case.case_id, None)
        state["last_viewed_case_id"] = case.case_id
        state["elapsed_active_seconds"] = int(
            payload.get("elapsed_active_seconds", state.get("elapsed_active_seconds", 0))
        )
        event = self._event(
            event_type="DENSE_MASK_CORRECTION_SAVED",
            case_id=case.case_id,
            prior_decision=prior.get("decision") if isinstance(prior, dict) else None,
            new_decision=record["decision"],
            notes=None,
            state=state,
            input_source=str(payload.get("input_source", "dense_mask_correction_ui")),
            extra={
                "client_event_id": client_event_id,
                "idempotency_key": idempotency_key,
                "expected_server_state_hash": expected_hash,
                "original_mask_uuid": mask_uuid,
                "correction": record,
                "original_c1_mutated": False,
            },
        )
        record["event_sequence"] = event["event_sequence"]
        record["client_event_id"] = client_event_id
        record["idempotency_key"] = idempotency_key
        event["correction"] = record
        return self._response(self._persist(state, event))

    def completion_eligibility(
        self,
        state: dict[str, Any],
        *,
        pending_outbox_events: int = 0,
        unresolved_draft_count: int = 0,
    ) -> dict[str, Any]:
        expected = {
            str(item["original_mask_uuid"])
            for case in self.manifest.cases
            for item in case.visible_metadata.get("repair_items", [])
        }
        corrections = state.get("corrections", {})
        corrected = [row for row in corrections.values() if row.get("decision") == CORRECTED_DECISION]
        checks = {
            "exact_flagged_mask_set": set(corrections) == expected and len(expected) == 20,
            "all_corrected_polygons_valid": all(row.get("validation", {}).get("valid") for row in corrected),
            "zero_corrected_self_intersections": all(
                not row.get("validation", {}).get("self_intersection_edge_pairs") for row in corrected
            ),
            "candidate_coverage_revalidation_complete": all(
                row.get("coverage_review_status") in {"COMPLETE", "UNRESOLVED"} for row in corrections.values()
            ),
            "occlusion_revalidation_complete_or_unresolved": all(
                row.get("occlusion_review_status") in {"COMPLETE", "UNRESOLVED"} for row in corrections.values()
            ),
            "pending_outbox_empty": int(pending_outbox_events) == 0,
            "stale_drafts_absent": int(unresolved_draft_count) == 0,
            "original_c1_immutable": state.get("original_c1_mutated") is False,
        }
        return {
            "eligible": all(checks.values()) and state.get("completed") is not True,
            "already_completed": state.get("completed") is True,
            "checks": checks,
            "required_mask_count": len(expected),
            "saved_mask_count": len(corrections),
        }

    @synchronized
    def complete_corrections(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = self.ensure_state()
        if state.get("completed") is True:
            return {**self._response(state, duplicate=True), "bundle": validate_completion_bundle(self.decisions_root)}
        idempotency_key = str(payload.get("idempotency_key") or "")
        client_event_id = str(payload.get("client_event_id") or "")
        if not idempotency_key or not client_event_id:
            raise ValueError("completion requires client_event_id and idempotency_key")
        expected_hash = payload.get("expected_server_state_hash")
        if expected_hash not in (None, "", stable_hash(canonical_decision_state(state))):
            raise ValueError("server state divergence; recover before completion")
        eligibility = self.completion_eligibility(
            state,
            pending_outbox_events=int(payload.get("pending_outbox_events", 0)),
            unresolved_draft_count=int(payload.get("unresolved_draft_count", 0)),
        )
        if not eligibility["eligible"]:
            failed = [key for key, passed in eligibility["checks"].items() if not passed]
            raise ValueError(f"dense-mask correction completion is blocked: {failed}")

        before_state = self.state_path.read_bytes()
        before_events = self.events_path.read_bytes()
        prospective = copy.deepcopy(state)
        prospective["completed"] = True
        prospective["completed_at"] = utc_now()
        prospective["elapsed_active_seconds"] = int(
            payload.get("elapsed_active_seconds", prospective.get("elapsed_active_seconds", 0))
        )
        decision_state_hash = stable_hash(canonical_decision_state(prospective))
        transaction_id = f"dense_correction_{decision_state_hash[:32]}"
        event = self._event(
            event_type="REVIEW_COMPLETED",
            case_id=None,
            prior_decision=None,
            new_decision=None,
            notes=None,
            state=prospective,
            input_source=str(payload.get("input_source", "dense_mask_correction_ui")),
            extra={
                "client_event_id": client_event_id,
                "idempotency_key": idempotency_key,
                "completion_transaction_id": transaction_id,
                "completion_eligibility": eligibility,
            },
        )
        self._persist(prospective, event)
        persisted = self.ensure_state()
        summary_counts = self.counts(persisted)
        corrections = persisted["corrections"]
        corrected_count = sum(row["decision"] == CORRECTED_DECISION for row in corrections.values())
        unreliable_count = sum(row["decision"] == UNRELIABLE_DECISION for row in corrections.values())
        common = {
            "review_id": self.manifest.review_id,
            "stage_id": self.manifest.stage_id,
            "manifest_hash": self.manifest_hash_value,
            "ui_config_hash": self.ui_config_hash_value,
            "decision_state_hash": stable_hash(canonical_decision_state(persisted)),
            "completion_transaction_id": transaction_id,
        }
        summary = {
            "schema_version": "football_intelligence.m5_5g4_r1.completed_summary.v1",
            **common,
            **summary_counts,
            "completion_scope": "CORRECTION_TRANCHE",
            "completed": True,
            "repair_case_count": len(self.manifest.cases),
            "required_mask_count": 20,
            "corrected_mask_count": corrected_count,
            "unreliable_mask_count": unreliable_count,
            "pending_outbox_events": 0,
            "actual_human_active_minutes": round(float(persisted.get("elapsed_active_seconds", 0)) / 60, 3),
            "original_c1_mutated": False,
            **safety_payload(),
        }
        export = {
            "schema_version": "football_intelligence.m5_5g4_r1.completed_review.v1",
            **common,
            "created_at": utc_now(),
            "state": persisted,
            "summary": summary,
            **safety_payload(),
        }
        manifest = {
            "schema_version": "football_intelligence.m5_5g4_r1.completed_manifest.v1",
            **common,
            "case_ids": [case.case_id for case in self.manifest.cases],
            "original_mask_uuids": sorted(corrections),
            "correction_schema": CORRECTION_SCHEMA,
            "original_c1_mutated": False,
            **safety_payload(),
        }
        try:
            bundle = write_completion_transaction(
                decisions_root=self.decisions_root,
                completed_review=export,
                completed_events=self.events_path.read_bytes(),
                completed_manifest=manifest,
                completed_summary=summary,
            )
            overlay = {
                "schema_version": "football_intelligence.m5_5g4_r1.overlay_application_manifest.v1",
                "correction_schema": CORRECTION_SCHEMA,
                "completion_transaction_id": transaction_id,
                "correction_count": len(corrections),
                "correction_hashes": {mask_uuid: stable_hash(row) for mask_uuid, row in sorted(corrections.items())},
                "original_c1_mutated": False,
                "derived_dense_gold_v2_status": "READY_FOR_DETERMINISTIC_OVERLAY_APPLICATION",
                **safety_payload(),
            }
            ledger = {
                "schema_version": "football_intelligence.m5_5g4_r1.original_corrected_hash_ledger.v1",
                "rows": [
                    {
                        "original_mask_uuid": mask_uuid,
                        "original_polygon_hash": row["original_polygon_hash"],
                        "corrected_polygon_hash": row["corrected_polygon_hash"],
                        "decision": row["decision"],
                        "original_mask_mutated": False,
                    }
                    for mask_uuid, row in sorted(corrections.items())
                ],
                **safety_payload(),
            }
            atomic_write_json(self.decisions_root / "correction_overlay_application_manifest.json", overlay)
            atomic_write_json(self.decisions_root / "original_vs_corrected_hash_ledger.json", ledger)
        except Exception:
            self.state_path.write_bytes(before_state)
            self.events_path.write_bytes(before_events)
            for name in (
                "completed_review.json",
                "completed_review_events.jsonl",
                "completed_review_manifest.json",
                "completed_review_summary.json",
                "correction_overlay_application_manifest.json",
                "original_vs_corrected_hash_ledger.json",
            ):
                (self.decisions_root / name).unlink(missing_ok=True)
            raise
        return {**self._response(persisted), "bundle": bundle, "completion_eligibility": eligibility}


def apply_correction_overlay(
    original_annotations: Mapping[str, Mapping[str, Any]], corrections: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Apply reviewed corrections to a copy while retaining immutable lineage."""

    output = copy.deepcopy(dict(original_annotations))
    applied = 0
    for case_id, annotation in output.items():
        masks = annotation.get("visible_masks", [])
        for index, mask in enumerate(masks):
            correction = corrections.get(str(mask.get("annotation_uuid")))
            if correction is None:
                continue
            revised = copy.deepcopy(mask)
            revised["correction_overlay_lineage"] = {
                "correction_uuid": correction["correction_uuid"],
                "original_polygon_hash": correction["original_polygon_hash"],
                "decision": correction["decision"],
            }
            if correction["decision"] == CORRECTED_DECISION:
                revised["polygon_original_pixels"] = copy.deepcopy(correction["corrected_polygon_original_pixels"])
                revised["tight_visible_box"] = copy.deepcopy(correction["corrected_tight_visible_box"])
                revised["mask_quality"] = correction["mask_quality"]
            else:
                revised["mask_quality"] = correction["mask_quality"]
                revised["excluded_from_mask_iou"] = True
            masks[index] = revised
            applied += 1
    return {
        "schema_version": "C1_DENSE_GOLD_V2_APPLIED_OVERLAY",
        "annotations": output,
        "applied_correction_count": applied,
        "original_annotations_mutated": False,
        **safety_payload(),
    }


def iter_original_mask_hashes(annotations: Mapping[str, Mapping[str, Any]]) -> Iterable[tuple[str, str]]:
    for annotation in annotations.values():
        for mask in annotation.get("visible_masks", []):
            yield str(mask["annotation_uuid"]), polygon_hash(mask["polygon_original_pixels"])
