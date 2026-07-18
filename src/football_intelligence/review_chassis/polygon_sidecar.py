from __future__ import annotations

import math
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.review_chassis.hashing import stable_hash


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    text = __import__("json").dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def canonical_vertices(vertices: Any) -> list[dict[str, float]]:
    if not isinstance(vertices, list):
        raise ValueError("polygon vertices must be a list")
    result: list[dict[str, float]] = []
    for vertex in vertices:
        if not isinstance(vertex, dict):
            raise ValueError("polygon vertices must be objects")
        try:
            x = float(vertex["x"])
            y = float(vertex["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("polygon vertices require numeric x/y coordinates") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("polygon vertices must be finite")
        result.append({"x": round(x, 6), "y": round(y, 6)})
    return result


def _orientation(a: dict[str, float], b: dict[str, float], c: dict[str, float]) -> float:
    return (b["x"] - a["x"]) * (c["y"] - a["y"]) - (b["y"] - a["y"]) * (c["x"] - a["x"])


def _segments_intersect(a: dict[str, float], b: dict[str, float], c: dict[str, float], d: dict[str, float]) -> bool:
    eps = 1e-8
    ab1 = _orientation(a, b, c)
    ab2 = _orientation(a, b, d)
    cd1 = _orientation(c, d, a)
    cd2 = _orientation(c, d, b)

    def on_segment(p: dict[str, float], q: dict[str, float], r: dict[str, float]) -> bool:
        return (
            min(p["x"], r["x"]) - eps <= q["x"] <= max(p["x"], r["x"]) + eps
            and min(p["y"], r["y"]) - eps <= q["y"] <= max(p["y"], r["y"]) + eps
        )

    if abs(ab1) <= eps and on_segment(a, c, b):
        return True
    if abs(ab2) <= eps and on_segment(a, d, b):
        return True
    if abs(cd1) <= eps and on_segment(c, a, d):
        return True
    if abs(cd2) <= eps and on_segment(c, b, d):
        return True
    return (ab1 > eps) != (ab2 > eps) and (cd1 > eps) != (cd2 > eps)


def polygon_area(vertices: list[dict[str, float]]) -> float:
    return abs(
        sum(
            vertices[index]["x"] * vertices[(index + 1) % len(vertices)]["y"]
            - vertices[(index + 1) % len(vertices)]["x"] * vertices[index]["y"]
            for index in range(len(vertices))
        )
        / 2.0
    )


def validate_polygon(
    *,
    vertices: Any,
    tolerance_pixels: Any,
    source_image_hash: str,
    expected_source_image_hash: str,
    image_width: int,
    image_height: int,
    expected_width: int,
    expected_height: int,
) -> dict[str, Any]:
    points = canonical_vertices(vertices)
    if len(points) < 4:
        raise ValueError("pitch polygon requires at least four vertices")
    if source_image_hash != expected_source_image_hash:
        raise ValueError("source image hash mismatch")
    if (int(image_width), int(image_height)) != (int(expected_width), int(expected_height)):
        raise ValueError("source image dimensions mismatch")
    tolerance = float(tolerance_pixels)
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 100:
        raise ValueError("pitch-polygon tolerance must be between 0 and 100 pixels")
    for point in points:
        if not (0 <= point["x"] <= expected_width and 0 <= point["y"] <= expected_height):
            raise ValueError("polygon vertex is outside the source image")
    for index, point in enumerate(points):
        other = points[(index + 1) % len(points)]
        if math.hypot(point["x"] - other["x"], point["y"] - other["y"]) <= 1e-6:
            raise ValueError("adjacent polygon vertices must be distinct")
    for left in range(len(points)):
        left_end = (left + 1) % len(points)
        for right in range(left + 1, len(points)):
            right_end = (right + 1) % len(points)
            if left in {right, right_end} or left_end in {right, right_end}:
                continue
            if _segments_intersect(points[left], points[left_end], points[right], points[right_end]):
                raise ValueError("pitch polygon must not self-intersect")
    area = polygon_area(points)
    minimum_area = max(1.0, float(expected_width * expected_height) * 0.001)
    if area < minimum_area:
        raise ValueError("pitch polygon area is too small")
    diagonal = math.hypot(expected_width, expected_height)
    if any(
        math.hypot(
            points[index]["x"] - points[(index + 1) % len(points)]["x"],
            points[index]["y"] - points[(index + 1) % len(points)]["y"],
        )
        > diagonal * 2
        for index in range(len(points))
    ):
        raise ValueError("pitch polygon edge length is unbounded")
    return {
        "vertices_original_pixels": points,
        "tolerance_pixels": round(tolerance, 6),
        "source_image_hash": expected_source_image_hash,
        "source_dimensions": {"width": int(expected_width), "height": int(expected_height)},
        "area_pixels": area,
        "area_fraction": area / float(expected_width * expected_height),
        "validation_state": "VALID",
    }


class PolygonSidecarStore:
    """Mutable, match-local polygon state kept outside the immutable evidence package."""

    def __init__(
        self,
        root: Path,
        *,
        review_id: str,
        reviewer_session_id: str,
        match_id: str,
        proposal_vertices: list[dict[str, Any]],
        proposal_tolerance: float,
        proposal_polygon_hash: str,
        source_image_hash: str,
        image_width: int,
        image_height: int,
        immutable_package_manifest_hash: str,
        evidence_manifest_hash: str,
    ) -> None:
        self.root = root.resolve()
        self.review_id = review_id
        self.reviewer_session_id = reviewer_session_id
        self.match_id = match_id
        self.proposal_vertices = canonical_vertices(proposal_vertices)
        self.proposal_tolerance = float(proposal_tolerance)
        self.proposal_polygon_hash = proposal_polygon_hash
        self.source_image_hash = source_image_hash
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.immutable_package_manifest_hash = immutable_package_manifest_hash
        self.evidence_manifest_hash = evidence_manifest_hash
        self.lock = threading.RLock()

    @property
    def draft_path(self) -> Path:
        return self.root / "polygon_draft.json"

    @property
    def draft_events_path(self) -> Path:
        return self.root / "polygon_draft_events.jsonl"

    @property
    def snapshots_root(self) -> Path:
        return self.root / "polygon_draft_snapshots"

    @property
    def approved_path(self) -> Path:
        return self.root / "approved_polygon.json"

    @property
    def approval_events_path(self) -> Path:
        return self.root / "polygon_approval_events.jsonl"

    @property
    def approved_manifest_path(self) -> Path:
        return self.root / "approved_polygon_manifest.json"

    @property
    def proposal_contract(self) -> dict[str, Any]:
        return {
            "proposal_polygon_hash": self.proposal_polygon_hash,
            "source_image_hash": self.source_image_hash,
            "source_dimensions": {"width": self.image_width, "height": self.image_height},
            "immutable_package_manifest_hash": self.immutable_package_manifest_hash,
            "evidence_manifest_hash": self.evidence_manifest_hash,
        }

    def _draft_hash(self, vertices: list[dict[str, float]], tolerance: float) -> str:
        return stable_hash(
            {
                "source_image_hash": self.source_image_hash,
                "source_dimensions": {"width": self.image_width, "height": self.image_height},
                "vertices_original_pixels": vertices,
                "tolerance_pixels": tolerance,
            }
        )

    def _approved_hash(self, vertices: list[dict[str, float]], tolerance: float) -> str:
        return stable_hash(
            {
                "review_id": self.review_id,
                "source_image_hash": self.source_image_hash,
                "source_dimensions": {"width": self.image_width, "height": self.image_height},
                "vertices_original_pixels": vertices,
                "tolerance_pixels": tolerance,
            }
        )

    def _read(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        import json

        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}

    def _event(self, event_type: str, payload: dict[str, Any]) -> None:
        _append_jsonl(
            self.draft_events_path if event_type.startswith("draft_") else self.approval_events_path,
            {
                "schema_version": "football_intelligence.m5_5f1a2.polygon_event.v1",
                "event_type": event_type,
                "event_id": f"polygon_event_{uuid.uuid4().hex}",
                "timestamp": _now(),
                "review_id": self.review_id,
                "reviewer_session_id": self.reviewer_session_id,
                **payload,
            },
        )

    def ensure(self) -> dict[str, Any]:
        with self.lock:
            self.root.mkdir(parents=True, exist_ok=True)
            self.snapshots_root.mkdir(parents=True, exist_ok=True)
            self.draft_events_path.touch(exist_ok=True)
            self.approval_events_path.touch(exist_ok=True)
            draft = self._read(self.draft_path)
            if not draft:
                validated = validate_polygon(
                    vertices=self.proposal_vertices,
                    tolerance_pixels=self.proposal_tolerance,
                    source_image_hash=self.source_image_hash,
                    expected_source_image_hash=self.source_image_hash,
                    image_width=self.image_width,
                    image_height=self.image_height,
                    expected_width=self.image_width,
                    expected_height=self.image_height,
                )
                draft = self._draft_payload(
                    vertices=validated["vertices_original_pixels"],
                    tolerance=validated["tolerance_pixels"],
                    revision=0,
                    migration_source="immutable_proposal",
                    status="PROPOSAL",
                    validation=validated,
                )
                _atomic_json(self.draft_path, draft)
                _atomic_json(self.snapshots_root / "polygon_draft_000000.json", draft)
            if not self.approved_path.exists():
                _atomic_json(
                    self.approved_path,
                    {
                        "schema_version": "football_intelligence.m5_5f1a2.approved_polygon.v1",
                        "status": "UNAPPROVED",
                        **self.proposal_contract,
                    },
                )
            if not self.approved_manifest_path.exists():
                _atomic_json(
                    self.approved_manifest_path,
                    {
                        "schema_version": "football_intelligence.m5_5f1a2.approved_manifest.v1",
                        "status": "UNAPPROVED",
                        **self.proposal_contract,
                    },
                )
            return self.state()

    def _draft_payload(
        self,
        *,
        vertices: list[dict[str, float]],
        tolerance: float,
        revision: int,
        migration_source: str,
        status: str,
        validation: dict[str, Any],
    ) -> dict[str, Any]:
        now = _now()
        return {
            "schema_version": "football_intelligence.m5_5f1a2.polygon_draft.v1",
            "review_id": self.review_id,
            "reviewer_session_id": self.reviewer_session_id,
            "match_id": self.match_id,
            "source_image_hash": self.source_image_hash,
            "source_dimensions": {"width": self.image_width, "height": self.image_height},
            "proposal_polygon_hash": self.proposal_polygon_hash,
            "vertices_original_pixels": vertices,
            "tolerance_pixels": tolerance,
            "draft_polygon_hash": self._draft_hash(vertices, tolerance),
            "draft_revision": revision,
            "created_at": self._read(self.draft_path).get("created_at", now),
            "updated_at": now,
            "migration_source": migration_source,
            "status": status,
            "validation_state": validation,
        }

    def save_draft(self, payload: dict[str, Any], *, migration_source: str = "browser_autosave") -> dict[str, Any]:
        with self.lock:
            self.ensure()
            validated = validate_polygon(
                vertices=payload.get("vertices_original_pixels", payload.get("polygon_vertices")),
                tolerance_pixels=payload.get("tolerance_pixels"),
                source_image_hash=str(payload.get("source_image_hash", "")),
                expected_source_image_hash=self.source_image_hash,
                image_width=int(payload.get("image_width", self.image_width)),
                image_height=int(payload.get("image_height", self.image_height)),
                expected_width=self.image_width,
                expected_height=self.image_height,
            )
            current = self._read(self.draft_path)
            prior_approved = self._read(self.approved_path)
            revision = int(current.get("draft_revision", 0)) + 1
            draft = self._draft_payload(
                vertices=validated["vertices_original_pixels"],
                tolerance=validated["tolerance_pixels"],
                revision=revision,
                migration_source=migration_source,
                status="DRAFT",
                validation=validated,
            )
            _atomic_json(self.draft_path, draft)
            _atomic_json(self.snapshots_root / f"polygon_draft_{revision:06d}.json", draft)
            if prior_approved.get("status") == "APPROVED":
                superseded_at = _now()
                _atomic_json(
                    self.approved_path,
                    {
                        **prior_approved,
                        "status": "SUPERSEDED",
                        "approved": False,
                        "superseded_at": superseded_at,
                        "superseded_by_draft_revision": revision,
                    },
                )
                _atomic_json(
                    self.approved_manifest_path,
                    {
                        **self._read(self.approved_manifest_path),
                        "status": "SUPERSEDED",
                        "superseded_at": superseded_at,
                        "superseded_by_draft_revision": revision,
                    },
                )
                self._event(
                    "polygon_edit_started",
                    {
                        "prior_approved_polygon_hash": prior_approved.get("approved_polygon_hash"),
                        "draft_revision": revision,
                    },
                )
            self._event(
                "draft_saved",
                {
                    "draft_revision": revision,
                    "draft_polygon_hash": draft["draft_polygon_hash"],
                    "migration_source": migration_source,
                },
            )
            return self.state()

    def migrate_draft(self, payload: dict[str, Any]) -> dict[str, Any]:
        backup_root = self.root / "polygon_draft_migration_backups"
        backup_root.mkdir(parents=True, exist_ok=True)
        backup = backup_root / f"legacy_{uuid.uuid4().hex}.json"
        _atomic_json(backup, payload)
        return self.save_draft(payload, migration_source="same_origin_legacy_localstorage")

    def approve(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.lock:
            self.ensure()
            if payload:
                self.save_draft(payload, migration_source="before_approval")
                draft = self._read(self.draft_path)
            else:
                draft = self._read(self.draft_path)
            validated = validate_polygon(
                vertices=draft.get("vertices_original_pixels"),
                tolerance_pixels=draft.get("tolerance_pixels"),
                source_image_hash=str(draft.get("source_image_hash", "")),
                expected_source_image_hash=self.source_image_hash,
                image_width=int(draft.get("source_dimensions", {}).get("width", 0)),
                image_height=int(draft.get("source_dimensions", {}).get("height", 0)),
                expected_width=self.image_width,
                expected_height=self.image_height,
            )
            approved_hash = self._approved_hash(validated["vertices_original_pixels"], validated["tolerance_pixels"])
            approved = {
                "schema_version": "football_intelligence.m5_5f1a2.approved_polygon.v1",
                "status": "APPROVED",
                "approved": True,
                "approved_polygon_hash": approved_hash,
                "proposal_polygon_hash": self.proposal_polygon_hash,
                "source_image_hash": self.source_image_hash,
                "source_dimensions": {"width": self.image_width, "height": self.image_height},
                "vertices_original_pixels": validated["vertices_original_pixels"],
                "tolerance_pixels": validated["tolerance_pixels"],
                "draft_revision": draft.get("draft_revision", 0),
                "approved_at": _now(),
                "validation_results_hash": stable_hash(validated),
                "immutable_package_manifest_hash": self.immutable_package_manifest_hash,
                "evidence_manifest_hash": self.evidence_manifest_hash,
            }
            manifest_body = {
                "schema_version": "football_intelligence.m5_5f1a2.approved_manifest.v1",
                "status": "APPROVED",
                "approved_polygon_hash": approved_hash,
                "proposal_polygon_hash": self.proposal_polygon_hash,
                "source_image_hash": self.source_image_hash,
                "source_dimensions": {"width": self.image_width, "height": self.image_height},
                "tolerance_pixels": validated["tolerance_pixels"],
                "review_id": self.review_id,
                "reviewer_session_id": self.reviewer_session_id,
                "approval_timestamp": approved["approved_at"],
                "draft_revision": draft.get("draft_revision", 0),
                "validation_results_hash": approved["validation_results_hash"],
                "immutable_package_manifest_hash": self.immutable_package_manifest_hash,
                "evidence_manifest_hash": self.evidence_manifest_hash,
            }
            manifest_body["approved_polygon_manifest_hash"] = stable_hash(manifest_body)
            _atomic_json(self.approved_path, approved)
            _atomic_json(self.approved_manifest_path, manifest_body)
            self._event(
                "polygon_approved",
                {
                    "approved_polygon_hash": approved_hash,
                    "approved_polygon_manifest_hash": manifest_body["approved_polygon_manifest_hash"],
                    "draft_revision": draft.get("draft_revision", 0),
                },
            )
            return self.state()

    def revoke(self, reason: str = "reviewer_requested") -> dict[str, Any]:
        with self.lock:
            self.ensure()
            prior = self._read(self.approved_path)
            revoked = {
                **prior,
                "status": "REVOKED",
                "approved": False,
                "revoked_at": _now(),
                "revocation_reason": reason,
            }
            manifest = {
                **self._read(self.approved_manifest_path),
                "status": "REVOKED",
                "revoked_at": revoked["revoked_at"],
                "revocation_reason": reason,
            }
            _atomic_json(self.approved_path, revoked)
            _atomic_json(self.approved_manifest_path, manifest)
            self._event(
                "polygon_revoked", {"prior_approved_polygon_hash": prior.get("approved_polygon_hash"), "reason": reason}
            )
            return self.state()

    def state(self) -> dict[str, Any]:
        draft = self._read(self.draft_path)
        approved = self._read(self.approved_path)
        manifest = self._read(self.approved_manifest_path)
        return {
            "schema_version": "football_intelligence.m5_5f1a2.polygon_sidecar_state.v1",
            "draft": draft,
            "approved": approved,
            "approved_manifest": manifest,
            "approved_polygon_hash": approved.get("approved_polygon_hash")
            if approved.get("status") == "APPROVED"
            else None,
            "approved_polygon_manifest_hash": manifest.get("approved_polygon_manifest_hash")
            if manifest.get("status") == "APPROVED"
            else None,
            "is_approved": approved.get("status") == "APPROVED" and manifest.get("status") == "APPROVED",
            "is_revoked": approved.get("status") == "REVOKED",
            "proposal": self.proposal_contract,
        }
