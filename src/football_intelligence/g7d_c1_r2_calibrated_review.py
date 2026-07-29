"""R2 calibration gate layered on the immutable C1 reviewer protocol."""

from __future__ import annotations

import json
from http import HTTPStatus
from pathlib import Path
from typing import Any, Mapping

from football_intelligence import g7d_c1_r1_novice_review as r1

REVISION = "G7D_C1_R2_CALIBRATED_TARGET_BOX_NOVICE_REVIEW_V1"


class CalibratedReviewStore(r1.ReviewStore):
    """Reject final truth unless the installed all-target mapping audit is valid."""

    review_revision = REVISION
    compatible_revisions = (r1.LEGACY_REVISION, r1.REVISION, REVISION)

    def __init__(self, package: Path):
        super().__init__(package)
        status_path = package / "target_box_calibration_status.json"
        if not status_path.is_file():
            raise RuntimeError("R2 reviewer package has no target-box calibration status")
        self.calibration = json.loads(status_path.read_text(encoding="utf-8"))
        self._calibration_problem = self._calibration_error()

    def _calibration_error(self) -> str | None:
        expected = {target_id for target_id in self.by_target}
        audited = set(self.calibration.get("target_ids", []))
        if self.calibration.get("review_revision") != REVISION:
            return "installed calibration revision mismatch"
        if self.calibration.get("verified") is not True:
            return "target mapping was not verified"
        if self.calibration.get("target_count") != len(expected) or audited != expected:
            return "target mapping audit does not bind every target"
        if self.calibration.get("failure_count") != 0:
            return "target mapping audit contains failures"
        return None

    def _mapping_response(self) -> tuple[int, dict[str, Any]]:
        return (
            HTTPStatus.CONFLICT,
            r1.error(
                "TARGET_MAPPING_NOT_VERIFIED",
                "target_mapping",
                "This box could not be positioned safely. Please stop and report it.",
                reason=self._calibration_problem,
            ),
        )

    def save(self, payload: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        if self._calibration_problem:
            return self._mapping_response()
        return super().save(payload)

    def complete(self, payload: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        if self._calibration_problem:
            return self._mapping_response()
        return super().complete(payload)

    def state(self) -> dict[str, Any]:
        state = super().state()
        state["target_mapping"] = {
            "verified": self._calibration_problem is None,
            "target_count": self.calibration.get("target_count", 0),
            "failure_count": self.calibration.get("failure_count", -1),
            "plain_error": (
                None
                if self._calibration_problem is None
                else "This box could not be positioned safely. Please stop and report it."
            ),
        }
        return state


def serve(package: Path, port: int = 8814) -> None:
    r1.serve(package, port, CalibratedReviewStore)
