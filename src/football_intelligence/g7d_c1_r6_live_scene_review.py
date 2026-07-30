"""R6 live full-frame scene reviewer with immutable R1-R5 compatibility."""

from __future__ import annotations

import json
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from football_intelligence import g7d_c1_r3_loaded_review as r3
from football_intelligence.g7d_c1_r2_calibrated_review import CalibratedReviewStore

REVISION = "G7D_C1_R6_LIVE_FULL_FRAME_SCENE_REVIEW_V1"
R4_REVISION = "G7D_C1_R4_STABLE_BOOT_NOVICE_REVIEW_V1"
R5_REVISION = "G7D_C1_R5_FULL_FRAME_SCENE_REVIEW_V1"


class LiveSceneReviewStore(r3.LoadedReviewStore):
    """Expose exact B3 overlays while preserving acknowledged earlier revisions."""

    review_revision = REVISION
    compatible_revisions = (*r3.LoadedReviewStore.compatible_revisions, R4_REVISION, R5_REVISION, REVISION)

    def __init__(self, package: Path):
        super().__init__(package)
        overlay_path = package / "scene_candidate_overlays.json"
        document = json.loads(overlay_path.read_text(encoding="utf-8"))
        if document.get("review_revision") != REVISION:
            raise RuntimeError("R6 overlay package revision mismatch")
        self.overlays = {row["scene_id"]: row for row in document["scenes"]}
        if set(self.overlays) != set(self.by_scene):
            raise RuntimeError("R6 overlay scene coverage mismatch")

    def state(self) -> dict[str, Any]:
        state = CalibratedReviewStore.state(self)
        retained: dict[str, Any] = {}
        discarded = 0
        for key, draft in state["drafts"].items():
            valid_key = key in self.by_target or key in self.by_scene
            valid_step = isinstance(draft.get("step_index"), int) and draft["step_index"] >= 0
            if draft.get("revision") in {R4_REVISION, R5_REVISION, REVISION} and valid_key and valid_step:
                retained[key] = draft
            else:
                discarded += 1
        state["drafts"] = retained
        state["discarded_stale_draft_count"] = discarded
        return state

    def scene_detail(self, scene_id: str) -> dict[str, Any] | None:
        detail = super().scene_detail(scene_id)
        if detail is None:
            return None
        overlay = self.overlays[scene_id]
        detail["scene"] = {
            **detail["scene"],
            "scene_candidate_overlays": overlay["candidates"],
            "scene_candidate_count": overlay["candidate_count"],
            "frame_provenance": overlay["frame_provenance"],
        }
        return detail


def create_server(package: Path, port: int = 0) -> ThreadingHTTPServer:
    return r3.create_server(package, port, LiveSceneReviewStore)


def serve(package: Path, port: int = 8814) -> None:
    server = create_server(package, port)
    print(f"R6 live scene reviewer listening on http://127.0.0.1:{server.server_port}/")
    server.serve_forever()
