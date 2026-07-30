"""R7 atomic scene-transition reviewer with immutable R1-R6 compatibility."""

from __future__ import annotations

import json
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from football_intelligence import g7d_c1_r3_loaded_review as r3
from football_intelligence.g7d_c1_r2_calibrated_review import CalibratedReviewStore
from football_intelligence.g7d_c1_r6_live_scene_review import LiveSceneReviewStore, REVISION as R6_REVISION

REVISION = "G7D_C1_R7_ATOMIC_SCENE_TRANSITION_REVIEW_V1"


class AtomicTransitionReviewStore(LiveSceneReviewStore):
    """Retain R6 truth and expose R7 drafts used as transition checkpoints."""

    review_revision = REVISION
    compatible_revisions = (*LiveSceneReviewStore.compatible_revisions, REVISION)

    def __init__(self, package: Path):
        r3.LoadedReviewStore.__init__(self, package)
        document = json.loads((package / "scene_candidate_overlays.json").read_text(encoding="utf-8"))
        if document.get("review_revision") != REVISION:
            raise RuntimeError("R7 overlay package revision mismatch")
        self.overlays = {row["scene_id"]: row for row in document["scenes"]}
        if set(self.overlays) != set(self.by_scene):
            raise RuntimeError("R7 overlay scene coverage mismatch")

    def state(self) -> dict[str, Any]:
        state = CalibratedReviewStore.state(self)
        retained: dict[str, Any] = {}
        discarded = 0
        for key, draft in state["drafts"].items():
            valid_key = key in self.by_target or key in self.by_scene
            valid_step = isinstance(draft.get("step_index"), int) and draft["step_index"] >= 0
            if draft.get("revision") in {*self.compatible_revisions, R6_REVISION} and valid_key and valid_step:
                retained[key] = draft
            else:
                discarded += 1
        state["drafts"] = retained
        state["discarded_stale_draft_count"] = discarded
        return state


def next_incomplete_scene(cases: list[dict[str, Any]], saved: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first server-unsaved scene in frozen case order."""
    return next((case for case in cases if case["scene_id"] not in saved), None)


def create_server(package: Path, port: int = 0) -> ThreadingHTTPServer:
    return r3.create_server(package, port, AtomicTransitionReviewStore)


def serve(package: Path, port: int = 8814) -> None:
    server = create_server(package, port)
    print(f"R7 atomic-transition reviewer listening on http://127.0.0.1:{server.server_port}/")
    server.serve_forever()
