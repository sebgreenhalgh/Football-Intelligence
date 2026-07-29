"""R5 full-frame scene-review server identity."""

from __future__ import annotations

from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from football_intelligence.g7d_c1_r2_calibrated_review import CalibratedReviewStore
from football_intelligence import g7d_c1_r3_loaded_review as r3

REVISION = "G7D_C1_R5_FULL_FRAME_SCENE_REVIEW_V1"
R4_REVISION = "G7D_C1_R4_STABLE_BOOT_NOVICE_REVIEW_V1"


class FullFrameSceneReviewStore(r3.LoadedReviewStore):
    review_revision = REVISION
    compatible_revisions = (*r3.LoadedReviewStore.compatible_revisions, R4_REVISION, REVISION)

    def state(self) -> dict[str, Any]:
        state = CalibratedReviewStore.state(self)
        retained: dict[str, Any] = {}
        discarded = 0
        for key, draft in state["drafts"].items():
            if (
                draft.get("revision") in {R4_REVISION, REVISION}
                and (key in self.by_target or key in self.by_scene)
                and isinstance(draft.get("step_index"), int)
                and draft["step_index"] >= 0
            ):
                retained[key] = draft
            else:
                discarded += 1
        state["drafts"] = retained
        state["discarded_stale_draft_count"] = discarded
        return state


def create_server(package: Path, port: int = 0) -> ThreadingHTTPServer:
    return r3.create_server(package, port, FullFrameSceneReviewStore)


def serve(package: Path, port: int = 8814) -> None:
    server = create_server(package, port)
    print(f"R5 full-frame scene reviewer listening on http://127.0.0.1:{server.server_port}/")
    server.serve_forever()
