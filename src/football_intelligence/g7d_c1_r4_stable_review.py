"""R4 stable-boot server identity for the calibrated C1 reviewer."""

from __future__ import annotations

from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from football_intelligence import g7d_c1_r3_loaded_review as r3
from football_intelligence.g7d_c1_r2_calibrated_review import CalibratedReviewStore

REVISION = "G7D_C1_R4_STABLE_BOOT_NOVICE_REVIEW_V1"


class StableBootReviewStore(r3.LoadedReviewStore):
    """Retain immutable truth while accepting only R4 working drafts."""

    review_revision = REVISION
    compatible_revisions = (*r3.LoadedReviewStore.compatible_revisions, REVISION)

    def state(self) -> dict[str, Any]:
        state = CalibratedReviewStore.state(self)
        retained: dict[str, Any] = {}
        discarded = 0
        for key, draft in state["drafts"].items():
            valid_key = key in self.by_target or key in self.by_scene
            valid_step = isinstance(draft.get("step_index"), int) and draft["step_index"] >= 0
            if draft.get("revision") == REVISION and valid_key and valid_step:
                retained[key] = draft
            else:
                discarded += 1
        state["drafts"] = retained
        state["discarded_stale_draft_count"] = discarded
        return state


def create_server(package: Path, port: int = 0) -> ThreadingHTTPServer:
    return r3.create_server(package, port, StableBootReviewStore)


def serve(package: Path, port: int = 8814) -> None:
    server = create_server(package, port)
    print(f"R4 stable-boot reviewer listening on http://127.0.0.1:{server.server_port}/")
    server.serve_forever()
