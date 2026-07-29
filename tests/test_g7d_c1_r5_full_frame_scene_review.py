from __future__ import annotations

import json
from pathlib import Path

from football_intelligence.g7d_c1_r5_full_frame_review import FullFrameSceneReviewStore, REVISION


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT.parent
    / "experiments/football_observation_reasoner/part 6"
    / "G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_v1"
    / "02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE"
)


def test_r5_preserves_frozen_targets_and_uses_dedicated_scene_surface() -> None:
    document = json.loads((PACKAGE / "review_cases.json").read_text(encoding="utf-8"))
    app = (PACKAGE / "app.js").read_text(encoding="utf-8")
    index = (PACKAGE / "index.html").read_text(encoding="utf-8")
    assert document["review_revision"] == REVISION
    assert len(document["cases"]) == 24
    assert sum(len(case["targets"]) for case in document["cases"]) == 192
    assert 'id="sceneReviewSurface"' in index
    assert 'id="sceneCanvas"' in index
    assert "function drawSceneReview()" in app
    assert "function drawCandidateViews()" in app
    assert "TargetBoxCalibration.displayPointToSource(viewState.scene" in app
    assert "Review the entire frame, not the previous yellow box." in app


def test_r5_store_retains_compatible_r4_drafts_without_rewriting_events() -> None:
    store = FullFrameSceneReviewStore(PACKAGE)
    state = store.state()
    assert state["review_revision"] == REVISION
    assert state["target_mapping"]["verified"] is True
    assert state["target_mapping"]["target_count"] == 192
    assert state["discarded_stale_draft_count"] == 0
    assert len(state["saved_candidates"]) == 8
