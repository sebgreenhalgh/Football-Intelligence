from __future__ import annotations

import json
from pathlib import Path

from football_intelligence.g7d_c1_r7_atomic_transition_review import (
    REVISION,
    AtomicTransitionReviewStore,
    next_incomplete_scene,
)

ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    ROOT.parent / "experiments/football_observation_reasoner/part 6/G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_v1"
)
PACKAGE = STAGE / "02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE"
EVIDENCE = STAGE / "17_R7_SCENE_TRANSITION_QUESTION_INITIALIZATION_REPAIR"
HANDOFF = STAGE / "18_R7_REVIEW_PACK/CHATGPT_HANDOFF"


def test_r7_identity_counts_and_preserved_truth() -> None:
    store = AtomicTransitionReviewStore(PACKAGE)
    state = store.state()
    assert store.review_revision == REVISION
    assert len(state["cases"]) == 24
    assert sum(len(case["targets"]) for case in state["cases"]) == 192
    preservation = json.loads((EVIDENCE / "EVENT_PRESERVATION.json").read_text(encoding="utf-8"))
    assert preservation["candidate_event_count"] == 8
    assert preservation["scene_event_count"] == 1
    assert state["saved_scenes"]["scene_01_118575_118575_first_half_13"]


def test_atomic_runtime_contract_and_question_one() -> None:
    app = (PACKAGE / "app.js").read_text(encoding="utf-8")
    required = [
        "COMPLETE_SCENE_AND_OPEN_NEXT_TARGET",
        "SAVING_SCENE_FINAL",
        "ADVANCING_TO_NEXT_SCENE",
        "RESETTING_REVIEW_MODE",
        "LOADING_NEXT_TARGET",
        "INITIALIZING_NEXT_QUESTION",
        "Opening the next scene…",
        "What is inside the highlighted box?",
        "CANDIDATE_REVIEW_MODE",
        "transition_checkpoint",
    ]
    assert all(value in app for value in required if not value.startswith("Opening the next scene"))
    assert "Opening the next scene" in app
    assert "await completeSceneAndOpenNextTarget(completedSceneId)" in app
    assert "await refreshState(); advanceScene();" not in app
    assert "if (transitionInFlight ||" in app


def test_live_edge_and_boundary_evidence() -> None:
    live = json.loads((EVIDENCE / "LIVE_EDGE_RESULTS.json").read_text(encoding="utf-8"))
    assert live["classification"] == "PASS_LIVE_EDGE_ATOMIC_SCENE_1_TO_2"
    assert live["scene_1_truth_byte_identical"] is True
    assert live["scene_2_id"] == "scene_02_118575_118575_first_half_01"
    assert live["target_id"] == "s02t01"
    assert live["question"] == "What is inside the highlighted box?"
    assert live["temporary_draft_restored"] is True
    assert live["back_forward_scene_1_complete"] is True
    boundaries = json.loads((EVIDENCE / "BOUNDARY_REGRESSIONS.json").read_text(encoding="utf-8"))
    assert [row["actual_next"] for row in boundaries["fixtures"]] == [2, 3, 24, "ALL_CASES_COMPLETE"]


def test_next_scene_helper_all_boundaries() -> None:
    cases = json.loads((PACKAGE / "review_cases.json").read_text(encoding="utf-8"))["cases"]
    for completed, expected in ((1, 2), (2, 3), (23, 24), (24, None)):
        saved = {case["scene_id"]: {} for case in cases[:completed]}
        result = next_incomplete_scene(cases, saved)
        assert (cases.index(result) + 1 if result else None) == expected


def test_exact_two_visuals_and_ten_file_handoff() -> None:
    visuals = sorted((EVIDENCE / "visual_qa").glob("*.png"))
    assert [path.name for path in visuals] == ["01_OPENING_NEXT_SCENE.png", "02_SCENE_2_TARGET_1_QUESTION_1.png"]
    files = sorted(path for path in HANDOFF.iterdir() if path.is_file())
    assert len(files) == 10
    manifest = json.loads((HANDOFF / "10_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["file_count_excluding_manifest"] == 9


def test_scope_has_no_forbidden_runtime_calls() -> None:
    changed = "\n".join(
        [
            (ROOT / "scripts/g7d_c1_r7_build_atomic_scene_transition.py").read_text(encoding="utf-8"),
            (ROOT / "src/football_intelligence/g7d_c1_r7_atomic_transition_review.py").read_text(encoding="utf-8"),
        ]
    ).lower()
    assert "validation/holdout" not in changed
    assert "run inference" not in changed
