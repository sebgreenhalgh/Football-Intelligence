from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
APP = REPO / "src/football_intelligence/dense_person_reviewer_static/app.js"
RUNNER = REPO / "scripts/g7f_c_run_calibration_adjudication_reviewer_r2.py"


def section(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def test_explicit_busy_state_disables_every_mutating_surface() -> None:
    source = APP.read_text(encoding="utf-8")
    controls = section(source, "function renderControls()", "function renderAll()")
    assert "const busy = isBusy();" in controls
    assert "const mutable = hasDocument && !state.finalized && !busy && coherent;" in controls
    for element_id in (
        "panEdit",
        "newPerson",
        "newIgnore",
        "addComponent",
        "undo",
        "redo",
        "assertion",
        "editRelevance",
        "repairChecklistAddressed",
        "adjudicationAssertion",
        "deletePerson",
        "deleteIgnore",
        "relevance",
        "ignoreReason",
    ):
        assert f'$("{element_id}").disabled = !mutable;' in controls or f'$("{element_id}").disabled = ' in controls
    assert "button.disabled = !mutable;" in controls
    assert '$("finalize").disabled = !mutable || !adjudicationReady;' in controls


def test_navigation_stages_target_before_atomic_commit_and_self_heals_selector() -> None:
    source = APP.read_text(encoding="utf-8")
    load = section(source, "async function loadImage", "function fit")
    assert "if (rejectBusyOperation()) return false;" in load
    assert "const nextDocument = clone(saved.document" in load
    assert "const nextMetadata = clone(saved.adjudication_metadata" in load
    assert "const nextImage = await new Promise" in load
    commit = load.index("state.index = targetIndex")
    assert load.index("const nextDocument") < commit
    assert load.index("const nextImage") < commit
    assert load.index("state.currentImageId = targetItem.anonymous_dense_image_id") >= commit
    rejected = section(source, "function rejectBusyOperation", "function clientStateInvariant")
    assert "syncSelector();" in rejected
    assert "BUSY_REJECTION_MESSAGE" in rejected


def test_idle_invariant_binds_selector_status_revision_and_workflow() -> None:
    source = APP.read_text(encoding="utf-8")
    invariant = section(source, "function clientStateInvariant", "function enforceClientStateInvariant")
    for name in (
        "selector_index_matches",
        "selector_text_matches",
        "current_image_matches",
        "revision_image_matches",
        "status_image_matches",
        "workflow_badge_matches",
    ):
        assert name in invariant
    enforcement = section(source, "function enforceClientStateInvariant", "let noticeTimer")
    assert "state.coherenceFailure = true;" in enforcement
    assert "Reload the reviewer before editing." in enforcement


def test_r2_runner_preserves_truth_bindings_and_targets_new_release() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'REVIEWER_RELEASE = "G7F_C_CALIBRATION_ADJUDICATION_REVIEWER_R2"' in source
    assert 'BASELINE = "548f1d17bb75632670fb576e3614eb341db133df"' in source
    assert '"adjudication_truth_bindings"' in source
    assert "G7F_C_CALIBRATION_SUPERSEDING_ADJUDICATION_AND_REAUDIT_v1" in source
    assert "G7F_C_ADJUDICATION_R1_NAVIGATION_STATE_DESYNC_REPAIR_v1" in source
