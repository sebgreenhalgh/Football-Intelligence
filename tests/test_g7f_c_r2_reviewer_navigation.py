from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
APP = REPO / "src/football_intelligence/dense_person_reviewer_static/app.js"


def test_load_image_uses_an_explicit_atomic_operation_lifecycle() -> None:
    source = APP.read_text(encoding="utf-8")
    load_image = source.split("async function loadImage", 1)[1].split("function fit", 1)[0]
    assert "OperationState.SAVING_FOR_NAVIGATION" in load_image
    assert "OperationState.LOADING_IMAGE" in load_image
    assert "if (rejectBusyOperation()) return false;" in load_image
    assert "const nextDocument = clone(saved.document" in load_image
    assert "const nextImage = await new Promise" in load_image
    assert load_image.index("const nextImage = await new Promise") < load_image.index("state.index = targetIndex")
    assert load_image.index("state.index = targetIndex") < load_image.index("leaveOperation(token)")
    assert "syncSelector();" in load_image


def test_navigation_enablement_is_independent_of_finalized_state() -> None:
    source = APP.read_text(encoding="utf-8")
    controls = source.split("function renderControls()", 1)[1].split("function renderAll()", 1)[0]
    assert '$("previous").disabled = busy || !coherent || !hasDocument || state.index <= 0;' in controls
    assert (
        '$("next").disabled = busy || !coherent || !hasDocument || state.index >= state.queue.length - 1;' in controls
    )
    assert '$("imageSelect").disabled = busy || !coherent || !hasDocument;' in controls
    assert "state.finalized" not in "\n".join(line for line in controls.splitlines() if "previous" in line)
    assert "state.finalized" not in "\n".join(line for line in controls.splitlines() if "next" in line)
    assert "state.finalized" not in "\n".join(line for line in controls.splitlines() if "imageSelect" in line)


def test_save_and_finalize_responses_are_bound_to_captured_image_state() -> None:
    source = APP.read_text(encoding="utf-8")
    save = source.split("async function saveDraft", 1)[1].split("async function recoverAfterFinalizeFailure", 1)[0]
    finalize = source.split("async function finalize", 1)[1].split("async function reveal", 1)[0]
    assert "const imageId = state.currentImageId;" in save
    assert "const expectedRevision = state.serverRevision;" in save
    assert "const documentSnapshot = clone(state.document);" in save
    assert "response.anonymous_dense_image_id !== imageId" in save
    assert "state.currentImageId !== imageId" in save
    assert "Saved ${imageId} revision" in save
    assert "OperationState.FINALIZING" in finalize
    assert "response.anonymous_dense_image_id !== imageId" in finalize
    assert "response.revision !== expectedRevision + 1" in finalize
