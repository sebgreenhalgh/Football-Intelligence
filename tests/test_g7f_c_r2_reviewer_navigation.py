from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
APP = REPO / "src/football_intelligence/dense_person_reviewer_static/app.js"


def test_load_image_rerenders_controls_after_loading_clears() -> None:
    source = APP.read_text(encoding="utf-8")
    finally_block = source.split("async function loadImage", 1)[1].split("function fit", 1)[0]
    assert "finally {\n    state.loading = false;\n    renderControls();\n  }" in finally_block
    assert finally_block.index("state.loading = false") < finally_block.index("renderControls()")


def test_navigation_enablement_is_independent_of_finalized_state() -> None:
    source = APP.read_text(encoding="utf-8")
    controls = source.split("function renderControls()", 1)[1].split("function renderAll()", 1)[0]
    assert '$("previous").disabled = state.loading || state.index <= 0;' in controls
    assert '$("next").disabled = state.loading || state.index >= state.queue.length - 1;' in controls
    assert '$("imageSelect").disabled = state.loading;' in controls
    assert "state.finalized" not in "\n".join(line for line in controls.splitlines() if "previous" in line)
    assert "state.finalized" not in "\n".join(line for line in controls.splitlines() if "next" in line)
    assert "state.finalized" not in "\n".join(line for line in controls.splitlines() if "imageSelect" in line)
