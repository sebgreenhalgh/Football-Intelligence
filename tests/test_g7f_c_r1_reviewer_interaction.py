from __future__ import annotations

import json
from pathlib import Path

from football_intelligence.dense_person_reviewer import DensePersonHTTPServer, DensePersonReviewerConfig


REPO = Path(__file__).resolve().parents[1]
STATIC = REPO / "src/football_intelligence/dense_person_reviewer_static"


def _frame(image_id: str, selection_status: str, position: int) -> dict[str, object]:
    return {
        "anonymous_dense_image_id": image_id,
        "selection_status": selection_status,
        "source_frame_sha256": f"{position:064x}",
        "source_width": 64,
        "source_height": 48,
        "review_queue_position": position,
        "all_frame_instance_lineage": [{"burst_id": "burst", "frame_sequence": position}],
    }


def _config(tmp_path: Path, *, decisions_root: Path | None = None) -> DensePersonReviewerConfig:
    selection = {
        "images": [
            _frame("DG-001", "CALIBRATION_ONLY", 1),
            _frame("DG-007", "SCORED_DENSE_GOLD", 7),
        ]
    }
    reveal = {"reveal_payloads": {"DG-001": {"runs": {}}, "DG-007": {"runs": {}}}}
    selection_path = tmp_path / "selection.json"
    reveal_path = tmp_path / "reveal.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    reveal_path.write_text(json.dumps(reveal), encoding="utf-8")
    return DensePersonReviewerConfig(
        selection_manifest_path=selection_path,
        assets_root=tmp_path / "assets",
        decisions_root=decisions_root or tmp_path / "decisions",
        binding_hashes={},
        reviewer_release="G7F_C_DENSE_PERSON_REVIEWER_R1",
        reveal_payload_path=reveal_path,
        allowed_selection_statuses=("CALIBRATION_ONLY",),
        port=0,
    )


def test_r1_bootstrap_exposes_modes_and_calibration_only(tmp_path: Path) -> None:
    server = DensePersonHTTPServer(_config(tmp_path))
    try:
        bootstrap = server.blind_bootstrap()
        assert [row["anonymous_dense_image_id"] for row in bootstrap["queue"]] == ["DG-001"]
        assert bootstrap["queue"][0]["workflow_group"] == "CALIBRATION"
        assert bootstrap["tools"]["default_interaction_mode"] == "PAN_EDIT"
        assert bootstrap["tools"]["persistent_interaction_modes"] == [
            "PAN_EDIT",
            "DRAW_PERSON",
            "ADD_VISIBLE_COMPONENT",
            "DRAW_IGNORE_REGION",
        ]
        assert bootstrap["tools"]["temporary_pan"] == "SPACE_HOLD"
        assert bootstrap["production_ready"] is False
    finally:
        server.server_close()


def test_loading_existing_v1_draft_does_not_rewrite_it(tmp_path: Path) -> None:
    decisions = tmp_path / "decisions"
    draft_path = decisions / "drafts/first_pass__DG-001.json"
    draft_path.parent.mkdir(parents=True)
    draft = {
        "anonymous_dense_image_id": "DG-001",
        "pass_kind": "FIRST_PASS",
        "revision": 27,
        "document": {
            "people": [
                {
                    "instance_id": "person-002",
                    "relevance": "MATCH_RELEVANT",
                    "visible_mask_components": [],
                }
            ],
            "ignore_regions": [],
            "reviewed_exhaustiveness_strips": [],
            "unfinished_polygon": None,
        },
    }
    draft_path.write_text(json.dumps(draft, indent=1), encoding="utf-8", newline="\r\n")
    before = draft_path.read_bytes()
    server = DensePersonHTTPServer(_config(tmp_path, decisions_root=decisions))
    try:
        loaded = server.store.state("DG-001", "FIRST_PASS")
        assert loaded["revision"] == 27
        assert loaded["document"] == draft["document"]
        assert draft_path.read_bytes() == before
    finally:
        server.server_close()


def test_client_has_single_guarded_vertex_boundary_and_explicit_controls() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    styles = (STATIC / "styles.css").read_text(encoding="utf-8")

    for mode in ("PAN_EDIT", "DRAW_PERSON", "ADD_VISIBLE_COMPONENT", "DRAW_IGNORE_REGION"):
        assert mode in script
        assert mode in page
    assert script.count("state.working.push(") == 1
    assert "function addWorkingVertex(point)" in script
    assert "state.finalized || isBusy() || state.coherenceFailure || state.spaceHeld || !isDrawingMode()" in script
    assert "Finish or cancel the current polygon first." in script
    assert "state.mode = InteractionMode.PAN_EDIT" in script
    assert "window.confirm" in script
    assert "Reset pan" not in page
    assert "Reset view" in page
    assert 'id="panEdit"' in page
    assert 'id="modeBadge"' in page
    assert 'id="deletePerson"' in page
    assert 'id="deleteIgnore"' in page
    assert 'data-effective-mode="PAN_EDIT"' in page
    assert 'data-effective-mode="TEMPORARY_PAN"' in styles
