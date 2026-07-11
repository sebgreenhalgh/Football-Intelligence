from __future__ import annotations

from football_intelligence.replay.true_m4_documents import build_summary, build_validation_documents
from football_intelligence.replay.test_dependency_fixtures import synthetic_m3t_payloads
from football_intelligence.step2_visual_continuity.sparse_handoff_package import build_m4_handoff_rows


def test_true_m4_documents_keep_guardrails_false(tmp_path) -> None:
    m3t = synthetic_m3t_payloads()
    pathlets, edges = build_m4_handoff_rows(m3t["pathlets"]["rows"], m3t["edges"], m3t["decisions"]["rows"])
    viewer = tmp_path / "reconstructed_m4/step2m4_sparse_handoff_viewer.html"
    viewer.parent.mkdir(parents=True)
    viewer.write_text("<html></html>", encoding="utf-8")
    summary = build_summary(
        pathlets=pathlets,
        edges=edges,
        quarantined_edge_count=0,
        decision_rows=m3t["decisions"]["rows"],
        m3t_manifest=m3t["handoff"],
        overlay_summary={
            "overlay_gif_count": 1,
            "overlay_strip_count": 1,
            "overlay_frame_count": 1,
            "overlay_asset_count": 3,
        },
        m3t_root=tmp_path / "m3t",
        viewer_path=viewer,
    )
    _manifest, validation, _audit, _issues, freeze = build_validation_documents(
        summary=summary,
        pathlets=pathlets,
        edges=edges,
        m3t_handoff=m3t["handoff"],
        m3t_progress=m3t["progress"],
        m3t_validation=m3t["validation"],
        step2_visual_continuity_root=tmp_path,
        m3t_root=tmp_path / "m3t",
        m4_output_root=tmp_path / "reconstructed_m4",
        viewer_path=viewer,
        validation_summary_path=tmp_path / "validation.json",
        handoff_manifest_path=tmp_path / "manifest.json",
    )
    assert validation["production_ready"] is False
    assert freeze["human_approved"] is False
    assert freeze["safe_to_apply_globally"] is False
