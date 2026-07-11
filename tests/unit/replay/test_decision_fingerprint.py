from __future__ import annotations

import json

from football_intelligence.replay.decision_fingerprint import reconcile_decision_fingerprint
from football_intelligence.replay.test_dependency_fixtures import synthetic_m3t_payloads


def test_decision_fingerprint_validates_references(tmp_path) -> None:
    m3t = synthetic_m3t_payloads()
    decision_path = tmp_path / "decisions.json"
    decision_path.write_text(json.dumps(m3t["decisions"]), encoding="utf-8")
    report = reconcile_decision_fingerprint(
        decision_payload=m3t["decisions"],
        decision_path=decision_path,
        review_candidates=m3t["review_candidates"],
        m3t_pathlets=m3t["pathlets"]["rows"],
        selected_edges=m3t["edges"],
    )
    assert report["missing_candidate_ids"] == []
    assert report["missing_pathlet_refs"] == []
    assert report["missing_edge_refs"] == []


def test_decision_fingerprint_accepts_quarantined_edge_references(tmp_path) -> None:
    m3t = synthetic_m3t_payloads()
    quarantined_edge = dict(m3t["edges"][1])
    m3t["edges"] = [m3t["edges"][0]]
    m3t["quarantined_edges"] = [quarantined_edge]
    decision_path = tmp_path / "decisions.json"
    decision_path.write_text(json.dumps(m3t["decisions"]), encoding="utf-8")

    report = reconcile_decision_fingerprint(
        decision_payload=m3t["decisions"],
        decision_path=decision_path,
        review_candidates=m3t["review_candidates"],
        m3t_pathlets=m3t["pathlets"]["rows"],
        selected_edges=m3t["edges"],
        quarantined_edges=m3t["quarantined_edges"],
    )

    assert report["missing_edge_refs"] == []
    assert report["selected_edge_reference_count"] == 1
    assert report["quarantined_edge_reference_count"] == 1
