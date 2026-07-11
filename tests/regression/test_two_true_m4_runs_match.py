from __future__ import annotations

import json

from football_intelligence.replay.differential import compare_true_replay_runs


def test_two_true_m4_runs_match_on_sealed_hashes(tmp_path) -> None:
    for name in ["a", "b"]:
        path = tmp_path / name / "validation"
        path.mkdir(parents=True)
        (path / "true_replay_validation_summary.json").write_text(
            json.dumps(
                {
                    "true_input_closure_hash": "h1",
                    "replay_config_hash": "h2",
                    "code_commit": "c",
                    "recovered_m1_semantic_hash": "n",
                    "reconstructed_structured_content_hash": "s",
                    "evidence_inventory_hash": "e",
                    "viewer_semantic_hash": "v",
                    "canonical_m3t_decision_semantic_hash": "d",
                    "counts": {"x": 1},
                    "guardrail_passed": True,
                    "source_mutation_passed": True,
                    "source_access_passed": True,
                }
            ),
            encoding="utf-8",
        )
    assert compare_true_replay_runs(tmp_path / "a", tmp_path / "b")["passed"] is True
