"""Check compact canonical state/docs offline; never open historical evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path


def check(root: Path) -> dict:
    status = json.loads((root / "PROJECT_STATUS.json").read_bytes())
    registry = json.loads((root / "configs/detectors/registry.json").read_bytes())
    assert status["production_ready"] is False and registry["production_ready"] is False
    assert status["repository"]["canonical_branch"] == "main"
    for key in ("candidate_unblinded", "final_promotion", "provisional_candidate"):
        assert status["detection"][key] == registry[key], f"Detector metadata disagreement: {key}"
    assert status["detection"]["status"] == registry["status"] == "OPERATOR_SELECTED_PROVISIONAL"
    assert status["detection"]["formal_stopping_decision"] == registry["stopping_audit_decision"] == (
        "CONTINUE_G7F_D_INTERIM_DENSE_GOLD_NEXT_TRANCHE_REQUIRED"
    )
    assert status["detection"]["candidate_unblinded"] is True
    assert status["detection"]["sealed_validation"] is registry["sealed_validation"] is False
    assert status["detection"]["final_promotion"] is registry["final_promotion"] is False
    assert registry["anonymous_label_mapping_disclosed"] is True
    assert registry["provisional_candidate"] in {row["configuration_id"] for row in registry["configurations"]}
    assert re.fullmatch(r"[0-9a-f]{64}", registry["provisional_manifest_sha256"])
    assert registry["operator_decision"] == "SELECT_CANDIDATE_C_AND_PAUSE_DENSE_ANNOTATION"
    gold = status["gold_corpus"]
    assert re.fullmatch(r"[0-9a-f]{64}", gold["latest_manifest_sha256"])
    assert re.fullmatch(r"gold-v\d+\.\d+\.\d+", gold["current_release"])
    assert gold["detection_gold_frames"] >= status["annotation_campaign"]["completed_scored_first_pass_frames"]
    assert gold["detection_gold_people"] >= 0
    if status["current_stage"] == "G7G_A_TEMPORAL_GOLD_CORPUS_AND_SEQUENCE_FOUNDATION_v1":
        assert status["stage_status"] == "COMPLETE"
        assert status["last_completed_stage"] == status["current_stage"]
        assert status["next_authorized_stage"] == "G7G_B_TEMPORAL_GOLD_HUMAN_PILOT_v1_SEPARATE_REQUEST_REQUIRED"
        assert gold["current_release"] == "gold-v0.1.0"
        assert gold["detection_gold_frames"] == 16 and gold["detection_gold_people"] == 862
        assert gold["layers_available"] == ["DETECTION"]
        temporal = status["temporal_gold"]
        assert temporal == {
            "status": "REVIEWER_READY_FOR_HUMAN_PILOT",
            "primary_sequences": 6,
            "reserve_sequences": 2,
            "real_sequences_annotated": 0,
            "real_temporal_events_created": 0,
            "candidate_data_used": False,
        }
    if status["current_stage"] in {
        "G7G_B_TEMPORAL_GOLD_HUMAN_PILOT_v1",
        "G7G_B_R1_TEMPORAL_REVIEWER_USABILITY_AND_SEQUENTIAL_FLOW_REPAIR_v1",
        "G7G_B_R2_FAST_LAUNCH_GATE_AND_GOLD_VALIDATION_PERFORMANCE_REPAIR_v1",
        "G7G_B_R3_TEMPORAL_REVIEWER_HIGH_MAGNIFICATION_ZOOM_REPAIR_v1",
    }:
        assert status["stage_status"] == "READY_FOR_HUMAN_PILOT"
        assert status["last_completed_stage"] == "G7G_A_TEMPORAL_GOLD_CORPUS_AND_SEQUENCE_FOUNDATION_v1"
        assert status["next_authorized_stage"] == "COMPLETE_THE_AUTHORIZED_G7G_B_SINGLE_SEQUENCE_HUMAN_PILOT"
        assert gold["current_release"] == "gold-v0.1.0"
        assert gold["detection_gold_frames"] == 16 and gold["detection_gold_people"] == 862
        assert gold["layers_available"] == ["DETECTION"]
        assert status["temporal_gold"] == {
            "status": ("SINGLE_SEQUENCE_PILOT_AUTHORIZED_WITH_R3_REVIEWER" if status["current_stage"].startswith("G7G_B_R3_")
                       else "SINGLE_SEQUENCE_PILOT_AUTHORIZED" if status["current_stage"] == "G7G_B_TEMPORAL_GOLD_HUMAN_PILOT_v1"
                       else "SINGLE_SEQUENCE_PILOT_AUTHORIZED_WITH_R2_REVIEWER"),
            "primary_sequences": 6,
            "reserve_sequences": 2,
            "authorized_pilot_sequences": 1,
            "real_sequences_annotated": 0,
            "real_temporal_events_created": 0,
            "candidate_data_used": False,
        }
    assert status["next_authorized_stage"]
    docs = [root / "README.md", root / "AGENTS.md"] + [root / p for p in status["repository"]["canonical_docs"]]
    for path in docs:
        assert path.is_file(), f"Missing canonical document: {path}"
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            resolved = (path.parent / target.split("#")[0]).resolve()
            assert resolved.is_relative_to(root.resolve()), f"Canonical link leaves repository: {target}"
            assert resolved.exists(), f"Broken canonical link: {path.name} -> {target}"
    words = len((root / "AGENTS.md").read_text(encoding="utf-8").split())
    assert words <= 1500, "Agent context exceeds 1,500 words"
    assert len((root / ".cursorrules").read_text(encoding="utf-8")) < 200
    return {"valid": True, "canonical_documents": len(docs), "agents_words": words, "production_ready": False}


if __name__ == "__main__":
    print(json.dumps(check(Path(__file__).resolve().parents[1]), sort_keys=True))
