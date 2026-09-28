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
