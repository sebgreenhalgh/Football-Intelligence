"""Focused R2 target-box calibration and package-preservation checks."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from football_intelligence.g7d_c1_r2_calibrated_review import CalibratedReviewStore, REVISION

EXPECTED_HEAD = "161e47c22e0585eabecf2bd53851879a71018b38"
ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT.parent / (
    "experiments/football_observation_reasoner/part 6/G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_v1"
)
PACKAGE = STAGE / "02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE"
EVIDENCE = STAGE / "07_R2_TARGET_BOX_CALIBRATION_AND_CROP_ALIGNMENT_REPAIR"
HANDOFF = STAGE / "08_R2_REVIEW_PACK/CHATGPT_HANDOFF"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def test_expected_head_and_frozen_inputs_are_preserved() -> None:
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == EXPECTED_HEAD
    cases = read(PACKAGE / "review_cases.json")
    preservation = read(EVIDENCE / "TARGET_PRESERVATION.json")
    assert cases["review_revision"] == REVISION
    assert len(cases["cases"]) == 24
    assert sum(len(case["targets"]) for case in cases["cases"]) == 192
    assert preservation["classification"] == "PASS"
    assert preservation["frames_candidate_ids_source_boxes_and_selection_reasons_unchanged"] is True
    assert preservation["selection_sha256_before"] == preservation["selection_sha256_after"]
    for case in cases["cases"]:
        assert sha256(PACKAGE / "assets" / case["asset_name"]) == case["frame_sha256"]


def test_root_cause_and_shared_transform_contract_are_explicit() -> None:
    cause = read(EVIDENCE / "ROOT_CAUSE.json")
    audit = read(EVIDENCE / "target_box_calibration_audit.json")
    module = (ROOT / "src/football_intelligence/g7d_c1_r2_static/calibration.js").read_text(encoding="utf-8")
    app = (PACKAGE / "app.js").read_text(encoding="utf-8")
    assert cause["classification"] == "PROVEN_UI_TRANSFORM_AND_VISUAL_SEPARATION_DEFECT"
    assert cause["underlying_b3_candidate_geometry_declared_wrong"] is False
    assert "containTransform" in module and "sourceBoxToDisplay" in module and "displayPointToSource" in module
    assert "TargetBoxCalibration" in app and "strokeRect" in app and "drawCropFrame" in app
    assert "#ffcf33" in app and "#58b7ff" in app and "showFeet" in app
    assert audit["target_count"] == 192 and audit["failure_count"] == 0
    assert all(record["passed"] for record in audit["records"])
    assert all(record["source_to_display_to_source_error_px"] <= 0.5 for record in audit["records"])
    assert all(record["display_to_source_to_display_error_css_px"] <= 1.0 for record in audit["records"])
    assert all(record["crop_containment"] and record["in_bounds"] for record in audit["records"])


def test_fixture_resize_dpr_and_mapping_gate() -> None:
    audit = read(EVIDENCE / "target_box_calibration_audit.json")
    assert len(audit["fixtures"]) == 9
    assert all(item["passed"] and item["crop_containment"] for item in audit["fixtures"])
    assert {item["dpr"] for item in audit["dpr_and_resize_metrics"]} == {1, 2}
    assert audit["resized_layout_passed"] is True
    assert CalibratedReviewStore(PACKAGE).state()["target_mapping"]["verified"] is True
    temporary = Path(__import__("tempfile").mkdtemp())
    try:
        shutil.copy2(PACKAGE / "review_cases.json", temporary / "review_cases.json")
        status = read(PACKAGE / "target_box_calibration_status.json")
        status["verified"] = False
        (temporary / "target_box_calibration_status.json").write_text(json.dumps(status), encoding="utf-8")
        store = CalibratedReviewStore(temporary)
        code, response = store.save({"event_type": "candidate"})
        assert code == 409 and response["error_code"] == "TARGET_MAPPING_NOT_VERIFIED"
    finally:
        shutil.rmtree(temporary)


def test_r1_workflow_handoff_and_three_preview_cap() -> None:
    index = (PACKAGE / "index.html").read_text(encoding="utf-8")
    app = (PACKAGE / "app.js").read_text(encoding="utf-8")
    status = read(PACKAGE / "target_box_calibration_status.json")
    manifest = read(HANDOFF / "10_MANIFEST.json")
    previews = sorted((EVIDENCE / "visual_qa").glob("*.png"))
    assert "Target mapping: CHECKING" in index and "Yellow box = the exact box" in index
    assert "Blue dashed frame = the larger zoom area" in index
    for preserved in ("setupTutorial", "saveDraft", "renderBottlenecks", "startSceneReview", "completeReview"):
        assert preserved in app
    assert status["verified"] is True and status["target_count"] == 192 and status["failure_count"] == 0
    assert len(previews) == 3
    assert len(list(HANDOFF.iterdir())) == 10 and len(manifest["files"]) == 9
    assert all(row["filename"] != "10_MANIFEST.json" for row in manifest["files"])
    for row in manifest["files"]:
        file = HANDOFF / row["filename"]
        assert file.is_file() and file.stat().st_size == row["byte_size"] and sha256(file) == row["sha256"]
    result = read(EVIDENCE / "stage_result.json")
    assert result["human_review_started"] is False and result["g7d_c2_started"] is False
