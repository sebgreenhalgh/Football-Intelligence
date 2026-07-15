from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from football_intelligence.replay.occlusion_path_review_repair import (
    EXPECTED_URL,
    build_m5_5a_occlusion_path_review_repair,
    validate_m5_5a_occlusion_path_review_repair_pack,
)
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.manifest import load_manifest
from football_intelligence.review_chassis.server import ReviewChassisServerConfig, create_server
from football_intelligence.review_chassis.spatial_annotations import scan_forbidden_browser_payload
from football_intelligence.review_chassis.validation import validate_review_chassis_package


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _bbox(x: float, y: float) -> dict[str, float]:
    return {"x1": x, "y1": y, "x2": x + 12, "y2": y + 32}


def _write_frame(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 180), color).save(path)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspace"
    prompt_root = tmp_path / "prompt"
    historical = tmp_path / "historical"
    prior = tmp_path / "prior_v3"
    stateful = prior / "03_STATEFUL_OCCLUSION_BASELINE"
    invalid = stateful / "HUMAN_REVIEW"
    frame_root = historical / "continuity_v11" / "unseen_window" / "frames" / "extraction_a"

    frames = []
    for index, sequence in enumerate([78, 82, 84, 86, 181, 183, 185, 187, 299, 301, 304, 306]):
        frame_path = frame_root / f"frame_{sequence:06d}.jpg"
        _write_frame(frame_path, (20 + index * 5, 80, 120))
        frames.append(
            {
                "frame_sequence": sequence,
                "sequence": sequence,
                "source_frame_index": 40500 + sequence,
                "frame_file": str(frame_path),
                "filename": frame_path.name,
                "width": 320,
                "height": 180,
            }
        )
    _write_json(
        historical / "continuity_v11" / "unseen_window" / "canonical_frame_manifest.json",
        {"artifact": "fixture", "frames": frames, "dimensions": {"width": 320, "height": 180}},
    )

    observations = []
    hypotheses = []
    transitions = []
    for case_number, source_frame, target_frame in [("008", 183, 185), ("010", 301, 304), ("013", 82, 84)]:
        observations.extend(
            [
                {
                    "case_number": case_number,
                    "observation_id": f"case_{case_number}_source",
                    "node_type": "DETECTION",
                    "frame_sequence": source_frame,
                    "bbox": _bbox(40, 70),
                },
                {
                    "case_number": case_number,
                    "observation_id": f"case_{case_number}_target_01",
                    "node_type": "DETECTION",
                    "frame_sequence": target_frame,
                    "bbox": _bbox(45, 72),
                },
                {
                    "case_number": case_number,
                    "observation_id": f"case_{case_number}_target_02",
                    "node_type": "DETECTION",
                    "frame_sequence": target_frame,
                    "bbox": _bbox(72, 70),
                },
            ]
        )
        hypotheses.extend(
            [
                {
                    "case_number": case_number,
                    "source_observation_id": f"case_{case_number}_source",
                    "target_observation_id": f"case_{case_number}_target_01",
                    "hypothesis_rank": 1,
                    "path_cost": 0.2,
                    "cost_breakdown": {"bbox_iou": 0.3},
                },
                {
                    "case_number": case_number,
                    "source_observation_id": f"case_{case_number}_source",
                    "target_observation_id": f"case_{case_number}_target_02",
                    "hypothesis_rank": 2,
                    "path_cost": 0.4,
                    "cost_breakdown": {"bbox_iou": 0.1},
                },
                {
                    "case_number": case_number,
                    "source_observation_id": f"case_{case_number}_source",
                    "target_observation_id": f"case_{case_number}_merged_observation",
                    "hypothesis_rank": 3,
                    "path_cost": 0.95,
                    "cost_breakdown": {"merged_observation_cost": 0.95},
                },
            ]
        )
        transitions.append(
            {
                "case_number": case_number,
                "source_state": "MULTI_HYPOTHESIS_REENTRY",
                "target_state": "HUMAN_REVIEW_REQUIRED",
                "review_trigger": True,
            }
        )
    _write_jsonl(stateful / "observation_rows.jsonl", observations)
    _write_jsonl(stateful / "hypothesis_rows.jsonl", hypotheses)
    _write_jsonl(stateful / "state_transition_rows.jsonl", transitions)
    _write_json(
        stateful / "case_results.json",
        {
            "rows": [
                {"case_number": "008", "outcome": "review_required_unresolved_no_forced_assignment"},
                {"case_number": "010", "outcome": "review_required_unresolved_no_forced_assignment"},
                {"case_number": "013", "outcome": "review_required_unresolved_no_forced_assignment"},
            ]
        },
    )
    _write_json(
        invalid / "reviewer_manifest.json",
        {
            "schema_version": "invalid",
            "cases": [
                {
                    "case_id": "old",
                    "source_frame_sequence": None,
                    "target_frame_sequence": None,
                    "allowed_decisions": ["PATH_C_CONTINUES_SOURCE"],
                }
            ],
        },
    )
    _write_json(invalid / "ui_config.json", {"schema_version": "invalid"})
    _write_json(invalid / "sealed" / "server_mapping.json", {"diagnostic": True})
    (invalid / "static").mkdir(parents=True)
    (invalid / "decisions").mkdir(parents=True)
    (invalid / "decisions" / "review_decision_events.jsonl").write_text("", encoding="utf-8")

    for name in [
        "00_READ_ME_FIRST.md",
        "01_M5_5A_OCCLUSION_PATH_REVIEW_REPAIR_CODEX_PROMPT.md",
        "04_PROMPT_PACK_MANIFEST.json",
    ]:
        (prompt_root / name).parent.mkdir(parents=True, exist_ok=True)
        (prompt_root / name).write_text("fixture\n", encoding="utf-8")
    _write_json(
        prompt_root / "02_REPAIR_WORKSPACE_CONTRACT.json",
        {
            "workspace_root": str(workspace),
            "historical_stage_root": str(historical),
            "prior_m5_5a_workspace_read_only": str(prior),
            "invalid_review_package_read_only": str(invalid),
            "repository": {"historical_baseline_commit": "HEAD"},
        },
    )
    _write_json(prompt_root / "03_GENERIC_REVIEW_PACKAGE_CONTRACT.json", {"historical_sources_must_not_mutate": []})
    return prompt_root, workspace, historical


def test_occlusion_path_repair_builds_valid_blinded_generic_package(tmp_path: Path) -> None:
    prompt_root, workspace, _historical = _fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    result = build_m5_5a_occlusion_path_review_repair(
        repo_root=repo_root,
        prompt_root=prompt_root,
        output_root=workspace,
    )

    package = result["package"]
    manifest = load_manifest(Path(package["manifest_path"]))
    ui_config = load_ui_config(Path(package["ui_config_path"]))
    state = json.loads((Path(package["decisions_root"]) / "review_decisions.json").read_text(encoding="utf-8"))
    events = (Path(package["decisions_root"]) / "review_decision_events.jsonl").read_text(encoding="utf-8")

    assert result["final_classification"] == "PASS_REPAIRED_GENERIC_OCCLUSION_PATH_REVIEW_READY"
    assert manifest.schema_version.endswith(".v2")
    assert ui_config.schema_version.endswith(".v2")
    assert len(manifest.cases) == 3
    assert all(
        case.source_frame_sequence is not None and case.target_frame_sequence is not None for case in manifest.cases
    )
    assert all("PATH_C_CONTINUES_SOURCE" not in case.allowed_decisions for case in manifest.cases)
    assert "PATH_C_CONTINUES_SOURCE" not in {option.value for option in ui_config.decisions}
    assert all(any(asset.asset_type == "animated_gif" for asset in case.evidence_assets) for case in manifest.cases)
    assert all(any(asset.asset_type == "image_sequence" for asset in case.evidence_assets) for case in manifest.cases)
    assert state["decisions"] == {}
    assert events == ""
    assert package["review_url"] == EXPECTED_URL

    validation = validate_review_chassis_package(
        manifest_path=Path(package["manifest_path"]),
        ui_config_path=Path(package["ui_config_path"]),
        evidence_root=Path(package["evidence_root"]),
        decisions_root=Path(package["decisions_root"]),
    )
    assert validation["passed"]


def test_browser_payload_is_sanitized_and_review_pack_is_flat(tmp_path: Path) -> None:
    prompt_root, workspace, _historical = _fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    result = build_m5_5a_occlusion_path_review_repair(
        repo_root=repo_root,
        prompt_root=prompt_root,
        output_root=workspace,
    )
    package = result["package"]
    server = create_server(
        ReviewChassisServerConfig(
            manifest_path=Path(package["manifest_path"]),
            ui_config_path=Path(package["ui_config_path"]),
            evidence_root=Path(package["evidence_root"]),
            decisions_root=Path(package["decisions_root"]),
            sealed_mapping_path=Path(package["sealed_mapping_path"]),
            port=0,
        )
    )
    try:
        browser_manifest = server.manifest_payload()
        browser_ui = server.ui_config_payload()
    finally:
        server.server_close()

    assert scan_forbidden_browser_payload(browser_manifest)["predecision_answer_key_delivered_to_client"] is False
    assert scan_forbidden_browser_payload(browser_ui)["predecision_answer_key_delivered_to_client"] is False
    assert "hidden_metadata" not in json.dumps(browser_manifest)
    assert "reveal_metadata" not in json.dumps(browser_manifest)
    assert "candidate_id" not in json.dumps(browser_manifest)
    assert "decision_to_output_mapping" not in json.dumps(browser_ui)
    assert result["review_pack"]["file_count"] <= 20
    assert result["review_pack"]["passed"]
    pack_validation = validate_m5_5a_occlusion_path_review_repair_pack(Path(result["review_pack"]["review_pack_root"]))
    assert pack_validation["passed"]
