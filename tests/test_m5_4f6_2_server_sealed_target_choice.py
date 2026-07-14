from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from PIL import Image

from football_intelligence.replay.blind_target_choice_review import (
    _comparison_identity,
    _dedupe_reversed_comparisons,
    _forbidden_text_hits,
    _predecision_answer_key_audit,
    _safe_target_choice_ui_config,
    _stage_ui_copy_count,
    _target_assignment,
    _walk_forbidden_answer_key,
)
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.models import (
    GenericEvidenceAsset,
    GenericReviewCase,
    GenericReviewManifest,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from football_intelligence.review_chassis.server import ReviewChassisServerConfig, create_server
from football_intelligence.review_chassis.validation import validate_review_chassis_package


def _bbox(x1: float, y1: float, x2: float, y2: float) -> dict[str, float]:
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _write_image(path: Path, *, fmt: str = "JPEG") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), (90, 120, 150)).save(path, format=fmt)


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _safe_package(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    evidence_root = tmp_path / "evidence"
    image_path = evidence_root / "case_001" / "target_a.jpg"
    gif_path = evidence_root / "case_001" / "clip.gif"
    _write_image(image_path)
    _write_image(gif_path, fmt="GIF")
    assets = [
        GenericEvidenceAsset(
            asset_id="target_a_crop",
            asset_type="crop",
            label="Target A",
            relative_path="target_a.jpg",
            sha256=sha256_file(image_path),
            media_type="image/jpeg",
            group_id="target_a",
        ),
        GenericEvidenceAsset(
            asset_id="temporal_clip",
            asset_type="animated_gif",
            label="Temporal GIF",
            relative_path="clip.gif",
            sha256=sha256_file(gif_path),
            media_type="image/gif",
            group_id="temporal",
        ),
    ]
    case = GenericReviewCase(
        case_id="case_001",
        task_type="visual_continuity_target_choice_review",
        candidate_id="candidate_001",
        candidate_hash=stable_hash("candidate_001"),
        evidence_hash=stable_hash([asset.model_dump(mode="json") for asset in assets]),
        allowed_decisions=["target_a_continues_source", "unresolved"],
        concise_question="Which target continues the highlighted source person?",
        evidence_assets=assets,
        source_frame_sequence=10,
        target_frame_sequence=11,
        frame_gap=1,
        visible_metadata={"target_a_id": "case_001_target_a", "target_b_id": "case_001_target_b"},
        hidden_metadata={},
        reveal_metadata={},
        safety_payload=safety_payload(),
    )
    manifest = GenericReviewManifest(
        review_id="m5_4f6_2_test",
        stage_id="m5_4f6_2",
        task_type="visual_continuity_target_choice_review",
        title="Server sealed target choice",
        cases=[case],
        evidence_manifest_hash=stable_hash([case.evidence_hash]),
        source_manifest_hash=stable_hash([]),
    )
    ui_config = ReviewUIConfig.model_validate(_safe_target_choice_ui_config())
    manifest_path = tmp_path / "target_choice_reviewer_manifest.json"
    ui_config_path = tmp_path / "target_choice_ui_config.json"
    decisions_root = tmp_path / "decisions"
    sealed_path = tmp_path / "sealed" / "target_choice_server_sealed_mapping.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    ui_payload = ui_config.model_dump(mode="json")
    ui_payload.pop("decision_to_output_mapping", None)
    ui_config_path.write_text(json.dumps(ui_payload, sort_keys=True), encoding="utf-8")
    sealed_path.parent.mkdir(parents=True, exist_ok=True)
    sealed_path.write_text(
        json.dumps(
            {
                "schema_version": "test.sealed_mapping.v1",
                "reveal_payloads": {
                    "case_001": {
                        "__case_metadata__": {
                            "accepted_target_panel": "target_a",
                            "decision_mapping": {
                                "target_a_continues_source": {"conflict_if_chosen_panel_is_not_prior_accept": False}
                            },
                        }
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    GenericReviewPersistence(
        manifest=manifest,
        ui_config=ui_config,
        decisions_root=decisions_root,
        reviewer_session_id="test",
    ).ensure_state()
    return manifest_path, ui_config_path, evidence_root, decisions_root, sealed_path


def _row(index: int, *, duplicate_of_first: bool = False) -> dict[str, object]:
    source = "source_001" if duplicate_of_first else f"source_{index:03d}"
    source_frame = 10 if duplicate_of_first else index * 10
    target_frame = 11 if duplicate_of_first else index * 10 + 1
    accepted = "target_x" if duplicate_of_first else f"accepted_{index:03d}"
    alternative = "target_y" if duplicate_of_first else f"alternative_{index:03d}"
    return {
        "candidate_id": f"candidate_{index:03d}",
        "source_visible_person_base_id": source,
        "source_frame_sequence": source_frame,
        "target_frame_sequence": target_frame,
        "accepted_target_visible_person_base_id": accepted,
        "alternative_target_visible_person_base_id": alternative,
        "accepted_target_candidate_id": f"accepted_candidate_{index:03d}",
        "alternative_target_candidate_id": f"alternative_candidate_{index:03d}",
        "accepted_target_bbox": _bbox(index, 0, index + 10, 20),
        "alternative_target_bbox": _bbox(index + 30, 0, index + 40, 20),
        "source_bbox": _bbox(index, 0, index + 10, 20),
        "frame_gap": 1,
        "local_assignment_neighbourhood_id": "n_001" if duplicate_of_first else f"n_{index:03d}",
        "candidate_type": "review_only_local_same_frame_wrong_target",
    }


def test_reviewer_manifest_and_ui_have_no_answer_key_fields_or_values(tmp_path: Path) -> None:
    manifest_path, ui_config_path, evidence_root, decisions_root, sealed_path = _safe_package(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ui_config = json.loads(ui_config_path.read_text(encoding="utf-8"))
    initial_state = json.loads((decisions_root / "review_decisions.json").read_text(encoding="utf-8"))

    audit = _predecision_answer_key_audit(
        reviewer_manifest=manifest,
        ui_config=ui_config,
        initial_state=initial_state,
        evidence_root=evidence_root,
        sealed_mapping_path=sealed_path,
    )

    assert audit["predecision_answer_key_delivered_to_client"] is False
    assert audit["browser_served_answer_key_field_count"] == 0
    assert _walk_forbidden_answer_key(manifest, source="manifest") == []
    assert _walk_forbidden_answer_key(ui_config, source="ui_config") == []
    assert _forbidden_text_hits("Target A Target B Source", source="bootstrap") == []


def test_sealed_mapping_is_not_static_and_reveal_requires_saved_decision(tmp_path: Path) -> None:
    manifest_path, ui_config_path, evidence_root, decisions_root, sealed_path = _safe_package(tmp_path)
    server = create_server(
        ReviewChassisServerConfig(
            manifest_path=manifest_path,
            ui_config_path=ui_config_path,
            evidence_root=evidence_root,
            decisions_root=decisions_root,
            sealed_mapping_path=sealed_path,
            port=0,
        )
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(f"{base}/api/review/manifest", timeout=10) as response:  # noqa: S310
            manifest_payload = json.loads(response.read().decode("utf-8"))
        with urlopen(f"{base}/api/review/ui-config", timeout=10) as response:  # noqa: S310
            ui_payload = json.loads(response.read().decode("utf-8"))
        assert _walk_forbidden_answer_key(manifest_payload, source="api_manifest") == []
        assert "decision_to_output_mapping" not in ui_payload
        with pytest.raises(HTTPError) as missing:
            urlopen(f"{base}/sealed/target_choice_server_sealed_mapping.json", timeout=10)  # noqa: S310
        assert missing.value.code == 404
        with pytest.raises(HTTPError) as blocked:
            _post_json(
                f"{base}/api/review/reveal",
                {"case_id": "case_001", "reveal_group_id": "__case_metadata__"},
            )
        assert blocked.value.code == 400

        _post_json(
            f"{base}/api/review/decision",
            {"case_id": "case_001", "decision": "target_a_continues_source", "input_source": "click"},
        )
        state = _post_json(
            f"{base}/api/review/reveal",
            {"case_id": "case_001", "reveal_group_id": "__case_metadata__"},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    reveal_payload = state["server_reveal_payloads"]["case_001"]["__case_metadata__"]
    assert reveal_payload["accepted_target_panel"] == "target_a"
    events = (decisions_root / "review_decision_events.jsonl").read_text(encoding="utf-8")
    assert '"event_type": "reveal"' in events
    assert '"decision_exists_at_reveal": true' in events


def test_reversed_ab_comparisons_are_deduplicated_to_six_unique_cases() -> None:
    rows = [_row(1)]
    duplicate = _row(2, duplicate_of_first=True)
    duplicate["source_visible_person_base_id"] = rows[0]["source_visible_person_base_id"]
    duplicate["source_frame_sequence"] = rows[0]["source_frame_sequence"]
    duplicate["target_frame_sequence"] = rows[0]["target_frame_sequence"]
    duplicate["accepted_target_visible_person_base_id"] = rows[0]["alternative_target_visible_person_base_id"]
    duplicate["alternative_target_visible_person_base_id"] = rows[0]["accepted_target_visible_person_base_id"]
    duplicate["local_assignment_neighbourhood_id"] = rows[0]["local_assignment_neighbourhood_id"]
    rows.append(duplicate)
    rows.extend(_row(index) for index in range(3, 8))

    unique, audit = _dedupe_reversed_comparisons(rows)
    panel_distribution = [
        _target_assignment(row, index)["accepted_target_panel"] for index, row in enumerate(unique, start=1)
    ]

    assert _comparison_identity(rows[0]) == _comparison_identity(rows[1])
    assert audit["reversed_comparison_duplicate_count_before"] == 1
    assert audit["reversed_comparison_duplicate_count_after"] == 0
    assert len(unique) == 6
    assert len({row["local_assignment_neighbourhood_id"] for row in unique}) == 6
    assert panel_distribution.count("target_a") == 3
    assert panel_distribution.count("target_b") == 3


def test_gif_only_package_and_no_stage_specific_frontend(tmp_path: Path) -> None:
    manifest_path, ui_config_path, evidence_root, decisions_root, _sealed_path = _safe_package(tmp_path)
    validation = validate_review_chassis_package(
        manifest_path=manifest_path,
        ui_config_path=ui_config_path,
        evidence_root=evidence_root,
        decisions_root=decisions_root,
    )

    assert validation["passed"] is True
    assert validation["gif_asset_count"] == 1
    assert validation["mp4_asset_count"] == 0
    assert validation["video_element_present"] is False
    assert _stage_ui_copy_count(tmp_path) == 0
