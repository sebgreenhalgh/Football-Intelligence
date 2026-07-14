from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_intelligence.research_handoff.stage_workspace import safety_payload, sha256_file, utc_now

FORBIDDEN_REVIEW_KEYS = {
    "answer_key",
    "accepted_target_panel",
    "historical_correct_panel",
    "human_selected_panel",
    "decision_to_output_mapping",
    "sealed_mapping",
    "target_visible_person_base_id",
    "source_visible_person_base_id",
    "persistent_identity",
    "player_slot",
    "goalkeeper_slot",
}

OCCLUSION_DECISIONS = [
    "PATH_A_CONTINUES_INCOMING_TRACK",
    "PATH_B_CONTINUES_INCOMING_TRACK",
    "PATH_C_CONTINUES_INCOMING_TRACK",
    "MULTIPLE_PATHS_REMAIN_PLAUSIBLE",
    "TRACK_FULLY_HIDDEN_NO_REENTRY_YET",
    "VISIBLE_TARGET_NOT_IN_CANDIDATE_SET",
    "MERGED_OBSERVATION_CANNOT_BE_SEPARATED",
    "TRACK_TERMINATES_OR_EXITS",
    "EVIDENCE_UNRESOLVED",
]


def _scan_forbidden_keys(value: Any, *, path: str = "$") -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in FORBIDDEN_REVIEW_KEYS:
                hits.append({"path": f"{path}.{key}", "field": key})
            hits.extend(_scan_forbidden_keys(nested, path=f"{path}.{key}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            hits.extend(_scan_forbidden_keys(nested, path=f"{path}[{index}]"))
    return hits


def build_occlusion_human_review_package(
    *,
    output_root: Path,
    unresolved_cases: list[dict[str, Any]],
    evidence_root_hint: Path | None = None,
) -> dict[str, Any]:
    """Write a reviewer-safe unresolved-interval package for the generic review chassis."""

    output_root = output_root.resolve()
    static_root = output_root / "static"
    decisions_root = output_root / "decisions"
    sealed_root = output_root / "sealed"
    for path in (output_root, static_root, decisions_root, sealed_root):
        path.mkdir(parents=True, exist_ok=True)

    cases = []
    sealed_cases = []
    for index, case in enumerate(unresolved_cases, start=1):
        case_id = f"m5_5a_occlusion_unresolved_interval_{index:03d}"
        frame_sequence = case.get("frame_sequence") or case.get("source_frame_sequence")
        safe_case = {
            "case_id": case_id,
            "task_type": "anonymous_occlusion_path_review",
            "question": "Which anonymous path is best supported through this occlusion interval?",
            "case_label": case.get("case_id"),
            "frame_sequence": frame_sequence,
            "source_frame_sequence": case.get("source_frame_sequence"),
            "target_frame_sequence": case.get("target_frame_sequence"),
            "incoming_anonymous_paths": case.get("incoming_anonymous_paths", ["tracklet_a"]),
            "outgoing_hypothesis_panels": ["PATH_A", "PATH_B", "PATH_C"],
            "allowed_decisions": OCCLUSION_DECISIONS,
            "evidence": {
                "mode": "gif_only_or_static_contact_sheet",
                "evidence_root_hint": str(evidence_root_hint) if evidence_root_hint is not None else None,
                "anonymous_only": True,
                "hidden_answers_present": False,
            },
            "uncertainty_reasons": case.get(
                "uncertainty_reasons",
                ["near_equal_path_costs", "crossing_or_crowding_interval", "review_required_before_reentry"],
            ),
            "integrity": {"case_hash_material": f"{case_id}:{case.get('case_id')}:{frame_sequence}"},
            **safety_payload(),
        }
        cases.append(safe_case)
        sealed_cases.append(
            {
                "case_id": case_id,
                "source_case_id": case.get("case_id"),
                "mapping_purpose": "post-decision interpretation only",
                "historical_panel_reference_present": bool(case.get("human_decision")),
                "stored_outside_static_package": True,
            }
        )

    manifest = {
        "schema_version": "football_intelligence.occlusion_review_manifest.v1",
        "generated_at": utc_now(),
        "review_mode": "gif_only_anonymous_occlusion_path_review",
        "case_count": len(cases),
        "cases": cases,
        "predecision_answer_key_delivered_to_client": False,
        "browser_served_json_forbidden_key_hits": [],
        **safety_payload(),
    }
    hits = _scan_forbidden_keys(manifest)
    if hits:
        raise ValueError(f"reviewer-safe manifest contains forbidden keys: {hits}")

    ui_config = {
        "schema_version": "football_intelligence.review_chassis_ui.v2",
        "task_type": "anonymous_occlusion_path_review",
        "asset_mode": "gif_only",
        "prefill_decisions": False,
        "allowed_decisions": OCCLUSION_DECISIONS,
        "reveal_requires_persisted_decision": True,
        "predecision_answer_key_delivered_to_client": False,
        **safety_payload(),
    }
    ui_hits = _scan_forbidden_keys(ui_config)
    if ui_hits:
        raise ValueError(f"UI config contains forbidden keys: {ui_hits}")

    (output_root / "reviewer_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "ui_config.json").write_text(
        json.dumps(ui_config, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (sealed_root / "server_mapping.json").write_text(
        json.dumps(
            {
                "schema_version": "football_intelligence.occlusion_review_sealed_mapping.v1",
                "served_to_browser_before_decision": False,
                "cases": sealed_cases,
                **safety_payload(),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (decisions_root / "review_decision_events.jsonl").write_text("", encoding="utf-8")
    (output_root / "README.md").write_text(
        "\n".join(
            [
                "# M5.5A Anonymous Occlusion Path Review",
                "",
                "This is a reviewer-safe package for unresolved anonymous path intervals.",
                "It contains no hidden answer key in the browser-served manifest or UI config.",
                "The sealed mapping is stored under `sealed/` for post-decision interpretation only.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    launcher = output_root / "launch_occlusion_review.ps1"
    launcher.write_text(
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                "$Root = Split-Path -Parent $MyInvocation.MyCommand.Path",
                "fi-pipeline review-chassis serve `",
                "  --manifest (Join-Path $Root 'reviewer_manifest.json') `",
                "  --ui-config (Join-Path $Root 'ui_config.json') `",
                "  --evidence-root (Join-Path $Root 'static') `",
                "  --decisions-root (Join-Path $Root 'decisions') `",
                "  --sealed-mapping (Join-Path $Root 'sealed\\server_mapping.json')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "review_root": str(output_root),
        "case_count": len(cases),
        "reviewer_manifest": str(output_root / "reviewer_manifest.json"),
        "ui_config": str(output_root / "ui_config.json"),
        "sealed_mapping": {
            "path": str(sealed_root / "server_mapping.json"),
            "sha256": sha256_file(sealed_root / "server_mapping.json"),
            "accessible_through_static_route": False,
        },
        "launcher_path": str(launcher),
        "local_review_url": "http://127.0.0.1:8776/",
        "predecision_answer_key_delivered_to_client": False,
        "forbidden_key_hits": [],
        **safety_payload(),
    }
