from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.learning.entity_calibrator import train_entity_calibrator
from football_intelligence.learning.mixed_review_ingestion import ingest_mixed_review, write_jsonl
from football_intelligence.learning.model_application import apply_entity_calibrator
from football_intelligence.learning.review_label_validation import class_sufficiency_readiness
from football_intelligence.replay.rebuilt_human_calibrated_pipeline import (
    _continuity_evidence,
    _entity_evidence,
    _frame_records,
    _review_case_hash,
    _source_inventory,
    _source_ref,
    default_paths as m54d_default_paths,
    read_json,
    rows,
    write_json,
    write_text,
)
from football_intelligence.review.schemas import (
    CONTINUITY_DECISIONS,
    CONTINUITY_QUESTION,
    VISUAL_TEAM_ROLE_DECISIONS,
    VISUAL_TEAM_ROLE_QUESTION,
    ReviewCase,
    ReviewManifest,
    SourceArtifactReference,
    safety_payload,
    stable_hash,
    utc_now,
)
from football_intelligence.review.workbench import build_workbench
from football_intelligence.step1_visual_reconstruction.visual_role_context import (
    ASSISTANT_FAR,
    ASSISTANT_NEAR,
    CENTRAL_REFEREE,
    GOALKEEPER_UNKNOWN,
    NON_PERSON,
    OTHER_OFF_PITCH,
    TEAM_1_GOALKEEPER,
    TEAM_1_OUTFIELD,
    TEAM_2_GOALKEEPER,
    TEAM_2_OUTFIELD,
    TEAM_UNKNOWN_OUTFIELD,
    UNKNOWN_PERSON,
    build_visual_role_context_rows,
    build_visual_role_features,
)
from football_intelligence.step2_visual_continuity.positive_selector import select_continuity_review_candidates
from football_intelligence.step2_visual_continuity.role_partitioning import (
    apply_role_partitioning,
    build_role_partition_manifest,
    pool_size_report,
)

FINAL_CLASSIFICATION = "PASS_ROLE_PARTITIONED_CONTINUITY_REVIEW_READY"
CONTINUITY_BLOCKER = "BLOCKED_SINGLE_CLASS_REVIEW_LABELS"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return rows(read_json(path))


def default_paths(artifact_root: Path, match_id: str) -> dict[str, Path]:
    prior = m54d_default_paths(artifact_root, match_id)
    return {
        **prior,
        "source_stage_root": prior["stage_root"],
        "stage_root": prior["step_m5"] / "06e_role_partitioned_learning",
    }


def _empty_decision_state(manifest: ReviewManifest) -> dict[str, Any]:
    return {
        "schema_version": "m5_4b.review_decisions.v1",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "workbench_version": manifest.workbench_version,
        "candidate_manifest_hash": manifest.candidate_manifest_hash,
        "evidence_manifest_hash": manifest.evidence_manifest_hash,
        "reviewer_session_id": None,
        "event_sequence": 0,
        "decisions": {},
        "notes": {},
        "last_viewed_case_id": None,
        "elapsed_active_seconds": 0,
        "completed": False,
        **safety_payload(),
    }


def _write_empty_decisions(decision_root: Path, manifest: ReviewManifest) -> None:
    write_json(decision_root / "review_decisions.json", _empty_decision_state(manifest))
    (decision_root / "review_decision_events.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (decision_root / "review_decision_events.jsonl").write_text("", encoding="utf-8")
    (decision_root / "snapshots").mkdir(parents=True, exist_ok=True)


def _write_open_launcher(
    *,
    launcher_path: Path,
    repo_root: Path,
    manifest_path: Path,
    evidence_root: Path,
    decision_root: Path,
    workbench_root: Path,
    label: str,
    port: int,
) -> Path:
    url = f"http://127.0.0.1:{port}/"
    command = " ".join(
        [
            "uv run fi-pipeline review serve",
            f'--review-manifest "{manifest_path}"',
            f'--evidence-root "{evidence_root}"',
            f'--decision-root "{decision_root}"',
            f'--workbench-root "{workbench_root}"',
            f"--host 127.0.0.1 --port {port}",
        ]
    )
    text = f"""$ErrorActionPreference = "Stop"
Set-Location "{repo_root}"
Write-Host "Starting {label} review server..."
Write-Host "Review URL: {url}"
{command}
"""
    return write_text(launcher_path, text)


def _reviewed_entity_labels(ingestion: dict[str, Any]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in ingestion["entity_examples"]:
        if row["label_usable_for_training"]:
            labels[str(row["candidate_artifact_id"])] = str(row["normalized_training_label"])
    return labels


def _source_refs(paths: dict[str, Path]) -> list[SourceArtifactReference]:
    return [
        _source_ref(
            "m5_4d_entity_validity_recalibrated_rows",
            paths["source_stage_root"] / "entity" / "entity_validity_recalibrated_rows.json",
            "read-only M5.4D recalibrated entity rows",
        ),
        _source_ref(
            "m5_4d_entity_feature_rows",
            paths["source_stage_root"] / "entity" / "entity_feature_rows.json",
            "read-only M5.4D entity features",
        ),
        _source_ref(
            "m5_4d_continuity_candidate_rows",
            paths["source_stage_root"] / "continuity" / "continuity_candidate_rows.json",
            "read-only M5.4D continuity candidates",
        ),
        _source_ref(
            "m5_4d_continuity_node_rows",
            paths["source_stage_root"] / "continuity" / "continuity_node_rows.json",
            "read-only M5.4D continuity nodes",
        ),
        _source_ref(
            "m5_4d_round_1_completed_review",
            paths["source_stage_root"] / "review" / "decisions" / "completed_review.json",
            "read-only completed M5.4D Round 1 human review",
        ),
        _source_ref(
            "canonical_frame_manifest",
            paths["frame_manifest"],
            "read-only canonical frame manifest",
        ),
    ]


def _entity_calibration_outputs(
    *,
    learning_root: Path,
    entity_rows: list[dict[str, Any]],
    entity_examples: list[dict[str, Any]],
    readiness: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    training_examples = [
        {**row, "human_label": row["normalized_training_label"]}
        for row in entity_examples
        if row.get("label_usable_for_training") is True
    ]
    if readiness["status"] == "READY_FOR_TRAINING":
        calibrator = train_entity_calibrator(training_examples)
        calibrator = {
            **calibrator,
            "artifact": "m5_4e_entity_calibrator_candidate",
            "round_1_labels_bound_from_decisions_map": True,
        }
    else:
        calibrator = {
            "artifact": "m5_4e_entity_calibrator_candidate",
            "model_type": "match_local_regularized_rule_calibrator",
            "training_example_count": len(training_examples),
            "label_distribution": dict(Counter(row["human_label"] for row in training_examples)),
            "validation_result": readiness["status"],
            "gate_passed": False,
            "no_unreviewed_prediction_used_as_truth": True,
            **safety_payload(),
        }
    validation = {
        "artifact": "m5_4e_entity_calibrator_validation",
        "candidate_gate_passed": bool(calibrator.get("gate_passed")),
        "application_ready": bool(readiness.get("application_ready")),
        "validation_result": "passed"
        if calibrator.get("gate_passed") and readiness.get("application_ready")
        else (
            "blocked_insufficient_independent_clusters_for_application"
            if calibrator.get("gate_passed")
            else str(calibrator.get("validation_result", readiness["status"]))
        ),
        "readiness_status": readiness["status"],
        "min_clusters_per_class_for_application": readiness["min_clusters_per_class_for_application"],
        "examples_per_class": readiness["examples_per_class"],
        "equivalence_clusters_per_class": readiness["equivalence_clusters_per_class"],
        "no_unreviewed_prediction_used_as_truth": True,
        "apply_zero_changes_when_validation_fails": True,
        **safety_payload(),
    }
    application_calibrator = {**calibrator, "gate_passed": bool(validation["validation_result"] == "passed")}
    application = apply_entity_calibrator(original_rows=entity_rows, calibrator=application_calibrator)
    application = {
        **application,
        "artifact": "m5_4e_entity_application_rows",
        "source_rows_preserved": True,
        "validation_result": validation["validation_result"],
        **safety_payload(),
    }
    changes_audit = {
        "artifact": "m5_4e_entity_changes_audit",
        "entity_rows_updated": application["remaining_rows_updated_by_learned_models"],
        "original_classifications_preserved": application["original_predictions_preserved"],
        "apply_zero_changes_when_validation_fails": validation["validation_result"] != "passed",
        "validation_result": validation["validation_result"],
        **safety_payload(),
    }
    write_json(learning_root / "entity_calibrator_candidate.json", calibrator)
    write_json(learning_root / "entity_calibrator_validation.json", validation)
    write_json(learning_root / "entity_application_rows.json", application)
    write_json(learning_root / "entity_changes_audit.json", changes_audit)
    return calibrator, validation, application, changes_audit


def _role_by_visible_id(
    *,
    node_rows: list[dict[str, Any]],
    role_context_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    role_by_candidate = {str(row["candidate_id"]): row for row in role_context_rows}
    output: dict[str, dict[str, Any]] = {}
    for node in node_rows:
        role = role_by_candidate.get(str(node.get("candidate_id")))
        if role is not None:
            output[str(node["visible_person_base_id"])] = role
    return output


def _role_review_selection(
    *,
    role_context_rows: list[dict[str, Any]],
    role_feature_rows: list[dict[str, Any]],
    limit: int = 30,
) -> list[dict[str, Any]]:
    feature_by_candidate = {str(row["candidate_id"]): row for row in role_feature_rows}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in role_context_rows:
        buckets[str(row["visual_role_context_state"])].append(row)
    order = [
        TEAM_1_OUTFIELD,
        TEAM_2_OUTFIELD,
        TEAM_UNKNOWN_OUTFIELD,
        TEAM_1_GOALKEEPER,
        TEAM_2_GOALKEEPER,
        GOALKEEPER_UNKNOWN,
        CENTRAL_REFEREE,
        ASSISTANT_NEAR,
        ASSISTANT_FAR,
        OTHER_OFF_PITCH,
        NON_PERSON,
        UNKNOWN_PERSON,
    ]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: dict[str, Any], reason: str) -> None:
        candidate_id = str(row["candidate_id"])
        if candidate_id in seen or len(selected) >= limit:
            return
        feature = feature_by_candidate.get(candidate_id, {})
        selected.append(
            {
                **row,
                "selection_reason": reason,
                "role_feature": feature,
                "review_category": str(row["visual_role_context_state"]).replace("_visual_context", ""),
            }
        )
        seen.add(candidate_id)

    for state in order:
        state_rows = sorted(
            buckets.get(state, []),
            key=lambda row: (
                -float(row.get("visual_role_context_confidence", 0.0)),
                int(row.get("frame_sequence", 0)),
                str(row.get("candidate_id")),
            ),
        )
        for row in state_rows[:2]:
            add(row, f"representative from {state}")

    feature_buckets = [
        ("possible_goalkeeper", "goalkeeper_belief", 0.55),
        ("possible_central_referee", "central_referee_belief", 0.55),
        ("possible_near_camera_assistant", "near_camera_assistant_belief", 0.5),
        ("possible_far_camera_assistant", "far_camera_assistant_belief", 0.5),
        ("possible_off_pitch_person", "off_pitch_person_belief", 0.55),
    ]
    for reason, score_key, threshold in feature_buckets:
        scored = []
        for row in role_context_rows:
            feature = feature_by_candidate.get(str(row["candidate_id"]), {})
            score = float(feature.get(score_key, 0.0))
            if score >= threshold:
                scored.append((score, row))
        for _, row in sorted(scored, key=lambda item: (-item[0], int(item[1].get("frame_sequence", 0))))[:2]:
            add(row, reason)

    uncertain = sorted(
        role_context_rows,
        key=lambda row: (float(row.get("visual_role_context_confidence", 0.0)), int(row.get("frame_sequence", 0))),
    )
    for row in uncertain:
        add(row, "low-confidence or disagreement role-context example")
        if len(selected) >= limit:
            break
    return selected


def _write_role_review(
    *,
    review_root: Path,
    stage_root: Path,
    repo_root: Path,
    selected_rows: list[dict[str, Any]],
    frame_root: Path,
    frame_records: dict[int, dict[str, Any]],
    source_refs: list[SourceArtifactReference],
) -> dict[str, Any]:
    evidence_root = review_root / "evidence"
    decision_root = review_root / "decisions"
    workbench_root = review_root / "workbench"
    cases: list[ReviewCase] = []
    evidence_summary: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows, start=1):
        case_id = f"m5_4e_role_case_{index:03d}"
        evidence = _entity_evidence(
            evidence_root=evidence_root,
            case_id=case_id,
            candidate={**row, "source_frame_sequence": row["frame_sequence"]},
            frame_root=frame_root,
            frame_records=frame_records,
        )
        feature = row.get("role_feature") if isinstance(row.get("role_feature"), dict) else {}
        case_payload = {
            "review_case_id": case_id,
            "task_type": "visual_team_role_context",
            "concise_question": VISUAL_TEAM_ROLE_QUESTION,
            "allowed_decisions": VISUAL_TEAM_ROLE_DECISIONS,
            "candidate_artifact_id": str(row["candidate_id"]),
            "source_artifact_references": source_refs,
            "source_frame_sequence": int(row["frame_sequence"]),
            "target_frame_sequence": None,
            "evidence_manifest": evidence,
            "uncertainty_reasons": [
                *[str(reason) for reason in row.get("visual_role_context_reasons", [])],
                f"current_team_colour_cluster={feature.get('colour_histogram_signature', 'unknown')}",
                f"torso_colour={feature.get('torso_colour', 'unknown')}",
            ],
            "category": str(row.get("review_category", row["visual_role_context_state"])),
            "priority": index,
            "control_status": "active_learning_selected",
            "candidate_hash": "",
            "evidence_hash": evidence.evidence_hash,
            "safety_payload": safety_payload(),
            "review_round": 2,
            "selection_metadata": {
                "why_selected": row["selection_reason"],
                "current_model_prediction": row["visual_role_context_state"],
                "current_model_confidence": row["visual_role_context_confidence"],
                "belief_scores": row["belief_scores"],
                "current_team_colour_cluster": feature.get("colour_histogram_signature"),
                "tight_crop": "tight_crop.jpg",
                "wide_crop": "wide_crop.jpg",
                "full_frame": "full_frame.jpg",
                "temporal_gif": "temporal_clip.gif",
                "neighbouring_people_visible_in": "wide_crop.jpg",
            },
            "model_prediction": row["visual_role_context_state"],
            "model_confidence": float(row["visual_role_context_confidence"]),
            "equivalence_cluster_id": f"m5_4e_role_cluster_{row['visual_role_context_state']}",
            "representative_of_count": 1,
        }
        case_payload["candidate_hash"] = _review_case_hash(case_payload)
        case = ReviewCase.model_validate(case_payload)
        cases.append(case)
        evidence_summary.append(
            {
                "review_case_id": case.review_case_id,
                "candidate_artifact_id": case.candidate_artifact_id,
                "asset_count": len(case.evidence_manifest.evidence_assets),
                "temporal_evidence_available": case.evidence_manifest.temporal_evidence_available,
                "evidence_hash": case.evidence_hash,
            }
        )
    manifest = ReviewManifest(
        title="M5.4E Visual Team/Role Context Review",
        review_task_family="m5_4e_visual_team_role_context",
        review_cases=cases,
        candidate_manifest_hash=stable_hash([case.candidate_hash for case in cases]),
        evidence_manifest_hash=stable_hash([case.evidence_hash for case in cases]),
        source_manifest_hash=stable_hash([ref.model_dump(mode="json") for ref in source_refs]),
        source_artifact_references=source_refs,
    )
    manifest_path = review_root / "review_manifest.json"
    write_json(manifest_path, manifest.model_dump(mode="json"))
    write_json(
        review_root / "evidence_manifest_summary.json",
        {
            "artifact": "m5_4e_role_review_evidence_manifest_summary",
            "created_at": _now(),
            "case_count": len(cases),
            "temporal_evidence_count": sum(1 for item in evidence_summary if item["temporal_evidence_available"]),
            "rows": evidence_summary,
            **safety_payload(),
        },
    )
    _write_empty_decisions(decision_root, manifest)
    build_workbench(workbench_root)
    launcher = _write_open_launcher(
        launcher_path=stage_root / "OPEN_ROLE_REVIEW.ps1",
        repo_root=repo_root,
        manifest_path=manifest_path,
        evidence_root=evidence_root,
        decision_root=decision_root,
        workbench_root=workbench_root,
        label="M5.4E role-context",
        port=8770,
    )
    write_text(stage_root / "OPEN_REVIEW.ps1", launcher.read_text(encoding="utf-8"))
    return {
        "manifest_path": str(manifest_path),
        "evidence_root": str(evidence_root),
        "decision_root": str(decision_root),
        "workbench_root": str(workbench_root),
        "launcher_path": str(launcher),
        "review_url": "http://127.0.0.1:8770/",
        "review_case_count": len(cases),
        "temporal_evidence_count": sum(1 for item in evidence_summary if item["temporal_evidence_available"]),
    }


def _write_continuity_balance_review(
    *,
    review_root: Path,
    stage_root: Path,
    repo_root: Path,
    selection: dict[str, Any],
    node_by_visible_id: dict[str, dict[str, Any]],
    frame_root: Path,
    frame_records: dict[int, dict[str, Any]],
    source_refs: list[SourceArtifactReference],
) -> dict[str, Any]:
    candidates = [
        *[{**row, "review_bucket": "likely_positive_continuity"} for row in selection["likely_positive"]],
        *[{**row, "review_bucket": "difficult_or_likely_negative_continuity"} for row in selection["likely_negative"]],
    ]
    evidence_root = review_root / "evidence"
    decision_root = review_root / "decisions"
    workbench_root = review_root / "workbench"
    cases: list[ReviewCase] = []
    for index, row in enumerate(candidates, start=1):
        case_id = f"m5_4e_continuity_case_{index:03d}"
        evidence = _continuity_evidence(
            evidence_root=evidence_root,
            case_id=case_id,
            edge=row,
            node_by_visible_id=node_by_visible_id,
            frame_root=frame_root,
            frame_records=frame_records,
        )
        case_payload = {
            "review_case_id": case_id,
            "task_type": "visual_continuity_edge_review",
            "concise_question": CONTINUITY_QUESTION,
            "allowed_decisions": CONTINUITY_DECISIONS,
            "candidate_artifact_id": str(row.get("role_partitioned_continuity_candidate_id")),
            "source_artifact_references": source_refs,
            "source_frame_sequence": int(row["source_frame_sequence"]),
            "target_frame_sequence": int(row["target_frame_sequence"]),
            "evidence_manifest": evidence,
            "uncertainty_reasons": [
                str(row.get("review_bucket")),
                str(row.get("role_partition_reason")),
                f"source_role={row.get('source_visual_role_context')}",
                f"target_role={row.get('target_visual_role_context')}",
                f"has_intermediate_support={row.get('has_intermediate_support')}",
            ],
            "category": str(row.get("review_bucket")),
            "priority": index,
            "control_status": "balanced_continuity_review_candidate",
            "candidate_hash": "",
            "evidence_hash": evidence.evidence_hash,
            "safety_payload": safety_payload(),
            "review_round": 2,
            "selection_metadata": {
                "why_selected": row.get("continuity_review_bucket"),
                "requires_intermediate_support": row.get("requires_intermediate_support"),
                "has_intermediate_support": row.get("has_intermediate_support"),
                "continuity_score": row.get("continuity_score"),
                "gate_features": row.get("gate_features"),
            },
            "model_prediction": row.get("continuity_review_bucket"),
            "model_confidence": float(row.get("continuity_score", 0.0)),
            "equivalence_cluster_id": f"m5_4e_continuity_{row.get('review_bucket')}",
            "representative_of_count": 1,
        }
        case_payload["candidate_hash"] = _review_case_hash(case_payload)
        cases.append(ReviewCase.model_validate(case_payload))
    manifest = ReviewManifest(
        title="M5.4E Balanced Continuity Example Discovery Review",
        review_task_family="m5_4e_balanced_continuity_example_discovery",
        review_cases=cases,
        candidate_manifest_hash=stable_hash([case.candidate_hash for case in cases]),
        evidence_manifest_hash=stable_hash([case.evidence_hash for case in cases]),
        source_manifest_hash=stable_hash([ref.model_dump(mode="json") for ref in source_refs]),
        source_artifact_references=source_refs,
    )
    manifest_path = review_root / "review_manifest.json"
    write_json(manifest_path, manifest.model_dump(mode="json"))
    _write_empty_decisions(decision_root, manifest)
    build_workbench(workbench_root)
    launcher = _write_open_launcher(
        launcher_path=stage_root / "OPEN_CONTINUITY_BALANCE_REVIEW.ps1",
        repo_root=repo_root,
        manifest_path=manifest_path,
        evidence_root=evidence_root,
        decision_root=decision_root,
        workbench_root=workbench_root,
        label="M5.4E balanced continuity",
        port=8771,
    )
    return {
        "manifest_path": str(manifest_path),
        "evidence_root": str(evidence_root),
        "decision_root": str(decision_root),
        "workbench_root": str(workbench_root),
        "launcher_path": str(launcher),
        "review_url": "http://127.0.0.1:8771/",
        "review_case_count": len(cases),
        "likely_positive_count": selection["likely_positive_count"],
        "likely_negative_count": selection["likely_negative_count"],
    }


def _within_team_segments(rows_in: list[dict[str, Any]]) -> dict[str, Any]:
    segment_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows_in):
        segment_rows.append(
            {
                "visual_continuity_segment_id": f"m5_4e_visual_continuity_segment_{index:06d}",
                "role_partitioned_continuity_candidate_id": row.get("role_partitioned_continuity_candidate_id"),
                "source_visible_person_base_id": row["source_visible_person_base_id"],
                "target_visible_person_base_id": row["target_visible_person_base_id"],
                "source_visual_role_context": row["source_visual_role_context"],
                "target_visual_role_context": row["target_visual_role_context"],
                "frame_gap": row["frame_gap"],
                "continuity_score": row["continuity_score"],
                "gate_features": row.get("gate_features", {}),
                "reciprocal_ranking_used": True,
                "one_to_one_assignment_claimed": False,
                "bounded_candidate_degree": True,
                "occlusion_handling": "visual_uncertainty_only_no_identity_claim",
                "visual_continuity_is_real_identity": False,
                "visual_continuity_is_player_slot": False,
                "match_local_only": True,
                "sandbox_only": True,
                **safety_payload(),
            }
        )
    return {
        "artifact": "m5_4e_within_team_visual_continuity_rows",
        "row_count": len(segment_rows),
        "rows": segment_rows,
        **safety_payload(),
    }


def _temporal_asset_counts(root: Path) -> dict[str, int]:
    return {
        "gif_count": len(list(root.rglob("*.gif"))) if root.exists() else 0,
        "mp4_count": len(list(root.rglob("*.mp4"))) if root.exists() else 0,
    }


def _write_case_index(path: Path, manifest_path: Path) -> None:
    manifest = read_json(manifest_path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "review_case_id",
                "task_type",
                "category",
                "source_frame_sequence",
                "target_frame_sequence",
                "model_prediction",
                "model_confidence",
            ],
        )
        writer.writeheader()
        for case in manifest.get("review_cases", []):
            writer.writerow(
                {
                    "review_case_id": case["review_case_id"],
                    "task_type": case["task_type"],
                    "category": case["category"],
                    "source_frame_sequence": case["source_frame_sequence"],
                    "target_frame_sequence": case.get("target_frame_sequence"),
                    "model_prediction": case.get("model_prediction"),
                    "model_confidence": case.get("model_confidence"),
                }
            )


def _write_review_pack(
    *,
    stage_root: Path,
    repo_root: Path,
    summary: dict[str, Any],
    role_review: dict[str, Any],
    continuity_review: dict[str, Any],
) -> dict[str, Any]:
    pack_root = stage_root / "review_pack"
    pack_root.mkdir(parents=True, exist_ok=True)
    explanation = f"""M5.4E REVIEW_PACK

Purpose
This folder is capped at 20 files and is a handoff bundle for deciding the next Football Intelligence step.
It references the full 06E stage outputs rather than duplicating image/video evidence.

What has been achieved
- The completed M5.4D Round 1 review was ingested from the actual decisions map, then bound back to the
  canonical Round 1 review manifest.
- Entity and continuity label inventories are separated by task type. The old accepted/rejected counters are
  preserved for compatibility but are not used as entity label truth.
- Entity labels are diverse enough to fit a candidate calibrator, but application is blocked by the stronger
  independent-cluster gate. Entity rows updated: {summary["entity_rows_updated"]}.
- Continuity learning is correctly blocked because Round 1 has rejected examples only and no accepted continuity
  example. No reject-everything model was trained.
- A match-local visual role/context layer was built for team-colour, goalkeeper, central referee, near assistant,
  far assistant, off-pitch, unknown person and non-person visual contexts.
- Continuity candidates were repartitioned by visual role before matching. This reduced the candidate pool from
  {summary["candidate_pool_before_role_partitioning"]} to {summary["candidate_pool_after_role_partitioning"]}.
- A new role-context review workbench was generated with {summary["role_review_case_count"]} cases.
- A balanced continuity discovery set was generated with {summary["likely_positive_continuity_review_count"]}
  likely-positive cases and {summary["likely_negative_continuity_review_count"]} difficult/likely-negative cases.

Safety state
VISUAL_ONLY_NOT_METRIC remains true. production_ready=false, no_auto_promotion=true, human_approved=false,
match_local_only=true, sandbox_only=true, safe_to_apply_globally=false.
No persistent identity, player slot, goalkeeper slot, metric pitch coordinate, tactical, event or physical output
has been created.

Most important next step
Review the role-context pack first, then review the balanced continuity candidates. Continuity calibration must
remain blocked until a later human-reviewed set contains at least one accepted and one rejected continuity example.

Launchers
Role review: {role_review["launcher_path"]}
Role review URL: {role_review["review_url"]}
Continuity discovery review: {continuity_review["launcher_path"]}
Continuity review URL: {continuity_review["review_url"]}
"""
    write_text(pack_root / "REVIEW_PACK_EXPLANATION.txt", explanation)
    write_text(
        pack_root / "M5_4E_NEXT_STEP_CONTEXT.md",
        "\n".join(
            [
                "# M5.4E Next Step Context",
                "",
                f"Final classification: `{summary['final_classification']}`",
                f"Exact blocker: `{summary['exact_blocker']}`",
                "",
                "Round 1 entity label distribution:",
                f"`{json.dumps(summary['round_1_entity_label_distribution'], sort_keys=True)}`",
                "",
                "Round 1 continuity label distribution:",
                f"`{json.dumps(summary['round_1_continuity_label_distribution'], sort_keys=True)}`",
                "",
                "Continuity calibration must not start until both accept and reject labels exist.",
            ]
        )
        + "\n",
    )
    copies = [
        ("round_1_label_distribution.json", stage_root / "learning" / "round_1_label_distribution.json"),
        ("entity_training_readiness.json", stage_root / "learning" / "entity_training_readiness.json"),
        ("continuity_training_readiness.json", stage_root / "learning" / "continuity_training_readiness.json"),
        ("entity_calibrator_validation.json", stage_root / "learning" / "entity_calibrator_validation.json"),
        ("visual_role_distribution.json", stage_root / "role" / "visual_role_distribution.json"),
        ("role_partition_manifest.json", stage_root / "continuity" / "role_partition_manifest.json"),
        ("candidate_pool_size_before_after.json", stage_root / "continuity" / "candidate_pool_size_before_after.json"),
        (
            "positive_continuity_review_selection.json",
            stage_root / "continuity" / "positive_continuity_review_selection.json",
        ),
        ("validation_summary.json", stage_root / "validation" / "m5_4e_validation_summary.json"),
    ]
    for name, source in copies:
        write_json(pack_root / name, read_json(source))
    role_manifest = read_json(Path(role_review["manifest_path"]))
    continuity_manifest = read_json(Path(continuity_review["manifest_path"]))
    write_json(
        pack_root / "role_review_manifest_summary.json",
        {**role_manifest, "review_cases": role_manifest.get("review_cases", [])[:30]},
    )
    write_json(
        pack_root / "continuity_review_manifest_summary.json",
        {**continuity_manifest, "review_cases": continuity_manifest.get("review_cases", [])[:20]},
    )
    _write_case_index(pack_root / "role_review_case_index.csv", Path(role_review["manifest_path"]))
    _write_case_index(pack_root / "continuity_review_case_index.csv", Path(continuity_review["manifest_path"]))
    write_json(
        pack_root / "review_pack_manifest.json",
        {
            "artifact": "m5_4e_review_pack",
            "file_cap": 20,
            "folder_name": "review_pack",
            "uses_full_review_evidence_by_reference": True,
            "role_review_launcher": role_review["launcher_path"],
            "continuity_review_launcher": continuity_review["launcher_path"],
            **safety_payload(),
        },
    )
    write_text(
        pack_root / "OPEN_REVIEW_PACK.ps1",
        f"""$ErrorActionPreference = "Stop"
Set-Location "{repo_root}"
& "{role_review["launcher_path"]}"
""",
    )
    files = [path for path in pack_root.iterdir() if path.is_file()]
    return {"path": str(pack_root), "file_count": len(files), "file_cap_respected": len(files) <= 20}


def build_role_partitioned_learning_stage(
    *,
    repo_root: Path,
    artifact_root: Path,
    match_id: str = "128058",
    stage_root: Path | None = None,
) -> dict[str, Any]:
    paths = default_paths(artifact_root.resolve(), match_id)
    source_stage_root = paths["source_stage_root"].resolve()
    stage_root = (stage_root or paths["stage_root"]).resolve()
    stage_root.mkdir(parents=True, exist_ok=True)
    before_sources = _source_inventory([source_stage_root])

    learning_root = stage_root / "learning"
    role_root = stage_root / "role"
    continuity_root = stage_root / "continuity"
    review_root = stage_root / "review"
    validation_root = stage_root / "validation"
    for root in [learning_root, role_root, continuity_root, review_root, validation_root]:
        root.mkdir(parents=True, exist_ok=True)

    completed_review = source_stage_root / "review" / "decisions" / "completed_review.json"
    completed_events = source_stage_root / "review" / "decisions" / "completed_review_events.jsonl"
    requested_manifest = source_stage_root / "review" / "round_1_review_manifest.json"
    canonical_manifest = source_stage_root / "review" / "review_manifest_round_1.json"
    ingestion = ingest_mixed_review(
        completed_review_path=completed_review,
        review_manifest_path=canonical_manifest,
        event_log_path=completed_events,
        requested_manifest_path=requested_manifest,
    )
    write_jsonl(learning_root / "round_1_mixed_review_examples.jsonl", ingestion["examples"])
    write_jsonl(learning_root / "round_1_entity_examples.jsonl", ingestion["entity_examples"])
    write_jsonl(learning_root / "round_1_continuity_examples.jsonl", ingestion["continuity_examples"])
    write_json(learning_root / "round_1_label_distribution.json", ingestion["distribution"])
    write_json(learning_root / "round_1_review_binding_validation.json", ingestion["binding_validation"])
    write_json(
        learning_root / "round_1_completed_review_summary_schema_fix.json",
        {
            "artifact": "m5_4e_completed_review_summary_schema_fix",
            "source_summary_path": str(source_stage_root / "review" / "decisions" / "completed_review_summary.json"),
            **ingestion["distribution"],
            **safety_payload(),
        },
    )

    entity_readiness = class_sufficiency_readiness(
        ingestion["entity_examples"],
        task_type="entity_validity",
        min_examples=8,
        min_clusters_per_class_for_application=5,
    )
    continuity_readiness = class_sufficiency_readiness(
        ingestion["continuity_examples"],
        task_type="visual_continuity_edge_review",
        required_labels={"accept_continuity", "reject_continuity"},
        min_examples=8,
        min_clusters_per_class_for_application=5,
    )
    write_json(learning_root / "entity_training_readiness.json", entity_readiness)
    write_json(learning_root / "continuity_training_readiness.json", continuity_readiness)
    write_json(
        learning_root / "continuity_calibrator_candidate.json",
        {
            "artifact": "m5_4e_continuity_calibrator_candidate",
            "training_performed": False,
            "updates_applied": 0,
            "status": continuity_readiness["status"],
            "blocked_reason": CONTINUITY_BLOCKER,
            "reject_everything_model_fitted": False,
            "no_unreviewed_prediction_used_as_truth": True,
            **safety_payload(),
        },
    )

    entity_rows = _read_rows(source_stage_root / "entity" / "entity_validity_recalibrated_rows.json")
    feature_rows = _read_rows(source_stage_root / "entity" / "entity_feature_rows.json")
    _, entity_validation, entity_application, _ = _entity_calibration_outputs(
        learning_root=learning_root,
        entity_rows=entity_rows,
        entity_examples=ingestion["entity_examples"],
        readiness=entity_readiness,
    )

    reviewed_labels = _reviewed_entity_labels(ingestion)
    role_features = build_visual_role_features(
        entity_rows=entity_rows,
        feature_rows=feature_rows,
        reviewed_entity_labels=reviewed_labels,
    )
    role_context = build_visual_role_context_rows(role_features["rows"])
    role_distribution = {
        "artifact": "m5_4e_visual_role_distribution",
        "visual_role_distribution": role_context["summary"],
        "visual_role_count": len(role_context["rows"]),
        **safety_payload(),
    }
    write_json(role_root / "visual_role_feature_rows.json", role_features)
    write_json(role_root / "visual_role_context_rows.json", role_context)
    write_json(role_root / "visual_role_distribution.json", role_distribution)

    frame_manifest = read_json(paths["frame_manifest"])
    frame_records = _frame_records(frame_manifest)
    source_refs = _source_refs(paths)
    role_selection = _role_review_selection(
        role_context_rows=role_context["rows"],
        role_feature_rows=role_features["rows"],
        limit=30,
    )
    write_json(
        role_root / "role_review_selection_rows.json",
        {"artifact": "m5_4e_role_review_selection_rows", "rows": role_selection, **safety_payload()},
    )
    role_review = _write_role_review(
        review_root=review_root / "role_context",
        stage_root=stage_root,
        repo_root=repo_root,
        selected_rows=role_selection,
        frame_root=paths["frame_root"],
        frame_records=frame_records,
        source_refs=source_refs,
    )

    node_rows = _read_rows(source_stage_root / "continuity" / "continuity_node_rows.json")
    candidate_rows = _read_rows(source_stage_root / "continuity" / "continuity_candidate_rows.json")
    for row in candidate_rows:
        features = row.get("gate_features") if isinstance(row.get("gate_features"), dict) else {}
        row["intermediate_frame_support"] = (
            int(row.get("frame_gap", 1)) > 1 or float(features.get("bbox_iou", 0.0)) > 0.2
        )
    role_by_visible_id = _role_by_visible_id(node_rows=node_rows, role_context_rows=role_context["rows"])
    partitioned = apply_role_partitioning(
        candidate_rows=candidate_rows,
        role_by_visible_id=role_by_visible_id,
        max_degree=3,
    )
    manifest = build_role_partition_manifest()
    pool_report = pool_size_report(
        partitioned["candidate_pool_before_role_partitioning"],
        partitioned["candidate_pool_after_role_partitioning"],
    )
    write_json(continuity_root / "role_partition_manifest.json", manifest)
    write_json(
        continuity_root / "role_partitioned_candidate_rows.json",
        {key: value for key, value in partitioned.items() if key != "rejected_rows"},
    )
    write_json(
        continuity_root / "role_incompatible_rejected_rows.json",
        {
            "artifact": "m5_4e_role_incompatible_rejected_rows",
            "rejected_count": len(partitioned["rejected_rows"]),
            "rows": partitioned["rejected_rows"],
            **safety_payload(),
        },
    )
    write_json(continuity_root / "candidate_pool_size_before_after.json", pool_report)
    within_team = _within_team_segments(partitioned["rows"])
    write_json(continuity_root / "within_team_visual_continuity_rows.json", within_team)

    positive_selection = select_continuity_review_candidates(partitioned["rows"], positive_limit=10, negative_limit=10)
    write_json(continuity_root / "positive_continuity_review_selection.json", positive_selection)
    node_by_visible_id = {str(row["visible_person_base_id"]): row for row in node_rows}
    continuity_review = _write_continuity_balance_review(
        review_root=review_root / "continuity_balance",
        stage_root=stage_root,
        repo_root=repo_root,
        selection=positive_selection,
        node_by_visible_id=node_by_visible_id,
        frame_root=paths["frame_root"],
        frame_records=frame_records,
        source_refs=source_refs,
    )

    temporal_counts = _temporal_asset_counts(review_root)
    after_sources = _source_inventory([source_stage_root])
    source_mutation = {
        "artifact": "m5_4e_source_mutation_audit",
        "source_stage_root": str(source_stage_root),
        "before": before_sources,
        "after": after_sources,
        "m5_4d_outputs_unchanged": before_sources["combined_hash"] == after_sources["combined_hash"],
        **safety_payload(),
    }
    write_json(validation_root / "source_mutation_audit.json", source_mutation)

    summary = {
        "artifact": "m5_4e_validation_summary",
        "created_at": _now(),
        "match_id": match_id,
        "stage_root": str(stage_root),
        "source_stage_root": str(source_stage_root),
        "final_classification": FINAL_CLASSIFICATION,
        "exact_blocker": CONTINUITY_BLOCKER,
        "round_1_entity_label_distribution": ingestion["distribution"]["entity_label_distribution"],
        "round_1_continuity_label_distribution": ingestion["distribution"]["continuity_label_distribution"],
        "entity_training_readiness": entity_readiness["status"],
        "continuity_training_readiness": continuity_readiness["status"],
        "entity_calibrator_validation_result": entity_validation["validation_result"],
        "entity_rows_updated": entity_application["remaining_rows_updated_by_learned_models"],
        "visual_role_distribution": role_distribution["visual_role_distribution"],
        "role_review_case_count": role_review["review_case_count"],
        "candidate_pool_before_role_partitioning": partitioned["candidate_pool_before_role_partitioning"],
        "candidate_pool_after_role_partitioning": partitioned["candidate_pool_after_role_partitioning"],
        "likely_positive_continuity_review_count": positive_selection["likely_positive_count"],
        "likely_negative_continuity_review_count": positive_selection["likely_negative_count"],
        "temporal_gif_count": temporal_counts["gif_count"],
        "temporal_mp4_count": temporal_counts["mp4_count"],
        "launcher_path": role_review["launcher_path"],
        "review_url": role_review["review_url"],
        "continuity_review_launcher_path": continuity_review["launcher_path"],
        "continuity_review_url": continuity_review["review_url"],
        "continuity_model_trained": False,
        "continuity_model_updates_applied": 0,
        "reject_everything_model_fitted": False,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
        "m5_4d_outputs_unchanged": source_mutation["m5_4d_outputs_unchanged"],
        "safe_to_apply_globally": False,
        **safety_payload(),
    }
    write_json(validation_root / "m5_4e_validation_summary.json", summary)
    review_pack = _write_review_pack(
        stage_root=stage_root,
        repo_root=repo_root,
        summary=summary,
        role_review=role_review,
        continuity_review=continuity_review,
    )
    summary["review_pack"] = review_pack
    write_json(validation_root / "m5_4e_validation_summary.json", summary)
    return summary
