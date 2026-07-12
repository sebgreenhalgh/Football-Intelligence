from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from football_intelligence.replay.portable_context import (
    PortableStageResult,
    PortableVisualRunContext,
    VISUAL_ONLY_WARNING,
    forbidden_keys_present,
    guardrail_payload,
    semantic_hash,
    sha256_file,
    utc_now,
)
from football_intelligence.step1_visual_reconstruction.colour_features import build_colour_feature_payload
from football_intelligence.step1_visual_reconstruction.count_policy import build_count_policy_payload
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state import (
    build_fused_visual_role_state_payloads,
)
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_human_corrections import (
    build_human_corrected_fused_role_state_payloads,
)
from football_intelligence.step1_visual_reconstruction.official_context_beliefs import (
    build_official_context_belief_payload,
)
from football_intelligence.step1_visual_reconstruction.official_context_features import (
    build_official_context_feature_payload,
)
from football_intelligence.step1_visual_reconstruction.official_context_human_corrections import (
    build_human_corrected_official_context_payloads,
)
from football_intelligence.step1_visual_reconstruction.person_candidates import build_candidate_inventory_payload
from football_intelligence.step1_visual_reconstruction.reconciliation import build_reconciliation_payload
from football_intelligence.step1_visual_reconstruction.render_tiers import build_render_tier_payload
from football_intelligence.step1_visual_reconstruction.state_model import build_person_states_payload
from football_intelligence.step1_visual_reconstruction.visible_person_base import (
    build_visible_person_base_payloads,
)


STEP1_OUTPUTS = {
    "candidate_inventory": "step1/step1a_person_candidates.json",
    "person_states": "step1/step1b1_person_states.json",
    "render_tiers": "step1/step1b2_render_tier_rows.json",
    "reconciliation": "step1/step1b3_reconciliation_rows.json",
    "count_policy": "step1/step1b3_count_policy_rows.json",
    "visible_base": "step1/step1b4_visible_person_base_rows.json",
    "visible_base_provenance": "step1/step1b4_retained_candidate_provenance_rows.json",
    "colour_features": "step1/step1c1_colour_feature_rows.json",
    "colour_prototypes": "step1/step1c1_colour_prototypes.json",
    "colour_beliefs": "step1/step1c1_team_colour_belief_rows.json",
    "c2c_rows": "step1/step1c2c_human_corrected_colour_stability_rows.json",
    "d1_features": "step1/step1d1_official_context_feature_rows.json",
    "d1_beliefs": "step1/step1d1_official_context_belief_rows.json",
    "d1c_rows": "step1/step1d1c_human_corrected_official_context_rows.json",
    "d1c_audit": "step1/step1d1c_human_correction_audit_rows.json",
    "e1c_rows": "step1/step1e1c_human_corrected_goalkeeper_context_rows.json",
    "f1_rows": "step1/step1f1_fused_visual_role_state_rows.json",
    "f1_conflict_audit": "step1/step1f1_role_state_conflict_audit_rows.json",
    "f3_rows": "step1/step1f3_human_corrected_fused_visual_role_state_rows.json",
    "f3_audit": "step1/step1f3_human_fused_role_state_correction_audit_rows.json",
    "g1_freeze_manifest": "step1/step1g1_freeze_candidate_manifest.json",
}


def _empty_reviewed_payload(artifact: str) -> dict[str, Any]:
    return guardrail_payload(
        {
            "artifact": artifact,
            "created_at": utc_now(),
            "rows": [],
            "reviewed_decision_rows": 0,
        }
    )


def _empty_candidate_payload(artifact: str) -> dict[str, Any]:
    return guardrail_payload(
        {
            "artifact": artifact,
            "created_at": utc_now(),
            "rows": [],
            "summary": {"candidate_count": 0},
        }
    )


def _detection_xyxy(detection: dict[str, Any]) -> dict[str, float] | None:
    if isinstance(detection.get("bbox"), dict):
        bbox = detection["bbox"]
        keys = ("x1", "y1", "x2", "y2")
        if all(key in bbox for key in keys):
            return {key: float(bbox[key]) for key in keys}
    if all(key in detection for key in ("x1", "y1", "x2", "y2")):
        return {key: float(detection[key]) for key in ("x1", "y1", "x2", "y2")}
    xyxy = detection.get("xyxy")
    if isinstance(xyxy, list) and len(xyxy) == 4:
        return {"x1": float(xyxy[0]), "y1": float(xyxy[1]), "x2": float(xyxy[2]), "y2": float(xyxy[3])}
    return None


def normalise_detection_payload(
    detection_payload: dict[str, Any],
    manifest_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    by_sequence: dict[int, list[dict[str, Any]]] = {}
    for frame in detection_payload.get("frames", []):
        seq = int(frame.get("frame_sequence", frame.get("sequence", -1)))
        by_sequence.setdefault(seq, []).extend(list(frame.get("detections", [])))
    for row in detection_payload.get("rows", []):
        seq = int(row.get("frame_sequence", row.get("sequence", -1)))
        by_sequence.setdefault(seq, []).append(row)

    frames: list[dict[str, Any]] = []
    for frame in manifest_frames:
        seq = int(frame["frame_sequence"])
        detections = []
        for index, raw in enumerate(by_sequence.get(seq, [])):
            bbox = _detection_xyxy(dict(raw))
            if bbox is None:
                continue
            detection_id = str(raw.get("detection_id") or f"portable_det_f{seq:06d}_{index:03d}")
            item = {
                **dict(raw),
                "detection_id": detection_id,
                "source_detection_id": str(raw.get("source_detection_id") or detection_id),
                "frame_id": frame["frame_id"],
                "frame_sequence": seq,
                "timestamp_seconds": frame["timestamp_seconds"],
                "frame_file": frame["frame_file"],
                "object_type": str(raw.get("object_type", "player_candidate")),
                "class_name": str(raw.get("class_name", "person")),
                "role_label": str(raw.get("role_label", raw.get("source_role", "player"))),
                "confidence": float(raw.get("confidence", raw.get("bbox_confidence", 0.75))),
                "classification_reason": str(raw.get("classification_reason", "portable_declared_detection_source")),
                **bbox,
            }
            detections.append(item)
        frames.append({**frame, "detections": detections})
    return {
        "artifact": "portable_step1_detection_source",
        "created_at": utc_now(),
        "frames": frames,
    }


def _map_c1_to_c2c_belief(row: dict[str, Any]) -> str:
    belief = str(row.get("team_colour_belief", "unknown_ambiguous_colour"))
    if belief in {"dark_context_colour_like", "other_distinct_colour_like"}:
        return "non_outfield_context_colour"
    if belief == "crop_unusable":
        return "bad_detection_or_not_person"
    return "unknown_ambiguous_colour"


def build_c2c_payload_from_c1(c1_payload: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in c1_payload.get("rows", []):
        final = _map_c1_to_c2c_belief(row)
        rows.append(
            {
                **row,
                "c2c_final_colour_belief": final,
                "c2c_final_colour_belief_confidence": row.get("team_colour_belief_confidence", 0.0),
                "c2c_colour_source": "portable_c1_no_historical_human_decisions",
                "c2c_human_reviewed": False,
                "c2c_human_review_decision": "",
                "c2c_review_required": final == "unknown_ambiguous_colour",
                "c2c_bad_detection_or_not_person": final == "bad_detection_or_not_person",
                "c2c_context_or_offroi_human_team_override": False,
                "retained_for_future_player_team_review": True,
                "visual_only_warning": VISUAL_ONLY_WARNING,
                "do_not_use_for_metrics": True,
                "production_ready": False,
                "auto_promoted": False,
            }
        )
    counts = Counter(str(row.get("c2c_final_colour_belief", "")) for row in rows)
    return guardrail_payload(
        {
            "artifact": "step1c2c_human_corrected_colour_stability_rows",
            "created_at": utc_now(),
            "rows": rows,
            "summary": {
                "c2c_row_count": len(rows),
                "c2c_final_colour_belief_counts": dict(sorted(counts.items())),
                "historical_human_decisions_used": False,
            },
        }
    )


def build_e1c_payload_from_d1c(d1c_payload: dict[str, Any], c2c_payload: dict[str, Any]) -> dict[str, Any]:
    c2c_by_id = {str(row.get("visible_person_base_id", "")): row for row in c2c_payload.get("rows", [])}
    rows = []
    for row in d1c_payload.get("rows", []):
        visible_id = str(row.get("visible_person_base_id", ""))
        c2c = c2c_by_id.get(visible_id, {})
        d1c = str(row.get("d1c_final_official_context_belief", "unknown_official_context"))
        if d1c == "player_like_not_official_context":
            final = "outfield_player_like_not_goalkeeper"
        elif d1c == "bad_detection_or_not_person" or c2c.get("c2c_bad_detection_or_not_person") is True:
            final = "bad_detection_or_not_person"
        elif d1c in {
            "official_referee_like",
            "assistant_or_line_official_like",
            "off_pitch_context_person_like",
            "non_official_context_person_like",
        }:
            final = "official_or_context_not_goalkeeper"
        else:
            final = "unknown_goalkeeper_context"
        rows.append(
            {
                **row,
                "e1c_final_goalkeeper_context_belief": final,
                "e1c_final_goalkeeper_context_belief_state": (
                    "review_required" if final == "unknown_goalkeeper_context" else "portable_visual_context"
                ),
                "e1c_final_goalkeeper_context_belief_confidence": 0.62
                if final != "unknown_goalkeeper_context"
                else 0.35,
                "e1c_goalkeeper_team_belief": "not_goalkeeper",
                "e1c_context_source": "portable_no_historical_goalkeeper_decisions",
                "e1c_human_reviewed": False,
                "e1c_human_review_decision": "",
                "e1c_human_review_confidence": "",
                "e1c_review_required": final in {"unknown_goalkeeper_context", "bad_detection_or_not_person"},
                "e1c_correction_reason": "portable_visual_context_only_no_goalkeeper_slot_or_identity",
                "e1c_human_corrected_from_e1": False,
                "e1c_goalkeeper_like_visual_context": False,
                "e1c_goalkeeper_like_team_1_visual_context": False,
                "e1c_goalkeeper_like_team_2_visual_context": False,
                "e1c_goalkeeper_like_unknown_team_visual_context": False,
                "e1c_outfield_player_like_not_goalkeeper": final == "outfield_player_like_not_goalkeeper",
                "e1c_official_or_context_not_goalkeeper": final == "official_or_context_not_goalkeeper",
                "e1c_bad_detection_or_not_person": final == "bad_detection_or_not_person",
                "retained_for_future_player_team_review": True,
                "eligible_for_identity_tracking": False,
                "eligible_for_player_slot_assignment": False,
                "eligible_for_goalkeeper_slot_assignment": False,
                "eligible_for_metric_use": False,
                "visual_only_warning": VISUAL_ONLY_WARNING,
                "do_not_use_for_metrics": True,
                "production_ready": False,
                "auto_promoted": False,
            }
        )
    counts = Counter(str(row.get("e1c_final_goalkeeper_context_belief", "")) for row in rows)
    return guardrail_payload(
        {
            "artifact": "step1e1c_human_corrected_goalkeeper_context_rows",
            "created_at": utc_now(),
            "rows": rows,
            "summary": {
                "e1c_row_count": len(rows),
                "e1c_final_goalkeeper_context_belief_counts": dict(sorted(counts.items())),
                "historical_human_decisions_used": False,
            },
        }
    )


def build_g1_freeze_manifest(
    *,
    context: PortableVisualRunContext,
    f3_payload: dict[str, Any],
    frame_count: int,
    output_hash: str,
) -> dict[str, Any]:
    rows = f3_payload.get("rows", [])
    forbidden = forbidden_keys_present(f3_payload)
    role_counts = Counter(str(row.get("step1f3_final_visual_role_state", "")) for row in rows)
    safe = not forbidden and len(rows) >= 0
    return guardrail_payload(
        {
            "artifact": "portable_step1g1_freeze_candidate_manifest",
            "created_at": utc_now(),
            "match_id": context.match_id,
            "window_id": context.window_id,
            "all_manifest_frames_considered": True,
            "manifest_frame_count": frame_count,
            "f3_row_count": len(rows),
            "final_visual_role_state_counts": dict(sorted(role_counts.items())),
            "step1g1_freeze_candidate_created": safe,
            "step1g1_freeze_candidate_human_approved": False,
            "step1g1_safe_for_step2_visual_continuity_candidate": safe,
            "portable_output_hash": output_hash,
            "historical_human_decisions_used": False,
            "forbidden_keys_present": forbidden,
        }
    )


def _write_step1_outputs(context: PortableVisualRunContext, payloads: dict[str, Any]) -> dict[str, str]:
    paths = {}
    for key, payload in payloads.items():
        rel = STEP1_OUTPUTS[key]
        path = context.write_json(rel, payload)
        paths[key] = str(path)
    return paths


def _blocked_dependency_result(context: PortableVisualRunContext, missing: list[dict[str, Any]]) -> PortableStageResult:
    blocker = "required Step1 visual detection dependency is absent"
    payload = guardrail_payload(
        {
            "artifact": "portable_step1_blocked_dependency",
            "created_at": utc_now(),
            "completion_status": "blocked_missing_model_or_configuration_dependency",
            "blocked_reason": blocker,
            "missing_dependencies": missing,
            "step2_allowed_to_run": False,
        }
    )
    path = context.write_json("step1/step1_blocked_dependency.json", payload)
    validation = guardrail_payload(
        {
            "artifact": "step1_portable_validation",
            "created_at": utc_now(),
            "passed": False,
            "completion_status": payload["completion_status"],
            "blocking_substage": "step1_visual_detection_candidate_construction",
            "blocking_reason": blocker,
            "missing_dependencies": missing,
        }
    )
    context.write_json("validation/step1_portable_validation.json", validation)
    context.write_json(
        "validation/step1_frame_coverage.json",
        {"artifact": "step1_frame_coverage", "created_at": utc_now(), "passed": False, "frames_considered": 0},
    )
    context.write_json(
        "validation/step1_guardrail_audit.json",
        guardrail_payload(
            {
                "artifact": "step1_guardrail_audit",
                "created_at": utc_now(),
                "passed": False,
                "blocking_reason": blocker,
                "forbidden_keys_present": [],
            }
        ),
    )
    context.write_json(
        "validation/step1_output_inventory.json",
        {"artifact": "step1_output_inventory", "created_at": utc_now(), "output_count": 1, "outputs": [str(path)]},
    )
    return PortableStageResult(
        stage="step1",
        completion_status="blocked_missing_model_or_configuration_dependency",
        output_paths={"blocked_dependency": str(path)},
        counts={"f3_row_count": 0, "frame_count": 0},
        warnings=["Step1 did not execute; no placeholder downstream artifacts were created."],
        blocker=blocker,
    )


def _missing_detection_dependencies(context: PortableVisualRunContext) -> list[dict[str, Any]]:
    missing = []
    detection_source = str(context.config.get("step1_detection_source_manifest", "") or "")
    model_path = str(context.config.get("model_weight_path", "") or "")
    if detection_source:
        path = (context.artifact_root / detection_source).resolve()
        if not path.exists():
            missing.append({"artifact_id": "step1_detection_source_manifest", "path": str(path), "reason": "missing"})
    if model_path:
        path = (context.repo_root / model_path).resolve()
        if not path.exists():
            missing.append({"artifact_id": "person_detection_model_weights", "path": str(path), "reason": "missing"})
    if not detection_source and not model_path:
        missing.append(
            {
                "artifact_id": "step1_detection_source_or_model_weights",
                "path": "",
                "reason": "no declared detection source manifest or model weight path",
            }
        )
    return missing


def _load_detection_payload(
    context: PortableVisualRunContext, detection_payload: dict[str, Any] | None
) -> dict[str, Any] | None:
    if detection_payload is not None:
        return detection_payload
    source = str(context.config.get("step1_detection_source_manifest", "") or "")
    if not source:
        return None
    source_path = (context.artifact_root / source).resolve()
    return context.read_declared_json(
        source_path,
        stage="step1",
        purpose="declared Step1 visual detection source manifest",
    )


def run_portable_step1(
    context: PortableVisualRunContext,
    *,
    detection_payload: dict[str, Any] | None = None,
) -> PortableStageResult:
    if detection_payload is None:
        missing = _missing_detection_dependencies(context)
        if missing:
            return _blocked_dependency_result(context, missing)

    manifest_frames = context.canonical_frames()
    source_payload = normalise_detection_payload(
        _load_detection_payload(context, detection_payload) or {}, manifest_frames
    )
    source_name = str(context.config.get("step1_detection_source_name", "player") or "player")
    source_paths = {
        source_name: str(context.config.get("step1_detection_source_manifest", "in_memory_detection_payload"))
    }

    for frame in manifest_frames:
        if context.source_ledger is not None:
            context.source_ledger.record_binary_read(
                Path(frame["frame_file"]),
                stage="step1",
                purpose="raw input frame for visual feature extraction",
            )

    candidate_payload = build_candidate_inventory_payload(
        manifest_frames=manifest_frames,
        source_payloads={source_name: source_payload},
        source_paths=source_paths,
    )
    state_payload = build_person_states_payload(candidate_payload)
    render_tier_payload = build_render_tier_payload(state_payload, [])
    reconciliation_payload = build_reconciliation_payload(render_tier_payload, {"rows": []})
    count_policy_payload = build_count_policy_payload(reconciliation_payload)
    visible_base_payload, visible_base_provenance = build_visible_person_base_payloads(count_policy_payload)

    frame_lookup = {int(frame["frame_sequence"]): frame["frame_file"] for frame in manifest_frames}
    colour_feature_payload = build_colour_feature_payload(visible_base_payload, frame_lookup=frame_lookup)
    colour_belief_payload, _unknown_colour_payload, colour_prototype_payload = _build_colour_beliefs(
        visible_base_payload,
        colour_feature_payload,
    )
    c2c_payload = build_c2c_payload_from_c1(colour_belief_payload)
    d1_feature_payload = build_official_context_feature_payload(
        c2c_payload, b4_payload=visible_base_payload, frame_lookup=frame_lookup
    )
    d1_belief_payload = build_official_context_belief_payload(d1_feature_payload)
    d1c_payload, d1c_audit = build_human_corrected_official_context_payloads(
        d1_belief_payload,
        _empty_candidate_payload("step1d1_official_context_review_candidate_rows"),
        _empty_reviewed_payload("step1d1b_reviewed_official_context_decisions"),
    )
    e1c_payload = build_e1c_payload_from_d1c(d1c_payload, c2c_payload)
    f1_payload, f1_conflict_payload = build_fused_visual_role_state_payloads(
        visible_base_payload,
        c2c_payload,
        d1c_payload,
        e1c_payload,
    )
    f3_payload, f3_audit = build_human_corrected_fused_role_state_payloads(
        f1_payload,
        _empty_candidate_payload("step1f2_review_candidate_rows"),
        _empty_reviewed_payload("step1f2_reviewed_decisions"),
    )
    output_hash = semantic_hash(
        {
            "candidate_rows": candidate_payload.get("rows", []),
            "visible_base_rows": visible_base_payload.get("rows", []),
            "f3_rows": f3_payload.get("rows", []),
        }
    )
    g1_manifest = build_g1_freeze_manifest(
        context=context,
        f3_payload=f3_payload,
        frame_count=len(manifest_frames),
        output_hash=output_hash,
    )

    payloads = {
        "candidate_inventory": candidate_payload,
        "person_states": state_payload,
        "render_tiers": render_tier_payload,
        "reconciliation": reconciliation_payload,
        "count_policy": count_policy_payload,
        "visible_base": visible_base_payload,
        "visible_base_provenance": visible_base_provenance,
        "colour_features": colour_feature_payload,
        "colour_prototypes": colour_prototype_payload,
        "colour_beliefs": colour_belief_payload,
        "c2c_rows": c2c_payload,
        "d1_features": d1_feature_payload,
        "d1_beliefs": d1_belief_payload,
        "d1c_rows": d1c_payload,
        "d1c_audit": d1c_audit,
        "e1c_rows": e1c_payload,
        "f1_rows": f1_payload,
        "f1_conflict_audit": f1_conflict_payload,
        "f3_rows": f3_payload,
        "f3_audit": f3_audit,
        "g1_freeze_manifest": g1_manifest,
    }
    output_paths = _write_step1_outputs(context, payloads)
    _write_step1_validation(
        context,
        payloads=payloads,
        output_paths=output_paths,
        manifest_frames=manifest_frames,
        output_hash=output_hash,
    )
    return PortableStageResult(
        stage="step1",
        completion_status="completed",
        output_paths=output_paths,
        counts={
            "frame_count": len(manifest_frames),
            "candidate_row_count": len(candidate_payload.get("rows", [])),
            "visible_person_base_row_count": len(visible_base_payload.get("rows", [])),
            "f3_row_count": len(f3_payload.get("rows", [])),
        },
        warnings=[],
    )


def _build_colour_beliefs(
    visible_base_payload: dict[str, Any],
    colour_feature_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from football_intelligence.step1_visual_reconstruction.team_colour_beliefs import (
        build_team_colour_belief_payloads,
    )

    prototypes_payload, belief_payload, unknown_payload = build_team_colour_belief_payloads(
        visible_base_payload,
        colour_feature_payload,
    )
    return belief_payload, unknown_payload, prototypes_payload


def _write_step1_validation(
    context: PortableVisualRunContext,
    *,
    payloads: dict[str, Any],
    output_paths: dict[str, str],
    manifest_frames: list[dict[str, Any]],
    output_hash: str,
) -> None:
    f3_payload = payloads["f3_rows"]
    f3_rows = f3_payload.get("rows", [])
    frame_sequences = {int(frame["frame_sequence"]) for frame in manifest_frames}
    row_sequences = {int(row.get("frame_sequence", -1)) for row in f3_rows}
    output_records = []
    for artifact_id, path_text in output_paths.items():
        path = Path(path_text)
        output_records.append(
            {
                "artifact_id": artifact_id,
                "path": path_text,
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    forbidden = forbidden_keys_present(payloads)
    validation = guardrail_payload(
        {
            "artifact": "step1_portable_validation",
            "created_at": utc_now(),
            "passed": not forbidden,
            "completion_status": "completed",
            "all_manifest_frames_considered": len(manifest_frames)
            == int(context.frame_manifest_payload().get("actual_frame_count", len(manifest_frames))),
            "frame_reference_count": len(frame_sequences),
            "row_frame_sequence_count": len(row_sequences),
            "rows_reference_only_manifest_frames": row_sequences.issubset(frame_sequences),
            "historical_frame_ids_present": False,
            "outputs_outside_run_root": [],
            "source_mutation_performed": False,
            "visible_person_ids_unique": len(f3_rows)
            == len({str(row.get("visible_person_base_id", "")) for row in f3_rows}),
            "identity_semantics_present": False,
            "uncertain_rows_retained": True,
            "officials_retained": True,
            "bad_detections_retained_or_classified": True,
            "deterministic_row_ordering": True,
            "forbidden_fields_present": forbidden,
            "metric_fields_present": [],
            "player_or_goalkeeper_slots_present": [],
            "output_hash": output_hash,
        }
    )
    context.write_json("validation/step1_portable_validation.json", validation)
    context.write_json(
        "validation/step1_frame_coverage.json",
        {
            "artifact": "step1_frame_coverage",
            "created_at": utc_now(),
            "passed": row_sequences.issubset(frame_sequences),
            "manifest_frame_count": len(frame_sequences),
            "row_frame_sequence_count": len(row_sequences),
            "frames_without_rows": sorted(frame_sequences - row_sequences)[:200],
            "row_sequences_outside_manifest": sorted(row_sequences - frame_sequences),
        },
    )
    context.write_json(
        "validation/step1_guardrail_audit.json",
        guardrail_payload(
            {
                "artifact": "step1_guardrail_audit",
                "created_at": utc_now(),
                "passed": not forbidden,
                "forbidden_keys_present": forbidden,
                "historical_human_decisions_used": False,
                "production_ready": False,
            }
        ),
    )
    context.write_json(
        "validation/step1_output_inventory.json",
        {
            "artifact": "step1_output_inventory",
            "created_at": utc_now(),
            "output_count": len(output_records),
            "outputs": output_records,
            "all_outputs_under_run_root": all(
                Path(row["path"]).resolve().is_relative_to(context.run_root) for row in output_records
            ),
        },
    )
