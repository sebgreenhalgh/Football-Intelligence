"""Read-only integrity check for the G7F-F operator-selected provisional detector.

Only named candidate-identity artifacts are opened. Blind-repeat identity files
are outside this check's allowlist and are never enumerated or read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT.parent
PART9 = EXTERNAL / "experiments/football_observation_reasoner/part 9"
G7FD = PART9 / "G7F_D_INTERIM_DENSE_GOLD_BAKEOFF_AND_STOPPING_AUDIT_v1"
AUDIT = G7FD / "runs/N010/20260928T023910741887Z"
REVIEW = AUDIT / "REVIEW_PACK"
G7FB = PART9 / "G7F_B_R1_DETECTION_CANDIDATE_BAKEOFF_v1"
STAGE = PART9 / "G7F_F_OPERATOR_SELECTION_UNBLIND_AND_PROVISIONAL_DETECTOR_FREEZE_v1"
MANIFEST = EXTERNAL / "models/detectors/provisional/detector-provisional-v0.1.0/manifest.json"
GOLD = EXTERNAL / "datasets/gold_corpus"
FORMAL = "CONTINUE_G7F_D_INTERIM_DENSE_GOLD_NEXT_TRANCHE_REQUIRED"
RUNS = {
    "LOCAL_DEFAULT_RERUN": "g7f_b_r1_local_default_rerun",
    "RECALL_ORIENTED_VARIANT": "g7f_b_r1_recall_conf_012",
    "MULTIPLICITY_REDUCTION_VARIANT": "g7f_b_r1_multiplicity_nms_iou_050",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def gold_tree_digest() -> tuple[int, str]:
    """Hash every Gold Corpus path, length and byte hash; make no Gold writes."""
    rows = []
    for path in sorted(GOLD.rglob("*")):
        if path.is_file():
            rows.append(f"{path.relative_to(GOLD).as_posix()}|{path.stat().st_size}|{sha(path)}")
    digest = hashlib.sha256((("\n".join(rows)) + "\n").encode()).hexdigest()
    return len(rows), digest


def check() -> dict:
    status = read(ROOT / "PROJECT_STATUS.json")
    registry = read(ROOT / "configs/detectors/registry.json")
    manifest = read(MANIFEST)
    operator = read(STAGE / "operator_decision.json")
    mapping = read(STAGE / "candidate_identity_mapping.json")

    require(status["current_stage"] in {
        "G7F_F_OPERATOR_SELECTION_UNBLIND_AND_PROVISIONAL_DETECTOR_FREEZE_v1",
        "G7G_A_TEMPORAL_GOLD_CORPUS_AND_SEQUENCE_FOUNDATION_v1",
        "G7G_B_TEMPORAL_GOLD_HUMAN_PILOT_v1",
        "G7G_B_R1_TEMPORAL_REVIEWER_USABILITY_AND_SEQUENTIAL_FLOW_REPAIR_v1",
        "G7G_B_R2_FAST_LAUNCH_GATE_AND_GOLD_VALIDATION_PERFORMANCE_REPAIR_v1",
        "G7G_B_R3_TEMPORAL_REVIEWER_HIGH_MAGNIFICATION_ZOOM_REPAIR_v1",
        "G7G_B_R4_FAST_FRAME_ASSET_LAUNCH_VALIDATION_REPAIR_v1",
    }, "wrong current stage")
    expected_status = "READY_FOR_HUMAN_PILOT" if status["current_stage"].startswith("G7G_B_") else "COMPLETE"
    require(status["stage_status"] == expected_status, "stage status mismatch")
    expected_next = {
        "G7F_F_OPERATOR_SELECTION_UNBLIND_AND_PROVISIONAL_DETECTOR_FREEZE_v1": "G7G_A_TEMPORAL_GOLD_CORPUS_AND_SEQUENCE_FOUNDATION_v1_SEPARATE_REQUEST_REQUIRED",
        "G7G_A_TEMPORAL_GOLD_CORPUS_AND_SEQUENCE_FOUNDATION_v1": "G7G_B_TEMPORAL_GOLD_HUMAN_PILOT_v1_SEPARATE_REQUEST_REQUIRED",
        "G7G_B_TEMPORAL_GOLD_HUMAN_PILOT_v1": "COMPLETE_THE_AUTHORIZED_G7G_B_SINGLE_SEQUENCE_HUMAN_PILOT",
        "G7G_B_R1_TEMPORAL_REVIEWER_USABILITY_AND_SEQUENTIAL_FLOW_REPAIR_v1": "COMPLETE_THE_AUTHORIZED_G7G_B_SINGLE_SEQUENCE_HUMAN_PILOT",
        "G7G_B_R2_FAST_LAUNCH_GATE_AND_GOLD_VALIDATION_PERFORMANCE_REPAIR_v1": "COMPLETE_THE_AUTHORIZED_G7G_B_SINGLE_SEQUENCE_HUMAN_PILOT",
        "G7G_B_R3_TEMPORAL_REVIEWER_HIGH_MAGNIFICATION_ZOOM_REPAIR_v1": "COMPLETE_THE_AUTHORIZED_G7G_B_SINGLE_SEQUENCE_HUMAN_PILOT",
        "G7G_B_R4_FAST_FRAME_ASSET_LAUNCH_VALIDATION_REPAIR_v1": "COMPLETE_THE_AUTHORIZED_G7G_B_SINGLE_SEQUENCE_HUMAN_PILOT",
    }[status["current_stage"]]
    require(status["next_authorized_stage"] == expected_next, "wrong next stage")
    require(status["annotation_campaign"]["status"] == "INTENTIONALLY_PAUSED_BY_OPERATOR", "annotation pause changed")
    require(status["annotation_campaign"]["completed_scored_first_pass_frames"] == 10, "scored frame count changed")
    require(status["annotation_campaign"]["planned_total"] == 48, "planned campaign size changed")
    require(status["gold_corpus"]["current_release"] == "gold-v0.1.0", "Gold release changed")
    require(status["gold_corpus"]["detection_gold_frames"] == 16, "Gold frames changed")
    require(status["gold_corpus"]["detection_gold_people"] == 862, "Gold people changed")
    require(
        sha(GOLD / "corpus_manifest.json") == status["gold_corpus"]["latest_manifest_sha256"] == manifest["gold_manifest_sha256"],
        "Gold corpus manifest changed",
    )
    require(
        sha(GOLD / "releases/gold-v0.1.0/manifest.json")
        == status["gold_corpus"]["release_manifest_sha256"]
        == manifest["gold_release_manifest_sha256"],
        "Gold release manifest changed",
    )
    gold_files, gold_digest = gold_tree_digest()
    require(gold_files == 458 and gold_digest == "6119c20dd378553e019c13595d7c83c4a0620cbc0fe2337c4967854029fa29c3", "Gold bytes changed")

    audit_manifest = read(REVIEW / "12_MANIFEST.json")
    require(sha(REVIEW / "12_MANIFEST.json") == manifest["selection_evidence_sha256"]["g7fd_review_manifest"], "audit manifest changed")
    for entry in audit_manifest["files"]:
        require(sha(REVIEW / entry["path"]) == entry["sha256"], f"G7F-D audit bytes changed: {entry['path']}")
    summary = read(REVIEW / "00_EXECUTIVE_SUMMARY.json")
    decision = read(REVIEW / "07_STOPPING_DECISION.json")
    scores = read(REVIEW / "04_BLINDED_AGGREGATE_METRICS.json")["metrics"]
    require(summary["classification"] == FORMAL and decision["decision"] == "CONTINUE_ANNOTATION", "formal decision changed")
    require(summary["N"] == 10 and summary["candidate_identities_unblinded"] is False, "historical audit altered")
    require([scores[label]["AP_50_95"] for label in ("Candidate A", "Candidate B", "Candidate C")] == [0.429629, 0.428944, 0.448901], "AP values changed")
    require([scores[label]["recall_50_95"] for label in ("Candidate A", "Candidate B", "Candidate C")] == [0.474669, 0.474669, 0.500567], "recall values changed")
    require([scores[label]["FP_per_frame_iou50"] for label in ("Candidate A", "Candidate B", "Candidate C")] == [5.7, 6.0, 9.4], "FP values changed")
    conditions = decision["conditions"]
    require(conditions["1_material_AP_separation"] == {"actual": 0.019272, "minimum": 0.02, "passed": False}, "AP separation changed")
    require(conditions["2_bootstrap_stability"]["actual"] == 0.9999 and conditions["2_bootstrap_stability"]["passed"], "bootstrap changed")
    require(conditions["3_jackknife_stability"]["actual"] == 10 and conditions["3_jackknife_stability"]["passed"], "jackknife changed")
    require(conditions["4_no_material_recall_regression"]["actual"] == 0.025898 and conditions["4_no_material_recall_regression"]["passed"], "recall condition changed")
    require(conditions["5_no_serious_pathology"]["FP_burden_bad"] and not conditions["5_no_serious_pathology"]["passed"], "pathology condition changed")
    require(conditions["6_match_frame_robustness"]["passed"] and conditions["6_match_frame_robustness"]["maximum_removal_influence_fraction"] == 0.368098796180988, "frame robustness changed")
    corpus = read(REVIEW / "02_FINALIZED_SCORED_CORPUS_BINDING.json")
    require(corpus["finalized_scored_count"] == 10, "scored corpus changed")
    require(sum(sum(item["person_counts"].values()) for item in corpus["corpus"]) == 529, "evaluable people changed")

    require(sha(STAGE / "operator_decision.json") == mapping["unblinded_after_operator_decision_sha256"] == manifest["selection_evidence_sha256"]["g7ff_operator_decision"], "operator decision binding failed")
    require(operator["formal_statistical_decision"] == FORMAL, "operator artifact changed formal decision")
    require(operator["operator_decision"] == "SELECT_CANDIDATE_C_AND_PAUSE_DENSE_ANNOTATION", "operator decision changed")
    require(operator["operator_override_of_stopping_rule"] is True and operator["candidate_identity_known_at_decision_freeze"] is False, "operator decision not independently frozen")
    require(operator["formal_stopping_rule_passed"] is False, "statistical pass falsely claimed")
    require(sha(STAGE / "candidate_identity_mapping.json") == manifest["selection_evidence_sha256"]["g7ff_candidate_mapping"], "mapping artifact hash mismatch")
    public = read(REVIEW / "03_FROZEN_CANDIDATE_BINDING.json")
    source_map_path = G7FD / "SEALED/DO_NOT_OPEN_CANDIDATE_IDENTITY_MAPPING.json"
    exact_path = AUDIT / "SEALED/exact_candidate_bindings.json"
    require(sha(source_map_path) == public["identity_mapping_sealed_commitment"] == mapping["source_sealed_mapping_sha256"], "sealed mapping commitment mismatch")
    require(sha(exact_path) == mapping["source_exact_candidate_bindings_sha256"], "exact mapping hash mismatch")
    source_map = read(source_map_path)["mapping"]
    exact = read(exact_path)
    require(set(source_map) == {"Candidate A", "Candidate B", "Candidate C"}, "candidate labels changed")
    require(len(set(source_map.values())) == 3 and set(source_map.values()) == set(RUNS), "mapping is not a bijection to frozen shortlist")
    plan = read(G7FB / "01_EXPERIMENT_PLAN/PREDECLARED_EXPERIMENT_PLAN.json")
    require(sha(G7FB / "01_EXPERIMENT_PLAN/PREDECLARED_EXPERIMENT_PLAN.json") == manifest["frozen_configuration"]["predeclared_plan_sha256"], "frozen plan changed")
    for label, role in source_map.items():
        candidate = mapping["candidates"][label]
        original = exact[label]
        require(role == original["role"] == candidate["role"], f"role mismatch: {label}")
        require(candidate["candidate_run_id"] == RUNS[role], f"run ID mismatch: {label}")
        require(candidate["candidate_run_sha256"] == original["sha256"] == sha(Path(original["path"])), f"run output mismatch: {label}")
        require(candidate["frozen_config_sha256"] == original["config_sha256"], f"config binding mismatch: {label}")
        require(original["candidate_run_v2_coverage"]["declared_instance_count"] == 1080 and original["candidate_run_v2_coverage"]["missing_instance_count"] == 0, f"coverage mismatch: {label}")
    require(mapping["candidates"]["Candidate C"]["candidate_name"] == "recall", "Candidate C not recall")
    selected = mapping["candidates"]["Candidate C"]
    run = read(Path(exact["Candidate C"]["path"]))
    require(run["run_id"] == selected["candidate_run_id"] == manifest["selection"]["candidate_run_id"], "selected run mismatch")
    require(run["system_id"] == "RECALL_ORIENTED_VARIANT" and run["weight_sha256"] == manifest["model"]["checkpoint_sha256"], "selected model mismatch")
    require(run["run_metadata"]["plan_sha256"] == manifest["frozen_configuration"]["predeclared_plan_sha256"], "run plan mismatch")
    require(run["code_commit"] == manifest["frozen_configuration"]["source_repository_commit"], "run source commit mismatch")
    require(len(run["processed_frame_instances"]) == 1080 and len({x["source_frame_sha256"] for x in run["processed_frame_instances"]}) == 1044, "run coverage incomplete")
    require(len(run["candidates"]) == 55419, "selected candidate rows changed")
    config_bytes = (json.dumps({k: v for k, v in run.items() if k not in {"candidates", "processed_frame_instances"}}, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
    require(hashlib.sha256(config_bytes).hexdigest() == selected["frozen_config_sha256"] == manifest["frozen_configuration"]["candidate_run_config_sha256"], "frozen config hash mismatch")
    role_plan = next(x for x in plan["candidate_roles"] if x["role"] == "RECALL_ORIENTED_VARIANT")
    require(role_plan["run_id"] == run["run_id"] and role_plan["confidence"] == 0.12 and role_plan["detector_nms_iou"] == 0.7, "threshold/NMS changed")
    require(plan["inference_contract"]["views"] == manifest["frozen_configuration"]["view_plan"], "view/tiling changed")
    require(plan["inference_contract"]["consolidation"] == manifest["frozen_configuration"]["proposal_consolidation"], "consolidation changed")
    checkpoint = Path(manifest["model"]["checkpoint_path"])
    require(checkpoint.stat().st_size == manifest["model"]["checkpoint_bytes"] == registry["model"]["bytes"], "checkpoint bytes changed")
    require(sha(checkpoint) == manifest["model"]["checkpoint_sha256"] == registry["model"]["sha256"], "checkpoint SHA changed")
    require(sha(Path(registry["model_manifest_path"])) == manifest["model"]["checkpoint_object_manifest_sha256"], "checkpoint object manifest changed")
    require(sha(Path(registry["model"]["provenance_file"])) == manifest["model"]["model_provenance_sha256"], "model provenance changed")
    require(sha(G7FB / "10_REVIEW_PACK/CHATGPT_HANDOFF/03_RUN_PROVENANCE_AND_V2_COVERAGE.json") == manifest["frozen_output"]["g7fb_run_provenance_sha256"], "G7F-B provenance changed")
    require(sha(G7FB / "10_REVIEW_PACK/CHATGPT_HANDOFF/12_MANIFEST.json") == manifest["frozen_output"]["g7fb_review_manifest_sha256"], "G7F-B review manifest changed")
    require(sha(Path(manifest["frozen_output"]["candidate_run_path"])) == manifest["frozen_output"]["candidate_run_sha256"], "selected output hash changed")

    require(registry["candidate_unblinded"] is True and registry["anonymous_label_mapping_disclosed"] is True, "registry still blinded")
    require(registry["provisional_candidate"] == selected["candidate_run_id"] == status["detection"]["provisional_candidate"], "provisional candidate metadata disagrees")
    require(registry["status"] == status["detection"]["status"] == manifest["status"] == "OPERATOR_SELECTED_PROVISIONAL", "status mismatch")
    require(registry["provisional_manifest_sha256"] == sha(MANIFEST), "provisional manifest registry hash mismatch")
    require(registry["provisional_manifest_path"] == str(MANIFEST), "provisional manifest registry path mismatch")
    require(registry["stopping_audit_decision"] == status["detection"]["formal_stopping_decision"] == FORMAL, "formal decision metadata mismatch")
    require(len(registry["configurations"]) == 3 and {x["configuration_id"] for x in registry["configurations"]} == set(RUNS.values()), "shortlist provenance lost")
    require(next(x for x in registry["configurations"] if x["configuration_id"] == selected["candidate_run_id"])["status"] == "OPERATOR_SELECTED_PROVISIONAL", "selected config not provisional")
    require(registry["sealed_validation"] is False and status["detection"]["sealed_validation"] is False, "sealed validation falsely claimed")
    require(all(x["production_ready"] is False for x in (status, registry, manifest)), "production promotion falsely claimed")
    require(all(x["final_promotion"] is False for x in (status["detection"], registry, manifest)), "final promotion falsely claimed")
    require(manifest["formal_stopping_rule_passed"] is False and manifest["promotion_eligible"] is False, "promotion eligibility falsely claimed")
    require(mapping["blind_repeat_image_identities_accessed"] is False and manifest["blind_repeat_image_identities_accessed"] is False, "blind-repeat access claimed")
    return {"valid": True, "gold_files": gold_files, "gold_tree_sha256": gold_digest, "candidate_c": selected["candidate_run_id"], "manifest_sha256": sha(MANIFEST), "blind_repeat_identities_accessed": False}


if __name__ == "__main__":
    print(json.dumps(check(), sort_keys=True))
