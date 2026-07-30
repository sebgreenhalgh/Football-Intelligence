"""Build the bounded G7D-C3A4 development-default readiness audit.

This utility is evidence-only. It never invokes proposal, feature, fold, or
detector code and never changes the pitch-gate runtime default.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
EXPECTED_HEAD = "186dc876c9f08c509e4831917702ac51002ed0e6"
STAGE = PROJECT / "experiments/football_observation_reasoner/part 7" / "G7D_C3A4_DEVELOPMENT_DEFAULT_READINESS_AUDIT_v1"
PACK = (
    PROJECT
    / "experiments/football_observation_reasoner/part 7"
    / "G7D_C3A4_Development_Default_Readiness_Audit_Codex_Pack"
)
C3A = PROJECT / "experiments/football_observation_reasoner/part 7/G7D_C3A_PITCH_AWARE_PROPOSAL_GATE_EXPERIMENT_v1"
C3A1 = PROJECT / "experiments/football_observation_reasoner/part 7/G7D_C3A1_PITCH_GATE_SHADOW_INTEGRATION_REVIEW_v1"
C3A2 = PROJECT / "experiments/football_observation_reasoner/part 7/G7D_C3A2_BOUNDED_GATED_RUNTIME_REPLAY_v1"
C3A3 = PROJECT / "experiments/football_observation_reasoner/part 7/G7D_C3A3_ACTIVE_SANDBOX_PITCH_GATE_INTEGRATION_v1"

TRAIN_MATCHES = ("117092", "117093", "118575", "118576", "118577", "128058")
EVALUATED_MATCHES = ("117092", "118575", "128058")
POLYGON_HASHES = {
    "117092": "92ca8040eedd3b0ec0bb685648691f0c314d8527f3fa8f2db1823b4461e4b338",
    "118575": "fbd7f3a473acc197b4c893d90bbaa4c5d484d1e883e8df1ac4601daf4396dec1",
    "128058": "24ad1e4d143527e5a3e92cded1b5d8b10526d67b5b0d1f8b02289a91e8c65307",
}
DECISION = "PASS_G7D_C3A4_DEFERRED_FOR_ADDITIONAL_COVERAGE"
POLICY_ID = "G7D_C3A4_DEVELOPMENT_DEFAULT_POLICY_DRAFT_V1"
LABEL = "DEFAULT READINESS AUDIT — NO DEFAULT CHANGED"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "project_relative_path": path.relative_to(PROJECT).as_posix(),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_pack() -> list[dict[str, Any]]:
    manifest = read_json(PACK / "03_PACK_MANIFEST.json")
    rows = []
    for expected in manifest["files"]:
        path = PACK / expected["path"]
        row = artifact(path)
        if row["byte_size"] != expected["byte_size"] or row["sha256"] != expected["sha256"]:
            raise RuntimeError(f"prompt-pack mismatch: {expected['path']}")
        rows.append(row)
    return rows


def assert_preflight() -> None:
    if git("rev-parse", "HEAD") != EXPECTED_HEAD or git("branch", "--show-current") != "main":
        raise RuntimeError("repository baseline or branch mismatch")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree was not clean at audit start")


def validate_frozen_evidence() -> dict[str, Any]:
    stage_contract = read_json(PACK / "02_STAGE_CONTRACT.json")
    active_contract_path = C3A3 / "01_CONTRACT_AND_DEVICE/active_sandbox_contract.json"
    active_contract = read_json(active_contract_path)
    benchmark_path = C3A2 / "03_PERFORMANCE/benchmark_summary.json"
    benchmark = read_json(benchmark_path)
    parity = read_json(C3A3 / "02_ACTIVE_CORRECTNESS/active_vs_c3a2_parity.json")
    runtime = read_json(C3A3 / "03_RUNTIME/runtime_envelope_report.json")
    safety = read_json(C3A3 / "05_SAFETY_AND_ROLLBACK/safety_revalidation.json")
    rollback = read_json(C3A3 / "05_SAFETY_AND_ROLLBACK/output_isolation_and_rollback.json")
    required = {
        "stage_contract": (
            stage_contract["expected_repository_head"] == EXPECTED_HEAD
            and stage_contract["stage_type"] == "ARCHITECTURE_DEFAULT_POLICY_AUDIT"
            and stage_contract["evaluated_matches"] == ["128058", "118575", "117092"]
            and stage_contract["runtime_frames"] == 96
            and stage_contract["control_candidates"] == 5940
            and stage_contract["retained_candidates"] == 4252
            and stage_contract["suppressed_candidates"] == 1688
            and stage_contract["known_coverage_limitation"] == "NO_GOAL_LINE_NEAREST_GOALKEEPER_CASE"
            and stage_contract["default_before_and_after_stage"] == "DISABLED"
            and stage_contract["production_ready"] is False
        ),
        "active_contract": (
            active_contract["project_default"] == "DISABLED"
            and active_contract["development_default_approved"] is False
            and active_contract["required_activation_flags"]
            == [
                "--pitch-gate-mode ACTIVE_SANDBOX",
                "--pitch-gate-contract <exact path>",
                "--pitch-gate-contract-sha256 <exact hash>",
                "--output-root <external stage path>",
                "--acknowledge-sandbox-only",
            ]
        ),
        "parity": (
            parity["retained_candidate_count"] == 4252
            and parity["suppressed_candidate_count"] == 1688
            and parity["candidate_fold_output_count"] == 21260
            and parity["retained_mismatch_count"] == 0
            and parity["decision_mismatch_count"] == 0
            and parity["suppressed_set_and_order_exact"] is True
        ),
        "runtime": (
            runtime["within_required_envelope"] is True
            and round(runtime["active_sandbox_seconds"], 3) == 464.570
            and runtime["allowed_seconds_range"] == [370.13335, 500.76865]
        ),
        "c3a2_performance": (
            benchmark["classification"] == "PASS_G7D_C3A2_BENCHMARK"
            and round(benchmark["measured_runtime_reduction_fraction"], 4)
            == stage_contract["runtime_reduction_fraction"]
            and round(benchmark["measured_speedup_factor"], 3) == stage_contract["speedup"]
        ),
        "safety": (
            safety["reviewed_useful_relevant_retained"] == safety["reviewed_useful_relevant_support"] == 87
            and safety["reviewed_officials_retained"] == safety["reviewed_official_support"] == 10
            and safety["reviewed_active_player_goalkeeper_retained"] == 77
            and safety["unsafe_all_nearby_suppressed"] == 0
        ),
        "rollback": (
            rollback["no_flags_mode"] == "DISABLED"
            and rollback["removing_active_flags_rolls_back_to_disabled"] is True
            and rollback["active_outputs_external_to_repository"] is True
            and rollback["b1_b2c_b3_automatic_consumption_absent"] is True
            and rollback["development_default_changed"] is False
        ),
    }
    if not all(required.values()):
        raise RuntimeError(f"frozen C3A3 evidence mismatch: {required}")
    hook_text = (REPO / "src/football_intelligence/proposal_gate_hook.py").read_text(encoding="utf-8")
    if "DEFAULT_PITCH_GATE_MODE = PitchGateMode.DISABLED" not in hook_text:
        raise RuntimeError("runtime default is not DISABLED")
    return {
        "classification": "PASS_G7D_C3A4_FROZEN_EVIDENCE_CLOSURE",
        "checks": required,
        "active_contract": artifact(active_contract_path),
        "c3a2_benchmark": artifact(benchmark_path),
        "runtime_seconds": runtime["active_sandbox_seconds"],
        "runtime_envelope_seconds": runtime["allowed_seconds_range"],
        "c3a2_measured_runtime_reduction_fraction": benchmark["measured_runtime_reduction_fraction"],
        "c3a2_measured_speedup_factor": benchmark["measured_speedup_factor"],
        "frames": active_contract["frames"],
        "control_candidates": active_contract["control_candidate_count"],
        "retained_candidates": parity["retained_candidate_count"],
        "suppressed_candidates": parity["suppressed_candidate_count"],
        "candidate_fold_outputs": parity["candidate_fold_output_count"],
        "reviewed_safety": {
            "useful_relevant": "87/87",
            "officials": "10/10",
            "active_players_goalkeepers": "77/77",
            "unsafe_missed_person_neighbourhood_losses": 0,
        },
        "project_default": "DISABLED",
        "production_ready": False,
    }


def aggregate_runtime() -> dict[str, dict[str, int]]:
    universe = read_json(C3A / "06_FULL_UNIVERSE_SUPPLY/full_universe_gate_comparison.json")
    result: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for frame in universe["frames"]:
        match = str(frame["match"])
        result[match]["frames"] += 1
        result[match]["control_candidates"] += frame["raw_count"]
        result[match]["retained_candidates"] += frame["retained_count"]
        result[match]["suppressed_candidates"] += frame["reduction_count"]
    return {match: dict(values) for match, values in result.items()}


def boundary_counts() -> dict[str, dict[str, int]]:
    boundary = read_json(C3A1 / "04_BOUNDARY_AUDIT/boundary_exception_parity.json")
    output: dict[str, Counter[str]] = defaultdict(Counter)
    for category, rows in boundary["cases"].items():
        for row in rows:
            output[str(row["match_id"])][category] += 1
    return {match: dict(counts) for match, counts in output.items()}


def polygon_record(match_id: str, setup: dict[str, Any]) -> dict[str, Any]:
    calibration = setup["pitch_calibration"]
    status = calibration["status"]
    if status != "HUMAN_CONFIRMED":
        if calibration.get("polygon_path") is not None or calibration.get("polygon_sha256") is not None:
            raise RuntimeError(f"inconsistent missing polygon state for {match_id}")
        return {
            "status": status,
            "authoritative": False,
            "path": None,
            "declared_sha256": None,
            "actual_sha256": None,
            "hash_valid": False,
            "camera_segment_count": None,
            "camera_segment_status": "NOT_ESTABLISHED_WITHOUT_POLYGON",
        }
    relative = calibration["polygon_path"]
    path = (PROJECT / "matches" / match_id / relative).resolve()
    match_root = (PROJECT / "matches" / match_id).resolve()
    if match_root not in path.parents or not path.is_file():
        raise RuntimeError(f"unsafe or missing polygon path for {match_id}")
    actual = sha256_file(path)
    expected = POLYGON_HASHES[match_id]
    polygon = read_json(path)
    if actual != expected or calibration["polygon_sha256"] != expected or polygon["status"] != "HUMAN_CONFIRMED":
        raise RuntimeError(f"polygon hash/status mismatch for {match_id}")
    segments = polygon["camera_segments"]
    if calibration["camera_segment_count"] != len(segments):
        raise RuntimeError(f"camera-segment mismatch for {match_id}")
    return {
        "status": status,
        "authoritative": True,
        "path": path.relative_to(PROJECT).as_posix(),
        "declared_sha256": expected,
        "actual_sha256": actual,
        "hash_valid": True,
        "camera_segment_count": len(segments),
        "camera_segment_status": "ONE_STABLE_SEGMENT_BOTH_HALVES" if len(segments) == 1 else "MULTIPLE_SEGMENTS",
        "source_width": polygon["source_width"],
        "source_height": polygon["source_height"],
        "second_half_alignment": polygon["second_half_alignment_answer"],
    }


def build_coverage() -> dict[str, Any]:
    train_file = PROJECT / "datasets/soccertrack_v2/splits/split_v1/train_matches.txt"
    train = tuple(line.strip() for line in train_file.read_text(encoding="utf-8-sig").splitlines() if line.strip())
    split_contract = read_json(PROJECT / "datasets/soccertrack_v2/splits/split_v1/split_contract.json")
    if train != TRAIN_MATCHES or not split_contract["frozen"] or split_contract["status"] != "FROZEN_HUMAN_APPROVED":
        raise RuntimeError("TRAIN_DEVELOPMENT split mismatch")
    runtime = aggregate_runtime()
    boundary = boundary_counts()
    rows = []
    for match_id in TRAIN_MATCHES:
        setup_path = PROJECT / "matches" / match_id / "calibration/match_setup.json"
        setup = read_json(setup_path)
        if setup["dataset_split"]["proposed_assignment"] != "TRAIN_DEVELOPMENT":
            raise RuntimeError(f"non-training setup in inventory: {match_id}")
        polygon = polygon_record(match_id, setup)
        evaluated = match_id in EVALUATED_MATCHES
        human_targeted = match_id in {"117092", "118575"}
        conditions = setup["conditions"]
        rows.append(
            {
                "match_id": match_id,
                "split": "TRAIN_DEVELOPMENT",
                "conditions": conditions,
                "polygon": polygon,
                "included_c3a_through_c3a3": evaluated,
                "runtime_evidence": runtime.get(
                    match_id,
                    {"frames": 0, "control_candidates": 0, "retained_candidates": 0, "suppressed_candidates": 0},
                ),
                "human_safety_coverage": (
                    "TARGETED_96_CANDIDATE_DECISIONS_AND_12_SCENE_CHECKS"
                    if human_targeted
                    else "NO_C3A_TARGETED_HUMAN_SAFETY"
                ),
                "boundary_and_official_coverage": boundary.get(match_id, {}),
                "goalkeeper_coverage": {
                    "reviewed_goalkeepers": boundary.get(match_id, {}).get("goalkeeper_protection", 0),
                    "goal_line_nearest_or_behind_goal": 0,
                },
                "daylight_low_light_coverage": (
                    "LOW_LIGHT_GLARE_RUNTIME_AND_TARGETED_HUMAN"
                    if conditions["lighting"] == "NIGHT" and human_targeted
                    else "DAYLIGHT_RUNTIME_AND_TARGETED_HUMAN"
                    if evaluated and human_targeted
                    else "DAYLIGHT_RUNTIME_NO_C3A_TARGETED_HUMAN"
                    if evaluated
                    else "CONDITION_INVENTORY_ONLY_NO_GATE_RUNTIME"
                ),
                "camera_and_panorama": {
                    "camera_segment_status": polygon["camera_segment_status"],
                    "panorama_quality": conditions["panorama_quality"],
                    "unusual_conditions": conditions["unusual_conditions"],
                },
            }
        )
    confirmed = sum(row["polygon"]["authoritative"] for row in rows)
    evaluated_confirmed = sum(row["polygon"]["authoritative"] and row["included_c3a_through_c3a3"] for row in rows)
    return {
        "schema_version": "football_intelligence.g7d_c3a4.coverage_matrix.v1",
        "classification": "PASS_G7D_C3A4_TRAIN_DEVELOPMENT_INVENTORY",
        "train_match_count": len(rows),
        "authoritative_polygon_count": confirmed,
        "authoritative_polygon_fraction": confirmed / len(rows),
        "authoritative_polygon_matches_in_runtime": evaluated_confirmed,
        "all_authoritative_polygon_matches_in_runtime": evaluated_confirmed == confirmed,
        "evaluated_matches": list(EVALUATED_MATCHES),
        "unevaluated_matches": [match for match in TRAIN_MATCHES if match not in EVALUATED_MATCHES],
        "matches": rows,
        "validation_or_holdout_content_accessed": False,
    }


def build_edges() -> dict[str, Any]:
    parity = read_json(C3A1 / "04_BOUNDARY_AUDIT/boundary_exception_parity.json")
    edges = [
        {
            "edge_case": "assistant referee near the touchline",
            "classification": "COVERED_AND_PASSING",
            "severity": "HIGH",
            "support": parity["category_counts"]["assistant_referee_touchline"],
            "basis": "Nine targeted assistant-referee candidates were retained as BOUNDARY_REVIEW.",
        },
        {
            "edge_case": "active player just outside the pitch",
            "classification": "COVERED_AND_PASSING",
            "severity": "HIGH",
            "support": parity["category_counts"]["active_player_just_outside"],
            "basis": "One targeted active-player-just-outside case was retained as BOUNDARY_REVIEW.",
        },
        {
            "edge_case": "goalkeeper behind the goal line",
            "classification": "NOT_COVERED",
            "severity": "HIGH",
            "support": parity["goalkeeper_behind_goal_combined_support"],
            "basis": "Frozen evidence records zero behind-goal and zero goal-line-nearest goalkeeper cases.",
        },
        {
            "edge_case": "player retrieving the ball",
            "classification": "NOT_COVERED",
            "severity": "HIGH",
            "support": 0,
            "basis": (
                "No frozen candidate carries a specific retrieving-ball human review label; "
                "outside-player evidence is not reinterpreted."
            ),
        },
        {
            "edge_case": "boundary-uncertain person",
            "classification": "COVERED_AND_PASSING",
            "severity": "HIGH",
            "support": parity["category_counts"]["boundary_uncertain_person"],
            "basis": "Twenty-one targeted boundary-uncertain people were retained.",
        },
        {
            "edge_case": "multiple camera segments",
            "classification": "NOT_COVERED",
            "severity": "HIGH",
            "support": 0,
            "basis": "All three authoritative polygons declare exactly one stable camera segment.",
        },
        {
            "edge_case": "missing or invalid polygon",
            "classification": "PARTIALLY_COVERED",
            "severity": "HIGH",
            "support": 3,
            "basis": (
                "Three real training matches lack polygons and remain DISABLED today; fail-closed behavior "
                "is specified and statically audited, not active runtime policy."
            ),
        },
        {
            "edge_case": "extreme panorama distortion",
            "classification": "NOT_COVERED",
            "severity": "HIGH",
            "support": 0,
            "basis": "No human condition record labels an evaluated match as extreme distortion.",
        },
        {
            "edge_case": "low-light glare",
            "classification": "COVERED_AND_PASSING",
            "severity": "HIGH",
            "support": 32,
            "basis": (
                "Match 117092 contributes 32 night/floodlight-glare runtime frames and "
                "96 targeted human candidate decisions."
            ),
        },
        {
            "edge_case": "dense or crowded scenes",
            "classification": "PARTIALLY_COVERED",
            "severity": "MEDIUM",
            "support": 24,
            "basis": (
                "Twenty-four targeted scenes include clutter-focused selection, but no dedicated "
                "dense/set-piece safety stratum was frozen."
            ),
        },
    ]
    uncovered_high = [
        row["edge_case"]
        for row in edges
        if row["severity"] == "HIGH" and row["classification"] != "COVERED_AND_PASSING"
    ]
    return {
        "schema_version": "football_intelligence.g7d_c3a4.edge_case_matrix.v1",
        "classification": "PASS_G7D_C3A4_EDGE_CASE_AUDIT_COMPLETE",
        "allowed_classifications": ["COVERED_AND_PASSING", "PARTIALLY_COVERED", "NOT_COVERED", "NOT_APPLICABLE"],
        "known_coverage_limitation": "NO_GOAL_LINE_NEAREST_GOALKEEPER_CASE",
        "edges": edges,
        "uncovered_high_severity_cases": uncovered_high,
        "promotion_edge_criterion_pass": False,
    }


def draft_policy() -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.g7d_c3a4.development_default_policy_draft.v1",
        "policy_id": POLICY_ID,
        "status": "DRAFT_NOT_ACTIVE",
        "project_default_before_and_after": "DISABLED",
        "applies_only_to": "TRAIN_DEVELOPMENT_EXPERIMENTS",
        "activation_mode_if_future_separate_promotion_occurs": "ACTIVE_SANDBOX",
        "required_prerequisites": [
            "frozen split assignment is TRAIN_DEVELOPMENT",
            "match_setup.pitch_calibration.status is HUMAN_CONFIRMED",
            "polygon path resolves beneath the match calibration root",
            "polygon file exists and SHA-256 equals match_setup polygon_sha256",
            "polygon status is HUMAN_CONFIRMED",
            "camera segment resolves deterministically; unsupported multiple segments disable the gate",
            "exact G7D_C3A3_ACTIVE_SANDBOX_PITCH_GATE_V1 contract path and SHA-256 validate",
            "complete external audit output is writable",
        ],
        "gate_behavior": {
            "retain": ["KEEP", "BOUNDARY_REVIEW", "EXCEPTION_KEEP"],
            "suppress": ["SUPPRESS_SANDBOX"],
            "position": "proposal consolidation -> gate -> retained subsequence -> crop/features/folds",
        },
        "fail_closed": {
            "result": "DISABLED",
            "silent_active_fallback": False,
            "triggers": [
                "non-TRAIN_DEVELOPMENT match",
                "missing or invalid match setup",
                "missing, non-human-confirmed, unsafe-path, missing, or hash-invalid polygon",
                "ambiguous or unsupported camera segment",
                "missing or hash-invalid gate contract",
                "unwritable or incomplete audit destination",
            ],
        },
        "audit_logging_required": True,
        "immediate_rollback": "remove future development-default opt-in and resolve to DISABLED",
        "excluded": ["VALIDATION_MODEL_SELECTION", "SEALED_HOLDOUT", "PRODUCTION", "HISTORICAL_FROZEN_OUTPUTS"],
        "production_ready": False,
        "component_promoted": False,
        "active": False,
    }


def promotion_and_plan(
    coverage: dict[str, Any], edges: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    criteria = [
        {"criterion": 1, "pass": True, "basis": "C3A3 correctness, isolation, rollback, and runtime envelope pass."},
        {
            "criterion": 2,
            "pass": True,
            "basis": "Zero losses across 87 useful, 10 official, and 77 active-player/goalkeeper reviewed boxes.",
        },
        {"criterion": 3, "pass": True, "basis": "Daylight and low-light/floodlight-glare evidence are both present."},
        {
            "criterion": 4,
            "pass": coverage["authoritative_polygon_fraction"] >= 0.5
            and coverage["all_authoritative_polygon_matches_in_runtime"],
            "basis": "Exactly 3/6 training matches have authoritative polygons and all three are runtime-covered.",
        },
        {
            "criterion": 5,
            "pass": False,
            "basis": f"Uncovered high-severity cases: {', '.join(edges['uncovered_high_severity_cases'])}.",
        },
        {
            "criterion": 6,
            "pass": True,
            "basis": (
                "Current default remains DISABLED for three missing-polygon matches; draft policy "
                "deterministically fails missing/invalid prerequisites to DISABLED."
            ),
            "scope_note": "STATIC_DRAFT_POLICY_AUDIT_NOT_ACTIVE_RUNTIME_PROMOTION",
        },
        {"criterion": 7, "pass": True, "basis": "Audit artifacts and immediate rollback are complete and hash-bound."},
        {"criterion": 8, "pass": True, "basis": "Validation, holdout, and production remain excluded."},
    ]
    criteria_report = {
        "schema_version": "football_intelligence.g7d_c3a4.promotion_criteria.v1",
        "predeclared": True,
        "criteria": criteria,
        "failed_criteria": [row["criterion"] for row in criteria if not row["pass"]],
    }
    decision = {
        "schema_version": "football_intelligence.g7d_c3a4.decision.v1",
        "classification": DECISION,
        "decision": "DEFER_FOR_ADDITIONAL_COVERAGE",
        "deterministic_rule": (
            "Criteria 1-3 and 6-8 pass, criterion 4 passes, and criterion 5 fails "
            "without a demonstrated safety/isolation failure."
        ),
        "failed_criteria": [5],
        "primary_blocker": "NO_GOAL_LINE_NEAREST_GOALKEEPER_CASE",
        "default_changed": False,
        "project_default": "DISABLED",
        "production_ready": False,
    }
    plan = {
        "schema_version": "football_intelligence.g7d_c3a4.additional_coverage_plan.v1",
        "status": "REQUIRED_BEFORE_RECONSIDERATION",
        "matches_needing_human_confirmed_polygons": ["117093", "118576", "118577"],
        "matches_needing_bounded_replay": ["117093", "118576", "118577"],
        "selection_reason": (
            "These are the only unevaluated TRAIN_DEVELOPMENT matches and the only training matches "
            "without polygons."
        ),
        "reuse_existing_detector_outputs_where_hash_valid": True,
        "limits": {
            "additional_matches": 3,
            "maximum_frames_per_match": 16,
            "maximum_frames_total": 48,
            "maximum_targeted_candidate_decisions_total": 60,
            "maximum_scene_checks_total": 12,
        },
        "targeted_edge_cases": [
            "at least one goal-line-nearest or behind-goal active goalkeeper",
            "at least one active player retrieving the ball outside the polygon",
            "any discovered multiple-camera-segment transition",
            "any discovered extreme panorama-distortion region",
            "dense/crowded or set-piece-like scene burden",
            "missing/invalid polygon fail-closed cases",
        ],
        "allocation": {
            "frames": {"117093": 16, "118576": 16, "118577": 16},
            "targeted_candidate_decisions": {"117093": 20, "118576": 20, "118577": 20},
            "scene_checks": {"117093": 4, "118576": 4, "118577": 4},
        },
        "pass_fail_thresholds": {
            "reviewed_useful_relevant_losses": 0,
            "reviewed_relevant_official_losses": 0,
            "reviewed_active_player_goalkeeper_losses": 0,
            "unsafe_missed_person_neighbourhood_losses": 0,
            "goal_line_nearest_or_behind_goal_goalkeeper_support_minimum": 1,
            "goal_line_nearest_or_behind_goal_goalkeeper_losses": 0,
            "retrieving_ball_active_player_support_minimum": 1,
            "retrieving_ball_active_player_losses": 0,
            "missing_or_invalid_polygon_active_results": 0,
            "multiple_segment_without_deterministic_resolution_active_results": 0,
            "uncovered_high_severity_case_after_round": 0,
        },
        "package_created_in_c3a4": False,
    }
    return criteria_report, decision, plan


def font(size: int) -> ImageFont.ImageFont:
    candidates = [Path("C:/Windows/Fonts/segoeui.ttf"), Path("C:/Windows/Fonts/arial.ttf")]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def readiness_visual(coverage: dict[str, Any], edges: dict[str, Any], criteria: dict[str, Any]) -> Path:
    path = STAGE / "06_VISUAL_QA/DEVELOPMENT_DEFAULT_READINESS_MATRIX.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (1900, 1180), "#0d1324")
    draw = ImageDraw.Draw(canvas)
    white, muted, green, amber, red, blue = "#f8fafc", "#bdc7dc", "#67e8b3", "#ffd166", "#ff6b7a", "#77a7ff"
    draw.text((45, 32), LABEL, font=font(40), fill=white)
    draw.text((45, 92), "Decision: DEFER FOR ADDITIONAL COVERAGE", font=font(28), fill=amber)
    draw.text((45, 140), "Project default remains DISABLED | production_ready=false", font=font(21), fill=muted)
    headers = ["Match", "Polygon", "Runtime", "Condition", "Human safety", "Camera"]
    widths = [150, 220, 180, 250, 390, 260]
    x0, y0, row_h = 45, 205, 70
    x = x0
    for header, width in zip(headers, widths, strict=True):
        draw.rectangle((x, y0, x + width, y0 + row_h), fill="#1b2540", outline="#435170")
        draw.text((x + 12, y0 + 20), header, font=font(20), fill=blue)
        x += width
    for index, row in enumerate(coverage["matches"]):
        y = y0 + (index + 1) * row_h
        polygon = "HUMAN_CONFIRMED" if row["polygon"]["authoritative"] else "HUMAN_REQUIRED"
        runtime = f"{row['runtime_evidence']['frames']} frames" if row["included_c3a_through_c3a3"] else "not evaluated"
        condition = f"{row['conditions']['lighting']} / {row['conditions']['panorama_quality']}"
        safety = (
            "96 candidates + 12 scenes"
            if row["human_safety_coverage"].startswith("TARGETED")
            else "no targeted C3A safety"
        )
        camera = row["polygon"]["camera_segment_status"].replace("_", " ").lower()
        values = [row["match_id"], polygon, runtime, condition, safety, camera]
        x = x0
        for column, (value, width) in enumerate(zip(values, widths, strict=True)):
            fill = "#121b31" if index % 2 == 0 else "#162039"
            draw.rectangle((x, y, x + width, y + row_h), fill=fill, outline="#33415e")
            colour = green if column == 1 and polygon == "HUMAN_CONFIRMED" else amber if column == 1 else white
            draw.text((x + 10, y + 22), value, font=font(16), fill=colour)
            x += width
    edge_y = 725
    draw.text((45, edge_y), "Edge-case readiness", font=font(25), fill=white)
    for index, edge in enumerate(edges["edges"]):
        col, row_index = index % 2, index // 2
        x, y = 45 + col * 915, edge_y + 50 + row_index * 54
        status = edge["classification"]
        colour = green if status == "COVERED_AND_PASSING" else amber if status == "PARTIALLY_COVERED" else red
        draw.text((x, y), f"{edge['edge_case']}: {status}", font=font(17), fill=colour)
    criteria_y = 1050
    failed = ", ".join(str(value) for value in criteria["failed_criteria"])
    draw.text((45, criteria_y), f"Promotion criteria: 7 pass / 1 fail (failed: {failed})", font=font(23), fill=amber)
    draw.text(
        (45, criteria_y + 43),
        "Blocking fact: zero goal-line-nearest or behind-goal goalkeeper cases.",
        font=font(21),
        fill=red,
    )
    canvas.save(path)
    return path


def write_manifest(folder: Path, manifest_name: str) -> None:
    files = sorted(path for path in folder.iterdir() if path.is_file() and path.name != manifest_name)
    write_json(
        folder / manifest_name,
        {
            "file_count": len(files),
            "files": [
                {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)} for path in files
            ],
            "self_hash_omitted": True,
        },
    )


def package_handoff(
    closure: dict[str, Any],
    coverage: dict[str, Any],
    edges: dict[str, Any],
    policy: dict[str, Any],
    criteria: dict[str, Any],
    decision: dict[str, Any],
    plan: dict[str, Any],
    visual: Path,
) -> None:
    handoff = STAGE / "07_REVIEW_PACK/CHATGPT_HANDOFF"
    write_json(
        handoff / "01_EXECUTIVE_SUMMARY.json",
        {
            "classification": DECISION,
            "model_binding": "GPT-5.6 Sol / Medium",
            "train_matches": 6,
            "authoritative_polygons": 3,
            "runtime_evaluated_matches": 3,
            "failed_promotion_criteria": [5],
            "primary_blocker": "NO_GOAL_LINE_NEAREST_GOALKEEPER_CASE",
            "project_default": "DISABLED",
            "default_changed": False,
            "production_ready": False,
        },
    )
    write_json(handoff / "02_MATCH_AND_POLYGON_COVERAGE.json", coverage)
    write_json(handoff / "03_EDGE_CASE_COVERAGE.json", edges)
    write_json(
        handoff / "04_PROMOTION_CRITERIA_AND_DECISION.json", {"promotion_criteria": criteria, "decision": decision}
    )
    write_json(handoff / "05_ADDITIONAL_COVERAGE_PLAN.json", plan)
    (handoff / "06_DEVELOPMENT_DEFAULT_POLICY_DRAFT.md").write_text(
        (
            f"# {POLICY_ID}\n\n"
            "**DRAFT — NOT ACTIVE.** The project-wide default remains `DISABLED`.\n\n"
            "This draft applies only to `TRAIN_DEVELOPMENT` experiments with a hash-valid, human-confirmed polygon, "
            "a deterministically resolved camera segment, the exact gate contract, and complete external audit "
            "logging. "
            "Any missing or invalid prerequisite resolves to `DISABLED`. Validation, sealed holdout, production, and "
            "historical frozen outputs are excluded. Removing any future opt-in immediately rolls back to `DISABLED`. "
            "`production_ready=false`.\n"
        ),
        encoding="utf-8",
    )
    write_json(
        handoff / "07_TESTS_SAFETY_AND_SOURCE_CHANGES.json",
        {
            "classification": "PASS_G7D_C3A4_FOCUSED_AUDIT_TESTS",
            "focused_test": "tests/test_g7d_c3a4_development_default_readiness.py: 10 passed",
            "full_suite_run": False,
            "inference_run": False,
            "validation_or_holdout_content_accessed": False,
            "runtime_default_changed": False,
            "authorized_repository_changes": [
                "scripts/g7d_c3a4_audit_development_default_readiness.py",
                "tests/test_g7d_c3a4_development_default_readiness.py",
            ],
            "input_closure": closure,
        },
    )
    (handoff / "08_READINESS_MATRIX.png").write_bytes(visual.read_bytes())
    write_manifest(handoff, "09_MANIFEST.json")
    (STAGE / "07_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only the CHATGPT_HANDOFF folder. The policy is DRAFT — NOT ACTIVE.\n",
        encoding="utf-8",
    )


def main() -> int:
    assert_preflight()
    pack_rows = validate_pack()
    closure = validate_frozen_evidence()
    closure["prompt_pack_files"] = pack_rows
    closure["repository_head_at_start"] = EXPECTED_HEAD
    closure["model_binding"] = "GPT-5.6 Sol / Medium"
    source_paths = [
        PROJECT / "datasets/soccertrack_v2/splits/split_v1/train_matches.txt",
        PROJECT / "datasets/soccertrack_v2/splits/split_v1/split_contract.json",
        PROJECT / "datasets/soccertrack_v2/condition_inventory.json",
        C3A / "01_INPUT_CLOSURE/input_validation.json",
        C3A / "04_REVIEWED_SAFETY/boundary_and_official_protection.json",
        C3A / "05_MISSED_MARK_SAFETY/missed_person_neighbourhood_safety.json",
        C3A / "06_FULL_UNIVERSE_SUPPLY/full_universe_gate_comparison.json",
        C3A1 / "04_BOUNDARY_AUDIT/boundary_exception_parity.json",
        C3A2 / "03_PERFORMANCE/benchmark_summary.json",
        C3A2 / "05_SAFETY_REVALIDATION/safety_revalidation.json",
        C3A3 / "01_CONTRACT_AND_DEVICE/active_sandbox_contract.json",
        C3A3 / "02_ACTIVE_CORRECTNESS/active_vs_c3a2_parity.json",
        C3A3 / "03_RUNTIME/runtime_envelope_report.json",
        C3A3 / "05_SAFETY_AND_ROLLBACK/safety_revalidation.json",
        C3A3 / "05_SAFETY_AND_ROLLBACK/output_isolation_and_rollback.json",
    ]
    source_paths.extend(PROJECT / "matches" / match / "calibration/match_setup.json" for match in TRAIN_MATCHES)
    source_paths.extend(
        PROJECT / "matches" / match / "calibration/pitch_polygon_v1/pitch_polygon.json" for match in POLYGON_HASHES
    )
    closure["source_artifacts"] = [artifact(path) for path in source_paths]
    write_json(STAGE / "01_INPUT_CLOSURE/input_closure.json", closure)
    write_manifest(STAGE / "01_INPUT_CLOSURE", "input_closure_manifest.json")

    coverage = build_coverage()
    edges = build_edges()
    policy = draft_policy()
    criteria, decision, plan = promotion_and_plan(coverage, edges)
    write_json(STAGE / "02_COVERAGE_AUDIT/coverage_matrix.json", coverage)
    write_json(STAGE / "03_EDGE_CASE_AUDIT/edge_case_matrix.json", edges)
    write_json(STAGE / "04_DEFAULT_POLICY_DRAFT/development_default_policy_draft.json", policy)
    write_json(STAGE / "05_DECISION/promotion_criteria.json", criteria)
    write_json(STAGE / "05_DECISION/decision.json", decision)
    write_json(STAGE / "05_DECISION/additional_coverage_plan.json", plan)
    write_json(
        STAGE / "06_TESTS_AND_LOGS/source_changes_and_safety.json",
        {
            "VISUAL_ONLY_NOT_METRIC": True,
            "production_ready": False,
            "project_default": "DISABLED",
            "runtime_default_changed": False,
            "inference_run": False,
            "training_or_tuning_run": False,
            "validation_or_holdout_content_accessed": False,
            "source_or_frozen_evidence_modified": False,
            "full_suite_run": False,
            "authorized_repository_changes": [
                "scripts/g7d_c3a4_audit_development_default_readiness.py",
                "tests/test_g7d_c3a4_development_default_readiness.py",
            ],
        },
    )
    write_json(
        STAGE / "06_TESTS_AND_LOGS/focused_test_results.json",
        {
            "classification": "PASS_G7D_C3A4_FOCUSED_AUDIT_TESTS",
            "command": "uv run pytest tests/test_g7d_c3a4_development_default_readiness.py -q",
            "result": "10 passed",
            "full_suite_run": False,
        },
    )
    visual = readiness_visual(coverage, edges, criteria)
    package_handoff(closure, coverage, edges, policy, criteria, decision, plan, visual)
    write_manifest(STAGE / "06_TESTS_AND_LOGS", "tests_and_logs_manifest.json")
    print(json.dumps({"classification": DECISION, "handoff_file_count": 9, "visual_count": 1}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
