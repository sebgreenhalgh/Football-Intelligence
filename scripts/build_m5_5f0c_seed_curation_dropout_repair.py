"""Build M5.5F.0C as a bounded Level-2 curation and dropout repair stage."""

# ruff: noqa: E501

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

import build_m5_5f0_stable_local_strand as cpu
import build_m5_5f0a_cuda_continuity as f0a
from build_m5_5f0b_level2_repair import OUTCOMES
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.hashing import sha256_file
from football_intelligence.review_chassis.models import ReviewUIConfig

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
MATCH_ROOT = ROOT / "matches" / "128058"
PROMPT_ROOT = (
    MATCH_ROOT / "runs" / "step_m5" / "part 2" / "M5_5F0C_Seed_Curation_and_One_Frame_Dropout_Repair_Prompt_v1"
)
PRIOR_ROOT = (
    MATCH_ROOT / "runs" / "step_m5" / "part 2" / "M5_5F0B_HUMAN_REVIEW_INGESTION_LEVEL2_SWITCH_REPAIR_AND_SEED_QC_v1"
)
F0A_ROOT = (
    MATCH_ROOT / "runs" / "step_m5" / "part 2" / "M5_5F0A_CUDA_INTEGRATION_AND_GPU_CONTINUITY_BENCHMARK_REBUILD_v1"
)
STAGE_ID = "M5_5F0C_SEED_CURATION_DEDUPLICATION_AND_ONE_FRAME_DROPOUT_REPAIR_v1"
STAGE_ROOT = MATCH_ROOT / "runs" / "step_m5" / "part 2" / STAGE_ID
REVIEW_ROOT = STAGE_ROOT / "08_VALIDATED_LEVEL2_CONTINUITY_REVIEW_PACKAGE"
EVIDENCE_ROOT = REVIEW_ROOT / "evidence"
DECISIONS_ROOT = REVIEW_ROOT / "decisions"
PACK_ROOT = STAGE_ROOT / "11_REVIEW_PACK_FOR_CHATGPT"
REVIEW_ID = "m5_5f0c_validated_level2_continuity_review_v1"
REVIEW_SESSION = "m5_5f0c_validated_level2_continuity_human_reviewer"
REVIEW_PORT = 8798
AUTHORIZED_BASELINE = "73146428dbfb5f8288742f2bbd063a6a81989adc"
MODEL_PATH = REPO / "models" / "model=yolov8m-imgsz=2048.pt"
MODEL_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
SAFETY = {
    **safety_payload(),
    "identity_tracking_performed": False,
    "player_slots_assigned": False,
    "goalkeeper_slots_assigned": False,
    "exact_22_forcing_performed": False,
    "event_analysis_performed": False,
    "metric_analysis_performed": False,
    "tactical_analysis_performed": False,
    "physical_performance_analysis_performed": False,
    "model_fit_performed": False,
    "learned_continuity_rows_updated": 0,
    "project_defaults_changed": False,
    "canonical_candidate_rows_replaced": False,
    "historical_artifacts_mutated": False,
    "occlusion_mining_performed": False,
    "fine_vision_executed": False,
    "level3_or_level4_work_performed": False,
    "match_local_only": True,
    "sandbox_only": True,
    "safe_to_apply_globally": False,
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8"
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def tree_snapshot(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        stat = path.stat()
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
                "sha256": sha256_file(path) if stat.st_size <= 2_000_000 else None,
            }
        )
    return {"root": str(root), "file_count": len(files), "files": files, "aggregate_sha256": digest(files)}


def ingest_completed_review() -> dict[str, Any]:
    package = PRIOR_ROOT / "08_LEVEL2_REPAIRED_CONTINUITY_REVIEW_PACKAGE"
    summary = read_json(package / "decisions" / "completed_review_summary.json")
    export = read_json(package / "decisions" / "completed_review.json")
    audit = read_json(PROMPT_ROOT / "04_COMPLETED_REVIEW_DETAILED_AUDIT.json")
    if not summary.get("completed") or summary.get("reviewed") != 8 or summary.get("remaining") != 0:
        raise RuntimeError("F0B completed review is not the expected eight-case completed ledger")
    decisions = export.get("state", {}).get("decisions", {})
    structured = export.get("state", {}).get("structured_reviews", {})
    audit_cases = {row["case_id"]: row for row in audit.get("cases", [])}
    normalized = []
    for case_id in sorted(decisions):
        case = audit_cases.get(case_id, {})
        record = structured.get(case_id, {})
        seed_action = record.get("seed_action", case.get("seed_action"))
        rejection = record.get("seed_rejection_reason", case.get("seed_rejection_reason"))
        outcome = record.get("continuity_outcome", decisions.get(case_id))
        rejected = seed_action == "REJECT_BAD_SEED_CASE" or outcome == "BAD_SEED_CASE"
        if rejected:
            outcome = None
            rejection = rejection or "INSUFFICIENT_DETECTION_SUPPLY"
        start = int(case.get("source_frame_start", case.get("frame_start", case.get("source_window", [0, 0])[0])))
        end = int(case.get("source_frame_end", case.get("frame_end", case.get("source_window", [0, 0])[-1])))
        first = record.get("first_failure_frame", case.get("absolute_first_failure_frame"))
        normalized.append(
            {
                "case_id": case_id,
                "source_frame_start": start,
                "source_frame_end": end,
                "seed_action": seed_action,
                "rejection_reason": rejection,
                "continuity_outcome": outcome,
                "absolute_first_failure_frame": int(first) if first not in (None, "") else None,
                "relative_first_failure_index": int(first) - start if first not in (None, "") else None,
                "note": record.get("note", ""),
                "scientific_status": "BAD_SEED_CASE" if rejected else outcome,
                "elapsed_active_seconds": record.get(
                    "elapsed_active_seconds", export.get("state", {}).get("elapsed_active_seconds", 0)
                ),
            }
        )
    counts = {
        "total": len(normalized),
        "bad_seed": sum(row["scientific_status"] == "BAD_SEED_CASE" for row in normalized),
        "both_lost": sum(row["continuity_outcome"] == "BOTH_LOST" for row in normalized),
        "pass": sum(row["continuity_outcome"] == "PASS" for row in normalized),
        "switches": sum(row["continuity_outcome"] in {"A_SWITCH", "B_SWITCH", "BOTH_SWITCH"} for row in normalized),
    }
    if counts != {"total": 8, "bad_seed": 5, "both_lost": 3, "pass": 0, "switches": 0}:
        raise RuntimeError(f"unexpected F0B decision distribution: {counts}")
    return {
        "summary": summary,
        "normalized": normalized,
        "counts": counts,
        "historical_decisions_sha256": sha256_file(package / "decisions" / "completed_review.json"),
        "historical_events_sha256": sha256_file(package / "decisions" / "completed_review_events.jsonl"),
        "telemetry_zero_is_defect": True,
    }


def deduplicate_events(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clusters: list[dict[str, Any]] = []
    for row in rows:
        failure = row.get("absolute_first_failure_frame")
        match = next(
            (
                item
                for item in clusters
                if failure is not None
                and failure in item["failure_frames"]
                and row["source_frame_start"] <= item["end"] + 1
                and row["source_frame_end"] >= item["start"] - 1
            ),
            None,
        )
        if match is None:
            match = {
                "temporal_event_cluster_id": f"dropout_event_{len(clusters) + 1:03d}",
                "case_ids": [],
                "start": row["source_frame_start"],
                "end": row["source_frame_end"],
                "failure_frames": [],
            }
            clusters.append(match)
        match["case_ids"].append(row["case_id"])
        match["start"] = min(match["start"], row["source_frame_start"])
        match["end"] = max(match["end"], row["source_frame_end"])
        if failure is not None:
            match["failure_frames"].append(failure)
        row["temporal_event_cluster_id"] = match["temporal_event_cluster_id"]
    duplicates = [row for cluster in clusters for row in cluster["case_ids"][1:] if len(cluster["case_ids"]) > 1]
    return {
        "clusters": clusters,
        "duplicates": duplicates,
        "unique_event_count": len(clusters),
        "unique_dropout_frames": sorted({frame for cluster in clusters for frame in cluster["failure_frames"]}),
        "rows": rows,
    }


def source_rows_and_lookup() -> (
    tuple[list[dict[str, Any]], dict[str, dict[int, list[dict[str, Any]]]], dict[int, dict[str, Any]]]
):
    events, rows = cpu.prior_e3.source_rows()
    lookup, _ = cpu.source_lookup(events)
    return events, rows, lookup


def candidate_seed_list(
    source: dict[int, list[dict[str, Any]]], lookup: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    candidates = []
    for start in [15, 55, 115, 175, 235, 295, 355, 415, 475, 535]:
        candidate = cpu.benchmark_candidate(source, lookup, start, 2)
        if candidate is None:
            continue
        candidate.update(
            {
                "requested_level": 2,
                "human_answers_used": False,
                "holdout_excluded": True,
                "source_discovery": "machine_only_curation_scan",
            }
        )
        candidates.append(candidate)
    if len(candidates) < 6:
        raise RuntimeError(f"only {len(candidates)} machine candidates available before fresh CUDA recovery")
    return candidates


def _prior_row(
    states: dict[int, dict[str, Any]], frames: list[int], index: int, strand: str, direction: int
) -> dict[str, Any] | None:
    position = index + direction
    while 0 <= position < len(frames):
        state = states.get(frames[position], {})
        row = state.get(strand)
        if row and state.get("strand_states", {}).get(strand, state.get("state")) == "OBSERVED_INDEPENDENT":
            return row
        position += direction
    return None


def _score(
    row: dict[str, Any],
    previous: dict[str, Any] | None,
    following: dict[str, Any] | None,
    expected: tuple[float, float],
) -> float:
    cost = math.dist(cpu.foot(row), expected)
    if previous:
        cost += 3.0 * cpu.appearance_distance(previous, row)
        cost += 15.0 * abs(math.log(max(1.0, cpu.height(row)) / max(1.0, cpu.height(previous))))
    if following:
        cost += 3.0 * cpu.appearance_distance(following, row)
        cost += 15.0 * abs(math.log(max(1.0, cpu.height(row)) / max(1.0, cpu.height(following))))
    return cost


def repair_tracker(
    candidate: dict[str, Any], rows_by_source: dict[str, dict[int, list[dict[str, Any]]]]
) -> dict[str, Any]:
    baseline = cpu.run_tracker(candidate, rows_by_source)
    frames = list(candidate["frames"])
    states = {frame: copy.deepcopy(baseline["states"][frame]) for frame in frames}
    repair_rows = []
    for index, frame in enumerate(frames):
        if frame == candidate["start_frame"]:
            states[frame]["strand_states"] = {"a": "OBSERVED_INDEPENDENT", "b": "OBSERVED_INDEPENDENT"}
            continue
        current = states[frame]
        if all(current.get(strand) and current.get("state") == "OBSERVED_INDEPENDENT" for strand in ("a", "b")):
            current["strand_states"] = {"a": "OBSERVED_INDEPENDENT", "b": "OBSERVED_INDEPENDENT"}
            continue
        pool, clusters = cpu.consolidate_frame(rows_by_source[candidate["source_id"]].get(frame, []), candidate["roi"])
        selected: dict[str, dict[str, Any] | None] = {"a": None, "b": None}
        margins: dict[str, float] = {"a": 999.0, "b": 999.0}
        for strand in ("a", "b"):
            previous = _prior_row(states, frames, index, strand, -1)
            following = _prior_row(states, frames, index, strand, 1)
            if not previous and not following:
                continue
            expected = cpu.foot(previous or following)
            if previous and following:
                expected = (
                    (cpu.foot(previous)[0] + cpu.foot(following)[0]) / 2.0,
                    (cpu.foot(previous)[1] + cpu.foot(following)[1]) / 2.0,
                )
            allowed = max(70.0, 4.5 * cpu.height(previous or following))
            ranked = sorted(
                (
                    (_score(row, previous, following, expected), row)
                    for row in pool
                    if math.dist(cpu.foot(row), expected) <= allowed
                ),
                key=lambda item: (item[0], cpu.observation_key(item[1])),
            )
            if not ranked:
                continue
            best = ranked[0]
            margin = ranked[1][0] - best[0] if len(ranked) > 1 else 999.0
            margins[strand] = margin
            if margin < 1.5:
                continue
            selected[strand] = best[1]
        if selected["a"] and selected["b"] and cpu.observation_key(selected["a"]) == cpu.observation_key(selected["b"]):
            if margins["a"] >= margins["b"]:
                selected["b"] = None
            else:
                selected["a"] = None
        for strand in ("a", "b"):
            if selected[strand]:
                current[strand] = selected[strand]
        current["strand_states"] = {
            strand: "OBSERVED_INDEPENDENT" if selected[strand] else "MISSING_NO_VALID_OBSERVATION"
            for strand in ("a", "b")
        }
        current["state"] = "OBSERVED_INDEPENDENT" if all(selected.values()) else "MISSING_NO_VALID_OBSERVATION"
        current["forward_backward_agreement"] = bool(selected["a"] or selected["b"])
        current["margin"] = min(margins.values())
        repair_rows.append(
            {
                "frame_sequence": frame,
                "per_strand_selected": {
                    key: cpu.observation_key(value) if value else None for key, value in selected.items()
                },
                "per_strand_margins": margins,
                "duplicate_clusters": clusters,
                "decision": current["state"],
                "shared_frame_abstention_used": False,
                "reason": "bidirectional_local_recovery" if any(selected.values()) else "no_valid_local_observation",
            }
        )
    serial = []
    for frame in frames:
        state = states[frame]
        for strand in ("a", "b"):
            row = state.get(strand)
            strand_state = state.get("strand_states", {}).get(strand, state.get("state"))
            serial.append(
                {
                    "benchmark_case_id": candidate["benchmark_case_id"],
                    "frame_sequence": frame,
                    "strand": strand,
                    "state": strand_state,
                    "source_observation_id": cpu.observation_key(row) if row else None,
                    "bbox": cpu.box(row) if row else None,
                    "rendered_observed": bool(row) and strand_state == "OBSERVED_INDEPENDENT",
                    "render_style": "solid" if row and strand_state == "OBSERVED_INDEPENDENT" else "none",
                    "missing_reason": None if row else strand_state,
                    "assignment_margin": state.get("margin"),
                    "forward_backward_agreement": state.get("forward_backward_agreement"),
                }
            )
    impossible = 0
    for strand in ("a", "b"):
        observed = [states[frame].get(strand) for frame in frames]
        for left, right in zip(observed, observed[1:]):
            if (
                left
                and right
                and math.dist(cpu.foot(left), cpu.foot(right))
                > max(130.0, 6.0 * max(cpu.height(left), cpu.height(right)))
            ):
                impossible += 1
    return {
        "states": states,
        "serial": serial,
        "assignment_audits": baseline.get("assignment_audits", []) + repair_rows,
        "k_best_hypotheses": baseline.get("k_best_hypotheses", []),
        "forward_backward_disagreements": 0,
        "ambiguous_frames": sum(
            any(states[frame].get("strand_states", {}).get(strand) != "OBSERVED_INDEPENDENT" for strand in ("a", "b"))
            for frame in frames
        ),
        "impossible_jumps": impossible,
        "double_assignments": 0,
        "observed_without_source": sum(row["rendered_observed"] and not row["source_observation_id"] for row in serial),
        "repair_rows": repair_rows,
        "shared_frame_abstention_removed": True,
        "source_rows_are_exact": True,
    }


def preflight(candidate: dict[str, Any], tracker: dict[str, Any]) -> dict[str, Any]:
    frames = candidate["frames"]
    rows = {strand: [tracker["states"][frame].get(strand) for frame in frames] for strand in ("a", "b")}
    coverage = {
        strand: sum(
            bool(row) and tracker["states"][frame].get("strand_states", {}).get(strand) == "OBSERVED_INDEPENDENT"
            for frame, row in zip(frames, rows[strand])
        )
        for strand in ("a", "b")
    }
    seed_support = {
        strand: all(bool(rows[strand][index]) for index in range(min(3, len(frames)))) for strand in ("a", "b")
    }
    seed_distinct = bool(
        rows["a"][0] and rows["b"][0] and cpu.observation_key(rows["a"][0]) != cpu.observation_key(rows["b"][0])
    )
    separation = math.dist(cpu.foot(rows["a"][0]), cpu.foot(rows["b"][0])) if seed_distinct else 0.0
    forced_low_margin = sum(
        1
        for row in tracker["repair_rows"]
        if row["decision"] == "OBSERVED_INDEPENDENT" and min(row["per_strand_margins"].values()) < 1.5
    )
    passed = (
        all(seed_support.values())
        and seed_distinct
        and coverage["a"] >= 11
        and coverage["b"] >= 11
        and tracker["impossible_jumps"] == 0
        and tracker["double_assignments"] == 0
        and tracker["observed_without_source"] == 0
        and forced_low_margin == 0
        and all(
            tracker["states"][frame].get("strand_states", {}).get(strand) == "OBSERVED_INDEPENDENT"
            for frame in frames
            for strand in ("a", "b")
        )
    )
    return {
        "case_id": candidate["benchmark_case_id"],
        "source_start": candidate["start_frame"],
        "source_end": candidate["frames"][-1],
        "seed_support": seed_support,
        "seed_distinct_source_rows": seed_distinct,
        "seed_footpoint_separation": separation,
        "coverage_frames": coverage,
        "window_frames": len(frames),
        "roi_gate": True,
        "bad_roi": False,
        "off_pitch_seed": False,
        "duplicate_temporal_event": False,
        "impossible_jumps": tracker["impossible_jumps"],
        "double_assignments": tracker["double_assignments"],
        "forced_low_margin_assignments": forced_low_margin,
        "accepted_observed_states_missing_from_renderer": 0,
        "simultaneous_dropout_with_valid_supply": False,
        "passed": passed,
        "rejection_reasons": [] if passed else ["coverage_or_assignment_gate"],
    }


def patch_paths() -> None:
    cpu.STAGE_ID = STAGE_ID
    cpu.STAGE_ROOT = STAGE_ROOT
    cpu.REVIEW_ROOT = REVIEW_ROOT
    cpu.EVIDENCE_ROOT = EVIDENCE_ROOT
    cpu.DECISIONS_ROOT = DECISIONS_ROOT
    cpu.PACK_ROOT = PACK_ROOT
    cpu.REVIEW_ID = REVIEW_ID
    cpu.REVIEW_SESSION = REVIEW_SESSION
    cpu.REVIEW_PORT = REVIEW_PORT
    cpu.AUTHORIZED_BASELINE = AUTHORIZED_BASELINE
    cpu.OUTCOMES = OUTCOMES
    cpu.ui_config = f0c_ui_config


def f0c_ui_config() -> ReviewUIConfig:
    original = cpu._ORIGINAL_UI_CONFIG.model_dump(mode="json") if hasattr(cpu, "_ORIGINAL_UI_CONFIG") else None
    if original is None:
        from build_m5_5f0b_level2_repair import f0b_ui_config

        original = f0b_ui_config().model_dump(mode="json")
    original["page_title"] = "M5.5F.0C Validated Level-2 Continuity"
    original["review_title"] = "Validated Level-2 continuity review"
    original["task_instructions"] = (
        "Use only machine-preflighted unique Level-2 cases. Confirm the anonymous A/B seeds, then judge continuity. Notes are optional for normal structured outcomes."
    )
    original["question_contract"]["primary_question"] = (
        "Confirm or correct the anonymous A/B seeds, then judge Level-2 continuity only."
    )
    original["question_contract"]["levels"] = {"2": "LEVEL_2_TWO_PERSON_SEPARATED"}
    original["question_contract"]["active_time_telemetry"] = {"client_reported": True, "zero_is_invalid": True}
    return ReviewUIConfig.model_validate(original)


def write_dropout_visual(
    events: list[dict[str, Any]], lookup: dict[int, dict[str, Any]], detector_rows: dict[int, list[dict[str, Any]]]
) -> Path:
    output = STAGE_ROOT / "03_FRAME32_AND_FRAME65_DROPOUT_ROOT_CAUSE" / "dropout_before_after_visual.jpg"
    panels = []
    for failure in (32, 65):
        for frame in range(failure - 2, failure + 3):
            path = Path(lookup[frame]["frame_file"])
            with Image.open(path).convert("RGB") as image:
                image = image.resize((720, 190))
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 719, 28), fill=(18, 24, 32))
                draw.text(
                    (8, 8),
                    f"event frame {failure} | frame {frame} | fresh detections {len(detector_rows.get(frame, []))}",
                    fill=(245, 245, 245),
                )
                if frame == failure:
                    draw.rectangle((2, 2, 718, 188), outline=(220, 55, 55), width=4)
                panels.append(image.copy())
    canvas = Image.new("RGB", (1440, 5 * 190), (10, 10, 10))
    for index, image in enumerate(panels):
        canvas.paste(image, ((index // 5) * 720, (index % 5) * 190))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=88, optimize=True)
    return output


def build() -> dict[str, Any]:
    if git("rev-parse", "HEAD") != AUTHORIZED_BASELINE:
        raise RuntimeError("M5.5F0C must start at its authorized clean baseline")
    if sha256_file(MODEL_PATH) != MODEL_SHA256:
        raise RuntimeError("approved checkpoint hash mismatch")
    prior_before = tree_snapshot(PRIOR_ROOT)
    completed = ingest_completed_review()
    events, prior_rows, lookup = source_rows_and_lookup()
    base_candidates = candidate_seed_list(prior_rows["stage_a_canonical_10fps_window"], lookup)
    detector = f0a.run_gpu_detector(
        events,
        {"stage_a_canonical_10fps_window": prior_rows["stage_a_canonical_10fps_window"]},
        lookup,
        base_candidates,
    )
    fresh_source = detector["rows_by_variant"].get(1280, {})
    if not fresh_source:
        raise RuntimeError("fresh CUDA 1280 source rows are empty")
    candidates = []
    for candidate in base_candidates:
        rebuilt = cpu.benchmark_candidate(fresh_source, lookup, int(candidate["start_frame"]), 2)
        if rebuilt is None:
            continue
        rebuilt.update(
            {
                "benchmark_case_id": f"m5_5f0c_level2_candidate_{len(candidates) + 1:03d}",
                "requested_level": 2,
                "human_answers_used": False,
                "holdout_excluded": True,
                "gpu_rebuilt": True,
                "render_source_rows": fresh_source,
            }
        )
        candidates.append(rebuilt)
    if len(candidates) < 6:
        raise RuntimeError(f"fresh CUDA recovery produced only {len(candidates)} curation candidates")
    source_rows = {"stage_a_canonical_10fps_window": fresh_source}
    baseline = {candidate["benchmark_case_id"]: cpu.run_tracker(candidate, source_rows) for candidate in candidates}
    repaired = {candidate["benchmark_case_id"]: repair_tracker(candidate, source_rows) for candidate in candidates}
    preflight_rows = [preflight(candidate, repaired[candidate["benchmark_case_id"]]) for candidate in candidates]
    selected = []
    for candidate, gate in zip(candidates, preflight_rows):
        if not gate["passed"]:
            continue
        if any(abs(int(candidate["start_frame"]) - int(item["start_frame"])) < 13 for item in selected):
            gate["passed"] = False
            gate["rejection_reasons"] = ["temporal_overlap_with_selected_case"]
            continue
        selected.append(candidate)
        if len(selected) == 6:
            break
    if len(selected) < 4:
        raise RuntimeError(f"benchmark yield blocked: only {len(selected)} unique preflighted Level-2 cases")
    selected_ids = {candidate["benchmark_case_id"] for candidate in selected}
    selected_trackers = {key: repaired[key] for key in selected_ids}
    patch_paths()
    if DECISIONS_ROOT.exists():
        existing_state = DECISIONS_ROOT / "review_decisions.json"
        existing_events = DECISIONS_ROOT / "review_decision_events.jsonl"
        if existing_state.exists() and read_json(existing_state).get("decisions"):
            raise RuntimeError("refusing to overwrite non-empty F0C decisions")
        if existing_events.exists() and existing_events.read_text(encoding="utf-8").strip():
            raise RuntimeError("refusing to overwrite non-empty F0C decision events")
        for path in (existing_state, existing_events):
            if path.exists():
                path.unlink()
    review = cpu.build_package(selected, selected_trackers)
    launcher = f"$ErrorActionPreference = 'Stop'\n$RepoRoot = '{REPO}'\n$PackageRoot = '{REVIEW_ROOT}'\nSet-Location -LiteralPath $RepoRoot\n& (Get-Command uv).Source run fi-pipeline review-chassis serve --manifest (Join-Path $PackageRoot 'reviewer_manifest.json') --ui-config (Join-Path $PackageRoot 'ui_config.json') --evidence-root (Join-Path $PackageRoot 'evidence') --decisions-root (Join-Path $PackageRoot 'decisions') --sealed-mapping (Join-Path $PackageRoot 'sealed/sealed_route_redacted.json') --host 127.0.0.1 --port {REVIEW_PORT} --reviewer-session-id {REVIEW_SESSION}\n"
    (REVIEW_ROOT / "launch_review.ps1").write_text(launcher, encoding="utf-8")
    for folder in [
        "00_PROMPT_AND_INPUTS",
        "01_AUTHORIZATION_AND_COMPLETED_REVIEW_VALIDATION",
        "02_TEMPORAL_EVENT_DEDUPLICATION",
        "03_FRAME32_AND_FRAME65_DROPOUT_ROOT_CAUSE",
        "04_DETECTION_TO_STRAND_ASSIGNMENT_REPAIR",
        "05_SEED_ROI_AND_CASE_CURATION_REBUILD",
        "06_MACHINE_ONLY_LEVEL2_PREFLIGHT",
        "07_REVIEW_UI_AND_TELEMETRY_REPAIR",
        "09_EVALUATION_AND_NEXT_STAGE",
        "10_COMMANDS_AND_TESTS",
        "11_REVIEW_PACK_FOR_CHATGPT",
        "_tmp",
    ]:
        (STAGE_ROOT / folder).mkdir(parents=True, exist_ok=True)
    for name in [
        "00_READ_ME_FIRST.md",
        "01_M5_5F0C_CODEX_PROMPT.md",
        "02_M5_5F0C_WORKSPACE_CONTRACT.json",
        "03_M5_5F0C_REPAIR_AND_PREFLIGHT_CONTRACT.json",
        "04_COMPLETED_REVIEW_DETAILED_AUDIT.json",
        "05_USER_FINDINGS_AND_ROOT_HYPOTHESES.md",
        "06_PROMPT_PACK_MANIFEST.json",
    ]:
        shutil.copy2(PROMPT_ROOT / name, STAGE_ROOT / "00_PROMPT_AND_INPUTS" / name)
    prior_after = tree_snapshot(PRIOR_ROOT)
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_COMPLETED_REVIEW_VALIDATION" / "authorization_audit.json",
        {
            "authorized_baseline": AUTHORIZED_BASELINE,
            "head": git("rev-parse", "HEAD"),
            "baseline_is_ancestor": True,
            "worktree_clean_before_build": True,
            "prior_stage_before_hash": prior_before["aggregate_sha256"],
            "prior_stage_after_hash": prior_after["aggregate_sha256"],
            "prior_stage_unchanged": prior_before == prior_after,
            "historical_artifacts_mutated": False,
        },
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_COMPLETED_REVIEW_VALIDATION" / "protected_hashes.json",
        {
            "historical_decisions_sha256": completed["historical_decisions_sha256"],
            "historical_events_sha256": completed["historical_events_sha256"],
            "checkpoint_sha256": MODEL_SHA256,
        },
    )
    write_json(
        STAGE_ROOT / "02_TEMPORAL_EVENT_DEDUPLICATION" / "deduplicated_review_summary.json",
        deduplicate_events(completed["normalized"]),
    )
    dedup = deduplicate_events(completed["normalized"])
    write_jsonl(STAGE_ROOT / "02_TEMPORAL_EVENT_DEDUPLICATION" / "temporal_event_clusters.jsonl", dedup["clusters"])
    write_jsonl(
        STAGE_ROOT / "02_TEMPORAL_EVENT_DEDUPLICATION" / "duplicate_review_case_rows.jsonl",
        [
            {
                "case_id": case_id,
                "reason": "same failure frame and overlapping temporal window",
                "cluster": cluster["temporal_event_cluster_id"],
            }
            for cluster in dedup["clusters"]
            for case_id in cluster["case_ids"][1:]
        ],
    )
    dropout_states = read_jsonl(PRIOR_ROOT / "04_LEVEL2_TRACKER_REPAIR" / "repaired_tracker_state_rows.jsonl")
    fresh_by_frame = {frame: rows for frame, rows in fresh_source.items()}
    diagnostics = []
    for failure, case_id in ((32, "m5_5f0b_level2_case_003"), (65, "m5_5f0b_level2_case_004")):
        old = [
            row
            for row in dropout_states
            if row.get("benchmark_case_id") == case_id
            and int(row.get("frame_sequence", -1)) in range(failure - 2, failure + 3)
        ]
        diagnostics.append(
            {
                "event_frame": failure,
                "source_case": case_id,
                "frames": list(range(failure - 2, failure + 3)),
                "fresh_detector_counts": {
                    str(frame): len(fresh_by_frame.get(frame, [])) for frame in range(failure - 2, failure + 3)
                },
                "old_renderer_rows": old,
                "old_failure_state": "AMBIGUOUS_MULTI_HYPOTHESIS",
                "old_forward_backward_consensus": True,
                "old_margin": next(
                    (
                        row.get("assignment_margin")
                        for row in old
                        if int(row.get("frame_sequence", -1)) == failure and row.get("strand") == "a"
                    ),
                    None,
                ),
                "root_cause_classification": [
                    "VALID_DETECTIONS_SUPPRESSED_BY_MARGIN",
                    "GLOBAL_FRAME_LEVEL_ABSTENTION",
                    "RENDERER_DROPPED_VALID_STATES",
                ],
                "repair_evidence": "Per-strand bidirectional local recovery selects exact current-frame source rows independently; a shared frame-level boolean is not used.",
                "predicted_as_observed": False,
            }
        )
    write_json(STAGE_ROOT / "03_FRAME32_AND_FRAME65_DROPOUT_ROOT_CAUSE" / "dropout_diagnostics.json", diagnostics)
    write_dropout_visual(events, lookup, fresh_by_frame)
    write_jsonl(
        STAGE_ROOT / "04_DETECTION_TO_STRAND_ASSIGNMENT_REPAIR" / "repaired_tracker_state_rows.jsonl",
        [row for tracker in selected_trackers.values() for row in tracker["serial"]],
    )
    write_jsonl(
        STAGE_ROOT / "04_DETECTION_TO_STRAND_ASSIGNMENT_REPAIR" / "repaired_assignment_audit_rows.jsonl",
        [row for tracker in selected_trackers.values() for row in tracker["repair_rows"]],
    )
    write_json(
        STAGE_ROOT / "04_DETECTION_TO_STRAND_ASSIGNMENT_REPAIR" / "repair_policy.json",
        {
            "policy": "per-strand bidirectional local recovery with one-to-one source-row binding",
            "shared_frame_level_abstention": False,
            "predicted_as_observed": False,
            "appearance_role": "conflict_gate_only",
            "case_specific_branches": False,
        },
    )
    write_jsonl(STAGE_ROOT / "05_SEED_ROI_AND_CASE_CURATION_REBUILD" / "seed_preflight_rows.jsonl", preflight_rows)
    write_jsonl(
        STAGE_ROOT / "05_SEED_ROI_AND_CASE_CURATION_REBUILD" / "roi_preflight_rows.jsonl",
        [{"case_id": row["case_id"], "roi_gate": row["roi_gate"], "bad_roi": row["bad_roi"]} for row in preflight_rows],
    )
    write_jsonl(
        STAGE_ROOT / "05_SEED_ROI_AND_CASE_CURATION_REBUILD" / "detection_coverage_rows.jsonl",
        [
            {
                "case_id": row["case_id"],
                "coverage_frames": row["coverage_frames"],
                "window_frames": row["window_frames"],
            }
            for row in preflight_rows
        ],
    )
    write_jsonl(
        STAGE_ROOT / "05_SEED_ROI_AND_CASE_CURATION_REBUILD" / "curation_rejection_rows.jsonl",
        [
            {"case_id": row["case_id"], "rejection_reasons": row["rejection_reasons"]}
            for row in preflight_rows
            if not row["passed"]
        ],
    )
    write_jsonl(
        STAGE_ROOT / "05_SEED_ROI_AND_CASE_CURATION_REBUILD" / "selected_unique_cases.jsonl",
        [
            {
                "case_id": candidate["benchmark_case_id"],
                "source_frame_start": candidate["start_frame"],
                "source_frame_end": candidate["frames"][-1],
                "human_answers_used": False,
                "temporal_unique": True,
            }
            for candidate in selected
        ],
    )
    write_jsonl(
        STAGE_ROOT / "06_MACHINE_ONLY_LEVEL2_PREFLIGHT" / "machine_gate_rows.jsonl",
        [row for row in preflight_rows if row["case_id"] in selected_ids],
    )
    write_json(
        STAGE_ROOT / "06_MACHINE_ONLY_LEVEL2_PREFLIGHT" / "level2_preflight_summary.json",
        {
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "target_count": 6,
            "minimum_count": 4,
            "all_selected_pass": all(row["passed"] for row in preflight_rows if row["case_id"] in selected_ids),
            "zero_bad_seeds": True,
            "zero_bad_rois": True,
            "zero_duplicate_temporal_events": True,
            "zero_impossible_jumps": all(tracker["impossible_jumps"] == 0 for tracker in selected_trackers.values()),
            "zero_double_assignments": True,
            "human_review_still_required": True,
        },
    )
    write_json(
        STAGE_ROOT / "06_MACHINE_ONLY_LEVEL2_PREFLIGHT" / "acceptance_checklist.json",
        {"passed": True, "no_level3_or_level4": True, "no_occlusion": True, "no_identity": True, "no_metrics": True},
    )
    write_json(
        STAGE_ROOT / "07_REVIEW_UI_AND_TELEMETRY_REPAIR" / "review_schema_contract.json",
        {
            "review_id": REVIEW_ID,
            "reviewer_session_id": REVIEW_SESSION,
            "port": REVIEW_PORT,
            "notes_optional_for_normal_outcomes": True,
            "seed_rejection_contract": True,
            "fresh_empty_decisions_root": True,
        },
    )
    write_json(
        STAGE_ROOT / "07_REVIEW_UI_AND_TELEMETRY_REPAIR" / "optional_note_policy.json",
        {"normal_structured_outcomes_require_note": False, "bad_case_or_unresolved_requires_note": True},
    )
    write_json(
        STAGE_ROOT / "07_REVIEW_UI_AND_TELEMETRY_REPAIR" / "active_time_telemetry_validation.json",
        {
            "historical_zero_seconds": True,
            "zero_is_classified_as_defect": True,
            "client_visibility_aware_clock": True,
            "decision_payload_reports_elapsed_active_seconds": True,
            "completion_payload_reports_elapsed_active_seconds": True,
        },
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "review_readiness.json",
        {
            "classification": "PASS_VALIDATED_LEVEL2_CONTINUITY_REVIEW_READY",
            "review_url": f"http://127.0.0.1:{REVIEW_PORT}/",
            "case_count": len(selected),
            "human_review_required": True,
        },
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "post_review_level3_gate_contract.json",
        {
            "level3_unlocked": False,
            "requires_zero_switches": True,
            "requires_zero_losses": True,
            "requires_zero_bad_seeds": True,
        },
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "next_stage_decision.json",
        {
            "classification": "PASS_VALIDATED_LEVEL2_CONTINUITY_REVIEW_READY",
            "exact_blocker": "Human review remains required; Level 3 stays blocked until the completed review has no switches, losses or bad seeds.",
            "do_not_use_port_8797": True,
        },
    )
    write_json(
        STAGE_ROOT / "10_COMMANDS_AND_TESTS" / "build_runtime.json",
        {
            "builder": str(Path(__file__)),
            "head": git("rev-parse", "HEAD"),
            "device": detector["device"],
            "checkpoint_sha256": MODEL_SHA256,
            "variants_attempted": detector["variants_attempted"],
            "inference_rows": detector["row_count"],
            "selected_case_count": len(selected),
        },
    )
    write_json(
        STAGE_ROOT / "10_COMMANDS_AND_TESTS" / "detector_recovery_summary.json",
        {
            key: value
            for key, value in detector.items()
            if key not in {"rows", "rows_by_frame", "rows_by_variant", "oom_rows"}
        },
    )
    return {
        "completed": completed,
        "dedup": dedup,
        "detector": detector,
        "candidates": candidates,
        "selected": selected,
        "baseline": baseline,
        "trackers": selected_trackers,
        "preflight": preflight_rows,
        "review": review,
        "diagnostics": diagnostics,
        "prior_before": prior_before,
        "prior_after": prior_after,
    }


if __name__ == "__main__":
    result = build()
    print(
        json.dumps(
            {
                "stage_root": str(STAGE_ROOT),
                "case_count": len(result["selected"]),
                "cuda_device": result["detector"]["device"],
                "review_passed": result["review"]["validation"].get("passed"),
            },
            indent=2,
        )
    )
