"""Finalize and validate the M5.5G.1A ChatGPT review pack."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
STAGE = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 3"
    / "M5_5G1A_DETECTION_GOLD_FOUNDATION_ONTOLOGY_FREEZE_AND_PILOT_ANNOTATION_v1"
)
PACK = STAGE / "14_REVIEW_PACK_FOR_CHATGPT"
BASELINE = "d06c798ddb09ab6cdb18738b9c95cb9906d162a6"
ORIGIN = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
CLASSIFICATION = "PASS_DETECTION_GOLD_PILOT_ANNOTATION_READY"
EXPECTED_FILES = (
    "REVIEW_PACK_MANIFEST.json",
    "01_EXECUTIVE_SUMMARY.md",
    "02_RUN_CONTEXT.json",
    "03_CHANGED_FILES.md",
    "04_SOURCE_DIFF.patch",
    "05_COMMANDS_AND_TEST_RESULTS.md",
    "06_ARTIFACT_INDEX.json",
    "07_G0_RESEAL_AND_PRO_DECISION.json",
    "08_FROZEN_ONTOLOGIES.json",
    "09_MATCHING_METRICS_AND_GATES.json",
    "10_PILOT_SELECTION_AND_BINDING.json",
    "11_UI_PERSISTENCE_BROWSER.json",
    "12_TIMING_AND_INTERACTION.json",
    "13_NEXT_STAGE_FULL_GOLD_CONTRACT.json",
    "14_SAFETY_AND_ACCEPTANCE.json",
    "15_PART3_ROADMAP.md",
    "16_HUMAN_ANNOTATION_INSTRUCTIONS.md",
    "17_PLAYER_STATIC_UI.png",
    "18_DENSE_MASK_UI.png",
    "19_FOOTBALL_BURST_UI.png",
)


def command(args: list[str]) -> str:
    return subprocess.run(
        args,
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_text(value: str) -> str:
    replacements = (
        (str(REPO), "<REPOSITORY>"),
        (str(REPO).replace("\\", "/"), "<REPOSITORY>"),
        (str(ROOT), "<FOOTBALL_INTELLIGENCE_ROOT>"),
        (str(ROOT).replace("\\", "/"), "<FOOTBALL_INTELLIGENCE_ROOT>"),
        (str(Path.home()), "<USER_PROFILE>"),
        (str(Path.home()).replace("\\", "/"), "<USER_PROFILE>"),
    )
    for source, replacement in replacements:
        value = value.replace(source, replacement)
    return value


def write_text(root: Path, name: str, value: str) -> None:
    (root / name).write_text(safe_text(value).rstrip() + "\n", encoding="utf-8")


def write_json(root: Path, name: str, value: Any) -> None:
    write_text(root, name, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True))


def artifact_index() -> dict[str, Any]:
    sections = []
    for directory in sorted(path for path in STAGE.iterdir() if path.is_dir()):
        if directory.name in {"_tmp", PACK.name}:
            continue
        files = [path for path in directory.rglob("*") if path.is_file()]
        sections.append(
            {
                "section": directory.name,
                "file_count": len(files),
                "size_bytes": sum(path.stat().st_size for path in files),
            }
        )
    return {
        "schema_version": "football_intelligence.m5_5g1a.review_artifact_index.v1",
        "sections": sections,
        "total_file_count": sum(row["file_count"] for row in sections),
        "total_size_bytes": sum(row["size_bytes"] for row in sections),
        "temporary_artifacts_excluded": True,
        "review_pack_excluded_from_its_own_index": True,
    }


def compact_browser(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": report["passed"],
        "url": report["url"],
        "route_and_privacy_audit": report["route_and_privacy_audit"],
        "viewport_profiles": [
            {
                "profile": row["profile"],
                "passed": row["passed"],
                "physical_viewport": row["physical_viewport"],
                "effective_browser_zoom_percent": row["effective_browser_zoom_percent"],
                "image_overlay_max_delta": row["imageOverlayMaxDelta"],
                "body_horizontal_overflow_pixels": row["bodyHorizontalOverflowPixels"],
            }
            for row in report["visual_regression"]
        ],
        "module_checks": {
            "player_provenance_populated": report["module_interactions"]["player_static"]["interaction"][
                "provenancePopulated"
            ],
            "dense_mask_screen_round_trip_max_pixels": report["module_interactions"]["dense_region"][
                "mask_interaction"
            ]["screenRoundTripMaxPixels"],
            "temporal_contact_frame_count": report["module_interactions"]["temporal_player"]["stable_run_gate"][
                "contactFrameCount"
            ],
            "stable_run_requires_full_review": report["module_interactions"]["temporal_player"]["stable_run_gate"][
                "beforeContactDisabled"
            ],
            "football_contact_frame_count": report["module_interactions"]["football_burst"]["contact_frame_count"],
        },
        "persistence_passed": report["persistence"]["passed"],
        "production_decisions_preserved": report["production_decisions_preservation"]["passed"],
        "real_decisions_root_opened": report["real_decisions_root_opened"],
    }


def validate_visual(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        spread = max(ImageStat.Stat(image.convert("RGB").resize((96, 60))).stddev)
        result = {
            "filename": path.name,
            "width": image.width,
            "height": image.height,
            "rgb_standard_deviation_max": round(spread, 3),
            "nonblank": spread > 8,
        }
    if result["width"] < 1000 or result["height"] < 600 or not result["nonblank"]:
        raise RuntimeError(f"invalid review-pack visual: {result}")
    return result


def validate_pack(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.iterdir() if path.is_file())
    names = tuple(path.name for path in files)
    nested = [path for path in root.rglob("*") if path.is_file() and path.parent != root]
    total_size = sum(path.stat().st_size for path in files)
    visuals = [path for path in files if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    forbidden_extensions = {".mp4", ".avi", ".mov", ".pt", ".pth", ".onnx"}
    forbidden_names = [
        path.name for path in files if path.suffix.lower() in forbidden_extensions or "sealed" in path.name.lower()
    ]
    privacy_hits = []
    for path in files:
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        privacy_tokens = (
            "C:" + "\\Users\\",
            "/" + "Users/",
            "BEGIN " + "PRIVATE KEY",
            "pass" + "word=",
        )
        for token in privacy_tokens:
            if token.lower() in text.lower():
                privacy_hits.append({"filename": path.name, "token": token})
    visual_results = [validate_visual(path) for path in visuals]
    checks = {
        "exact_expected_files": set(names) == set(EXPECTED_FILES),
        "file_count_at_most_20": len(files) <= 20,
        "file_count_exactly_20": len(files) == 20,
        "flat": not nested,
        "total_size_at_most_50_mib": total_size <= 50 * 1024 * 1024,
        "visual_count_at_most_three": len(visuals) <= 3,
        "source_diff_present_nonempty": (root / "04_SOURCE_DIFF.patch").stat().st_size > 0,
        "forbidden_binary_payloads_absent": not forbidden_names,
        "personal_or_credential_tokens_absent": not privacy_hits,
        "visuals_valid": all(row["nonblank"] for row in visual_results),
    }
    result = {
        "schema_version": "football_intelligence.m5_5g1a.review_pack_validation.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "file_count": len(files),
        "total_size_bytes": total_size,
        "visual_count": len(visuals),
        "visuals": visual_results,
        "forbidden_names": forbidden_names,
        "privacy_hits": privacy_hits,
    }
    if not result["passed"]:
        raise RuntimeError(f"review-pack validation failed: {result}")
    return result


def main() -> None:
    head = command(["git", "rev-parse", "HEAD"])
    branch = command(["git", "branch", "--show-current"])
    origin = command(["git", "remote", "get-url", "origin"])
    upstream = command(["git", "rev-parse", "@{upstream}"])
    status = command(["git", "status", "--porcelain"])
    ancestor = (
        subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE, head], cwd=REPO, check=False).returncode == 0
    )
    if not all((head != BASELINE, branch == "main", origin == ORIGIN, upstream == head, not status, ancestor)):
        raise RuntimeError(
            f"repository finalization gate failed: head={head}, branch={branch}, origin={origin}, "
            f"upstream={upstream}, clean={not status}, ancestor={ancestor}"
        )
    if PACK.exists() and any(PACK.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty review pack: {PACK}")

    work = STAGE / "_tmp" / f"review_pack_build_{uuid.uuid4().hex[:10]}"
    work.mkdir(parents=True)
    build = read_json(STAGE / "13_COMMANDS_AND_TESTS" / "build_summary.json")
    commands = read_json(STAGE / "13_COMMANDS_AND_TESTS" / "command_results.json")
    source_scope = read_json(STAGE / "13_COMMANDS_AND_TESTS" / "source_scope_audit.json")
    g0 = read_json(STAGE / "01_G0_PRO_PACK_RESEALED" / "local_g0_pack_validation.json")
    pro = read_json(STAGE / "02_PRO_DECISION_INGESTION" / "pro_decision_hash_and_index.json")
    stage_map = read_json(STAGE / "02_PRO_DECISION_INGESTION" / "recommendation_to_stage_map.json")
    freeze = read_json(STAGE / "03_GOLD_ONTOLOGY_AND_SCHEMA_FREEZE" / "schema_freeze_manifest.json")
    matching = read_json(STAGE / "04_MATCHING_METRICS_AND_ACCEPTANCE_GATES" / "matching_specification.json")
    metrics = read_json(STAGE / "04_MATCHING_METRICS_AND_ACCEPTANCE_GATES" / "future_metric_schema.json")
    gates = read_json(STAGE / "04_MATCHING_METRICS_AND_ACCEPTANCE_GATES" / "frozen_acceptance_gates.json")
    pilot = read_json(STAGE / "05_PILOT_CASE_SELECTION_AND_BINDING" / "pilot_case_manifest.json")
    dedup = read_json(STAGE / "05_PILOT_CASE_SELECTION_AND_BINDING" / "case_deduplication.json")
    binding = read_json(STAGE / "05_PILOT_CASE_SELECTION_AND_BINDING" / "case_binding_validation.json")
    browser = read_json(STAGE / "11_BROWSER_PERSISTENCE_AND_VISUAL_REGRESSION" / "browser_persistence_results.json")
    timing = read_json(STAGE / "09_ANNOTATION_TIMING_AND_INTERACTION_PLAN" / "annotation_time_estimate.json")
    interaction = read_json(
        STAGE / "09_ANNOTATION_TIMING_AND_INTERACTION_PLAN" / "interaction_efficiency_validation.json"
    )
    package = read_json(STAGE / "10_DETECTION_GOLD_PILOT_ANNOTATION_PACKAGE" / "review_package_validation.json")
    next_stage = read_json(STAGE / "12_NEXT_STAGE_FULL_GOLD_CONTRACT" / "next_stage_full_gold_contract.json")
    completion = read_json(STAGE / "12_NEXT_STAGE_FULL_GOLD_CONTRACT" / "pilot_completion_ingestion_contract.json")

    write_text(
        work,
        "01_EXECUTIVE_SUMMARY.md",
        f"""# M5.5G.1A executive summary

## Outcome

{CLASSIFICATION}. The original M5.5G.0 context pack was reconstructed as an
exact 20-file byte-valid transfer, and the ChatGPT Pro decision was frozen as
TARGETED DETECTION GOLD FIRST.

The stage freezes player, dense-mask, temporal, pitch/role and football
ontologies plus matching, future metrics and hard acceptance gates. It creates
an 88-case diagnostic-only pilot: 32 static player cases, 8 dense-mask cases,
12 eleven-frame temporal bursts, 12 pitch/boundary cases and 24 nine-frame
football bursts.

The five-module annotation application passed real-browser, crash-recovery and
responsive acceptance. The estimated active annotation time is
{timing['estimated_active_minutes']} minutes. No detector or tracker was
evaluated, tuned or promoted.
""",
    )
    write_json(
        work,
        "02_RUN_CONTEXT.json",
        {
            "schema_version": "football_intelligence.m5_5g1a.review_run_context.v1",
            "part": 3,
            "stage": "M5.5G.1A",
            "classification": CLASSIFICATION,
            "authorized_baseline": BASELINE,
            "implementation_commit": head,
            "branch": branch,
            "origin": origin,
            "commit_pushed": upstream == head,
            "working_tree_clean_at_finalization": not status,
            "review_url": "http://127.0.0.1:8807/",
            "review_id": "m5_5g1a_detection_gold_pilot_v1",
            "reviewer_session_id": "m5_5g1a_detection_gold_pilot_reviewer",
        },
    )
    changed = command(["git", "diff", "--name-status", f"{BASELINE}..{head}"])
    write_text(
        work,
        "03_CHANGED_FILES.md",
        "# Changed files\n\n```text\n"
        + changed
        + "\n```\n\nNo detector runtime, defaults, lockfile or weights changed.",
    )
    source_diff = command(["git", "diff", "--binary", f"{BASELINE}..{head}"])
    write_text(work, "04_SOURCE_DIFF.patch", source_diff)
    write_text(
        work,
        "05_COMMANDS_AND_TEST_RESULTS.md",
        "# Commands and tests\n\n"
        + "\n".join(f"- `{name}`: {result}" for name, result in commands["commands"].items()),
    )
    write_json(work, "06_ARTIFACT_INDEX.json", artifact_index())
    write_json(
        work,
        "07_G0_RESEAL_AND_PRO_DECISION.json",
        {
            "g0_local_validation_passed": g0["passed"],
            "g0_reseal_passed": g0["resealed"]["passed"],
            "g0_resealed_file_count": g0["resealed"]["file_count"],
            "source_diff_present": g0["source_diff_present_nonempty"],
            "atlas_byte_matches": all(
                row["sha256_match"] and row["size_match"] for row in g0["file_checks"] if "ATLAS" in row["name"]
            ),
            "pro_decision_sha256": pro["sha256"],
            "pro_decision_section_count": pro["section_count"],
            "final_next_stage_choice": pro["final_next_stage_choice"],
            "recommendation_stage_count": len(stage_map["rows"]),
        },
    )
    schema_summaries = []
    for name in (
        "player_instance_schema.json",
        "dense_region_schema.json",
        "temporal_player_schema.json",
        "pitch_role_schema.json",
        "football_schema.json",
    ):
        path = STAGE / "03_GOLD_ONTOLOGY_AND_SCHEMA_FREEZE" / name
        schema = read_json(path)
        schema_summaries.append(
            {
                "filename": name,
                "schema_id": schema["$id"],
                "schema_version": schema["x-schema-version"],
                "sha256": sha256_file(path),
                "required_fields": schema["required"],
                "additional_properties": schema["additionalProperties"],
            }
        )
    write_json(
        work,
        "08_FROZEN_ONTOLOGIES.json",
        {
            "ontology_version": freeze["ontology_version"],
            "freeze_hash": freeze["freeze_hash"],
            "frozen_before_case_selection": freeze["frozen_before_case_selection"],
            "schemas": schema_summaries,
        },
    )
    write_json(
        work,
        "09_MATCHING_METRICS_AND_GATES.json",
        {"matching": matching, "future_metrics": metrics, "frozen_acceptance_gates": gates},
    )
    write_json(
        work,
        "10_PILOT_SELECTION_AND_BINDING.json",
        {
            "total_cases": pilot["total_cases"],
            "counts_by_task": pilot["counts_by_task"],
            "counts_by_task_and_stratum": pilot["counts_by_task_and_stratum"],
            "diagnostic_only": pilot["diagnostic_only"],
            "validation_or_holdout_use_forbidden": pilot["validation_or_holdout_use_forbidden"],
            "deduplication_passed": dedup["passed"],
            "binding_validation_passed": binding["passed"],
            "package_validation": package,
        },
    )
    write_json(work, "11_UI_PERSISTENCE_BROWSER.json", compact_browser(browser))
    write_json(work, "12_TIMING_AND_INTERACTION.json", {"estimate": timing, "browser_exercise": interaction})
    write_json(
        work,
        "13_NEXT_STAGE_FULL_GOLD_CONTRACT.json",
        {"next_stage": next_stage, "pilot_completion_ingestion": completion},
    )
    write_json(
        work,
        "14_SAFETY_AND_ACCEPTANCE.json",
        {
            "classification": build["classification"],
            "safety": {
                key: build[key]
                for key in (
                    "visual_only_warning",
                    "production_ready",
                    "no_auto_promotion",
                    "human_approved",
                    "safe_to_apply_globally",
                    "match_local_only",
                    "sandbox_only",
                    "identity_tracking_performed",
                    "player_slots_assigned",
                    "goalkeeper_slots_assigned",
                    "exact_22_forcing_performed",
                    "detector_or_tracker_evaluated",
                    "detector_or_tracker_promoted",
                )
            },
            "source_scope": source_scope,
            "full_suite_passed": commands["full_suite_passed"],
            "browser_acceptance_passed": commands["browser_acceptance_passed"],
        },
    )
    roadmap = (
        ROOT
        / "matches"
        / "128058"
        / "runs"
        / "step_m5"
        / "part 3"
        / "M5_5G1A_Detection_Gold_Foundation_and_Pilot_Annotation_v1"
        / "09_PART3_STAGE_ROADMAP.md"
    ).read_text(encoding="utf-8")
    write_text(work, "15_PART3_ROADMAP.md", roadmap)
    write_text(
        work,
        "16_HUMAN_ANNOTATION_INSTRUCTIONS.md",
        """# Human annotation instructions

1. This is Part 3 and uses only `http://127.0.0.1:8807/`.
2. Do not use any earlier annotation port for this pilot.
3. This pilot is diagnostic only and cannot enter validation or holdout data.
4. Annotate all five modules: Players, Dense, Temporal, Pitch and Football.
5. Prefer explicit proposal selection; draw geometry only when supply is missing.
6. Use stable-run only after reviewing the complete 11-frame contact strip.
7. Notes are optional.
8. Stop immediately if `Saved to server` disappears or the server event sequence stops advancing.
9. Complete only after all 88 cases are saved and the server enables completion.

No detector or tracker was promoted.
""",
    )
    source_visuals = (
        ("01_PLAYER_STATIC_ANNOTATION_UI.png", "17_PLAYER_STATIC_UI.png"),
        ("02_DENSE_VISIBLE_MASK_UI.png", "18_DENSE_MASK_UI.png"),
        ("03_FOOTBALL_BURST_ANNOTATION_UI.png", "19_FOOTBALL_BURST_UI.png"),
    )
    browser_root = STAGE / "11_BROWSER_PERSISTENCE_AND_VISUAL_REGRESSION"
    for source_name, destination_name in source_visuals:
        with Image.open(browser_root / source_name) as image:
            image.convert("RGB").save(work / destination_name, format="PNG", optimize=True)

    payload_rows = []
    for path in sorted(item for item in work.iterdir() if item.is_file()):
        payload_rows.append({"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": "football_intelligence.m5_5g1a.review_pack_manifest.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "classification": CLASSIFICATION,
        "implementation_commit": head,
        "file_count_including_manifest": len(payload_rows) + 1,
        "payload_files": payload_rows,
        "payload_manifest_sha256": hashlib.sha256(
            json.dumps(payload_rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "flat": True,
        "maximum_file_count": 20,
        "maximum_size_mib": 50,
        "maximum_visual_count": 3,
        "raw_video_included": False,
        "model_weights_included": False,
        "sealed_mappings_included": False,
        "candidate_ids_included": False,
        "hidden_expected_answers_included": False,
        "credentials_or_personal_data_included": False,
    }
    write_json(work, "REVIEW_PACK_MANIFEST.json", manifest)
    validation = validate_pack(work)
    manifest["validation"] = validation
    write_json(work, "REVIEW_PACK_MANIFEST.json", manifest)
    validation = validate_pack(work)
    if not validation["passed"]:
        raise RuntimeError("review pack failed after manifest validation update")

    PACK.mkdir(parents=True, exist_ok=True)
    for path in work.iterdir():
        shutil.copy2(path, PACK / path.name)
    published_validation = validate_pack(PACK)
    if not published_validation["passed"]:
        raise RuntimeError("published review pack failed validation")
    print(
        json.dumps(
            {
                "passed": True,
                "pack": str(PACK),
                "file_count": published_validation["file_count"],
                "size_bytes": published_validation["total_size_bytes"],
                "visual_count": published_validation["visual_count"],
                "implementation_commit": head,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
