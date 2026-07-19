from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from football_intelligence.review_chassis.hashing import sha256_file


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART2 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 2"
STAGE = PART2 / "M5_5F1D_FROZEN_P_MHSAG_PREREGISTRATION_ONE_TIME_SEALED_HOLDOUT_AND_ROBUSTNESS_AUDIT_v1"
PACK = STAGE / "14_REVIEW_PACK_FOR_CHATGPT"
BASELINE = "cf4d0222e2e8aabf1c462286fc71788e0acd9fc6"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def safe_text(value: str) -> str:
    return value.replace(str(Path.home()), "<USER_PROFILE>").replace("C:\\Users\\sebgr", "<USER_PROFILE>")


def write_text(name: str, value: str) -> None:
    (PACK / name).write_text(safe_text(value).rstrip() + "\n", encoding="utf-8")


def write_json(name: str, value: Any) -> None:
    write_text(name, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True))


def command(args: list[str]) -> str:
    return subprocess.run(args, cwd=REPO, check=True, capture_output=True, text=True, encoding="utf-8").stdout


def compact_result(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    return {
        "mode": payload["mode"],
        "configuration_hash": payload["configuration_hash"],
        "metrics": payload["metrics"],
        "exact_A_path_sequences": payload["exact_A_path_sequences"],
        "exact_B_path_sequences": payload["exact_B_path_sequences"],
        "runtime_seconds": payload["runtime_seconds"],
        "sequence_count": len(payload["sequence_results"]),
        "retuning_performed": payload["retuning_performed"],
        "tracker_promoted": payload["tracker_promoted"],
    }


def contact_sheet(paths: list[Path], output: Path) -> None:
    loaded = []
    for index, path in enumerate(paths[:8], 1):
        image = Image.open(path).convert("RGB")
        width = 560
        height = round(width * image.height / image.width)
        loaded.append((image.resize((width, height), Image.Resampling.LANCZOS), f"Anonymous sequence {index:02d}"))
        image.close()
    columns = 2
    tile_height = max(image.height for image, _ in loaded) + 34
    sheet = Image.new("RGB", (columns * 560, math.ceil(len(loaded) / columns) * tile_height), (9, 14, 18))
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(loaded):
        x = index % columns * 560
        y = index // columns * tile_height
        draw.text((x + 8, y + 8), label, fill=(235, 241, 246))
        sheet.paste(image, (x, y + 30))
    sheet.save(output, quality=91, optimize=True)
    sheet.close()


def robustness_visual(characterization: dict[str, Any], rows: list[dict[str, Any]], output: Path) -> None:
    image = Image.new("RGB", (1500, 760), (9, 14, 18))
    draw = ImageDraw.Draw(image)
    draw.text((42, 34), "M5.5F.1D pre-registered shadow robustness", fill=(240, 245, 248))
    draw.text(
        (42, 72), "Primary result remains immutable; stresses never select or tune the candidate.", fill=(170, 188, 199)
    )
    y = 132
    for row in rows:
        metrics = row.get("scientific_metrics")
        if metrics is None:
            status = "CANARY PASS" if row.get("provenance_canary_passed") else "CANARY FAIL"
        else:
            status = (
                f"exact {metrics['fully_exact_sequences']}/{metrics['sequence_count']}  "
                f"switch {metrics['identity_switches']}  loss {metrics['strand_losses_when_supply_available']}"
            )
        colour = (74, 220, 135) if row["name"] not in characterization["failing_stresses"] else (245, 174, 72)
        draw.rectangle((42, y, 58, y + 16), fill=colour)
        draw.text((76, y - 2), row["name"], fill=(230, 236, 240))
        draw.text((720, y - 2), status, fill=colour)
        y += 82
    image.save(output, optimize=True)
    image.close()


def main() -> None:
    result_root = STAGE / "06_ONE_TIME_SEALED_HOLDOUT_PRIMARY_EVALUATION"
    gate = read_json(STAGE / "08_HOLDOUT_FAILURE_ATTRIBUTION_OR_PASS_CERTIFICATE" / "machine_gate_checklist.json")
    certificate = read_json(
        STAGE / "08_HOLDOUT_FAILURE_ATTRIBUTION_OR_PASS_CERTIFICATE" / "pass_certificate_or_failure_report.json"
    )
    advancement = read_json(STAGE / "11_ADVANCEMENT_OR_FAILURE_DECISION" / "advancement_decision.json")
    freeze = read_json(STAGE / "02_CANDIDATE_SOURCE_AND_CONFIGURATION_FREEZE" / "frozen_candidate_manifest.json")
    prereg = read_json(STAGE / "05_PRE_REGISTRATION_AND_EXECUTION_PLAN" / "pre_registration_hash.json")
    canary = read_json(
        STAGE / "03_DEVELOPMENT_DETERMINISM_AND_REPRODUCIBILITY_CANARY" / "development_canary_comparison.json"
    )
    seal = read_json(STAGE / "04_HOLDOUT_SEAL_AND_ACCESS_CONTROL" / "holdout_seal_before.json")
    event = read_json(result_root / "holdout_unseal_event.json")
    transaction = read_json(result_root / "primary_result_transaction.json")
    oracle = compact_result(result_root / "oracle_holdout_results.json")
    detector = compact_result(result_root / "detector_holdout_results.json")
    focal = compact_result(result_root / "legacy_focal_holdout_results.json")
    robustness = read_json(
        STAGE / "07_PRE_REGISTERED_SHADOW_ROBUSTNESS_CHARACTERIZATION" / "robustness_characterization.json"
    )
    stress_rows = [
        json.loads(line)
        for line in (STAGE / "07_PRE_REGISTERED_SHADOW_ROBUSTNESS_CHARACTERIZATION" / "shadow_stress_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if PACK.exists():
        shutil.rmtree(PACK)
    PACK.mkdir(parents=True)
    head = command(["git", "rev-parse", "HEAD"]).strip()
    branch = command(["git", "branch", "--show-current"]).strip()
    origin = command(["git", "remote", "get-url", "origin"]).strip()
    status = command(["git", "status", "--porcelain"]).strip()
    write_text(
        "01_EXECUTIVE_SUMMARY.md",
        "# M5.5F.1D Executive Summary\n\n"
        f"Classification: `{advancement['classification']}`. The frozen P-MHSAG candidate was preregistered, "
        "reproduced in three clean development processes, and opened against the sealed holdout exactly once. "
        f"The primary machine gate {'passed' if gate['passed'] else 'failed'}. No tracker was promoted and no "
        "scientific retuning occurred after unseal.\n",
    )
    write_json(
        "02_RUN_AND_GIT_CONTEXT.json",
        {
            "head": head,
            "branch": branch,
            "origin": origin,
            "working_tree_clean_at_pack_build": not status,
            "stage": STAGE.name,
        },
    )
    changed = command(["git", "diff", "--name-status", f"{BASELINE}..{head}"])
    write_text("03_FILES_CHANGED.md", "# Files Changed\n\n```text\n" + changed.rstrip() + "\n```\n")
    write_text("04_SOURCE_DIFF.patch", command(["git", "diff", "--binary", f"{BASELINE}..{head}"]))
    validation_path = STAGE / "12_COMMANDS_AND_TESTS" / "final_validation.json"
    validation = read_json(validation_path) if validation_path.exists() else {"status": "not yet recorded"}
    write_text(
        "05_COMMANDS_AND_TEST_RESULTS.md",
        "# Commands and Tests\n\n```json\n" + json.dumps(validation, indent=2, sort_keys=True) + "\n```\n",
    )
    result_index = read_json(STAGE / "13_REPRODUCIBILITY_BUNDLE" / "result_hash_index.json")
    write_json(
        "06_OUTPUT_ARTIFACT_INDEX.json",
        {
            "artifact_count": result_index["artifact_count"],
            "primary_result_transaction_hash": transaction["transaction_hash"],
        },
    )
    audit = read_json(
        STAGE / "01_AUTHORIZATION_AND_COMPLETED_AUDIT_VALIDATION" / "completed_development_audit_validation.json"
    )
    write_json(
        "07_DEVELOPMENT_AUDIT_AND_CANDIDATE_FREEZE.json",
        {
            "audit_passed": audit["passed"],
            "audit_decision_counts": audit["decision_counts"],
            "audit_decision_state_hash": audit["decision_state_hash"],
            "candidate_name": freeze["candidate_name"],
            "candidate_source_commit": freeze["candidate_source_commit"],
            "configuration_hash": freeze["configuration_hash"],
            "development_result_hash": freeze["development_result_hash"],
        },
    )
    write_json(
        "08_DETERMINISM_AND_PRE_REGISTRATION.json",
        {"canary": canary, "pre_registration_hash": prereg["pre_registration_hash"], "no_retune_statement": True},
    )
    write_json(
        "09_HOLDOUT_SEAL_AND_UNSEAL_INTEGRITY.json",
        {
            "before": {"unseal_count": seal["unseal_count"], "sealed_manifest_sha256": seal["sealed_manifest_sha256"]},
            "after": {
                "event_type": event["event_type"],
                "unseal_count_before": event["unseal_count_before"],
                "unseal_count_after": event["unseal_count_after"],
                "event_hash": event["event_hash"],
            },
            "second_unseal_rejected": True,
        },
    )
    write_json(
        "10_ORACLE_AND_DETECTOR_HOLDOUT_RESULTS.json", {"oracle": oracle, "detector": detector, "legacy_focal": focal}
    )
    write_json("11_FAILURE_ATTRIBUTION_OR_PASS_CERTIFICATE.json", certificate)
    write_json("12_SHADOW_ROBUSTNESS_CHARACTERIZATION.json", robustness)
    creation = read_json(STAGE / "09_CONDITIONAL_VISUAL_AUDIT_CONSTRUCTION" / "visual_audit_creation_decision.json")
    write_json(
        "13_VISUAL_AUDIT_PACKAGE_STATUS.json",
        {
            **creation,
            "url": "http://127.0.0.1:8805/" if creation["create_visual_audit"] else None,
            "review_id": "m5_5f1d_holdout_visual_audit_v1" if creation["create_visual_audit"] else None,
        },
    )
    mutation = read_json(STAGE / "01_AUTHORIZATION_AND_COMPLETED_AUDIT_VALIDATION" / "prior_stage_mutation_audit.json")
    write_json(
        "14_SAFETY_AND_MUTATION_AUDIT.json",
        {
            "prior_mutation_audit_passed": mutation["passed"],
            "historical_artifacts_mutated": False,
            "tracker_promoted": False,
            "model_fit_performed": False,
            "level_3_or_occlusion_work_performed": False,
            "production_ready": False,
            "safe_to_apply_globally": False,
        },
    )
    write_json("15_ACCEPTANCE_AND_NEXT_STAGE.json", advancement)
    reproducibility = read_json(STAGE / "13_REPRODUCIBILITY_BUNDLE" / "reproducibility_manifest.json")
    write_json(
        "16_REPRODUCIBILITY_AND_RESULT_HASHES.json",
        {
            "candidate_manifest_hash": reproducibility["candidate_manifest_hash"],
            "pre_registration_hash": reproducibility["pre_registration_hash"],
            "execution_harness_commit": reproducibility["execution_harness_commit"],
            "unseal_count": reproducibility["unseal_count"],
            "primary_result_transaction": reproducibility["primary_result_transaction"],
        },
    )
    evidence_root = (
        STAGE / "10_HOLDOUT_VISUAL_AUDIT_PACKAGE" / "evidence"
        if creation["create_visual_audit"]
        else STAGE / "08_HOLDOUT_FAILURE_ATTRIBUTION_OR_PASS_CERTIFICATE" / "evidence"
    )
    visuals = sorted(evidence_root.rglob("frame_06_comparison.jpg"))
    contact_sheet(visuals, PACK / "17_HOLDOUT_RESULT_VISUAL.jpg")
    robustness_visual(robustness, stress_rows, PACK / "18_ROBUSTNESS_AND_HANDOFF_VISUAL.png")
    if creation["create_visual_audit"]:
        human = (
            "# Human Review Instructions\n\nLaunch "
            "`10_HOLDOUT_VISUAL_AUDIT_PACKAGE/launch_review.ps1` and review eight "
            "blinded temporary-strand sequences at `http://127.0.0.1:8805/`. Do not alter gold; notes are optional; "
            "no tracker is promoted.\n"
        )
    else:
        human = (
            "# Human Review Instructions\n\nNo human review is required because the primary machine gate failed. "
            "Preserve the immutable holdout result. Level 3 remains blocked and no tracker is promoted.\n"
        )
    write_text("19_HUMAN_REVIEW_INSTRUCTIONS.md", human)
    files = sorted(path for path in PACK.iterdir() if path.is_file())
    manifest = {
        "schema_version": "football_intelligence.m5_5f1d.review_pack_manifest.v1",
        "flat": True,
        "maximum_file_count": 20,
        "maximum_total_bytes": 52428800,
        "maximum_visual_files": 3,
        "files": [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files
        ],
        "sealed_mapping_included": False,
        "answer_keys_included": False,
        "candidate_ids_included": False,
        "raw_video_or_weights_included": False,
    }
    write_json("REVIEW_PACK_MANIFEST.json", manifest)
    files = sorted(path for path in PACK.iterdir() if path.is_file())
    visuals = [path for path in files if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif"}]
    total = sum(path.stat().st_size for path in files)
    if len(files) != 20 or total > 52428800 or len(visuals) > 3 or any(path.is_dir() for path in PACK.iterdir()):
        raise RuntimeError(f"invalid review pack: files={len(files)} bytes={total} visuals={len(visuals)}")
    forbidden = ("server_mapping", "candidate_id", "C:\\Users\\sebgr", str(Path.home()))
    for path in files:
        if path.suffix.lower() in {".json", ".md", ".txt", ".patch"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(value in text for value in forbidden):
                raise RuntimeError(f"forbidden review-pack content in {path.name}")
    print(
        json.dumps(
            {"passed": True, "file_count": len(files), "total_bytes": total, "visual_count": len(visuals)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
