"""Build the R6 server-authoritative reviewer from the immutable R5 corpus."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
R5 = PART7 / "G7E_B_R5_REVIEWER_STATE_MACHINE_AND_FULL_CORPUS_STABILIZATION_v1"
R6 = PART7 / "G7E_B_R6_SERVER_AUTHORITATIVE_ACTION_REDUCER_AND_EXACT_BRANCH_REPAIR_v1"
SOURCE = R5 / "02_CANONICAL_STATE_CONTRACT/temporal_reviewer_r5"
PACKAGE = R6 / "03_SERVER_AUTHORITATIVE_ACTION_REDUCER/temporal_reviewer_r6"
REAL_ROOT = PART7 / (
    "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1/"
    "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
)
PRACTICE_ROOT = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/practice_decisions"
ASSET_ROOT = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/assets"
REVISION = "G7E_B_R6_SERVER_AUTHORITATIVE_ACTION_REDUCER_V1"

OUTPUT_DIRS = (
    "00_BASELINE_REAL_EVENT_AND_DRAFT_CLOSURE",
    "01_EXACT_REAL_BROWSER_PATH_REPRODUCTION",
    "02_FAILED_REAL_DRAFT_RECOVERY",
    "03_SERVER_AUTHORITATIVE_ACTION_REDUCER",
    "04_PRODUCTION_PATH_CHALLENGE_SUITE",
    "05_FULL_120_BURST_BROWSER_AUDIT",
    "06_FAULT_AND_RACE_CHALLENGE",
    "07_RELEASE_GATE_AND_REAL_STATE_ACCEPTANCE",
    "08_VISUAL_QA",
    "09_TESTS_AND_LOGS",
    "10_REVIEW_PACK",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def transform_cases(name: str) -> dict[str, Any]:
    document = read_json(SOURCE / name)
    before = hashlib.sha256(
        json.dumps(
            [(case["burst_id"], case["frames"], case["frame_candidates"]) for case in document["cases"]],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    document["schema_version"] = "football_intelligence.g7e_b_r6.review_cases.v1"
    document["review_revision"] = REVISION
    for case in document["cases"]:
        case["schema_version"] = "football_intelligence.g7e_b_r6.review_case.v1"
        case["review_revision"] = REVISION
        case["server_authoritative_action_reducer_required"] = True
    after = hashlib.sha256(
        json.dumps(
            [(case["burst_id"], case["frames"], case["frame_candidates"]) for case in document["cases"]],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if before != after:
        raise RuntimeError("R6 build altered immutable frame or candidate records")
    return document


def build() -> None:
    for relative in OUTPUT_DIRS:
        (R6 / relative).mkdir(parents=True, exist_ok=True)
    PACKAGE.mkdir(parents=True, exist_ok=True)
    canonical_source = REPO / "src/football_intelligence/g7e_b_r5_canonical_reviewer_state_contract.json"
    action_source = REPO / "src/football_intelligence/g7e_b_r6_server_action_contract.json"
    canonical = read_json(canonical_source)
    action = read_json(action_source)
    write_json(PACKAGE / "review_cases.json", transform_cases("review_cases.json"))
    write_json(PACKAGE / "practice_cases.json", transform_cases("practice_cases.json"))
    for name in (
        "candidate_states_by_reference.json",
        "candidate_state_fixtures.json",
        "tranche_manifest.jsonl",
        "review_server.py",
    ):
        shutil.copyfile(SOURCE / name, PACKAGE / name)
    shutil.copyfile(canonical_source, PACKAGE / "canonical_reviewer_state_contract.json")
    shutil.copyfile(action_source, PACKAGE / "server_action_contract.json")
    generated = (
        '"use strict";\n'
        f"window.__G7E_B_R6_CANONICAL_CONTRACT__ = {json.dumps(canonical, sort_keys=True, separators=(',', ':'))};\n"
        f'window.__G7E_B_R6_CANONICAL_CONTRACT_SHA256__ = "{sha256(canonical_source)}";\n'
        f"window.__G7E_B_R6_ACTION_CONTRACT__ = {json.dumps(action, sort_keys=True, separators=(',', ':'))};\n"
        f'window.__G7E_B_R6_ACTION_CONTRACT_SHA256__ = "{sha256(action_source)}";\n'
    )
    (PACKAGE / "generated_client_contract.js").write_text(generated, encoding="utf-8", newline="\n")
    html = (REPO / "src/football_intelligence/g7e_b_r2_temporal_review.html").read_text(encoding="utf-8")
    html = html.replace('<html lang="en">', f'<html lang="en" data-review-revision="{REVISION}">', 1)
    html = html.replace("R2 REVIEWER PREVIEW", "R6 SERVER-AUTHORITATIVE PREVIEW")
    html = html.replace(
        '<script src="/review.js"></script>',
        '<script src="/generated_client_contract.js"></script>\n  <script src="/review.js"></script>',
    )
    (PACKAGE / "index.html").write_text(html, encoding="utf-8", newline="\n")
    shutil.copyfile(REPO / "src/football_intelligence/g7e_b_r6_temporal_review.js", PACKAGE / "review.js")
    css = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            REPO / "src/football_intelligence/g7e_b_temporal_review.css",
            REPO / "src/football_intelligence/g7e_b_r2_temporal_review.css",
        )
    )
    css += (
        "\n.release-gate{border:2px solid #2cc9a0;border-radius:14px;padding:12px;"
        "background:#effcf7;color:#132039}"
        ".blocking-error[data-error-kind='image-asset']{border-color:#e76d6d}"
        ".blocking-error[data-error-kind='server-action']{border-color:#e7a51a}\n"
    )
    (PACKAGE / "review.css").write_text(css, encoding="utf-8", newline="\n")
    launcher = (
        '$ErrorActionPreference = "Stop"\n'
        f'$Repo = "{REPO}"\n'
        '$Python = Join-Path $Repo ".venv\\Scripts\\python.exe"\n'
        f'& $Python "{PACKAGE / "review_server.py"}" --package "{PACKAGE}" '
        f'--asset-root "{ASSET_ROOT}" --decisions-root "{REAL_ROOT}" '
        f'--practice-root "{PRACTICE_ROOT}" --port 8818\n'
    )
    (R6 / "launch_temporal_burst_review_r6.ps1").write_text(launcher, encoding="utf-8", newline="\n")
    (R6 / "HUMAN_RESUME_INSTRUCTIONS.md").write_text(
        "# Resume the recovered burst\n\n"
        "Run `launch_temporal_burst_review_r6.ps1`, open http://127.0.0.1:8818/, and inspect the "
        "server-verified summary. All 27 marks and human answers are preserved. No event was created by R6. "
        "Press final Save only when you choose to acknowledge the recovered burst. Do not begin the next burst "
        "in this stage.\n",
        encoding="utf-8",
        newline="\n",
    )
    build_manifest = {
        "schema_version": "football_intelligence.g7e_b_r6.reviewer_build.v1",
        "review_revision": REVISION,
        "package": str(PACKAGE),
        "canonical_contract_sha256": sha256(canonical_source),
        "server_action_contract_sha256": sha256(action_source),
        "production_browser_bundle_sha256": sha256(PACKAGE / "review.js"),
        "server_reducer_sha256": sha256(REPO / "src/football_intelligence/g7e_b_r6_action_reducer.py"),
        "real_case_count": 120,
        "practice_case_count": 3,
        "frame_reference_count": 1080,
        "candidate_records_changed": False,
        "project_defaults_changed": False,
        "production_ready": False,
    }
    write_json(PACKAGE / "build_manifest.json", build_manifest)
    write_json(R6 / "03_SERVER_AUTHORITATIVE_ACTION_REDUCER/server_action_reducer_build.json", build_manifest)
    print("PASS_G7E_B_R6_SERVER_ACTION_REVIEWER_BUILT")


if __name__ == "__main__":
    build()
