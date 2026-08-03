"""Build the R5 canonical-state reviewer from exact R4 corpus assets."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
R4 = PART7 / "G7E_B_R4_CANDIDATE_RELATIONSHIP_BRANCH_INTEGRITY_AND_REAL_DRAFT_RECOVERY_v1"
R5 = PART7 / "G7E_B_R5_REVIEWER_STATE_MACHINE_AND_FULL_CORPUS_STABILIZATION_v1"
SOURCE = R4 / "02_BRANCH_COMPATIBILITY_ENGINE/temporal_reviewer_r4"
PACKAGE = R5 / "02_CANONICAL_STATE_CONTRACT/temporal_reviewer_r5"
REAL_ROOT = (
    R4.parent
    / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1"
    / "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
)
PRACTICE_ROOT = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/practice_decisions"
ASSET_ROOT = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/assets"
REVISION = "G7E_B_R5_REVIEWER_STATE_MACHINE_V1"


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
    source_digest = hashlib.sha256(
        json.dumps(
            [(case["burst_id"], case["frames"], case["frame_candidates"]) for case in document["cases"]],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    document["schema_version"] = "football_intelligence.g7e_b_r5.review_cases.v1"
    document["review_revision"] = REVISION
    for case in document["cases"]:
        case["schema_version"] = "football_intelligence.g7e_b_r5.review_case.v1"
        case["review_revision"] = REVISION
        case["canonical_state_contract_required"] = True
    result_digest = hashlib.sha256(
        json.dumps(
            [(case["burst_id"], case["frames"], case["frame_candidates"]) for case in document["cases"]],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if source_digest != result_digest:
        raise RuntimeError("R5 build changed an immutable case, frame, or candidate record")
    return document


def build() -> None:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    contract_source = REPO / "src/football_intelligence/g7e_b_r5_canonical_reviewer_state_contract.json"
    contract = read_json(contract_source)
    contract_hash = sha256(contract_source)
    write_json(PACKAGE / "review_cases.json", transform_cases("review_cases.json"))
    write_json(PACKAGE / "practice_cases.json", transform_cases("practice_cases.json"))
    for name in (
        "candidate_states_by_reference.json",
        "candidate_state_fixtures.json",
        "tranche_manifest.jsonl",
        "review_server.py",
    ):
        shutil.copyfile(SOURCE / name, PACKAGE / name)
    shutil.copyfile(contract_source, PACKAGE / "canonical_reviewer_state_contract.json")
    generated = (
        '"use strict";\n'
        f"window.__G7E_B_R5_CANONICAL_CONTRACT__ = {json.dumps(contract, sort_keys=True, separators=(',', ':'))};\n"
        f'window.__G7E_B_R5_CANONICAL_CONTRACT_SHA256__ = "{contract_hash}";\n'
    )
    (PACKAGE / "generated_client_contract.js").write_text(generated, encoding="utf-8", newline="\n")
    html = (REPO / "src/football_intelligence/g7e_b_r2_temporal_review.html").read_text(encoding="utf-8")
    html = html.replace('<html lang="en">', f'<html lang="en" data-review-revision="{REVISION}">', 1)
    html = html.replace(
        "R2 REVIEWER PREVIEW — NO HUMAN DECISION",
        "R5 RELEASE-CANDIDATE PREVIEW — NO NEW HUMAN TRUTH",
    )
    html = html.replace(
        '<script src="/review.js"></script>',
        '<script src="/generated_client_contract.js"></script>\n  <script src="/review.js"></script>',
    )
    (PACKAGE / "index.html").write_text(html, encoding="utf-8", newline="\n")
    shutil.copyfile(REPO / "src/football_intelligence/g7e_b_r2_temporal_review.js", PACKAGE / "review.js")
    css = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            REPO / "src/football_intelligence/g7e_b_temporal_review.css",
            REPO / "src/football_intelligence/g7e_b_r2_temporal_review.css",
        )
    )
    css += "\n.release-gate{border:2px solid #2cc9a0;border-radius:12px;padding:10px;font-weight:800;}\n"
    (PACKAGE / "review.css").write_text(css, encoding="utf-8", newline="\n")
    launcher = (
        '$ErrorActionPreference = "Stop"\n'
        f'$Repo = "{REPO}"\n'
        '$Python = Join-Path $Repo ".venv\\Scripts\\python.exe"\n'
        f'& $Python "{PACKAGE / "review_server.py"}" --package "{PACKAGE}" '
        f'--asset-root "{ASSET_ROOT}" --decisions-root "{REAL_ROOT}" '
        f'--practice-root "{PRACTICE_ROOT}" --port 8818\n'
    )
    (R5 / "launch_temporal_burst_review_r5.ps1").write_text(launcher, encoding="utf-8", newline="\n")
    instructions = (
        "# R5 real review resume\n\n"
        "Launch `launch_temporal_burst_review_r5.ps1` and open http://127.0.0.1:8818/. "
        "Burst 1 remains immutable and acknowledged. Burst 2 opens at Question 1 with no answer selected. "
        "The reviewer remains locked unless the complete R5 release gate validates.\n"
    )
    (R5 / "HUMAN_RESUME_INSTRUCTIONS.md").write_text(instructions, encoding="utf-8", newline="\n")
    implementation = {
        "schema_version": "football_intelligence.g7e_b_r5.reviewer_build.v1",
        "review_revision": REVISION,
        "package": str(PACKAGE),
        "canonical_contract_sha256": contract_hash,
        "real_case_count": 120,
        "practice_case_count": 3,
        "frame_reference_count": 1080,
        "candidate_records_changed": False,
        "project_defaults_changed": False,
        "production_ready": False,
    }
    write_json(PACKAGE / "build_manifest.json", implementation)
    write_json(R5 / "02_CANONICAL_STATE_CONTRACT/canonical_state_implementation.json", implementation)
    print("PASS_G7E_B_R5_STATE_MACHINE_REVIEWER_BUILT")


if __name__ == "__main__":
    build()
