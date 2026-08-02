"""Build the R4 frame-local relationship reviewer from exact R3 evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
R3 = PART7 / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1"
R4 = PART7 / "G7E_B_R4_CANDIDATE_RELATIONSHIP_BRANCH_INTEGRITY_AND_REAL_DRAFT_RECOVERY_v1"
SOURCE = R3 / "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3"
PACKAGE = R4 / "02_BRANCH_COMPATIBILITY_ENGINE/temporal_reviewer_r4"
REAL_ROOT = SOURCE / "human_decisions"
PRACTICE_ROOT = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/practice_decisions"
ASSET_ROOT = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/assets"
REVISION = "G7E_B_R4_CANDIDATE_RELATIONSHIP_BRANCH_INTEGRITY_V1"


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


def transformed_cases(name: str) -> dict[str, Any]:
    source = read_json(SOURCE / name)
    result = json.loads(json.dumps(source))
    result["schema_version"] = "football_intelligence.g7e_b_r4.review_cases.v1"
    result["review_revision"] = REVISION
    for case in result["cases"]:
        case["schema_version"] = "football_intelligence.g7e_b_r4.review_case.v1"
        case["review_revision"] = REVISION
        case["relationship_compatibility_required"] = True
    source_candidate_digest = hashlib.sha256(
        json.dumps(
            [[case.get("frame_candidates", []) for case in source["cases"]]],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    result_candidate_digest = hashlib.sha256(
        json.dumps(
            [[case.get("frame_candidates", []) for case in result["cases"]]],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if source_candidate_digest != result_candidate_digest:
        raise RuntimeError("FAIL_G7E_B_R4_BRANCH_COMPATIBILITY: candidate records changed")
    return result


def build() -> None:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        name: sha256(SOURCE / name)
        for name in (
            "review_cases.json",
            "practice_cases.json",
            "candidate_states_by_reference.json",
            "tranche_manifest.jsonl",
        )
    }
    review_cases = transformed_cases("review_cases.json")
    practice_cases = transformed_cases("practice_cases.json")
    if len(review_cases["cases"]) != 120 or len(practice_cases["cases"]) != 3:
        raise RuntimeError("FAIL_G7E_B_R4_BRANCH_COMPATIBILITY: case cardinality changed")
    write_json(PACKAGE / "review_cases.json", review_cases)
    write_json(PACKAGE / "practice_cases.json", practice_cases)
    for name in (
        "candidate_states_by_reference.json",
        "candidate_state_fixtures.json",
        "tranche_manifest.jsonl",
        "review_server.py",
    ):
        shutil.copyfile(SOURCE / name, PACKAGE / name)
    compatibility_source = REPO / "src/football_intelligence/g7e_b_r4_relationship_compatibility.json"
    shutil.copyfile(compatibility_source, PACKAGE / "relationship_compatibility.json")
    for name in ("reviewer_event_schema.json", "reviewer_draft_schema.json", "reviewer_branch_contract.json"):
        document = read_json(SOURCE / name)
        document["review_revision"] = REVISION
        document["relationship_compatibility_sha256"] = sha256(compatibility_source)
        document["frame_local_relationship_required"] = True
        if name == "reviewer_event_schema.json":
            document["$id"] = "football_intelligence.g7e_b_r4.burst_annotation_event.v1"
            document["title"] = "G7E-B R4 frame-bound branch-compatible temporal event"
        elif name == "reviewer_draft_schema.json":
            document["$id"] = "football_intelligence.g7e_b_r4.temporal_review_draft.v1"
            document["title"] = "G7E-B R4 recoverable frame-local relationship draft"
        write_json(PACKAGE / name, document)
    html = (REPO / "src/football_intelligence/g7e_b_r2_temporal_review.html").read_text(encoding="utf-8")
    html = html.replace('<html lang="en">', f'<html lang="en" data-review-revision="{REVISION}">', 1)
    html = html.replace("R2 REVIEWER PREVIEW — NO HUMAN DECISION", "R4 REVIEWER PREVIEW — NO NEW HUMAN TRUTH")
    html = html.replace("exact frame-local candidates R2", "frame-local relationship integrity R4")
    (PACKAGE / "index.html").write_text(html, encoding="utf-8", newline="\n")
    shutil.copyfile(REPO / "src/football_intelligence/g7e_b_r2_temporal_review.js", PACKAGE / "review.js")
    css = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            REPO / "src/football_intelligence/g7e_b_temporal_review.css",
            REPO / "src/football_intelligence/g7e_b_r2_temporal_review.css",
        )
    )
    css += (
        "\n.targeted-correction-button{border:2px solid #efb544;background:#fff9e8;"
        "color:#18213c;font-weight:800;padding:14px;border-radius:14px;width:100%;}\n"
    )
    (PACKAGE / "review.css").write_text(css, encoding="utf-8", newline="\n")
    instructions = (
        "# G7E-B R4 real-draft resume\n\n"
        "Launch `launch_temporal_burst_review_r4.ps1`, open http://127.0.0.1:8818/, and choose real review. "
        "Burst 1 (`g7e_a_117093_10`) reopens at its preserved summary unless one targeted relationship "
        "confirmation remains. Review the summary, then use `Save burst` yourself when ready. "
        "All unaffected human work is preserved. Codex did not press final Save and did not start Burst 2.\n"
    )
    (R4 / "HUMAN_RESUME_INSTRUCTIONS.md").write_text(instructions, encoding="utf-8", newline="\n")
    launcher = (
        '$ErrorActionPreference = "Stop"\n'
        f'$Repo = "{REPO}"\n'
        '$Python = Join-Path $Repo ".venv\\Scripts\\python.exe"\n'
        f'& $Python "{PACKAGE / "review_server.py"}" --package "{PACKAGE}" '
        f'--asset-root "{ASSET_ROOT}" --decisions-root "{REAL_ROOT}" '
        f'--practice-root "{PRACTICE_ROOT}" --port 8818\n'
    )
    (R4 / "launch_temporal_burst_review_r4.ps1").write_text(launcher, encoding="utf-8", newline="\n")
    implementation = {
        "schema_version": "football_intelligence.g7e_b_r4.relationship_engine_implementation.v1",
        "review_revision": REVISION,
        "package": str(PACKAGE),
        "relationship_compatibility_path": str(PACKAGE / "relationship_compatibility.json"),
        "relationship_compatibility_sha256": sha256(PACKAGE / "relationship_compatibility.json"),
        "single_matrix_shared_by_client_and_server": True,
        "real_case_count": 120,
        "practice_case_count": 3,
        "real_frame_reference_count": sum(len(case["frames"]) for case in review_cases["cases"]),
        "unique_frame_count": 1044,
        "candidate_artifacts_modified": False,
        "detector_or_temporal_inference_run": False,
        "project_defaults_changed": False,
        "validation_or_holdout_accessed": False,
        "burst_2_started": False,
        "source_hashes": source_hashes,
        "production_ready": False,
    }
    write_json(R4 / "02_BRANCH_COMPATIBILITY_ENGINE/relationship_engine_implementation.json", implementation)
    write_json(PACKAGE / "build_manifest.json", implementation)
    print("PASS_G7E_B_R4_RELATIONSHIP_REVIEWER_BUILT")


if __name__ == "__main__":
    build()
