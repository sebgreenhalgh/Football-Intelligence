"""Build the G7E-B R3 frame-bound reviewer without changing frozen evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
R2 = PART7 / "G7E_B_R2_FULL_TEMPORAL_CANDIDATE_CLOSURE_AND_REVIEWER_REPAIR_v1"
B0 = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1"
STAGE = PART7 / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1"
SOURCE = R2 / "06_REVIEWER_REPAIR/temporal_reviewer_r2"
PACKAGE = STAGE / "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3"
REVISION = "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_V1"


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


def unique_index() -> dict[str, dict[str, Any]]:
    path = R2 / "01_UNIQUE_FRAME_INDEX/unique_temporal_frame_index.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 1044 or len({row["unique_frame_id"] for row in rows}) != 1044:
        raise RuntimeError("FAIL_G7E_B_R3_FRAME_BINDING: unique-frame closure changed")
    return {row["unique_frame_id"]: row for row in rows}


def identity(case: dict[str, Any], sequence: int, index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    frame = case["frames"][sequence]
    state = case["per_frame_candidate_states"][sequence]
    unique = index[state["unique_frame_id"]]
    if (
        frame["frame_reference_id"] not in unique["frame_reference_ids"]
        or frame["source_frame_pixel_sha256"] != unique["frame_pixel_sha256"]
        or state["frame_pixel_sha256"] != unique["frame_pixel_sha256"]
    ):
        raise RuntimeError("FAIL_G7E_B_R3_FRAME_BINDING: frame provenance mismatch")
    return {
        "burst_id": case["burst_id"],
        "frame_id": frame["frame_reference_id"],
        "unique_frame_id": state["unique_frame_id"],
        "frame_index": unique["frame_index_zero_based"],
        "frame_pixel_sha256": unique["frame_pixel_sha256"],
    }


def transform_cases(source: Path, index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload = read_json(source)
    payload["schema_version"] = "football_intelligence.g7e_b_r3.review_cases.v1"
    payload["review_revision"] = REVISION
    references = 0
    for case in payload["cases"]:
        case["schema_version"] = "football_intelligence.g7e_b_r3.review_case.v1"
        case["review_revision"] = REVISION
        case["canonical_frame_binding_required"] = True
        for sequence, frame in enumerate(case["frames"]):
            canonical = identity(case, sequence, index)
            frame["canonical_frame_identity"] = canonical
            case["per_frame_candidate_states"][sequence]["canonical_frame_identity"] = canonical
            references += 1
    expected = 27 if source.name == "practice_cases.json" else 1080
    if references != expected:
        raise RuntimeError(f"FAIL_G7E_B_R3_FRAME_BINDING: expected {expected} references, got {references}")
    return payload


def schemas() -> tuple[dict[str, Any], dict[str, Any]]:
    identity_fields = ["burst_id", "frame_id", "unique_frame_id", "frame_index", "frame_pixel_sha256"]
    event = read_json(SOURCE / "reviewer_event_schema.json")
    event.update(
        {
            "$id": "football_intelligence.g7e_b_r3.burst_annotation_event.v1",
            "title": "G7E-B R3 canonically frame-bound temporal event",
            "canonical_frame_identity_fields": identity_fields,
            "final_save_preflight_required": True,
            "idempotent_event_then_acknowledgement": True,
        }
    )
    event["properties"]["review_revision"] = {"const": REVISION}
    draft = read_json(SOURCE / "reviewer_draft_schema.json")
    draft.update(
        {
            "$id": "football_intelligence.g7e_b_r3.temporal_review_draft.v1",
            "title": "G7E-B R3 optimistic canonically frame-bound temporal draft",
            "canonical_frame_identity_fields": identity_fields,
            "optimistic_concurrency_required": True,
            "silent_migration_forbidden": True,
        }
    )
    draft["properties"]["review_revision"] = {"const": REVISION}
    return event, draft


def build() -> None:
    index = unique_index()
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
    write_json(PACKAGE / "review_cases.json", transform_cases(SOURCE / "review_cases.json", index))
    write_json(PACKAGE / "practice_cases.json", transform_cases(SOURCE / "practice_cases.json", index))
    states = read_json(SOURCE / "candidate_states_by_reference.json")
    states["review_revision"] = REVISION
    by_frame_reference = {
        frame_id: row for unique in index.values() for frame_id in unique["frame_reference_ids"] for row in [unique]
    }
    for frame_id, state in states["frames"].items():
        unique = by_frame_reference[frame_id]
        state["canonical_frame_provenance"] = {
            "frame_id": frame_id,
            "unique_frame_id": unique["unique_frame_id"],
            "frame_index": unique["frame_index_zero_based"],
            "frame_pixel_sha256": unique["frame_pixel_sha256"],
        }
    write_json(PACKAGE / "candidate_states_by_reference.json", states)
    event_schema, draft_schema = schemas()
    write_json(PACKAGE / "reviewer_event_schema.json", event_schema)
    write_json(PACKAGE / "reviewer_draft_schema.json", draft_schema)
    branch = read_json(SOURCE / "reviewer_branch_contract.json")
    branch.update(
        {
            "review_revision": REVISION,
            "canonical_frame_binding_required": True,
            "click_transaction_atomicity_required": True,
            "optimistic_draft_locking_required": True,
            "final_preflight_required": True,
            "idempotency_required": True,
        }
    )
    write_json(PACKAGE / "reviewer_branch_contract.json", branch)
    for name in ("candidate_state_fixtures.json", "tranche_manifest.jsonl", "review_server.py"):
        shutil.copyfile(SOURCE / name, PACKAGE / name)
    html = (REPO / "src/football_intelligence/g7e_b_r2_temporal_review.html").read_text(encoding="utf-8")
    html = html.replace('<html lang="en">', f'<html lang="en" data-review-revision="{REVISION}">', 1)
    html = html.replace("exact frame-local candidates R2", "frame-bound atomic save R3")
    html = html.replace(
        "R2 REVIEWER PREVIEW — NO HUMAN DECISION",
        "R3 REVIEWER PREVIEW — NO HUMAN TRUTH",
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
    css += (
        "\n.targeted-correction-button{border:2px solid #efb544;background:#fff9e8;"
        "color:#18213c;font-weight:800;padding:14px;border-radius:14px;width:100%;}\n"
    )
    (PACKAGE / "review.css").write_text(css, encoding="utf-8", newline="\n")
    instructions = (
        "# G7E-B R3 temporal practice review\n\n"
        "Launch `launch_temporal_burst_review_r3.ps1`, then open http://127.0.0.1:8818/. "
        "Resume practice only. Your completed answers and coordinates were preserved; only their already-proven "
        "frame identity metadata was repaired. Frame clicks now wait for a hash-bound server draft acknowledgement "
        "before navigation. Final Save checks bindings, writes one immutable event, then writes and validates its "
        "acknowledgement. Do not begin real Tranche 1 in this repair stage.\n"
    )
    (PACKAGE / "HUMAN_REVIEW_INSTRUCTIONS.md").write_text(instructions, encoding="utf-8", newline="\n")
    launcher = STAGE / "launch_temporal_burst_review_r3.ps1"
    launcher.write_text(
        "$ErrorActionPreference = \"Stop\"\n"
        f'$Repo = "{REPO}"\n'
        '$Python = Join-Path $Repo ".venv\\Scripts\\python.exe"\n'
        f'& $Python "{PACKAGE / "review_server.py"}" --package "{PACKAGE}" '
        f'--asset-root "{B0 / "03_TEMPORAL_REVIEWER/assets"}" '
        f'--decisions-root "{PACKAGE / "human_decisions"}" '
        f'--practice-root "{B0 / "03_TEMPORAL_REVIEWER/practice_decisions"}" --port 8818\n',
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        STAGE / "03_FRAME_BINDING_IMPLEMENTATION/frame_binding_implementation.json",
        {
            "schema_version": "football_intelligence.g7e_b_r3.frame_binding_implementation.v1",
            "review_revision": REVISION,
            "canonical_frame_identity_fields": [
                "burst_id",
                "frame_id",
                "unique_frame_id",
                "frame_index",
                "frame_pixel_sha256",
            ],
            "real_frame_references_bound": 1080,
            "practice_frame_references_bound": 27,
            "unique_frames_reused": 1044,
            "r2_source_hashes": source_hashes,
            "candidate_inference_rerun": False,
            "team_or_identity_added": False,
            "production_ready": False,
            "passed": True,
        },
    )
    write_json(
        STAGE / "03_FRAME_BINDING_IMPLEMENTATION/reviewer_package_manifest.json",
        {
            "review_revision": REVISION,
            "files": [
                {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256(path)}
                for path in sorted(PACKAGE.iterdir())
                if path.is_file()
            ],
            "production_ready": False,
        },
    )
    print("PASS_G7E_B_R3_FRAME_BOUND_REVIEWER_BUILT")


if __name__ == "__main__":
    build()
