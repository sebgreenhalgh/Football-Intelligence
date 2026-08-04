"""Build the hash-bound R6.1 reviewer and deterministic visual derivatives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from football_intelligence.temporal_reviewer.visual import build_visual_modes, write_visual_manifest

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
PART8 = PROJECT / "experiments/football_observation_reasoner/part 8"
R6 = PART7 / "G7E_B_R6_SERVER_AUTHORITATIVE_ACTION_REDUCER_AND_EXACT_BRANCH_REPAIR_v1"
R6_PACKAGE = R6 / "03_SERVER_AUTHORITATIVE_ACTION_REDUCER/temporal_reviewer_r6"
STAGE = PART8 / "G7E_B_R6_1_FINAL_BYTE_VISUAL_RUNTIME_AND_REPOSITORY_CLOSURE_v1"
PACKAGE = STAGE / "03_VISUAL_REPAIR_IMPLEMENTATION/temporal_reviewer_r6_1"
REAL_ROOT = PART7 / (
    "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1/"
    "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
)
PRACTICE_ROOT = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/practice_decisions"
ASSET_ROOT = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/assets"
RELEASE_REVISION = "G7E_B_R6_1_FINAL_BYTE_VISUAL_RUNTIME_CLOSURE_V1"


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


def frozen_digest(document: dict[str, Any]) -> str:
    rows = []
    for case in document["cases"]:
        frame_rows = [
            {
                "frame_reference_id": frame["frame_reference_id"],
                "canonical_frame_identity": frame["canonical_frame_identity"],
                "source_frame_pixel_sha256": frame["source_frame_pixel_sha256"],
                "panorama_url": frame["panorama_url"],
                "panorama_sha256": frame["panorama_sha256"],
                "focus_url": frame["focus_url"],
                "focus_sha256": frame["focus_sha256"],
                "source_width": frame["source_width"],
                "source_height": frame["source_height"],
            }
            for frame in case["frames"]
        ]
        rows.append((case["burst_id"], frame_rows, case["frame_candidates"], case["candidates"]))
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def package_hashes() -> dict[str, dict[str, int | str]]:
    result: dict[str, dict[str, int | str]] = {}
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_file() and path.name != "package_manifest.json":
            result[path.relative_to(PACKAGE).as_posix()] = {"sha256": sha256(path), "byte_size": path.stat().st_size}
    return result


def main() -> None:
    if PACKAGE.exists():
        if STAGE.resolve() not in PACKAGE.resolve().parents:
            raise RuntimeError("refusing to replace a reviewer package outside the R6.1 stage")
        shutil.rmtree(PACKAGE)
    PACKAGE.mkdir(parents=True, exist_ok=True)
    visual_root = PACKAGE / "review_assets"
    manifests: list[dict[str, Any]] = []
    frozen_before: dict[str, str] = {}
    frozen_after: dict[str, str] = {}
    for name in ("review_cases.json", "practice_cases.json"):
        document = read_json(R6_PACKAGE / name)
        frozen_before[name] = frozen_digest(document)
        transformed, rows = build_visual_modes(document, ASSET_ROOT, visual_root)
        transformed["r6_1_release_revision"] = RELEASE_REVISION
        for case in transformed["cases"]:
            case["r6_1_release_revision"] = RELEASE_REVISION
        frozen_after[name] = frozen_digest(transformed)
        if frozen_before[name] != frozen_after[name]:
            raise RuntimeError(f"R6.1 visual build altered frozen truth: {name}")
        write_json(PACKAGE / name, transformed)
        manifests.extend(rows)
    unique = {(row["kind"], row["original_sha256"]): row for row in manifests}
    write_visual_manifest(PACKAGE / "visual_asset_manifest.json", list(unique.values()))

    for name in (
        "candidate_states_by_reference.json",
        "candidate_state_fixtures.json",
        "tranche_manifest.jsonl",
        "review_server.py",
        "canonical_reviewer_state_contract.json",
    ):
        shutil.copyfile(R6_PACKAGE / name, PACKAGE / name)
    shutil.copyfile(
        REPO / "src/football_intelligence/g7e_b_r6_server_action_contract.json",
        PACKAGE / "server_action_contract.json",
    )
    source_snapshots = (
        "temporal_review.py",
        "g7e_b_r6_action_reducer.py",
        "g7e_b_r5_reviewer_state.py",
    )
    for name in source_snapshots:
        destination = PACKAGE / "runtime_source_snapshot/football_intelligence" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / "src/football_intelligence" / name, destination)
    for source in sorted((REPO / "src/football_intelligence/temporal_reviewer").glob("*.py")):
        destination = PACKAGE / "runtime_source_snapshot/football_intelligence/temporal_reviewer" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    canonical = read_json(PACKAGE / "canonical_reviewer_state_contract.json")
    action = read_json(PACKAGE / "server_action_contract.json")
    generated = (
        '"use strict";\n'
        f"window.__G7E_B_R6_CANONICAL_CONTRACT__ = {json.dumps(canonical, sort_keys=True, separators=(',', ':'))};\n"
        "window.__G7E_B_R6_CANONICAL_CONTRACT_SHA256__ = "
        f'"{sha256(PACKAGE / "canonical_reviewer_state_contract.json")}";\n'
        f"window.__G7E_B_R6_ACTION_CONTRACT__ = {json.dumps(action, sort_keys=True, separators=(',', ':'))};\n"
        f'window.__G7E_B_R6_ACTION_CONTRACT_SHA256__ = "{sha256(PACKAGE / "server_action_contract.json")}";\n'
        f'window.__G7E_B_R6_1_RELEASE_REVISION__ = "{RELEASE_REVISION}";\n'
    )
    (PACKAGE / "generated_client_contract.js").write_text(generated, encoding="utf-8", newline="\n")
    html = (REPO / "src/football_intelligence/g7e_b_r2_temporal_review.html").read_text(encoding="utf-8")
    html = html.replace('<html lang="en">', f'<html lang="en" data-release-revision="{RELEASE_REVISION}">', 1)
    html = html.replace("R2 REVIEWER PREVIEW", "R6.1 FINAL-BYTE REVIEW PREVIEW")
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
        "background:#effcf7;color:#132039}.blocking-error[data-error-kind='image-asset']{border-color:#e76d6d}"
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
    (STAGE / "launch_temporal_burst_review_r6_1.ps1").write_text(launcher, encoding="utf-8", newline="\n")
    (STAGE / "HUMAN_RESUME_INSTRUCTIONS.md").write_text(
        "# Resume Tranche 1 safely\n\n"
        "Run `launch_temporal_burst_review_r6_1.ps1`, open http://127.0.0.1:8818/, and confirm the restored "
        "burst and question. Auto, Original, and Enhanced change review appearance only. They do not change "
        "source truth, geometry, coordinates, or answers. Continue only when you choose. "
        "`production_ready=false`.\n",
        encoding="utf-8",
        newline="\n",
    )
    build = {
        "schema_version": "football_intelligence.g7e_b_r6_1.reviewer_build.v1",
        "release_revision": RELEASE_REVISION,
        "review_protocol_revision": action["review_revision"],
        "package": str(PACKAGE),
        "frozen_digest_before": frozen_before,
        "frozen_digest_after": frozen_after,
        "frozen_truth_changed": False,
        "canonical_contract_sha256": sha256(PACKAGE / "canonical_reviewer_state_contract.json"),
        "action_contract_sha256": sha256(PACKAGE / "server_action_contract.json"),
        "browser_bundle_sha256": sha256(PACKAGE / "review.js"),
        "css_sha256": sha256(PACKAGE / "review.css"),
        "server_module_sha256": sha256(REPO / "src/football_intelligence/temporal_review.py"),
        "reducer_sha256": sha256(REPO / "src/football_intelligence/g7e_b_r6_action_reducer.py"),
        "runtime_source_snapshot": {
            path.relative_to(PACKAGE).as_posix(): sha256(path)
            for path in sorted((PACKAGE / "runtime_source_snapshot").rglob("*.py"))
        },
        "visual_asset_manifest_sha256": sha256(PACKAGE / "visual_asset_manifest.json"),
        "cases": 120,
        "frame_references": 1080,
        "production_ready": False,
    }
    write_json(PACKAGE / "build_manifest.json", build)
    write_json(STAGE / "03_VISUAL_REPAIR_IMPLEMENTATION/reviewer_build.json", build)
    write_json(
        PACKAGE / "package_manifest.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6_1.package_manifest.v1",
            "files": package_hashes(),
            "self_hash_excluded": True,
            "production_ready": False,
        },
    )
    print("PASS_G7E_B_R6_1_REVIEWER_BUILT")


if __name__ == "__main__":
    main()
