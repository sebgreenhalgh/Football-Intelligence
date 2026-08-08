"""Build the hash-bound R6.3 reviewer from the existing R6.2 package builder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
from types import ModuleType
from typing import Any

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART8 = PROJECT / "experiments/football_observation_reasoner/part 8"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
STAGE = PART8 / "G7E_B_R6_3_FAST_ACTION_AND_STALE_DRAFT_RECOVERY_v1"
PACKAGE = STAGE / "03_FAST_ACTION_IMPLEMENTATION/temporal_reviewer_r6_3"
RELEASE_REVISION = "G7E_B_R6_3_FAST_ACTION_AND_STALE_DRAFT_RECOVERY_V1"
REAL_ROOT = PART7 / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1/03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
PRACTICE_ROOT = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/practice_decisions"
ASSET_ROOT = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/assets"


def load_previous_builder() -> ModuleType:
    path = REPO / "scripts/g7e_b_r6_2_build_precision_reviewer.py"
    specification = importlib.util.spec_from_file_location("g7e_b_r6_3_previous_builder", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("R6.2 package builder could not be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.STAGE = STAGE
    module.PACKAGE = PACKAGE
    module.RELEASE_REVISION = RELEASE_REVISION
    return module


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_hashes() -> dict[str, dict[str, int | str]]:
    return {
        path.relative_to(PACKAGE).as_posix(): {"sha256": sha256(path), "byte_size": path.stat().st_size}
        for path in sorted(PACKAGE.rglob("*"))
        if path.is_file() and path.name != "package_manifest.json"
    }


def main() -> None:
    builder = load_previous_builder()
    builder.main()
    for legacy in ("03_VISUAL_REPAIR_IMPLEMENTATION", "03_PRECISION_NAVIGATION_IMPLEMENTATION"):
        path = STAGE / legacy
        if path.exists() and path.resolve() != PACKAGE.resolve():
            shutil.rmtree(path)

    generated = PACKAGE / "generated_client_contract.js"
    with generated.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f'window.__G7E_B_R6_3_RELEASE_REVISION__ = "{RELEASE_REVISION}";\n')

    build = read_json(PACKAGE / "build_manifest.json")
    build.update(
        {
            "schema_version": "football_intelligence.g7e_b_r6_3.reviewer_build.v1",
            "classification": "PASS_G7E_B_R6_3_FAST_REVIEWER_BUILT",
            "release_revision": RELEASE_REVISION,
            "package": str(PACKAGE),
            "browser_bundle_sha256": sha256(PACKAGE / "review.js"),
            "viewport_transform_sha256": sha256(PACKAGE / "viewport_transform.js"),
            "css_sha256": sha256(PACKAGE / "review.css"),
            "html_sha256": sha256(PACKAGE / "index.html"),
            "production_ready": False,
        }
    )
    write_json(PACKAGE / "build_manifest.json", build)
    write_json(STAGE / "03_FAST_ACTION_IMPLEMENTATION/reviewer_build.json", build)
    write_json(
        PACKAGE / "package_manifest.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6_3.package_manifest.v1",
            "files": package_hashes(),
            "self_hash_excluded": True,
            "production_ready": False,
        },
    )

    launcher = (
        '$ErrorActionPreference = "Stop"\n'
        f'$Repo = "{REPO}"\n'
        '$Python = Join-Path $Repo ".venv\\Scripts\\python.exe"\n'
        f'& $Python "{PACKAGE / "review_server.py"}" --package "{PACKAGE}" '
        f'--asset-root "{ASSET_ROOT}" --decisions-root "{REAL_ROOT}" '
        f'--practice-root "{PRACTICE_ROOT}" --port 8818\n'
    )
    (STAGE / "launch_temporal_burst_review_r6_3.ps1").write_text(launcher, encoding="utf-8", newline="\n")
    (STAGE / "HUMAN_RESUME_INSTRUCTIONS.md").write_text(
        "# Resume Tranche 1 safely\n\n"
        "Run `launch_temporal_burst_review_r6_3.ps1`, open http://127.0.0.1:8818/, and confirm the restored "
        "burst and question before answering. Stale browser state is restored from the server and the rejected "
        "click is never replayed. `production_ready=false`.\n",
        encoding="utf-8",
        newline="\n",
    )
    print("PASS_G7E_B_R6_3_FAST_REVIEWER_BUILT")


if __name__ == "__main__":
    main()
