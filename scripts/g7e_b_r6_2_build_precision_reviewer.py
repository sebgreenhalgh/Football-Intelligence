"""Build the hash-bound R6.2 precision-navigation reviewer package."""

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
STAGE = PART8 / "G7E_B_R6_2_PRECISION_ZOOM_PAN_AND_COORDINATE_SAFE_MARKING_v1"
PACKAGE = STAGE / "03_PRECISION_NAVIGATION_IMPLEMENTATION/temporal_reviewer_r6_2"
RELEASE_REVISION = "G7E_B_R6_2_PRECISION_ZOOM_PAN_COORDINATE_SAFE_MARKING_V1"


def load_base() -> ModuleType:
    path = REPO / "scripts/g7e_b_r6_1_build_final_byte_reviewer.py"
    specification = importlib.util.spec_from_file_location("g7e_b_r6_2_build_base", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("R6.1 package builder could not be loaded")
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    base = load_base()
    base.main()
    legacy_evidence = STAGE / "03_VISUAL_REPAIR_IMPLEMENTATION"
    if legacy_evidence.exists():
        shutil.rmtree(legacy_evidence)
    legacy_launcher = STAGE / "launch_temporal_burst_review_r6_1.ps1"
    if legacy_launcher.exists():
        legacy_launcher.unlink()
    viewport_source = REPO / "src/football_intelligence/g7e_b_r6_2_viewport.js"
    viewport_package = PACKAGE / "viewport_transform.js"
    shutil.copyfile(viewport_source, viewport_package)
    css_source = REPO / "src/football_intelligence/g7e_b_r6_2_temporal_review.css"
    with (PACKAGE / "review.css").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write("\n" + css_source.read_text(encoding="utf-8") + "\n")
    generated = PACKAGE / "generated_client_contract.js"
    with generated.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f'window.__G7E_B_R6_2_RELEASE_REVISION__ = "{RELEASE_REVISION}";\n')

    build = read_json(PACKAGE / "build_manifest.json")
    build.update(
        {
            "schema_version": "football_intelligence.g7e_b_r6_2.reviewer_build.v1",
            "classification": "PASS_G7E_B_R6_2_PRECISION_REVIEWER_BUILT",
            "release_revision": RELEASE_REVISION,
            "package": str(PACKAGE),
            "browser_bundle_sha256": base.sha256(PACKAGE / "review.js"),
            "viewport_transform_sha256": base.sha256(viewport_package),
            "viewport_transform_source_sha256": base.sha256(viewport_source),
            "css_sha256": base.sha256(PACKAGE / "review.css"),
            "html_sha256": base.sha256(PACKAGE / "index.html"),
            "production_ready": False,
        }
    )
    write_json(PACKAGE / "build_manifest.json", build)
    write_json(STAGE / "03_PRECISION_NAVIGATION_IMPLEMENTATION/reviewer_build.json", build)
    write_json(
        PACKAGE / "package_manifest.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6_2.package_manifest.v1",
            "files": base.package_hashes(),
            "self_hash_excluded": True,
            "production_ready": False,
        },
    )

    launcher = (
        '$ErrorActionPreference = "Stop"\n'
        f'$Repo = "{REPO}"\n'
        '$Python = (Get-ChildItem "$env:APPDATA\\uv\\python\\cpython-3.12*-windows-x86_64-none\\python.exe" '
        '| Sort-Object FullName -Descending | Select-Object -First 1).FullName\n'
        'if (-not $Python) { throw "A trusted uv-managed CPython 3.12 runtime is required." }\n'
        '$env:PYTHONPATH = "$(Join-Path $Repo ".venv\\Lib\\site-packages");$(Join-Path $Repo "src")"\n'
        f'& $Python "{PACKAGE / "review_server.py"}" --package "{PACKAGE}" '
        f'--asset-root "{base.ASSET_ROOT}" --decisions-root "{base.REAL_ROOT}" '
        f'--practice-root "{base.PRACTICE_ROOT}" --port 8818\n'
    )
    (STAGE / "launch_temporal_burst_review_r6_2.ps1").write_text(launcher, encoding="utf-8", newline="\n")
    (STAGE / "HUMAN_RESUME_INSTRUCTIONS.md").write_text(
        "# Resume Tranche 1 safely\n\n"
        "Stop the prior local reviewer if it is still running, then run "
        "`launch_temporal_burst_review_r6_2.ps1` and open http://127.0.0.1:8818/. "
        "Confirm the restored burst and question before answering. Wheel zooms toward the cursor; "
        "Pan, Space + left-drag, and middle-drag move either viewer without marking. "
        "Fit/Reset restores the whole image. Navigation is display-only. `production_ready=false`.\n",
        encoding="utf-8",
        newline="\n",
    )
    print("PASS_G7E_B_R6_2_PRECISION_REVIEWER_BUILT")


if __name__ == "__main__":
    main()
