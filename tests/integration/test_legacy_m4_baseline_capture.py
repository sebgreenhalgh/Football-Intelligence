from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from football_intelligence.cli.app import app  # noqa: E402
from football_intelligence.core.config import SafetyConfig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_fake_legacy_m4(artifact_root: Path) -> Path:
    safety = SafetyConfig().model_dump(mode="json")
    legacy = artifact_root / "matches/128058/calibration/step2_visual_continuity/step2m4_sparse_handoff_package"
    m3t = artifact_root / "matches/128058/calibration/step2_visual_continuity/step2m3t_sparse_pathlets"
    summary = {
        "m4_handoff_pathlet_count": 795,
        "m4_handoff_edge_count": 7393,
        "overlay_asset_count": 857,
        "pathlets_over_cap": 0,
        "duplicate_frame_pathlets": 0,
        "branch_merge_pathlets": 0,
        "forbidden_keys_present": [],
        **safety,
    }
    validation = {"forbidden_keys_present": [], "gate_checks": {}, **summary}
    guardrail = {"forbidden_keys_present": [], **safety}
    handoff = {"artifact": "step2m4_handoff_manifest", **summary}
    freeze = {"artifact": "step2m4_freeze_candidate_manifest", **summary}
    pathlets = {"artifact": "step2m4_sparse_handoff_pathlets", "rows": [{"m4_handoff_pathlet_id": "p1"}]}
    decisions = {"artifact": "step2m3t_reviewed_sparse_pathlet_decisions", "rows": [{"pathlet_id": "p1"}]}
    write_json(legacy / "step2m4_sparse_handoff_summary.json", summary)
    write_json(legacy / "step2m4_validation_summary.json", validation)
    write_json(legacy / "step2m4_safety_guardrail_audit.json", guardrail)
    write_json(legacy / "step2m4_handoff_manifest.json", handoff)
    write_json(legacy / "step2m4_freeze_candidate_manifest.json", freeze)
    write_json(legacy / "step2m4_sparse_handoff_pathlets.json", pathlets)
    write_json(m3t / "step2m3t_reviewed_sparse_pathlet_decisions.json", decisions)
    with gzip.open(legacy / "step2m4_sparse_handoff_edges.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"edge_id": "e1"}) + "\n")
    media = legacy / "step2m4_pathlet_overlay_strips/sample.jpg"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"stable media-like artifact")
    return legacy


def test_cli_captures_and_validates_legacy_m4_baseline(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifact_root"
    (artifact_root / "matches").mkdir(parents=True)
    legacy = write_fake_legacy_m4(artifact_root)
    config = REPO_ROOT / "configs/pipeline/visual_only_v1.yaml"
    runner = CliRunner()
    args = [
        "baseline",
        "capture",
        "--config",
        str(config),
        "--repo-root",
        str(REPO_ROOT),
        "--artifact-root",
        str(artifact_root),
        "--legacy-m4-root",
        str(legacy),
        "--allow-missing-ffprobe",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    run_dir = Path(result.output.strip())
    assert run_dir.is_relative_to(artifact_root)
    assert "SoccerTrack-v2" not in run_dir.as_posix()
    validate = runner.invoke(
        app,
        [
            "baseline",
            "validate",
            "--run-dir",
            str(run_dir),
            "--repo-root",
            str(REPO_ROOT),
            "--artifact-root",
            str(artifact_root),
        ],
    )
    assert validate.exit_code == 0, validate.output
