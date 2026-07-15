from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image

from football_intelligence.replay.m5_5b_repaired_reviews_stage import (
    BASELINE_COMMIT,
    MANDATORY_REVIEW_PACK_FILES,
    authorization_audit,
    validate_m5_5b_review_pack,
    _review_pack_rows_without_sensitive_ids,
)


def _write_text(path: Path, text: str = "ok\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    _write_text(path, json.dumps(payload, sort_keys=True))


def _write_animated_gif(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for offset in (0, 40):
        image = Image.new("RGB", (320, 220), (20 + offset, 80, 120))
        pixels = image.load()
        for x in range(image.width):
            for y in range(image.height):
                pixels[x, y] = ((x + offset) % 255, (y * 2) % 255, (x + y + offset) % 255)
        frames.append(image)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=120, loop=0)


def _write_review_pack(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for filename in sorted(MANDATORY_REVIEW_PACK_FILES):
        if filename.endswith(".json"):
            _write_json(root / filename, {"safe": True})
        elif filename.endswith(".jsonl"):
            _write_text(root / filename, json.dumps({"case_number": "008", "safe": True}) + "\n")
        else:
            _write_text(root / filename, "safe\n")
    Image.new("RGB", (360, 220), (30, 80, 120)).save(root / "17_PRIMARY_VISUAL_EVIDENCE.jpg", quality=92)
    _write_animated_gif(root / "18_SECONDARY_VISUAL_EVIDENCE.gif")


def test_authorization_audit_accepts_clean_descendant_and_reports_dirty(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    _write_text(repo / "tracked.txt", "one\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True, capture_output=True)
    baseline = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    _write_text(repo / "second.txt", "two\n")
    subprocess.run(["git", "add", "second.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "second"], cwd=repo, check=True, capture_output=True)

    clean = authorization_audit(repo, baseline_commit=baseline)
    _write_text(repo / "dirty.txt", "dirty\n")
    dirty = authorization_audit(repo, baseline_commit=baseline)

    assert clean["baseline_commit_exists"] is True
    assert clean["baseline_is_ancestor_of_head"] is True
    assert clean["worktree_clean_at_builder_run"] is True
    assert dirty["baseline_is_ancestor_of_head"] is True
    assert dirty["worktree_clean_at_builder_run"] is False


def test_m5_5b_review_pack_validator_requires_mandatory_flat_animated_visuals(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    _write_review_pack(pack)

    result = validate_m5_5b_review_pack(pack)

    assert result["passed"] is True, result["errors"]
    assert result["file_count"] == 20
    assert result["visual_file_count"] == 2


def test_m5_5b_review_pack_validator_rejects_sensitive_answer_fields(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    _write_review_pack(pack)
    _write_json(pack / "13_SEQUENCE_RESOLVER_RESULTS.json", {"candidate_id": "m5_4h1_pc_secret"})

    result = validate_m5_5b_review_pack(pack)

    assert result["passed"] is False
    assert any("candidate_id" in error for error in result["errors"])


def test_review_pack_case_rows_remove_sealed_and_canonical_identifiers() -> None:
    rows = _review_pack_rows_without_sensitive_ids(
        [
            {
                "case_id": "m5_5a_occlusion_path_case_008",
                "case_number": "008",
                "human_chosen_anonymous_target": "case_008_target_02",
                "target_candidate_id": "m5_4h1_pc_secret",
                "visible_person_base_id": "m5_4h1_vpb_secret",
                "human_decision": "PATH_A_CONTINUES_SOURCE",
                "correct_path_in_top2": True,
                "review_escalation": True,
            }
        ]
    )

    text = json.dumps(rows)
    assert "candidate_id" not in text
    assert "visible_person_base_id" not in text
    assert "m5_4h1" not in text
    assert rows[0]["case_number"] == "008"


def test_repository_baseline_constant_matches_prompt() -> None:
    assert BASELINE_COMMIT == "1bc576b21da6039d1c262c004e78a22a6d33cd72"
