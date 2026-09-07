from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import g7f_c_run_calibration_adjudication_reaudit as reaudit
import g7f_c_run_dg005_sequence2_reviewer as sequence2_runner


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fake_git(values: dict[tuple[str, ...], str]):
    def run(_repo: Path, *args: str) -> str:
        return values[args]

    return run


def repository_values(*, head: str, changed: str = "", count: str = "0") -> dict[tuple[str, ...], str]:
    r2 = reaudit.R2_REVIEWER_COMMIT
    baseline = reaudit.PHASE_B_COMPATIBILITY_COMMIT
    return {
        ("rev-parse", "HEAD"): head,
        ("rev-parse", "origin/main"): head,
        ("status", "--porcelain"): "",
        ("merge-base", r2, head): r2,
        ("merge-base", baseline, head): baseline,
        ("diff", "--name-only", baseline, head): changed,
        ("rev-list", "--count", f"{baseline}..{head}"): count,
    }


def test_phase_b_compatibility_baseline_without_sequence2_release_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        reaudit,
        "git",
        fake_git(repository_values(head=reaudit.PHASE_B_COMPATIBILITY_COMMIT)),
    )
    with pytest.raises(RuntimeError, match="exact sequence-2 release"):
        reaudit.verify_repository_compatibility(tmp_path, reaudit.R2_REVIEWER_COMMIT)


def test_unrelated_repository_drift_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    head = "f" * 40
    values = repository_values(head=head, changed="docs/unrelated.md", count="1")
    monkeypatch.setattr(reaudit, "git", fake_git(values))
    with pytest.raises(RuntimeError, match="not the exact sequence-2 release"):
        reaudit.verify_repository_compatibility(tmp_path, reaudit.R2_REVIEWER_COMMIT)


def test_exact_sequence2_release_drift_is_accepted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert reaudit.ALLOWED_SEQUENCE2_RELEASE_PATHS == sequence2_runner.EXPECTED_RELEASE_PATHS
    head = "e" * 40
    changed = "\n".join(sorted(reaudit.ALLOWED_SEQUENCE2_RELEASE_PATHS))
    monkeypatch.setattr(reaudit, "git", fake_git(repository_values(head=head, changed=changed, count="1")))
    result = reaudit.verify_repository_compatibility(tmp_path, reaudit.R2_REVIEWER_COMMIT)
    assert result["commits_above_phase_b_compatibility"] == 1
    assert result["changed_paths_above_phase_b_compatibility"] == sorted(reaudit.ALLOWED_SEQUENCE2_RELEASE_PATHS)


def test_r2_manifest_and_stable_phase_a_truth_bindings_are_verified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    released_file = tmp_path / "released.py"
    released_file.write_text("accepted R2 bytes\n", encoding="utf-8")
    truth = {"repository_commit": "548f1d17", "selection_manifest_sha256": "a" * 64}
    paths = {
        "r2_stage": tmp_path / "r2",
        "phase_a": tmp_path / "phase_a",
    }
    manifest_path = paths["r2_stage"] / "05_REVIEWER/reviewer_release_manifest.json"
    manifest = {
        "reviewer_release": reaudit.CLOSURE_REVIEWER_RELEASE,
        "required_baseline": reaudit.R2_BASELINE,
        "repository_commit": reaudit.R2_REVIEWER_COMMIT,
        "files": [
            {"path": str(released_file), "byte_size": released_file.stat().st_size, "sha256": sha256(released_file)}
        ],
    }
    write_json(manifest_path, manifest)
    monkeypatch.setattr(reaudit, "R2_RELEASE_MANIFEST_SHA256", sha256(manifest_path))
    write_json(
        paths["r2_stage"] / "05_REVIEWER/reviewer_binding_hashes.json",
        {
            "adjudication_truth_bindings": truth,
            "reviewer_release_manifest_sha256": sha256(manifest_path),
        },
    )
    write_json(paths["phase_a"] / "05_REVIEWER/reviewer_binding_hashes.json", truth)
    release, bindings, closure = reaudit.verify_r2_release(paths)
    assert release["reviewer_release"] == reaudit.CLOSURE_REVIEWER_RELEASE
    assert bindings == truth
    assert closure == {
        "closure_reviewer_release": reaudit.CLOSURE_REVIEWER_RELEASE,
        "closure_reviewer_release_manifest_sha256": sha256(manifest_path),
        "closure_repository_commit": reaudit.R2_REVIEWER_COMMIT,
    }


@pytest.mark.parametrize(
    "releases",
    [
        [reaudit.R1_REVIEWER_RELEASE] * 6,
        [reaudit.CLOSURE_REVIEWER_RELEASE] * 6,
        [reaudit.R1_REVIEWER_RELEASE] + [reaudit.CLOSURE_REVIEWER_RELEASE] * 5,
    ],
)
def test_r1_r2_authoritative_event_lineage_is_valid(releases: list[str]) -> None:
    rows = [
        {
            "anonymous_dense_image_id": image_id,
            "event": {"reviewer_release": release},
        }
        for image_id, release in zip(reaudit.CALIBRATION_IDS, releases, strict=True)
    ]
    by_image, observed = reaudit.validate_event_reviewer_releases(rows)
    assert by_image == dict(zip(reaudit.CALIBRATION_IDS, releases, strict=True))
    assert observed == sorted(set(releases))


def test_targeted_release_is_valid_only_for_dg005_sequence2() -> None:
    rows = [
        {
            "anonymous_dense_image_id": image_id,
            "event": {
                "reviewer_release": (
                    reaudit.SEQUENCE2_REVIEWER_RELEASE if image_id == "DG-005" else reaudit.CLOSURE_REVIEWER_RELEASE
                ),
                "adjudication_sequence": 2 if image_id == "DG-005" else 1,
            },
        }
        for image_id in reaudit.CALIBRATION_IDS
    ]
    by_image, observed = reaudit.validate_event_reviewer_releases(rows)
    assert by_image["DG-005"] == reaudit.SEQUENCE2_REVIEWER_RELEASE
    assert reaudit.SEQUENCE2_REVIEWER_RELEASE in observed
    rows[0]["event"] = {
        "reviewer_release": reaudit.SEQUENCE2_REVIEWER_RELEASE,
        "adjudication_sequence": 2,
    }
    with pytest.raises(RuntimeError, match="only on DG-005"):
        reaudit.validate_event_reviewer_releases(rows)


def test_phase_b_output_stays_in_original_adjudication_stage(tmp_path: Path) -> None:
    paths = reaudit.phase_b_paths(tmp_path / "SoccerTrack-v2")
    assert paths["stage"] == paths["phase_a"]
    assert paths["stage"].name == "G7F_C_CALIBRATION_SUPERSEDING_ADJUDICATION_AND_REAUDIT_v1"
    assert paths["stage"] != paths["r2_stage"]
    assert paths["sequence2_stage"].name == "G7F_C_DG005_SEQUENCE2_RELEVANCE_ADJUDICATION_AND_PHASE_B_CLOSURE_v1"


def test_status_requires_dg005_sequence2_then_reports_seven_pairs_and_six_authoritative() -> None:
    base = {
        "missing": [],
        "rows": [{"anonymous_dense_image_id": image_id} for image_id in reaudit.CALIBRATION_IDS],
        "all_rows": [{} for _ in range(6)],
        "authoritative_sequences": {image_id: 1 for image_id in reaudit.CALIBRATION_IDS},
        "authoritative_event_reviewer_releases": {},
        "adjudication_reviewer_releases_observed": [],
        "closure_release": {},
        "sequence2_release": {},
        "dg005_sequence2_complete": False,
    }
    blocked = reaudit.status_result(base)
    assert blocked["decision"] == reaudit.SEQUENCE2_INCOMPLETE
    assert blocked["valid_event_ack_pairs"] == 6
    complete = {
        **base,
        "all_rows": [{} for _ in range(7)],
        "authoritative_sequences": {**base["authoritative_sequences"], "DG-005": 2},
        "dg005_sequence2_complete": True,
    }
    ready = reaudit.status_result(complete)
    assert ready["decision"] == reaudit.VISUAL_REQUIRED
    assert ready["valid_event_ack_pairs"] == 7
    assert ready["authoritative_adjudications"] == 6


def test_temp_decisions_remain_byte_identical_and_source_assets_are_candidate_free(tmp_path: Path) -> None:
    decisions = tmp_path / "TEMP_decisions"
    write_json(decisions / "events/event.json", {"human": True})
    before = reaudit.decisions_inventory(decisions)
    for image_id in reaudit.CALIBRATION_IDS:
        path = reaudit.source_image_path(tmp_path / "source", image_id)
        assert path == tmp_path / "source/05_REVIEWER/assets" / f"{image_id}.png"
        assert "candidate" not in path.as_posix().lower()
    assert reaudit.decisions_inventory(decisions) == before
