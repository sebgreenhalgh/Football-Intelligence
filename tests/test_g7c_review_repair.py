import hashlib
import json
from pathlib import Path

from PIL import Image


def _paths() -> tuple[Path, Path]:
    root = Path(r"C:\Users\sebgr\Documents\football-intelligence")
    return root, root / r"experiments\football_observation_reasoner\part 5\G7C_DATASET_INVENTORY_AND_SPLIT_v1"


def test_real_contact_sheet_and_human_form() -> None:
    _, workspace = _paths()
    sheet = workspace / "05_CONDITION_REVIEW" / "ten_match_contact_sheet.png"
    form = json.loads((workspace / "05_CONDITION_REVIEW" / "HUMAN_CONDITION_REVIEW.json").read_text())
    assert Image.open(sheet).size == (1280, 1250)
    assert len(form["matches"]) == 10
    assert all(
        not row["representative_frame_approved"] and not row["proposed_split_approved"] for row in form["matches"]
    )
    assert all(row["lighting"] == "" and row["weather"] == "" for row in form["matches"])


def test_manifest_is_nonrecursive_and_not_self_hashed() -> None:
    _, workspace = _paths()
    review = workspace / "08_REVIEW_PACK"
    manifest = json.loads((review / "07_MANIFEST.json").read_text())
    assert len(list(review.iterdir())) == 7
    assert len(manifest["files"]) == 6
    assert all(entry["filename"] != "07_MANIFEST.json" for entry in manifest["files"])
    for entry in manifest["files"]:
        path = review / entry["filename"]
        assert path.stat().st_size == entry["byte_size"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def test_split_and_source_mutation_evidence_preserved() -> None:
    root, workspace = _paths()
    split = json.loads((root / r"datasets\soccertrack_v2\splits\split_v1\proposed_split.json").read_text())
    mutation = json.loads(
        (workspace / "07_TESTS_AND_LOGS" / "source_mutation_check.json").read_text(encoding="utf-8-sig")
    )
    assert split["status"] == "PROVISIONAL_PENDING_HUMAN_APPROVAL"
    assert split["frozen"] is False
    assert split["allocation"]["TRAIN_DEVELOPMENT"][-1] == "128058"
    assert mutation["changed"] == 0
