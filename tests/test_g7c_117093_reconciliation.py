import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
WORKSPACE = ROOT / r"experiments\football_observation_reasoner\part 5\G7C_DATASET_INVENTORY_AND_SPLIT_v1"


def test_117093_manifest_correction_and_source_hash() -> None:
    manifest_dir = ROOT / r"matches\117093\manifests"
    manifest = json.loads((manifest_dir / "match_manifest.json").read_text())
    hashes = json.loads((manifest_dir / "source_file_hashes.json").read_text())
    event = json.loads((manifest_dir / "source_correction_events.json").read_text())[0]
    corrected = ROOT / Path(event["new_source_path"])
    assert event["event_type"] == "AUTHORIZED_PRE_FREEZE_SOURCE_CORRECTION"
    assert event["new_hash"] == hashlib.sha256(corrected.read_bytes()).hexdigest()
    assert manifest["representative_source"]["timestamp_seconds"] == 1.0
    assert hashes[event["new_source_path"]] == event["new_hash"]
    assert len(hashes) == manifest["source_file_count"]


def test_117093_panel_and_split_preservation() -> None:
    sheet = Image.open(WORKSPACE / r"05_CONDITION_REVIEW\ten_match_contact_sheet.png")
    provenance = json.loads((WORKSPACE / r"05_CONDITION_REVIEW\contact_sheet_provenance.json").read_text())
    split = json.loads((ROOT / r"datasets\soccertrack_v2\splits\split_v1\proposed_split.json").read_text())
    assert sheet.size == (1280, 1250)
    assert provenance["117093"]["relative_path"].endswith("117093_panorama_1st_half-008.mp4")
    assert provenance["117093"]["timestamp_seconds"] == 1.0
    assert split["status"] == "PROVISIONAL_PENDING_HUMAN_APPROVAL"
    assert split["frozen"] is False
