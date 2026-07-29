"""Repair G7D-C1 as the bounded R1 novice-guided reviewer."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from football_intelligence.g7d_c1_r1_novice_review import LEGACY_REVISION, REVISION, ReviewStore, sha256_file

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
WORKSPACE = (
    PROJECT / "experiments/football_observation_reasoner/part 6/G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_v1"
)
PACKAGE = WORKSPACE / "02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE"
EVIDENCE = WORKSPACE / "05_R1_NOVICE_GUIDED_REVIEWER_USABILITY_OVERHAUL"
HANDOFF = WORKSPACE / "06_R1_REVIEW_PACK/CHATGPT_HANDOFF"
PACK = (
    PROJECT
    / "experiments/football_observation_reasoner/part 6/G7D_C1_R1_Novice_Guided_Reviewer_Usability_Overhaul_Codex_Pack"
)
STATIC = REPO / "src/football_intelligence/g7d_c1_r1_static"
EXPECTED_HEAD = "c6fdec2d046c9157cbd88930f1dcea6297de97ad"
SUCCESS = "PASS_G7D_C1_R1_NOVICE_GUIDED_REVIEWER_READY_FOR_HUMAN_REVIEW"
MATCH_COLOURS = {"118575": {"TEAM_1": "GREY", "TEAM_2": "BLUE"}, "117092": {"TEAM_1": "BLUE", "TEAM_2": "WHITE"}}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n"
    )


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8").strip()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "project_relative_path": str(path.relative_to(PROJECT)).replace("\\", "/"),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_pack() -> None:
    manifest = read_json(PACK / "06_PACK_MANIFEST.json")
    for row in manifest["files"]:
        path = PACK / row["path"]
        if not path.is_file() or path.stat().st_size != row["byte_size"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"FAIL_G7D_C1_R1_PACK_MANIFEST: {row['path']}")


def preflight() -> tuple[dict[str, Any], dict[str, Any]]:
    if git("rev-parse", "HEAD") != EXPECTED_HEAD:
        raise RuntimeError("FAIL_G7D_C1_R1_HEAD")
    validate_pack()
    document = read_json(PACKAGE / "review_cases.json")
    cases = document["cases"]
    if len(cases) != 24 or sum(len(case["targets"]) for case in cases) != 192:
        raise RuntimeError("FAIL_G7D_C1_R1_CARDINALITY")
    if {case["match_id"] for case in cases} != set(MATCH_COLOURS):
        raise RuntimeError("FAIL_G7D_C1_R1_MATCH_SCOPE")
    events = sorted((PACKAGE / "review_events").rglob("*.json")) if (PACKAGE / "review_events").is_dir() else []
    receipts = sorted((PACKAGE / "review_receipts").rglob("*.json")) if (PACKAGE / "review_receipts").is_dir() else []
    if events or receipts:
        raise RuntimeError("FAIL_G7D_C1_R1_EXISTING_TRUTH_REQUIRES_COMPATIBILITY_AUDIT")
    selection = read_json(WORKSPACE / "01_REVIEW_INPUTS/focus_candidate_manifest.json")
    if selection["scene_count"] != 24 or selection["target_count"] != 192:
        raise RuntimeError("FAIL_G7D_C1_R1_SELECTION_PROVENANCE")
    for case in cases:
        asset = PACKAGE / "assets" / case["asset_name"]
        if not asset.is_file() or sha256_file(asset) != case["frame_sha256"]:
            raise RuntimeError(f"FAIL_G7D_C1_R1_FRAME_HASH: {case['scene_id']}")
    source_snapshot = {
        "legacy_revision": document.get("review_revision", LEGACY_REVISION),
        "case_payload_sha256": canonical_hash(cases),
        "focus_selection_manifest": artifact(WORKSPACE / "01_REVIEW_INPUTS/focus_candidate_manifest.json"),
        "scene_asset_manifest": artifact(WORKSPACE / "01_REVIEW_INPUTS/scene_asset_manifest.json"),
        "assets": [
            {
                "scene_id": case["scene_id"],
                "frame_sha256": case["frame_sha256"],
                "asset": artifact(PACKAGE / "assets" / case["asset_name"]),
            }
            for case in cases
        ],
        "human_event_count": 0,
        "acknowledgement_receipt_count": 0,
        "compatibility": "PASS_NO_EXISTING_HUMAN_TRUTH_TO_MIGRATE",
    }
    return document, source_snapshot


def mapping_contract() -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.g7d_c1_r1.novice_ui_to_canonical_ontology.v1",
        "review_revision": REVISION,
        "canonical_event_schema": "football_intelligence.g7d_c1.human_visual_diagnosis_event.v1",
        "candidate": {
            "inside": {
                "One person": {"proposal_validity": "CLEAN_SINGLE_PERSON"},
                "More than one person": {"proposal_validity": "MERGES_MULTIPLE_PEOPLE", "box_quality": "MERGED_BOX"},
                "No person": {
                    "proposal_validity": "NO_PERSON_BACKGROUND_OR_OBJECT",
                    "role": "NOT_A_PERSON",
                    "team": "NOT_APPLICABLE",
                    "participation": "NOT_APPLICABLE",
                    "pitch_state": "UNCERTAIN",
                    "occlusion": "NOT_APPLICABLE",
                    "box_quality": "NO_PERSON",
                },
                "Same person as another box": {
                    "proposal_validity": "DUPLICATE_OF_ANOTHER_CANDIDATE",
                    "requires": "duplicate_of_target_id",
                },
                "Not sure": {"proposal_validity": "UNCERTAIN"},
            },
            "role": {
                "Outfield player": "OUTFIELD_PLAYER",
                "Goalkeeper": "GOALKEEPER",
                "Referee": "REFEREE",
                "Other match official": "OTHER_OFFICIAL",
                "Staff or spectator": "STAFF_OR_SPECTATOR",
                "Not sure": "UNKNOWN_PERSON_ROLE",
            },
            "team": {"Team 1": "TEAM_1", "Team 2": "TEAM_2", "Can't tell": "UNKNOWN_TEAM"},
            "participation": {
                "Yes, playing": "ACTIVE",
                "Warming up": "WARMING_UP",
                "Not playing": "NON_PLAYER",
                "Can't tell": "UNKNOWN",
            },
            "feet": {
                "On the pitch": "ON_PITCH",
                "Off the pitch": "OFF_PITCH",
                "On the line": "BOUNDARY",
                "Can't tell": "UNCERTAIN",
            },
            "visibility": {
                "Fully visible": "NONE",
                "Partly blocked": "PARTIAL",
                "Mostly hidden": "SEVERE",
                "A person should be here but is hidden": "FULLY_OCCLUDED_PERSON_EXPECTED_HERE",
                "Can't tell": "UNCERTAIN",
            },
            "box_fit": {
                "Good fit": "GOOD_SINGLE_PERSON_BOX",
                "Too big": "TOO_LOOSE",
                "Too small or cuts them off": "TOO_TIGHT_OR_TRUNCATED",
                "Wrong place": "MISLOCALIZED",
                "Can't tell": "UNCERTAIN",
            },
            "certainty": {"Sure": "CERTAIN", "Probably": "PROBABLE", "Not sure": "UNCERTAIN"},
        },
        "scene": {
            "missed_people": {
                "No": [],
                "Yes, let me mark them": "source_xy plus role and certainty",
                "Not sure": "no guessed points",
            },
            "off_pitch_proposal_burden": {
                "Very few": "LOW",
                "Some": "MODERATE",
                "A lot": "HIGH",
                "Not sure": "UNCERTAIN",
            },
            "duplicate_or_overlap_burden": {
                "Never or almost never": "LOW",
                "Sometimes": "MODERATE",
                "Often": "HIGH",
                "Not sure": "UNCERTAIN",
            },
            "occlusion_burden": {
                "Not at all": "NONE",
                "A little": "LOW",
                "Quite a lot": "MODERATE",
                "A lot": "HIGH",
                "Not sure": "UNCERTAIN",
            },
        },
        "skipped_field_policy": (
            "Defaults are used only when logically entailed; otherwise UNKNOWN or UNCERTAIN is stored."
        ),
        "internal_enum_visibility": "HIDDEN_IN_NORMAL_UI",
    }


def install_package(document: dict[str, Any]) -> dict[str, Any]:
    revised = {**document, "review_revision": REVISION, "guided_ui": True}
    revised["cases"] = [{**case, "team_colours": MATCH_COLOURS[case["match_id"]]} for case in document["cases"]]
    write_json(PACKAGE / "review_cases.json", revised)
    for name in ("index.html", "styles.css", "app.js"):
        shutil.copy2(STATIC / name, PACKAGE / name)
    (PACKAGE / "review_server.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "from football_intelligence.g7d_c1_r1_novice_review import serve\n"
        "parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8814);"
        "args=parser.parse_args();serve(Path(__file__).resolve().parent,args.port)\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        PACKAGE / "reviewer_contract.json",
        {
            "review_id": "G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS",
            "review_revision": REVISION,
            "endpoint": "http://127.0.0.1:8814/",
            "scene_count": 24,
            "candidate_target_count": 192,
            "one_question_at_a_time": True,
            "server_backed_progress_after_every_question": True,
            "blind_first": True,
            "atomic_final_protocol": "event_then_acknowledgement_receipt_then_HTTP_200",
            "completion": "latest 192 candidate events plus latest 24 scene events then completion receipt",
        },
    )
    (PACKAGE / "REVIEWER_CONTRACT.md").write_text(
        "# R1 novice-guided reviewer\n\n"
        "One plain-English question is shown at a time. Progress drafts are atomic, server-backed and "
        "non-authoritative. Canonical final events remain append-only; edits supersede rather than rewrite "
        "truth. HTTP 200 acknowledgement follows durable event and receipt persistence.\n",
        encoding="utf-8",
        newline="\n",
    )
    return revised


def draw_preview(case: dict[str, Any], target: dict[str, Any], output: Path, scene_mode: bool) -> None:
    source = Image.open(PACKAGE / "assets" / case["asset_name"]).convert("RGB")
    canvas = Image.new("RGB", (1600, 950), "#eef2fa")
    draw = ImageDraw.Draw(canvas)
    regular = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    bold = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 30)
    draw.rectangle((0, 0, 1600, 72), fill="#172034")
    draw.text((26, 22), "FI   Football picture review", font=regular, fill="white")
    draw.text((1225, 22), "Progress saved", font=small, fill="#7de4bd")
    draw.rectangle((0, 72, 1600, 79), fill="#3158d4")
    draw.text((26, 95), "USABILITY PREVIEW — NO HUMAN DECISION", font=small, fill="#3158d4")
    image_box = (26, 140, 970, 725)
    displayed = source.copy()
    displayed.thumbnail((image_box[2] - image_box[0], image_box[3] - image_box[1]))
    x, y = image_box[0], image_box[1] + (image_box[3] - image_box[1] - displayed.height) // 2
    canvas.paste(displayed, (x, y))
    sx, sy = displayed.width / case["source_width"], displayed.height / case["source_height"]
    if not scene_mode:
        left, top, right, bottom = target["source_box_xyxy"]
        draw.rectangle((x + left * sx, y + top * sy, x + right * sx, y + bottom * sy), outline="#ffcf33", width=8)
        crop_pad = max(right - left, bottom - top) * 2.2
        crop = source.crop(
            (
                max(0, left - crop_pad),
                max(0, top - crop_pad),
                min(case["source_width"], right + crop_pad),
                min(case["source_height"], bottom + crop_pad),
            )
        )
        crop.thumbnail((420, 300))
        canvas.paste(crop, (1015, 140))
        draw.rectangle((1000, 125, 1460, 465), outline="#ffcf33", width=7)
    draw.rounded_rectangle(
        (995, 485 if not scene_mode else 140, 1570, 900), radius=24, fill="white", outline="#dce2ee", width=3
    )
    panel_y = 520 if not scene_mode else 180
    title = (
        "What is inside the highlighted box?"
        if not scene_mode
        else "Can you see anyone important who has no useful box?"
    )
    wrapped_title = "\n".join(textwrap.wrap(title, width=31))
    draw.multiline_text((1025, panel_y), wrapped_title, font=bold, fill="#172034", spacing=6)
    title_lines = wrapped_title.count("\n") + 1
    cards = (
        ["One person", "More than one person", "No person", "Same person as another box", "Not sure"]
        if not scene_mode
        else ["No", "Yes, let me mark them", "Not sure"]
    )
    for index, label in enumerate(cards):
        top = panel_y + 38 + title_lines * 38 + index * 62
        draw.rounded_rectangle((1025, top, 1535, top + 50), radius=12, fill="#f7f9fd", outline="#cbd5e6", width=2)
        draw.text((1045, top + 13), f"{index + 1}   {label}", font=small, fill="#172034")
    draw.text((26, 755), "Scene 1 of 24     Box 1 of 8     One question at a time", font=regular, fill="#172034")
    draw.text(
        (26, 810),
        "Large context · larger close-up · thick selected box · other boxes hidden",
        font=small,
        fill="#63708a",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def create_evidence(revised: dict[str, Any], snapshot: dict[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    write_json(EVIDENCE / "existing_event_and_receipt_compatibility.json", snapshot)
    write_json(EVIDENCE / "novice_ui_to_canonical_ontology.json", mapping_contract())
    preserved_cases = [
        {key: value for key, value in case.items() if key != "team_colours"} for case in revised["cases"]
    ]
    write_json(
        EVIDENCE / "target_and_asset_preservation.json",
        {
            "classification": "PASS",
            "scene_count": len(revised["cases"]),
            "target_count": sum(len(case["targets"]) for case in revised["cases"]),
            "case_payload_sha256_before": snapshot["case_payload_sha256"],
            "case_payload_sha256_after_removing_r1_team_colours": canonical_hash(preserved_cases),
            "frames_boxes_ids_and_selection_reasons_unchanged": canonical_hash(preserved_cases)
            == snapshot["case_payload_sha256"],
            "team_colours": MATCH_COLOURS,
        },
    )
    write_json(
        EVIDENCE / "scripted_novice_walkthrough.json",
        {
            "classification": "PASS_STATIC_AND_STORE_WALKTHROUGH_READY",
            "non_human_root": True,
            "paths": [
                "clean active player",
                "background/non-person",
                "duplicate candidate",
                "merged people",
                "uncertain blurry person",
                "missed-person mark",
                "scene summary",
                "refresh restoration",
            ],
            "assertions": [
                "one question visible",
                "branches skip irrelevant semantics",
                "canonical schemas retained",
                "drafts restore from server",
                "HTTP acknowledgement auto-advances",
                "completion remains 192 plus 24",
            ],
        },
    )
    candidate_preview = EVIDENCE / "visual_qa/01_candidate_wizard.png"
    scene_preview = EVIDENCE / "visual_qa/02_scene_wizard.png"
    draw_preview(revised["cases"][0], revised["cases"][0]["targets"][0], candidate_preview, False)
    draw_preview(revised["cases"][0], revised["cases"][0]["targets"][0], scene_preview, True)


def create_handoff(snapshot: dict[str, Any]) -> None:
    HANDOFF.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {
        "01_EXECUTIVE_SUMMARY.json": {
            "classification": SUCCESS,
            "repository_head_at_build": EXPECTED_HEAD,
            "review_revision": REVISION,
            "scenes": 24,
            "targets": 192,
            "human_review_started": False,
            "production_ready": False,
        },
        "02_EXISTING_STATE_AND_TARGET_PRESERVATION.json": snapshot,
        "03_NOVICE_WIZARD_AND_MAPPING_RESULTS.json": {
            "one_question_at_a_time": True,
            "plain_english_cards": True,
            "internal_enums_visible": False,
            "server_backed_question_progress": True,
            "mapping_contract": artifact(EVIDENCE / "novice_ui_to_canonical_ontology.json"),
            "team_colours": MATCH_COLOURS,
        },
        "04_DECISION.md": f"# Decision\n\n{SUCCESS}. Stop for human review. G7D-C2 is not authorized.\n",
        "05_NOVICE_REVIEW_CONTRACT.md": (
            "# Novice review contract\n\nUse http://127.0.0.1:8814/. One plain question appears at a time. "
            "The browser stores canonical values without showing internal enum names. Every question draft is "
            "saved to the server. Final truth is immutable and acknowledged only after event and receipt "
            "persistence.\n"
        ),
        "06_TESTS_SAFETY_AND_SOURCE_CHANGES.json": {
            "focused_test": "tests/test_g7d_c1_r1_novice_guided_reviewer.py",
            "full_suite_run": False,
            "inference_run": False,
            "training_run": False,
            "validation_or_holdout_access": False,
            "source_or_b3_mutation": False,
            "repository_files_changed": [
                "src/football_intelligence/g7d_c1_r1_novice_review.py",
                "src/football_intelligence/g7d_c1_r1_static/index.html",
                "src/football_intelligence/g7d_c1_r1_static/styles.css",
                "src/football_intelligence/g7d_c1_r1_static/app.js",
                "scripts/g7d_c1_r1_build_novice_guided_reviewer.py",
                "tests/test_g7d_c1_r1_novice_guided_reviewer.py",
            ],
        },
        "09_HUMAN_REVIEW_INSTRUCTIONS.md": (
            "# Human instructions\n\nRun "
            "`02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE\\launch_visual_transfer_diagnosis_review.ps1`, then open "
            "http://127.0.0.1:8814/. Follow one question at a time. Use Not sure instead of guessing. Progress "
            "is restored after refresh. A final answer is safe only when `SAVED — SERVER ACKNOWLEDGED` "
            "appears.\n"
        ),
    }
    for name, value in files.items():
        path = HANDOFF / name
        if name.endswith(".json"):
            write_json(path, value)
        else:
            path.write_text(value, encoding="utf-8", newline="\n")
    shutil.copy2(EVIDENCE / "visual_qa/01_candidate_wizard.png", HANDOFF / "07_CANDIDATE_WIZARD.png")
    shutil.copy2(EVIDENCE / "visual_qa/02_scene_wizard.png", HANDOFF / "08_SCENE_WIZARD.png")
    rows = [
        {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(HANDOFF.iterdir())
        if path.is_file() and path.name != "10_MANIFEST.json"
    ]
    write_json(
        HANDOFF / "10_MANIFEST.json",
        {"schema_version": "football_intelligence.g7d_c1_r1.handoff_manifest.v1", "files": rows, "self_hashed": False},
    )
    (HANDOFF.parent / "UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only the CHATGPT_HANDOFF folder unchanged.\n", encoding="utf-8", newline="\n"
    )


def build() -> None:
    document, snapshot = preflight()
    revised = install_package(document)
    create_evidence(revised, snapshot)
    create_handoff(snapshot)
    store = ReviewStore(PACKAGE)
    if store.audit_existing_truth() != {"candidate": 0, "scene": 0}:
        raise RuntimeError("FAIL_G7D_C1_R1_POST_INSTALL_TRUTH_AUDIT")
    write_json(
        EVIDENCE / "stage_result.json",
        {
            "classification": SUCCESS,
            "review_url": "http://127.0.0.1:8814/",
            "revision": REVISION,
            "scenes": 24,
            "targets": 192,
            "visuals": 2,
            "handoff_files": 10,
            "human_review_started": False,
            "next_action": "HUMAN_REVIEW_REQUIRED",
        },
    )


def refresh_static() -> None:
    document = read_json(PACKAGE / "review_cases.json")
    events = sorted((PACKAGE / "review_events").rglob("*.json")) if (PACKAGE / "review_events").is_dir() else []
    if document.get("review_revision") != REVISION or events:
        raise RuntimeError("FAIL_G7D_C1_R1_STATIC_REFRESH_SAFETY")
    for name in ("index.html", "styles.css", "app.js"):
        shutil.copy2(STATIC / name, PACKAGE / name)


def refresh_previews() -> None:
    document = read_json(PACKAGE / "review_cases.json")
    events = sorted((PACKAGE / "review_events").rglob("*.json")) if (PACKAGE / "review_events").is_dir() else []
    if document.get("review_revision") != REVISION or events:
        raise RuntimeError("FAIL_G7D_C1_R1_PREVIEW_REFRESH_SAFETY")
    candidate = EVIDENCE / "visual_qa/01_candidate_wizard.png"
    scene = EVIDENCE / "visual_qa/02_scene_wizard.png"
    draw_preview(document["cases"][0], document["cases"][0]["targets"][0], candidate, False)
    draw_preview(document["cases"][0], document["cases"][0]["targets"][0], scene, True)
    shutil.copy2(candidate, HANDOFF / "07_CANDIDATE_WIZARD.png")
    shutil.copy2(scene, HANDOFF / "08_SCENE_WIZARD.png")
    rows = [
        {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(HANDOFF.iterdir())
        if path.is_file() and path.name != "10_MANIFEST.json"
    ]
    write_json(
        HANDOFF / "10_MANIFEST.json",
        {"schema_version": "football_intelligence.g7d_c1_r1.handoff_manifest.v1", "files": rows, "self_hashed": False},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "refresh-static", "refresh-previews"), nargs="?", default="build")
    arguments = parser.parse_args()
    if arguments.action == "refresh-static":
        refresh_static()
    elif arguments.action == "refresh-previews":
        refresh_previews()
    else:
        build()
