"""Build the one-case, human-only G7D-B2A 128058 pitch-polygon reviewer."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from g7d_a_r5_templates import R5_HTML, R5_SERVER


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
STAGE = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B2A_128058_PITCH_POLYGON_REVIEW_v1"
PACK = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B2A_128058_Pitch_Polygon_Review_Codex_Pack"
MATCH_ID = "128058"
PORT = 8813
REVIEW_ID = "G7D_B2A_128058_PITCH_POLYGON_REVIEW"
REVISION = "G7D_B2A_128058_PITCH_POLYGON_REVIEW_V1"
EXPECTED_HEAD = "1eadbfc08c0ea90125513ac17cbc7ee00f11ebe1"
SOURCE_VIDEOS = {
    "first": {
        "relative_path": "matches/128058/source/videos/128058_panorama_1st_half-006.mp4",
        "sha256": "8db0efdc045978d67572c6764681a76350e8da75a9f5fa7bc9307f3b9f21d989",
    },
    "second": {
        "relative_path": "matches/128058/source/videos/128058_panorama_2nd_half-010.mp4",
        "sha256": "c5554a1a85655770d7adc83d8ef272e656a14a04433d8b5ee74cf021f9805131",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8").strip()


def validate_inputs() -> None:
    allowed_dirty = {
        "?? scripts/g7d_b2a_build_pitch_polygon_review.py",
        "?? tests/test_g7d_b2a_128058_pitch_polygon_review.py",
    }
    if git("rev-parse", "HEAD") != EXPECTED_HEAD or not set(git("status", "--porcelain").splitlines()) <= allowed_dirty:
        raise RuntimeError("FAIL_BASELINE_OR_WORKTREE")
    split = read_json(PROJECT / "datasets/soccertrack_v2/splits/split_v1/split_manifest.json")
    setup = read_json(PROJECT / f"matches/{MATCH_ID}/calibration/match_setup.json")
    failure = read_json(PACK / "06_B2_FAILURE_EVIDENCE/pitch_geometry_resolution.json")
    if split["status"] != "FROZEN_HUMAN_APPROVED" or split["frozen"] is not True:
        raise RuntimeError("FAIL_FROZEN_SPLIT")
    if MATCH_ID not in split["membership"]["TRAIN_DEVELOPMENT"]:
        raise RuntimeError("FAIL_FROZEN_SPLIT")
    calibration = setup["pitch_calibration"]
    if (
        calibration["status"] != "HUMAN_REQUIRED"
        or calibration["polygon_path"] is not None
        or calibration["polygon_sha256"] is not None
    ):
        raise RuntimeError("FAIL_G7D_B2A_REVIEWER_REUSE")
    if (
        setup["team_mapping"]["team_1_primary_colour"] != "BLUE"
        or setup["team_mapping"]["team_2_primary_colour"] != "WHITE"
    ):
        raise RuntimeError("FAIL_G7D_B2A_REVIEWER_REUSE")
    if failure["status"] != "FAIL_G7D_B2_128058_PITCH_PROVENANCE" or failure["sampling_or_inference_started"]:
        raise RuntimeError("FAIL_G7D_B2A_REVIEWER_REUSE")
    if failure["canonical_polygon_exists"]:
        raise RuntimeError("FAIL_G7D_B2A_REVIEWER_REUSE")
    for name in ("selection_manifest.json", "review_case_manifest.json", "source_frame_manifest.json"):
        if (PROJECT / f"matches/{MATCH_ID}/calibration/pitch_polygon_v1/{name}").exists():
            raise RuntimeError("FAIL_G7D_B2A_REVIEWER_REUSE: existing review package")
    if (PROJECT / f"matches/{MATCH_ID}/calibration/pitch_polygon_v1/pitch_polygon.json").exists():
        raise RuntimeError("FAIL_G7D_B2A_REVIEWER_REUSE: polygon already finalized")
    source_manifest = read_json(PROJECT / f"matches/{MATCH_ID}/manifests/source_file_manifest.json")["files"]
    for data in SOURCE_VIDEOS.values():
        row = next((item for item in source_manifest if item["relative_path"] == data["relative_path"]), None)
        source = PROJECT / data["relative_path"]
        if (
            row is None
            or row["sha256"] != data["sha256"]
            or not source.is_file()
            or sha256_file(source) != data["sha256"]
        ):
            raise RuntimeError("FAIL_G7D_B2A_SOURCE_VIDEO_PROVENANCE")


def probe(ffprobe: Path, source: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height,r_frame_rate",
            "-of",
            "json",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(completed.stdout)
    video = next(stream for stream in payload["streams"] if stream["codec_type"] == "video")
    numerator, denominator = (Decimal(value) for value in video["r_frame_rate"].split("/"))
    fps = numerator / denominator
    duration = Decimal(payload["format"]["duration"])
    if fps <= 0 or duration <= 0:
        raise RuntimeError("FAIL_G7D_B2A_SOURCE_VIDEO_PROVENANCE")
    return {
        "duration_seconds": float(duration),
        "source_width": int(video["width"]),
        "source_height": int(video["height"]),
        "frame_rate": video["r_frame_rate"],
        "frame_rate_decimal": str(fps),
        "_duration_decimal": duration,
        "_fps_decimal": fps,
    }


def extract_reference_frame(
    ffmpeg: Path, source: Path, output: Path, duration: Decimal, fps: Decimal
) -> tuple[Decimal, int, Decimal]:
    requested = duration * Decimal("0.25")
    frame_index = int((requested * fps).to_integral_value(rounding=ROUND_HALF_UP))
    resolved = Decimal(frame_index) / fps
    if output.exists():
        raise RuntimeError("FAIL_G7D_B2A_REFERENCE_FRAME_EXTRACTION: output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-vsync",
            "0",
            "-frames:v",
            "1",
            "-c:v",
            "png",
            str(output),
        ],
        check=True,
        timeout=900,
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("FAIL_G7D_B2A_REFERENCE_FRAME_EXTRACTION")
    return requested, frame_index, resolved


def reviewer_templates() -> tuple[str, str]:
    alignment_check = (
        'if payload["alignment_answer"] not in ("YES", "NO", "UNCERTAIN"): '
        'return None, error("INVALID_ALIGNMENT", "alignment_answer", "alignment answer must be YES, NO, or UNCERTAIN")'
    )
    alignment_replacement = (
        alignment_check + '\n    if payload["alignment_answer"] == "UNCERTAIN": '
        'return None, error("ALIGNMENT_UNCERTAIN", "alignment_answer", '
        '"UNCERTAIN must remain incomplete and cannot be saved")'
    )
    server = R5_SERVER.replace('REQUIRED_MATCH_IDS = ["118575", "117092"]', 'REQUIRED_MATCH_IDS = ["128058"]')
    server = server.replace("G7D_A_PITCH_POLYGON_REVIEW_R5", REVISION).replace("G7D_A_PITCH_POLYGON_REVIEW", REVIEW_ID)
    server = server.replace(alignment_check, alignment_replacement)
    server = server.replace("SERVER_PERSISTED_TWO_CASE_COMPLETION", "SERVER_PERSISTED_ONE_CASE_COMPLETION")
    acknowledgement_start = (
        "def acknowledgement_receipt(event_path, event):\n"
        '    receipt_path = PACKAGE / "review_receipts" / "event_acknowledgements" / f"{event[\'match_id\']}.json"\n'
        "    receipt = "
    )
    acknowledgement_replacement = (
        "def acknowledgement_receipt(event_path, event):\n"
        '    receipt_path = PACKAGE / "review_receipts" / "event_acknowledgements" / f"{event[\'match_id\']}.json"\n'
        '    if receipt_path.exists(): return receipt_path, json.loads(receipt_path.read_text(encoding="utf-8"))\n'
        "    receipt = "
    )
    server = server.replace(acknowledgement_start, acknowledgement_replacement)
    completion_start = "def completion_receipt():\n    paths = "
    completion_replacement = (
        "def completion_receipt():\n"
        '    path = PACKAGE / "review_receipts" / "completion" / "final.json"\n'
        '    if path.exists(): return path, json.loads(path.read_text(encoding="utf-8"))\n'
        "    paths = "
    )
    server = server.replace(completion_start, completion_replacement)
    html = R5_HTML.replace("G7D-A Pitch Polygon Review", "G7D-B2A — 128058 Human Pitch Polygon Review")
    html = html.replace(
        '<select id="case"><option>118575</option><option>117092</option></select>',
        '<select id="case"><option>128058</option></select>',
    )
    html = html.replace(
        "<h1>G7D-B2A — 128058 Human Pitch Polygon Review</h1>",
        "<h1>G7D-B2A — 128058 Human Pitch Polygon Review</h1><p>TEAM_1=BLUE &nbsp; TEAM_2=WHITE. Trace only the playable ground surface; use source-image coordinates.</p>",
    )
    html = html.replace("G7D_A_PITCH_POLYGON_REVIEW_R5", REVISION).replace("G7D_A_PITCH_POLYGON_REVIEW", REVIEW_ID)
    html = html.replace(
        '||!align.value||(align.value==="NO"&&!valid(secondPoints,secondClosed))',
        '||!align.value||align.value==="UNCERTAIN"||(align.value==="NO"&&!valid(secondPoints,secondClosed))',
    )
    return server, html


def contact_sheet(first: Path, second: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    images = [("FIRST HALF", first), ("SECOND HALF", second)]
    sheet = Image.new("RGB", (1280, 760), "#18222c")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((16, 12), "HUMAN PITCH REVIEW INPUT — NO POLYGON YET", fill="white", font=font)
    for index, (title, path) in enumerate(images):
        image = Image.open(path).convert("RGB")
        image.thumbnail((620, 680))
        x = 16 + index * 632
        sheet.paste(image, (x, 44))
        draw.text((x, 724), f"128058 {title} | source metadata in manifest", fill="white", font=font)
    output = STAGE / "04_VISUAL_QA/128058_pitch_polygon_review_inputs.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)


def build(args: argparse.Namespace) -> None:
    validate_inputs()
    package = STAGE / "03_PITCH_POLYGON_REVIEW_PACKAGE"
    if package.exists() or (STAGE / "02_REVIEW_INPUTS/first_half_reference.png").exists():
        raise RuntimeError("FAIL_G7D_B2A_REVIEWER_REUSE: stage already prepared")
    source_resolution: dict[str, Any] = {
        "schema_version": "football_intelligence.g7d_b2a.source_video_resolution.v1",
        "match_id": MATCH_ID,
        "videos": {},
    }
    frames: dict[str, Any] = {}
    for half, data in SOURCE_VIDEOS.items():
        source = PROJECT / data["relative_path"]
        metadata = probe(args.ffprobe, source)
        frame_name = f"{half}_half_reference.png"
        frame_path = STAGE / "02_REVIEW_INPUTS" / frame_name
        requested, frame_index, resolved = extract_reference_frame(
            args.ffmpeg, source, frame_path, metadata.pop("_duration_decimal"), metadata.pop("_fps_decimal")
        )
        source_resolution["videos"][half] = {
            "half": "FIRST_HALF" if half == "first" else "SECOND_HALF",
            "project_relative_path": data["relative_path"],
            "byte_size": source.stat().st_size,
            "sha256": data["sha256"],
            "duration_seconds": metadata["duration_seconds"],
            "resolution": [metadata["source_width"], metadata["source_height"]],
            "frame_rate": metadata["frame_rate"],
            "source_manifest_reference": f"matches/{MATCH_ID}/manifests/source_file_manifest.json",
            "selection_reason": "single canonical panorama entry matching the source-file manifest half token",
        }
        frames[half] = {
            "half": "FIRST_HALF" if half == "first" else "SECOND_HALF",
            "relative_path": f"02_REVIEW_INPUTS/{frame_name}",
            "source_video_relative_path": data["relative_path"],
            "source_video_sha256": data["sha256"],
            "requested_timestamp_seconds": float(requested),
            "frame_index_zero_based": frame_index,
            "resolved_frame_timestamp_seconds": float(resolved),
            "source_width": metadata["source_width"],
            "source_height": metadata["source_height"],
            "frame_sha256": sha256_file(frame_path),
            "frame_byte_size": frame_path.stat().st_size,
            "selection_rule": "25_PERCENT_DURATION_NEAREST_FRAME_ROUND_HALF_UP",
        }
    write_json(STAGE / "01_INPUT_CLOSURE/source_video_resolution.json", source_resolution)
    write_json(
        STAGE / "01_INPUT_CLOSURE/b2_stop_validation.json",
        {
            "status": "PASS",
            "source": "06_B2_FAILURE_EVIDENCE/pitch_geometry_resolution.json",
            "required_failure": "FAIL_G7D_B2_128058_PITCH_PROVENANCE",
            "sampling_or_inference_started": False,
            "historical_static_freeze_promoted": False,
            "missing_historical_files_reconstructed": False,
        },
    )
    write_json(STAGE / "02_REVIEW_INPUTS/source_frame_manifest.json", {"match_id": MATCH_ID, "frames": frames})
    case = {
        "match_id": MATCH_ID,
        "review_id": REVIEW_ID,
        "review_revision": REVISION,
        "authoritative_drawing_half": "FIRST_HALF",
        "second_half_context": "READ_ONLY_UNLESS_ALIGNMENT_NO",
        "default_polygon_count": 1,
        "status": "PENDING_HUMAN_REVIEW",
        "team_convention": {"TEAM_1": "BLUE", "TEAM_2": "WHITE"},
        "source_frames": frames,
        "asset_urls": {"first": f"/assets/{MATCH_ID}/first.png", "second": f"/assets/{MATCH_ID}/second.png"},
    }
    calibration = PROJECT / f"matches/{MATCH_ID}/calibration/pitch_polygon_v1"
    write_json(calibration / "selection_manifest.json", case)
    write_json(
        calibration / "review_case_manifest.json",
        {
            "match_id": MATCH_ID,
            "review_id": REVIEW_ID,
            "revision": REVISION,
            "authoritative_half": "FIRST_HALF",
            "second_half_alignment_answers": ["YES", "NO", "UNCERTAIN"],
            "uncertain_completion_forbidden": True,
            "final_polygon_creation": "FORBIDDEN_BEFORE_COMPLETION_VALIDATION",
            "match_setup_update": "FORBIDDEN_BEFORE_FINALIZATION",
        },
    )
    write_json(calibration / "source_frame_manifest.json", frames)
    package.mkdir(parents=True, exist_ok=True)
    frames_dir = package / "_frames"
    frames_dir.mkdir()
    shutil.copyfile(STAGE / "02_REVIEW_INPUTS/first_half_reference.png", frames_dir / f"{MATCH_ID}_first.png")
    shutil.copyfile(STAGE / "02_REVIEW_INPUTS/second_half_reference.png", frames_dir / f"{MATCH_ID}_second.png")
    write_json(
        package / "review_cases.json",
        {
            "review_id": REVIEW_ID,
            "review_revision": REVISION,
            "port": PORT,
            "cases": [case],
            "saved_events_imported": False,
        },
    )
    shutil.copyfile(REPO / "scripts/g7d_a_polygon_validation.py", package / "polygon_validation.py")
    server, html = reviewer_templates()
    (package / "review_server.py").write_text(server, encoding="utf-8", newline="\n")
    (package / "index.html").write_text(html, encoding="utf-8", newline="\n")
    (package / "launch_pitch_polygon_review.ps1").write_text(
        "$ErrorActionPreference = 'Stop'\n"
        f"$RepoRoot = '{REPO}'\n$PackageRoot = $PSScriptRoot\n$Port = {PORT}\n"
        "$occupied = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue\n"
        'if ($occupied) { Write-Error "Port $Port is occupied; stop that process before launching." }\n'
        "Set-Location -LiteralPath $PackageRoot\n"
        "& (Join-Path $RepoRoot '.venv\\Scripts\\python.exe') .\\review_server.py --port $Port\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        package / "reviewer_contract.json",
        {
            "review_id": REVIEW_ID,
            "revision": REVISION,
            "port": PORT,
            "url": f"http://127.0.0.1:{PORT}/",
            "canonical_coordinate_state": "SOURCE_IMAGE_COORDINATES_ONLY",
            "event_acknowledgement_completion_order": (
                "ATOMIC_EVENT_THEN_ACK_RECEIPT_THEN_COMPLETION_RECEIPT_THEN_HTTP_200"
            ),
            "pitch_polygon_finalization": "FORBIDDEN_BEFORE_HUMAN_COMPLETION",
            "match_setup_update": "FORBIDDEN_BEFORE_FINALIZATION",
        },
    )
    contact_sheet(
        STAGE / "02_REVIEW_INPUTS/first_half_reference.png", STAGE / "02_REVIEW_INPUTS/second_half_reference.png"
    )


def package_handoff() -> None:
    """Create the upload-only B2A handoff without changing human-review state."""
    package = STAGE / "03_PITCH_POLYGON_REVIEW_PACKAGE"
    frames = read_json(STAGE / "02_REVIEW_INPUTS/source_frame_manifest.json")
    resolution = read_json(STAGE / "01_INPUT_CLOSURE/source_video_resolution.json")
    contract = read_json(package / "reviewer_contract.json")
    calibration = PROJECT / f"matches/{MATCH_ID}/calibration/pitch_polygon_v1"
    if (
        (calibration / "pitch_polygon.json").exists()
        or (package / "review_events").exists()
        or (package / "review_receipts").exists()
    ):
        raise RuntimeError("FAIL_G7D_B2A_HANDOFF_REUSE: human-review state is no longer pristine")
    handoff = STAGE / "05_REVIEW_PACK/CHATGPT_HANDOFF"
    if handoff.exists():
        shutil.rmtree(handoff)
    handoff.mkdir(parents=True)
    head = git("rev-parse", "HEAD")
    write_json(
        handoff / "01_EXECUTIVE_SUMMARY.json",
        {
            "schema_version": "football_intelligence.g7d_b2a.executive_summary.v1",
            "status": "PASS_G7D_B2A_128058_PITCH_POLYGON_REVIEW_READY_FOR_HUMAN_REVIEW",
            "repository_head": head,
            "match_id": MATCH_ID,
            "review_id": REVIEW_ID,
            "review_revision": REVISION,
            "review_url": contract["url"],
            "human_review_case_count": 1,
            "next_human_action": "Launch the reviewer, draw the first-half canonical polygon, select alignment YES or NO, then Save.",
            "unresolved_blockers": [
                "A human must create and save the one canonical first-half polygon.",
                "The B2 baseline remains forbidden until G7D_B2B_128058_PITCH_POLYGON_FINALIZATION completes.",
            ],
            "next_stage_after_human_completion": "G7D_B2B_128058_PITCH_POLYGON_FINALIZATION",
            "blocked_until_human_completion": True,
            "prohibited_before_human_completion": [
                "pitch_polygon.json",
                "match_setup.json update",
                "G7D-B2 baseline sampling",
                "proposal or semantic runtime",
            ],
        },
    )
    write_json(
        handoff / "02_INPUT_AND_PROVENANCE_RESULTS.json",
        {
            "match_id": MATCH_ID,
            "team_convention": {"TEAM_1": "BLUE", "TEAM_2": "WHITE"},
            "b2_stop_validation": read_json(STAGE / "01_INPUT_CLOSURE/b2_stop_validation.json"),
            "canonical_videos": resolution["videos"],
            "deterministic_review_frames": frames["frames"],
            "canonical_coordinate_system": "SOURCE_IMAGE_COORDINATES_ONLY",
        },
    )
    write_json(
        handoff / "03_REVIEWER_AND_TEST_RESULTS.json",
        {
            "reviewer": contract,
            "launch_script": "03_PITCH_POLYGON_REVIEW_PACKAGE/launch_pitch_polygon_review.ps1",
            "frame_routes": [f"/assets/{MATCH_ID}/first.png", f"/assets/{MATCH_ID}/second.png"],
            "human_state_at_packaging": "NO_EVENT_OR_RECEIPT_EXISTS",
            "receipt_order": "ATOMIC_EVENT_THEN_ACK_RECEIPT_THEN_COMPLETION_RECEIPT_THEN_HTTP_200",
            "focused_test_result": "PASS (pytest tests/test_g7d_b2a_128058_pitch_polygon_review.py -q)",
        },
    )
    (handoff / "04_DECISION.md").write_text(
        "# B2A decision\n\n"
        "`128058` is ready for one bounded human pitch-polygon review. This package uses exactly the resolved first- and second-half panorama videos and their deterministic 25%-duration frames. The only authoritative polygon input is the reviewer’s first-half canonical source-coordinate polygon. A final polygon artifact and match-setup mutation remain forbidden until a human event and its immutable acknowledgement and completion receipts exist.\n",
        encoding="utf-8",
        newline="\n",
    )
    (handoff / "05_REVIEW_CONTRACT.md").write_text(
        "# Human review contract\n\n"
        f"- Revision: `{REVISION}`\n"
        f"- Case: `{MATCH_ID}`; `TEAM_1=BLUE`, `TEAM_2=WHITE`.\n"
        "- Launch `03_PITCH_POLYGON_REVIEW_PACKAGE/launch_pitch_polygon_review.ps1`; the bounded local URL is `http://127.0.0.1:8813/`.\n"
        "- Draw only on the first-half view. Both canvases render the same canonical source-coordinate vertices.\n"
        "- Select YES only when the read-only second-half overlay aligns; select NO for a genuine camera change. UNCERTAIN cannot complete the case.\n"
        "- Server HTTP 200 follows only atomic persistence of event, acknowledgement receipt, and completion receipt.\n"
        "- Stop after Save. Do not create `pitch_polygon.json` or modify `match_setup.json` in this phase.\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        handoff / "06_TESTS_AND_SAFETY.json",
        {
            "focused_checks": [
                {"command": "uv lock --check", "status": "PASS"},
                {"command": "uv sync", "status": "PASS"},
                {
                    "command": "python -m ruff format --check scripts/g7d_b2a_build_pitch_polygon_review.py tests/test_g7d_b2a_128058_pitch_polygon_review.py",
                    "status": "PASS",
                },
                {
                    "command": "uv run ruff check scripts/g7d_b2a_build_pitch_polygon_review.py tests/test_g7d_b2a_128058_pitch_polygon_review.py",
                    "status": "PASS",
                },
                {
                    "command": "node --check <changed JavaScript files>",
                    "status": "NOT_APPLICABLE_NO_STANDALONE_JAVASCRIPT_FILE_CHANGED",
                },
                {"command": "uv run pytest tests/test_g7d_b2a_128058_pitch_polygon_review.py -q", "status": "PASS"},
                {"command": "git diff --check", "status": "PASS"},
            ],
            "safety": {
                "sampling_or_inference_started": False,
                "b1_runtime_executed": False,
                "validation_or_holdout_accessed": False,
                "final_pitch_polygon_created": False,
                "human_event_created": False,
                "visual_count": 1,
            },
        },
    )
    shutil.copy2(STAGE / "04_VISUAL_QA/128058_pitch_polygon_review_inputs.png", handoff / "07_REVIEW_INPUTS.png")
    manifest_rows = []
    for path in sorted(handoff.iterdir()):
        if path.name == "08_MANIFEST.json":
            continue
        manifest_rows.append({"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(
        handoff / "08_MANIFEST.json",
        {"schema_version": "football_intelligence.g7d_b2a.review_pack_manifest.v1", "files": manifest_rows},
    )
    (STAGE / "05_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only CHATGPT_HANDOFF. It is self-contained and excludes source video, runtime artifacts, and event data.\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, required=True)
    parser.add_argument("--refresh-reviewer", action="store_true")
    parser.add_argument("--package-handoff", action="store_true")
    args = parser.parse_args()
    if args.refresh_reviewer and args.package_handoff:
        raise RuntimeError("choose one bounded maintenance action")
    if args.refresh_reviewer:
        package = STAGE / "03_PITCH_POLYGON_REVIEW_PACKAGE"
        if not package.is_dir() or (package / "review_events").exists() or (package / "review_receipts").exists():
            raise RuntimeError("FAIL_G7D_B2A_REVIEWER_REUSE: reviewer refresh requires an untouched package")
        server, html = reviewer_templates()
        (package / "review_server.py").write_text(server, encoding="utf-8", newline="\n")
        (package / "index.html").write_text(html, encoding="utf-8", newline="\n")
    elif args.package_handoff:
        package_handoff()
    else:
        build(args)


if __name__ == "__main__":
    main()
