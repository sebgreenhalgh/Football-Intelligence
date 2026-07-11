# ruff: noqa: E501

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from football_intelligence.paths import (
    MATCH_ROOT,
    SOCCERTRACK_ROOT,
    STAGE3C_ASSIGNMENTS_10FPS_SMOOTHED_PATH,
    STAGE3C_OFFICIAL_CANDIDATES_10FPS_PATH,
    STAGE3C_REFEREE_CANDIDATES_10FPS_PATH,
    STAGE3C13_GOLD20_MANUAL_LABELS_PATH,
    STAGE3C14E_GOLD8_BEFORE_AFTER_COMPARISON_REPORT_PATH,
    STAGE3C14E_GOLD8_REPAIRED_EVAL_SUMMARY_PATH,
    STAGE3C15_ERROR_ROWS_PATH,
    STAGE3C15_GOLD8_EVAL_SUMMARY_PATH,
    STAGE3C15_REPORT_PATH,
    STAGE3C15_REVIEW_CONTACT_SHEET_PATH,
    STAGE3C15_SPECIALIST_ROLE_CANDIDATES_PATH,
    ensure_dir,
    require_file,
)
from football_intelligence.stage3d_anchor_warp import VISUAL_ONLY_WARNING


STAGE = "STAGE_REVIEW_PACK_PRO_EXTENDED_STEP1_PLAYER_ROLE_DETECTION_RESET"
PACK_DIR = MATCH_ROOT / "review_packs" / "pro_extended_step1_player_role_detection_reset"
PACK_ZIP_PATH = MATCH_ROOT / "review_packs" / "pro_extended_step1_player_role_detection_reset.zip"
MANIFEST_PATH = PACK_DIR / "04_RECOMMENDED_UPLOAD_MANIFEST.json"
PROMPT_PATH = PACK_DIR / "05_PRO_EXTENDED_PROMPT.txt"
MAX_PACK_FILES = 20

GENERATED_FILES = [
    "00_REVIEW_INDEX.md",
    "01_PRO_EXTENDED_CONTEXT.md",
    "02_CURRENT_PIPELINE_MAP.md",
    "03_KEY_FAILURE_SUMMARY.md",
    "04_RECOMMENDED_UPLOAD_MANIFEST.json",
    "05_PRO_EXTENDED_PROMPT.txt",
]

SELECTED_FILES = [
    {
        "filename": "06_stage3c_10_smooth_hq_short_slots_10fps.py",
        "source": SOCCERTRACK_ROOT / "scripts" / "stage3c_10_smooth_hq_short_slots_10fps.py",
        "kind": "source_code",
        "description": "Creates team_constrained_slot_assignments_10fps_smoothed.json with conservative same-slot smoothing.",
        "question_answered": "Where is the current core smoothed 10fps player-slot assignment artifact created?",
    },
    {
        "filename": "07_stage3c_09_track_hq_short_slots_10fps.py",
        "source": SOCCERTRACK_ROOT / "scripts" / "stage3c_09_track_hq_short_slots_10fps.py",
        "kind": "source_code",
        "description": "Builds the true-10fps team-constrained slot assignments before smoothing.",
        "question_answered": "How does the current system assign expected team slots from detections?",
    },
    {
        "filename": "08_stage3c_08_classify_hq_short_team_colours_10fps.py",
        "source": SOCCERTRACK_ROOT / "scripts" / "stage3c_08_classify_hq_short_team_colours_10fps.py",
        "kind": "source_code",
        "description": "Frame-level team colour classification feeding player slot assignment.",
        "question_answered": "How does the current system decide team colour?",
    },
    {
        "filename": "09_stage3c_07_detect_hq_short_candidates_10fps.py",
        "source": SOCCERTRACK_ROOT / "scripts" / "stage3c_07_detect_hq_short_candidates_10fps.py",
        "kind": "source_code",
        "description": "Current 10fps candidate detection and person-role candidate generation.",
        "question_answered": "How are people, player candidates, officials, staff, and unknowns detected per frame?",
    },
    {
        "filename": "10_stage3c15_specialist_role_repair.py",
        "source": SOCCERTRACK_ROOT / "src" / "football_intelligence" / "stage3c15_specialist_role_repair.py",
        "kind": "source_code",
        "description": "Specialist role repair attempt for officials and goalkeepers.",
        "question_answered": "Why did official exclusion improve some errors while reducing player recall, and why did GK repair not help?",
    },
    {
        "filename": "11_stage3c14e_gold8_repair.py",
        "source": SOCCERTRACK_ROOT / "src" / "football_intelligence" / "stage3c14e_gold8_repair.py",
        "kind": "source_code",
        "description": "Gold-8 strict one-to-one matching and repaired evaluation harness.",
        "question_answered": "How is correctness measured against manually labelled visible people?",
    },
    {
        "filename": "12_stage3c13_gold20.py",
        "source": SOCCERTRACK_ROOT / "src" / "football_intelligence" / "stage3c13_gold20.py",
        "kind": "source_code",
        "description": "Gold-20 manual labelling pack and candidate-person row construction.",
        "question_answered": "What manual truth schema defines visible person, role, team, identity, and uncertainty?",
    },
    {
        "filename": "13_gold8_manual_labels_excerpt.json",
        "kind": "evidence_schema",
        "description": "Compact excerpt of completed Gold manual labels used as visual truth.",
        "question_answered": "What does correct player/GK/official/off-pitch labelling look like?",
        "generated_excerpt": "gold8_manual_labels",
    },
    {
        "filename": "14_stage3c14e_repaired_eval_summary.json",
        "source": STAGE3C14E_GOLD8_REPAIRED_EVAL_SUMMARY_PATH,
        "kind": "evidence_json",
        "description": "Gold-8 repaired strict matching summary after duplicate-harness fix.",
        "question_answered": "What did the repaired harness fix before Stage 3C.15?",
    },
    {
        "filename": "15_stage3c14e_before_after_eval_comparison.md",
        "source": STAGE3C14E_GOLD8_BEFORE_AFTER_COMPARISON_REPORT_PATH,
        "kind": "evidence_report",
        "description": "Before/after summary of the Stage 3C.14E matching-harness repair.",
        "question_answered": "Which previous failure was an evaluation harness artifact rather than a detection issue?",
    },
    {
        "filename": "16_stage3c15_gold8_eval_summary.json",
        "source": STAGE3C15_GOLD8_EVAL_SUMMARY_PATH,
        "kind": "evidence_json",
        "description": "Gold-8 evaluation summary for the Stage 3C.15 specialist role repair candidate.",
        "question_answered": "How did Stage 3C.15 change official, GK, matched, and missed counts?",
    },
    {
        "filename": "17_stage3c15_specialist_role_repair_report.md",
        "source": STAGE3C15_REPORT_PATH,
        "kind": "evidence_report",
        "description": "Short Stage 3C.15 report with known limitations and recommendation.",
        "question_answered": "Why should Stage 3C.15 not be promoted as the next default?",
    },
    {
        "filename": "18_current_role_detection_schemas_excerpt.json",
        "kind": "schema_excerpt",
        "description": "Focused schemas and samples from smoothed assignments, official/referee candidates, Stage 3C.15 role candidates, and error rows.",
        "question_answered": "What are the exact data shapes used by current frame-level role logic and its failure evidence?",
        "generated_excerpt": "role_detection_schemas",
    },
    {
        "filename": "19_stage3c15_review_contact_sheet.jpg",
        "source": STAGE3C15_REVIEW_CONTACT_SHEET_PATH,
        "kind": "image",
        "description": "Visual contact sheet showing Stage 3C.15 role-repair successes and failures.",
        "question_answered": "What do the current frame-level role/person failures look like in the actual footage?",
    },
]


def visual_stamp(payload: dict[str, Any]) -> dict[str, Any]:
    payload["stage"] = payload.get("stage", STAGE)
    payload["visual_only_warning"] = VISUAL_ONLY_WARNING
    payload["do_not_use_for_metrics"] = True
    payload["production_ready"] = False
    payload["project_wide_defaults_changed"] = False
    return payload


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path, str(path)).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(visual_stamp(payload), indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    if VISUAL_ONLY_WARNING not in text:
        text = f"- Warning: `{VISUAL_ONLY_WARNING}`.\n\n{text}"
    path.write_text(text, encoding="utf-8")


def clean_pack_dir() -> None:
    ensure_dir(PACK_DIR)
    for path in PACK_DIR.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def short_abs(path: Path | None) -> str:
    if path is None:
        return ""
    return str(path.resolve())


def copy_file(source: Path, destination: Path) -> None:
    require_file(source, str(source))
    ensure_dir(destination.parent)
    shutil.copy2(source, destination)


def compact_bbox(value: Any) -> dict[str, Any] | list[float] | None:
    if isinstance(value, dict):
        return {key: value.get(key) for key in ["x1", "y1", "x2", "y2"] if key in value}
    if isinstance(value, list) and len(value) >= 4:
        return value[:4]
    return None


def compact_observation(obs: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "player_slot_id",
        "slot_id",
        "detection_id",
        "source_detection_id",
        "status",
        "team",
        "team_label",
        "team_colour_label",
        "role",
        "role_gold",
        "visible_person_type_gold",
        "specialist_role",
        "specialist_role_confidence",
        "image_footpoint_xy",
        "selected_anchor_xy",
        "pitch_xy_visual_only",
        "projection_warning",
        "identity_source",
    ]
    compact = {key: obs.get(key) for key in keys if key in obs}
    bbox = compact_bbox(obs.get("bbox"))
    if bbox:
        compact["bbox"] = bbox
    if "footpoint_x" in obs or "footpoint_y" in obs:
        compact["footpoint_xy"] = [obs.get("footpoint_x"), obs.get("footpoint_y")]
    return compact


def compact_frame(frame: dict[str, Any], observation_limit: int = 8) -> dict[str, Any]:
    return {
        "frame_id": frame.get("frame_id"),
        "frame_sequence": frame.get("frame_sequence"),
        "timestamp_seconds": frame.get("timestamp_seconds"),
        "observed_player_slot_count": frame.get("observed_player_slot_count"),
        "active_player_slot_count": frame.get("active_player_slot_count"),
        "slot_observations_sample": [compact_observation(item) for item in frame.get("slot_observations", [])[:observation_limit]],
        "missing_slot_states_sample": [compact_observation(item) for item in frame.get("missing_slot_states", [])[:5]],
    }


def compact_detection_frame(frame: dict[str, Any], detection_limit: int = 8) -> dict[str, Any]:
    detections = frame.get("detections") or frame.get("candidates") or frame.get("persons") or []
    return {
        "frame_id": frame.get("frame_id"),
        "frame_sequence": frame.get("frame_sequence"),
        "timestamp_seconds": frame.get("timestamp_seconds"),
        "detection_count": len(detections) if isinstance(detections, list) else None,
        "detections_sample": [compact_observation(item) for item in detections[:detection_limit]] if isinstance(detections, list) else [],
    }


def first_frames(payload: dict[str, Any], limit: int = 2) -> list[dict[str, Any]]:
    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        return []
    return frames[:limit]


def gold8_manual_labels_excerpt() -> dict[str, Any]:
    payload = read_json(STAGE3C13_GOLD20_MANUAL_LABELS_PATH)
    frames = payload.get("frames", [])
    completed = [frame for frame in frames if frame.get("labels_complete") is True]
    selected = completed[:8] if completed else frames[:8]
    out_frames = []
    for frame in selected:
        people = []
        for person in frame.get("persons", []):
            people.append(
                {
                    "gold_person_id": person.get("gold_person_id"),
                    "candidate_row_id": person.get("candidate_row_id"),
                    "source_detection_id": person.get("source_detection_id"),
                    "bbox": compact_bbox(person.get("bbox")),
                    "visible_person_type_gold": person.get("visible_person_type_gold"),
                    "team_colour_gold": person.get("team_colour_gold"),
                    "role_gold": person.get("role_gold"),
                    "stable_visual_identity_id_gold": person.get("stable_visual_identity_id_gold"),
                    "image_footpoint_xy_gold": person.get("image_footpoint_xy_gold"),
                    "location_confidence_gold": person.get("location_confidence_gold"),
                    "team_confidence_gold": person.get("team_confidence_gold"),
                    "identity_confidence_gold": person.get("identity_confidence_gold"),
                    "occlusion_state_gold": person.get("occlusion_state_gold"),
                    "should_be_projected_as_observed_dot_gold": person.get("should_be_projected_as_observed_dot_gold"),
                    "notes": person.get("notes", ""),
                    "visual_only_warning": VISUAL_ONLY_WARNING,
                    "do_not_use_for_metrics": True,
                }
            )
        out_frames.append(
            {
                "gold20_frame_id": frame.get("gold20_frame_id"),
                "frame_id": frame.get("frame_id"),
                "frame_sequence": frame.get("frame_sequence"),
                "timestamp_seconds": frame.get("timestamp_seconds"),
                "labels_complete": frame.get("labels_complete"),
                "person_count": len(people),
                "persons": people,
                "visual_only_warning": VISUAL_ONLY_WARNING,
                "do_not_use_for_metrics": True,
            }
        )
    return visual_stamp(
        {
            "artifact": "gold8_manual_labels_excerpt",
            "source_path": short_abs(STAGE3C13_GOLD20_MANUAL_LABELS_PATH),
            "completed_frames_in_source": len(completed),
            "frames_in_excerpt": len(out_frames),
            "frames": out_frames,
        }
    )


def role_detection_schemas_excerpt() -> dict[str, Any]:
    smoothed = read_json(STAGE3C_ASSIGNMENTS_10FPS_SMOOTHED_PATH)
    officials = read_json(STAGE3C_OFFICIAL_CANDIDATES_10FPS_PATH)
    referees = read_json(STAGE3C_REFEREE_CANDIDATES_10FPS_PATH)
    specialist = read_json(STAGE3C15_SPECIALIST_ROLE_CANDIDATES_PATH)
    errors = read_json(STAGE3C15_ERROR_ROWS_PATH)

    def sample_errors(limit: int = 12) -> list[dict[str, Any]]:
        rows = errors.get("error_rows", errors.get("rows", []))
        if not isinstance(rows, list):
            return []
        out = []
        for row in rows[:limit]:
            out.append(
                {
                    "frame_id": row.get("frame_id"),
                    "frame_sequence": row.get("frame_sequence"),
                    "gold_person_id": row.get("gold_person_id"),
                    "candidate_source": row.get("candidate_source"),
                    "error_type": row.get("error_type"),
                    "gold_team": row.get("gold_team"),
                    "gold_role": row.get("gold_role"),
                    "candidate_team": row.get("candidate_team"),
                    "candidate_role": row.get("candidate_role"),
                    "candidate_slot_id": row.get("candidate_slot_id"),
                    "notes": row.get("notes", ""),
                    "visual_only_warning": VISUAL_ONLY_WARNING,
                    "do_not_use_for_metrics": True,
                }
            )
        return out

    return visual_stamp(
        {
            "artifact": "current_role_detection_schemas_excerpt",
            "sources": {
                "smoothed_assignments": short_abs(STAGE3C_ASSIGNMENTS_10FPS_SMOOTHED_PATH),
                "official_candidates_10fps": short_abs(STAGE3C_OFFICIAL_CANDIDATES_10FPS_PATH),
                "referee_candidates_10fps": short_abs(STAGE3C_REFEREE_CANDIDATES_10FPS_PATH),
                "stage3c15_specialist_role_candidates": short_abs(STAGE3C15_SPECIALIST_ROLE_CANDIDATES_PATH),
                "stage3c15_error_rows": short_abs(STAGE3C15_ERROR_ROWS_PATH),
            },
            "smoothed_assignment_summary": smoothed.get("summary", {}),
            "smoothed_assignment_frame_sample": [compact_frame(frame) for frame in first_frames(smoothed, 2)],
            "official_candidate_summary": officials.get("summary", {}),
            "official_candidate_frame_sample": [compact_detection_frame(frame) for frame in first_frames(officials, 2)],
            "referee_candidate_summary": referees.get("summary", {}),
            "referee_candidate_frame_sample": [compact_detection_frame(frame) for frame in first_frames(referees, 2)],
            "specialist_role_candidate_summary": specialist.get("summary", {}),
            "specialist_role_candidate_frame_sample": [compact_detection_frame(frame) for frame in first_frames(specialist, 2)],
            "stage3c15_error_rows_sample": sample_errors(),
            "note": "Projection fields are retained only as visual QA context. They are not metric truth.",
        }
    )


def generated_excerpt_payload(kind: str) -> dict[str, Any]:
    if kind == "gold8_manual_labels":
        return gold8_manual_labels_excerpt()
    if kind == "role_detection_schemas":
        return role_detection_schemas_excerpt()
    raise ValueError(f"Unknown excerpt kind: {kind}")


def selected_entries() -> list[dict[str, Any]]:
    entries = []
    for index, spec in enumerate(SELECTED_FILES, start=6):
        filename = str(spec["filename"])
        source = spec.get("source")
        destination = PACK_DIR / filename
        entries.append(
            {
                "sort_order": index,
                "filename": filename,
                "original_absolute_path": short_abs(source) if isinstance(source, Path) else "generated_excerpt",
                "copied_pack_path": short_abs(destination),
                "kind": spec["kind"],
                "description": spec["description"],
                "question_answered": spec["question_answered"],
                "visual_only_warning": VISUAL_ONLY_WARNING,
                "do_not_use_for_metrics": True,
                "production_ready": False,
            }
        )
    return entries


def generated_entries() -> list[dict[str, Any]]:
    descriptions = {
        "00_REVIEW_INDEX.md": "Human-readable index for the Pro Extended review pack.",
        "01_PRO_EXTENDED_CONTEXT.md": "Stage context, hard restrictions, and current evidence.",
        "02_CURRENT_PIPELINE_MAP.md": "Current image-space person/role detection pipeline map.",
        "03_KEY_FAILURE_SUMMARY.md": "Concise summary of the key Stage 3C.14E/3C.15 failure evidence.",
        "04_RECOMMENDED_UPLOAD_MANIFEST.json": "Machine-readable manifest for all pack files.",
        "05_PRO_EXTENDED_PROMPT.txt": "Prompt to paste into ChatGPT Pro Extended with the pack files.",
    }
    return [
        {
            "sort_order": index,
            "filename": filename,
            "original_absolute_path": "generated",
            "copied_pack_path": short_abs(PACK_DIR / filename),
            "kind": "prompt" if filename.endswith(".txt") else "generated_markdown_json",
            "description": descriptions[filename],
            "question_answered": "How should Pro Extended use this pack safely?",
            "visual_only_warning": VISUAL_ONLY_WARNING,
            "do_not_use_for_metrics": True,
            "production_ready": False,
        }
        for index, filename in enumerate(GENERATED_FILES)
    ]


def build_manifest() -> dict[str, Any]:
    files = generated_entries() + selected_entries()
    for item in files:
        path = Path(item["copied_pack_path"])
        item["exists"] = path.exists()
        item["file_size_bytes"] = path.stat().st_size if path.exists() else 0
    return visual_stamp(
        {
            "artifact": "pro_extended_step1_player_role_detection_reset_manifest",
            "file_count": len(files),
            "max_file_count": MAX_PACK_FILES,
            "review_pack_dir": short_abs(PACK_DIR),
            "review_pack_zip_path": short_abs(PACK_ZIP_PATH),
            "files": files,
            "selection_priorities": [
                "current code that creates or mutates player/person/role/team assignments",
                "evidence showing recall regression and official/GK failure",
                "Gold manual labels/evaluation that define visual correctness",
                "compact visuals showing actual frame failures",
            ],
            "hard_restrictions": [
                VISUAL_ONLY_WARNING,
                "no_speed_distance_fatigue_load_team_shape_pass_dribble_tactical_physical_or_football_conclusion_metrics",
                "do_not_change_stage3d4g_stage3d4h_or_stage3d4k_registries",
                "do_not_change_project_wide_defaults",
                "do_not_promote_stage3c11_stage3c12_or_stage3c15",
            ],
        }
    )


def render_review_index(manifest: dict[str, Any]) -> str:
    lines = [
        "# Pro Extended Step 1 Player/Role Detection Reset Review Pack",
        "",
        f"- Warning: `{VISUAL_ONLY_WARNING}`.",
        "- Purpose: diagnose frame-level person/player/goalkeeper/official role detection before identity tracking.",
        "- Production ready: **false**.",
        "- Project-wide defaults changed: **false**.",
        "- Do not calculate speed, distance, fatigue, load, team shape, pass, dribble, tactical, physical performance, or football conclusion metrics.",
        "",
        "## Selected Files",
        "",
        "| file | kind | why selected | helps answer | original path | size |",
        "|---|---|---|---|---|---:|",
    ]
    for item in manifest["files"]:
        lines.append(
            f"| `{item['filename']}` | {item['kind']} | {item['description']} | {item['question_answered']} | `{item['original_absolute_path']}` | {item['file_size_bytes']} |"
        )
    lines.extend(
        [
            "",
            "## Upload Guidance",
            "",
            "Upload the entire folder or zip to Pro Extended. If the file budget is tight, keep files `00` through `05`, the current core scripts `06` through `10`, the Gold/eval evidence `13` through `18`, and the contact sheet `19`.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_context() -> str:
    return f"""# Pro Extended Context

- Warning: `{VISUAL_ONLY_WARNING}`.
- Stage: `{STAGE}`.
- Scope: image-space frame/burst person and role detection reset.
- Production ready: **false**.
- Project-wide defaults changed: **false**.

The first step to redesign is not identity tracking and not football analysis. It is to identify and locate visible people per frame or short burst:

- team_1 outfield players
- team_1 goalkeeper
- team_2 outfield players
- team_2 goalkeeper
- officials/referees/linesmen as separate context people
- unknown / off-pitch / false-positive people

The system may maintain a model of 22 expected player roles, but it must not hallucinate invisible players. Each expected role should carry an explicit state such as `observed_clear`, `observed_partial`, `occluded`, `carried_missing`, `not_visible`, or `unknown`.

Current status:

- Stage 3C.13 created Gold-20 manual annotation assets.
- Stage 3C.14 / 3C.14E created partial Gold-8 validation and fixed strict one-to-one matching.
- Stage 3C.15 wired a specialist official/GK role repair layer.
- Stage 3C.15 is not safe to promote because official filtering reduced some official-as-player errors but also regressed matched Gold rows and did not repair goalkeeper mistakes.

Projection fields may appear in evidence files only as visual QA context. They are not metric truth and must not be used for football conclusions.
"""


def render_pipeline_map() -> str:
    return f"""# Current Pipeline Map

- Warning: `{VISUAL_ONLY_WARNING}`.
- This map is for image-space role detection diagnosis only.

## Source Inputs

- Source frame window: `matches/128058/frames/goal_window_stage3c_hq_short/`
- Candidate detection files: `matches/128058/detections/goal_window_stage3c_hq_short/`
- Current assignment files: `matches/128058/tracks/goal_window_stage3c_hq_short/`

## Current Step 1 Flow

1. `09_stage3c_07_detect_hq_short_candidates_10fps.py`
   Creates player, official, referee, staff, unknown, and person candidate JSONs at 10fps.

2. `08_stage3c_08_classify_hq_short_team_colours_10fps.py`
   Applies image-space team colour labels to player candidates.

3. `07_stage3c_09_track_hq_short_slots_10fps.py`
   Uses team-constrained logic to assign expected slot IDs to candidate detections.

4. `06_stage3c_10_smooth_hq_short_slots_10fps.py`
   Creates `{STAGE3C_ASSIGNMENTS_10FPS_SMOOTHED_PATH.name}` by changing only short same-slot missing gaps to carried occlusion.

5. Stage 3C.11 and Stage 3C.12 experimented with identity/global hybrid candidates, but those candidates were not promoted.

6. `10_stage3c15_specialist_role_repair.py`
   Attempts to reclassify officials and goalkeepers using specialist role candidates. It remains a candidate-only repair and should not be promoted.

## Evaluation / Truth Flow

- `12_stage3c13_gold20.py` builds the manual Gold-20 review pack.
- `11_stage3c14e_gold8_repair.py` evaluates candidate outputs against complete Gold-8 labels with strict one-to-one matching.
- Files `13` through `18` show Gold labels, repaired harness evidence, Stage 3C.15 regression evidence, and current data schemas.

## Current Break Point

The pipeline currently mixes frame-level role detection, team colour assignment, expected-slot assignment, official filtering, goalkeeper heuristics, and short temporal smoothing. Stage 3C.15 proves the specialist role layer is connected, but it is not safe: removing officials also removes or fails to match some true player observations, and goalkeeper-specific repair detects candidates without correcting the evaluated GK errors.
"""


def render_failure_summary() -> str:
    return f"""# Key Failure Summary

- Warning: `{VISUAL_ONLY_WARNING}`.
- These are visual QA/evaluation counts, not football performance metrics.

Stage 3C.14E fixed a matching-harness artifact:

- Duplicate candidates went from `81` to `0`.
- The repaired harness is the safer basis for evaluating role/person detection.

Stage 3C.15 specialist role repair is wired but not safe:

- Official candidates loaded: `894`.
- Official candidates matched: `147`.
- Officials excluded: `147`.
- Official-as-player errors improved from `8` to `5`.
- Matched Gold rows regressed from `151` to `145`.
- Missed rows worsened from `18` to `24`.
- Goalkeeper candidates detected: `432`.
- Goalkeeper candidates repaired: `0`.
- Goalkeeper-as-outfield errors stayed `2` to `2`.

Interpretation:

- Official/referee classification is useful but currently too entangled with player recall.
- Goalkeeper detection appears to find candidate evidence but does not produce successful slot/role repairs.
- The current specialist repair candidate should not be promoted.
- The next architecture should prioritize safe frame-level person and role detection before identity tracking, 2D projection conclusions, or downstream football analysis.
"""


def render_prompt() -> str:
    return f"""You are reviewing a compact code/evidence pack for the football-intelligence project.

Hard restrictions:
- {VISUAL_ONLY_WARNING}
- Do not calculate speed, distance, fatigue, load, team shape, pass, dribble, tactical, physical performance, or football conclusion metrics.
- Do not do identity tracking yet.
- Do not promote Stage 3C.11, Stage 3C.12, Stage 3C.15, or any new candidate.
- Do not change Stage 3D.4g/3D.4h/3D.4k calibration registries.
- Do not change project-wide defaults.
- Production-ready remains false.

Goal:
Diagnose and redesign Step 1 of the CV pipeline: per-frame or short-burst image-space person/player/role detection. The system should identify and locate team_1 outfield players, team_1 goalkeeper, team_2 outfield players, team_2 goalkeeper, officials/referees/linesmen, off-pitch people, unknowns, and false positives.

Expected output from you:

1. Deeply diagnose the current frame-level player/role detection architecture using the uploaded files.
2. Decide whether the current Stage 3C/Stage 3C.15 logic should be fixed incrementally or rewritten.
3. Propose a Step 1 architecture covering:
   - person detection
   - visible/candidate person states
   - team colour classification
   - goalkeeper classification
   - official/referee classification
   - expected 22-player role model with observed/partial/occluded/carried_missing/not_visible/unknown states
   - confidence and uncertainty handling
   - human review loop
4. Explain why official exclusion currently reduces player recall.
5. Explain why goalkeeper candidates can be detected but produce zero successful GK repairs.
6. Identify the first files/functions to edit.
7. Specify tests and visual QA artifacts required before any promotion.
8. Keep 2D projection only as visual QA context, not metric truth.

Important framing:
The first step may maintain the concept of 22 expected player roles, but it must not hallucinate invisible players. Observed dots/boxes should represent visible observed people only. Missing, hidden, or carried states must be explicit and should not be treated as observed people.
"""


def generate_selected_files() -> None:
    for spec in SELECTED_FILES:
        destination = PACK_DIR / str(spec["filename"])
        if "generated_excerpt" in spec:
            write_json(destination, generated_excerpt_payload(str(spec["generated_excerpt"])))
        else:
            copy_file(require_file(spec["source"], str(spec["source"])), destination)


def write_generated_docs(manifest: dict[str, Any]) -> None:
    write_text(PACK_DIR / "00_REVIEW_INDEX.md", render_review_index(manifest))
    write_text(PACK_DIR / "01_PRO_EXTENDED_CONTEXT.md", render_context())
    write_text(PACK_DIR / "02_CURRENT_PIPELINE_MAP.md", render_pipeline_map())
    write_text(PACK_DIR / "03_KEY_FAILURE_SUMMARY.md", render_failure_summary())
    write_json(MANIFEST_PATH, manifest)
    write_text(PROMPT_PATH, render_prompt())


def zip_pack() -> None:
    if PACK_ZIP_PATH.exists():
        PACK_ZIP_PATH.unlink()
    ensure_dir(PACK_ZIP_PATH.parent)
    files = sorted(path for path in PACK_DIR.iterdir() if path.is_file())
    if len(files) > MAX_PACK_FILES:
        raise RuntimeError(f"Review pack has {len(files)} files; maximum is {MAX_PACK_FILES}: {PACK_DIR}")
    with zipfile.ZipFile(PACK_ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)


def validate_pack() -> None:
    files = sorted(path for path in PACK_DIR.iterdir() if path.is_file())
    if len(files) > MAX_PACK_FILES:
        raise AssertionError(f"Pack contains {len(files)} files, expected <= {MAX_PACK_FILES}")
    required = set(GENERATED_FILES + [str(spec["filename"]) for spec in SELECTED_FILES])
    present = {path.name for path in files}
    missing = sorted(required - present)
    if missing:
        raise AssertionError(f"Missing review pack files: {missing}")
    if not MANIFEST_PATH.exists():
        raise AssertionError(f"Manifest missing: {MANIFEST_PATH}")
    if not PROMPT_PATH.exists():
        raise AssertionError(f"Prompt missing: {PROMPT_PATH}")
    if not PACK_ZIP_PATH.exists():
        raise AssertionError(f"Zip missing: {PACK_ZIP_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        copied = Path(item["copied_pack_path"])
        if not copied.exists():
            raise AssertionError(f"Manifest-selected file does not exist: {copied}")
    for path in files:
        if path.suffix.lower() in {".md", ".json", ".txt"} and path.name not in {"06_stage3c_10_smooth_hq_short_slots_10fps.py"}:
            text = path.read_text(encoding="utf-8")
            if VISUAL_ONLY_WARNING not in text:
                raise AssertionError(f"{path.name} is missing {VISUAL_ONLY_WARNING}")
    if manifest.get("production_ready") is not False:
        raise AssertionError("Manifest production_ready must be false")
    if manifest.get("project_wide_defaults_changed") is not False:
        raise AssertionError("Manifest project_wide_defaults_changed must be false")


def build_pro_extended_step1_review_pack() -> dict[str, Any]:
    clean_pack_dir()
    generate_selected_files()
    manifest = build_manifest()
    write_generated_docs(manifest)
    manifest = build_manifest()
    write_json(MANIFEST_PATH, manifest)
    write_text(PACK_DIR / "00_REVIEW_INDEX.md", render_review_index(manifest))
    zip_pack()
    validate_pack()

    result = visual_stamp(
        {
            "review_pack_dir": short_abs(PACK_DIR),
            "review_pack_zip_path": short_abs(PACK_ZIP_PATH),
            "selected_file_count": len([path for path in PACK_DIR.iterdir() if path.is_file()]),
            "manifest_path": short_abs(MANIFEST_PATH),
            "pro_extended_prompt_path": short_abs(PROMPT_PATH),
            "current_core_assignment_file_selected": (PACK_DIR / "06_stage3c_10_smooth_hq_short_slots_10fps.py").exists(),
            "stage3c15_evidence_selected": (PACK_DIR / "16_stage3c15_gold8_eval_summary.json").exists() and (PACK_DIR / "17_stage3c15_specialist_role_repair_report.md").exists(),
            "gold8_evidence_selected": (PACK_DIR / "13_gold8_manual_labels_excerpt.json").exists() and (PACK_DIR / "14_stage3c14e_repaired_eval_summary.json").exists(),
            "review_contact_sheet_selected": (PACK_DIR / "19_stage3c15_review_contact_sheet.jpg").exists(),
        }
    )
    return result


def print_result(result: dict[str, Any]) -> None:
    for key in [
        "review_pack_dir",
        "review_pack_zip_path",
        "selected_file_count",
        "manifest_path",
        "pro_extended_prompt_path",
        "current_core_assignment_file_selected",
        "stage3c15_evidence_selected",
        "gold8_evidence_selected",
        "review_contact_sheet_selected",
        "production_ready",
        "project_wide_defaults_changed",
        "visual_only_warning",
    ]:
        value = result[key]
        if isinstance(value, bool):
            value = str(value).lower()
        print(f"{key} {value}")


def main() -> None:
    print_result(build_pro_extended_step1_review_pack())


if __name__ == "__main__":
    main()
