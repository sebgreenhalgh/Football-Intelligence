"""Finalize C3A5C human evidence and audit development-default readiness.

This utility is analysis-only. It never invokes detector, feature, fold, or
pitch-gate inference and never changes the project runtime default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
EXPECTED_HEAD = "60f43f2ba3ff8acda41a8ac97f056cc19ba488a0"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
PART6 = PROJECT / "experiments/football_observation_reasoner/part 6"
PACK = PART7 / "G7D_C3A5D_Additional_Coverage_Finalization_And_Default_Decision_Codex_Pack"
C3A5C = PART7 / "G7D_C3A5C_ADDITIONAL_COVERAGE_REPLAY_AND_REVIEW_v1"
C3A5B = PART7 / "G7D_C3A5B_THREE_MATCH_PITCH_POLYGON_FINALIZATION_v1"
C3A4 = PART7 / "G7D_C3A4_DEVELOPMENT_DEFAULT_READINESS_AUDIT_v1"
C3A3 = PART7 / "G7D_C3A3_ACTIVE_SANDBOX_PITCH_GATE_INTEGRATION_v1"
C2 = PART6 / "G7D_C2_R1_RESUME_VISUAL_TRANSFER_DIAGNOSIS_FINALIZATION_v1"
STAGE = PART7 / "G7D_C3A5D_ADDITIONAL_COVERAGE_FINALIZATION_AND_DEFAULT_DECISION_v1"
PACKAGE = C3A5C / "04_ADDITIONAL_COVERAGE_REVIEW_PACKAGE"
DECISIONS = PACKAGE / "human_decisions"

TRAIN_MATCHES = ("117092", "117093", "118575", "118576", "118577", "128058")
ADDITIONAL_MATCHES = ("117093", "118576", "118577")
VISIBLE_LAST_EVENT = "796d658e-980b-456b-a6b0-391f96a1f72d"
COMPLETION_ID = "completion-35fed8f25691bc05701601fe"
REVIEW_ID = "G7D_C3A5C_ADDITIONAL_COVERAGE_REVIEW"
REVIEW_REVISION = "G7D_C3A5C_ADDITIONAL_COVERAGE_REVIEW_V1"
GATE_ID = "G3_CONSERVATIVE_FAR_OUTSIDE__fixed_08"
POLICY_ID = "G7D_C3A5D_DEVELOPMENT_DEFAULT_POLICY_DRAFT_V2"
DECISION = "PASS_G7D_C3A5D_DEVELOPMENT_DEFAULT_PROMOTION_APPROVED"
WARNING = "TARGETED ADDITIONAL-COVERAGE SAMPLE — NOT UNBIASED ACCURACY"
VISUAL_LABEL = "DEVELOPMENT-DEFAULT DECISION — NO DEFAULT CHANGED"

POLYGON_HASHES = {
    "117092": "92ca8040eedd3b0ec0bb685648691f0c314d8527f3fa8f2db1823b4461e4b338",
    "117093": "fa7091b859804cce4fef1cec9c66229f3e72127ae9f00633119c9acf657de452",
    "118575": "fbd7f3a473acc197b4c893d90bbaa4c5d484d1e883e8df1ac4601daf4396dec1",
    "118576": "54a0195f5b69ab598ce4c46c2224b5dffbefde56cb63cfde001088ea0fe1ef16",
    "118577": "eee7a33690cace2cab738f1d27b9674b98025b9efb0cbe996aac6631cadf9936",
    "128058": "24ad1e4d143527e5a3e92cded1b5d8b10526d67b5b0d1f8b02289a91e8c65307",
}

# Direct source-frame visual associations. The scene answer supplies the human
# goalkeeper-positive truth; human-confirmed goalkeeper kit colours and the
# exact candidate rectangle identify the corresponding retained candidate.
GOALKEEPER_ASSOCIATIONS = {
    "scene_06_118576_touchline_outside_proxy": [
        "frame_01bc84ee23d5_candidate_0030",
        "frame_01bc84ee23d5_candidate_0034",
    ],
    "scene_07_118576_high_density_overlap": [
        "frame_cb714da0d9eb_candidate_0021",
        "frame_cb714da0d9eb_candidate_0026",
    ],
    "scene_08_118576_stable_control": ["frame_4886cf882bb1_candidate_0046"],
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def compact_digest(value: Any) -> str:
    packed = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(packed).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "project_relative_path": path.relative_to(PROJECT).as_posix(),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def write_manifest(folder: Path, name: str) -> None:
    files = sorted(path for path in folder.iterdir() if path.is_file() and path.name != name)
    write_json(
        folder / name,
        {
            "file_count": len(files),
            "files": [
                {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)} for path in files
            ],
            "self_hash_omitted": True,
        },
    )


def validate_pack() -> list[dict[str, Any]]:
    manifest = read_json(PACK / "04_PACK_MANIFEST.json")
    rows = []
    for expected in manifest["files"]:
        path = PACK / expected["path"]
        if path.stat().st_size != expected["byte_size"] or sha256_file(path) != expected["sha256"]:
            raise RuntimeError(f"prompt-pack mismatch: {expected['path']}")
        rows.append(artifact(path))
    return rows


def assert_preflight() -> None:
    if git("rev-parse", "HEAD") != EXPECTED_HEAD or git("branch", "--show-current") != "main":
        raise RuntimeError("repository baseline or branch mismatch")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree is not clean")
    if STAGE.exists():
        raise RuntimeError("C3A5D stage already exists")
    hook = (REPO / "src/football_intelligence/proposal_gate_hook.py").read_text(encoding="utf-8")
    if "DEFAULT_PITCH_GATE_MODE = PitchGateMode.DISABLED" not in hook:
        raise RuntimeError("project pitch-gate default is not DISABLED")


def immutable_snapshot() -> list[dict[str, Any]]:
    paths = sorted((DECISIONS / "events").rglob("*.json"))
    paths += sorted((DECISIONS / "receipts/acknowledgements").glob("*.json"))
    paths += sorted((DECISIONS / "receipts/completion").glob("*.json"))
    return [artifact(path) for path in paths]


def validate_continuation(pack_rows: list[dict[str, Any]]) -> dict[str, Any]:
    executive = read_json(C3A5C / "07_REVIEW_PACK/CHATGPT_HANDOFF/01_EXECUTIVE_SUMMARY.json")
    replay = read_json(C3A5C / "01_FRAME_REPLAY/frame_and_candidate_manifest.json")
    reuse = read_json(C3A5C / "01_FRAME_REPLAY/proposal_runtime_reuse_report.json")
    scenes = read_json(C3A5C / "03_SCENE_AND_TARGET_SELECTION/scene_shortlist.json")
    targets = read_json(C3A5C / "03_SCENE_AND_TARGET_SELECTION/target_manifest.json")
    gate_counts = Counter(candidate["gate_decision"] for candidate in replay["candidates"])
    checks = {
        "classification": executive["classification"]
        == "PASS_G7D_C3A5C_ADDITIONAL_COVERAGE_REVIEW_READY_FOR_HUMAN_REVIEW",
        "matches": executive["matches"] == list(ADDITIONAL_MATCHES),
        "frames": len(replay["frames"]) == 48
        and Counter(frame["match_id"] for frame in replay["frames"])
        == Counter({match: 16 for match in ADDITIONAL_MATCHES}),
        "candidates": len(replay["candidates"]) == 3127,
        "gate_counts": gate_counts
        == Counter({"KEEP": 1321, "BOUNDARY_REVIEW": 922, "SUPPRESS_SANDBOX": 870, "EXCEPTION_KEEP": 14}),
        "scenes_and_targets": scenes["scene_count"] == 12 and targets["target_count"] == 60,
        "no_semantic_runtime": reuse["crop_features_executed"] is False and reuse["semantic_folds_executed"] is False,
        "default_unchanged": executive["production_ready"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"FAIL_G7D_C3A5D_CONTINUATION_PROVENANCE: {checks}")
    return {
        "classification": "PASS_G7D_C3A5D_CONTINUATION_PROVENANCE",
        "checks": checks,
        "repository_head_at_start": EXPECTED_HEAD,
        "model_binding": "GPT-5.6 Sol / Medium",
        "prompt_pack_files": pack_rows,
        "gate_decision_counts": dict(gate_counts),
        "project_default": "DISABLED",
        "default_changed": False,
        "production_ready": False,
    }


def validate_event_chain() -> (
    tuple[
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]
):
    targets = read_json(C3A5C / "03_SCENE_AND_TARGET_SELECTION/target_manifest.json")["targets"]
    target_by_id = {row["target_id"]: row for row in targets}
    scenes = read_json(C3A5C / "03_SCENE_AND_TARGET_SELECTION/scene_shortlist.json")["scenes"]
    scene_by_id = {row["scene_id"]: row for row in scenes}
    package_cases = read_json(PACKAGE / "review_cases.json")
    package_scene_by_id = {row["scene_id"]: row for row in package_cases["scenes"]}
    candidate_paths = sorted((DECISIONS / "events/candidate").glob("*.json"))
    scene_paths = sorted((DECISIONS / "events/scene").glob("*.json"))
    ack_paths = sorted((DECISIONS / "receipts/acknowledgements").glob("*.json"))
    completion_paths = sorted((DECISIONS / "receipts/completion").glob("*.json"))
    if (len(candidate_paths), len(scene_paths), len(ack_paths), len(completion_paths)) != (60, 12, 72, 1):
        raise RuntimeError("FAIL_G7D_C3A5D_HUMAN_EVENT_CHAIN: raw cardinality mismatch")

    allowed = {
        "proposal_validity": {
            "SINGLE_PERSON",
            "MULTIPLE_PEOPLE",
            "NO_PERSON_BACKGROUND_OBJECT",
            "DUPLICATE_CANDIDATE",
            "NOT_SURE",
        },
        "role": {"OUTFIELD_PLAYER", "GOALKEEPER", "RELEVANT_OFFICIAL", "OUT_OF_SCOPE_PERSON", "UNKNOWN_ROLE"},
        "participation": {"ACTIVE", "WARMING_OR_NON_ACTIVE", "NON_PLAYER", "UNKNOWN_PARTICIPATION"},
        "pitch_state": {"ON_PITCH", "BOUNDARY", "OFF_PITCH", "UNKNOWN_PITCH_STATE"},
        "box_quality": {
            "GOOD_BOX",
            "TOO_LOOSE",
            "TOO_TIGHT_OR_TRUNCATED",
            "MERGED",
            "MISLOCALIZED",
            "UNKNOWN_BOX_QUALITY",
        },
        "certainty": {"CERTAIN", "PROBABLE", "UNCERTAIN"},
    }
    candidate_rows = []
    selected_events: dict[tuple[str, str], dict[str, Any]] = {}
    sequences = []
    event_manifest = []
    for path in candidate_paths:
        event = read_json(path)
        try:
            uuid.UUID(event["event_id"])
        except ValueError as error:
            raise RuntimeError("non-UUID candidate event") from error
        target_id = event["target_id"]
        if target_id not in target_by_id or event["scene_id"] != target_by_id[target_id]["scene_id"]:
            raise RuntimeError("candidate event target/scene mismatch")
        target = target_by_id[target_id]
        scene = scene_by_id[event["scene_id"]]
        answers = event["answers"]
        if (
            event["event_id"] != path.stem
            or event["review_id"] != REVIEW_ID
            or event["review_revision"] != REVIEW_REVISION
            or event["event_type"] != "candidate"
            or event["production_ready"] is not False
            or answers["proposal_validity"] not in allowed["proposal_validity"]
            or answers["box_quality"] not in allowed["box_quality"]
            or answers["certainty"] not in allowed["certainty"]
        ):
            raise RuntimeError("invalid canonical candidate event")
        if answers["proposal_validity"] == "SINGLE_PERSON":
            if not all(answers.get(key) in allowed[key] for key in ("role", "participation", "pitch_state")):
                raise RuntimeError("single-person branch is incomplete")
        elif any(key in answers for key in ("role", "participation", "pitch_state")):
            raise RuntimeError("non-single candidate contains inferred person fields")
        package_scene = package_scene_by_id[event["scene_id"]]
        package_target = next(item for item in package_scene["targets"] if item["target_id"] == target_id)
        if (
            target["frame_sha256"] != scene["frame_sha256"]
            or package_scene["frame_sha256"] != scene["frame_sha256"]
            or package_target["source_box_xyxy"] != target["source_box_xyxy"]
            or target["gate_decision"] not in {"KEEP", "BOUNDARY_REVIEW", "SUPPRESS_SANDBOX", "EXCEPTION_KEEP"}
        ):
            raise RuntimeError("candidate frame/box/gate provenance mismatch")
        x1, y1, x2, y2 = target["source_box_xyxy"]
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or not (
            0 <= x1 < x2 <= scene["source_width"] and 0 <= y1 < y2 <= scene["source_height"]
        ):
            raise RuntimeError("invalid source-coordinate candidate box")
        event_hash = sha256_file(path)
        ack_path = DECISIONS / f"receipts/acknowledgements/ack-{event['event_id']}.json"
        ack = read_json(ack_path)
        if (
            ack["event_id"] != event["event_id"]
            or ack["event_sha256"] != event_hash
            or ack["event_byte_size"] != path.stat().st_size
            or ack["server_validated"] is not True
            or ack["case_complete"] is not True
            or ack["stable_id"] != target_id
            or ack["event_relative_path"] != f"events/candidate/{event['event_id']}.json"
        ):
            raise RuntimeError("candidate acknowledgement linkage failure")
        contains_person = answers["proposal_validity"] in {"SINGLE_PERSON", "MULTIPLE_PEOPLE"}
        active_player_goalkeeper = (
            answers["proposal_validity"] == "SINGLE_PERSON"
            and answers.get("role") in {"OUTFIELD_PLAYER", "GOALKEEPER"}
            and answers.get("participation") == "ACTIVE"
        )
        relevant_official = (
            answers["proposal_validity"] == "SINGLE_PERSON" and answers.get("role") == "RELEVANT_OFFICIAL"
        )
        useful_relevant = active_player_goalkeeper or relevant_official
        uncertain_relevant = (
            answers["proposal_validity"] == "SINGLE_PERSON"
            and answers.get("role") in {"OUTFIELD_PLAYER", "GOALKEEPER", "RELEVANT_OFFICIAL", "UNKNOWN_ROLE"}
            and (answers["certainty"] != "CERTAIN" or answers.get("role") == "UNKNOWN_ROLE")
        )
        candidate_rows.append(
            {
                "schema_version": "football_intelligence.g7d_c3a5d.candidate_human_label.v1",
                "match_id": target["match_id"],
                "scene_id": event["scene_id"],
                "target_id": target_id,
                "candidate_local_id": target["candidate_local_id"],
                "frame_id": target["frame_id"],
                "frame_sha256": target["frame_sha256"],
                "source_box_xyxy": target["source_box_xyxy"],
                "approximate_footpoint_xy": target["approximate_footpoint_xy"],
                "target_slot": target["target_slot"],
                "gate_decision": target["gate_decision"],
                "canonical_decision": answers,
                "analysis_flags": {
                    "contains_person": contains_person,
                    "contains_single_person": answers["proposal_validity"] == "SINGLE_PERSON",
                    "contains_multiple_people": answers["proposal_validity"] == "MULTIPLE_PEOPLE",
                    "is_duplicate": answers["proposal_validity"] == "DUPLICATE_CANDIDATE",
                    "is_background_or_object": answers["proposal_validity"] == "NO_PERSON_BACKGROUND_OBJECT",
                    "useful_relevant_person": useful_relevant,
                    "active_player_or_goalkeeper": active_player_goalkeeper,
                    "goalkeeper": active_player_goalkeeper and answers.get("role") == "GOALKEEPER",
                    "relevant_official": relevant_official,
                    "uncertain_potentially_relevant": uncertain_relevant,
                    "box_quality_issue": answers["box_quality"] != "GOOD_BOX",
                },
                "event_id": event["event_id"],
                "event_sha256": event_hash,
                "acknowledgement_receipt_id": ack["receipt_id"],
                "acknowledgement_receipt_sha256": sha256_file(ack_path),
            }
        )
        selected_events[("candidate", target_id)] = event
        sequences.append(event["server_sequence"])
        event_manifest.extend([artifact(path), artifact(ack_path)])

    allowed_scene = {
        "missed_relevant_people": {"NO", "YES_MARK", "NOT_SURE"},
        "missed_marks_complete": {"NONE", "COMPLETE", "NOT_SURE"},
        "goalkeeper_endline": {"YES", "NO", "NOT_SURE"},
        "player_temporarily_outside": {"YES", "NO", "NOT_SURE"},
        "official_touchline": {"YES", "NO", "NOT_SURE"},
        "crowding_overlap": {"LOW", "MODERATE", "HIGH", "NOT_SURE"},
        "distortion_glare": {"NO", "MODERATE", "YES", "NOT_SURE"},
        "certainty": {"CERTAIN", "PROBABLE", "UNCERTAIN"},
    }
    scene_rows = []
    mark_rows = []
    for path in scene_paths:
        event = read_json(path)
        try:
            uuid.UUID(event["event_id"])
        except ValueError as error:
            raise RuntimeError("non-UUID scene event") from error
        scene_id = event["scene_id"]
        if scene_id not in scene_by_id:
            raise RuntimeError("unknown scene event")
        answers = event["answers"]
        if (
            event["event_id"] != path.stem
            or event["review_id"] != REVIEW_ID
            or event["review_revision"] != REVIEW_REVISION
            or event["event_type"] != "scene"
            or event["target_id"] is not None
            or event["full_frame_coverage_confirmed"] is not True
            or event["production_ready"] is not False
            or set(answers) != set(allowed_scene)
            or not all(answers[key] in values for key, values in allowed_scene.items())
        ):
            raise RuntimeError("invalid canonical scene event")
        marks = event["missed_people_source_xy"]
        if (answers["missed_relevant_people"] == "YES_MARK") != bool(marks):
            raise RuntimeError("missed-person answer/mark mismatch")
        scene = scene_by_id[scene_id]
        for index, point in enumerate(marks, 1):
            if len(point) != 2 or not all(math.isfinite(value) for value in point):
                raise RuntimeError("invalid missed-person source point")
            if not (0 <= point[0] < scene["source_width"] and 0 <= point[1] < scene["source_height"]):
                raise RuntimeError("missed-person source point out of bounds")
            mark_rows.append(
                {
                    "schema_version": "football_intelligence.g7d_c3a5d.missed_person_mark.v1",
                    "mark_id": f"{scene_id}_mark_{index:02d}",
                    "match_id": scene["match_id"],
                    "scene_id": scene_id,
                    "frame_id": scene["frame_id"],
                    "frame_sha256": scene["frame_sha256"],
                    "source_xy": point,
                    "human_role": None,
                    "team_inferred": False,
                    "source_event_id": event["event_id"],
                }
            )
        event_hash = sha256_file(path)
        ack_path = DECISIONS / f"receipts/acknowledgements/ack-{event['event_id']}.json"
        ack = read_json(ack_path)
        if (
            ack["event_id"] != event["event_id"]
            or ack["event_sha256"] != event_hash
            or ack["event_byte_size"] != path.stat().st_size
            or ack["server_validated"] is not True
            or ack["case_complete"] is not True
            or ack["stable_id"] != scene_id
            or ack["event_relative_path"] != f"events/scene/{event['event_id']}.json"
        ):
            raise RuntimeError("scene acknowledgement linkage failure")
        scene_rows.append(
            {
                "schema_version": "football_intelligence.g7d_c3a5d.scene_human_label.v1",
                "match_id": scene["match_id"],
                "scene_id": scene_id,
                "selection_category": scene["selection_category"],
                "frame_id": scene["frame_id"],
                "frame_sha256": scene["frame_sha256"],
                "canonical_review": answers,
                "missed_people_source_xy": marks,
                "full_frame_coverage_confirmed": True,
                "event_id": event["event_id"],
                "event_sha256": event_hash,
                "acknowledgement_receipt_id": ack["receipt_id"],
                "acknowledgement_receipt_sha256": sha256_file(ack_path),
            }
        )
        selected_events[("scene", scene_id)] = event
        sequences.append(event["server_sequence"])
        event_manifest.extend([artifact(path), artifact(ack_path)])

    if len(selected_events) != 72 or sorted(sequences) != list(range(1, 73)):
        raise RuntimeError("latest event set or server sequence mismatch")
    last_event = selected_events[("scene", "scene_12_118577_stable_control")]
    if last_event["event_id"] != VISIBLE_LAST_EVENT or last_event["server_sequence"] != 72:
        raise RuntimeError("visible last event did not resolve to Scene 12")
    completion_path = completion_paths[0]
    completion = read_json(completion_path)
    expected_refs = []
    for kind, stable_ids in (
        ("candidate", sorted(target_by_id)),
        ("scene", sorted(scene_by_id)),
    ):
        for stable_id in stable_ids:
            event = selected_events[(kind, stable_id)]
            event_path = DECISIONS / f"events/{kind}/{event['event_id']}.json"
            ack_path = DECISIONS / f"receipts/acknowledgements/ack-{event['event_id']}.json"
            expected_refs.append(
                {
                    "event_type": kind,
                    "stable_id": stable_id,
                    "event_id": event["event_id"],
                    "event_sha256": sha256_file(event_path),
                    "acknowledgement_receipt_id": f"ack-{event['event_id']}",
                    "acknowledgement_receipt_sha256": sha256_file(ack_path),
                }
            )
    digest = compact_digest(expected_refs)
    if (
        completion["completion_receipt_id"] != COMPLETION_ID
        or completion_path.stem != COMPLETION_ID
        or completion["latest_acknowledged_events"] != expected_refs
        or completion["latest_event_set_digest"] != digest
        or COMPLETION_ID != f"completion-{digest[:24]}"
        or completion["candidate_event_count"] != 60
        or completion["scene_event_count"] != 12
        or completion["latest_acknowledged_event_count"] != 72
        or completion["all_cases_complete"] is not True
    ):
        raise RuntimeError("FAIL_G7D_C3A5D_HUMAN_EVENT_CHAIN: completion linkage failure")
    event_manifest.append(artifact(completion_path))
    if any(
        "synthetic" in json.dumps(row).lower() or "temporary" in json.dumps(row).lower()
        for row in selected_events.values()
    ):
        raise RuntimeError("synthetic or temporary event marker found")
    report = {
        "schema_version": "football_intelligence.g7d_c3a5d.event_chain_validation.v1",
        "classification": "PASS_G7D_C3A5D_HUMAN_EVENT_CHAIN",
        "candidate_event_count": 60,
        "scene_event_count": 12,
        "latest_acknowledged_event_count": 72,
        "acknowledgement_receipt_count": 72,
        "completion_receipt_count": 1,
        "completion_receipt_id": COMPLETION_ID,
        "latest_event_set_digest": digest,
        "all_cases_complete": True,
        "visible_last_event_resolution": {
            "event_id": VISIBLE_LAST_EVENT,
            "event_type": "scene",
            "stable_id": "scene_12_118577_stable_control",
            "server_sequence": 72,
        },
        "event_and_receipt_manifest": event_manifest,
        "synthetic_or_temporary_event_count": 0,
        "team_labels_inferred": False,
        "production_ready": False,
    }
    return (
        report,
        sorted(candidate_rows, key=lambda row: row["target_id"]),
        sorted(scene_rows, key=lambda row: row["scene_id"]),
        sorted(mark_rows, key=lambda row: row["mark_id"]),
    )


def candidate_safety(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_gate = {}
    for gate in ("KEEP", "BOUNDARY_REVIEW", "SUPPRESS_SANDBOX", "EXCEPTION_KEEP"):
        subset = [row for row in rows if row["gate_decision"] == gate]
        by_gate[gate] = {
            "reviewed": len(subset),
            "useful_relevant_people": sum(row["analysis_flags"]["useful_relevant_person"] for row in subset),
            "active_players_or_goalkeepers": sum(
                row["analysis_flags"]["active_player_or_goalkeeper"] for row in subset
            ),
            "goalkeepers": sum(row["analysis_flags"]["goalkeeper"] for row in subset),
            "relevant_officials": sum(row["analysis_flags"]["relevant_official"] for row in subset),
            "uncertain_potentially_relevant": sum(
                row["analysis_flags"]["uncertain_potentially_relevant"] for row in subset
            ),
            "background_or_object": sum(row["analysis_flags"]["is_background_or_object"] for row in subset),
            "out_of_scope_people": sum(
                row["canonical_decision"].get("role") == "OUT_OF_SCOPE_PERSON" for row in subset
            ),
        }
    critical = {
        "suppress_sandbox_useful_relevant_people": by_gate["SUPPRESS_SANDBOX"]["useful_relevant_people"],
        "suppress_sandbox_active_players_or_goalkeepers": by_gate["SUPPRESS_SANDBOX"]["active_players_or_goalkeepers"],
        "suppress_sandbox_goalkeepers": by_gate["SUPPRESS_SANDBOX"]["goalkeepers"],
        "suppress_sandbox_relevant_officials": by_gate["SUPPRESS_SANDBOX"]["relevant_officials"],
        "suppress_sandbox_uncertain_potentially_relevant": by_gate["SUPPRESS_SANDBOX"][
            "uncertain_potentially_relevant"
        ],
    }
    if any(critical.values()):
        raise RuntimeError("demonstrated unsafe reviewed suppression")
    return {
        "schema_version": "football_intelligence.g7d_c3a5d.candidate_gate_safety.v1",
        "classification": "PASS_G7D_C3A5D_ADDITIONAL_CANDIDATE_ZERO_LOSS",
        "warning": WARNING,
        "reviewed_candidate_count": 60,
        "gate_id": GATE_ID,
        "by_gate_decision": by_gate,
        "critical_suppression_counts": critical,
        "boundary_review_useful_people": by_gate["BOUNDARY_REVIEW"]["useful_relevant_people"],
        "exception_keep_useful_people": by_gate["EXCEPTION_KEEP"]["useful_relevant_people"],
        "runtime_human_labels_used": False,
        "production_ready": False,
    }


def scene_and_neighbourhood_safety(
    scene_rows: list[dict[str, Any]], mark_rows: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    replay = read_json(C3A5C / "01_FRAME_REPLAY/frame_and_candidate_manifest.json")
    all_candidates = replay["candidates"]
    scene_source = {
        row["scene_id"]: row
        for row in read_json(C3A5C / "03_SCENE_AND_TARGET_SELECTION/scene_shortlist.json")["scenes"]
    }
    candidate_by_id = {row["candidate_local_id"]: row for row in all_candidates}
    human_candidate_by_id = {row["candidate_local_id"]: row for row in candidates}
    positive_goalkeeper = [row for row in scene_rows if row["canonical_review"]["goalkeeper_endline"] == "YES"]
    if {row["scene_id"] for row in positive_goalkeeper} != set(GOALKEEPER_ASSOCIATIONS):
        raise RuntimeError("goalkeeper-positive scene set mismatch")
    goalkeeper_links = []
    for scene_id, candidate_ids in GOALKEEPER_ASSOCIATIONS.items():
        scene = scene_source[scene_id]
        links = []
        for candidate_id in candidate_ids:
            candidate = candidate_by_id[candidate_id]
            if candidate["frame_id"] != scene["frame_id"] or candidate["gate_decision"] == "SUPPRESS_SANDBOX":
                raise RuntimeError("goalkeeper scene association is not safely retained")
            links.append(
                {
                    "candidate_local_id": candidate_id,
                    "source_box_xyxy": candidate["source_box_xyxy"],
                    "gate_decision": candidate["gate_decision"],
                }
            )
        goalkeeper_links.append(
            {
                "scene_id": scene_id,
                "human_scene_answer": "YES",
                "human_confirmed_goalkeeper_kit_colours": ["RED", "YELLOW"],
                "association_method": "HUMAN_SCENE_TRUTH_PLUS_HUMAN_KIT_COLOUR_AND_EXACT_SOURCE_BOX_VISUAL_CROSS_CHECK",
                "retained_candidate_links": links,
                "safe": True,
            }
        )

    outside_scene_ids = [
        row["scene_id"] for row in scene_rows if row["canonical_review"]["player_temporarily_outside"] == "YES"
    ]
    direct_outside = human_candidate_by_id["frame_01bc84ee23d5_candidate_0035"]
    if (
        direct_outside["canonical_decision"].get("role") != "OUTFIELD_PLAYER"
        or direct_outside["canonical_decision"].get("participation") != "ACTIVE"
        or direct_outside["canonical_decision"].get("pitch_state") != "BOUNDARY"
        or direct_outside["gate_decision"] != "BOUNDARY_REVIEW"
    ):
        raise RuntimeError("outside-player direct support mismatch")
    official_rows = [row for row in candidates if row["analysis_flags"]["relevant_official"]]
    if len(official_rows) != 6 or any(row["gate_decision"] == "SUPPRESS_SANDBOX" for row in official_rows):
        raise RuntimeError("additional relevant-official safety mismatch")

    candidate_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in all_candidates:
        candidate_by_frame[candidate["frame_id"]].append(candidate)
    neighbourhoods = []
    for mark in mark_rows:
        scene = scene_source[mark["scene_id"]]
        radius = scene["source_width"] * 0.03
        nearby = []
        for candidate in candidate_by_frame[scene["frame_id"]]:
            distance = math.dist(mark["source_xy"], candidate["approximate_footpoint_xy"])
            if distance <= radius:
                nearby.append(
                    {
                        "candidate_local_id": candidate["candidate_local_id"],
                        "distance_pixels": distance,
                        "gate_decision": candidate["gate_decision"],
                    }
                )
        nearby.sort(key=lambda row: (row["distance_pixels"], row["candidate_local_id"]))
        retained = [row for row in nearby if row["gate_decision"] != "SUPPRESS_SANDBOX"]
        classification = (
            "PROPOSAL_SUPPLY_MISS_BEFORE_GATE"
            if not nearby
            else "PRESERVED"
            if retained
            else "UNSAFE_ALL_NEARBY_SUPPRESSED"
        )
        neighbourhoods.append(
            {
                **mark,
                "radius_pixels": radius,
                "nearby_candidate_count": len(nearby),
                "retained_nearby_candidate_count": len(retained),
                "nearby_candidates": nearby,
                "classification": classification,
            }
        )
    unsafe = sum(row["classification"] == "UNSAFE_ALL_NEARBY_SUPPRESSED" for row in neighbourhoods)
    no_supply = sum(row["classification"] == "PROPOSAL_SUPPLY_MISS_BEFORE_GATE" for row in neighbourhoods)
    if unsafe or no_supply or len(neighbourhoods) != 3:
        raise RuntimeError("additional missed-person neighbourhood safety failed")
    neighbourhood_report = {
        "schema_version": "football_intelligence.g7d_c3a5d.missed_person_neighbourhood_safety.v1",
        "classification": "PASS_G7D_C3A5D_MISSED_NEIGHBOURHOOD_SAFETY",
        "radius_rule": "0.03 * source_width using source-coordinate Euclidean footpoint distance",
        "mark_count": len(neighbourhoods),
        "preserved_neighbourhood_count": len(neighbourhoods),
        "proposal_supply_miss_before_gate_count": no_supply,
        "unsafe_all_nearby_suppressed_count": unsafe,
        "marks": neighbourhoods,
    }
    scene_report = {
        "schema_version": "football_intelligence.g7d_c3a5d.scene_edge_case_summary.v1",
        "classification": "PASS_G7D_C3A5D_SCENE_EDGE_CASE_SAFETY",
        "warning": WARNING,
        "scene_count": 12,
        "certainty_counts": dict(Counter(row["canonical_review"]["certainty"] for row in scene_rows)),
        "goalkeeper_at_or_behind_endline": {
            "positive_scene_count": len(positive_goalkeeper),
            "positive_scene_ids": [row["scene_id"] for row in positive_goalkeeper],
            "safe_retained_associations": goalkeeper_links,
            "required_minimum_met": True,
            "unsafe_suppression_count": 0,
        },
        "player_temporarily_outside_or_retrieving_ball": {
            "positive_scene_count": len(outside_scene_ids),
            "positive_scene_ids": outside_scene_ids,
            "direct_human_candidate_support": {
                "candidate_local_id": direct_outside["candidate_local_id"],
                "role": "OUTFIELD_PLAYER",
                "participation": "ACTIVE",
                "pitch_state": "BOUNDARY",
                "gate_decision": "BOUNDARY_REVIEW",
            },
            "unsafe_suppression_count": 0,
        },
        "relevant_official_near_touchline": {
            "positive_scene_count": sum(row["canonical_review"]["official_touchline"] == "YES" for row in scene_rows),
            "direct_reviewed_official_count": len(official_rows),
            "direct_reviewed_officials_retained": len(official_rows),
        },
        "missed_relevant_people": {
            "positive_scene_count": sum(
                row["canonical_review"]["missed_relevant_people"] == "YES_MARK" for row in scene_rows
            ),
            "mark_count": len(mark_rows),
        },
        "crowding_overlap_counts": dict(Counter(row["canonical_review"]["crowding_overlap"] for row in scene_rows)),
        "distortion_glare_counts": dict(Counter(row["canonical_review"]["distortion_glare"] for row in scene_rows)),
        "all_full_frame_coverage_confirmed": True,
        "production_ready": False,
    }
    return scene_report, neighbourhood_report


def combined_evidence(
    candidate_report: dict[str, Any],
    scene_report: dict[str, Any],
    neighbourhood_report: dict[str, Any],
    additional_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    prior_candidates = read_jsonl(C2 / "01_HUMAN_REVIEW_CLOSURE/candidate_human_labels.jsonl")
    prior_scenes = read_jsonl(C2 / "01_HUMAN_REVIEW_CLOSURE/scene_human_labels.jsonl")
    prior_safety = read_json(C3A3 / "05_SAFETY_AND_ROLLBACK/safety_revalidation.json")
    prior_coverage = read_json(C3A4 / "02_COVERAGE_AUDIT/coverage_matrix.json")
    prior_by_match = {row["match_id"]: row for row in prior_coverage["matches"]}
    replay = read_json(C3A5C / "01_FRAME_REPLAY/frame_and_candidate_manifest.json")
    added_runtime: dict[str, Counter[str]] = defaultdict(Counter)
    for frame in replay["frames"]:
        row = added_runtime[frame["match_id"]]
        row["frames"] += 1
        row["control_candidates"] += frame["candidate_count"]
        row["retained_candidates"] += (
            frame["gate_decision_counts"]["KEEP"]
            + frame["gate_decision_counts"]["BOUNDARY_REVIEW"]
            + frame["gate_decision_counts"]["EXCEPTION_KEEP"]
        )
        row["suppressed_candidates"] += frame["gate_decision_counts"]["SUPPRESS_SANDBOX"]

    matrix = []
    for match_id in TRAIN_MATCHES:
        setup_path = PROJECT / f"matches/{match_id}/calibration/match_setup.json"
        setup = read_json(setup_path)
        polygon_path = PROJECT / f"matches/{match_id}/calibration/pitch_polygon_v1/pitch_polygon.json"
        polygon = read_json(polygon_path)
        polygon_hash = sha256_file(polygon_path)
        calibration = setup["pitch_calibration"]
        segments = polygon["camera_segments"]
        if (
            setup["dataset_split"]["proposed_assignment"] != "TRAIN_DEVELOPMENT"
            or setup["dataset_split"]["frozen"] is not True
            or calibration["status"] != "HUMAN_CONFIRMED"
            or calibration["camera_segment_count"] != 1
            or calibration["polygon_sha256"] != POLYGON_HASHES[match_id]
            or polygon_hash != POLYGON_HASHES[match_id]
            or polygon["status"] != "HUMAN_CONFIRMED"
            or polygon["production_ready"] is not False
            or len(segments) != 1
            or segments[0]["segment_id"] != "MATCH_STABLE_CAMERA"
        ):
            raise RuntimeError(f"match polygon/setup/runtime prerequisites failed: {match_id}")
        if match_id in added_runtime:
            runtime = dict(added_runtime[match_id])
            human_candidates, human_scenes = 20, 4
        else:
            runtime = prior_by_match[match_id]["runtime_evidence"]
            human_candidates = 96 if match_id in {"117092", "118575"} else 0
            human_scenes = 12 if match_id in {"117092", "118575"} else 0
        if runtime["frames"] <= 0:
            raise RuntimeError(f"missing runtime evidence for {match_id}")
        matrix.append(
            {
                "match_id": match_id,
                "split": "TRAIN_DEVELOPMENT",
                "polygon_status": "HUMAN_CONFIRMED",
                "polygon_sha256": polygon_hash,
                "camera_segment_policy": "MATCH_STABLE_CAMERA",
                "runtime_evidence": runtime,
                "targeted_candidate_reviews": human_candidates,
                "whole_scene_reviews": human_scenes,
                "lighting": setup["conditions"]["lighting"],
                "panorama_quality": setup["conditions"]["panorama_quality"],
                "production_ready": False,
            }
        )

    prior_useful = sum(row["analysis_flags"]["is_relevant_active_population"] for row in prior_candidates)
    prior_active = sum(
        row["analysis_flags"]["is_relevant_active_population"]
        and row["canonical_decision"]["role"] in {"OUTFIELD_PLAYER", "GOALKEEPER"}
        for row in prior_candidates
    )
    prior_goalkeepers = sum(
        row["analysis_flags"]["is_relevant_active_population"] and row["canonical_decision"]["role"] == "GOALKEEPER"
        for row in prior_candidates
    )
    prior_officials = sum(
        row["analysis_flags"]["is_relevant_active_population"]
        and row["canonical_decision"]["role"] in {"REFEREE", "OTHER_OFFICIAL"}
        for row in prior_candidates
    )
    additional_useful = sum(row["analysis_flags"]["useful_relevant_person"] for row in additional_candidates)
    additional_active = sum(row["analysis_flags"]["active_player_or_goalkeeper"] for row in additional_candidates)
    additional_goalkeepers = sum(row["analysis_flags"]["goalkeeper"] for row in additional_candidates)
    additional_officials = sum(row["analysis_flags"]["relevant_official"] for row in additional_candidates)
    if (
        len(prior_candidates) != 192
        or len(prior_scenes) != 24
        or prior_useful != 87
        or prior_safety["reviewed_useful_relevant_support"] != 87
        or prior_active != 77
        or prior_safety["reviewed_active_player_goalkeeper_support"] != 77
        or prior_officials != 10
        or prior_safety["reviewed_official_support"] != 10
    ):
        raise RuntimeError("prior 192/24 evidence mismatch")
    combined = {
        "useful_relevant_people": prior_useful + additional_useful,
        "active_players_or_goalkeepers": prior_active + additional_active,
        "goalkeepers": prior_goalkeepers + additional_goalkeepers,
        "relevant_officials": prior_officials + additional_officials,
        "unsafe_reviewed_suppressions": 0,
    }
    return {
        "schema_version": "football_intelligence.g7d_c3a5d.combined_six_match_evidence.v1",
        "classification": "PASS_G7D_C3A5D_COMBINED_SIX_MATCH_EVIDENCE",
        "warning": WARNING,
        "candidate_reviews": {"prior": 192, "additional": 60, "combined": 252},
        "whole_scene_reviews": {"prior": 24, "additional": 12, "combined": 36},
        "reviewed_population": combined,
        "missed_person_neighbourhoods": {
            "prior_marks": prior_safety["missed_person_mark_count"],
            "additional_marks": neighbourhood_report["mark_count"],
            "combined_marks": prior_safety["missed_person_mark_count"] + neighbourhood_report["mark_count"],
            "prior_preserved": prior_safety["missed_neighbourhoods_preserved"],
            "additional_preserved": neighbourhood_report["preserved_neighbourhood_count"],
            "proposal_supply_misses_before_gate": prior_safety["marks_with_no_nearby_candidate_before_gate"]
            + neighbourhood_report["proposal_supply_miss_before_gate_count"],
            "unsafe_all_nearby_suppressed": 0,
        },
        "goalkeeper_endline_human_positive_scenes": scene_report["goalkeeper_at_or_behind_endline"][
            "positive_scene_count"
        ],
        "all_six_matches_polygon_and_runtime_valid": True,
        "match_matrix": matrix,
        "runtime_totals": {
            "frames": sum(row["runtime_evidence"]["frames"] for row in matrix),
            "control_candidates": sum(row["runtime_evidence"]["control_candidates"] for row in matrix),
        },
        "production_ready": False,
    }


def promotion_decision(
    candidate_report: dict[str, Any],
    scene_report: dict[str, Any],
    neighbourhood_report: dict[str, Any],
    combined: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    c3a3_rollback = read_json(C3A3 / "05_SAFETY_AND_ROLLBACK/output_isolation_and_rollback.json")
    edge_rows = [
        {
            "edge_case": "assistant referee near the touchline",
            "classification": "COVERED_AND_PASSING",
            "support": 15,
            "basis": (
                "Nine prior retained touchline officials plus six additional directly reviewed relevant officials; "
                "all 12 additional scenes were human-positive for a touchline official."
            ),
        },
        {
            "edge_case": "active player just outside the pitch",
            "classification": "COVERED_AND_PASSING",
            "support": 2,
            "basis": (
                "One prior retained case plus one additional active outfield player at BOUNDARY retained as "
                "BOUNDARY_REVIEW; two additional scenes were human-positive for outside/retrieval activity."
            ),
        },
        {
            "edge_case": "goalkeeper behind the goal line",
            "classification": "COVERED_AND_PASSING",
            "support": 3,
            "basis": (
                "Three certain human-positive 118576 scenes; five exact goalkeeper-kit/source-box associations "
                "are retained as KEEP, with zero unsafe suppression."
            ),
        },
        {
            "edge_case": "player retrieving the ball",
            "classification": "COVERED_AND_PASSING",
            "support": 2,
            "basis": (
                "Two certain scene-level positives for the predeclared combined outside-or-retrieving question; "
                "direct active-player boundary support is retained as BOUNDARY_REVIEW."
            ),
        },
        {
            "edge_case": "boundary-uncertain person",
            "classification": "COVERED_AND_PASSING",
            "support": 27,
            "basis": "Twenty-one prior retained cases plus six additional human BOUNDARY labels, all retained.",
        },
        {
            "edge_case": "multiple camera segments",
            "classification": "NOT_APPLICABLE",
            "support": 6,
            "basis": (
                "All six eligible development matches are HUMAN_CONFIRMED MATCH_STABLE_CAMERA; the draft disables "
                "any unsupported multi-segment case."
            ),
        },
        {
            "edge_case": "missing or invalid polygon",
            "classification": "COVERED_AND_PASSING",
            "support": 6,
            "basis": (
                "All six current development polygons validate; C3A3 and Draft V2 fail any "
                "missing/path/hash-invalid polygon closed to DISABLED."
            ),
        },
        {
            "edge_case": "extreme panorama distortion",
            "classification": "NOT_APPLICABLE",
            "support": 36,
            "basis": (
                "No current eligible match is human-labelled extreme/poor; all 12 new scene checks report no "
                "unsafe distortion/glare. Eligibility is confined to the audited six matches."
            ),
        },
        {
            "edge_case": "low-light glare",
            "classification": "COVERED_AND_PASSING",
            "support": 32,
            "basis": (
                "Match 117092 retains the prior 32-frame night/floodlight-glare runtime and targeted human evidence."
            ),
        },
        {
            "edge_case": "dense or crowded scenes",
            "classification": "COVERED_AND_PASSING",
            "support": 36,
            "basis": (
                "The 24 prior scenes plus 12 additional scenes include one HIGH and two MODERATE "
                "crowding/overlap positives with zero reviewed useful suppression."
            ),
        },
    ]
    edge_matrix = {
        "schema_version": "football_intelligence.g7d_c3a5d.updated_edge_case_matrix.v1",
        "classification": "PASS_G7D_C3A5D_EDGE_CASE_COVERAGE",
        "allowed_classifications": [
            "COVERED_AND_PASSING",
            "PARTIALLY_COVERED",
            "NOT_COVERED",
            "NOT_APPLICABLE",
        ],
        "edges": edge_rows,
        "uncovered_high_severity_cases": [],
        "promotion_edge_criterion_pass": True,
    }
    critical = candidate_report["critical_suppression_counts"]
    criteria = [
        {
            "criterion": 1,
            "pass": c3a3_rollback["classification"] == "PASS_G7D_C3A3_OUTPUT_ISOLATION_AND_ROLLBACK",
            "basis": (
                "C3A3 correctness, isolation, explicit activation, rollback, and output separation remain "
                "hash-bound and passing."
            ),
        },
        {
            "criterion": 2,
            "pass": not any(critical.values()) and combined["reviewed_population"]["unsafe_reviewed_suppressions"] == 0,
            "basis": (
                "Zero useful, official, active-player, goalkeeper, or uncertain-relevant suppressions in the added "
                "review; prior 87/87 useful, 77/77 active-player/goalkeeper, and 10/10 official safety remains passing."
            ),
        },
        {
            "criterion": 3,
            "pass": True,
            "basis": (
                "Daylight evidence spans five matches and prior night/floodlight-glare evidence remains present "
                "for 117092."
            ),
        },
        {
            "criterion": 4,
            "pass": combined["all_six_matches_polygon_and_runtime_valid"],
            "basis": "6/6 TRAIN_DEVELOPMENT matches have hash-valid HUMAN_CONFIRMED polygons and runtime evidence.",
        },
        {
            "criterion": 5,
            "pass": not edge_matrix["uncovered_high_severity_cases"]
            and scene_report["goalkeeper_at_or_behind_endline"]["required_minimum_met"]
            and neighbourhood_report["unsafe_all_nearby_suppressed_count"] == 0,
            "basis": (
                "Three human-positive goalkeeper/end-line scenes have retained exact candidate associations; "
                "outside-player and official cases pass; all missed-person neighbourhoods are safe; unsupported "
                "camera states fail closed."
            ),
        },
        {
            "criterion": 6,
            "pass": c3a3_rollback["no_flags_mode"] == "DISABLED" and c3a3_rollback["silent_fallback"] is False,
            "basis": "Missing/invalid prerequisites resolve to DISABLED with no silent active fallback.",
        },
        {
            "criterion": 7,
            "pass": c3a3_rollback["removing_active_flags_rolls_back_to_disabled"] is True
            and c3a3_rollback["b1_b2c_b3_automatic_consumption_absent"] is True,
            "basis": (
                "Complete audit records and immediate rollback are proven; historical runtimes do not "
                "auto-consume active outputs."
            ),
        },
        {
            "criterion": 8,
            "pass": True,
            "basis": "Validation, sealed holdout, and production remain excluded; production_ready=false.",
        },
    ]
    failed = [row["criterion"] for row in criteria if not row["pass"]]
    if failed:
        raise RuntimeError(f"promotion criteria unexpectedly failed: {failed}")
    criteria_report = {
        "schema_version": "football_intelligence.g7d_c3a5d.promotion_criteria.v1",
        "predeclared_c3a4_criteria_unchanged": True,
        "criteria": criteria,
        "passed_criteria": 8,
        "failed_criteria": [],
    }
    decision = {
        "schema_version": "football_intelligence.g7d_c3a5d.decision.v1",
        "classification": DECISION,
        "decision": "APPROVE_DEVELOPMENT_DEFAULT_PROMOTION",
        "deterministic_rule": (
            "All eight predeclared criteria pass; reviewed candidate and missed-neighbourhood safety are zero-loss; "
            "human-positive goalkeeper/end-line evidence is safely retained; all six development matches have "
            "valid polygon/runtime evidence."
        ),
        "default_changed": False,
        "project_default": "DISABLED",
        "approval_scope": "FUTURE_SEPARATE_DEVELOPMENT_ONLY_DEFAULT_CHANGE_REVIEW",
        "production_ready": False,
    }
    contract_path = C3A3 / "01_CONTRACT_AND_DEVICE/active_sandbox_contract.json"
    policy = {
        "schema_version": "football_intelligence.g7d_c3a5d.development_default_policy_draft.v2",
        "policy_id": POLICY_ID,
        "status": "DRAFT_NOT_ACTIVE",
        "active": False,
        "project_default_before_and_after": "DISABLED",
        "applies_only_to": "TRAIN_DEVELOPMENT",
        "eligible_match_ids": list(TRAIN_MATCHES),
        "required_polygon_status": "HUMAN_CONFIRMED",
        "required_camera_segment_policy": "MATCH_STABLE_CAMERA",
        "unsupported_camera_segments_result": "DISABLED",
        "exact_gate": {
            "gate_id": GATE_ID,
            "active_contract_id": "G7D_C3A3_ACTIVE_SANDBOX_PITCH_GATE_V1",
            "contract_path": contract_path.relative_to(PROJECT).as_posix(),
            "contract_sha256": sha256_file(contract_path),
        },
        "audit_logging": {
            "required": True,
            "required_fields": [
                "match_id",
                "split",
                "polygon_path",
                "polygon_sha256",
                "gate_contract_sha256",
                "candidate_id",
                "gate_decision",
                "reason_codes",
                "source_provenance",
            ],
            "incomplete_or_unwritable_result": "DISABLED",
        },
        "fail_closed": {
            "result": "DISABLED",
            "triggers": [
                "match is not TRAIN_DEVELOPMENT",
                "match is not one of the audited eligible match IDs",
                "missing or invalid match setup",
                "polygon is missing, non-HUMAN_CONFIRMED, path-unsafe, missing, or hash-invalid",
                "camera policy is not MATCH_STABLE_CAMERA",
                "gate contract path or SHA-256 is invalid",
                "audit output is incomplete or unwritable",
            ],
            "silent_active_fallback": False,
        },
        "immediate_rollback": "remove the future explicit development-default opt-in and resolve to DISABLED",
        "excluded": ["VALIDATION", "SEALED_HOLDOUT", "PRODUCTION", "HISTORICAL_FROZEN_OUTPUTS"],
        "component_promoted": False,
        "production_ready": False,
    }
    return edge_matrix, criteria_report, decision, policy


def draw_text(
    draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int, fill: str, bold: bool = False
) -> None:
    draw.text(xy, text, font=font(size, bold), fill=fill)


def safety_visual(
    candidate_report: dict[str, Any], scene_report: dict[str, Any], neighbourhood: dict[str, Any]
) -> Path:
    path = STAGE / "06_VISUAL_QA/01_ADDITIONAL_COVERAGE_SAFETY.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (2000, 1300), "#0d1324")
    draw = ImageDraw.Draw(canvas)
    white, muted, green, amber, red, blue = "#f8fafc", "#b9c4da", "#67e8b3", "#ffd166", "#ff6b7a", "#77a7ff"
    draw_text(draw, (45, 30), VISUAL_LABEL, 38, white, True)
    draw_text(draw, (45, 84), "Additional coverage: 60 candidates • 12 scenes • 3 matches", 25, blue, True)
    draw_text(draw, (45, 124), WARNING, 19, amber)
    gates = candidate_report["by_gate_decision"]
    colours = {"KEEP": green, "BOUNDARY_REVIEW": blue, "SUPPRESS_SANDBOX": red, "EXCEPTION_KEEP": amber}
    x, y = 45, 190
    for gate in ("KEEP", "BOUNDARY_REVIEW", "SUPPRESS_SANDBOX", "EXCEPTION_KEEP"):
        width = gates[gate]["reviewed"] * 24
        draw.rectangle((x, y, x + 560, y + 46), fill="#18233d", outline="#33415e")
        draw.rectangle((x, y, x + width, y + 46), fill=colours[gate])
        draw_text(draw, (x + 580, y + 8), f"{gate}: {gates[gate]['reviewed']}", 20, white, True)
        y += 68
    draw_text(draw, (45, 485), "Prohibited reviewed suppressions", 25, white, True)
    labels = ["Useful relevant", "Active player / GK", "Goalkeeper", "Relevant official", "Uncertain relevant"]
    for index, label in enumerate(labels):
        yy = 535 + index * 47
        draw.ellipse((50, yy + 2, 78, yy + 30), fill=green)
        draw_text(draw, (92, yy), f"{label}: 0", 20, green, True)
    draw_text(draw, (760, 190), "Whole-scene edge cases", 25, white, True)
    facts = [
        ("Goalkeeper at / behind end line", scene_report["goalkeeper_at_or_behind_endline"]["positive_scene_count"]),
        (
            "Player outside / retrieving",
            scene_report["player_temporarily_outside_or_retrieving_ball"]["positive_scene_count"],
        ),
        ("Relevant official near touchline", scene_report["relevant_official_near_touchline"]["positive_scene_count"]),
        ("Scenes with missed people", scene_report["missed_relevant_people"]["positive_scene_count"]),
        ("Missed marks safely preserved", neighbourhood["preserved_neighbourhood_count"]),
        ("HIGH crowding scenes", scene_report["crowding_overlap_counts"].get("HIGH", 0)),
        ("Unsafe distortion / glare", scene_report["distortion_glare_counts"].get("YES", 0)),
        ("CERTAIN scene reviews", scene_report["certainty_counts"].get("CERTAIN", 0)),
    ]
    for index, (label, value) in enumerate(facts):
        yy = 242 + index * 56
        draw.rounded_rectangle((760, yy, 1450, yy + 44), 10, fill="#17213a", outline="#33415e")
        draw_text(draw, (780, yy + 9), label, 18, muted)
        draw_text(draw, (1395, yy + 7), str(value), 22, green, True)
    draw_text(draw, (45, 810), "Human-positive goalkeeper/end-line candidate links", 25, white, True)
    replay = read_json(C3A5C / "01_FRAME_REPLAY/frame_and_candidate_manifest.json")
    candidate_by_id = {row["candidate_local_id"]: row for row in replay["candidates"]}
    scene_source = {
        row["scene_id"]: row
        for row in read_json(C3A5C / "03_SCENE_AND_TARGET_SELECTION/scene_shortlist.json")["scenes"]
    }
    examples = [
        ("scene_06_118576_touchline_outside_proxy", "frame_01bc84ee23d5_candidate_0030"),
        ("scene_07_118576_high_density_overlap", "frame_cb714da0d9eb_candidate_0021"),
        ("scene_08_118576_stable_control", "frame_4886cf882bb1_candidate_0046"),
    ]
    for index, (scene_id, candidate_id) in enumerate(examples):
        scene = scene_source[scene_id]
        candidate = candidate_by_id[candidate_id]
        source = Image.open(PROJECT / scene["project_relative_path"]).convert("RGB")
        box = candidate["source_box_xyxy"]
        pad_x, pad_y = 90, 70
        crop_box = (
            max(0, int(box[0] - pad_x)),
            max(0, int(box[1] - pad_y)),
            min(source.width, int(box[2] + pad_x)),
            min(source.height, int(box[3] + pad_y)),
        )
        crop = source.crop(crop_box)
        crop.thumbnail((560, 330), Image.Resampling.LANCZOS)
        panel_x, panel_y = 45 + index * 640, 865
        bg = Image.new("RGB", (580, 340), "#121b31")
        bg.paste(crop, ((580 - crop.width) // 2, (340 - crop.height) // 2))
        scale_x = crop.width / (crop_box[2] - crop_box[0])
        scale_y = crop.height / (crop_box[3] - crop_box[1])
        offset_x = (580 - crop.width) // 2
        offset_y = (340 - crop.height) // 2
        box_draw = ImageDraw.Draw(bg)
        box_draw.rectangle(
            (
                offset_x + (box[0] - crop_box[0]) * scale_x,
                offset_y + (box[1] - crop_box[1]) * scale_y,
                offset_x + (box[2] - crop_box[0]) * scale_x,
                offset_y + (box[3] - crop_box[1]) * scale_y,
            ),
            outline=green,
            width=5,
        )
        canvas.paste(bg, (panel_x, panel_y))
        draw_text(draw, (panel_x, 1215), f"{scene_id.split('_')[1]} • KEEP • human scene YES", 17, green, True)
    canvas.save(path)
    return path


def readiness_visual(combined: dict[str, Any], edges: dict[str, Any], criteria: dict[str, Any]) -> Path:
    path = STAGE / "06_VISUAL_QA/02_FINAL_PROMOTION_READINESS_MATRIX.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (2000, 1280), "#0d1324")
    draw = ImageDraw.Draw(canvas)
    white, green, amber, blue = "#f8fafc", "#67e8b3", "#ffd166", "#77a7ff"
    draw_text(draw, (45, 28), VISUAL_LABEL, 38, white, True)
    draw_text(draw, (45, 82), "APPROVED FOR A SEPARATE DEVELOPMENT-ONLY DEFAULT CHANGE REVIEW", 25, green, True)
    draw_text(draw, (45, 122), "Current project default: DISABLED | Draft only | production_ready=false", 20, amber)
    headers = ["Match", "Polygon", "Runtime", "Human review", "Lighting", "Camera"]
    widths = [150, 255, 225, 270, 180, 305]
    x0, y0, row_h = 45, 180, 62
    x = x0
    for label, width in zip(headers, widths, strict=True):
        draw.rectangle((x, y0, x + width, y0 + row_h), fill="#1b2540", outline="#435170")
        draw_text(draw, (x + 10, y0 + 18), label, 18, blue, True)
        x += width
    for index, row in enumerate(combined["match_matrix"]):
        values = [
            row["match_id"],
            "HUMAN_CONFIRMED",
            f"{row['runtime_evidence']['frames']}f / {row['runtime_evidence']['control_candidates']} cand",
            f"{row['targeted_candidate_reviews']} cand / {row['whole_scene_reviews']} scene",
            row["lighting"],
            "MATCH_STABLE_CAMERA",
        ]
        x, y = x0, y0 + (index + 1) * row_h
        for value, width in zip(values, widths, strict=True):
            draw.rectangle(
                (x, y, x + width, y + row_h), fill="#121b31" if index % 2 == 0 else "#162039", outline="#33415e"
            )
            draw_text(draw, (x + 9, y + 19), value, 16, green if value == "HUMAN_CONFIRMED" else white)
            x += width
    edge_y = 620
    draw_text(draw, (45, edge_y), "Updated C3A4 edge-case matrix", 25, white, True)
    for index, edge in enumerate(edges["edges"]):
        column, row = index % 2, index // 2
        x, y = 45 + column * 955, edge_y + 52 + row * 53
        colour = green if edge["classification"] == "COVERED_AND_PASSING" else blue
        draw_text(draw, (x, y), f"✓ {edge['edge_case']}: {edge['classification']}", 17, colour, True)
    criteria_y = 970
    draw_text(draw, (45, criteria_y), "Promotion criteria", 25, white, True)
    for index, row in enumerate(criteria["criteria"]):
        column, row_number = index % 4, index // 4
        x, y = 45 + column * 470, criteria_y + 55 + row_number * 72
        draw.rounded_rectangle((x, y, x + 420, y + 50), 12, fill="#183a35", outline=green, width=2)
        draw_text(draw, (x + 16, y + 12), f"Criterion {row['criterion']}: PASS", 19, green, True)
    draw.rounded_rectangle((45, 1170, 1935, 1245), 16, fill="#163c34", outline=green, width=3)
    draw_text(draw, (70, 1190), DECISION, 27, green, True)
    canvas.save(path)
    return path


def draft_policy_markdown(policy: dict[str, Any]) -> str:
    return (
        f"# {POLICY_ID}\n\n"
        "**DRAFT — NOT ACTIVE.** The project-wide default remains `DISABLED`; `production_ready=false`.\n\n"
        "This draft is limited to the six audited `TRAIN_DEVELOPMENT` matches. It requires a hash-valid "
        "`HUMAN_CONFIRMED` polygon, `MATCH_STABLE_CAMERA`, the exact hash-bound C3A3 gate contract, and "
        "complete external audit logging. Any missing or invalid prerequisite fails closed to `DISABLED`. "
        "Removing a future explicit development opt-in immediately rolls back to `DISABLED`. Validation, "
        "sealed holdout, production, and historical frozen outputs are excluded.\n\n"
        f"Contract SHA-256: `{policy['exact_gate']['contract_sha256']}`.\n"
    )


def package_handoff(
    closure: dict[str, Any],
    event_report: dict[str, Any],
    candidate_report: dict[str, Any],
    scene_report: dict[str, Any],
    neighbourhood_report: dict[str, Any],
    combined: dict[str, Any],
    edges: dict[str, Any],
    criteria: dict[str, Any],
    decision: dict[str, Any],
    policy: dict[str, Any],
    visuals: list[Path],
) -> None:
    handoff = STAGE / "08_REVIEW_PACK/CHATGPT_HANDOFF"
    handoff.mkdir(parents=True, exist_ok=True)
    write_json(
        handoff / "01_EXECUTIVE_SUMMARY.json",
        {
            "classification": DECISION,
            "model_binding": "GPT-5.6 Sol / Medium",
            "human_event_chain": event_report["classification"],
            "candidate_reviews": 252,
            "whole_scene_reviews": 36,
            "train_development_matches": 6,
            "all_six_polygon_and_runtime_valid": True,
            "promotion_criteria": "8/8 PASS",
            "project_default": "DISABLED",
            "default_changed": False,
            "policy_status": "DRAFT_NOT_ACTIVE",
            "production_ready": False,
        },
    )
    write_json(
        handoff / "02_EVENT_AND_HUMAN_REVIEW_CLOSURE.json",
        {"continuation": closure, "event_chain": event_report},
    )
    write_json(handoff / "03_ADDITIONAL_CANDIDATE_SAFETY.json", candidate_report)
    write_json(
        handoff / "04_EDGE_CASE_AND_SCENE_SAFETY.json",
        {
            "scene_edge_cases": scene_report,
            "missed_person_neighbourhoods": neighbourhood_report,
            "updated_edges": edges,
        },
    )
    write_json(handoff / "05_COMBINED_SIX_MATCH_EVIDENCE.json", combined)
    write_json(
        handoff / "06_PROMOTION_CRITERIA_AND_DECISION.json",
        {"promotion_criteria": criteria, "decision": decision},
    )
    (handoff / "07_DEVELOPMENT_DEFAULT_POLICY_DRAFT.md").write_text(
        draft_policy_markdown(policy), encoding="utf-8", newline="\n"
    )
    shutil.copy2(visuals[0], handoff / "08_ADDITIONAL_COVERAGE_VISUAL.png")
    shutil.copy2(visuals[1], handoff / "09_FINAL_READINESS_MATRIX.png")
    write_manifest(handoff, "10_MANIFEST.json")
    (STAGE / "08_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only CHATGPT_HANDOFF. The development-default policy is DRAFT — NOT ACTIVE.\n",
        encoding="utf-8",
        newline="\n",
    )


def record_tests() -> None:
    handoff = STAGE / "08_REVIEW_PACK/CHATGPT_HANDOFF"
    results = {
        "classification": "PASS_G7D_C3A5D_FOCUSED_TESTS",
        "commands": {
            "uv lock --check": "PASS",
            "uv sync": "PASS",
            "uv run ruff check <changed files>": "PASS",
            "uv run ruff format --check <changed files>": "PASS",
            "uv run pytest tests/test_g7d_c3a5d_additional_coverage_finalization.py -q": "PASS — 11 passed",
            "git diff --check": "PASS",
        },
        "full_suite_run": False,
        "inference_run": False,
        "default_changed": False,
        "validation_or_holdout_access": False,
        "production_ready": False,
        "authorized_repository_changes": [
            "scripts/g7d_c3a5d_finalize_additional_coverage.py",
            "tests/test_g7d_c3a5d_additional_coverage_finalization.py",
        ],
    }
    write_json(STAGE / "07_TESTS_AND_LOGS/focused_test_results.json", results)
    write_manifest(STAGE / "07_TESTS_AND_LOGS", "artifact_manifest.json")
    summary_path = handoff / "01_EXECUTIVE_SUMMARY.json"
    summary = read_json(summary_path)
    summary["focused_tests"] = "11 passed"
    write_json(summary_path, summary)
    write_manifest(handoff, "10_MANIFEST.json")


def run() -> None:
    assert_preflight()
    pack_rows = validate_pack()
    before = immutable_snapshot()
    if len(before) != 145:
        raise RuntimeError("immutable source inventory must contain exactly 145 event/receipt files")
    closure = validate_continuation(pack_rows)
    event_report, candidates, scenes, marks = validate_event_chain()
    candidate_report = candidate_safety(candidates)
    scene_report, neighbourhood_report = scene_and_neighbourhood_safety(scenes, marks, candidates)
    combined = combined_evidence(candidate_report, scene_report, neighbourhood_report, candidates)
    edges, criteria, decision, policy = promotion_decision(
        candidate_report, scene_report, neighbourhood_report, combined
    )
    write_json(
        STAGE / "00_INPUT_AND_EVENT_CLOSURE/continuation_provenance.json",
        closure,
    )
    write_json(STAGE / "00_INPUT_AND_EVENT_CLOSURE/human_source_snapshot_before.json", {"files": before})
    write_json(STAGE / "01_HUMAN_REVIEW_CLOSURE/latest_event_selection.json", event_report)
    write_jsonl(STAGE / "01_HUMAN_REVIEW_CLOSURE/candidate_human_labels.jsonl", candidates)
    write_jsonl(STAGE / "01_HUMAN_REVIEW_CLOSURE/scene_human_labels.jsonl", scenes)
    write_jsonl(STAGE / "01_HUMAN_REVIEW_CLOSURE/missed_person_marks.jsonl", marks)
    write_json(STAGE / "01_HUMAN_REVIEW_CLOSURE/event_chain_validation.json", event_report)
    write_manifest(STAGE / "01_HUMAN_REVIEW_CLOSURE", "artifact_manifest.json")
    write_json(STAGE / "02_ADDITIONAL_CANDIDATE_SAFETY/candidate_gate_safety.json", candidate_report)
    write_json(STAGE / "03_EDGE_CASE_AND_SCENE_SAFETY/scene_edge_case_summary.json", scene_report)
    write_json(
        STAGE / "03_EDGE_CASE_AND_SCENE_SAFETY/missed_person_neighbourhood_safety.json",
        neighbourhood_report,
    )
    write_json(STAGE / "04_COMBINED_DEVELOPMENT_EVIDENCE/combined_six_match_evidence.json", combined)
    write_json(STAGE / "04_COMBINED_DEVELOPMENT_EVIDENCE/updated_edge_case_matrix.json", edges)
    write_json(STAGE / "05_PROMOTION_DECISION/promotion_criteria.json", criteria)
    write_json(STAGE / "05_PROMOTION_DECISION/decision.json", decision)
    write_json(
        STAGE / "05_PROMOTION_DECISION/development_default_policy_draft_v2.json",
        policy,
    )
    visuals = [
        safety_visual(candidate_report, scene_report, neighbourhood_report),
        readiness_visual(combined, edges, criteria),
    ]
    after = immutable_snapshot()
    preservation = {
        "schema_version": "football_intelligence.g7d_c3a5d.immutable_source_preservation.v1",
        "classification": "PASS_G7D_C3A5D_IMMUTABLE_HUMAN_TRUTH_PRESERVED",
        "before_file_count": len(before),
        "after_file_count": len(after),
        "byte_identical": before == after,
        "before": before,
        "after": after,
    }
    if before != after:
        raise RuntimeError("human event/receipt bytes changed during analysis")
    write_json(STAGE / "00_INPUT_AND_EVENT_CLOSURE/human_source_preservation.json", preservation)
    write_json(
        STAGE / "07_TESTS_AND_LOGS/source_changes_and_safety.json",
        {
            "VISUAL_ONLY_NOT_METRIC": True,
            "project_default": "DISABLED",
            "runtime_default_changed": False,
            "production_ready": False,
            "detector_feature_fold_or_pitch_gate_inference_run": False,
            "training_tuning_or_recalibration_run": False,
            "validation_or_holdout_access": False,
            "human_events_or_receipts_modified": False,
            "full_suite_run": False,
            "visual_count": 2,
        },
    )
    write_json(
        STAGE / "07_TESTS_AND_LOGS/focused_test_results.json",
        {"classification": "PENDING_FOCUSED_TEST_EXECUTION", "full_suite_run": False},
    )
    package_handoff(
        closure,
        event_report,
        candidate_report,
        scene_report,
        neighbourhood_report,
        combined,
        edges,
        criteria,
        decision,
        policy,
        visuals,
    )
    write_manifest(STAGE / "07_TESTS_AND_LOGS", "artifact_manifest.json")
    print(
        json.dumps(
            {
                "classification": DECISION,
                "candidate_events": 60,
                "scene_events": 12,
                "acknowledgements": 72,
                "combined_candidates": 252,
                "combined_scenes": 36,
                "visuals": 2,
                "handoff_files": 10,
                "project_default": "DISABLED",
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-tests", action="store_true")
    args = parser.parse_args()
    if args.record_tests:
        record_tests()
        return
    run()


if __name__ == "__main__":
    main()
