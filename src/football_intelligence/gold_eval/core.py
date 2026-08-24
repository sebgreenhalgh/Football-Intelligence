from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


GOLD_CORPUS_VERSION = "G7F_A_GOLD_CORPUS_V1"
GOLD_SCOPE = "TEMPORAL_OBSERVATION_GOLD"
MISSING_HISTORICAL_FIELD = "FIELD_NOT_PRESENT_IN_HISTORICAL_EVENT"
EXPECTED_GLOBAL_RECEIPT = "global-9b8f980de82fabcea2164122"
EXPECTED_FINAL_EVENT = "f02167cc-b9b8-56ed-95dc-65cc2f7f30bc"
EXPECTED_COUNTS = {
    "bursts": 120,
    "subjects": 108,
    "subject_frames": 972,
    "missed_observations": 763,
    "accepted_actions": 9665,
}
MATCHES = ("117092", "117093", "118575", "118576", "118577", "128058")
TRANCHES = tuple(f"TRANCHE_{number}" for number in range(1, 7))
EXPECTED_TRANCHE_RECEIPTS = {
    "TRANCHE_1": "tranche-1-e2e61a550585aaa81a4afd25",
    "TRANCHE_2": "tranche-2-2fc70df6bb8e38d9da70a230",
    "TRANCHE_3": "tranche-3-16f74619fd56b95a7f44ad0a",
    "TRANCHE_4": "tranche-4-208840032552762e820c96d2",
    "TRANCHE_5": "tranche-5-5b9ea2676c643805f0791bf5",
    "TRANCHE_6": "tranche-6-5e1a0186528590a4eb5e0042",
}
SUPPORTED_EVENT_SCHEMAS = {
    (
        "football_intelligence.g7e_b_r4.burst_annotation_event.v1",
        "G7E_B_R4_CANDIDATE_RELATIONSHIP_BRANCH_INTEGRITY_V1",
    ),
    (
        "football_intelligence.g7e_b_r5.burst_annotation_event.v1",
        "G7E_B_R5_REVIEWER_STATE_MACHINE_V1",
    ),
    (
        "football_intelligence.g7e_b_r6.burst_annotation_event.v1",
        "G7E_B_R6_SERVER_AUTHORITATIVE_ACTION_REDUCER_V1",
    ),
}
NORMALIZED_FILENAMES = (
    "gold_bursts.jsonl",
    "gold_subjects.jsonl",
    "gold_subject_frames.jsonl",
    "gold_missed_observations.jsonl",
    "gold_selected_candidate_bindings.jsonl",
    "gold_source_event_index.jsonl",
)


class GoldEvalError(ValueError):
    """Raised when source or derived corpus integrity fails."""


class UnsupportedMetricError(GoldEvalError):
    """Raised when a requested metric is not supported by the gold ontology."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(payload)


def inventory_tree(root: Path, *, include_files: bool = False) -> dict[str, Any]:
    root = root.resolve()
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        byte_size = path.stat().st_size
        total_bytes += byte_size
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "byte_size": byte_size,
                "sha256": sha256_file(path),
            }
        )
    result: dict[str, Any] = {
        "root": str(root),
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "ordered_inventory_sha256": canonical_sha256(rows),
    }
    if include_files:
        result["files"] = rows
    return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GoldEvalError(message)


def _git_value(repository: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repository, check=True, capture_output=True, text=True).stdout.strip()


def evaluator_code_sha256() -> str:
    package_root = Path(__file__).resolve().parent
    rows = [
        {"name": path.name, "sha256": sha256_file(path), "byte_size": path.stat().st_size}
        for path in sorted(package_root.glob("*.py"))
    ]
    return canonical_sha256(rows)


def _lineage(event: dict[str, Any], event_path: Path, ack: dict[str, Any], ack_path: Path) -> dict[str, Any]:
    return {
        "source_event_id": event["event_id"],
        "source_event_sha256": sha256_file(event_path),
        "source_acknowledgement_id": ack["receipt_id"],
        "source_acknowledgement_sha256": sha256_file(ack_path),
        "source_event_schema_version": event["schema_version"],
        "source_review_revision": event["review_revision"],
        "source_review_revision_number": event.get("draft_version"),
    }


def _schema(title: str, required: list[str], properties: dict[str, Any]) -> dict[str, Any]:
    base = {
        "gold_corpus_version": {"const": GOLD_CORPUS_VERSION},
        "gold_scope": {"const": GOLD_SCOPE},
        "burst_id": {"type": "string", "minLength": 1},
    }
    base.update(properties)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "required": ["gold_corpus_version", "gold_scope", "burst_id", *required],
        "properties": base,
        "additionalProperties": True,
    }


def gold_schemas() -> dict[str, dict[str, Any]]:
    lineage = {
        "source_event_id": {"type": "string"},
        "source_event_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "source_acknowledgement_id": {"type": "string"},
        "source_acknowledgement_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    }
    return {
        "gold_bursts.schema.json": _schema(
            "Gold burst",
            ["tranche_id", "match_id", "subject_count", *lineage],
            {
                **lineage,
                "tranche_id": {"type": "string"},
                "match_id": {"type": "string"},
                "subject_count": {"type": "integer"},
            },
        ),
        "gold_subjects.schema.json": _schema(
            "Gold burst-local subject",
            ["subject_token", *lineage],
            {**lineage, "subject_token": {"type": "string"}},
        ),
        "gold_subject_frames.schema.json": _schema(
            "Gold subject-frame observation",
            ["subject_token", "frame_sequence", "source_frame_sha256", *lineage],
            {
                **lineage,
                "subject_token": {"type": "string"},
                "frame_sequence": {"type": "integer", "minimum": 0, "maximum": 8},
                "source_frame_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            },
        ),
        "gold_missed_observations.schema.json": _schema(
            "Gold frame-local missing-box observation",
            ["mark_id", "frame_sequence", "source_frame_sha256", "semantic_label", *lineage],
            {
                **lineage,
                "mark_id": {"type": "string"},
                "frame_sequence": {"type": "integer", "minimum": 0, "maximum": 8},
                "source_frame_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "semantic_label": {"const": "FRAME_LOCAL_MISSING_BOX_OBSERVATION"},
            },
        ),
        "gold_selected_candidate_bindings.schema.json": _schema(
            "Gold selected frozen-candidate binding",
            ["subject_token", "frame_sequence", "candidate_id", "source_box_xyxy", *lineage],
            {
                **lineage,
                "subject_token": {"type": "string"},
                "frame_sequence": {"type": "integer", "minimum": 0, "maximum": 8},
                "candidate_id": {"type": "string"},
                "source_box_xyxy": {"type": "array", "minItems": 4, "maxItems": 4},
            },
        ),
        "gold_source_event_index.schema.json": _schema(
            "Gold immutable source event index",
            [*lineage],
            lineage,
        ),
        "future_candidate_run.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Future source-frame candidate run",
            "type": "object",
            "required": ["schema_version", "run_id", "system_id", "code_commit", "candidates"],
            "properties": {
                "schema_version": {"const": "football_intelligence.g7f_a.candidate_run.v1"},
                "run_id": {"type": "string", "minLength": 1},
                "system_id": {"type": "string", "minLength": 1},
                "code_commit": {"type": "string", "minLength": 1},
                "weight_sha256": {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"},
                "run_metadata": {"type": ["object", "null"]},
                "candidates": {"type": "array", "items": {"$ref": "#/$defs/candidate"}},
            },
            "$defs": {
                "candidate": {
                    "type": "object",
                    "required": [
                        "source_frame_sha256",
                        "source_width",
                        "source_height",
                        "candidate_id",
                        "coordinate_space",
                        "box_xyxy",
                        "confidence",
                        "class_label",
                        "view_provenance",
                    ],
                    "properties": {
                        "source_frame_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        "source_width": {"type": "integer", "minimum": 1},
                        "source_height": {"type": "integer", "minimum": 1},
                        "candidate_id": {"type": "string", "minLength": 1},
                        "coordinate_space": {"type": "string"},
                        "box_xyxy": {"type": "array", "minItems": 4, "maxItems": 4},
                        "confidence": {"type": "number"},
                        "class_label": {"type": "string"},
                        "view_provenance": {"type": "object"},
                        "transform_to_source": {"type": "object"},
                        "runtime_metadata": {"type": ["object", "null"]},
                    },
                    "additionalProperties": False,
                }
            },
            "additionalProperties": False,
        },
    }


def metric_capabilities() -> dict[str, Any]:
    missing_person_denominator = (
        "Gold Corpus v1 does not enumerate every visible person and every arbitrary candidate in each frame; "
        "an exhaustive person denominator/false-positive ontology is absent."
    )
    missing_identity = "Gold Corpus v1 contains burst-local subjects, not persistent person identities or trajectories."
    missing_ball = "Gold Corpus v1 contains no exhaustive football/ball annotation population."
    return {
        "schema_version": "football_intelligence.g7f_a.metric_capabilities.v1",
        "gold_scope": GOLD_SCOPE,
        "production_ready": False,
        "tier_a_exact": sorted(
            [
                "burst_focus_distribution",
                "burst_context_distribution",
                "subject_count_distribution",
                "missed_check_distribution",
                "frame_local_missed_observation_burden",
                "subject_role_distribution",
                "subject_participation_distribution",
                "subject_certainty_distribution",
                "subject_frame_visibility_distribution",
                "historical_candidate_supply_distribution",
                "historical_candidate_relationship_distribution",
                "historical_selected_candidate_cardinality",
                "historical_selected_candidate_binding_validity",
                "frame_occlusion_phase_distribution",
                "explicit_named_occlusion_answer_coverage",
            ]
        ),
        "tier_b_point_support": sorted(
            [
                "subject_marker_candidate_support_rate",
                "subject_marker_candidate_multiplicity",
                "missed_mark_candidate_support_rate",
                "candidate_count_near_reviewed_subject_marker",
            ]
        ),
        "tier_c_unsupported": {
            "person_precision": missing_person_denominator,
            "precision": missing_person_denominator,
            "person_recall": missing_person_denominator,
            "recall": missing_person_denominator,
            "map": missing_person_denominator,
            "false_positive_rate": missing_person_denominator,
            "hota": missing_identity,
            "idf1": missing_identity,
            "mota": missing_identity,
            "long_term_identity_metrics": missing_identity,
            "unique_player_counts": missing_identity,
            "ball_precision": missing_ball,
            "ball_recall": missing_ball,
            "team_shape_metrics": "No team-shape or football-performance gold ontology exists in this corpus.",
            "player_performance_metrics": "No player-performance gold ontology exists in this corpus.",
        },
        "unknown_metric_policy": "FAIL_CLOSED",
    }


def _source_snapshot(decisions_root: Path, reviewer_package: Path) -> dict[str, Any]:
    decisions_root = decisions_root.resolve()
    reviewer_package = reviewer_package.resolve()
    _require(decisions_root.is_dir(), f"human decisions root not found: {decisions_root}")
    _require(reviewer_package.is_dir(), f"reviewer package not found: {reviewer_package}")

    cases = read_json(reviewer_package / "review_cases.json")["cases"]
    manifests = read_jsonl(reviewer_package / "tranche_manifest.jsonl")
    _require(len(cases) == 120 and len(manifests) == 120, "frozen package must contain exactly 120 cases")
    cases_by_burst = {case["burst_id"]: case for case in cases}
    manifests_by_burst = {row["burst_id"]: row for row in manifests}
    _require(len(cases_by_burst) == 120, "duplicate review-case burst ID")
    _require(set(cases_by_burst) == set(manifests_by_burst), "review-case/manifest membership mismatch")
    ordered_cases: list[dict[str, Any]] = []
    for tranche in TRANCHES:
        tranche_cases = sorted(
            (case for case in cases if case["tranche_id"] == tranche), key=lambda case: case["tranche_position"]
        )
        _require(len(tranche_cases) == 20, f"{tranche} must contain 20 cases")
        _require([case["tranche_position"] for case in tranche_cases] == list(range(1, 21)), f"{tranche} positions")
        ordered_cases.extend(tranche_cases)

    package_manifest_path = reviewer_package / "package_manifest.json"
    package_manifest = read_json(package_manifest_path)
    for relative_path, expected in package_manifest["files"].items():
        source = reviewer_package / relative_path
        _require(source.is_file(), f"package-manifest file absent: {relative_path}")
        _require(source.stat().st_size == expected["byte_size"], f"package byte-size mismatch: {relative_path}")
        _require(sha256_file(source) == expected["sha256"], f"package hash mismatch: {relative_path}")
    release_gate_path = reviewer_package / "G7E_B_R6_7_2_REAL_REVIEW_RELEASE_GATE.json"
    release_gate = read_json(release_gate_path)
    _require(release_gate["production_ready"] is False, "reviewer release gate must remain non-production")
    candidate_states_path = reviewer_package / "candidate_states_by_reference.json"
    candidate_states = read_json(candidate_states_path)["frames"]
    _require(len(candidate_states) == 1080, "frozen candidate state must bind 1,080 source frames")
    state_contract = read_json(reviewer_package / "canonical_reviewer_state_contract.json")
    relationship_states = state_contract["relationship_compatibility"]["supply_states"]
    relationship_families = state_contract["relationship_compatibility"]["question_families"]

    event_paths = sorted((decisions_root / "events").glob("*/*.json"))
    ack_paths = sorted((decisions_root / "receipts" / "acknowledgements").glob("*.json"))
    _require(len(event_paths) == 120, f"expected 120 immutable events, found {len(event_paths)}")
    _require(len(ack_paths) == 120, f"expected 120 acknowledgements, found {len(ack_paths)}")
    _require(not list((decisions_root / "drafts").glob("*.json")), "completed source contains a current draft")
    events = [(read_json(path), path) for path in event_paths]
    acks = [(read_json(path), path) for path in ack_paths]
    events_by_burst: dict[str, list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    for event, path in events:
        events_by_burst[event["burst_id"]].append((event, path))
    ack_by_event = {ack["event_id"]: (ack, path) for ack, path in acks}
    _require(len(ack_by_event) == 120, "duplicate acknowledgement event binding")

    normalized = {name: [] for name in NORMALIZED_FILENAMES}
    references_by_tranche: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_references: list[dict[str, Any]] = []
    latest_by_burst: dict[str, dict[str, Any]] = {}
    candidate_state_file_hash = sha256_file(candidate_states_path)
    for case in ordered_cases:
        burst_id = case["burst_id"]
        manifest = manifests_by_burst[burst_id]
        chain = sorted(events_by_burst.get(burst_id, []), key=lambda pair: int(pair[0]["server_sequence"]))
        _require(chain, f"{burst_id}: missing event")
        for previous, following in zip(chain, chain[1:]):
            _require(following[0].get("supersedes_event_id") == previous[0]["event_id"], f"{burst_id}: event chain")
        event, event_path = chain[-1]
        latest_by_burst[burst_id] = event
        _require((event["schema_version"], event["review_revision"]) in SUPPORTED_EVENT_SCHEMAS, f"{burst_id}: schema")
        _require(
            event["source_frame_hashes"] == [f["source_frame_pixel_sha256"] for f in case["frames"]],
            f"{burst_id}: frame hashes",
        )
        _require(
            event["burst_manifest_sha256"] == case["source_manifest_hashes"]["temporal_burst_manifest_sha256"],
            f"{burst_id}: manifest hash",
        )
        _require(event.get("summary_confirmed") is True, f"{burst_id}: summary not confirmed")
        _require(event.get("mode") == "real" and event.get("production_ready") is False, f"{burst_id}: source mode")
        ack_pair = ack_by_event.get(event["event_id"])
        _require(ack_pair is not None, f"{burst_id}: acknowledgement missing")
        ack, ack_path = ack_pair
        event_hash = sha256_file(event_path)
        _require(ack["event_sha256"] == event_hash, f"{burst_id}: acknowledgement event hash")
        _require(ack["event_byte_size"] == event_path.stat().st_size, f"{burst_id}: acknowledgement event bytes")
        _require(
            ack.get("server_validated") is True and ack.get("case_complete") is True,
            f"{burst_id}: acknowledgement state",
        )
        lineage = _lineage(event, event_path, ack, ack_path)
        reference = {
            "burst_id": burst_id,
            "event_id": event["event_id"],
            "event_sha256": event_hash,
            "acknowledgement_receipt_id": ack["receipt_id"],
            "acknowledgement_receipt_sha256": sha256_file(ack_path),
        }
        references_by_tranche[case["tranche_id"]].append(reference)
        all_references.append(reference)

        common = {
            "gold_corpus_version": GOLD_CORPUS_VERSION,
            "gold_scope": GOLD_SCOPE,
            "burst_id": burst_id,
            "tranche_id": case["tranche_id"],
            "match_id": str(case["match_id"]),
            **lineage,
        }
        subjects = event.get("subjects", [])
        missed_marks = event.get("whole_burst_missed_person_marks", [])
        _require(len(subjects) <= 3, f"{burst_id}: more than three subjects")
        _require(
            [subject["subject_token"] for subject in subjects]
            == [f"SUBJECT_{letter}" for letter in "ABC"[: len(subjects)]],
            f"{burst_id}: non-canonical subject order",
        )
        missed_answer = event["whole_burst_missed_person_answer"]
        _require(
            (missed_answer == "YES" and missed_marks) or (missed_answer in {"NO", "NOT_SURE"} and not missed_marks),
            f"{burst_id}: missed-answer branch mismatch",
        )
        normalized["gold_bursts.jsonl"].append(
            {
                **common,
                "tranche_position": case["tranche_position"],
                "half": case["half"],
                "perspective_band": manifest["perspective_band"],
                "primary_selection_class": manifest["primary_selection_class"],
                "source_global_receipt_id": EXPECTED_GLOBAL_RECEIPT,
                "original_focus_box_answer": event["original_focus_box_answer"],
                "context_answer": event["context_subject_answer"],
                "subject_count": len(subjects),
                "missed_check": event["whole_burst_missed_person_answer"],
                "frame_local_missed_observation_count": len(missed_marks),
            }
        )
        normalized["gold_source_event_index.jsonl"].append(
            {
                **common,
                "source_event_relative_path": event_path.relative_to(decisions_root).as_posix(),
                "source_acknowledgement_relative_path": ack_path.relative_to(decisions_root).as_posix(),
                "source_event_byte_size": event_path.stat().st_size,
                "source_acknowledgement_byte_size": ack_path.stat().st_size,
                "source_server_sequence": event["server_sequence"],
            }
        )

        frozen_by_frame: list[dict[str, dict[str, Any]]] = []
        state_by_frame: list[dict[str, Any]] = []
        for sequence, frame in enumerate(case["frames"]):
            _require(frame["burst_frame_sequence"] == sequence, f"{burst_id}: frame order")
            _require(
                frame["source_frame_pixel_sha256"] == frame["canonical_frame_identity"]["frame_pixel_sha256"],
                f"{burst_id}: canonical frame identity",
            )
            frame_state = candidate_states.get(frame["frame_reference_id"])
            _require(frame_state is not None, f"{burst_id}: frozen candidate state absent")
            _require(
                frame_state["frame_pixel_sha256"] == frame["source_frame_pixel_sha256"],
                f"{burst_id}: candidate frame hash",
            )
            _require(
                frame_state["candidates"] == case["frame_candidates"][sequence], f"{burst_id}: candidate-state bytes"
            )
            state_by_frame.append(frame_state)
            frozen_by_frame.append({candidate["candidate_id"]: candidate for candidate in frame_state["candidates"]})
            _require(
                all(
                    0 <= candidate["source_box_xyxy"][0] <= candidate["source_box_xyxy"][2] <= case["source_width"]
                    and 0 <= candidate["source_box_xyxy"][1] <= candidate["source_box_xyxy"][3] <= case["source_height"]
                    for candidate in frame_state["candidates"]
                ),
                f"{burst_id}: frozen candidate bounds",
            )

        selected_keys: set[tuple[str, int, str]] = set()
        for subject in subjects:
            subject_token = subject["subject_token"]
            _require(len(subject["frame_observations"]) == 9, f"{burst_id}/{subject_token}: nine frames")
            named_answer_present = "occlusion_sequence_answer" in subject
            normalized["gold_subjects.jsonl"].append(
                {
                    **common,
                    "subject_token": subject_token,
                    "subject_definition_source": subject["subject_definition_source"],
                    "anchor_frame_sequence": subject["anchor_frame_sequence"],
                    "anchor_source_xy": subject["anchor_source_xy"],
                    "role": subject["role"],
                    "participation": subject["participation"],
                    "certainty": subject["certainty"],
                    "marker_continuity": subject["marker_continuity_confirmation"],
                    "continuity": subject["continuity"],
                    "occlusion_confirmed": subject["occlusion_confirmed"],
                    "occlusion_sequence_answer": subject.get("occlusion_sequence_answer", MISSING_HISTORICAL_FIELD),
                    "occlusion_sequence_answer_availability": "PRESENT"
                    if named_answer_present
                    else MISSING_HISTORICAL_FIELD,
                }
            )
            for sequence, observation in enumerate(subject["frame_observations"]):
                frame = case["frames"][sequence]
                _require(
                    observation["frame_reference_id"] == frame["frame_reference_id"], f"{burst_id}: observation frame"
                )
                _require(
                    observation["canonical_frame_identity"] == frame["canonical_frame_identity"],
                    f"{burst_id}: observation canonical frame",
                )
                x = observation.get("subject_location_source_x")
                y = observation.get("subject_location_source_y")
                _require(
                    (x is None and y is None)
                    or (
                        x is not None
                        and y is not None
                        and 0 <= x <= case["source_width"]
                        and 0 <= y <= case["source_height"]
                    ),
                    f"{burst_id}: subject coordinate",
                )
                selected_ids = observation.get("selected_candidate_ids", [])
                _require(len(selected_ids) == len(set(selected_ids)), f"{burst_id}: duplicate selected candidate")
                supply = observation["observation_supply"]
                relationship = observation["candidate_relationship"]
                rule = relationship_states.get(supply)
                _require(rule is not None, f"{burst_id}: unknown candidate supply {supply}")
                maximum = rule["maximum_selected_count"]
                _require(
                    len(selected_ids) >= rule["minimum_selected_count"]
                    and (maximum is None or len(selected_ids) <= maximum),
                    f"{burst_id}: candidate cardinality",
                )
                if rule["relationship_applicable"]:
                    allowed = relationship_families[rule["question_family"]]["allowed_relationships"]
                    _require(relationship in allowed, f"{burst_id}: candidate relationship")
                else:
                    _require(relationship == "NOT_APPLICABLE", f"{burst_id}: relationship applicability")
                selected_candidates = []
                for candidate_id in selected_ids:
                    candidate = frozen_by_frame[sequence].get(candidate_id)
                    _require(candidate is not None, f"{burst_id}: selected candidate outside frozen frame")
                    selected_candidates.append(candidate)
                    selected_keys.add((subject_token, sequence, candidate_id))
                frame_state_hash = canonical_sha256(state_by_frame[sequence])
                normalized["gold_subject_frames.jsonl"].append(
                    {
                        **common,
                        "subject_token": subject_token,
                        "frame_sequence": sequence,
                        "frame_reference_id": frame["frame_reference_id"],
                        "source_frame_sha256": frame["source_frame_pixel_sha256"],
                        "source_width": case["source_width"],
                        "source_height": case["source_height"],
                        "human_confirmed_source_coordinate": [x, y] if x is not None else None,
                        "visibility": observation["visibility"],
                        "observation_supply": observation["observation_supply"],
                        "candidate_relationship": observation["candidate_relationship"],
                        "selected_candidate_ids": selected_ids,
                        "selected_candidate_count": len(selected_ids),
                        "selected_candidate_frozen_source_boxes": [c["source_box_xyxy"] for c in selected_candidates],
                        "occlusion_phase": observation["occlusion_phase"],
                        "source_candidate_state_file_sha256": candidate_state_file_hash,
                        "source_candidate_state_frame_sha256": frame_state_hash,
                    }
                )
                for candidate in selected_candidates:
                    state = state_by_frame[sequence]
                    normalized["gold_selected_candidate_bindings.jsonl"].append(
                        {
                            **common,
                            "subject_token": subject_token,
                            "frame_sequence": sequence,
                            "frame_reference_id": frame["frame_reference_id"],
                            "source_frame_sha256": frame["source_frame_pixel_sha256"],
                            "candidate_id": candidate["candidate_id"],
                            "source_box_xyxy": candidate["source_box_xyxy"],
                            "supply_state": observation["observation_supply"],
                            "relationship_state": observation["candidate_relationship"],
                            "source_candidate_state_file_sha256": candidate_state_file_hash,
                            "source_candidate_state_frame_sha256": frame_state_hash,
                            "source_post_gate_artifact": state["post_gate_artifact"],
                        }
                    )

        mapping_keys = {
            (mapping["subject_token"], mapping["frame_sequence"], mapping["candidate_id"])
            for mapping in event.get("candidate_mappings", [])
        }
        _require(mapping_keys == selected_keys, f"{burst_id}: flattened candidate mappings")
        for mapping in event.get("candidate_mappings", []):
            sequence = mapping["frame_sequence"]
            frame = case["frames"][sequence]
            candidate = frozen_by_frame[sequence].get(mapping["candidate_id"])
            _require(
                candidate is not None
                and mapping["source_box_xyxy"] == candidate["source_box_xyxy"]
                and mapping["frame_reference_id"] == frame["frame_reference_id"]
                and mapping["canonical_frame_identity"] == frame["canonical_frame_identity"],
                f"{burst_id}: flattened mapping provenance",
            )

        for mark in missed_marks:
            sequence = mark["frame_sequence"]
            frame = case["frames"][sequence]
            x, y = mark["source_xy"]
            _require(0 <= x <= case["source_width"] and 0 <= y <= case["source_height"], f"{burst_id}: missed bounds")
            _require(mark["frame_reference_id"] == frame["frame_reference_id"], f"{burst_id}: missed frame")
            _require(
                mark["canonical_frame_identity"] == frame["canonical_frame_identity"],
                f"{burst_id}: missed canonical frame",
            )
            normalized["gold_missed_observations.jsonl"].append(
                {
                    **common,
                    "mark_id": mark["mark_id"],
                    "frame_sequence": sequence,
                    "frame_reference_id": frame["frame_reference_id"],
                    "source_frame_sha256": frame["source_frame_pixel_sha256"],
                    "source_width": case["source_width"],
                    "source_height": case["source_height"],
                    "source_coordinate": mark["source_xy"],
                    "original_mark_role": mark["role"],
                    "original_mark_certainty": mark["certainty"],
                    "role_was_explicitly_elicited": False,
                    "certainty_was_explicitly_elicited": False,
                    "role_availability": "NOT_ELICITED_BY_REVIEW_FLOW",
                    "certainty_availability": "NOT_ELICITED_BY_REVIEW_FLOW",
                    "semantic_label": "FRAME_LOCAL_MISSING_BOX_OBSERVATION",
                }
            )

    _require(len(latest_by_burst) == 120, "latest event set must contain 120 bursts")
    tranche_receipt_refs: list[dict[str, Any]] = []
    for tranche in TRANCHES:
        references = references_by_tranche[tranche]
        digest = canonical_sha256(references)
        expected_id = EXPECTED_TRANCHE_RECEIPTS[tranche]
        computed_id = f"tranche-{tranche.removeprefix('TRANCHE_').lower()}-{digest[:24]}"
        _require(computed_id == expected_id, f"{tranche}: receipt derivation mismatch")
        receipt_path = decisions_root / "receipts" / "tranche_completion" / f"{expected_id}.json"
        receipt = read_json(receipt_path)
        _require(receipt["latest_acknowledged_events"] == references, f"{tranche}: receipt event set")
        _require(receipt["latest_event_set_digest"] == digest, f"{tranche}: receipt digest")
        tranche_receipt_refs.append(
            {
                "tranche_id": tranche,
                "tranche_completion_receipt_id": expected_id,
                "tranche_completion_receipt_sha256": sha256_file(receipt_path),
            }
        )
    global_digest = canonical_sha256({"events": all_references, "tranches": tranche_receipt_refs})
    _require(f"global-{global_digest[:24]}" == EXPECTED_GLOBAL_RECEIPT, "global receipt derivation mismatch")
    global_path = decisions_root / "receipts" / "global_completion" / f"{EXPECTED_GLOBAL_RECEIPT}.json"
    global_receipt = read_json(global_path)
    _require(global_receipt["latest_acknowledged_events"] == all_references, "global event references")
    _require(global_receipt["current_tranche_receipts"] == tranche_receipt_refs, "global tranche references")
    _require(global_receipt["all_cases_complete"] is True, "global receipt is not complete")
    _require(all_references[-1]["event_id"] == EXPECTED_FINAL_EVENT, "final acknowledged event mismatch")
    _require(len(list((decisions_root / "receipts" / "actions").glob("*.json"))) == 9665, "accepted action count")

    for name in NORMALIZED_FILENAMES:
        normalized[name].sort(
            key=lambda row: tuple(
                str(row.get(key, ""))
                for key in ("burst_id", "subject_token", "frame_sequence", "mark_id", "candidate_id")
            )
        )
    actual_counts = {
        "bursts": len(normalized["gold_bursts.jsonl"]),
        "subjects": len(normalized["gold_subjects.jsonl"]),
        "subject_frames": len(normalized["gold_subject_frames.jsonl"]),
        "missed_observations": len(normalized["gold_missed_observations.jsonl"]),
        "accepted_actions": 9665,
        "selected_candidate_bindings": len(normalized["gold_selected_candidate_bindings.jsonl"]),
        "source_event_index": len(normalized["gold_source_event_index.jsonl"]),
    }
    for key, expected in EXPECTED_COUNTS.items():
        _require(actual_counts[key] == expected, f"{key}: expected {expected}, found {actual_counts[key]}")
    match_counts = Counter(row["match_id"] for row in normalized["gold_bursts.jsonl"])
    _require(match_counts == Counter({match: 20 for match in MATCHES}), f"match distribution: {match_counts}")

    repair = {
        "g7e_a_117092_03": {"subjects": ["SUBJECT_A", "SUBJECT_B"], "marks": 24},
        "g7e_a_118577_14": {"subjects": ["SUBJECT_A", "SUBJECT_B"], "marks": 9},
        "g7e_a_117092_10": {"subjects": ["SUBJECT_A"], "marks": 8},
    }
    repair_report = []
    for burst_id, expected in repair.items():
        event = latest_by_burst[burst_id]
        passed = [subject["subject_token"] for subject in event["subjects"]] == expected["subjects"]
        passed &= len(event["whole_burst_missed_person_marks"]) == expected["marks"]
        if burst_id == "g7e_a_118577_14":
            subject_b = event["subjects"][1]
            passed &= subject_b.get("occlusion_sequence_answer") == "OCCLUDED"
            passed &= [row["occlusion_phase"] for row in subject_b["frame_observations"]] == ["ENTERING_OCCLUSION"] * 9
        _require(passed, f"repair-trigger burst mismatch: {burst_id}")
        repair_report.append({"burst_id": burst_id, "passed": bool(passed), **expected})

    schema_revision_counts = Counter(
        (event["schema_version"], event["review_revision"]) for event in latest_by_burst.values()
    )
    _require(set(schema_revision_counts) == SUPPORTED_EVENT_SCHEMAS, "source must exercise R4, R5 and R6 schemas")

    return {
        "normalized": normalized,
        "cases_by_burst": cases_by_burst,
        "manifests_by_burst": manifests_by_burst,
        "counts": actual_counts,
        "repair_bursts": repair_report,
        "schema_revision_counts": [
            {"schema_version": pair[0], "review_revision": pair[1], "count": count}
            for pair, count in sorted(schema_revision_counts.items())
        ],
        "package_bindings": {
            "source_package_commit": package_manifest["commit"],
            "reviewer_package_manifest_sha256": sha256_file(package_manifest_path),
            "release_gate_sha256": sha256_file(release_gate_path),
            "tranche_manifest_sha256": sha256_file(reviewer_package / "tranche_manifest.jsonl"),
            "candidate_states_by_reference_sha256": candidate_state_file_hash,
            "review_cases_sha256": sha256_file(reviewer_package / "review_cases.json"),
            "canonical_reviewer_state_contract_sha256": sha256_file(
                reviewer_package / "canonical_reviewer_state_contract.json"
            ),
            "server_action_contract_sha256": sha256_file(reviewer_package / "server_action_contract.json"),
            "visual_asset_manifest_sha256": sha256_file(reviewer_package / "visual_asset_manifest.json"),
            "build_manifest_sha256": sha256_file(reviewer_package / "build_manifest.json"),
        },
        "receipt_chain": {
            "global_receipt_id": EXPECTED_GLOBAL_RECEIPT,
            "global_receipt_sha256": sha256_file(global_path),
            "final_acknowledged_event_id": EXPECTED_FINAL_EVENT,
            "tranche_receipts": tranche_receipt_refs,
        },
    }


def _split_manifest(bursts: list[dict[str, Any]]) -> dict[str, Any]:
    by_match = {match: sorted(row["burst_id"] for row in bursts if row["match_id"] == match) for match in MATCHES}
    folds = []
    all_ids = {row["burst_id"] for row in bursts}
    for held_out in MATCHES:
        evaluation = by_match[held_out]
        development = sorted(all_ids - set(evaluation))
        _require(len(evaluation) == 20 and len(development) == 100, f"fold size for {held_out}")
        _require(not (set(evaluation) & set(development)), f"fold leakage for {held_out}")
        folds.append(
            {
                "fold_id": f"HOLDOUT_MATCH_{held_out}",
                "held_out_match_id": held_out,
                "tuning_exposure": "HELD_OUT_INTERNAL",
                "evaluation_burst_ids": evaluation,
                "development_burst_ids": development,
            }
        )
    return {
        "schema_version": "football_intelligence.g7f_a.split_manifest.v1",
        "strategy": "SIX_FOLD_LEAVE_ONE_MATCH_OUT_INTERNAL",
        "match_is_atomic": True,
        "all_nine_frames_remain_with_burst": True,
        "promotion_eligible_from_current_120_only": False,
        "folds": folds,
        "production_ready": False,
    }


def _distribution(rows: Iterable[dict[str, Any]], field: str) -> dict[str, Any]:
    counter = Counter(row[field] for row in rows)
    denominator = sum(counter.values())
    return {
        "capability_tier": "A_EXACT",
        "denominator": denominator,
        "counts": dict(sorted(counter.items())),
        "percent": {key: round(100 * value / denominator, 3) for key, value in sorted(counter.items())},
    }


def _exact_gold_metrics(normalized: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    bursts = normalized["gold_bursts.jsonl"]
    subjects = normalized["gold_subjects.jsonl"]
    frames = normalized["gold_subject_frames.jsonl"]
    missed = normalized["gold_missed_observations.jsonl"]
    bindings = normalized["gold_selected_candidate_bindings.jsonl"]
    return {
        "burst_focus_distribution": _distribution(bursts, "original_focus_box_answer"),
        "burst_context_distribution": _distribution(bursts, "context_answer"),
        "subject_count_distribution": _distribution(bursts, "subject_count"),
        "missed_check_distribution": _distribution(bursts, "missed_check"),
        "frame_local_missed_observation_burden": {
            "capability_tier": "A_EXACT",
            "count": len(missed),
            "denominator": len(bursts),
            "unit": "person_x_exact_frame_marks_per_reviewed_burst",
        },
        "subject_role_distribution": _distribution(subjects, "role"),
        "subject_participation_distribution": _distribution(subjects, "participation"),
        "subject_certainty_distribution": _distribution(subjects, "certainty"),
        "subject_frame_visibility_distribution": _distribution(frames, "visibility"),
        "historical_candidate_supply_distribution": _distribution(frames, "observation_supply"),
        "historical_candidate_relationship_distribution": _distribution(frames, "candidate_relationship"),
        "historical_selected_candidate_cardinality": _distribution(frames, "selected_candidate_count"),
        "historical_selected_candidate_binding_validity": {
            "capability_tier": "A_EXACT",
            "numerator": len(bindings),
            "denominator": len(bindings),
            "value": 1.0,
            "unit": "human_selected_candidates_exactly_bound_to_frozen_candidate_state",
        },
        "frame_occlusion_phase_distribution": _distribution(frames, "occlusion_phase"),
        "explicit_named_occlusion_answer_coverage": _distribution(subjects, "occlusion_sequence_answer_availability"),
    }


def _baseline(
    subject_frames: list[dict[str, Any]], missed: list[dict[str, Any]], bursts: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    expected_supply = {
        "ONE_USEFUL_CANDIDATE": 718,
        "MULTIPLE_CANDIDATES": 222,
        "MERGED_WITH_OTHER_PEOPLE": 23,
        "FRAGMENT_ONLY": 1,
        "NO_CANDIDATE": 8,
    }
    supply = Counter(row["observation_supply"] for row in subject_frames)
    _require(dict(supply) == expected_supply, f"frozen candidate-supply baseline mismatch: {supply}")
    relationship = Counter(row["candidate_relationship"] for row in subject_frames)
    for name, expected in {
        "SAME_PERSON_FRAGMENTS": 197,
        "MERGED_MULTI_PERSON": 22,
        "CORRECT_INNER_BAD_OUTER": 25,
    }.items():
        _require(relationship[name] == expected, f"relationship baseline mismatch for {name}")
    grouped = {
        "CLEAN_SINGLE_SUPPLY": {"ONE_USEFUL_CANDIDATE"},
        "MULTIPLICITY_OR_FRAGMENTATION": {"MULTIPLE_CANDIDATES", "FRAGMENT_ONLY"},
        "MERGED_OR_FRAGMENTED_SUPPLY": {"MERGED_WITH_OTHER_PEOPLE", "FRAGMENT_ONLY"},
        "NO_SUBJECT_CANDIDATE_SUPPLY": {"NO_CANDIDATE"},
    }
    ledger = []
    for row in subject_frames:
        groups = sorted(name for name, members in grouped.items() if row["observation_supply"] in members)
        ledger.append(
            {
                "population": "SUBJECT_FRAME_TRUTH",
                "burst_id": row["burst_id"],
                "match_id": row["match_id"],
                "subject_token": row["subject_token"],
                "frame_sequence": row["frame_sequence"],
                "source_frame_sha256": row["source_frame_sha256"],
                "candidate_supply": row["observation_supply"],
                "candidate_relationship": row["candidate_relationship"],
                "rd_groupings": groups,
            }
        )
    for row in missed:
        ledger.append(
            {
                "population": "FRAME_LOCAL_MISSED_PERSON_TRUTH",
                "burst_id": row["burst_id"],
                "match_id": row["match_id"],
                "frame_sequence": row["frame_sequence"],
                "mark_id": row["mark_id"],
                "source_frame_sha256": row["source_frame_sha256"],
                "semantic_label": row["semantic_label"],
                "rd_groupings": ["FRAME_LOCAL_VISIBLE_PERSON_MISSING_BOX"],
            }
        )
    by_match = {}
    for match in MATCHES:
        frame_rows = [row for row in subject_frames if row["match_id"] == match]
        mark_rows = [row for row in missed if row["match_id"] == match]
        by_match[match] = {
            "candidate_supply": _distribution(frame_rows, "observation_supply"),
            "candidate_relationship": _distribution(frame_rows, "candidate_relationship"),
            "frame_local_missed_observations": {
                "capability_tier": "A_EXACT",
                "denominator": 20,
                "count": len(mark_rows),
                "unit": "person_x_exact_frame_marks_per_20_bursts",
            },
        }
    baseline = {
        "schema_version": "football_intelligence.g7f_a.frozen_baseline.v1",
        "gold_scope": GOLD_SCOPE,
        "candidate_supply": _distribution(subject_frames, "observation_supply"),
        "candidate_relationship": _distribution(subject_frames, "candidate_relationship"),
        "selected_candidate_cardinality": _distribution(subject_frames, "selected_candidate_count"),
        "frame_local_missed_observations": {
            "capability_tier": "A_EXACT",
            "count": len(missed),
            "denominator": len(bursts),
            "unit": "person_x_exact_frame_marks_per_120_reviewed_bursts",
            "detector_recall": None,
        },
        "population_separation_enforced": True,
        "production_ready": False,
    }
    by_fold = {
        f"HOLDOUT_MATCH_{match}": {
            "held_out_match_id": match,
            "tuning_exposure": "HELD_OUT_INTERNAL",
            **by_match[match],
        }
        for match in MATCHES
    }
    return baseline, {"matches": by_match}, [{"fold_id": key, **value} for key, value in by_fold.items()], ledger


def _priority_artifacts(baseline: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    supply = baseline["candidate_supply"]
    priority = {
        "schema_version": "football_intelligence.g7f_a.rd_priority_map.v1",
        "gold_scope": GOLD_SCOPE,
        "subject_frame_candidate_supply": {
            "denominator": supply["denominator"],
            "one_useful": {"count": 718, "percent": 73.868},
            "multiple": {"count": 222, "percent": 22.840},
            "merged_or_fragment": {"count": 24, "percent": 2.469},
            "no_candidate": {"count": 8, "percent": 0.823},
        },
        "frame_local_missed_observations": {
            "count": 763,
            "unit": "visible_relevant_person_x_exact_affected_frame_missing_useful_box",
            "exhaustive_person_frame_denominator_available": False,
        },
        "relationship_burden": {"same_person_fragments": {"count": 197, "denominator": 972, "percent": 20.267}},
        "immediately_testable": [
            "source-point candidate containment and multiplicity",
            "candidate supply and relationship distributions on reviewed subject frames",
            "frame-local support at existing missed marks",
        ],
        "requires_additional_gold": [
            "exhaustive precision, recall, mAP and false-positive rate",
            "arbitrary merged-person correctness and duplicate rate",
            "final unbiased model promotion",
        ],
        "architecture_decision_from_aggregates_authorized": False,
        "production_ready": False,
    }
    markdown = "\n".join(
        [
            "# R&D priority map v1",
            "",
            "Gold Corpus v1 is human-reviewed temporal observation gold, not exhaustive person-detection truth.",
            "",
            "- Subject-frame supply (denominator 972): 718 one-useful (73.868%), 222 multiple (22.840%),",
            "  24 merged/fragment (2.469%), and 8 no-candidate (0.823%).",
            "- Frame-local missed burden: 763 visible-person x affected-frame marks. No exhaustive",
            "  person-frame denominator exists.",
            "- Relationship burden: 197/972 reviewed subject frames (20.267%) are SAME_PERSON_FRAGMENTS.",
            "",
            "The corpus can test source-point support, multiplicity, historical supply, and reviewed",
            "relationships now. Precision, recall, mAP, arbitrary false positives/duplicates, and final",
            "promotion require additional gold. These aggregates do not by themselves authorize a",
            "detector architecture decision.",
            "",
        ]
    )
    gap = {
        "schema_version": "football_intelligence.g7f_a.gold_gap_register.v1",
        "gaps": [
            {
                "gold": "EXHAUSTIVE_FULL_FRAME_PERSON_GOLD",
                "required_before": ["precision", "recall", "mAP", "false-positive rate"],
            },
            {
                "gold": "DENSE_FULL_OR_VISIBLE_BOXES_OR_MASKS",
                "required_before": [
                    "merged-person correctness",
                    "distinct-person suppression claims",
                    "arbitrary-system true duplicate rate",
                ],
            },
            {"gold": "NEW_SEALED_SOURCE_MATCHES", "required_before": ["final unbiased architecture promotion"]},
            {
                "gold": "FOOTBALL_OR_BALL_GOLD",
                "required_before": ["ball precision", "ball recall", "ball centre-error"],
            },
        ],
        "production_ready": False,
    }
    return priority, markdown, gap


def build_gold_corpus(
    decisions_root: Path, reviewer_package: Path, output_root: Path, repository: Path
) -> dict[str, Any]:
    decisions_root = decisions_root.resolve()
    reviewer_package = reviewer_package.resolve()
    output_root = output_root.resolve()
    repository = repository.resolve()
    before = inventory_tree(decisions_root)
    head = _git_value(repository, "rev-parse", "HEAD")
    source = _source_snapshot(decisions_root, reviewer_package)
    normalized = source["normalized"]
    schemas = gold_schemas()

    source_dir = output_root / "00_SOURCE_FREEZE"
    schema_dir = output_root / "01_GOLD_SCHEMAS"
    gold_dir = output_root / "02_NORMALIZED_GOLD"
    split_dir = output_root / "03_SPLITS"
    harness_dir = output_root / "04_EVALUATION_HARNESS"
    baseline_dir = output_root / "05_FROZEN_BASELINE"
    taxonomy_dir = output_root / "06_ERROR_TAXONOMY"
    priority_dir = output_root / "07_RD_PRIORITY_MAP"
    integrity_dir = output_root / "08_TESTS_AND_INTEGRITY"
    evidence_dir = output_root / "09_RELEASE_EVIDENCE"
    for directory in (
        source_dir,
        schema_dir,
        gold_dir,
        split_dir,
        harness_dir,
        baseline_dir,
        taxonomy_dir,
        priority_dir,
        integrity_dir,
        evidence_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    for filename, schema in schemas.items():
        write_json(schema_dir / filename, schema)
    for filename, rows in normalized.items():
        write_jsonl(gold_dir / filename, rows)
    split = _split_manifest(normalized["gold_bursts.jsonl"])
    future = {
        "schema_version": "football_intelligence.g7f_a.future_sealed_holdout.v1",
        "tuning_exposure": "FUTURE_SEALED",
        "current_gold_corpus_burst_ids": [],
        "sealed_source_matches": [],
        "promotion_eligible_from_current_120_only": False,
        "production_ready": False,
    }
    write_json(split_dir / "gold_split_manifest_v1.json", split)
    write_json(split_dir / "future_sealed_holdout_manifest.json", future)
    capabilities = metric_capabilities()
    write_json(harness_dir / "metric_capabilities.json", capabilities)
    write_json(harness_dir / "future_candidate_run.schema.json", schemas["future_candidate_run.schema.json"])
    write_json(
        harness_dir / "evaluator_binding.json",
        {
            "repository_commit": head,
            "evaluator_code_sha256": evaluator_code_sha256(),
            "cli": "python -m football_intelligence.gold_eval",
            "production_ready": False,
        },
    )

    baseline, by_match, by_fold, ledger = _baseline(
        normalized["gold_subject_frames.jsonl"],
        normalized["gold_missed_observations.jsonl"],
        normalized["gold_bursts.jsonl"],
    )
    baseline["exact_gold_metrics"] = _exact_gold_metrics(normalized)
    write_json(baseline_dir / "baseline_current_candidate_system.json", baseline)
    write_json(baseline_dir / "baseline_by_match.json", by_match)
    write_json(baseline_dir / "baseline_by_fold.json", {"folds": by_fold})
    write_jsonl(baseline_dir / "baseline_error_ledger.jsonl", ledger)
    write_json(
        taxonomy_dir / "error_taxonomy.json",
        {
            "candidate_supply_classes": [
                "ONE_USEFUL_CANDIDATE",
                "MULTIPLE_CANDIDATES",
                "MERGED_WITH_OTHER_PEOPLE",
                "FRAGMENT_ONLY",
                "NO_CANDIDATE",
                "UNCERTAIN",
            ],
            "derived_rd_groupings_are_additive": True,
            "derived_rd_groupings": [
                "CLEAN_SINGLE_SUPPLY",
                "MULTIPLICITY_OR_FRAGMENTATION",
                "MERGED_OR_FRAGMENTED_SUPPLY",
                "NO_SUBJECT_CANDIDATE_SUPPLY",
                "FRAME_LOCAL_VISIBLE_PERSON_MISSING_BOX",
            ],
            "missed_semantic_label": "FRAME_LOCAL_MISSING_BOX_OBSERVATION",
            "subject_no_candidate_and_missed_populations_combined": False,
        },
    )
    priority, priority_md, gap = _priority_artifacts(baseline)
    write_json(priority_dir / "rd_priority_map_v1.json", priority)
    (priority_dir / "rd_priority_map_v1.md").write_text(priority_md, encoding="utf-8", newline="\n")
    write_json(priority_dir / "gold_gap_register.json", gap)

    normalized_hashes = {name: sha256_file(gold_dir / name) for name in NORMALIZED_FILENAMES}
    schema_hashes = {name: sha256_file(schema_dir / name) for name in schemas}
    manifest = {
        "schema_version": "football_intelligence.g7f_a.normalization_manifest.v1",
        "gold_corpus_version": GOLD_CORPUS_VERSION,
        "gold_scope": GOLD_SCOPE,
        "exhaustive_person_detection_ground_truth": False,
        "repository_commit": head,
        "source_real_root_inventory_sha256": before["ordered_inventory_sha256"],
        "source_global_receipt_id": EXPECTED_GLOBAL_RECEIPT,
        "source_tranche_receipt_ids": EXPECTED_TRANCHE_RECEIPTS,
        **source["package_bindings"],
        "normalized_file_sha256": normalized_hashes,
        "schema_sha256": schema_hashes,
        "row_counts": source["counts"],
        "normalization_code_commit": head,
        "normalization_code_sha256": evaluator_code_sha256(),
        "production_ready": False,
    }
    write_json(gold_dir / "gold_normalization_manifest.json", manifest)
    evaluation_bindings = {
        "gold_corpus_manifest_sha256": sha256_file(gold_dir / "gold_normalization_manifest.json"),
        "split_manifest_sha256": sha256_file(split_dir / "gold_split_manifest_v1.json"),
        "candidate_run_sha256": sha256_file(gold_dir / "gold_selected_candidate_bindings.jsonl"),
        "evaluator_code_commit": head,
        "evaluator_code_sha256": evaluator_code_sha256(),
        "metric_contract_sha256": sha256_file(harness_dir / "metric_capabilities.json"),
    }
    baseline["evaluation_bindings"] = evaluation_bindings
    by_match["evaluation_bindings"] = evaluation_bindings
    fold_output = {"folds": by_fold, "evaluation_bindings": evaluation_bindings}
    bound_ledger = [{**row, "evaluation_bindings": evaluation_bindings} for row in ledger]
    write_json(baseline_dir / "baseline_current_candidate_system.json", baseline)
    write_json(baseline_dir / "baseline_by_match.json", by_match)
    write_json(baseline_dir / "baseline_by_fold.json", fold_output)
    write_jsonl(baseline_dir / "baseline_error_ledger.jsonl", bound_ledger)
    after = inventory_tree(decisions_root)
    _require(before == after, "real human-decision root changed during normalization")
    write_json(source_dir / "real_root_inventory_before.json", before)
    write_json(source_dir / "real_root_inventory_after.json", after)
    write_json(source_dir / "source_package_bindings.json", source["package_bindings"])
    write_json(source_dir / "receipt_chain.json", source["receipt_chain"])
    write_json(
        integrity_dir / "source_and_gold_integrity.json",
        {
            "source_before": before,
            "source_after": after,
            "source_byte_identical": True,
            "zero_source_writes": True,
            "zero_current_drafts": True,
            "counts": source["counts"],
            "repair_bursts": source["repair_bursts"],
            "schema_revision_counts": source["schema_revision_counts"],
            "production_ready": False,
        },
    )
    report = validate_gold(output_root)
    write_json(integrity_dir / "gold_validation_report.json", report)
    return {"manifest_sha256": sha256_file(gold_dir / "gold_normalization_manifest.json"), **report}


def validate_gold(workspace: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    gold_dir = workspace / "02_NORMALIZED_GOLD"
    split_dir = workspace / "03_SPLITS"
    harness_dir = workspace / "04_EVALUATION_HARNESS"
    manifest_path = gold_dir / "gold_normalization_manifest.json"
    manifest = read_json(manifest_path)
    _require(manifest["gold_scope"] == GOLD_SCOPE, "gold scope mismatch")
    _require(manifest.get("exhaustive_person_detection_ground_truth") is False, "exhaustive truth misclassification")
    rows = {name: read_jsonl(gold_dir / name) for name in NORMALIZED_FILENAMES}
    counts = {
        "bursts": len(rows["gold_bursts.jsonl"]),
        "subjects": len(rows["gold_subjects.jsonl"]),
        "subject_frames": len(rows["gold_subject_frames.jsonl"]),
        "missed_observations": len(rows["gold_missed_observations.jsonl"]),
        "selected_candidate_bindings": len(rows["gold_selected_candidate_bindings.jsonl"]),
        "source_event_index": len(rows["gold_source_event_index.jsonl"]),
    }
    for key, expected in EXPECTED_COUNTS.items():
        if key != "accepted_actions":
            _require(counts[key] == expected, f"normalized {key} count")
    _require(counts["source_event_index"] == 120, "source event index count")
    _require(len({row["burst_id"] for row in rows["gold_bursts.jsonl"]}) == 120, "duplicate burst key")
    _require(
        len({(row["burst_id"], row["subject_token"]) for row in rows["gold_subjects.jsonl"]}) == 108,
        "duplicate subject key",
    )
    _require(
        len(
            {
                (row["burst_id"], row["subject_token"], row["frame_sequence"])
                for row in rows["gold_subject_frames.jsonl"]
            }
        )
        == 972,
        "duplicate subject-frame key",
    )
    _require(
        len({(row["burst_id"], row["mark_id"]) for row in rows["gold_missed_observations.jsonl"]}) == 763,
        "duplicate missed key",
    )
    _require(
        len(
            {
                (row["burst_id"], row["subject_token"], row["frame_sequence"], row["candidate_id"])
                for row in rows["gold_selected_candidate_bindings.jsonl"]
            }
        )
        == counts["selected_candidate_bindings"],
        "duplicate selected-candidate binding key",
    )
    subject_keys = {(row["burst_id"], row["subject_token"]) for row in rows["gold_subjects.jsonl"]}
    _require(
        all((row["burst_id"], row["subject_token"]) in subject_keys for row in rows["gold_subject_frames.jsonl"]),
        "orphan subject frame",
    )
    _require(all(0 <= row["frame_sequence"] <= 8 for row in rows["gold_subject_frames.jsonl"]), "frame sequence")
    _require(
        all(
            not row["role_was_explicitly_elicited"] and not row["certainty_was_explicitly_elicited"]
            for row in rows["gold_missed_observations.jsonl"]
        ),
        "missed defaults reinterpreted",
    )
    _require(
        sum(
            row["occlusion_sequence_answer_availability"] == MISSING_HISTORICAL_FIELD
            for row in rows["gold_subjects.jsonl"]
        )
        == 106,
        "historical named-occlusion availability count",
    )
    frame_by_key = {
        (row["burst_id"], row["subject_token"], row["frame_sequence"]): row for row in rows["gold_subject_frames.jsonl"]
    }
    for binding in rows["gold_selected_candidate_bindings.jsonl"]:
        frame = frame_by_key[(binding["burst_id"], binding["subject_token"], binding["frame_sequence"])]
        candidate_index = frame["selected_candidate_ids"].index(binding["candidate_id"])
        _require(
            frame["selected_candidate_frozen_source_boxes"][candidate_index] == binding["source_box_xyxy"],
            "selected-candidate normalized box mismatch",
        )
    for mark in rows["gold_missed_observations.jsonl"]:
        x, y = mark["source_coordinate"]
        _require(0 <= x <= mark["source_width"] and 0 <= y <= mark["source_height"], "missed mark bounds")
    repair_expectations = {
        "g7e_a_117092_03": (["SUBJECT_A", "SUBJECT_B"], 24),
        "g7e_a_118577_14": (["SUBJECT_A", "SUBJECT_B"], 9),
        "g7e_a_117092_10": (["SUBJECT_A"], 8),
    }
    for burst_id, (expected_subjects, expected_marks) in repair_expectations.items():
        actual_subjects = sorted(
            row["subject_token"] for row in rows["gold_subjects.jsonl"] if row["burst_id"] == burst_id
        )
        actual_marks = sum(row["burst_id"] == burst_id for row in rows["gold_missed_observations.jsonl"])
        _require(actual_subjects == expected_subjects and actual_marks == expected_marks, f"{burst_id}: repair row")
    repaired_subject_b = next(
        row
        for row in rows["gold_subjects.jsonl"]
        if row["burst_id"] == "g7e_a_118577_14" and row["subject_token"] == "SUBJECT_B"
    )
    _require(repaired_subject_b["occlusion_sequence_answer"] == "OCCLUDED", "R6.6 named occlusion answer")
    repaired_phases = [
        row["occlusion_phase"]
        for row in rows["gold_subject_frames.jsonl"]
        if row["burst_id"] == "g7e_a_118577_14" and row["subject_token"] == "SUBJECT_B"
    ]
    _require(repaired_phases == ["ENTERING_OCCLUSION"] * 9, "R6.6 frame-phase sequence")
    for filename, expected_hash in manifest["normalized_file_sha256"].items():
        _require(sha256_file(gold_dir / filename) == expected_hash, f"normalized hash mismatch: {filename}")
    split = read_json(split_dir / "gold_split_manifest_v1.json")
    all_bursts = {row["burst_id"] for row in rows["gold_bursts.jsonl"]}
    for fold in split["folds"]:
        development = set(fold["development_burst_ids"])
        evaluation = set(fold["evaluation_burst_ids"])
        _require(not development & evaluation, f"split leakage: {fold['fold_id']}")
        _require(development | evaluation == all_bursts, f"split coverage: {fold['fold_id']}")
        held_out = fold["held_out_match_id"]
        _require(
            all(row["match_id"] == held_out for row in rows["gold_bursts.jsonl"] if row["burst_id"] in evaluation),
            f"match leakage: {fold['fold_id']}",
        )
    future = read_json(split_dir / "future_sealed_holdout_manifest.json")
    _require(not future["current_gold_corpus_burst_ids"], "future sealed holdout contains current cases")
    capabilities = read_json(harness_dir / "metric_capabilities.json")
    _require(capabilities["unknown_metric_policy"] == "FAIL_CLOSED", "metric guard not fail-closed")
    return {
        "classification": "PASS_G7F_A_GOLD_VALIDATION",
        "gold_scope": GOLD_SCOPE,
        "counts": counts,
        "fold_count": len(split["folds"]),
        "zero_match_leakage": True,
        "future_sealed_holdout_current_case_count": 0,
        "manifest_sha256": sha256_file(manifest_path),
        "split_manifest_sha256": sha256_file(split_dir / "gold_split_manifest_v1.json"),
        "metric_contract_sha256": sha256_file(harness_dir / "metric_capabilities.json"),
        "production_ready": False,
    }
