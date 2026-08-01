"""Deterministic contracts for G7E-A burst-local temporal selection."""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Iterable, Mapping

MATCHES = ("117092", "117093", "118575", "118576", "118577", "128058")

CLASS_PRIORITY = (
    "OCCLUSION_OR_MERGE_RISK",
    "PROPOSAL_MISS_RISK",
    "GOALMOUTH_OR_ENDLINE_CROWD",
    "OFFICIAL_OR_BOUNDARY_CONTINUITY",
    "FRAGMENT_OR_DUPLICATE_RISK",
    "FAR_SIDE_CROWDING",
    "STABLE_OPEN_PLAY_CONTROL",
)

QUOTAS = {
    "OCCLUSION_OR_MERGE_RISK": 4,
    "FRAGMENT_OR_DUPLICATE_RISK": 3,
    "PROPOSAL_MISS_RISK": 3,
    "FAR_SIDE_CROWDING": 3,
    "GOALMOUTH_OR_ENDLINE_CROWD": 2,
    "OFFICIAL_OR_BOUNDARY_CONTINUITY": 1,
    "STABLE_OPEN_PLAY_CONTROL": 4,
}

OFFSETS_SECONDS = tuple(Decimal(str(value)) for value in (-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8))

ONTOLOGY = {
    "visibility": (
        "VISIBLE_COMPLETE",
        "VISIBLE_PARTIAL",
        "FULLY_OCCLUDED_EXPECTED_PRESENT",
        "OUT_OF_FRAME_OR_LEFT_SCENE",
        "NOT_PRESENT",
        "UNCERTAIN",
    ),
    "observation_supply": (
        "ONE_USEFUL_CANDIDATE",
        "MULTIPLE_CANDIDATES",
        "MERGED_WITH_OTHER_PEOPLE",
        "FRAGMENT_ONLY",
        "NO_CANDIDATE",
        "NOT_APPLICABLE",
        "UNCERTAIN",
    ),
    "occlusion_phase": ("NONE", "ENTERING_OCCLUSION", "OCCLUDED", "EXITING_OCCLUSION", "UNCERTAIN"),
    "candidate_relationship": (
        "SAME_PERSON_DUPLICATES",
        "SAME_PERSON_FRAGMENTS",
        "DIFFERENT_PEOPLE",
        "CORRECT_INNER_BAD_OUTER",
        "MERGED_MULTI_PERSON",
        "OBJECT_OR_BACKGROUND",
        "UNCERTAIN",
    ),
    "continuity": (
        "SAME_BURST_LOCAL_SUBJECT",
        "DIFFERENT_SUBJECT",
        "CANNOT_TELL",
        "NOT_APPLICABLE",
    ),
    "role": ("OUTFIELD_PLAYER", "GOALKEEPER", "RELEVANT_OFFICIAL", "OTHER_PERSON", "UNKNOWN_ROLE"),
    "participation": (
        "ACTIVE_IN_MATCH",
        "WARMING_OR_INACTIVE",
        "NOT_PLAYER_OR_OFFICIAL",
        "UNKNOWN_PARTICIPATION",
    ),
    "certainty": ("CERTAIN", "PROBABLE", "NOT_SURE"),
}


def round_half_up_frame(timestamp_seconds: Decimal, fps: Decimal) -> int:
    """Resolve a timestamp with the repository's nearest-frame convention."""
    return int((timestamp_seconds * fps).to_integral_value(rounding=ROUND_HALF_UP))


def frame_indices_for_centre(centre_frame_index: int, fps: Decimal) -> tuple[int, ...]:
    """Return the exact nine zero-based frame indices for a burst centre."""
    return tuple(
        centre_frame_index + int((offset * fps).to_integral_value(rounding=ROUND_HALF_UP)) for offset in OFFSETS_SECONDS
    )


def overlap_count(left: Iterable[int], right: Iterable[int]) -> int:
    return len(set(left) & set(right))


def slot_plan() -> list[dict[str, str]]:
    """Return the frozen 20-slot class/half/perspective plan for every match."""
    layout = {
        "OCCLUSION_OR_MERGE_RISK": (
            ("FIRST_HALF", "FAR"),
            ("FIRST_HALF", "NEAR_MIDDLE"),
            ("SECOND_HALF", "FAR"),
            ("SECOND_HALF", "NEAR_MIDDLE"),
        ),
        "PROPOSAL_MISS_RISK": (
            ("SECOND_HALF", "FAR"),
            ("FIRST_HALF", "ANY"),
            ("SECOND_HALF", "ANY"),
        ),
        "GOALMOUTH_OR_ENDLINE_CROWD": (("FIRST_HALF", "NEAR_MIDDLE"), ("SECOND_HALF", "FAR")),
        "OFFICIAL_OR_BOUNDARY_CONTINUITY": (("SECOND_HALF", "ANY"),),
        "FRAGMENT_OR_DUPLICATE_RISK": (
            ("FIRST_HALF", "FAR"),
            ("SECOND_HALF", "FAR"),
            ("FIRST_HALF", "ANY"),
        ),
        "FAR_SIDE_CROWDING": (("FIRST_HALF", "FAR"), ("SECOND_HALF", "FAR"), ("FIRST_HALF", "FAR")),
        "STABLE_OPEN_PLAY_CONTROL": (
            ("FIRST_HALF", "NEAR_MIDDLE"),
            ("SECOND_HALF", "NEAR_MIDDLE"),
            ("FIRST_HALF", "NEAR_MIDDLE"),
            ("SECOND_HALF", "NEAR_MIDDLE"),
        ),
    }
    slots: list[dict[str, str]] = []
    for selection_class in CLASS_PRIORITY:
        for half, perspective in layout[selection_class]:
            slots.append(
                {
                    "selection_class": selection_class,
                    "required_half": half,
                    "preferred_perspective": perspective,
                }
            )
    assert Counter(slot["selection_class"] for slot in slots) == Counter(QUOTAS)
    assert len(slots) == 20
    return slots


def validate_burst_records(bursts: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate the frozen cardinality, balance, and companion invariants."""
    rows = list(bursts)
    errors: list[str] = []
    if len(rows) != 120:
        errors.append(f"burst_count={len(rows)}")
    by_match: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_match[str(row.get("match_id"))].append(row)
        indices = tuple(row.get("frame_indices_zero_based", ()))
        if len(indices) != 9 or len(set(indices)) != 9:
            errors.append(f"nine_distinct_frames:{row.get('burst_id')}")
    if tuple(sorted(by_match)) != MATCHES:
        errors.append(f"matches={tuple(sorted(by_match))}")
    for match_id in MATCHES:
        match_rows = by_match.get(match_id, [])
        if len(match_rows) != 20:
            errors.append(f"{match_id}:count={len(match_rows)}")
        counts = Counter(str(row.get("primary_selection_class")) for row in match_rows)
        if counts != Counter(QUOTAS):
            errors.append(f"{match_id}:quotas={dict(counts)}")
        halves = Counter(str(row.get("half")) for row in match_rows)
        if halves["FIRST_HALF"] < 8 or halves["SECOND_HALF"] < 8:
            errors.append(f"{match_id}:halves={dict(halves)}")
        perspectives = Counter(str(row.get("perspective_band")) for row in match_rows)
        if perspectives["FAR"] < 6 or perspectives["NEAR_MIDDLE"] < 4:
            errors.append(f"{match_id}:perspectives={dict(perspectives)}")
        if counts["STABLE_OPEN_PLAY_CONTROL"] != 4:
            errors.append(f"{match_id}:stable_controls")
        companions = [row for row in match_rows if row.get("companion")]
        if len(companions) > 6:
            errors.append(f"{match_id}:companions={len(companions)}")
        for index, left in enumerate(match_rows):
            if not left.get("companion"):
                continue
            for right in match_rows[:index]:
                if left.get("source_video_relative_path") != right.get("source_video_relative_path"):
                    continue
                if overlap_count(left["frame_indices_zero_based"], right["frame_indices_zero_based"]) > 4:
                    errors.append(f"companion_overlap:{left['burst_id']}:{right['burst_id']}")
    high_fallbacks = sum(int(row.get("fallback_level", 0)) >= 3 for row in rows)
    if high_fallbacks / max(len(rows), 1) > 0.15:
        errors.append(f"high_fallback_rate={high_fallbacks / len(rows):.6f}")
    return {"valid": not errors, "errors": errors, "high_fallback_count": high_fallbacks}


def validate_ontology(ontology: Mapping[str, Iterable[str]]) -> bool:
    return all(tuple(ontology.get(key, ())) == values for key, values in ONTOLOGY.items())
