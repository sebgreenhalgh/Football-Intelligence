from __future__ import annotations

import json
from pathlib import Path

from football_intelligence.replay.occlusion_pro_context_pack import (
    PACK_FILENAMES,
    _forbidden_field_scan,
    _line_range_for_symbol,
    _pack_hash,
    deterministic_pack_hash,
)


def test_context_pack_contract_is_exactly_twenty_flat_ordered_files() -> None:
    assert len(PACK_FILENAMES) == 20
    assert PACK_FILENAMES == [
        "01_EXECUTIVE_BRIEFING.md",
        "02_SYSTEM_ARCHITECTURE_MAP.md",
        "03_CURRENT_STATE_AND_PROVENANCE.json",
        "04_SAFETY_SCOPE_AND_IDENTITY_BOUNDARIES.md",
        "05_DATA_CONTRACTS_AND_SCHEMAS.md",
        "06_DETECTION_AND_PERSON_PIPELINE_EXCERPTS.py",
        "07_CONTINUITY_AND_CHALLENGE_PIPELINE_EXCERPTS.py",
        "08_REVIEW_CHASSIS_AND_PERSISTENCE_EXCERPTS.py",
        "09_LOCALIZATION_AND_UPSTREAM_AUDIT_EXCERPTS.py",
        "10_OCCLUSION_FAILURE_TAXONOMY_AND_KNOWN_WEAKNESSES.md",
        "11_CURRENT_EVALUATION_LABELS_AND_GROUPING.json",
        "12_OCCLUSION_RESEARCH_QUESTIONS_AND_DECISION_GATES.md",
        "13_REPRESENTATIVE_CASE_INDEX.csv",
        "14_CROSSING_FAILURE_CASE_008.gif",
        "15_CROSSING_FAILURE_CASE_010.gif",
        "16_CROSSING_FAILURE_CASE_013.gif",
        "17_SHARED_OCCLUSION_REGION_CASES_004_016.gif",
        "18_FULL_FRAME_DETECTION_CONTACT_SHEET.jpg",
        "19_TARGETED_CODEBASE_FILE_MAP.md",
        "20_PACK_MANIFEST.json",
    ]
    assert all("/" not in name and "\\" not in name for name in PACK_FILENAMES)


def test_pack_hash_excludes_generated_at_but_preserves_other_content(tmp_path: Path) -> None:
    output_root = tmp_path / "pack"
    output_root.mkdir()
    for filename in PACK_FILENAMES:
        (output_root / filename).write_text(f"{filename}\n", encoding="utf-8")
    files = [
        {
            "filename": filename,
            "sha256": "sha",
            "byte_size": 123,
            "media_type": "text/plain",
            "source_artifact_paths": [],
        }
        for filename in PACK_FILENAMES
    ]
    manifest = {
        "schema_version": "test",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "pack_hash": "<pending>",
        "files": files,
    }
    changed_timestamp = dict(manifest, generated_at="2026-01-01T00:00:01+00:00")
    changed_content = dict(manifest, schema_version="changed")

    assert _pack_hash(output_root, manifest) == _pack_hash(output_root, changed_timestamp)
    assert _pack_hash(output_root, manifest) != _pack_hash(output_root, changed_content)
    assert deterministic_pack_hash(manifest) == deterministic_pack_hash(changed_timestamp)


def test_forbidden_answer_key_scan_ignores_source_excerpts_but_flags_served_pack_text(tmp_path: Path) -> None:
    (tmp_path / "source_excerpt.py").write_text(
        "decision_to_output_mapping = 'schema field in source excerpt only'\n",
        encoding="utf-8",
    )
    (tmp_path / "served_context.md").write_text("accepted_target_panel\n", encoding="utf-8")

    result = _forbidden_field_scan(tmp_path)

    assert result["answer_key_field_count"] == 1
    assert result["hits"] == [{"filename": "served_context.md", "field": "accepted_target_panel"}]


def test_source_symbol_extraction_uses_ast_line_ranges(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        "\n".join(
            [
                "class First:",
                "    pass",
                "",
                "def target(value: int) -> int:",
                "    return value + 1",
                "",
                "def other() -> None:",
                "    pass",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    start, end, text = _line_range_for_symbol(source, "target")

    assert (start, end) == (4, 5)
    assert "def target(value: int) -> int:" in text
    assert json.loads(json.dumps({"start": start, "end": end})) == {"start": 4, "end": 5}
