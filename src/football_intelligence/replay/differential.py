from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any

from football_intelligence.core.fingerprint_policy import SemanticFingerprintPolicy
from football_intelligence.core.fingerprints import media_type_for_path, semantic_hash, sha256_file
from football_intelligence.replay.contracts import M4_STRUCTURED_FILES
from football_intelligence.replay.m4_renderer import evidence_inventory, evidence_records

RUNTIME_PATH_POLICY = SemanticFingerprintPolicy(
    excluded_json_paths=frozenset(
        {
            "$.source_m3t_folder",
            "$.viewer_path",
            "$.validation_summary_path",
            "$.handoff_manifest_path",
        }
    )
)

TRUE_REPLAY_RUNTIME_PATH_POLICY = SemanticFingerprintPolicy(
    excluded_json_paths=RUNTIME_PATH_POLICY.excluded_json_paths
    | frozenset(
        {
            "$.step2_visual_continuity_root",
            "$.m3t_read_root",
            "$.m4_write_root",
        }
    )
)


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def read_artifact(path: Path, mode: str) -> tuple[Any, list[Any] | None]:
    if mode == "jsonl_gzip_rows":
        rows: list[Any] = []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows, rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if mode == "rows":
        rows = data.get("rows", []) if isinstance(data, dict) else data
        return rows, rows
    return data, None


def m4_structured_fingerprints(
    m4_root: Path,
    m3t_decision_path: Path,
    artifact_root: Path,
    *,
    policy: SemanticFingerprintPolicy | None = None,
) -> dict[str, Any]:
    active_policy = policy or RUNTIME_PATH_POLICY
    records: list[dict[str, Any]] = []
    for artifact_name, (filename, mode, _key) in M4_STRUCTURED_FILES.items():
        path = m4_root / filename
        payload, rows = read_artifact(path, mode)
        records.append(
            {
                "artifact_name": artifact_name,
                "source_relative_uri": _relative_or_absolute(path, artifact_root),
                "source_byte_hash": sha256_file(path),
                "row_count": len(rows) if rows is not None else None,
                "schema_or_artifact_name": artifact_name,
                "semantic_content_hash": semantic_hash(payload, policy=active_policy),
                "ordering_policy": mode,
                "excluded_runtime_fields": sorted(active_policy.excluded_field_names),
                "excluded_json_paths": sorted(active_policy.excluded_json_paths),
                "parse_status": "ok",
            }
        )
    decision_payload, decision_rows = read_artifact(m3t_decision_path, "rows")
    records.append(
        {
            "artifact_name": "m3t_reviewed_decisions",
            "source_relative_uri": _relative_or_absolute(m3t_decision_path, artifact_root),
            "source_byte_hash": sha256_file(m3t_decision_path),
            "row_count": len(decision_rows or []),
            "schema_or_artifact_name": "m3t_reviewed_decisions",
            "semantic_content_hash": semantic_hash(decision_payload, policy=active_policy),
            "ordering_policy": "rows",
            "excluded_runtime_fields": sorted(active_policy.excluded_field_names),
            "excluded_json_paths": sorted(active_policy.excluded_json_paths),
            "parse_status": "ok",
        }
    )
    return {
        "schema_version": "m5.m4_structured_fingerprints.v1",
        "fingerprints": records,
        "structured_content_hash": semantic_hash(
            [
                {"artifact_name": record["artifact_name"], "semantic_content_hash": record["semantic_content_hash"]}
                for record in records
            ]
        ),
    }


def _keyed(rows: list[Any], key: str) -> dict[str, Any]:
    return {str(row.get(key, index)): row for index, row in enumerate(rows) if isinstance(row, dict)}


def structured_diff(
    baseline_root: Path,
    replay_root: Path,
    *,
    policy: SemanticFingerprintPolicy | None = None,
) -> dict[str, Any]:
    active_policy = policy or RUNTIME_PATH_POLICY
    reports = []
    for artifact_name, (filename, mode, key) in M4_STRUCTURED_FILES.items():
        left = baseline_root / filename
        right = replay_root / filename
        left_payload, left_rows = read_artifact(left, mode)
        right_payload, right_rows = read_artifact(right, mode)
        left_hash = semantic_hash(left_payload, policy=active_policy)
        right_hash = semantic_hash(right_payload, policy=active_policy)
        missing: list[str] = []
        extra: list[str] = []
        changed: list[str] = []
        sample: list[dict[str, Any]] = []
        if left_rows is not None and right_rows is not None:
            left_map = _keyed(left_rows, key)
            right_map = _keyed(right_rows, key)
            missing = sorted(set(left_map) - set(right_map))
            extra = sorted(set(right_map) - set(left_map))
            for row_key in sorted(set(left_map) & set(right_map)):
                if semantic_hash(left_map[row_key], policy=active_policy) != semantic_hash(
                    right_map[row_key], policy=active_policy
                ):
                    changed.append(row_key)
                    if len(sample) < 5:
                        sample.append({"row_key": row_key})
        ordering_match = semantic_hash(left_payload, policy=active_policy) == semantic_hash(
            right_payload, policy=active_policy
        )
        passed = left_hash == right_hash and not missing and not extra and not changed and ordering_match
        reports.append(
            {
                "artifact_name": artifact_name,
                "baseline_uri": left.as_posix(),
                "replay_uri": right.as_posix(),
                "baseline_byte_hash": sha256_file(left),
                "replay_byte_hash": sha256_file(right),
                "baseline_semantic_hash": left_hash,
                "replay_semantic_hash": right_hash,
                "row_count": len(left_rows) if left_rows is not None else None,
                "missing_row_keys": missing,
                "extra_row_keys": extra,
                "changed_row_keys": changed,
                "changed_field_sample": sample,
                "ordering_comparison": {"ordered_semantic_match": ordering_match},
                "classification": "exact_semantic_match" if passed else "unknown_requires_diagnosis",
                "passed": passed,
            }
        )
    return {
        "schema_version": "m5.replay.structured_diff.v1",
        "artifacts": reports,
        "passed": all(r["passed"] for r in reports),
    }


def media_diff(baseline_root: Path, replay_root: Path) -> dict[str, Any]:
    left = {record["relative_uri"]: record for record in evidence_records(baseline_root)}
    right = {record["relative_uri"]: record for record in evidence_records(replay_root)}
    records = []
    for relative_uri in sorted(set(left) | set(right)):
        left_record = left.get(relative_uri)
        right_record = right.get(relative_uri)
        if left_record is None:
            classification = "extra_asset"
            passed = False
        elif right_record is None:
            classification = "missing_asset"
            passed = False
        elif left_record["decoded_content_hash"] == right_record["decoded_content_hash"]:
            classification = (
                "exact_byte_match"
                if left_record["content_hash"] == right_record["content_hash"]
                else "decoded_pixel_match_container_differs"
            )
            passed = True
        else:
            classification = "decoded_content_mismatch"
            passed = False
        path = replay_root / relative_uri
        records.append(
            {
                "relative_asset_uri": relative_uri,
                "exists": left_record is not None and right_record is not None,
                "byte_size": None if right_record is None else right_record["byte_size"],
                "byte_hash": None if right_record is None else right_record["content_hash"],
                "media_type": media_type_for_path(path),
                "dimensions": None if right_record is None else right_record["dimensions"],
                "decoded_frame_count": None if right_record is None else right_record["decoded_frame_count"],
                "pixel_hash": None if right_record is None else right_record["decoded_content_hash"],
                "ordered_decoded_frame_hashes": None if right_record is None else right_record["decoded_frame_hashes"],
                "ffprobe_metadata": None,
                "classification": classification,
                "passed": passed,
            }
        )
    inventory = evidence_inventory(replay_root)
    return {
        "schema_version": "m5.replay.media_diff.v1",
        "asset_count": len(records),
        "evidence_inventory_hash": inventory["evidence_inventory_hash"],
        "records": records,
        "passed": all(record["passed"] for record in records),
    }


def _viewer_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"const data=(.*?);\nlet rows=", text, flags=re.S)
    payload = json.loads(match.group(1)) if match else {}
    return {"text": text, "payload": payload}


def viewer_diff(
    baseline_root: Path,
    replay_root: Path,
    *,
    policy: SemanticFingerprintPolicy | None = None,
) -> dict[str, Any]:
    active_policy = policy or RUNTIME_PATH_POLICY
    baseline = _viewer_payload(baseline_root / "step2m4_sparse_handoff_viewer.html")
    replay = _viewer_payload(replay_root / "step2m4_sparse_handoff_viewer.html")
    rows = replay["payload"].get("rows", [])
    links = [
        value for row in rows for value in (row.get("m4_overlay_gif_path"), row.get("m4_overlay_strip_path")) if value
    ]
    resolved = []
    escapes = []
    for link in links:
        path = (replay_root / link).resolve()
        if not path.is_relative_to(replay_root.resolve()):
            escapes.append(link)
        resolved.append({"link": link, "exists": path.exists()})
    summary = replay["payload"].get("summary", {})
    normalized_match = semantic_hash(baseline["payload"], policy=active_policy) == semantic_hash(
        replay["payload"], policy=active_policy
    )
    return {
        "schema_version": "m5.replay.viewer_diff.v1",
        "viewer_exists": (replay_root / "step2m4_sparse_handoff_viewer.html").exists(),
        "viewer_non_empty": bool(replay["text"]),
        "visual_only_warning_exists": "visual-only" in replay["text"].lower(),
        "identity_warning_exists": "do not infer identity" in replay["text"].lower()
        or "not identity" in replay["text"].lower(),
        "required_relative_evidence_links": resolved,
        "embedded_pathlet_row_count": len(rows),
        "embedded_summary_counts": {
            "m4_handoff_pathlet_count": summary.get("m4_handoff_pathlet_count"),
            "m4_handoff_edge_count": summary.get("m4_handoff_edge_count"),
            "overlay_asset_count": summary.get("overlay_asset_count"),
        },
        "normalized_embedded_json_semantic_hash": semantic_hash(replay["payload"], policy=active_policy),
        "normalized_embedded_json_matches_preserved": normalized_match,
        "links_to_preserved_m4_root": [link for link in links if "step2_visual_continuity/step2m4" in link],
        "link_escapes": escapes,
        "passed": bool(replay["text"])
        and normalized_match
        and all(item["exists"] for item in resolved)
        and not escapes,
    }


def compare_replay_runs(left_run: Path, right_run: Path) -> dict[str, Any]:
    left_summary = json.loads((left_run / "validation/replay_validation_summary.json").read_text(encoding="utf-8"))
    right_summary = json.loads((right_run / "validation/replay_validation_summary.json").read_text(encoding="utf-8"))
    keys = [
        "input_closure_hash",
        "replay_config_hash",
        "code_commit",
        "reconstructed_structured_content_hash",
        "evidence_inventory_hash",
        "viewer_semantic_hash",
    ]
    comparisons = {f"{key}_equal": left_summary.get(key) == right_summary.get(key) for key in keys}
    comparisons["counts_equal"] = left_summary.get("counts") == right_summary.get("counts")
    comparisons["guardrails_passed"] = left_summary.get("guardrail_passed") and right_summary.get("guardrail_passed")
    comparisons["source_mutation_passed"] = left_summary.get("source_mutation_passed") and right_summary.get(
        "source_mutation_passed"
    )
    return {
        "schema_version": "m5.replay.run_comparison.v1",
        "left_run": str(left_run),
        "right_run": str(right_run),
        "comparisons": comparisons,
        "passed": all(comparisons.values()),
    }


def compare_true_replay_runs(left_run: Path, right_run: Path) -> dict[str, Any]:
    left_summary = json.loads((left_run / "validation/true_replay_validation_summary.json").read_text(encoding="utf-8"))
    right_summary = json.loads(
        (right_run / "validation/true_replay_validation_summary.json").read_text(encoding="utf-8")
    )
    keys = [
        "true_input_closure_hash",
        "replay_config_hash",
        "code_commit",
        "recovered_m1_semantic_hash",
        "reconstructed_structured_content_hash",
        "evidence_inventory_hash",
        "viewer_semantic_hash",
        "canonical_m3t_decision_semantic_hash",
    ]
    comparisons = {f"{key}_equal": left_summary.get(key) == right_summary.get(key) for key in keys}
    comparisons["counts_equal"] = left_summary.get("counts") == right_summary.get("counts")
    comparisons["guardrails_passed"] = left_summary.get("guardrail_passed") and right_summary.get("guardrail_passed")
    comparisons["source_mutation_passed"] = left_summary.get("source_mutation_passed") and right_summary.get(
        "source_mutation_passed"
    )
    comparisons["source_access_passed"] = left_summary.get("source_access_passed") and right_summary.get(
        "source_access_passed"
    )
    return {
        "schema_version": "m5.true_replay.run_comparison.v1",
        "left_run": str(left_run),
        "right_run": str(right_run),
        "comparisons": comparisons,
        "passed": all(comparisons.values()),
    }
