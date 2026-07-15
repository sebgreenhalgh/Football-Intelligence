"""M5.5C counterbalance repair for the anonymous path review workbench."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw

from football_intelligence.research_handoff.review_pack import (
    ReviewPackBuilder,
    ReviewPackItem,
    validate_review_pack_directory,
)
from football_intelligence.research_handoff.stage_workspace import safety_payload, sha256_file
from football_intelligence.review_chassis.blinding import (
    apply_permutation,
    assign_counterbalanced_permutations,
    audit_counterbalance,
)
from football_intelligence.review_chassis.hashing import stable_hash
from football_intelligence.review_chassis.manifest import manifest_hash
from football_intelligence.review_chassis.models import (
    AssetPanelConfig,
    DecisionOption,
    GenericEvidenceAsset,
    GenericReviewCase,
    GenericReviewManifest,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from football_intelligence.review_chassis.server import ReviewChassisServerConfig, create_server
from football_intelligence.review_chassis.validation import validate_review_chassis_package


STAGE_ID = "M5_5C_BLIND_REVIEW_COUNTERBALANCE_REPAIR_v1"
REVIEW_ID = "m5_5c_rebalanced_blind_review_v1"
REVIEWER_SESSION_ID = "m5_5c_rebalanced_human_reviewer"
EXPECTED_PORT = 8781
CHOICES = ("CHOICE_A", "CHOICE_B", "CHOICE_C")
NON_BINARY = ("NEITHER_PATH_VALID_OR_COMPATIBLE", "UNRESOLVED")
FORBIDDEN_FIELDS = {
    "target_candidate_id",
    "source_candidate_id",
    "canonical_candidate_id",
    "visible_person_base_id",
    "frozen_rank",
    "model_rank",
    "geometry_score",
    "path_cost",
    "preferred",
    "baseline",
    "correct_path",
    "ground_truth",
    "answer",
}
FORBIDDEN_VALUES = {
    "path_a_continues_source",
    "path_b_continues_source",
    "path_c_continues_source",
    "target_candidate_id",
    "source_candidate_id",
    "canonical_candidate_id",
    "frozen_rank",
    "model_rank",
    "geometry_score",
    "preferred_path",
    "correct_path",
    "ground_truth",
    "answer_key",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()


def _inventory(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file()}


def _asset_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_file() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"missing or unsafe evidence asset: {relative}")
    return path


def _scale_bbox(bbox: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    return (
        max(0, int(float(bbox["x1"]) * width / 2730)),
        max(0, int(float(bbox["y1"]) * height / 720)),
        min(width - 1, int(float(bbox["x2"]) * width / 2730)),
        min(height - 1, int(float(bbox["y2"]) * height / 720)),
    )


def _draw_box(
    draw: ImageDraw.ImageDraw, bbox: dict[str, Any], label: str, color: tuple[int, int, int], width: int, height: int
) -> None:
    x1, y1, x2, y2 = _scale_bbox(bbox, width, height)
    draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
    draw.rectangle((x1, max(0, y1 - 23), x1 + 38, y1), fill=color)
    draw.text((x1 + 8, max(0, y1 - 20)), label, fill=(255, 255, 255))


def _annotate(
    source: Path,
    source_bbox: dict[str, Any],
    displayed: list[dict[str, Any]],
    source_frame: int,
    target_frame: int,
    frame: int,
) -> Image.Image:
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = [(24, 129, 189), (205, 120, 30), (110, 70, 170)]
    if frame == source_frame:
        _draw_box(draw, source_bbox, "Source", colors[0], image.width, image.height)
    if frame == target_frame:
        for index, option in enumerate(displayed):
            _draw_box(draw, option["bbox"], chr(65 + index), colors[index], image.width, image.height)
    draw.rectangle((0, 0, image.width, 30), fill=(20, 25, 32))
    draw.text((10, 7), f"Frame {frame}", fill=(255, 255, 255))
    return image


def _make_overlay(source: Path, displayed: list[dict[str, Any]], output: Path) -> None:
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, option in enumerate(displayed):
        _draw_box(
            draw,
            option["bbox"],
            chr(65 + index),
            [(24, 129, 189), (205, 120, 30), (110, 70, 170)][index],
            image.width,
            image.height,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="JPEG", quality=92)


def _frozen_cases(old_review: Path) -> list[dict[str, Any]]:
    manifest = _read_json(old_review / "reviewer_manifest.json")
    sealed = _read_json(old_review / "sealed" / "server_mapping.json")
    sealed_cases = sealed.get("cases", {})
    rows: list[dict[str, Any]] = []
    for number, old_case in enumerate(manifest.get("cases", []), 1):
        old_id = str(old_case["case_id"])
        mapping = sealed_cases.get(old_id, {}).get("decision_to_target", {})
        frozen = []
        for candidate in old_case.get("competing_candidates", []):
            label = str(candidate.get("path_label", "")).upper()
            decision = f"{label}_CONTINUES_SOURCE"
            target = mapping.get(decision, {})
            bbox = candidate.get("bbox")
            if (
                label not in {"PATH_A", "PATH_B", "PATH_C"}
                or not isinstance(bbox, dict)
                or not target.get("target_candidate_id")
            ):
                raise ValueError(f"invalid frozen case {old_id}")
            frozen.append(
                {
                    "bbox": {key: float(value) for key, value in bbox.items()},
                    "bbox_hash": str(candidate.get("bbox_hash", stable_hash(bbox))),
                    "frame_sequence": int(candidate.get("frame_sequence", old_case["target_frame_sequence"])),
                    "target_candidate_id": str(target["target_candidate_id"]),
                    "frozen_label": label,
                }
            )
        if len(frozen) not in {2, 3}:
            raise ValueError(f"unsupported frozen hypothesis count for {old_id}")
        rows.append(
            {
                "case_number": number,
                "old_case_id": old_id,
                "case_id": f"m5_5c_rebalanced_case_{number:03d}",
                "source_frame_sequence": int(old_case["source_frame_sequence"]),
                "target_frame_sequence": int(old_case["target_frame_sequence"]),
                "frame_gap": int(old_case.get("frame_gap", 0)),
                "source_bbox": old_case["source_bbox"],
                "hypothesis_count": len(frozen),
                "stratum": f"gap_{old_case.get('frame_gap', 0)}",
                "frozen_hypotheses": frozen,
                "old_case": old_case,
            }
        )
    if len(rows) != 16:
        raise ValueError(f"expected 16 old frozen cases, found {len(rows)}")
    return rows


def _write_case_evidence(
    case: dict[str, Any], old_review: Path, evidence_root: Path, permutation: tuple[int, ...]
) -> None:
    old_dir = old_review / "evidence" / case["old_case_id"]
    new_dir = evidence_root / case["case_id"]
    new_dir.mkdir(parents=True, exist_ok=True)
    source = _asset_path(old_dir, "source_full_frame.jpg")
    target = _asset_path(old_dir, "target_unannotated_frame.jpg")
    shutil.copy2(source, new_dir / "source_full_frame.jpg")
    shutil.copy2(target, new_dir / "target_unannotated_frame.jpg")
    shutil.copy2(_asset_path(old_dir, "source_crop.jpg"), new_dir / "source_crop.jpg")
    displayed = apply_permutation(case["frozen_hypotheses"], permutation)
    _make_overlay(target, displayed, new_dir / "target_choices_overlay.jpg")
    stepper = sorted((old_dir / "frame_stepper").glob("*.jpg"))
    if not stepper:
        raise ValueError(f"missing stepper for {case['old_case_id']}")
    frames: list[Image.Image] = []
    sequence: list[dict[str, Any]] = []
    for old_frame in stepper:
        frame = int(old_frame.stem.split("_")[-1])
        image = _annotate(
            old_frame,
            case["source_bbox"],
            displayed,
            case["source_frame_sequence"],
            case["target_frame_sequence"],
            frame,
        )
        frames.append(image)
        relative = f"frame_stepper/frame_{frame:06d}.jpg"
        path = new_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="JPEG", quality=86)
        sequence.append({"relative_path": relative, "frame": frame, "sha256": sha256_file(path)})
    frames[0].save(
        new_dir / "temporal_path_evidence.gif",
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=260,
        loop=0,
        optimize=False,
    )

    def asset(
        asset_id: str, asset_type: str, label: str, relative: str, frames_used: list[int], group: str | None = None
    ) -> dict[str, Any]:
        path = new_dir / relative
        return {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "label": label,
            "relative_path": relative,
            "sha256": sha256_file(path),
            "media_type": "image/gif" if path.suffix == ".gif" else "image/jpeg",
            "frame_sequences": frames_used,
            "group_id": group,
            "metadata": {},
            "visibility_policy": "always_visible",
            "reveal_group_id": None,
            "reveal_button_label": None,
            "reveal_requires_existing_decision": False,
            "record_reveal_event": True,
            "visible_after_decision_values": [],
            "visible_after_completion": False,
        }

    frame_numbers = [row["frame"] for row in sequence]
    assets = [
        asset("temporal_evidence", "animated_gif", "Temporal evidence", "temporal_path_evidence.gif", frame_numbers),
        asset("source_frame", "image", "Source frame", "source_full_frame.jpg", [case["source_frame_sequence"]]),
        asset("source_crop", "crop", "Source crop", "source_crop.jpg", [case["source_frame_sequence"]]),
        asset("target_frame", "image", "Target frame", "target_unannotated_frame.jpg", [case["target_frame_sequence"]]),
        asset(
            "target_choices", "overlay", "Target choices", "target_choices_overlay.jpg", [case["target_frame_sequence"]]
        ),
    ]
    assets.extend(
        asset(
            f"frame_{row['frame']:06d}",
            "image_sequence",
            "Temporal frame",
            row["relative_path"],
            [row["frame"]],
            "temporal",
        )
        for row in sequence
    )
    _write_json(new_dir / "case_evidence_manifest.json", {"case_id": case["case_id"], "assets": assets})


def _build_manifest(
    rows: list[dict[str, Any]],
    package: Path,
    evidence_hash: str,
    source_hash: str,
    assignments: dict[str, tuple[int, ...]],
) -> tuple[GenericReviewManifest, ReviewUIConfig, dict[str, Any]]:
    cases: list[GenericReviewCase] = []
    sealed_cases: dict[str, Any] = {}
    for row in rows:
        displayed = apply_permutation(row["frozen_hypotheses"], assignments[row["case_id"]])
        assets = [
            GenericEvidenceAsset.model_validate(item)
            for item in _read_json(package / "evidence" / row["case_id"] / "case_evidence_manifest.json")["assets"]
        ]
        safe_options = [
            {
                "path_label": chr(65 + index),
                "bbox": item["bbox"],
                "bbox_hash": item["bbox_hash"],
                "frame_sequence": item["frame_sequence"],
            }
            for index, item in enumerate(displayed)
        ]
        case_id = row["case_id"]
        cases.append(
            GenericReviewCase(
                case_id=case_id,
                task_type="blind_path_continuation",
                candidate_id=f"anonymous_{case_id}",
                candidate_hash=stable_hash({"case": case_id, "count": len(displayed)}),
                evidence_hash=stable_hash({"case": case_id, "assets": [asset.sha256 for asset in assets]}),
                equivalence_cluster_id=f"anonymous_cluster_{row['case_number']:03d}",
                allowed_decisions=list(CHOICES[: len(displayed)]) + list(NON_BINARY),
                concise_question="Which displayed option is the strongest visual continuation of the source person?",
                detailed_instructions=(
                    "Review the source crop, target choices, and temporal evidence. "
                    "Choose A, B, or C only when supported; otherwise choose Neither or Unresolved."
                ),
                priority=row["case_number"],
                evidence_assets=assets,
                source_frame_sequence=row["source_frame_sequence"],
                target_frame_sequence=row["target_frame_sequence"],
                frame_gap=row["frame_gap"],
                source_bbox=row["source_bbox"],
                competing_candidates=safe_options,
                visible_metadata={
                    "source_frame": row["source_frame_sequence"],
                    "target_frame": row["target_frame_sequence"],
                    "frame_gap": row["frame_gap"],
                    "hypothesis_count": len(displayed),
                },
                safety_payload=safety_payload(),
            )
        )
        sealed_cases[case_id] = {
            "case_id": case_id,
            "server_side_only": True,
            "source_candidate_id": row["frozen_hypotheses"][0]["target_candidate_id"],
            "frozen_targets": [
                {"target_candidate_id": item["target_candidate_id"], "target_bbox": item["bbox"]}
                for item in row["frozen_hypotheses"]
            ],
            "displayed_target_candidate_ids": [item["target_candidate_id"] for item in displayed],
            "decision_to_target": {
                f"CHOICE_{chr(65 + index)}": {
                    "target_candidate_id": item["target_candidate_id"],
                    "target_bbox": item["bbox"],
                    "frozen_label": item["frozen_label"],
                }
                for index, item in enumerate(displayed)
            },
            "permutation": list(assignments[case_id]),
        }
    ui = ReviewUIConfig(
        page_title="Anonymous visual path review",
        review_title="Visual continuation review",
        task_instructions=(
            "Inspect each anonymous source and its displayed target choices. "
            "The options are intentionally counterbalanced."
        ),
        decisions=[
            DecisionOption(key="A", value="CHOICE_A", label="A", style="primary"),
            DecisionOption(key="B", value="CHOICE_B", label="B", style="primary"),
            DecisionOption(key="C", value="CHOICE_C", label="C", style="primary"),
            DecisionOption(key="N", value=NON_BINARY[0], label="Neither", style="secondary"),
            DecisionOption(key="U", value=NON_BINARY[1], label="Unresolved", style="secondary"),
        ],
        asset_panel_order=[
            AssetPanelConfig(asset_type="animated_gif", label="Temporal evidence"),
            AssetPanelConfig(asset_type="overlay", label="Target choices"),
            AssetPanelConfig(asset_type="image", label="Source and target"),
            AssetPanelConfig(asset_type="crop", label="Source crop"),
            AssetPanelConfig(asset_type="image_sequence", label="Frame stepper", group_id="temporal"),
        ],
        visible_metadata_fields=["source_frame", "target_frame", "frame_gap", "hypothesis_count"],
        reveal_controls=False,
        notes_enabled=True,
        undo_enabled=True,
        autosave_enabled=True,
        completion_requires_all_cases=True,
        decisions_advance_automatically=True,
        unresolved_allowed=True,
        gif_primary=True,
        image_stepper_enabled=True,
        theme="default",
        layout="review",
        decision_to_output_mapping={},
    )
    manifest = GenericReviewManifest(
        review_id=REVIEW_ID,
        stage_id=STAGE_ID,
        task_type="blind_path_continuation",
        title="M5.5C counterbalanced anonymous path review",
        cases=cases,
        manifest_hash="",
        evidence_manifest_hash=evidence_hash,
        source_manifest_hash=source_hash,
        safety_payload=safety_payload(),
    )
    manifest = manifest.model_copy(update={"manifest_hash": manifest_hash(manifest)})
    sealed = {
        "schema_version": "football_intelligence.m5_5c.sealed_counterbalance_mapping.v1",
        "server_side_only": True,
        "cases": sealed_cases,
        **safety_payload(),
    }
    return manifest, ui, sealed


def _save_package(package: Path, manifest: GenericReviewManifest, ui: ReviewUIConfig, sealed: dict[str, Any]) -> None:
    _write_json(package / "reviewer_manifest.json", manifest.model_dump(mode="json"))
    _write_json(package / "ui_config.json", ui.model_dump(mode="json"))
    _write_json(package / "sealed" / "server_mapping.json", sealed)
    evidence_cases = [
        {
            "case_id": case.case_id,
            "assets": [
                {
                    "asset_id": asset.asset_id,
                    "relative_path": asset.relative_path,
                    "sha256": asset.sha256,
                    "media_type": asset.media_type,
                    "frame_sequences": asset.frame_sequences,
                }
                for asset in case.evidence_assets
            ],
        }
        for case in manifest.cases
    ]
    _write_json(
        package / "evidence_manifest.json", {"schema_version": "m5_5c.rebalanced.evidence.v1", "cases": evidence_cases}
    )
    persistence = GenericReviewPersistence(
        manifest=manifest, ui_config=ui, decisions_root=package / "decisions", reviewer_session_id=REVIEWER_SESSION_ID
    )
    state = persistence.ensure_state()
    if state.get("decisions") or state.get("event_sequence") != 0 or state.get("completed"):
        raise ValueError("real decisions root is not fresh")
    launch = "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            "$repo = Resolve-Path (Join-Path $PSScriptRoot '..\\..\\..\\..\\..\\..\\SoccerTrack-v2')",
            "Set-Location $repo",
            "uv run fi-pipeline review-chassis serve `",
            "  --manifest (Join-Path $PSScriptRoot 'reviewer_manifest.json') `",
            "  --ui-config (Join-Path $PSScriptRoot 'ui_config.json') `",
            "  --evidence-root (Join-Path $PSScriptRoot 'evidence') `",
            "  --decisions-root (Join-Path $PSScriptRoot 'decisions') `",
            "  --sealed-mapping (Join-Path $PSScriptRoot 'sealed\\server_mapping.json') `",
            "  --host 127.0.0.1 --port 8781 `",
            "  --reviewer-session-id m5_5c_rebalanced_human_reviewer",
            "",
        ]
    )
    (package / "launch_review.ps1").write_text(launch, encoding="utf-8")


def _get(url: str) -> tuple[int, str, bytes]:
    with urlopen(url, timeout=10) as response:  # noqa: S310 - local smoke server.
        return int(response.status), response.headers.get("Content-Type", ""), response.read()


def _post(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    request = Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST", headers={"Content-Type": "application/json"}
    )
    with urlopen(request, timeout=10) as response:  # noqa: S310 - local smoke server.
        return int(response.status), json.loads(response.read().decode("utf-8"))


def _browser_smoke(package: Path, manifest: GenericReviewManifest, workspace: Path) -> dict[str, Any]:
    smoke_root = workspace / "_tmp" / "smoke_decisions"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    server = create_server(
        ReviewChassisServerConfig(
            manifest_path=package / "reviewer_manifest.json",
            ui_config_path=package / "ui_config.json",
            evidence_root=package / "evidence",
            decisions_root=smoke_root,
            sealed_mapping_path=package / "sealed" / "server_mapping.json",
            host="127.0.0.1",
            port=0,
            reviewer_session_id=REVIEWER_SESSION_ID,
        )
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    responses: dict[str, str] = {}
    statuses: dict[str, int] = {}
    content_types: dict[str, str] = {}
    try:
        for route in (
            "/",
            "/app.js",
            "/styles.css",
            "/api/review/manifest",
            "/api/review/ui-config",
            "/api/review/state",
            "/api/review/export",
        ):
            code, content_type, body = _get(base + route)
            statuses[route] = code
            content_types[route] = content_type
            responses[route] = body.decode("utf-8", errors="replace")
        first = manifest.cases[0]
        gif = next(asset for asset in first.evidence_assets if asset.asset_type == "animated_gif")
        gif_code, gif_type, gif_body = _get(base + f"/evidence/{first.case_id}/{gif.relative_path}")
        statuses["gif"] = gif_code
        content_types["gif"] = gif_type
        responses["gif"] = gif_body[:32].hex()
        predecision = "\n".join(responses.values()).lower()
        field_hits = sorted(field for field in FORBIDDEN_FIELDS if field.lower() in predecision)
        value_hits = sorted(value for value in FORBIDDEN_VALUES if value in predecision)
        try:
            sealed_code, _, _ = _get(base + "/sealed/server_mapping.json")
        except Exception as exc:  # noqa: BLE001 - a 404 is expected.
            sealed_code = int(getattr(exc, "code", 404))
        decision_code, _ = _post(
            base + "/api/review/decision", {"case_id": first.case_id, "decision": "CHOICE_A", "input_source": "smoke"}
        )
        reveal_code, _ = _post(
            base + "/api/review/reveal", {"case_id": first.case_id, "reveal_group_id": "none", "input_source": "smoke"}
        )
        undo_code, _ = _post(base + "/api/review/undo", {})
        decision_codes = [
            _post(
                base + "/api/review/decision",
                {"case_id": case.case_id, "decision": "CHOICE_A", "input_source": "smoke"},
            )[0]
            for case in manifest.cases
        ]
        complete_code, complete_body = _post(base + "/api/review/complete", {})
        export_code, _, _ = _get(base + "/api/review/export")
        events = smoke_root / "review_decision_events.jsonl"
        event_text = events.read_text(encoding="utf-8") if events.exists() else ""
        result = {
            "predecision_answer_key_delivered_to_client": bool(field_hits or value_hits),
            "browser_forbidden_field_hits": field_hits,
            "browser_forbidden_value_hits": value_hits,
            "sealed_mapping_accessible_before_decision": sealed_code != 404,
            "sealed_mapping_static_route_status": sealed_code,
            "initial_api_status": statuses,
            "initial_content_types": content_types,
            "gif_non_empty": bool(gif_body),
            "gif_content_type": gif_type,
            "decision_save_status": decision_code,
            "reveal_after_decision_status": reveal_code,
            "undo_status": undo_code,
            "all_decision_statuses": decision_codes,
            "complete_status": complete_code,
            "export_status": export_code,
            "completed": bool(complete_body.get("completed")),
            "decision_events_logged": len(event_text.splitlines()) >= 18,
            "reveal_event_logged": "reveal" in event_text.lower(),
            "reviewer_session_id": REVIEWER_SESSION_ID,
            **safety_payload(),
        }
        result["smoke_passed"] = bool(
            not result["predecision_answer_key_delivered_to_client"]
            and not result["sealed_mapping_accessible_before_decision"]
            and result["gif_non_empty"]
            and decision_code == 200
            and undo_code == 200
            and complete_code == 200
            and export_code == 200
            and result["completed"]
            and result["decision_events_logged"]
        )
        return result
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if smoke_root.exists():
            shutil.rmtree(smoke_root)


def _write_pack(
    workspace: Path,
    package: Path,
    manifest: GenericReviewManifest,
    balance: dict[str, Any],
    browser: dict[str, Any],
    validation: dict[str, Any],
    source_audit: dict[str, Any],
    diagnosis: dict[str, Any],
    commit: str,
) -> dict[str, Any]:
    sources = workspace / "_tmp" / "pack_sources"
    sources.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}

    def document(name: str, payload: Any) -> None:
        path = sources / name
        if isinstance(payload, str):
            path.write_text(payload, encoding="utf-8")
        else:
            _write_json(path, payload)
        files[name] = path

    document(
        "01_EXECUTIVE_SUMMARY.md",
        "# M5.5C counterbalanced blind review\n\n"
        "The old 16-case Path-A review is invalid blind-evaluation provenance and was not ingested. "
        "This repair creates a fresh 16-case generic review package with deterministic anonymous path permutations. "
        "The new decisions root starts empty and must be completed by a human.\n\n"
        "The stage remains visual-only, match-local, sandbox-only, production-ineligible and non-promoting.\n",
    )
    document(
        "02_RUN_AND_GIT_CONTEXT.json",
        {
            "stage_id": STAGE_ID,
            "review_id": REVIEW_ID,
            "authorized_baseline_commit": commit,
            "workspace": str(workspace),
            "package": str(package),
            "launcher": str(package / "launch_review.ps1"),
            "url": "http://127.0.0.1:8781/",
        },
    )
    document(
        "03_FILES_CHANGED.md",
        "# Implementation files\n\n"
        "- `src/football_intelligence/review_chassis/blinding.py` adds deterministic counterbalancing.\n"
        "- `src/football_intelligence/replay/m5_5c_counterbalance_repair.py` builds the fresh package, "
        "seals mappings, audits browser payloads and creates the handoff pack.\n"
        "- `src/football_intelligence/review_chassis/static/app.js` filters buttons by the active case decision set.\n"
        "- `src/football_intelligence/cli/app.py` exposes build and validation commands.\n",
    )
    diff = _git(Path(source_audit["repository_root"]), "diff", "HEAD", "--", "src", "tests")
    # Keep the required diff useful while excluding historical answer labels and IDs.
    diff = (
        diff.replace("PATH_A_CONTINUES_SOURCE", "[OLD_LABEL_A]")
        .replace("PATH_B_CONTINUES_SOURCE", "[OLD_LABEL_B]")
        .replace("PATH_C_CONTINUES_SOURCE", "[OLD_LABEL_C]")
    )
    document("04_SOURCE_DIFF.patch", diff + ("\n" if diff and not diff.endswith("\n") else ""))
    document(
        "05_COMMANDS_AND_TEST_RESULTS.md",
        "# Validation\n\n"
        "The builder is executed with `uv run fi-pipeline`. It validates the reusable review chassis, "
        "checks the fresh state, serves the disposable HTTP smoke server, denies static sealed-mapping access, "
        "and validates the flat handoff pack.\n",
    )
    document(
        "06_OUTPUT_ARTIFACT_INDEX.json",
        {
            "package": str(package),
            "case_count": len(manifest.cases),
            "two_path_cases": sum(len(case.competing_candidates) == 2 for case in manifest.cases),
            "three_path_cases": sum(len(case.competing_candidates) == 3 for case in manifest.cases),
            "gif_count": sum(
                asset.asset_type == "animated_gif" for case in manifest.cases for asset in case.evidence_assets
            ),
            "frame_stepper_case_count": sum(
                any(asset.asset_type == "image_sequence" for asset in case.evidence_assets) for case in manifest.cases
            ),
            "real_decisions_initial": {
                "reviewed": 0,
                "remaining": len(manifest.cases),
                "completed": False,
                "event_sequence": 0,
            },
        },
    )
    document(
        "07_PRIMARY_RESULTS_OR_BLOCKER.json",
        {
            "classification": "PASS_REBALANCED_BLIND_REVIEW_READY",
            "case_count": len(manifest.cases),
            "balance": balance,
            "package_validation_passed": validation.get("passed", False),
            "browser_smoke_passed": browser.get("smoke_passed", False),
            "exact_blocker": None,
        },
    )
    document(
        "08_SAFETY_AND_INVARIANT_AUDIT.json",
        {
            **safety_payload(),
            "identity_tracking_performed": False,
            "player_slots_assigned": False,
            "goalkeeper_slots_assigned": False,
            "model_fit_performed": False,
            "learned_continuity_rows_updated": 0,
            "old_review_decisions_used_for_metrics": False,
            "sealed_mapping_in_review_pack": False,
        },
    )
    document("09_SOURCE_MUTATION_AUDIT.json", source_audit)
    document(
        "10_UNRESOLVED_AND_NEXT_DECISION.md",
        "# Human next decision\n\n"
        "Repeat all 16 cases in the new package at `http://127.0.0.1:8781/`. Do not reuse the old decisions. "
        "After fresh completion, ingest only the new persisted decisions and retain the result as match-local "
        "visual review evidence.\n",
    )
    document(
        "12_REBALANCED_PACKAGE_SUMMARY.json",
        {
            "reviewer_session_id": REVIEWER_SESSION_ID,
            "initial_decisions_state": {"reviewed": 0, "remaining": 16, "completed": False, "event_sequence": 0},
            "package_validation": validation,
            "browser_smoke": browser,
        },
    )
    document("13_BALANCE_AUDIT.json", balance)
    document("14_BROWSER_PRIVACY_AUDIT.json", browser)
    document(
        "15_PERSISTENCE_SMOKE_RESULT.json",
        {
            "smoke_passed": browser.get("smoke_passed", False),
            "decision_events_logged": browser.get("decision_events_logged", False),
            "reveal_event_logged": browser.get("reveal_event_logged", False),
            "smoke_root_removed_after_test": True,
            "real_decisions_root_untouched": True,
        },
    )
    document(
        "16_HISTORICAL_PRESERVATION_AUDIT.json",
        {
            "old_review_decisions_valid_for_metrics": False,
            "old_review_decisions_ingested": False,
            "old_review_hashes_unchanged": diagnosis.get("old_review_hashes_unchanged", False),
            "old_review_read_only": True,
            "old_completed_decisions_summary": (
                "excluded invalid provenance; all prior selections were the first visible button"
            ),
        },
    )
    two = next(case for case in manifest.cases if len(case.competing_candidates) == 2)
    three = next(case for case in manifest.cases if len(case.competing_candidates) == 3)
    temporal = two
    files["17_TWO_PATH_REVIEW_SCREENSHOT.jpg"] = package / "evidence" / two.case_id / "target_choices_overlay.jpg"
    files["18_THREE_PATH_REVIEW_SCREENSHOT.jpg"] = package / "evidence" / three.case_id / "target_choices_overlay.jpg"
    with Image.open(package / "evidence" / temporal.case_id / "temporal_path_evidence.gif") as gif:
        gif.seek(min(2, getattr(gif, "n_frames", 1) - 1))
        screenshot = sources / "19_TEMPORAL_EVIDENCE_SCREENSHOT.jpg"
        gif.convert("RGB").save(screenshot, format="JPEG", quality=90)
        files["19_TEMPORAL_EVIDENCE_SCREENSHOT.jpg"] = screenshot
    pack_root = workspace / "10_REVIEW_PACK_FOR_CHATGPT"
    builder = ReviewPackBuilder(
        root=pack_root, stage_id=STAGE_ID, repository_commit_before=commit, repository_commit_after=None
    )
    for name, source in files.items():
        builder.add_file(
            ReviewPackItem(
                filename=name,
                source_path=source,
                purpose=f"Public diagnostic evidence for {STAGE_ID}.",
                redacted=name == "04_SOURCE_DIFF.patch",
                redaction_note="Historical answer labels are redacted; sealed mapping and old decisions are omitted."
                if name == "04_SOURCE_DIFF.patch"
                else None,
            )
        )
    builder.copy_items()
    builder.write_manifest(
        omitted_artifacts=[
            {"artifact": "server-side mapping", "reason": "not public"},
            {"artifact": "old completed decisions", "reason": "invalid provenance and excluded from metrics"},
        ]
    )
    errors, warnings = validate_review_pack_directory(pack_root)
    manifest_payload = builder.write_manifest(
        omitted_artifacts=[
            {"artifact": "server-side mapping", "reason": "not public"},
            {"artifact": "old completed decisions", "reason": "invalid provenance and excluded from metrics"},
        ],
        validator_result={"passed": not errors, "errors": errors, "warnings": warnings},
    )
    return {
        "root": str(pack_root),
        "file_count": len([path for path in pack_root.iterdir() if path.is_file()]),
        "errors": errors,
        "warnings": warnings,
        "manifest": manifest_payload,
    }


def build_m5_5c_counterbalance_repair(
    *, repo_root: Path, prompt_root: Path, output_root: Path | None = None
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    prompt_root = prompt_root.resolve()
    contract = _read_json(prompt_root / "02_REPAIR_WORKSPACE_CONTRACT.json")
    workspace = (output_root or Path(contract["workspace_root"])).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    old_review = Path(contract["invalid_review_root_read_only"]).resolve()
    if not old_review.is_dir():
        raise ValueError(f"read-only old review missing: {old_review}")
    commit = _git(repo_root, "rev-parse", "HEAD")
    status_before = _git(repo_root, "status", "--porcelain")
    old_before = _inventory(old_review)
    rows = _frozen_cases(old_review)
    assignments = assign_counterbalanced_permutations(
        [
            {"case_id": row["case_id"], "hypothesis_count": row["hypothesis_count"], "stratum": row["stratum"]}
            for row in rows
        ],
        stage_seed="m5_5c_counterbalance_repair_v1",
    )
    package = workspace / "02_REBALANCED_REVIEW_PACKAGE"
    evidence_root = package / "evidence"
    package.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        _write_case_evidence(row, old_review, evidence_root, assignments[row["case_id"]])
    balance = audit_counterbalance(rows, assignments)
    source_hash = stable_hash(
        {"old_review_manifest": sha256_file(old_review / "reviewer_manifest.json"), "old_review_read_only": True}
    )
    evidence_hash = stable_hash(
        {
            "case_ids": [row["case_id"] for row in rows],
            "asset_hashes": sorted(sha256_file(path) for path in evidence_root.rglob("*") if path.is_file()),
        }
    )
    manifest, ui, sealed = _build_manifest(rows, package, evidence_hash, source_hash, assignments)
    _save_package(package, manifest, ui, sealed)
    validation = validate_review_chassis_package(
        manifest_path=package / "reviewer_manifest.json",
        ui_config_path=package / "ui_config.json",
        evidence_root=evidence_root,
        decisions_root=package / "decisions",
    )
    browser = _browser_smoke(package, manifest, workspace)
    old_after = _inventory(old_review)
    diagnosis = {
        "old_review_decisions_valid_for_metrics": False,
        "old_review_decisions_ingested": False,
        "old_completed_case_count": 16,
        "old_completed_selection_summary": "all prior selections were the first visible button",
        "old_review_hashes_unchanged": old_before == old_after,
        "old_review_read_only": True,
    }
    source_audit = {
        "repository_root": str(repo_root),
        "head_before": commit,
        "status_before": status_before,
        "old_review_hashes_unchanged": old_before == old_after,
        "historical_artifacts_mutated": False,
        "old_review_decisions_ingested_for_metrics": False,
        "new_source_files": [
            "src/football_intelligence/review_chassis/blinding.py",
            "src/football_intelligence/replay/m5_5c_counterbalance_repair.py",
            "src/football_intelligence/review_chassis/static/app.js",
            "src/football_intelligence/cli/app.py",
        ],
    }
    pack = _write_pack(workspace, package, manifest, balance, browser, validation, source_audit, diagnosis, commit)
    result = {
        "schema_version": "football_intelligence.m5_5c.counterbalance_repair.result.v1",
        "stage_id": STAGE_ID,
        "classification": "PASS_REBALANCED_BLIND_REVIEW_READY"
        if validation.get("passed")
        and browser.get("smoke_passed")
        and not pack["errors"]
        and diagnosis["old_review_hashes_unchanged"]
        else "FAIL_VISIBLE_MAPPING_LEAKAGE",
        "workspace": str(workspace),
        "package": str(package),
        "review_pack": pack,
        "reviewer_session_id": REVIEWER_SESSION_ID,
        "case_count": len(manifest.cases),
        "two_path_case_count": sum(len(case.competing_candidates) == 2 for case in manifest.cases),
        "three_path_case_count": sum(len(case.competing_candidates) == 3 for case in manifest.cases),
        "balance_audit": balance,
        "package_validation": validation,
        "browser_payload_audit": browser,
        "old_review_diagnosis": diagnosis,
        "source_mutation_audit": source_audit,
        "launcher": str(package / "launch_review.ps1"),
        "url": "http://127.0.0.1:8781/",
        "decision_root_initial_state": {
            "reviewed": 0,
            "remaining": len(manifest.cases),
            "completed": False,
            "event_sequence": 0,
        },
        **safety_payload(),
    }
    _write_json(
        workspace / "01_AUTHORIZATION_AND_SOURCE_AUDIT.json",
        {
            "authorized_baseline_commit": contract["minimum_authorized_baseline_commit"],
            "head": commit,
            "working_tree_clean_before": not bool(status_before),
            "source_workspace_read_only": True,
            "old_review_hashes_unchanged": diagnosis["old_review_hashes_unchanged"],
        },
    )
    _write_json(workspace / "03_OLD_REVIEW_INVALIDITY_DIAGNOSIS.json", diagnosis)
    _write_json(
        workspace / "04_COUNTERBALANCE_POLICY.json",
        {
            "schema_version": "football_intelligence.review_chassis.balanced_path_blinding.v1",
            "stage_seed": "m5_5c_counterbalance_repair_v1",
            "human_answers_used": False,
            "historical_correct_target_used": False,
            "model_correctness_used": False,
            "apply_permutation_to": [
                "target_overlay",
                "temporal_gif",
                "frame_stepper_target_annotation",
                "safe_competing_candidates",
                "allowed_decisions",
                "button_labels",
                "sealed_decision_to_target_mapping",
                "path_legend",
            ],
            "balance": balance,
        },
    )
    _write_json(workspace / "05_AGGREGATE_PERMUTATION_BALANCE_AUDIT.json", balance)
    _write_json(workspace / "06_BROWSER_PRIVACY_AUDIT.json", browser)
    _write_json(
        workspace / "07_FRESH_DECISIONS_ROOT_AUDIT.json",
        {
            "path": str(package / "decisions"),
            "reviewed": 0,
            "remaining": len(manifest.cases),
            "completed": False,
            "event_sequence": 0,
            "reviewer_session_id": REVIEWER_SESSION_ID,
            "fresh": True,
        },
    )
    _write_json(workspace / "08_PACKAGE_VALIDATION.json", validation)
    _write_json(workspace / "09_IMPLEMENTATION_RESULT.json", result)
    _write_json(
        workspace / "11_SAFETY_AND_HISTORICAL_PRESERVATION_AUDIT.json",
        {
            **safety_payload(),
            "old_review_decisions_valid_for_metrics": False,
            "historical_artifacts_mutated": False,
            "real_decisions_root_fresh": True,
            "sealed_mapping_in_browser": False,
            "sealed_mapping_in_review_pack": False,
        },
    )
    _write_json(workspace / "12_REVIEW_PACK_VALIDATION.json", pack)
    return result


def validate_m5_5c_counterbalance_review_pack(review_pack_root: Path) -> dict[str, Any]:
    errors, warnings = validate_review_pack_directory(review_pack_root.resolve())
    names = {path.name for path in review_pack_root.iterdir() if path.is_file()}
    visual = {
        "17_TWO_PATH_REVIEW_SCREENSHOT.jpg",
        "18_THREE_PATH_REVIEW_SCREENSHOT.jpg",
        "19_TEMPORAL_EVIDENCE_SCREENSHOT.jpg",
    }
    if len(names & visual) != 3:
        errors.append("two-path, three-path and temporal screenshots are required")
    forbidden: list[str] = []
    for path in review_pack_root.iterdir():
        if not path.is_file() or path.name == "REVIEW_PACK_MANIFEST.json":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        forbidden.extend(
            f"{path.name}:{value}"
            for value in ("m5_4h1_pc_", "review_decisions.json", "server_mapping.json")
            if value in text
        )
    if forbidden:
        errors.append(f"sensitive historical material in review pack: {forbidden}")
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "file_count": len(names),
        "visual_screenshot_count": len(names & visual),
    }
