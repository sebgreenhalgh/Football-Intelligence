"""Read-only, append-only interim dense-gold stopping audit."""

from __future__ import annotations

import hashlib
import json
import argparse
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PART9 = REPO.parent / "experiments/football_observation_reasoner/part 9"
SOURCE = PART9 / "G7F_C_DENSE_PERSON_GOLD_DISCRIMINATION_SET_v1"
WORKSPACE = PART9 / "G7F_D_INTERIM_DENSE_GOLD_BAKEOFF_AND_STOPPING_AUDIT_v1"
DECISIONS = SOURCE / "06_DENSE_DECISIONS"
PYTHON = Path(r"C:\Users\sebgr\anaconda3\envs\fi-reviewer\python.exe")
RELEASE = PART9 / "G7F_C_SCORED_DENSE_GOLD_FIRST_PASS_REVIEWER_RELEASE_v1"
BAKEOFF = PART9 / "G7F_B_R1_DETECTION_CANDIDATE_BAKEOFF_v1"
EXPECTED_COMMIT = "96b64c83f04350ea0507aeb77636120c7a1b2a14"
INITIAL_FREEZE = WORKSPACE / "00_FREEZE/real_decisions_before_20260928T022552882207Z.json"
sys.path.insert(0, str(REPO / "src"))

from football_intelligence.dense_person_gold import (  # noqa: E402
    build_final_event,
    canonical_json_bytes,
    evaluate_dense_boxes,
)
from football_intelligence.interim_dense_gold_audit import (  # noqa: E402
    LABELS,
    SAFETY,
    FrameMetricCache,
    next_tranche,
    paired_analysis,
    prepare_frame,
    stopping_decision,
)
from g7f_c_run_scored_first_pass_reviewer import FROZEN_HASHES, REVIEWER_RELEASE, scored_queue  # noqa: E402

import numpy as np  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def inventory(root: Path) -> dict:
    rows = [
        {"relative_path": p.relative_to(root).as_posix(), "byte_size": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]
    ordered = "\n".join(f"{r['relative_path']}\t{r['byte_size']}\t{r['sha256']}" for r in rows).encode()
    return {
        "root": str(root),
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(r["byte_size"] for r in rows),
        "ordered_inventory_sha256": hashlib.sha256(ordered).hexdigest(),
    }


def freeze() -> Path:
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError("Only the fi-reviewer interpreter is authorized")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = WORKSPACE / "00_FREEZE" / f"real_decisions_before_{stamp}.json"
    value = inventory(DECISIONS)
    write_new(path, value)
    print(
        json.dumps(
            {
                "freeze_path": str(path),
                "file_count": value["file_count"],
                "inventory_sha256": value["ordered_inventory_sha256"],
            }
        )
    )
    return path


def require(condition, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def git(*args) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def freeze_or_check(path: Path, value) -> None:
    if path.exists():
        require(read(path) == value, f"Immutable stage configuration mismatch: {path.name}")
    else:
        write_new(path, value)


def preregister() -> dict:
    protocol = {
        "schema_version": "g7f_d.stopping_protocol.v1",
        "bootstrap_seed": 20260928,
        "bootstrap_replicates": 10000,
        "resampling_unit": "SCORED_FRAME",
        "paired_sampling": True,
        "top_candidate": "HIGHEST_AGGREGATE_AP_50_95",
        "AP_gap_minimum": 0.020,
        "bootstrap_probability_minimum": 0.90,
        "jackknife_required_wins": "ceil(0.75*N)",
        "recall_delta_minimum": -0.020,
        "duplicate_rate_increase_failure": 0.05,
        "merge_rate_increase_failure": 0.05,
        "FP_pathology": "FP_per_frame > runner AND FP_per_frame >= 1.5*runner AND AP_gap < 0.04",
        "rate_denominators": "all candidate rows including ignore-suppressed rows; counts at IoU .50",
        "FP_zero_baseline": "zero vs zero is not an increase; positive vs zero fails when AP_gap < .04",
        "maximum_single_frame_advantage_fraction": 0.50,
        "frame_advantage_definition": "max(0,full_AP_gap - leave_one_frame_out_AP_gap)/full_AP_gap",
        "nonpositive_full_gap_robustness": "FAIL",
        "tie_break": "Candidate A then B then C for deterministic ordering; bootstrap win uses strict >",
        "matching": "EXACT_FROZEN_EVALUATOR_SEMANTICS; per-threshold six-decimal rounding preserved",
        "uncertain_relevance": "excluded from GT, does not create additional ignore regions",
        "bootstrap_implementation": "cached per-frame matching with multiplicity weights; no people-level resampling",
        "next_tranche": (
            "untouched scored IDs only; minimum completed+selected match count, "
            "then context count, then original queue position"
        ),
        "metadata_allowlist": [
            "anonymous_image_id",
            "source_frame_sha256",
            "match_group",
            "context_group",
            "queue_position",
        ],
        "context": "source width/height and frozen half; no inferred camera categories",
        "count_discrepancy": "current structurally valid scored corpus controls N; expected report was 8",
        "unblinding": "only after frozen STOP decision; CONTINUE never publishes mapping",
        "sampling_limits": (
            "small nonrandom enriched sample; repeated interim looks are not a sequentially calibrated test"
        ),
        **SAFETY,
    }
    freeze_or_check(WORKSPACE / "01_PROTOCOL/PRE_REGISTERED_STOPPING_RULE.json", protocol)
    return protocol


def discover(before: dict):
    require(inventory(DECISIONS) == before, "Human decisions changed since inventory freeze")
    for relative, expected in FROZEN_HASHES.items():
        require(sha(SOURCE / relative) == expected, f"Frozen contract mismatch: {relative}")
    release_manifest_path = RELEASE / "05_REVIEWER/reviewer_release_manifest.json"
    release_manifest = read(release_manifest_path)
    require(release_manifest["repository_commit"] == EXPECTED_COMMIT, "Reviewer release commit mismatch")
    require(
        release_manifest["candidate_reveal_enabled"] is False and release_manifest["blind_repeat_authorized"] is False,
        "Reviewer safety state mismatch",
    )
    for row in release_manifest["files"]:
        require(sha(Path(row["path"])) == row["sha256"], "Accepted reviewer release bytes changed")
    bindings = read(RELEASE / "05_REVIEWER/reviewer_binding_hashes.json")
    require(bindings["reviewer_release_manifest_sha256"] == sha(release_manifest_path), "Release binding mismatch")
    selection = read(SOURCE / "01_SELECTION/dense_gold_selection_manifest.json")
    queue = scored_queue(selection)
    require(
        {r["anonymous_dense_image_id"] for r in queue} == {f"DG-{i:03d}" for i in range(7, 55)}, "Scored IDs differ"
    )
    match_map = {m: f"Match-{i + 1:02d}" for i, m in enumerate(sorted({r["match_id"] for r in queue}))}
    require(len(match_map) == 6, "Scored queue must cover six matches")
    metadata, frames, corpus, drafts, touched = [], [], [], [], set()
    for row in queue:
        image_id = row["anonymous_dense_image_id"]
        halves = sorted({lineage["half"] for lineage in row["all_frame_instance_lineage"]})
        context = f"{row['source_width']}x{row['source_height']}/{'/'.join(halves)}"
        metadata.append(
            {
                "anonymous_image_id": image_id,
                "source_frame_sha256": row["source_frame_sha256"],
                "match_group": match_map[row["match_id"]],
                "context_group": context,
                "queue_position": row["review_queue_position"],
            }
        )
        ep = DECISIONS / "events" / f"first_pass__{image_id}.json"
        ap = DECISIONS / "acknowledgements" / f"first_pass__{image_id}.json"
        dp = DECISIONS / "drafts" / f"first_pass__{image_id}.json"
        if dp.exists():
            touched.add(image_id)
            if not ep.exists():
                drafts.append({"anonymous_image_id": image_id, "draft_sha256": sha(dp)})
        require(ep.exists() == ap.exists(), f"Event/ack orphan for {image_id}")
        if not ep.exists():
            continue
        event, ack = read(ep), read(ap)
        require(event["reviewer_release"] == REVIEWER_RELEASE, f"Reviewer mismatch: {image_id}")
        require(event["binding_hashes"] == bindings, f"Event bindings mismatch: {image_id}")
        require(type(event["final_revision"]) is int and event["final_revision"] > 0, "Invalid final revision")
        annotation = event["annotation"]
        # Rebuild in memory only, using the frozen source geometry and the event's exact human polygons.
        document = {
            "people": [
                {
                    "instance_id": p["instance_id"],
                    "relevance": p["relevance"],
                    "visible_mask_components": p["canonical_components"],
                }
                for p in annotation["people"]
            ],
            "ignore_regions": [
                {
                    "ignore_region_id": p["ignore_region_id"],
                    "reason": p["reason"],
                    "polygon": p["canonical_components"][0],
                }
                for p in annotation["ignore_regions"]
            ],
            "reviewed_exhaustiveness_strips": annotation["reviewed_exhaustiveness_strips"],
            "completion_assertion": annotation["completion_assertion"],
        }
        expected_event, expected_ack = build_final_event(
            row,
            document,
            binding_hashes=bindings,
            reviewer_release=REVIEWER_RELEASE,
            pass_kind="FIRST_PASS",
            final_revision=event["final_revision"],
        )
        require(expected_event == event and expected_ack == ack, f"Event/ack/geometry validation failed: {image_id}")
        frames.append(
            {
                "anonymous_image_id": image_id,
                "source_frame_sha256": event["source_frame_sha256"],
                "source_width": event["source_width"],
                "source_height": event["source_height"],
                "selection_status": event["selection_status"],
                "people": annotation["people"],
                "ignore_regions": annotation["ignore_regions"],
            }
        )
        corpus.append(
            {
                **metadata[-1],
                "event_file_sha256": sha(ep),
                "acknowledgement_file_sha256": sha(ap),
                "event_payload_sha256": event["event_sha256"],
                "person_counts": dict(Counter(p["relevance"] for p in annotation["people"])),
                "ignore_regions": len(annotation["ignore_regions"]),
                "valid_finalized_scored_FIRST_PASS": True,
            }
        )
        print(f"Validated scored truth chain {image_id}", flush=True)
    # A receipt means touched even if no draft remains. Do not use its document as truth.
    for receipt_path in (DECISIONS / "action_receipts").glob("*.json"):
        receipt = read(receipt_path)
        response = receipt.get("response", {})
        image_id = response.get("anonymous_dense_image_id")
        if image_id:
            touched.add(image_id)
    n = len(frames)
    require(n >= 2, "Insufficient finalized scored frames for paired stopping audit")
    public = {
        "frozen_scored_queue_count": 48,
        "finalized_scored_count": n,
        "operator_reported_count": 8,
        "count_discrepancy_explanation": (
            "The current snapshot contains additional valid acknowledged scored events beyond the operator's report; "
            "all pass the same release and truth chain."
            if n > 8
            else "Current structurally valid snapshot is authoritative; the operator count is not used as truth."
            if n != 8
            else None
        ),
        "calibration_used_in_primary_statistics": False,
        "unfinished_scored_drafts": drafts,
        "corpus": corpus,
        "match_coverage": dict(Counter(r["match_group"] for r in corpus)),
        **SAFETY,
    }
    return frames, public, metadata, touched


def candidate_inputs(frames: list[dict], run: Path):
    from football_intelligence.gold_eval.evaluation import validate_candidate_run

    registry_root = PART9 / "G7F_A_R1_FULL_SOURCE_FRAME_REGISTRY_AND_ADAPTER_COVERAGE_REPAIR_v1"
    for filename, expected in {
        "gold_frame_instances.jsonl": "2b8831b79ba04248c6fd05f488f6321e78b6f24b1339fd295b67c6802ad13bb9",
        "gold_source_frame_registry.jsonl": "dce19812fd6f09059b7dded19e14f01005f6c030fb117870712d01b9dd539154",
    }.items():
        require(sha(registry_root / "02_NORMALIZED_GOLD" / filename) == expected, "Frame registry binding failure")
    provenance_path = BAKEOFF / "10_REVIEW_PACK/CHATGPT_HANDOFF/03_RUN_PROVENANCE_AND_V2_COVERAGE.json"
    require(
        sha(provenance_path) == "3ad71f7be4fd05da2775a7792417796e56418bbb7c3f3696782e39afc1a125d5",
        "Frozen candidate provenance changed",
    )
    provenance = read(provenance_path)
    shortlist = read(BAKEOFF / "10_REVIEW_PACK/CHATGPT_HANDOFF/09_PROVISIONAL_SHORTLIST_AND_DENSE_GOLD_PLAN.json")
    require(
        shortlist["provisional_leader"] is None
        and {r["role"] for r in shortlist["pareto_shortlist"]} == set(provenance),
        "Frozen shortlist changed",
    )
    salt = hashlib.sha256(
        (
            WORKSPACE.name + ":candidate-blinding:v1:" + FROZEN_HASHES["03_METRICS/dense_gold_metric_protocol.json"]
        ).encode()
    ).hexdigest()
    roles = sorted(provenance, key=lambda role: hashlib.sha256((salt + role).encode()).hexdigest())
    mapping = {label: role for label, role in zip(LABELS, roles, strict=True)}
    sealed = {
        "stage_salt_sha256": salt,
        "mapping": mapping,
        "provenance": provenance,
        "unblinding_requires_frozen_STOP": True,
    }
    freeze_or_check(WORKSPACE / "SEALED/DO_NOT_OPEN_CANDIDATE_IDENTITY_MAPPING.json", sealed)
    # This public binding is a commitment, not an identity-linkable file hash per blinded candidate.
    public = {
        "candidate_labels": list(LABELS),
        "frozen_candidate_count": 3,
        "exact_predictions_reused": True,
        "source_hash_coverage_verified": True,
        "configuration_and_weight_changes": False,
        "identity_mapping_sealed_commitment": sha(WORKSPACE / "SEALED/DO_NOT_OPEN_CANDIDATE_IDENTITY_MAPPING.json"),
        **SAFETY,
    }
    per_label, private_bindings = {}, {}
    for label in LABELS:
        info = provenance[mapping[label]]
        path = Path(info["path"])
        require(sha(path) == info["sha256"], f"Frozen prediction hash failure for {label}")
        try:
            validation = validate_candidate_run(registry_root, path, require_exact_frame_coverage=True)
        except Exception as exc:
            raise RuntimeError(f"Frozen candidate-run v2 validation failed for {label}") from exc
        payload = validation["run"]
        processed = payload["processed_frame_instances"]
        require(
            len(processed) == 1080 and len({r["source_frame_sha256"] for r in processed}) == 1044,
            f"Coverage failure for {label}",
        )
        require(
            len(payload["candidates"]) == info["operational"]["candidate_rows"],
            f"Candidate row count failure for {label}",
        )
        by_source = defaultdict(list)
        ids = set()
        for candidate in payload["candidates"]:
            cid = candidate["candidate_id"]
            require(cid not in ids, f"Ambiguous duplicate candidate ID for {label}")
            ids.add(cid)
            by_source[candidate["source_frame_sha256"]].append(candidate)
        included = []
        for frame in frames:
            source_hash = frame["source_frame_sha256"]
            coverage = [p for p in processed if p["source_frame_sha256"] == source_hash]
            require(bool(coverage), f"Missing finalized frame coverage for {label}")
            candidates = by_source.get(source_hash, [])
            for candidate in candidates:
                box = candidate["box_xyxy"]
                w, h = frame["source_width"], frame["source_height"]
                require(
                    candidate["source_width"] == w and candidate["source_height"] == h, "Candidate dimensions mismatch"
                )
                require(
                    candidate["coordinate_space"] == "SOURCE" and candidate["class_label"] == "person",
                    "Invalid candidate class/coordinates",
                )
                require(
                    len(box) == 4
                    and all(np.isfinite(v) for v in box)
                    and 0 <= box[0] < box[2] <= w
                    and 0 <= box[1] < box[3] <= h,
                    "Invalid frozen candidate box",
                )
                require(
                    np.isfinite(candidate["confidence"]) and 0 <= candidate["confidence"] <= 1,
                    "Invalid original confidence",
                )
            included.append({**frame, "candidates": candidates})
        per_label[label] = included
        private_bindings[label] = {
            "path": str(path),
            "sha256": info["sha256"],
            "role": mapping[label],
            "candidate_run_v2_coverage": validation["coverage"],
            "config_sha256": hashlib.sha256(
                canonical_json_bytes(
                    {k: v for k, v in payload.items() if k not in {"candidates", "processed_frame_instances"}}
                )
            ).hexdigest(),
        }
    write_new(run / "SEALED/exact_candidate_bindings.json", private_bindings)
    return per_label, public, mapping


def run_audit(freeze_path: Path) -> None:
    require(Path(sys.executable).resolve() == PYTHON.resolve(), "Only fi-reviewer Python is authorized")
    head = git("rev-parse", "HEAD")
    require(head == git("rev-parse", "origin/main"), "HEAD must equal origin/main")
    require(
        subprocess.call(["git", "merge-base", "--is-ancestor", EXPECTED_COMMIT, head], cwd=REPO) == 0,
        "Accepted scored reviewer commit must remain an ancestor",
    )
    require(
        not git("diff", "--name-only") and not git("diff", "--cached", "--name-only"),
        "Pre-existing tracked files changed",
    )
    protocol = preregister()
    before = read(freeze_path)
    frames, corpus, metadata, touched = discover(before)
    n = len(frames)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = WORKSPACE / "runs" / f"N{n:03d}" / stamp
    run.mkdir(parents=True, exist_ok=False)
    write_new(
        run / "IMPLEMENTATION_BINDING.json",
        {
            "python_executable": str(PYTHON),
            "base_commit": head,
            "files": [
                {"path": relative, "sha256": sha(REPO / relative)}
                for relative in (
                    "scripts/g7f_d_interim_dense_gold_audit.py",
                    "src/football_intelligence/interim_dense_gold_audit.py",
                    "tests/test_g7f_d_interim_dense_gold_audit.py",
                )
            ],
            "new_source_files_uncommitted": bool(git("ls-files", "--others", "--exclude-standard")),
            "reviewer_modified": False,
            **SAFETY,
        },
    )
    for row in read(run / "IMPLEMENTATION_BINDING.json")["files"]:
        snapshot = run / "EXECUTED_SOURCE" / row["path"]
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with snapshot.open("xb") as stream:
            stream.write((REPO / row["path"]).read_bytes())
        require(sha(snapshot) == row["sha256"], "Implementation snapshot changed while freezing")
    write_new(
        run / "RUN_STARTED.json",
        {
            "run_id": f"N{n:03d}/{stamp}",
            "before_inventory_path": str(freeze_path),
            "repository_base_commit": head,
            "protocol_sha256": sha(WORKSPACE / "01_PROTOCOL/PRE_REGISTERED_STOPPING_RULE.json"),
            **SAFETY,
        },
    )
    print(f"Audit run N{n:03d}/{stamp}; no candidate identities will be printed", flush=True)
    pack = run / "REVIEW_PACK"
    try:
        write_new(pack / "02_FINALIZED_SCORED_CORPUS_BINDING.json", corpus)
        candidate_frames, public_binding, mapping = candidate_inputs(frames, run)
        write_new(pack / "03_FROZEN_CANDIDATE_BINDING.json", public_binding)
        caches = {}
        equivalence = {}
        for label in LABELS:
            prepared = []
            for frame in candidate_frames[label]:
                prepared.append(prepare_frame(frame))
                print(f"Prepared {label}, {frame['anonymous_image_id']}", flush=True)
            caches[label] = FrameMetricCache(prepared)
            # Frozen oracle on each real frame. Predecode ignore masks to avoid rebuilding
            # an unchanged mask for every candidate; evaluator and semantics remain untouched.
            from football_intelligence.dense_person_gold import _frame_ignore_mask

            reference_frames = [{**f, "ignore_mask": _frame_ignore_mask(f)} for f in candidate_frames[label]]
            reference = evaluate_dense_boxes(reference_frames)
            optimized = caches[label].score(np.ones(n, dtype=int))
            keys = (
                "AP_50_95",
                "AP50",
                "AP75",
                "recall_50_95",
                "recall_50",
                "recall_75",
                "evaluable_person_denominator",
            )
            require(all(reference[k] == optimized[k] for k in keys), f"Frozen evaluator equivalence failure: {label}")
            require(
                reference["DUPLICATE_CANDIDATE_AT_IOU50"] == optimized["duplicate_diagnostic_count"],
                "Duplicate equivalence failure",
            )
            require(
                reference["MULTI_PERSON_CANDIDATE_MASK_COVERAGE_030"] == optimized["merge_diagnostic_count"],
                "Merge equivalence failure",
            )
            require(
                reference["by_iou_threshold"]["0.50"]["unmatched_candidates"] == optimized["FP_count_iou50"],
                "FP equivalence failure",
            )
            equivalence[label] = {
                "all_metrics_and_diagnostics_equal": True,
                "frozen_evaluator_sha256": sha(REPO / "src/football_intelligence/dense_person_gold.py"),
            }
            print(f"Frozen evaluator equivalence passed: {label}", flush=True)
        ids = [f["anonymous_image_id"] for f in frames]
        groups = [row["match_group"] for row in corpus["corpus"]]
        print("Computing 10000 paired frame bootstrap replicates and jackknife", flush=True)
        aggregate, per_frame, statistics, raw = paired_analysis(
            caches, ids, groups, seed=protocol["bootstrap_seed"], replicates=protocol["bootstrap_replicates"]
        )
        raw_path = run / "BOOTSTRAP_DISTRIBUTIONS.npz"
        with raw_path.open("xb") as stream:
            np.savez_compressed(stream, **raw)
        statistics.update(
            {
                "full_distributions_path": "../BOOTSTRAP_DISTRIBUTIONS.npz",
                "full_distributions_sha256": sha(raw_path),
                "frozen_evaluator_equivalence": equivalence,
                **SAFETY,
            }
        )
        write_new(pack / "04_BLINDED_AGGREGATE_METRICS.json", {"metrics": aggregate, **SAFETY})
        write_new(pack / "05_PER_FRAME_METRICS.json", {"frames": per_frame, **SAFETY})
        write_new(pack / "06_BOOTSTRAP_AND_JACKKNIFE.json", statistics)
        decision = stopping_decision(aggregate, statistics)
        decision["N"] = n
        decision["phase_A_artifact_bindings"] = {
            name: sha(pack / name)
            for name in (
                "02_FINALIZED_SCORED_CORPUS_BINDING.json",
                "03_FROZEN_CANDIDATE_BINDING.json",
                "04_BLINDED_AGGREGATE_METRICS.json",
                "05_PER_FRAME_METRICS.json",
                "06_BOOTSTRAP_AND_JACKKNIFE.json",
            )
        }
        decision["protocol_sha256"] = sha(WORKSPACE / "01_PROTOCOL/PRE_REGISTERED_STOPPING_RULE.json")
        decision_path = pack / "07_STOPPING_DECISION.json"
        write_new(decision_path, decision)
        write_new(
            run / "PHASE_A_FREEZE_RECEIPT.json",
            {
                "blinded_decision_sha256": sha(decision_path),
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "candidate_identities_unblinded": False,
                "statistics_and_rule_frozen": True,
                **SAFETY,
            },
        )
        # Phase B cannot execute until the serialized Phase-A decision and receipt both exist.
        frozen_decision = read(decision_path)
        require(
            sha(decision_path) == read(run / "PHASE_A_FREEZE_RECEIPT.json")["blinded_decision_sha256"],
            "Phase-A freeze failed",
        )
        continued = frozen_decision["decision"] == "CONTINUE_ANNOTATION"
        if continued:
            tranche = {
                "frames": next_tranche(metadata, set(ids), touched),
                "target_size": 10,
                "candidate_performance_used": False,
                "started_automatically": False,
                **SAFETY,
            }
            write_new(run / "NEXT_TRANCHE_10.json", tranche)
            write_new(pack / "08_NEXT_TRANCHE_10.json", tranche)
            classification = "CONTINUE_G7F_D_INTERIM_DENSE_GOLD_NEXT_TRANCHE_REQUIRED"
        else:
            write_new(
                pack / "09_UNBLINDED_PROVISIONAL_SELECTION.json",
                {
                    "selected_provisional_candidate": mapping[decision["top_candidate"]],
                    "runner_up": mapping[decision["runner_up"]],
                    "mapping": mapping,
                    "blinded_decision_sha256_before_unblinding": sha(decision_path),
                    "unblinded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "unblind_count": 1,
                    "evidence": decision,
                    **SAFETY,
                },
            )
            classification = "PASS_G7F_D_INTERIM_DENSE_GOLD_STOP_PROVISIONAL_INTERNAL_SELECTION"
        after = inventory(DECISIONS)
        require(after == before, "Human-decision before/after inventory mismatch")
        for info in read(run / "SEALED/exact_candidate_bindings.json").values():
            require(sha(Path(info["path"])) == info["sha256"], "Frozen candidate bytes changed during audit")
        for relative, expected in FROZEN_HASHES.items():
            require(sha(SOURCE / relative) == expected, "Frozen gold contract changed during audit")
        for info in read(RELEASE / "05_REVIEWER/reviewer_release_manifest.json")["files"]:
            require(sha(Path(info["path"])) == info["sha256"], "Scored reviewer changed during audit")
        integrity = {
            "before": before,
            "after": after,
            "full_inventory_equal": True,
            "every_preexisting_file_byte_identical": True,
            "new_human_files": 0,
            **SAFETY,
        }
        write_new(pack / "01_REAL_DECISIONS_INTEGRITY.json", integrity)
        write_new(
            pack / "00_EXECUTIVE_SUMMARY.json",
            {
                "classification": classification,
                "N": n,
                "decision": decision["decision"],
                "conditions": decision["conditions"],
                "candidate_identities_unblinded": not continued,
                "NEXT_TRANCHE_10_created": continued,
                "human_decisions_before_after_equal": True,
                "run_id": f"N{n:03d}/{stamp}",
                "provisional_only": True,
                **SAFETY,
            },
        )
        architecture = (
            "# Additive future gold layers (design only)\n\n"
            "1. DETECTION_GOLD: frame -> visible person mask -> derived box -> relevance.\n"
            "2. TEMPORAL_TRACKLET_GOLD: frame person instance -> tracklet_id "
            "in short consecutive frame/video sequences.\n"
            "3. PLAYER_IDENTITY_GOLD: tracklet_id -> roster player_id or UNKNOWN; "
            "track identity and player identity remain separate.\n"
            "4. BALL_GOLD: timestamp/frame -> image-space ball location/mask and visibility state.\n"
            "5. MATCH_STATE_GOLD: intervals/events labelled OPEN_PLAY, THROW_IN, FREE_KICK, "
            "CORNER, GOAL_KICK, KICK_OFF, STOPPAGE, OTHER/UNKNOWN.\n"
            "6. PITCH_COORDINATE_GOLD: calibration-bound player/ball observations in pitch coordinates.\n"
            "7. DERIVED FOOTBALL INTELLIGENCE: possession, passes, pressures, shape and tactical phases, "
            "with explicit upstream provenance.\n\n"
            "Every layer is versioned and additive. Higher-level inference must never overwrite "
            "lower-level human observations. "
            "Corrections append superseding records and provenance links. "
            "Isolated stills alone do not establish temporal continuity. "
            "No new layer is implemented or authorized by this note. production_ready=false.\n"
        )
        with (pack / "10_FUTURE_GOLD_LAYER_ARCHITECTURE.md").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(architecture)
        lines = [
            classification,
            "",
            f"Valid scored FIRST_PASS N={n}; calibration excluded.",
            "",
            f"Blinded decision: {decision['decision']}.",
            "",
        ]
        lines += [f"- {k}: {'PASS' if v['passed'] else 'FAIL'}." for k, v in decision["conditions"].items()]
        lines += [
            "",
            "Candidate identities remain sealed."
            if continued
            else "Current scored Dense-Gold evidence shows material and sufficiently stable internal separation "
            "to pause additional dense annotation for detector selection.",
            "This is a provisional internal detector choice only. "
            "Final promotion remains blocked pending new sealed match footage.",
            "Small disagreement-enriched internal sample; the bootstrap is descriptive, "
            "not a large-sample or sequentially calibrated guarantee.",
            "Human decisions are byte-identical. production_ready=false. "
            "BLIND_REPEAT_AUTHORIZED=false. FINAL_DETECTOR_PROMOTION=false.",
        ]
        with (pack / "11_DECISION.md").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(lines) + "\n")
        write_new(
            pack / "12_MANIFEST.json",
            {
                "files": [
                    {"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)}
                    for p in sorted(pack.iterdir())
                    if p.is_file()
                ],
                "classification": classification,
                "blinded_decision_sha256": sha(decision_path),
                "self_hash_included": False,
                **SAFETY,
            },
        )
        write_new(
            run / "RUN_COMPLETED.json",
            {
                "classification": classification,
                "review_pack_manifest_sha256": sha(pack / "12_MANIFEST.json"),
                "human_inventory_sha256": after["ordered_inventory_sha256"],
                **SAFETY,
            },
        )
        print(
            json.dumps(
                {
                    "run_path": str(run),
                    "N": n,
                    "decision": decision["decision"],
                    "conditions": decision["conditions"],
                    "identities_unblinded": not continued,
                    "next_tranche_created": continued,
                    "human_inventory_equal": True,
                    **SAFETY,
                }
            ),
            flush=True,
        )
    except Exception as exc:
        write_new(
            run / "HOLD.json",
            {
                "classification": "HOLD_G7F_D_INTERIM_DENSE_GOLD_BAKEOFF_INTEGRITY_FAILURE",
                "reason": str(exc),
                "human_inventory_equal": inventory(DECISIONS) == before,
                **SAFETY,
            },
        )
        raise


def verify_run(run: Path) -> None:
    """Append verification evidence without rewriting the frozen decision or review pack."""
    run = run.resolve()
    require(run.is_relative_to(WORKSPACE.resolve() / "runs"), "Audit run must be inside this stage workspace")
    require((run / "RUN_COMPLETED.json").is_file() and not (run / "HOLD.json").exists(), "Run did not complete")
    pack = run / "REVIEW_PACK"
    manifest = read(pack / "12_MANIFEST.json")
    for row in manifest["files"]:
        path = pack / row["path"]
        require(sha(path) == row["sha256"] and path.stat().st_size == row["bytes"], "Review pack artifact changed")
    receipt = read(run / "PHASE_A_FREEZE_RECEIPT.json")
    decision = read(pack / "07_STOPPING_DECISION.json")
    require(sha(pack / "07_STOPPING_DECISION.json") == receipt["blinded_decision_sha256"], "Blinded decision changed")
    scores = read(pack / "04_BLINDED_AGGREGATE_METRICS.json")["metrics"]
    stats = read(pack / "06_BOOTSTRAP_AND_JACKKNIFE.json")
    recalculated = stopping_decision(scores, stats)
    require(all(decision[k] == value for k, value in recalculated.items()), "Stopping decision cannot be reproduced")
    top, runner = decision["top_candidate"], decision["runner_up"]
    full_gap = round(scores[top]["AP_50_95"] - scores[runner]["AP_50_95"], 6)
    match_influence = []
    for row in stats["match_sensitivity"]:
        without = row["leave_group_out"]
        remaining_gap = round(without[top]["AP_50_95"] - without[runner]["AP_50_95"], 6) if without else None
        match_influence.append(
            {
                "match_group": row["match_group"],
                "frame_count": row["frames"],
                "top_vs_runner_gap_without_match": remaining_gap,
                "reverses_top_vs_runner": remaining_gap is not None and remaining_gap < 0,
                "positive_AP_gap_removal_fraction": max(0, full_gap - remaining_gap) / full_gap
                if full_gap > 0 and remaining_gap is not None
                else None,
            }
        )
    data = np.load(run / "BOOTSTRAP_DISTRIBUTIONS.npz", allow_pickle=False)
    require(
        sha(run / "BOOTSTRAP_DISTRIBUTIONS.npz") == stats["full_distributions_sha256"], "Bootstrap distribution changed"
    )
    require(data["sampled_frame_indices"].shape == (10000, decision["N"]), "Bootstrap replication count mismatch")
    expected_indices = np.random.default_rng(stats["seed"]).integers(0, decision["N"], size=(10000, decision["N"]))
    require(np.array_equal(expected_indices, data["sampled_frame_indices"]), "Bootstrap seed replay differs")
    deltas = {}
    for pair in stats["pairs"]:
        left, right = pair["candidate_X"][-1], pair["candidate_Y"][-1]
        for metric in ("AP_50_95", "recall_50_95"):
            delta = data[f"{left}_{metric}"] - data[f"{right}_{metric}"]
            require(float((delta > 0).mean()) == pair[metric]["probability_X_gt_Y"], "Bootstrap probability differs")
            require(
                np.array_equal(
                    np.quantile(delta, [0.05, 0.50, 0.95]), [pair[metric][q] for q in ("p05", "p50", "p95")]
                ),
                "Bootstrap quantile replay differs",
            )
            deltas[f"{left}_minus_{right}_{metric}"] = delta
    data.close()
    with (run / "PAIRED_DELTA_DISTRIBUTIONS.npz").open("xb") as stream:
        np.savez_compressed(stream, **deltas)
    integrity = read(pack / "01_REAL_DECISIONS_INTEGRITY.json")
    current = inventory(DECISIONS)
    require(current == integrity["before"] == integrity["after"], "Human-decision inventory changed")
    continued = decision["decision"] == "CONTINUE_ANNOTATION"
    identities = (
        "LOCAL_DEFAULT_RERUN",
        "RECALL_ORIENTED_VARIANT",
        "MULTIPLICITY_REDUCTION_VARIANT",
        "g7f_b_r1_local_default",
        "g7f_b_r1_recall_conf",
        "g7f_b_r1_multiplicity_nms",
    )
    if continued:
        require(not (pack / "09_UNBLINDED_PROVISIONAL_SELECTION.json").exists(), "CONTINUE must not unblind")
        for path in pack.iterdir():
            if path.is_file():
                content = path.read_text(encoding="utf-8")
                require(not any(name in content for name in identities), "Identity leak in human-facing review pack")
        tranche = read(run / "NEXT_TRANCHE_10.json")
        require(tranche == read(pack / "08_NEXT_TRANCHE_10.json"), "Next-tranche copies differ")
        selected_ids = {r["anonymous_image_id"] for r in tranche["frames"]}
        corpus_ids = {r["anonymous_image_id"] for r in read(pack / "02_FINALIZED_SCORED_CORPUS_BINDING.json")["corpus"]}
        require(
            not selected_ids & corpus_ids and len(selected_ids) == len(tranche["frames"]), "Next tranche overlaps truth"
        )
        require(
            not any((DECISIONS / "drafts" / f"first_pass__{image_id}.json").exists() for image_id in selected_ids),
            "Next tranche contains touched drafts",
        )
    executed = read(run / "IMPLEMENTATION_BINDING.json")
    for row in executed["files"]:
        require(sha(run / "EXECUTED_SOURCE" / row["path"]) == row["sha256"], "Executed source snapshot mismatch")
    patch_sections = []
    for row in executed["files"]:
        relative = row["path"]
        lines = (REPO / relative).read_text(encoding="utf-8").splitlines()
        patch_sections.extend(
            [
                f"diff --git a/{relative} b/{relative}",
                "new file mode 100644",
                "--- /dev/null",
                f"+++ b/{relative}",
                f"@@ -0,0 +1,{len(lines)} @@",
                *[f"+{line}" for line in lines],
            ]
        )
    with (run / "SOURCE_ADDITIONS.patch").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(patch_sections) + "\n")
    evidence = {
        "run": str(run),
        "human_inventory_equal": True,
        "inventory_file_count": current["file_count"],
        "blinded_decision_hash_valid": True,
        "bootstrap_replay_exact": True,
        "match_sensitivity_interpretation": {
            "groups": match_influence,
            "any_match_removal_reverses_top_vs_runner": any(r["reverses_top_vs_runner"] for r in match_influence),
            "diagnostic_only_no_added_stopping_threshold": True,
        },
        "candidate_identities_unblinded": not continued,
        "human_facing_identity_scan_passed": continued,
        "reviewer_and_human_truth_modified": False,
        "implementation_snapshots_hash_valid": True,
        "current_source_files": [{"path": r["path"], "sha256": sha(REPO / r["path"])} for r in executed["files"]],
        "maintenance_since_execution": (
            "Runner permits future clean commits descended from accepted release; metrics and stopping code unchanged"
        ),
        "test_summary": read(WORKSPACE / "TESTS/TEST_SUMMARY.json"),
        "evidence_hashes": {
            name: sha(run / name) for name in ("SOURCE_ADDITIONS.patch", "PAIRED_DELTA_DISTRIBUTIONS.npz")
        },
        **SAFETY,
    }
    write_new(run / "FINAL_VERIFICATION.json", evidence)
    print(
        json.dumps(
            {
                "verification": "PASS",
                "N": decision["N"],
                "decision": decision["decision"],
                "human_inventory_equal": True,
                "candidate_identities_unblinded": not continued,
                **SAFETY,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "preflight", "run", "verify"), default="freeze", nargs="?")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
    elif args.phase == "preflight":
        preregister()
        frames, corpus, _, _ = discover(read(args.freeze or INITIAL_FREEZE))
        write_new(WORKSPACE / "00_FREEZE/initial_corpus_validation.json", corpus)
        print(
            json.dumps(
                {
                    "N": len(frames),
                    "match_coverage": corpus["match_coverage"],
                    "unfinished_drafts": corpus["unfinished_scored_drafts"],
                }
            )
        )
    elif args.phase == "verify":
        require(args.run is not None, "--run is required for verification")
        verify_run(args.run)
    else:
        run_audit(args.freeze or freeze())
