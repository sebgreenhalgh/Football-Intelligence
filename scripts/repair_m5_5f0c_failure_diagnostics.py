"""Run fresh CUDA evidence only for the historical frame-32/frame-65 failures."""

# ruff: noqa: E501

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_m5_5f0_stable_local_strand as cpu
import build_m5_5f0a_cuda_continuity as f0a
from football_intelligence.review_chassis.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1].parent
MATCH = ROOT / "matches" / "128058"
STAGE = MATCH / "runs" / "step_m5" / "part 2" / "M5_5F0C_SEED_CURATION_DEDUPLICATION_AND_ONE_FRAME_DROPOUT_REPAIR_v1"
PRIOR = MATCH / "runs" / "step_m5" / "part 2" / "M5_5F0B_HUMAN_REVIEW_INGESTION_LEVEL2_SWITCH_REPAIR_AND_SEED_QC_v1"
MODEL = Path(__file__).resolve().parents[1] / "models" / "model=yolov8m-imgsz=2048.pt"
MODEL_HASH = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8"
    )


def main() -> None:
    if sha256_file(MODEL) != MODEL_HASH:
        raise RuntimeError("approved detector hash mismatch")
    events, original = cpu.prior_e3.source_rows()
    lookup, _ = cpu.source_lookup(events)
    source = original["stage_a_canonical_10fps_window"]
    candidates = []
    for start in (26, 59):
        candidate = cpu.benchmark_candidate(source, lookup, start, 2)
        if candidate is None:
            raise RuntimeError(f"failure diagnostic candidate unavailable at {start}")
        candidate.update(
            {"benchmark_case_id": f"m5_5f0c_failure_window_{start}", "requested_level": 2, "human_answers_used": False}
        )
        candidates.append(candidate)
    detector = f0a.run_gpu_detector(events, {"stage_a_canonical_10fps_window": source}, lookup, candidates)
    rows = detector["rows_by_variant"].get(1280, {})
    out = STAGE / "03_FRAME32_AND_FRAME65_DROPOUT_ROOT_CAUSE"
    write_jsonl(out / "failure_window_detector_rows.jsonl", [row for frame_rows in rows.values() for row in frame_rows])
    diagnostics_path = out / "dropout_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    for item in diagnostics:
        item["fresh_failure_window_detector"] = True
        item["fresh_detector_counts"] = {
            str(frame): len(rows.get(frame, [])) for frame in range(item["event_frame"] - 2, item["event_frame"] + 3)
        }
        item["fresh_detector_row_count_at_failure"] = len(rows.get(item["event_frame"], []))
        item["fresh_detector_checkpoint_sha256"] = MODEL_HASH
        item["fresh_detector_device"] = detector["device"]
    write_json(diagnostics_path, diagnostics)
    write_json(
        out / "failure_window_detector_summary.json",
        {
            key: value
            for key, value in detector.items()
            if key not in {"rows", "rows_by_frame", "rows_by_variant", "oom_rows"}
        },
    )
    print(
        json.dumps(
            {
                "device": detector["device"],
                "frames": {str(frame): len(rows.get(frame, [])) for frame in (32, 65)},
                "row_count": detector["row_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
