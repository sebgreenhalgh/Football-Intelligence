from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_intelligence.replay.blind_pipeline import build_input_closure, run_blind_pipeline_boundary
from football_intelligence.replay.blind_window_extractor import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attempt isolated M5.3 blind-window visual pipeline run.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    closure_path = args.stage_root / "pipeline/input_closure.json"
    closure = (
        read_json(closure_path)
        if closure_path.exists()
        else build_input_closure(
            stage_root=args.stage_root,
            repo_root=args.repo_root,
            config_path=args.config,
            selection_seal=args.stage_root / "selection/blind_window_selection_seal.json",
            source_manifest=args.stage_root / "source/source_video_manifest.json",
            frame_manifest=args.stage_root / "frames/extraction_a/frame_manifest.json",
            retention_contract=args.stage_root / "source/artifact_retention_contract.json",
        )
    )
    run_root = args.stage_root / "runs" / args.run_id
    summary = run_blind_pipeline_boundary(
        run_root=run_root,
        repo_root=args.repo_root,
        frame_manifest=args.stage_root / "frames/extraction_a/frame_manifest.json",
        input_closure=closure,
        run_label=args.run_id,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
