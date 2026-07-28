from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_intelligence.dataset_inventory import run_inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--matches-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--expected-match-count", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_inventory(
        project_root=args.project_root,
        matches_root=args.matches_root,
        dataset_root=args.dataset_root,
        experiment_root=args.experiment_root,
        expected_count=args.expected_match_count,
        dry_run=args.dry_run,
    )
    out = args.experiment_root / "02_SOURCE_INVENTORY" / "inventory_result.json"
    if not args.dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "matches": result["matches"],
                "source_file_count": len(result["source_records"]),
                "media_metadata_count": len(result["media_metadata"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
