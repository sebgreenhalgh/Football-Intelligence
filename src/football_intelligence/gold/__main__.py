"""fi-gold: no inference, no review UI, no automatic promotion."""

import argparse
import json
import sys
from pathlib import Path

from .corpus import GoldCorpus, GoldError, canonical, immutable_write


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Defaults to FI_GOLD_ROOT or external datasets/gold_corpus")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    validate = sub.add_parser("validate")
    validate.add_argument("--release")
    ingest = sub.add_parser("ingest")
    ingest.add_argument("decision_root", type=Path)
    ingest.add_argument("--source-manifest", type=Path, required=True)
    ingest.add_argument("--code-commit", required=True)
    temporal = sub.add_parser("ingest-temporal", help="Later-stage complete sequence ingestion; never drafts")
    temporal.add_argument("decision_root", type=Path)
    temporal.add_argument("--selection-manifest", type=Path, required=True)
    temporal.add_argument("--code-commit", required=True)
    temporal_detection = sub.add_parser("ingest-temporal-detection", help="Later-stage human DETECTION frames; preserves canonical anchors")
    temporal_detection.add_argument("decision_root", type=Path)
    temporal_detection.add_argument("--selection-manifest", type=Path, required=True)
    temporal_detection.add_argument("--code-commit", required=True)
    evidence = sub.add_parser(
        "register-evidence", help="Preserve legacy provenance without asserting a new active layer"
    )
    evidence.add_argument("source_manifest", type=Path)
    evidence.add_argument("--code-commit", required=True)
    release = sub.add_parser("release")
    release.add_argument("version")
    release.add_argument("--code-commit", required=True)
    export = sub.add_parser("export-detection")
    export.add_argument("--release", required=True)
    export.add_argument(
        "--split",
        choices=["DENSE_GOLD_INTERNAL_VALIDATION", "CALIBRATION_ONLY", "ALL"],
        default="DENSE_GOLD_INTERNAL_VALIDATION",
    )
    export.add_argument("--output", type=Path, required=True)
    export_temporal = sub.add_parser("export-temporal")
    export_temporal.add_argument("--release", required=True)
    export_temporal.add_argument("--layer", choices=["TRACKLET", "BALL", "MATCH_STATE"], required=True)
    export_temporal.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    corpus = GoldCorpus(args.root)
    try:
        if args.command == "status":
            result = corpus.status()
        elif args.command == "validate":
            result = corpus.validate(release=args.release)
        elif args.command == "ingest":
            result = corpus.ingest(args.decision_root, args.source_manifest, code_commit=args.code_commit)
        elif args.command == "ingest-temporal":
            result = corpus.ingest_temporal(args.decision_root, args.selection_manifest, code_commit=args.code_commit)
        elif args.command == "ingest-temporal-detection":
            result = corpus.ingest_temporal_detection(args.decision_root, args.selection_manifest, code_commit=args.code_commit)
        elif args.command == "release":
            result = corpus.create_release(args.version, code_commit=args.code_commit)
        elif args.command == "register-evidence":
            result = corpus.register_evidence(args.source_manifest, code_commit=args.code_commit)
        elif args.command == "export-detection":
            rows = corpus.export_detection(release=args.release, split=None if args.split == "ALL" else args.split)
            immutable_write(args.output, b"".join(canonical(row) for row in rows))
            result = {"exported_frames": len(rows), "output": str(args.output), "production_ready": False}
        else:
            rows = corpus.export_temporal(release=args.release, layer=args.layer)
            immutable_write(args.output, b"".join(canonical(row) for row in rows))
            result = {"exported_sequences": len(rows), "layer": args.layer, "output": str(args.output), "production_ready": False}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (GoldError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"GOLD_VALIDATION_FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
