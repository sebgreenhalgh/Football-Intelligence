from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from football_intelligence.gold_eval.core import build_gold_corpus, metric_capabilities, validate_gold
from football_intelligence.gold_eval.evaluation import (
    build_error_ledger,
    compare_runs,
    evaluate_candidate_run,
    evaluate_frozen_baseline,
    render_review_sample,
    require_supported_metric,
)


def _path(value: str) -> Path:
    return Path(value).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="G7F-A temporal-observation gold evaluator")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-gold")
    build.add_argument("--decisions-root", required=True, type=_path)
    build.add_argument("--reviewer-package", required=True, type=_path)
    build.add_argument("--workspace", required=True, type=_path)
    build.add_argument("--repository", required=True, type=_path)
    for name in (
        "validate-gold",
        "show-capabilities",
        "evaluate-frozen-baseline",
        "build-error-ledger",
        "render-review-sample",
    ):
        command = commands.add_parser(name)
        command.add_argument("--workspace", required=True, type=_path)
        if name in {"evaluate-frozen-baseline", "build-error-ledger", "render-review-sample"}:
            command.add_argument("--output", required=True, type=_path)
        if name == "render-review-sample":
            command.add_argument("--limit", type=int, default=24)
    candidate = commands.add_parser("evaluate-candidate-run")
    candidate.add_argument("--workspace", required=True, type=_path)
    candidate.add_argument("--run", required=True, type=_path)
    candidate.add_argument("--output", required=True, type=_path)
    compare = commands.add_parser("compare-runs")
    compare.add_argument("--reports", required=True, type=_path, nargs="+")
    compare.add_argument("--output", required=True, type=_path)
    guard = commands.add_parser("check-metric")
    guard.add_argument("--workspace", required=True, type=_path)
    guard.add_argument("metric")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result: dict[str, Any]
    if args.command == "build-gold":
        result = build_gold_corpus(args.decisions_root, args.reviewer_package, args.workspace, args.repository)
    elif args.command == "validate-gold":
        result = validate_gold(args.workspace)
    elif args.command == "show-capabilities":
        result = metric_capabilities()
    elif args.command == "evaluate-frozen-baseline":
        result = evaluate_frozen_baseline(args.workspace, args.output)
    elif args.command == "evaluate-candidate-run":
        result = evaluate_candidate_run(args.workspace, args.run, args.output)
    elif args.command == "compare-runs":
        result = compare_runs(args.reports, args.output)
    elif args.command == "build-error-ledger":
        result = build_error_ledger(args.workspace, args.output)
    elif args.command == "render-review-sample":
        result = render_review_sample(args.workspace, args.output, args.limit)
    elif args.command == "check-metric":
        result = {"metric": args.metric, "capability_tier": require_supported_metric(args.workspace, args.metric)}
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
