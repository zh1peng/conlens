"""Command-line entry point for edge-table LENS analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .core import lens_stat, make_edge_statistics
from .enrichment import lens_enrich
from .permutation import lens_edge_permute


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="conlens", description="Run LENS enrichment")
    parser.add_argument("edges", type=Path, help="CSV edge table with node1,node2,statistic")
    parser.add_argument("sets", type=Path, help="JSON mapping set names to edge_id arrays")
    parser.add_argument("output", type=Path, help="Output LensResult JSON")
    parser.add_argument("--positive-direction", required=True)
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--diagonal", action="store_true")
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument(
        "--score-type", choices=["standard", "positive", "negative"], default="standard"
    )
    parser.add_argument("--min-size", type=int, default=5)
    parser.add_argument("--max-size", type=int)
    parser.add_argument("--n-permutations", type=int, default=0)
    parser.add_argument("--random-state", type=int)
    parser.add_argument("--family-name", default="default")
    parser.add_argument("--store-running-sum", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.n_permutations < 0:
        raise SystemExit("--n-permutations must be >= 0")
    table = pd.read_csv(args.edges)
    edge_sets = json.loads(args.sets.read_text(encoding="utf-8"))
    edge_statistics = make_edge_statistics(
        table,
        positive_direction=args.positive_direction,
        directed=args.directed,
        diagonal=args.diagonal,
    )
    observed = lens_stat(
        edge_statistics,
        edge_sets,
        weight=args.weight,
        score_type=args.score_type,
        store_running_sum=args.store_running_sum,
    )
    null_stats = None
    if args.n_permutations:
        null_edges = lens_edge_permute(
            edge_statistics,
            n_permutations=args.n_permutations,
            random_state=args.random_state,
        )
        null_stats = (
            lens_stat(item, edge_sets, weight=args.weight, score_type=args.score_type)
            for item in null_edges
        )
    result = lens_enrich(
        observed,
        null_stats,
        min_size=args.min_size,
        max_size=args.max_size,
        family_name=args.family_name,
    )
    result.save(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
