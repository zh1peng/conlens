"""Command-line entry point for reproducible table-based LENS analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .core import lens_enrich


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="conlens", description="Run LENS enrichment")
    parser.add_argument("edges", type=Path, help="CSV edge table with node1,node2,statistic")
    parser.add_argument("sets", type=Path, help="JSON mapping set names to edge_id arrays")
    parser.add_argument("output", type=Path, help="Output LensResult JSON")
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--diagonal", action="store_true")
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument(
        "--score-type", choices=["standard", "positive", "negative"], default="standard"
    )
    parser.add_argument("--min-size", type=int, default=5)
    parser.add_argument("--max-size", type=int)
    parser.add_argument("--null-method", choices=["edge_permutation", "provided_null"])
    parser.add_argument("--provided-null", type=Path)
    parser.add_argument("--provided-null-kind", choices=["es", "statistics", "ranks"], default="es")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--random-state", type=int)
    parser.add_argument("--positive-direction")
    parser.add_argument("--store-running-sum", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    edges = pd.read_csv(args.edges)
    edge_sets = json.loads(args.sets.read_text(encoding="utf-8"))
    supplied_payload = (
        json.loads(args.provided_null.read_text(encoding="utf-8"))
        if args.provided_null is not None
        else None
    )
    if args.null_method == "provided_null" and supplied_payload is None:
        raise SystemExit("--provided-null is required with --null-method provided_null")
    if args.null_method == "provided_null":
        required = {"data", "edge_ids", "edge_sets", "positive_direction"}
        if not isinstance(supplied_payload, dict) or not required.issubset(supplied_payload):
            raise SystemExit(
                "provided-null JSON must contain data, edge_ids, edge_sets, and positive_direction"
            )
        supplied = supplied_payload["data"]
        supplied_edge_ids = supplied_payload["edge_ids"]
        supplied_edge_sets = supplied_payload["edge_sets"]
        supplied_direction = supplied_payload["positive_direction"]
    else:
        supplied = supplied_edge_ids = supplied_edge_sets = supplied_direction = None
    result = lens_enrich(
        edges,
        edge_sets,
        directed=args.directed,
        diagonal=args.diagonal,
        weight=args.weight,
        score_type=args.score_type,
        min_size=args.min_size,
        max_size=args.max_size,
        null_method=args.null_method,
        provided_null=supplied,
        provided_null_kind=args.provided_null_kind,
        provided_null_edge_ids=supplied_edge_ids,
        provided_null_edge_sets=supplied_edge_sets,
        provided_null_direction=supplied_direction,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
        positive_direction=args.positive_direction or supplied_direction,
        store_running_sum=args.store_running_sum,
    )
    result.save(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
