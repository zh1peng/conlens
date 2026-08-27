"""Compare the NumPy null-LENS fast path with the materialized Pandas oracle."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterator, Mapping

import numpy as np

from conlens import (
    Contrast,
    DesignMatrix,
    EdgeStatistics,
    lens_edge_permute,
    lens_fl_permute,
    lens_stat,
    make_design,
    make_edge_statistics,
    matrix_to_edges,
)


def _edge_problem(
    *, nodes: int, n_sets: int, set_size: int, seed: int
) -> tuple[EdgeStatistics, dict[str, set[str]]]:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(nodes, nodes))
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    edges = make_edge_statistics(
        matrix_to_edges(matrix), positive_direction="larger statistic"
    )
    identifiers = edges.table["edge_id"].to_numpy(str)
    if set_size >= len(identifiers):
        raise ValueError("set_size must be smaller than the edge universe")
    edge_sets = {
        f"set_{index}": set(
            rng.choice(identifiers, size=set_size, replace=False).tolist()
        )
        for index in range(n_sets)
    }
    return edges, edge_sets


def _fl_problem(
    *, subjects: int, nodes: int, n_sets: int, set_size: int, seed: int
) -> tuple[
    np.ndarray,
    DesignMatrix,
    Mapping[str, Contrast],
    dict[str, set[str]],
]:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(subjects, nodes, nodes))
    connectomes = (raw + raw.transpose(0, 2, 1)) / 2
    for matrix in connectomes:
        np.fill_diagonal(matrix, 0)
    design = make_design(continuous={"age": np.linspace(20, 80, subjects)})
    contrasts = {"age": Contrast({"age": 1}, "partial_r", "increases with age")}
    edge_ids = matrix_to_edges(connectomes[0])["edge_id"].to_numpy(str)
    if set_size >= len(edge_ids):
        raise ValueError("set_size must be smaller than the edge universe")
    edge_sets = {
        f"set_{index}": set(rng.choice(edge_ids, size=set_size, replace=False).tolist())
        for index in range(n_sets)
    }
    return connectomes, design, contrasts, edge_sets


def _consume(
    generator: Iterator[EdgeStatistics | dict[str, EdgeStatistics]],
    edge_sets: Mapping[str, set[str]],
    *,
    materialize: bool,
) -> np.ndarray:
    scores: list[float] = []
    for replicate in generator:
        mapping = replicate if isinstance(replicate, dict) else {"single": replicate}
        if materialize:
            for item in mapping.values():
                _ = item.table
        result = lens_stat(mapping, edge_sets)
        for contrast in result.values():
            scores.extend(float(item.ES) for item in contrast.sets if item.ES is not None)
    return np.asarray(scores)


def _time(
    factory: Callable[[], Iterator], edge_sets, *, materialize: bool
) -> tuple[float, np.ndarray]:
    started = time.perf_counter()
    scores = _consume(factory(), edge_sets, materialize=materialize)
    return time.perf_counter() - started, scores


def _report(label: str, factory: Callable[[], Iterator], edge_sets) -> None:
    numpy_seconds, numpy_scores = _time(factory, edge_sets, materialize=False)
    pandas_seconds, pandas_scores = _time(factory, edge_sets, materialize=True)
    if not np.array_equal(numpy_scores, pandas_scores):
        raise AssertionError(f"{label} benchmark paths produced different scores")
    print(
        f"{label}: NumPy={numpy_seconds:.3f}s, Pandas={pandas_seconds:.3f}s, "
        f"speedup={pandas_seconds / numpy_seconds:.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["edge", "fl", "both"], default="both")
    parser.add_argument("--nodes", type=int, default=120)
    parser.add_argument("--subjects", type=int, default=60)
    parser.add_argument("--sets", type=int, default=12)
    parser.add_argument("--set-size", type=int, default=400)
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode in {"edge", "both"}:
        edges, edge_sets = _edge_problem(
            nodes=args.nodes, n_sets=args.sets, set_size=args.set_size, seed=args.seed
        )
        _report(
            "edge permutation",
            lambda: lens_edge_permute(
                edges, n_permutations=args.permutations, random_state=args.seed
            ),
            edge_sets,
        )
    if args.mode in {"fl", "both"}:
        connectomes, design, contrasts, edge_sets = _fl_problem(
            subjects=args.subjects,
            nodes=args.nodes,
            n_sets=args.sets,
            set_size=args.set_size,
            seed=args.seed,
        )
        _report(
            "Freedman-Lane",
            lambda: lens_fl_permute(
                connectomes,
                design=design,
                contrasts=contrasts,
                n_permutations=args.permutations,
                random_state=args.seed,
            ),
            edge_sets,
        )


if __name__ == "__main__":
    main()
