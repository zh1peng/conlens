"""Bootstrap-derived stability summaries without inventing subject-level data."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .core import lens_enrich
from .results import LensResult


def _jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    return 1.0 if not union else len(first & second) / len(union)


def _dice(first: set[str], second: set[str]) -> float:
    total = len(first) + len(second)
    return 1.0 if total == 0 else 2 * len(first & second) / total


def bootstrap_lens(
    edges: pd.DataFrame,
    edge_sets: Mapping[str, Iterable[str]],
    *,
    statistic_replicates: np.ndarray | None = None,
    results: Iterable[LensResult] | None = None,
    subject_data: np.ndarray | None = None,
    statistic_function: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    n_bootstraps: int = 1000,
    random_state: int | None = None,
    strata: Iterable[Any] | None = None,
    **lens_options: Any,
) -> list[LensResult]:
    """Analyze one explicit bootstrap source without inventing subject-level data.

    ``statistic_function`` receives the resampled subject-by-edge matrix and the
    original-row indices used for that replicate. The indices let callers resample
    phenotype labels and covariates with exactly the same bootstrap draw.
    """
    n_sources = sum(source is not None for source in (statistic_replicates, results, subject_data))
    if n_sources != 1:
        raise ValueError("provide exactly one of statistic_replicates, results, or subject_data")
    if results is not None:
        output = list(results)
        if not output:
            raise ValueError("results cannot be empty")
        return output
    if subject_data is not None:
        values = np.asarray(subject_data, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(edges):
            raise ValueError("subject_data must have shape (subjects, edges)")
        if not np.isfinite(values).all():
            raise ValueError("subject_data must be finite")
        if statistic_function is None:
            raise ValueError("statistic_function is required with subject_data")
        if n_bootstraps < 1:
            raise ValueError("n_bootstraps must be >= 1")
        strata_values = None if strata is None else np.asarray(list(strata), dtype=object)
        if strata_values is not None and (
            strata_values.ndim != 1 or len(strata_values) != len(values)
        ):
            raise ValueError("strata must contain one value per subject")
        rng = np.random.default_rng(random_state)
        generated = []
        all_indices = np.arange(len(values))
        for _ in range(n_bootstraps):
            if strata_values is None:
                indices = rng.choice(all_indices, size=len(values), replace=True)
            else:
                samples = []
                for label in pd.unique(strata_values):
                    members = all_indices[strata_values == label]
                    samples.append(rng.choice(members, size=len(members), replace=True))
                indices = np.concatenate(samples)
            statistic = np.asarray(statistic_function(values[indices], indices), dtype=float)
            if statistic.shape != (len(edges),) or not np.isfinite(statistic).all():
                raise ValueError("statistic_function must return one finite statistic per edge")
            generated.append(statistic)
        replicates = np.stack(generated)
    else:
        replicates = np.asarray(statistic_replicates, dtype=float)
    if replicates.ndim != 2 or replicates.shape[1] != len(edges):
        raise ValueError("statistic_replicates must have shape (replicates, edges)")
    if not np.isfinite(replicates).all():
        raise ValueError("statistic_replicates must be finite")
    output = []
    for values in replicates:
        frame = edges.copy()
        frame["statistic"] = values
        output.append(lens_enrich(frame, edge_sets, **lens_options))
    return output


def summarize_stability(
    results: Iterable[LensResult], *, significance_alpha: float = 0.05
) -> dict[str, Any]:
    replicates = list(results)
    if not replicates:
        raise ValueError("at least one LensResult is required")
    if not 0 < significance_alpha < 1:
        raise ValueError("significance_alpha must be in (0, 1)")
    set_names = [item.set_name for item in replicates[0].sets]
    if any([item.set_name for item in result.sets] != set_names for result in replicates[1:]):
        raise ValueError("all results must contain identically ordered set definitions")
    summaries: dict[str, Any] = {}
    for name in set_names:
        set_results = [result.get(name) for result in replicates]
        reference_members = set(set_results[0].edge_set_ids)
        if any(set(item.edge_set_ids) != reference_members for item in set_results[1:]):
            raise ValueError(f"edge-set definition changed across replicates for {name!r}")
        edge_universe = sorted(reference_members)
        node_universe: set[Any] = set()
        for result in replicates:
            if result.ranked_edges is not None:
                rows = result.ranked_edges[result.ranked_edges["edge_id"].isin(reference_members)]
                node_universe.update(rows["node1"])
                node_universe.update(rows["node2"])
        edge_frequency = {
            edge: sum(edge in item.leading_edge_ids for item in set_results) / len(set_results)
            for edge in edge_universe
        }
        node_frequency = {
            node: sum(node in item.leading_node_ids for item in set_results) / len(set_results)
            for node in sorted(node_universe, key=str)
        }
        pairs = list(combinations([set(item.leading_edge_ids) for item in set_results], 2))
        summaries[name] = {
            "edge_inclusion_frequency": edge_frequency,
            "node_inclusion_frequency": node_frequency,
            "significance_frequency": float(
                np.mean(
                    [
                        item.q_value is not None and item.q_value < significance_alpha
                        for item in set_results
                    ]
                )
            ),
            "significance_alpha": significance_alpha,
            "NES_distribution": [item.NES for item in set_results],
            "leading_edge_size_distribution": [item.leading_edge_size for item in set_results],
            "pairwise_jaccard": [_jaccard(first, second) for first, second in pairs],
            "pairwise_dice": [_dice(first, second) for first, second in pairs],
        }
    return {"n_replicates": len(replicates), "sets": summaries}


def consensus_network(
    results: Iterable[LensResult],
    set_name: str,
    *,
    threshold: float,
) -> pd.DataFrame:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be explicitly specified in [0, 1]")
    replicates = list(results)
    summary = summarize_stability(replicates)["sets"][set_name]
    selected = {
        edge
        for edge, frequency in summary["edge_inclusion_frequency"].items()
        if frequency >= threshold
    }
    template = next(
        (result.ranked_edges for result in replicates if result.ranked_edges is not None), None
    )
    if template is None:
        raise ValueError("ranked edges are required for a consensus network")
    output = template[template["edge_id"].isin(selected)].copy()
    output["inclusion_frequency"] = output["edge_id"].map(summary["edge_inclusion_frequency"])
    return output.reset_index(drop=True)
