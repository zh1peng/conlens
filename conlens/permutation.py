"""Streaming edge-label permutation for precomputed edge statistics."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np
import pandas as pd

from .core import _coerce_edge_statistics
from .results import EdgeStatistics


def _permuted_table(source: EdgeStatistics, indices: np.ndarray) -> pd.DataFrame:
    columns = ["node1", "node2", "edge_id", "canonical_edge_id"]
    table = source.table[columns].copy()
    table["statistic"] = source.table["statistic"].to_numpy(float)[indices]
    table.attrs.update(source.table.attrs)
    return table


def lens_edge_permute(
    edge_statistics: EdgeStatistics | pd.DataFrame | Mapping[str, EdgeStatistics],
    *,
    n_permutations: int,
    positive_direction: str | None = None,
    random_state: int | None = None,
) -> Iterator[EdgeStatistics | dict[str, EdgeStatistics]]:
    """Yield edge-label-permuted statistics without retaining a null matrix.

    A mapping is permuted with the same edge permutation in every contrast so
    the dependence among contrasts is preserved.
    """
    if not isinstance(n_permutations, int) or n_permutations < 1:
        raise ValueError("n_permutations must be a positive integer")
    is_mapping = isinstance(edge_statistics, Mapping) and not isinstance(
        edge_statistics, pd.DataFrame
    )
    if is_mapping:
        assert isinstance(edge_statistics, Mapping)
        if not edge_statistics:
            raise ValueError("edge_statistics mapping cannot be empty")
        prepared = {
            str(name): _coerce_edge_statistics(item) for name, item in edge_statistics.items()
        }
    else:
        prepared = {
            "__single__": _coerce_edge_statistics(
                edge_statistics,
                positive_direction=positive_direction,
            )
        }

    reference_ids: list[str] | None = None
    for item in prepared.values():
        current_ids = item.table["edge_id"].astype(str).tolist()
        if reference_ids is None:
            reference_ids = current_ids
        elif current_ids != reference_ids:
            raise ValueError("all contrasts must use the same ordered edge universe")
    assert reference_ids is not None

    rng = np.random.default_rng(random_state)
    for replicate in range(n_permutations):
        indices = rng.permutation(len(reference_ids))
        output: dict[str, EdgeStatistics] = {}
        for name, source in prepared.items():
            metadata = {
                **source.metadata,
                "source": "edge_permutation_null",
                "permutation_scheme": "edge_label_permutation",
                "permutation_index": replicate,
                "random_seed": random_state,
            }
            output[name] = EdgeStatistics(_permuted_table(source, indices), metadata)
        yield output if is_mapping else output["__single__"]
