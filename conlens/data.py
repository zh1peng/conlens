"""Validation and conversion of connectome data."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_EDGE_COLUMNS = ("node1", "node2", "statistic")


def _object_vector(values: Sequence[Any]) -> np.ndarray:
    output = np.empty(len(values), dtype=object)
    output[:] = list(values)
    return output


def _validate_node_order(node_order: Sequence[Any]) -> list[Any]:
    nodes = list(node_order)
    if len(nodes) != len(set(nodes)):
        raise ValueError("node labels must be unique")
    return nodes


def canonicalize_edges(
    edges: pd.DataFrame,
    *,
    node_order: Sequence[Any] | None = None,
    directed: bool = False,
) -> pd.DataFrame:
    """Canonicalize endpoints using node indices, never lexical endpoint order."""
    if not {"node1", "node2"}.issubset(edges.columns):
        raise ValueError("edge table must contain node1 and node2")
    frame = edges.copy()
    supplied_edge_ids = "edge_id" in frame.columns
    if supplied_edge_ids:
        if frame["edge_id"].isna().any():
            raise ValueError("edge_id values must not be missing")
        if frame["edge_id"].duplicated().any():
            duplicates = frame.loc[frame["edge_id"].duplicated(False), "edge_id"].tolist()
            raise ValueError(f"duplicate edge_id values: {duplicates!r}")
        frame["edge_id"] = frame["edge_id"].astype(str)
        if (frame["edge_id"].str.len() == 0).any():
            raise ValueError("edge_id values must not be empty")
        if frame["edge_id"].duplicated().any():
            duplicates = frame.loc[frame["edge_id"].duplicated(False), "edge_id"].tolist()
            raise ValueError(f"duplicate edge_id values after string normalization: {duplicates!r}")
    if node_order is None:
        nodes = list(dict.fromkeys([*frame["node1"].tolist(), *frame["node2"].tolist()]))
    else:
        nodes = _validate_node_order(node_order)
    index = {node: i for i, node in enumerate(nodes)}
    unknown = (set(frame["node1"]) | set(frame["node2"])) - set(index)
    if unknown:
        raise ValueError(f"unknown nodes: {sorted(unknown, key=str)!r}")
    i = frame["node1"].map(index).to_numpy()
    j = frame["node2"].map(index).to_numpy()
    if not directed:
        swap = i > j
        first = frame["node1"].to_numpy(copy=True)
        second = frame["node2"].to_numpy(copy=True)
        frame["node1"] = np.where(swap, second, first)
        frame["node2"] = np.where(swap, first, second)
        i, j = np.minimum(i, j), np.maximum(i, j)
    canonical_ids = [f"{a}->{b}" if directed else f"{a}--{b}" for a, b in zip(i, j, strict=True)]
    frame["canonical_edge_id"] = canonical_ids
    if frame["canonical_edge_id"].duplicated().any():
        duplicate = frame.loc[
            frame["canonical_edge_id"].duplicated(False), "canonical_edge_id"
        ].tolist()
        raise ValueError(f"duplicate edges after canonicalization: {duplicate!r}")
    if not supplied_edge_ids:
        frame["edge_id"] = frame["canonical_edge_id"]
    frame.attrs["node_order"] = nodes
    frame.attrs["directed"] = directed
    return frame


def validate_edge_table(
    edges: pd.DataFrame,
    *,
    node_order: Sequence[Any] | None = None,
    directed: bool = False,
    diagonal: bool = False,
    nan_policy: str = "raise",
) -> pd.DataFrame:
    missing = set(REQUIRED_EDGE_COLUMNS) - set(edges.columns)
    if missing:
        raise ValueError(f"missing required edge columns: {sorted(missing)!r}")
    if nan_policy not in {"raise", "omit"}:
        raise ValueError("nan_policy must be 'raise' or 'omit'")
    frame = edges.copy()
    try:
        statistic = pd.to_numeric(frame["statistic"], errors="raise").to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("statistic must be numeric") from exc
    finite = np.isfinite(statistic)
    omitted: list[dict[str, Any]] = []
    if not finite.all():
        if nan_policy == "raise":
            bad = np.flatnonzero(~finite).tolist()
            raise ValueError(f"non-finite statistics at rows {bad!r}")
        omitted = []
        for row in np.flatnonzero(~finite):
            record = {
                "row": int(row),
                "node1": frame.iloc[row]["node1"],
                "node2": frame.iloc[row]["node2"],
                "reason": "non-finite statistic",
            }
            if "edge_id" in frame:
                record["edge_id"] = frame.iloc[row]["edge_id"]
            omitted.append(record)
        frame = frame.loc[finite].copy()
    if not diagonal:
        diagonal_rows = frame["node1"] == frame["node2"]
        if diagonal_rows.any():
            raise ValueError("diagonal edges are not allowed when diagonal=False")
    frame = canonicalize_edges(frame, node_order=node_order, directed=directed)
    frame["statistic"] = pd.to_numeric(frame["statistic"]).astype(float)
    frame.attrs["omitted_edges"] = omitted
    frame.attrs["universe_size"] = len(frame)
    return frame.reset_index(drop=True)


def validate_connectome(
    matrix: np.ndarray,
    *,
    node_labels: Sequence[Any] | None = None,
    directed: bool = False,
    tolerance: float = 1e-12,
) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim not in {2, 3} or array.shape[-1] != array.shape[-2]:
        raise ValueError("connectome must have shape (nodes, nodes) or (subjects, nodes, nodes)")
    if node_labels is not None and len(_validate_node_order(node_labels)) != array.shape[-1]:
        raise ValueError("node_labels length must match matrix dimensions")
    if not np.isfinite(array).all():
        raise ValueError("connectome contains non-finite values")
    if not directed and not np.allclose(array, np.swapaxes(array, -1, -2), atol=tolerance, rtol=0):
        raise ValueError("undirected connectome must be symmetric")
    return array


def matrix_to_edges(
    matrix: np.ndarray,
    node_labels: Sequence[Any] | None = None,
    *,
    directed: bool = False,
    diagonal: bool = False,
    statistic_name: str = "statistic",
) -> pd.DataFrame:
    array = validate_connectome(matrix, node_labels=node_labels, directed=directed)
    n_nodes = array.shape[-1]
    labels = list(range(n_nodes)) if node_labels is None else list(node_labels)
    if directed:
        row, col = np.indices((n_nodes, n_nodes))
        mask = np.ones((n_nodes, n_nodes), dtype=bool)
        if not diagonal:
            mask &= row != col
        row, col = row[mask], col[mask]
    else:
        offset = 0 if diagonal else 1
        row, col = np.triu_indices(n_nodes, k=offset)
    label_array = _object_vector(labels)
    base = pd.DataFrame({"node1": label_array[row], "node2": label_array[col]})
    base = canonicalize_edges(base, node_order=labels, directed=directed)
    if array.ndim == 2:
        base[statistic_name] = array[row, col]
        if statistic_name != "statistic":
            base["statistic"] = base[statistic_name]
        return base
    chunks = []
    for subject, subject_matrix in enumerate(array):
        frame = base.copy()
        frame.insert(0, "subject", subject)
        frame[statistic_name] = subject_matrix[row, col]
        if statistic_name != "statistic":
            frame["statistic"] = frame[statistic_name]
        chunks.append(frame)
    output = pd.concat(chunks, ignore_index=True)
    output.attrs.update(base.attrs)
    return output


def edges_to_matrix(
    edges: pd.DataFrame,
    node_labels: Sequence[Any] | None = None,
    *,
    directed: bool = False,
    diagonal_value: float = 0.0,
    value_column: str = "statistic",
) -> np.ndarray:
    if value_column not in edges:
        raise ValueError(f"edge table does not contain value column {value_column!r}")
    value_frame = edges[["node1", "node2", value_column]].copy()
    if value_column != "statistic":
        value_frame = value_frame.rename(columns={value_column: "statistic"})
    frame = validate_edge_table(
        value_frame,
        node_order=node_labels,
        directed=directed,
        diagonal=True,
    )
    fallback_labels = [] if node_labels is None else list(node_labels)
    labels = frame.attrs.get("node_order", fallback_labels)
    index = {node: i for i, node in enumerate(labels)}
    matrix = np.full((len(labels), len(labels)), np.nan, dtype=float)
    np.fill_diagonal(matrix, diagonal_value)
    for row in frame.itertuples(index=False):
        i, j = index[row.node1], index[row.node2]
        value = float(row.statistic)
        matrix[i, j] = value
        if not directed:
            matrix[j, i] = value
    return matrix
