"""Optional Nilearn adapters; importing core conlens never imports Nilearn."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..data import matrix_to_edges


def from_nilearn_connectivity(
    connectomes: np.ndarray,
    labels: Sequence[Any],
    *,
    directed: bool = False,
    coordinates: np.ndarray | None = None,
    node_metadata: pd.DataFrame | None = None,
):
    """Convert ConnectivityMeasure output and retain atlas metadata."""
    label_list = list(labels)
    if coordinates is not None:
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.shape != (len(label_list), 3) or not np.isfinite(coordinates).all():
            raise ValueError("coordinates must be a finite node-by-3 array")
    if node_metadata is not None:
        if len(node_metadata) != len(label_list):
            raise ValueError("node_metadata must contain one row per label")
        if "node_id" in node_metadata and node_metadata["node_id"].tolist() != label_list:
            raise ValueError("node_metadata node_id order must match labels")
    output = matrix_to_edges(connectomes, label_list, directed=directed)
    output.attrs["atlas_labels"] = label_list
    output.attrs["node_coordinates"] = None if coordinates is None else coordinates.tolist()
    output.attrs["node_metadata"] = (
        None if node_metadata is None else node_metadata.to_dict(orient="records")
    )
    return output


def plot_nilearn_connectome(adjacency, coordinates, **kwargs):
    try:
        from nilearn import plotting
    except ImportError as exc:
        raise ImportError("install conlens[nilearn] to use Nilearn plotting") from exc
    return plotting.plot_connectome(adjacency, coordinates, **kwargs)
