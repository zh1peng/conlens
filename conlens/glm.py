"""Edge-wise GLM estimation and Freedman--Lane null generation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import null_space

from .data import matrix_to_edges, validate_connectome
from .design import Contrast, DesignMatrix
from .results import EdgeStatistics
from .stats import glm_contrast_statistics


def _edge_matrix(
    connectomes: np.ndarray,
    node_labels: Sequence[Any] | None,
    *,
    directed: bool,
    diagonal: bool,
) -> tuple[np.ndarray, pd.DataFrame, list[Any]]:
    array = validate_connectome(connectomes, node_labels=node_labels, directed=directed)
    if array.ndim != 3:
        raise ValueError("connectomes must have shape (subjects, nodes, nodes)")
    long_edges = matrix_to_edges(
        array,
        node_labels=node_labels,
        directed=directed,
        diagonal=diagonal,
    )
    template = long_edges[long_edges["subject"] == 0][
        ["node1", "node2", "edge_id", "canonical_edge_id"]
    ].reset_index(drop=True)
    data = np.stack(
        [
            long_edges[long_edges["subject"] == subject]["statistic"].to_numpy(float)
            for subject in range(len(array))
        ]
    )
    labels = list(range(array.shape[1])) if node_labels is None else list(node_labels)
    template.attrs["node_order"] = labels
    return data, template, labels


def _validate_glm_inputs(
    data: np.ndarray,
    design: DesignMatrix,
    contrasts: Mapping[str, Contrast],
) -> None:
    if not isinstance(design, DesignMatrix):
        raise TypeError("design must be created with make_design()")
    if design.n_observations != len(data):
        raise ValueError("design must contain one row per subject")
    if not isinstance(contrasts, Mapping) or not contrasts:
        raise ValueError("contrasts must be a non-empty mapping")
    if any(not isinstance(name, str) or not name for name in contrasts):
        raise ValueError("contrast names must be non-empty strings")
    if any(not isinstance(item, Contrast) for item in contrasts.values()):
        raise TypeError("every contrast must be a Contrast object")


def _edge_result(
    template: pd.DataFrame,
    statistics,
    *,
    contrast_name: str,
    contrast: Contrast,
    contrast_vector: np.ndarray,
    design: DesignMatrix,
    node_order: list[Any],
    directed: bool,
    diagonal: bool,
    include_audit_columns: bool,
    extra_metadata: Mapping[str, Any] | None = None,
) -> EdgeStatistics:
    table = template.copy()
    table["statistic"] = statistics.effect_size
    if include_audit_columns:
        table["effect_size"] = statistics.effect_size
        table["contrast_estimate"] = statistics.contrast_estimate
        table["standard_error"] = statistics.standard_error
        table["t_statistic"] = statistics.t_statistic
        table["residual_df"] = statistics.residual_df
        table["edge_p_value_two_sided"] = statistics.p_value_two_sided
        table["residual_sd"] = statistics.residual_sd
    table.attrs["node_order"] = node_order
    statistic_name = (
        "partial correlation"
        if contrast.effect_size == "partial_r"
        else "model-adjusted Hedges' g"
    )
    metadata = {
        "contrast_name": contrast_name,
        "contrast_vector": contrast_vector.tolist(),
        "effect_size": contrast.effect_size,
        "statistic_name": statistic_name,
        "positive_direction": contrast.positive_direction,
        "residual_df": statistics.residual_df,
        "design": design.metadata(),
        "analysis_signature": {
            "kind": "glm_contrast",
            "design": design.signature(),
            "contrast_name": contrast_name,
            "contrast_vector": contrast_vector.tolist(),
            "effect_size": contrast.effect_size,
        },
        "node_order": node_order,
        "directed": directed,
        "diagonal": diagonal,
    }
    metadata.update(extra_metadata or {})
    return EdgeStatistics(table=table, metadata=metadata)


def lens_glm(
    connectomes: np.ndarray,
    *,
    design: DesignMatrix,
    contrasts: Mapping[str, Contrast],
    node_labels: Sequence[Any] | None = None,
    directed: bool = False,
    diagonal: bool = False,
) -> dict[str, EdgeStatistics]:
    """Fit all named contrasts and return observed signed edge statistics."""
    data, template, node_order = _edge_matrix(
        connectomes,
        node_labels,
        directed=directed,
        diagonal=diagonal,
    )
    _validate_glm_inputs(data, design, contrasts)
    output: dict[str, EdgeStatistics] = {}
    for name, contrast in contrasts.items():
        vector = contrast.resolve(design)
        statistics = glm_contrast_statistics(
            data,
            design.values,
            vector,
            effect_size=contrast.effect_size,
        )
        output[name] = _edge_result(
            template,
            statistics,
            contrast_name=name,
            contrast=contrast,
            contrast_vector=vector,
            design=design,
            node_order=node_order,
            directed=directed,
            diagonal=diagonal,
            include_audit_columns=True,
            extra_metadata={"source": "observed"},
        )
    return output


def _contains_missing(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_contains_missing(item) for item in value)
    return bool(np.asarray(pd.isna(value), dtype=bool).any())


def _validate_blocks(blocks: Iterable[Any] | None, n_subjects: int) -> np.ndarray | None:
    if blocks is None:
        return None
    values = list(blocks)
    if len(values) != n_subjects:
        raise ValueError("exchangeability_blocks must contain one value per subject")
    if any(_contains_missing(value) for value in values):
        raise ValueError("exchangeability_blocks cannot contain missing values")
    try:
        codes, _ = pd.factorize(pd.Series(values, dtype=object), sort=False)
    except TypeError as exc:
        raise ValueError("exchangeability_blocks values must be hashable") from exc
    return np.asarray(codes, dtype=int)


def _permutation_indices(
    n_subjects: int,
    rng: np.random.Generator,
    block_codes: np.ndarray | None,
) -> np.ndarray:
    if block_codes is None:
        return rng.permutation(n_subjects)
    indices = np.arange(n_subjects)
    output = indices.copy()
    for code in np.unique(block_codes):
        members = indices[block_codes == code]
        output[members] = rng.permutation(members)
    return output


def lens_fl_permute(
    connectomes: np.ndarray,
    *,
    design: DesignMatrix,
    contrasts: Mapping[str, Contrast],
    n_permutations: int,
    node_labels: Sequence[Any] | None = None,
    directed: bool = False,
    diagonal: bool = False,
    exchangeability_blocks: Iterable[Any] | None = None,
    random_state: int | None = None,
) -> Iterator[dict[str, EdgeStatistics]]:
    """Yield Freedman--Lane null edge statistics without retaining an edge-null matrix."""
    if not isinstance(n_permutations, int) or n_permutations < 1:
        raise ValueError("n_permutations must be a positive integer")
    data, template, node_order = _edge_matrix(
        connectomes,
        node_labels,
        directed=directed,
        diagonal=diagonal,
    )
    _validate_glm_inputs(data, design, contrasts)
    block_codes = _validate_blocks(exchangeability_blocks, len(data))
    x = design.values
    prepared: dict[str, tuple[Contrast, np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, contrast in contrasts.items():
        vector = contrast.resolve(design)
        reduced_basis = null_space(vector.reshape(1, -1))
        reduced_design = x @ reduced_basis
        reduced_beta = np.linalg.pinv(reduced_design) @ data
        fitted = reduced_design @ reduced_beta
        prepared[name] = (contrast, vector, fitted, data - fitted)

    rng = np.random.default_rng(random_state)
    for replicate in range(n_permutations):
        indices = _permutation_indices(len(data), rng, block_codes)
        output: dict[str, EdgeStatistics] = {}
        for name, (contrast, vector, fitted, residuals) in prepared.items():
            permuted = fitted + residuals[indices]
            statistics = glm_contrast_statistics(
                permuted,
                x,
                vector,
                effect_size=contrast.effect_size,
            )
            output[name] = _edge_result(
                template,
                statistics,
                contrast_name=name,
                contrast=contrast,
                contrast_vector=vector,
                design=design,
                node_order=node_order,
                directed=directed,
                diagonal=diagonal,
                include_audit_columns=False,
                extra_metadata={
                    "source": "freedman_lane_null",
                    "permutation_scheme": "contrast_specific_freedman_lane",
                    "permutation_index": replicate,
                    "random_seed": random_state,
                    "exchangeability_blocks_used": block_codes is not None,
                },
            )
        yield output

