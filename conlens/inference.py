"""Explicit null models, empirical inference, normalization, and FDR."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .core import compute_enrichment_score, compute_running_sum, rank_edges
from .results import LensResult
from .stats import glm_statistic, two_group_ttest


def adjust_pvalues(pvalues: Iterable[float], method: str = "BH") -> np.ndarray:
    values = np.asarray(list(pvalues), dtype=float)
    if method.upper() != "BH":
        raise ValueError("only Benjamini-Hochberg ('BH') is supported")
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("pvalues must be finite values in [0, 1]")
    if len(values) == 0:
        return values
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def empirical_pvalue(observed: float, null_scores: Iterable[float]) -> dict[str, Any]:
    null = np.asarray(list(null_scores), dtype=float)
    if null.ndim != 1 or len(null) == 0 or not np.isfinite(null).all():
        raise ValueError("null_scores must be a non-empty finite one-dimensional array")
    positive = int(np.sum(null >= 0))
    negative = int(np.sum(null <= 0))
    if observed > 0:
        extreme = int(np.sum(null >= observed))
        p_value = (1 + extreme) / (1 + positive)
        minimum_resolvable = 1 / (1 + positive)
    elif observed < 0:
        extreme = int(np.sum(null <= observed))
        p_value = (1 + extreme) / (1 + negative)
        minimum_resolvable = 1 / (1 + negative)
    else:
        extreme, p_value = 0, 1.0
        minimum_resolvable = 1.0
    return {
        "p_value": float(p_value),
        "n_more_extreme": extreme,
        "n_null_positive": positive,
        "n_null_negative": negative,
        "n_permutations": len(null),
        "minimum_resolvable_p": float(minimum_resolvable),
        "p_value_method": "sign_specific_empirical_add_one",
    }


def normalize_enrichment_scores(
    observed: float, null_scores: Iterable[float]
) -> tuple[float | None, str]:
    null = np.asarray(list(null_scores), dtype=float)
    if null.ndim != 1 or len(null) == 0 or not np.isfinite(null).all():
        raise ValueError("null_scores must be a non-empty finite one-dimensional array")
    if observed == 0:
        return 0.0, "defined"
    directional = null[null >= 0] if observed > 0 else np.abs(null[null <= 0])
    if len(directional) == 0:
        return None, "undefined"
    scale = float(np.mean(directional))
    if scale == 0:
        return None, "undefined"
    return float(observed / scale), "defined"


def _score_sets(
    edge_ids: np.ndarray,
    statistics: np.ndarray,
    edge_sets: Mapping[str, set[str]],
    *,
    weight: float,
    score_type: str,
    canonical_edge_ids: np.ndarray | None = None,
) -> dict[str, float]:
    frame = pd.DataFrame({"edge_id": edge_ids, "statistic": statistics})
    if canonical_edge_ids is not None:
        if len(canonical_edge_ids) != len(edge_ids):
            raise ValueError("canonical_edge_ids must match edge_ids")
        frame["canonical_edge_id"] = canonical_edge_ids
    ranked, _ = rank_edges(frame)
    output: dict[str, float] = {}
    for name, members in edge_sets.items():
        hits = ranked["edge_id"].isin(members)
        profile, _ = compute_running_sum(ranked["statistic"], hits, weight=weight)
        output[name] = compute_enrichment_score(profile, score_type=score_type)["ES"]
    return output


def edge_permutation_null(
    edges: pd.DataFrame,
    edge_sets: Mapping[str, set[str]],
    *,
    n_permutations: int = 1000,
    random_state: int | None = None,
    weight: float = 1.0,
    score_type: str = "standard",
    n_jobs: int = 1,
) -> dict[str, np.ndarray]:
    """Globally permute edge-ID/statistic correspondence once per replicate."""
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")
    seed_sequence = np.random.SeedSequence(random_state)
    seeds = seed_sequence.spawn(n_permutations)
    identifiers = edges["edge_id"].astype(str).to_numpy()
    canonical_identifiers = (
        edges["canonical_edge_id"].astype(str).to_numpy()
        if "canonical_edge_id" in edges
        else identifiers
    )
    values = edges["statistic"].to_numpy(float)

    def one(seed: np.random.SeedSequence) -> dict[str, float]:
        permutation = np.random.default_rng(seed).permutation(len(values))
        return _score_sets(
            identifiers,
            values[permutation],
            edge_sets,
            weight=weight,
            score_type=score_type,
            canonical_edge_ids=canonical_identifiers,
        )

    if n_jobs == 1:
        replicates = [one(seed) for seed in seeds]
    else:
        from joblib import Parallel, delayed

        replicates = Parallel(n_jobs=n_jobs)(delayed(one)(seed) for seed in seeds)
    return {name: np.asarray([replicate[name] for replicate in replicates]) for name in edge_sets}


def _permutation_indices(
    n_observations: int,
    rng: np.random.Generator,
    blocks: Iterable[Any] | None,
) -> np.ndarray:
    if blocks is None:
        return rng.permutation(n_observations)
    block_values = np.asarray(list(blocks))
    if block_values.ndim != 1 or len(block_values) != n_observations:
        raise ValueError("exchangeability_blocks must contain one value per observation")
    indices = np.arange(n_observations)
    output = indices.copy()
    for block in np.unique(block_values):
        members = indices[block_values == block]
        output[members] = rng.permutation(members)
    return output


def label_permutation_null(
    data: np.ndarray,
    labels: Iterable[Any],
    edge_ids: Iterable[str],
    edge_sets: Mapping[str, set[str]],
    *,
    n_permutations: int = 1000,
    random_state: int | None = None,
    exchangeability_blocks: Iterable[Any] | None = None,
    weight: float = 1.0,
    score_type: str = "standard",
    statistic_function: Callable[[np.ndarray, np.ndarray], np.ndarray] = two_group_ttest,
    canonical_edge_ids: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    values = np.asarray(data, dtype=float)
    group_labels = np.asarray(list(labels))
    identifiers = np.asarray(list(edge_ids), dtype=str)
    canonical_identifiers = (
        None if canonical_edge_ids is None else np.asarray(list(canonical_edge_ids), dtype=str)
    )
    if values.ndim != 2 or values.shape != (len(group_labels), len(identifiers)):
        raise ValueError("data shape must match labels and edge_ids")
    if not np.isfinite(values).all():
        raise ValueError("subject-level data must be finite")
    rng = np.random.default_rng(random_state)
    output = {name: np.empty(n_permutations) for name in edge_sets}
    for replicate in range(n_permutations):
        indices = _permutation_indices(len(values), rng, exchangeability_blocks)
        statistics = np.asarray(statistic_function(values, group_labels[indices]), dtype=float)
        if statistics.shape != (len(identifiers),) or not np.isfinite(statistics).all():
            raise ValueError("statistic_function must return one finite statistic per edge")
        scores = _score_sets(
            identifiers,
            statistics,
            edge_sets,
            weight=weight,
            score_type=score_type,
            canonical_edge_ids=canonical_identifiers,
        )
        for name, score in scores.items():
            output[name][replicate] = score
    return output


def freedman_lane_null(
    data: np.ndarray,
    tested_design: np.ndarray,
    nuisance_design: np.ndarray,
    edge_ids: Iterable[str],
    edge_sets: Mapping[str, set[str]],
    *,
    contrast: np.ndarray | None = None,
    n_permutations: int = 1000,
    random_state: int | None = None,
    exchangeability_blocks: Iterable[Any] | None = None,
    weight: float = 1.0,
    score_type: str = "standard",
    canonical_edge_ids: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    y = np.asarray(data, dtype=float)
    tested = np.asarray(tested_design, dtype=float)
    nuisance = np.asarray(nuisance_design, dtype=float)
    identifiers = np.asarray(list(edge_ids), dtype=str)
    canonical_identifiers = (
        None if canonical_edge_ids is None else np.asarray(list(canonical_edge_ids), dtype=str)
    )
    if tested.ndim == 1:
        tested = tested[:, None]
    if nuisance.ndim == 1:
        nuisance = nuisance[:, None]
    if y.ndim != 2 or len(y) != len(tested) or len(y) != len(nuisance):
        raise ValueError("data and design matrices must have matching observation counts")
    if y.shape[1] != len(identifiers) or not np.isfinite(y).all():
        raise ValueError("data columns must match finite edge_ids")
    if not np.any(np.all(np.isclose(nuisance, 1.0), axis=0)):
        raise ValueError("nuisance_design must explicitly contain an intercept column")
    reduced_beta = np.linalg.pinv(nuisance) @ y
    fitted_reduced = nuisance @ reduced_beta
    residuals = y - fitted_reduced
    full_design = np.column_stack([tested, nuisance])
    tested_contrast = np.zeros(full_design.shape[1])
    if contrast is None:
        tested_contrast[0] = 1.0
    else:
        supplied = np.asarray(contrast, dtype=float).reshape(-1)
        if len(supplied) == tested.shape[1]:
            tested_contrast[: tested.shape[1]] = supplied
        elif len(supplied) == full_design.shape[1]:
            tested_contrast = supplied
        else:
            raise ValueError("contrast must match tested or full design columns")
    rng = np.random.default_rng(random_state)
    output = {name: np.empty(n_permutations) for name in edge_sets}
    for replicate in range(n_permutations):
        indices = _permutation_indices(len(y), rng, exchangeability_blocks)
        permuted = fitted_reduced + residuals[indices]
        statistics = glm_statistic(permuted, full_design, tested_contrast)
        scores = _score_sets(
            identifiers,
            statistics,
            edge_sets,
            weight=weight,
            score_type=score_type,
            canonical_edge_ids=canonical_identifiers,
        )
        for name, score in scores.items():
            output[name][replicate] = score
    return output


def provided_null(
    null_data: Mapping[str, Iterable[float]] | np.ndarray,
    *,
    kind: str = "es",
    edge_ids: Iterable[str] | None = None,
    edge_sets: Mapping[str, set[str]] | None = None,
    weight: float = 1.0,
    score_type: str = "standard",
    canonical_edge_ids: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    """Validate and score supplied ES, statistic-matrix, or rank-matrix null data.

    Rank matrices contain one rank per observed-order edge in every row. They are
    sufficient only for unweighted enrichment because ranks do not encode the
    statistic magnitudes required by weighted hit increments.
    """
    if kind not in {"es", "statistics", "ranks"}:
        raise ValueError("provided null kind must be 'es', 'statistics', or 'ranks'")
    if kind == "es":
        if not isinstance(null_data, Mapping):
            raise ValueError("kind='es' requires a mapping from set name to null scores")
        arrays = {name: np.asarray(list(scores), dtype=float) for name, scores in null_data.items()}
        lengths = {len(scores) for scores in arrays.values()}
        if (
            len(lengths) != 1
            or not lengths
            or 0 in lengths
            or any(not np.isfinite(scores).all() for scores in arrays.values())
        ):
            raise ValueError("provided null arrays must be finite and have equal replicate counts")
        return arrays
    if edge_ids is None or edge_sets is None:
        raise ValueError("edge_ids and edge_sets are required for provided matrix nulls")
    identifiers = np.asarray(list(edge_ids), dtype=str)
    canonical_identifiers = (
        identifiers
        if canonical_edge_ids is None
        else np.asarray(list(canonical_edge_ids), dtype=str)
    )
    matrix = np.asarray(null_data)
    if matrix.ndim != 2 or matrix.shape[1] != len(identifiers) or matrix.shape[0] == 0:
        raise ValueError("provided null matrix must have shape (replicates, observed edges)")
    if kind == "ranks":
        if weight != 0:
            raise ValueError("provided rank matrices require weight=0")
        if not np.issubdtype(matrix.dtype, np.integer):
            raise ValueError("provided rank matrices must contain integer ranks")
        expected_zero = np.arange(len(identifiers))
        expected_one = np.arange(1, len(identifiers) + 1)
        for row in matrix:
            ordered = np.sort(row)
            if not (
                np.array_equal(ordered, expected_zero) or np.array_equal(ordered, expected_one)
            ):
                raise ValueError("every provided rank row must be a complete rank permutation")
        offset = 0 if np.min(matrix) == 0 else 1
        statistic_matrix = -(matrix.astype(float) - offset)
    else:
        statistic_matrix = np.asarray(matrix, dtype=float)
        if not np.isfinite(statistic_matrix).all():
            raise ValueError("provided statistic matrices must be finite")
    replicates = [
        _score_sets(
            identifiers,
            row,
            edge_sets,
            weight=weight,
            score_type=score_type,
            canonical_edge_ids=canonical_identifiers,
        )
        for row in statistic_matrix
    ]
    return {name: np.asarray([replicate[name] for replicate in replicates]) for name in edge_sets}


def apply_null_inference(
    result: LensResult,
    null_scores: Mapping[str, Iterable[float]],
    *,
    correction_family_id: str = "default",
) -> LensResult:
    tested = []
    for set_result in result.sets:
        if set_result.status != "ok":
            continue
        if set_result.set_name not in null_scores:
            raise ValueError(f"missing null scores for {set_result.set_name!r}")
        if set_result.ES is None:
            raise ValueError(f"valid set {set_result.set_name!r} has no observed ES")
        observed = set_result.ES
        null = np.asarray(list(null_scores[set_result.set_name]), dtype=float)
        inference = empirical_pvalue(observed, null)
        for key, value in inference.items():
            setattr(set_result, key, value)
        set_result.NES, set_result.normalization_status = normalize_enrichment_scores(
            observed, null
        )
        tested.append(set_result)
    pvalues = []
    for item in tested:
        if item.p_value is None:
            raise ArithmeticError(f"inference produced no P value for {item.set_name!r}")
        pvalues.append(item.p_value)
    qvalues = adjust_pvalues(pvalues)
    for item, qvalue in zip(tested, qvalues, strict=True):
        item.q_value = float(qvalue)
    result.metadata.update(
        {
            "adjustment_method": "BH",
            "n_sets_tested": len(tested),
            "correction_family_id": correction_family_id,
        }
    )
    return result


permutation_test = edge_permutation_null
