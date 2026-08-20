"""Bootstrap-derived stability summaries without inventing subject-level data."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta

from .core import lens_enrich
from .results import LensResult, LensStabilityResult

SET_GATE_THRESHOLD = 0.50

SET_SUMMARY_COLUMNS = [
    "set_name",
    "observed_direction",
    "observed_es",
    "observed_nes",
    "observed_p_value",
    "observed_q_value",
    "detection_count",
    "same_direction_count",
    "different_direction_count",
    "detection_rate",
    "direction_consistency",
    "set_stability",
    "set_stability_lower",
    "set_stability_upper",
    "set_reproducibility_supported",
    "observed_leading_edge_size",
    "bootstrap_leading_edge_size_median",
    "bootstrap_leading_edge_size_lower",
    "bootstrap_leading_edge_size_upper",
    "median_jaccard_with_observed",
    "conditional_localization_supported",
    "conditional_core_reportable",
    "conditional_status",
    "full_pipeline_core_size",
    "conditional_core_size",
]

EDGE_SUMMARY_COLUMNS = [
    "set_name",
    "edge_id",
    "node1",
    "node2",
    "in_observed_leading_edge",
    "same_direction_inclusion_count",
    "conditional_stability",
    "conditional_stability_lower",
    "conditional_stability_upper",
    "full_pipeline_stability",
    "full_pipeline_stability_lower",
    "full_pipeline_stability_upper",
    "conditional_core",
    "full_pipeline_core",
]

REPLICATE_SUMMARY_COLUMNS = [
    "replicate",
    "set_name",
    "detected",
    "same_direction",
    "direction",
    "es",
    "nes",
    "p_value",
    "q_value",
    "leading_edge_size",
    "jaccard_with_observed",
]


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
    sampling_rng = None
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
        sampling_rng = np.random.default_rng(random_state)
        generated = []
        all_indices = np.arange(len(values))
        for _ in range(n_bootstraps):
            if strata_values is None:
                indices = sampling_rng.choice(all_indices, size=len(values), replace=True)
            else:
                samples = []
                for label in pd.unique(strata_values):
                    members = all_indices[strata_values == label]
                    samples.append(
                        sampling_rng.choice(members, size=len(members), replace=True)
                    )
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
    inference_seeds = None
    if lens_options.get("null_method") is not None:
        inference_rng = (
            sampling_rng if sampling_rng is not None else np.random.default_rng(random_state)
        )
        inference_seeds = inference_rng.integers(
            0, 2**32, size=len(replicates), dtype=np.uint64
        )
    output = []
    for replicate_index, values in enumerate(replicates):
        frame = edges.copy()
        frame["statistic"] = values
        parameters = lens_options.copy()
        if inference_seeds is not None:
            parameters["random_state"] = int(inference_seeds[replicate_index])
        output.append(lens_enrich(frame, edge_sets, **parameters))
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
                        item.q_value is not None and item.q_value <= significance_alpha
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


def _jeffreys_interval(
    successes: int | np.ndarray,
    total: int,
    interval_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(successes, dtype=float)
    tail = (1.0 - interval_level) / 2.0
    return (
        np.asarray(beta.ppf(tail, values + 0.5, total - values + 0.5), dtype=float),
        np.asarray(beta.ppf(1.0 - tail, values + 0.5, total - values + 0.5), dtype=float),
    )


def _is_empty_metadata(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return not value
    return False


def _validate_complete_result(result: LensResult, label: str) -> dict[str, Any]:
    if not isinstance(result, LensResult):
        raise TypeError(f"{label} must be a LensResult")
    if result.metadata.get("adjustment_method") != "BH":
        raise ValueError(f"{label} must contain completed BH-adjusted inference")
    required_metadata = (
        "edge_universe_hash",
        "correction_family_id",
        "node_order",
        "directed",
        "diagonal",
        "ranking_statistic_name",
        "analysis_signature",
        "positive_direction",
        "null_method",
        "permutation_scheme",
        "exchangeability_blocks_summary",
        "n_permutations",
        "n_sets_tested",
    )
    missing_metadata = [field for field in required_metadata if field not in result.metadata]
    if missing_metadata:
        raise ValueError(f"{label} is missing required metadata: {missing_metadata}")
    nonempty_metadata = (
        "edge_universe_hash",
        "correction_family_id",
        "node_order",
        "ranking_statistic_name",
        "analysis_signature",
        "positive_direction",
        "null_method",
        "permutation_scheme",
    )
    if any(_is_empty_metadata(result.metadata[field]) for field in nonempty_metadata):
        raise ValueError(f"{label} contains incomplete analysis metadata")
    if result.ranked_edges is None:
        raise ValueError(f"{label} must retain ranked_edges")
    required_edge_columns = {"edge_id", "canonical_edge_id", "node1", "node2"}
    if not required_edge_columns.issubset(result.ranked_edges.columns):
        raise ValueError(
            f"{label} ranked_edges must contain edge_id, canonical_edge_id, node1, and node2"
        )
    if result.ranked_edges["edge_id"].duplicated().any():
        raise ValueError(f"{label} ranked_edges must contain unique edge IDs")
    names = [item.set_name for item in result.sets]
    if len(names) != len(set(names)):
        raise ValueError(f"{label} contains duplicate set names")
    output = {item.set_name: item for item in result.sets}
    for item in result.sets:
        if len(item.edge_set_ids) != len(set(item.edge_set_ids)):
            raise ValueError(f"{label} contains duplicate members in set {item.set_name!r}")
        if len(item.leading_edge_ids) != len(set(item.leading_edge_ids)):
            raise ValueError(f"{label} contains duplicate leading edges for set {item.set_name!r}")
        if item.leading_edge_size != len(item.leading_edge_ids):
            raise ValueError(f"{label} leading-edge size is inconsistent for set {item.set_name!r}")
        members = set(item.edge_set_ids)
        if not set(item.leading_edge_ids).issubset(members):
            raise ValueError(
                f"{label} leading edges are not members of set {item.set_name!r}"
            )
        if item.status != "ok":
            continue
        if item.q_value is None or not np.isfinite(item.q_value):
            raise ValueError(f"{label} has no finite q value for set {item.set_name!r}")
        if not 0.0 <= item.q_value <= 1.0:
            raise ValueError(f"{label} q value is outside [0, 1] for set {item.set_name!r}")
    return output


def _validate_compatible_result(
    observed: LensResult,
    observed_sets: Mapping[str, Any],
    replicate: LensResult,
    replicate_index: int,
) -> dict[str, Any]:
    label = f"bootstrap result {replicate_index}"
    replicate_sets = _validate_complete_result(replicate, label)
    if set(replicate_sets) != set(observed_sets):
        raise ValueError(f"{label} does not contain the same set names as observed")
    compatibility_fields = (
        "edge_universe_hash",
        "correction_family_id",
        "node_order",
        "directed",
        "diagonal",
        "ranking_statistic_name",
        "analysis_signature",
        "positive_direction",
        "weight_exponent",
        "score_type",
        "set_size_filters",
        "null_method",
        "permutation_scheme",
        "n_permutations",
        "n_sets_tested",
    )
    for field in compatibility_fields:
        if replicate.metadata.get(field) != observed.metadata.get(field):
            raise ValueError(f"{label} metadata field {field!r} differs from observed")
    observed_uses_blocks = observed.metadata["exchangeability_blocks_summary"] is not None
    replicate_uses_blocks = replicate.metadata["exchangeability_blocks_summary"] is not None
    if replicate_uses_blocks != observed_uses_blocks:
        raise ValueError(
            f"{label} metadata field 'exchangeability_blocks_used' differs from observed"
        )
    observed_ranked_edges = observed.ranked_edges
    replicate_ranked_edges = replicate.ranked_edges
    if observed_ranked_edges is None or replicate_ranked_edges is None:
        raise ValueError(f"{label} and observed result must retain ranked_edges")
    observed_edge_mapping = dict(
        zip(
            observed_ranked_edges["edge_id"],
            observed_ranked_edges["canonical_edge_id"],
            strict=True,
        )
    )
    replicate_edge_mapping = dict(
        zip(
            replicate_ranked_edges["edge_id"],
            replicate_ranked_edges["canonical_edge_id"],
            strict=True,
        )
    )
    if replicate_edge_mapping != observed_edge_mapping:
        raise ValueError(f"{label} edge ID mapping differs from observed")
    if observed.metadata.get("null_method") == "provided_null":
        for field in ("provided_null_kind", "provided_null_direction"):
            if replicate.metadata.get(field) != observed.metadata.get(field):
                raise ValueError(f"{label} metadata field {field!r} differs from observed")
    for name, observed_set in observed_sets.items():
        replicate_set = replicate_sets[name]
        if replicate_set.status != observed_set.status:
            raise ValueError(f"{label} status differs for set {name!r}")
        if set(replicate_set.edge_set_ids) != set(observed_set.edge_set_ids):
            raise ValueError(f"{label} definition differs for set {name!r}")
    return replicate_sets


def _validate_stability_options(
    significance_alpha: float,
    interval_level: float,
    core_threshold: float,
    min_same_direction: int,
) -> None:
    if not 0.0 < significance_alpha < 1.0:
        raise ValueError("significance_alpha must be in (0, 1)")
    if not 0.0 < interval_level < 1.0:
        raise ValueError("interval_level must be in (0, 1)")
    if not 0.0 < core_threshold < 1.0:
        raise ValueError("core_threshold must be in (0, 1)")
    if not isinstance(min_same_direction, int) or min_same_direction < 1:
        raise ValueError("min_same_direction must be a positive integer")


def summarize_bootstrap_stability(
    observed: LensResult,
    bootstrap_results: Iterable[LensResult],
    *,
    significance_alpha: float = 0.05,
    interval_level: float = 0.95,
    core_threshold: float = 0.50,
    min_same_direction: int = 30,
    keep_bootstrap_results: bool = False,
) -> LensStabilityResult:
    """Summarize full-pipeline bootstrap stability around an observed result.

    Every bootstrap result must repeat inference and BH adjustment over the same
    correction family. Only a set that is BH-significant in the observed result is
    tracked. Edge inclusion counts only when that set is again BH-significant with
    the observed direction.

    Reported intervals are Monte Carlo intervals for empirical bootstrap
    frequencies. They are not confidence intervals for edge truth or future-study
    replication probabilities.
    """
    _validate_stability_options(
        significance_alpha,
        interval_level,
        core_threshold,
        min_same_direction,
    )

    observed_sets = _validate_complete_result(observed, "observed result")
    replicates = list(bootstrap_results)
    if not replicates:
        raise ValueError("bootstrap_results must contain at least one LensResult")
    replicate_sets = [
        _validate_compatible_result(observed, observed_sets, result, index)
        for index, result in enumerate(replicates)
    ]

    if observed.ranked_edges is None:
        raise ValueError("observed ranked_edges are required for edge-level stability")
    required_edge_columns = {"edge_id", "node1", "node2"}
    if not required_edge_columns.issubset(observed.ranked_edges.columns):
        raise ValueError("observed ranked_edges must contain edge_id, node1, and node2")
    edge_lookup = observed.ranked_edges.set_index("edge_id", drop=False)
    if not edge_lookup.index.is_unique:
        raise ValueError("observed ranked_edges must contain unique edge IDs")

    tracked_names: list[str] = []
    for item in observed.sets:
        if item.status != "ok" or item.q_value is None or item.q_value > significance_alpha:
            continue
        if item.direction not in {"positive", "negative"}:
            raise ValueError(
                f"observed significant set {item.set_name!r} has no positive/negative direction"
            )
        tracked_names.append(item.set_name)

    set_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    n_bootstraps = len(replicates)
    quantile_tail = (1.0 - interval_level) / 2.0

    for set_name in tracked_names:
        observed_set = observed_sets[set_name]
        observed_direction = observed_set.direction
        observed_leading = set(observed_set.leading_edge_ids)
        member_ids = sorted(set(observed_set.edge_set_ids))
        missing_members = set(member_ids) - set(edge_lookup.index)
        if missing_members:
            raise ValueError(f"observed ranked_edges are missing members of set {set_name!r}")
        member_index = {edge_id: index for index, edge_id in enumerate(member_ids)}
        inclusion_counts = np.zeros(len(member_ids), dtype=int)
        detection_count = 0
        same_direction_count = 0
        different_direction_count = 0
        same_direction_sizes: list[int] = []
        same_direction_jaccard: list[float] = []

        for replicate_index, sets_by_name in enumerate(replicate_sets):
            item = sets_by_name[set_name]
            detected = bool(item.q_value is not None and item.q_value <= significance_alpha)
            if detected and item.direction not in {"positive", "negative"}:
                raise ValueError(
                    f"bootstrap result {replicate_index} detected set {set_name!r} "
                    "without a positive/negative direction"
                )
            same_direction = bool(detected and item.direction == observed_direction)
            if detected:
                detection_count += 1
                if same_direction:
                    same_direction_count += 1
                else:
                    different_direction_count += 1
            replicate_leading = set(item.leading_edge_ids)
            jaccard_with_observed = np.nan
            if same_direction:
                for edge_id in replicate_leading:
                    inclusion_counts[member_index[edge_id]] += 1
                same_direction_sizes.append(len(replicate_leading))
                jaccard_with_observed = _jaccard(observed_leading, replicate_leading)
                same_direction_jaccard.append(jaccard_with_observed)
            replicate_rows.append(
                {
                    "replicate": replicate_index,
                    "set_name": set_name,
                    "detected": detected,
                    "same_direction": same_direction,
                    "direction": item.direction,
                    "es": item.ES,
                    "nes": item.NES,
                    "p_value": item.p_value,
                    "q_value": item.q_value,
                    "leading_edge_size": item.leading_edge_size,
                    "jaccard_with_observed": jaccard_with_observed,
                }
            )

        detection_rate = detection_count / n_bootstraps
        direction_consistency = (
            same_direction_count / detection_count if detection_count else np.nan
        )
        set_stability = same_direction_count / n_bootstraps
        set_lower, set_upper = _jeffreys_interval(
            same_direction_count, n_bootstraps, interval_level
        )
        full_stability = inclusion_counts / n_bootstraps
        full_lower, full_upper = _jeffreys_interval(
            inclusion_counts, n_bootstraps, interval_level
        )
        if same_direction_count:
            conditional_stability = inclusion_counts / same_direction_count
            conditional_lower, conditional_upper = _jeffreys_interval(
                inclusion_counts, same_direction_count, interval_level
            )
            if not np.allclose(
                full_stability,
                set_stability * conditional_stability,
                atol=1e-15,
                rtol=0,
            ):
                raise ArithmeticError("full-pipeline stability identity failed")
        else:
            conditional_stability = np.full(len(member_ids), np.nan)
            conditional_lower = np.full(len(member_ids), np.nan)
            conditional_upper = np.full(len(member_ids), np.nan)

        conditional_localization_supported = same_direction_count >= min_same_direction
        set_reproducibility_supported = bool(set_lower > SET_GATE_THRESHOLD)
        conditional_core_reportable = bool(
            conditional_localization_supported and set_reproducibility_supported
        )
        full_pipeline_core = full_lower > core_threshold
        conditional_core = (
            (conditional_lower > core_threshold)
            if conditional_core_reportable
            else np.zeros(len(member_ids), dtype=bool)
        )
        if not conditional_localization_supported:
            conditional_status = "insufficient same-direction detections"
        elif not set_reproducibility_supported:
            conditional_status = "set stability below gate"
        else:
            conditional_status = "reportable"

        if same_direction_sizes:
            size_lower, size_upper = np.quantile(
                same_direction_sizes, [quantile_tail, 1.0 - quantile_tail]
            )
            size_median = float(np.median(same_direction_sizes))
        else:
            size_lower = size_upper = size_median = np.nan
        set_rows.append(
            {
                "set_name": set_name,
                "observed_direction": observed_direction,
                "observed_es": observed_set.ES,
                "observed_nes": observed_set.NES,
                "observed_p_value": observed_set.p_value,
                "observed_q_value": observed_set.q_value,
                "detection_count": detection_count,
                "same_direction_count": same_direction_count,
                "different_direction_count": different_direction_count,
                "detection_rate": detection_rate,
                "direction_consistency": direction_consistency,
                "set_stability": set_stability,
                "set_stability_lower": float(set_lower),
                "set_stability_upper": float(set_upper),
                "set_reproducibility_supported": set_reproducibility_supported,
                "observed_leading_edge_size": observed_set.leading_edge_size,
                "bootstrap_leading_edge_size_median": size_median,
                "bootstrap_leading_edge_size_lower": float(size_lower),
                "bootstrap_leading_edge_size_upper": float(size_upper),
                "median_jaccard_with_observed": float(np.median(same_direction_jaccard))
                if same_direction_jaccard
                else np.nan,
                "conditional_localization_supported": conditional_localization_supported,
                "conditional_core_reportable": conditional_core_reportable,
                "conditional_status": conditional_status,
                "full_pipeline_core_size": int(full_pipeline_core.sum()),
                "conditional_core_size": int(conditional_core.sum()),
            }
        )

        for index, edge_id in enumerate(member_ids):
            edge = edge_lookup.loc[edge_id]
            edge_rows.append(
                {
                    "set_name": set_name,
                    "edge_id": edge_id,
                    "node1": edge["node1"],
                    "node2": edge["node2"],
                    "in_observed_leading_edge": edge_id in observed_leading,
                    "same_direction_inclusion_count": int(inclusion_counts[index]),
                    "conditional_stability": float(conditional_stability[index]),
                    "conditional_stability_lower": float(conditional_lower[index]),
                    "conditional_stability_upper": float(conditional_upper[index]),
                    "full_pipeline_stability": float(full_stability[index]),
                    "full_pipeline_stability_lower": float(full_lower[index]),
                    "full_pipeline_stability_upper": float(full_upper[index]),
                    "conditional_core": bool(conditional_core[index]),
                    "full_pipeline_core": bool(full_pipeline_core[index]),
                }
            )

    set_summary = pd.DataFrame(set_rows, columns=SET_SUMMARY_COLUMNS)
    edge_summary = pd.DataFrame(edge_rows, columns=EDGE_SUMMARY_COLUMNS)
    replicate_summary = pd.DataFrame(replicate_rows, columns=REPLICATE_SUMMARY_COLUMNS)
    metadata = {
        "n_bootstraps": n_bootstraps,
        "significance_alpha": significance_alpha,
        "interval_method": "Jeffreys bootstrap-frequency Monte Carlo interval",
        "interval_level": interval_level,
        "core_threshold": core_threshold,
        "conditional_set_gate_threshold": SET_GATE_THRESHOLD,
        "min_same_direction": min_same_direction,
        "correction_family_id": observed.metadata.get("correction_family_id"),
        "analysis_signature": observed.metadata.get("analysis_signature"),
        "positive_direction": observed.metadata.get("positive_direction"),
        "null_method": observed.metadata.get("null_method"),
        "permutation_scheme": observed.metadata.get("permutation_scheme"),
        "exchangeability_blocks_used": (
            observed.metadata.get("exchangeability_blocks_summary") is not None
        ),
        "n_permutations": observed.metadata.get("n_permutations"),
        "edge_universe_hash": observed.metadata.get("edge_universe_hash"),
        "node_order": observed.metadata.get("node_order"),
        "directed": observed.metadata.get("directed"),
        "diagonal": observed.metadata.get("diagonal"),
        "ranking_statistic_name": observed.metadata.get("ranking_statistic_name"),
        "weight_exponent": observed.metadata.get("weight_exponent"),
        "score_type": observed.metadata.get("score_type"),
        "observed_significant_sets": tracked_names,
        "interpretation": (
            "Empirical sampling stability; not an edge-truth probability, an FDP bound, "
            "or a precise future-study replication probability."
        ),
    }
    return LensStabilityResult(
        set_summary=set_summary,
        edge_summary=edge_summary,
        replicate_summary=replicate_summary,
        metadata=metadata,
        bootstrap_results=replicates if keep_bootstrap_results else None,
    )
