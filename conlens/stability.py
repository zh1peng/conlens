"""Full-pipeline subject bootstrap for LENS enrichment results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import beta

from .core import lens_stat
from .design import Contrast, DesignMatrix
from .enrichment import lens_enrich
from .glm import lens_fl_permute, lens_glm
from .results import GLMResult, LensResult, LensStabilityResult

SET_GATE_THRESHOLD = 0.50
SET_SUMMARY_COLUMNS = [
    "set_name", "observed_direction", "observed_es", "observed_nes",
    "observed_p_value", "observed_q_value", "detection_count",
    "same_direction_count", "different_direction_count", "detection_rate",
    "direction_consistency", "set_stability", "set_stability_lower",
    "set_stability_upper", "set_reproducibility_supported",
    "observed_leading_edge_size", "bootstrap_leading_edge_size_median",
    "bootstrap_leading_edge_size_lower", "bootstrap_leading_edge_size_upper",
    "median_jaccard_with_observed", "conditional_localization_supported",
    "conditional_core_reportable", "conditional_status", "full_pipeline_core_size",
    "conditional_core_size",
]
EDGE_SUMMARY_COLUMNS = [
    "set_name", "edge_id", "node1", "node2", "in_observed_leading_edge",
    "same_direction_inclusion_count", "conditional_stability",
    "conditional_stability_lower", "conditional_stability_upper",
    "full_pipeline_stability", "full_pipeline_stability_lower",
    "full_pipeline_stability_upper", "conditional_core", "full_pipeline_core",
]
REPLICATE_SUMMARY_COLUMNS = [
    "replicate", "set_name", "detected", "same_direction", "direction", "es",
    "nes", "p_value", "q_value", "leading_edge_size", "jaccard_with_observed",
]
_COMPATIBILITY_FIELDS = (
    "family_name", "edge_universe_hash", "edge_mapping_hash", "set_definition_hash",
    "analysis_signature", "positive_direction", "weight_exponent", "score_type",
    "min_size", "max_size", "n_permutations", "permutation_scheme",
    "exchangeability_blocks_used", "n_tests_in_family",
)


def _jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    return 1.0 if not union else len(first & second) / len(union)


def _jeffreys_interval(
    successes: int | np.ndarray, total: int, interval_level: float
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(successes, dtype=float)
    tail = (1.0 - interval_level) / 2.0
    return (
        np.asarray(beta.ppf(tail, values + 0.5, total - values + 0.5), dtype=float),
        np.asarray(beta.ppf(1.0 - tail, values + 0.5, total - values + 0.5), dtype=float),
    )


def _validate_stability_options(
    significance_alpha: float,
    interval_level: float,
    core_threshold: float,
    min_same_direction: int,
) -> None:
    if not 0 < significance_alpha < 1:
        raise ValueError("significance_alpha must be in (0, 1)")
    if not 0 < interval_level < 1:
        raise ValueError("interval_level must be in (0, 1)")
    if not 0 < core_threshold < 1:
        raise ValueError("core_threshold must be in (0, 1)")
    if not isinstance(min_same_direction, int) or min_same_direction < 1:
        raise ValueError("min_same_direction must be a positive integer")


def _validate_complete(result: LensResult, label: str) -> dict[str, Any]:
    if not isinstance(result, LensResult):
        raise TypeError(f"{label} must be a LensResult")
    if result.metadata.get("adjustment_method") != "BH":
        raise ValueError(f"{label} must contain completed BH-adjusted inference")
    for field in _COMPATIBILITY_FIELDS:
        if field not in result.metadata:
            raise ValueError(f"{label} is missing required metadata {field!r}")
    required = {"edge_id", "canonical_edge_id", "node1", "node2"}
    if not required.issubset(result.ranked_edges):
        raise ValueError(f"{label} ranked_edges are incomplete")
    if result.ranked_edges["edge_id"].duplicated().any():
        raise ValueError(f"{label} ranked_edges contain duplicate edge IDs")
    output = {item.set_name: item for item in result.sets}
    if len(output) != len(result.sets):
        raise ValueError(f"{label} contains duplicate set names")
    for item in result.sets:
        if len(item.leading_edge_ids) != len(set(item.leading_edge_ids)):
            raise ValueError(f"{label} has duplicate leading edges in {item.set_name!r}")
        if item.leading_edge_size != len(item.leading_edge_ids):
            raise ValueError(f"{label} has an inconsistent leading-edge size")
        if not set(item.leading_edge_ids).issubset(item.edge_set_ids):
            raise ValueError(f"{label} leading edges are not set members")
        if item.status == "ok":
            if item.q_value is None or not np.isfinite(item.q_value):
                raise ValueError(f"{label} lacks a valid q value for {item.set_name!r}")
            if not 0 <= item.q_value <= 1:
                raise ValueError(f"{label} has an invalid q value for {item.set_name!r}")
    return output


def _validate_compatible(
    observed: LensResult,
    observed_sets: Mapping[str, Any],
    replicate: LensResult,
    index: int,
) -> dict[str, Any]:
    label = f"bootstrap result {index}"
    current = _validate_complete(replicate, label)
    if set(current) != set(observed_sets):
        raise ValueError(f"{label} contains different set names")
    for field in _COMPATIBILITY_FIELDS:
        if replicate.metadata[field] != observed.metadata[field]:
            raise ValueError(f"{label} has incompatible {field}")
    observed_mapping = dict(zip(
        observed.ranked_edges["edge_id"],
        observed.ranked_edges["canonical_edge_id"],
        strict=True,
    ))
    current_mapping = dict(zip(
        replicate.ranked_edges["edge_id"],
        replicate.ranked_edges["canonical_edge_id"],
        strict=True,
    ))
    if current_mapping != observed_mapping:
        raise ValueError(f"{label} has an incompatible edge mapping")
    for name, observed_set in observed_sets.items():
        if set(current[name].edge_set_ids) != set(observed_set.edge_set_ids):
            raise ValueError(f"{label} changed the definition of {name!r}")
        if current[name].status != observed_set.status:
            raise ValueError(f"{label} changed the status of {name!r}")
    return current


def summarize_stability(
    observed: LensResult,
    bootstrap_results: Iterable[LensResult],
    *,
    significance_alpha: float = 0.05,
    interval_level: float = 0.95,
    core_threshold: float = 0.50,
    min_same_direction: int = 30,
) -> LensStabilityResult:
    """Summarize observed-anchored, full-pipeline bootstrap stability."""
    _validate_stability_options(
        significance_alpha, interval_level, core_threshold, min_same_direction
    )
    observed_sets = _validate_complete(observed, "observed result")
    replicates = list(bootstrap_results)
    if not replicates:
        raise ValueError("bootstrap_results must contain at least one result")
    replicate_sets = [
        _validate_compatible(observed, observed_sets, result, index)
        for index, result in enumerate(replicates)
    ]
    edge_lookup = observed.ranked_edges.set_index("edge_id", drop=False)
    tracked = [
        item.set_name for item in observed.sets
        if item.status == "ok"
        and item.q_value is not None
        and item.q_value <= significance_alpha
    ]
    set_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    total = len(replicates)
    quantile_tail = (1 - interval_level) / 2

    for set_name in tracked:
        observed_set = observed_sets[set_name]
        if observed_set.direction not in {"positive", "negative"}:
            raise ValueError(f"observed set {set_name!r} has no signed direction")
        observed_leading = set(observed_set.leading_edge_ids)
        member_ids = sorted(set(observed_set.edge_set_ids))
        member_index = {edge_id: index for index, edge_id in enumerate(member_ids)}
        inclusion = np.zeros(len(member_ids), dtype=int)
        detected_count = same_count = different_count = 0
        sizes: list[int] = []
        jaccards: list[float] = []
        for replicate_index, by_name in enumerate(replicate_sets):
            item = by_name[set_name]
            detected = bool(item.q_value is not None and item.q_value <= significance_alpha)
            same = bool(detected and item.direction == observed_set.direction)
            if detected:
                detected_count += 1
                if same:
                    same_count += 1
                else:
                    different_count += 1
            score = np.nan
            if same:
                leading = set(item.leading_edge_ids)
                unknown = leading - set(member_index)
                if unknown:
                    raise ValueError(f"bootstrap leading edges are not members of {set_name!r}")
                for edge_id in leading:
                    inclusion[member_index[edge_id]] += 1
                sizes.append(len(leading))
                score = _jaccard(observed_leading, leading)
                jaccards.append(score)
            replicate_rows.append({
                "replicate": replicate_index, "set_name": set_name,
                "detected": detected, "same_direction": same,
                "direction": item.direction, "es": item.ES, "nes": item.NES,
                "p_value": item.p_value, "q_value": item.q_value,
                "leading_edge_size": item.leading_edge_size,
                "jaccard_with_observed": score,
            })

        set_stability = same_count / total
        set_lower, set_upper = _jeffreys_interval(same_count, total, interval_level)
        full = inclusion / total
        full_lower, full_upper = _jeffreys_interval(inclusion, total, interval_level)
        if same_count:
            conditional = inclusion / same_count
            conditional_lower, conditional_upper = _jeffreys_interval(
                inclusion, same_count, interval_level
            )
        else:
            conditional = conditional_lower = conditional_upper = np.full(
                len(member_ids), np.nan
            )
        conditional_supported = same_count >= min_same_direction
        set_supported = bool(set_lower > SET_GATE_THRESHOLD)
        conditional_reportable = conditional_supported and set_supported
        full_core = full_lower > core_threshold
        conditional_core = (
            conditional_lower > core_threshold
            if conditional_reportable
            else np.zeros(len(member_ids), dtype=bool)
        )
        status = (
            "insufficient same-direction detections" if not conditional_supported
            else "set stability below gate" if not set_supported
            else "reportable"
        )
        if sizes:
            size_lower, size_upper = np.quantile(
                sizes, [quantile_tail, 1 - quantile_tail]
            )
            size_median = float(np.median(sizes))
        else:
            size_lower = size_upper = size_median = np.nan
        set_rows.append({
            "set_name": set_name,
            "observed_direction": observed_set.direction,
            "observed_es": observed_set.ES,
            "observed_nes": observed_set.NES,
            "observed_p_value": observed_set.p_value,
            "observed_q_value": observed_set.q_value,
            "detection_count": detected_count,
            "same_direction_count": same_count,
            "different_direction_count": different_count,
            "detection_rate": detected_count / total,
            "direction_consistency": same_count / detected_count if detected_count else np.nan,
            "set_stability": set_stability,
            "set_stability_lower": float(set_lower),
            "set_stability_upper": float(set_upper),
            "set_reproducibility_supported": set_supported,
            "observed_leading_edge_size": observed_set.leading_edge_size,
            "bootstrap_leading_edge_size_median": size_median,
            "bootstrap_leading_edge_size_lower": float(size_lower),
            "bootstrap_leading_edge_size_upper": float(size_upper),
            "median_jaccard_with_observed": float(np.median(jaccards)) if jaccards else np.nan,
            "conditional_localization_supported": conditional_supported,
            "conditional_core_reportable": conditional_reportable,
            "conditional_status": status,
            "full_pipeline_core_size": int(full_core.sum()),
            "conditional_core_size": int(conditional_core.sum()),
        })
        for index, edge_id in enumerate(member_ids):
            edge = edge_lookup.loc[edge_id]
            edge_rows.append({
                "set_name": set_name, "edge_id": edge_id,
                "node1": edge["node1"], "node2": edge["node2"],
                "in_observed_leading_edge": edge_id in observed_leading,
                "same_direction_inclusion_count": int(inclusion[index]),
                "conditional_stability": float(conditional[index]),
                "conditional_stability_lower": float(conditional_lower[index]),
                "conditional_stability_upper": float(conditional_upper[index]),
                "full_pipeline_stability": float(full[index]),
                "full_pipeline_stability_lower": float(full_lower[index]),
                "full_pipeline_stability_upper": float(full_upper[index]),
                "conditional_core": bool(conditional_core[index]),
                "full_pipeline_core": bool(full_core[index]),
            })

    metadata = {
        **{field: observed.metadata.get(field) for field in _COMPATIBILITY_FIELDS},
        "n_bootstraps": total,
        "significance_alpha": significance_alpha,
        "interval_method": "Jeffreys bootstrap-frequency Monte Carlo interval",
        "interval_level": interval_level,
        "core_threshold": core_threshold,
        "conditional_set_gate_threshold": SET_GATE_THRESHOLD,
        "min_same_direction": min_same_direction,
        "observed_significant_sets": tracked,
        "interpretation": (
            "Empirical sampling stability; not an edge-truth probability or a "
            "future-study replication probability."
        ),
    }
    return LensStabilityResult(
        pd.DataFrame(set_rows, columns=SET_SUMMARY_COLUMNS),
        pd.DataFrame(edge_rows, columns=EDGE_SUMMARY_COLUMNS),
        pd.DataFrame(replicate_rows, columns=REPLICATE_SUMMARY_COLUMNS),
        metadata,
    )


def _contains_missing(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_contains_missing(item) for item in value)
    return bool(np.asarray(pd.isna(value), dtype=bool).any())


def _bootstrap_draws(
    n_subjects: int,
    n_bootstraps: int,
    strata: Iterable[Any] | None,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    all_indices = np.arange(n_subjects)
    if strata is None:
        return [rng.choice(all_indices, n_subjects, replace=True) for _ in range(n_bootstraps)]
    values = list(strata)
    if len(values) != n_subjects:
        raise ValueError("strata must contain one value per subject")
    if any(_contains_missing(value) for value in values):
        raise ValueError("strata cannot contain missing values")
    codes, _ = pd.factorize(pd.Series(values, dtype=object), sort=False)
    draws = []
    for _ in range(n_bootstraps):
        parts = []
        for code in np.unique(codes):
            members = all_indices[codes == code]
            parts.append(rng.choice(members, len(members), replace=True))
        draws.append(np.concatenate(parts))
    return draws


def _fit_pipeline(
    connectomes: np.ndarray,
    edge_sets: Mapping[str, Iterable[str]],
    design: DesignMatrix,
    contrasts: Mapping[str, Contrast],
    *,
    n_permutations: int,
    node_labels: Sequence[Any] | None,
    directed: bool,
    diagonal: bool,
    exchangeability_blocks: Iterable[Any] | None,
    random_state: int,
    weight: float,
    score_type: str,
    min_size: int,
    max_size: int | None,
    family_name: str,
    store_running_sum: bool,
) -> GLMResult:
    observed_edges = lens_glm(
        connectomes, design=design, contrasts=contrasts, node_labels=node_labels,
        directed=directed, diagonal=diagonal,
    )
    observed_stat = lens_stat(
        observed_edges, edge_sets, weight=weight, score_type=score_type,
        store_running_sum=store_running_sum,
    )
    null_edges = lens_fl_permute(
        connectomes, design=design, contrasts=contrasts,
        n_permutations=n_permutations, node_labels=node_labels,
        directed=directed, diagonal=diagonal,
        exchangeability_blocks=exchangeability_blocks, random_state=random_state,
    )
    null_stats = (
        lens_stat(item, edge_sets, weight=weight, score_type=score_type)
        for item in null_edges
    )
    result = lens_enrich(
        observed_stat, null_stats, min_size=min_size, max_size=max_size,
        family_name=family_name,
    )
    assert isinstance(result, GLMResult)
    return result


def lens_bootstrap(
    connectomes: np.ndarray,
    edge_sets: Mapping[str, Iterable[str]],
    *,
    design: DesignMatrix,
    contrasts: Mapping[str, Contrast],
    n_bootstraps: int = 1000,
    n_permutations: int = 10000,
    node_labels: Sequence[Any] | None = None,
    directed: bool = False,
    diagonal: bool = False,
    exchangeability_blocks: Iterable[Any] | None = None,
    strata: Iterable[Any] | None = None,
    random_state: int | None = None,
    n_jobs: int = 1,
    weight: float = 1.0,
    score_type: str = "standard",
    min_size: int = 5,
    max_size: int | None = None,
    family_name: str = "default",
    significance_alpha: float = 0.05,
    interval_level: float = 0.95,
    core_threshold: float = 0.50,
    min_same_direction: int = 30,
) -> dict[str, LensStabilityResult]:
    """Repeat the complete GLM, FL, LENS, inference, and BH pipeline by bootstrap."""
    if not isinstance(n_bootstraps, int) or n_bootstraps < 1:
        raise ValueError("n_bootstraps must be a positive integer")
    if not isinstance(n_jobs, int) or n_jobs == 0:
        raise ValueError("n_jobs must be a nonzero integer")
    values = np.asarray(connectomes, dtype=float)
    if values.ndim != 3:
        raise ValueError("connectomes must have shape (subjects, nodes, nodes)")
    if design.n_observations != len(values):
        raise ValueError("design must contain one row per subject")
    blocks = (
        None if exchangeability_blocks is None
        else np.asarray(list(exchangeability_blocks), dtype=object)
    )
    if blocks is not None and len(blocks) != len(values):
        raise ValueError("exchangeability_blocks must contain one value per subject")
    seed_sequence = np.random.SeedSequence(random_state)
    draw_seed, observed_seed, bootstrap_seed = seed_sequence.spawn(3)
    draws = _bootstrap_draws(
        len(values), n_bootstraps, strata, np.random.default_rng(draw_seed)
    )
    fit_seeds = [
        int(seed.generate_state(1)[0]) for seed in bootstrap_seed.spawn(n_bootstraps)
    ]
    observed = _fit_pipeline(
        values, edge_sets, design, contrasts, n_permutations=n_permutations,
        node_labels=node_labels, directed=directed, diagonal=diagonal,
        exchangeability_blocks=blocks,
        random_state=int(observed_seed.generate_state(1)[0]), weight=weight,
        score_type=score_type, min_size=min_size, max_size=max_size,
        family_name=family_name, store_running_sum=True,
    )

    def fit_one(index: int) -> GLMResult:
        draw = draws[index]
        try:
            return _fit_pipeline(
                values[draw], edge_sets, design.take(draw), contrasts,
                n_permutations=n_permutations, node_labels=node_labels,
                directed=directed, diagonal=diagonal,
                exchangeability_blocks=None if blocks is None else blocks[draw],
                random_state=fit_seeds[index], weight=weight,
                score_type=score_type, min_size=min_size, max_size=max_size,
                family_name=family_name, store_running_sum=False,
            )
        except Exception as exc:
            raise RuntimeError(f"bootstrap replicate {index} failed: {exc}") from exc

    first = fit_one(0)
    for name in observed.contrast_names:
        observed_sets = _validate_complete(observed[name], "observed")
        _validate_compatible(observed[name], observed_sets, first[name], 0)
    if n_bootstraps == 1:
        replicates = [first]
    elif n_jobs == 1:
        replicates = [first, *(fit_one(index) for index in range(1, n_bootstraps))]
    else:
        remaining = Parallel(n_jobs=n_jobs)(
            delayed(fit_one)(index) for index in range(1, n_bootstraps)
        )
        replicates = [first, *remaining]
    return {
        name: summarize_stability(
            observed[name], [result[name] for result in replicates],
            significance_alpha=significance_alpha, interval_level=interval_level,
            core_threshold=core_threshold, min_same_direction=min_same_direction,
        )
        for name in observed.contrast_names
    }
