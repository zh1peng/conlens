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
    "node_identity_hash", "analysis_signature", "positive_direction", "weight_exponent",
    "score_type",
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


class _StabilityAccumulator:
    """Online accumulator that retains summaries, not full bootstrap results."""

    def __init__(
        self,
        observed: LensResult,
        *,
        significance_alpha: float,
        interval_level: float,
        core_threshold: float,
        min_same_direction: int,
    ) -> None:
        _validate_stability_options(
            significance_alpha, interval_level, core_threshold, min_same_direction
        )
        self.observed = observed
        self.observed_sets = _validate_complete(observed, "observed result")
        self.alpha = significance_alpha
        self.interval_level = interval_level
        self.core_threshold = core_threshold
        self.min_same_direction = min_same_direction
        self.edge_lookup = observed.ranked_edges.set_index("edge_id", drop=False)
        self.tracked = [
            item.set_name for item in observed.sets
            if item.status == "ok"
            and item.q_value is not None
            and item.q_value <= significance_alpha
        ]
        self.state: dict[str, dict[str, Any]] = {}
        for name in self.tracked:
            item = self.observed_sets[name]
            if item.direction not in {"positive", "negative"}:
                raise ValueError(f"observed set {name!r} has no signed direction")
            members = sorted(set(item.edge_set_ids))
            self.state[name] = {
                "members": members,
                "member_index": {edge_id: index for index, edge_id in enumerate(members)},
                "observed_leading": set(item.leading_edge_ids),
                "inclusion": np.zeros(len(members), dtype=int),
                "detected": 0,
                "same": 0,
                "different": 0,
                "sizes": [],
                "jaccards": [],
            }
        self.replicate_rows: list[dict[str, Any]] = []
        self.total = 0

    def add(self, result: LensResult) -> None:
        by_name = _validate_compatible(
            self.observed, self.observed_sets, result, self.total
        )
        for name in self.tracked:
            observed_item = self.observed_sets[name]
            item = by_name[name]
            state = self.state[name]
            detected = bool(item.q_value is not None and item.q_value <= self.alpha)
            same = bool(detected and item.direction == observed_item.direction)
            if detected:
                state["detected"] += 1
                state["same" if same else "different"] += 1
            score = np.nan
            if same:
                leading = set(item.leading_edge_ids)
                unknown = leading - set(state["member_index"])
                if unknown:
                    raise ValueError(f"bootstrap leading edges are not members of {name!r}")
                for edge_id in leading:
                    state["inclusion"][state["member_index"][edge_id]] += 1
                state["sizes"].append(len(leading))
                score = _jaccard(state["observed_leading"], leading)
                state["jaccards"].append(score)
            self.replicate_rows.append({
                "replicate": self.total, "set_name": name,
                "detected": detected, "same_direction": same,
                "direction": item.direction, "es": item.ES, "nes": item.NES,
                "p_value": item.p_value, "q_value": item.q_value,
                "leading_edge_size": item.leading_edge_size,
                "jaccard_with_observed": score,
            })
        self.total += 1

    def finalize(self) -> LensStabilityResult:
        if self.total == 0:
            raise ValueError("bootstrap_results must contain at least one result")
        set_rows: list[dict[str, Any]] = []
        edge_rows: list[dict[str, Any]] = []
        tail = (1 - self.interval_level) / 2
        for name in self.tracked:
            observed_item = self.observed_sets[name]
            state = self.state[name]
            detected, same, different = state["detected"], state["same"], state["different"]
            inclusion = state["inclusion"]
            stability = same / self.total
            lower, upper = _jeffreys_interval(same, self.total, self.interval_level)
            full = inclusion / self.total
            full_lower, full_upper = _jeffreys_interval(
                inclusion, self.total, self.interval_level
            )
            if same:
                conditional = inclusion / same
                conditional_lower, conditional_upper = _jeffreys_interval(
                    inclusion, same, self.interval_level
                )
            else:
                conditional = conditional_lower = conditional_upper = np.full(
                    len(state["members"]), np.nan
                )
            localization_supported = same >= self.min_same_direction
            set_supported = bool(lower > SET_GATE_THRESHOLD)
            reportable = localization_supported and set_supported
            full_core = full_lower > self.core_threshold
            conditional_core = (
                conditional_lower > self.core_threshold
                if reportable else np.zeros(len(state["members"]), dtype=bool)
            )
            status = (
                "insufficient same-direction detections" if not localization_supported
                else "set stability below gate" if not set_supported else "reportable"
            )
            sizes = state["sizes"]
            if sizes:
                size_lower, size_upper = np.quantile(sizes, [tail, 1 - tail])
                size_median = float(np.median(sizes))
            else:
                size_lower = size_upper = size_median = np.nan
            set_rows.append({
                "set_name": name, "observed_direction": observed_item.direction,
                "observed_es": observed_item.ES, "observed_nes": observed_item.NES,
                "observed_p_value": observed_item.p_value,
                "observed_q_value": observed_item.q_value,
                "detection_count": detected, "same_direction_count": same,
                "different_direction_count": different,
                "detection_rate": detected / self.total,
                "direction_consistency": same / detected if detected else np.nan,
                "set_stability": stability, "set_stability_lower": float(lower),
                "set_stability_upper": float(upper),
                "set_reproducibility_supported": set_supported,
                "observed_leading_edge_size": observed_item.leading_edge_size,
                "bootstrap_leading_edge_size_median": size_median,
                "bootstrap_leading_edge_size_lower": float(size_lower),
                "bootstrap_leading_edge_size_upper": float(size_upper),
                "median_jaccard_with_observed": (
                    float(np.median(state["jaccards"])) if state["jaccards"] else np.nan
                ),
                "conditional_localization_supported": localization_supported,
                "conditional_core_reportable": reportable,
                "conditional_status": status,
                "full_pipeline_core_size": int(full_core.sum()),
                "conditional_core_size": int(conditional_core.sum()),
            })
            for index, edge_id in enumerate(state["members"]):
                edge = self.edge_lookup.loc[edge_id]
                edge_rows.append({
                    "set_name": name, "edge_id": edge_id,
                    "node1": edge["node1"], "node2": edge["node2"],
                    "in_observed_leading_edge": edge_id in state["observed_leading"],
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
            **{field: self.observed.metadata.get(field) for field in _COMPATIBILITY_FIELDS},
            "n_bootstraps": self.total,
            "significance_alpha": self.alpha,
            "interval_method": "Jeffreys bootstrap-frequency Monte Carlo interval",
            "interval_level": self.interval_level,
            "core_threshold": self.core_threshold,
            "conditional_set_gate_threshold": SET_GATE_THRESHOLD,
            "min_same_direction": self.min_same_direction,
            "observed_significant_sets": self.tracked,
            "interpretation": (
                "Empirical sampling stability; not an edge-truth probability or a "
                "future-study replication probability."
            ),
        }
        return LensStabilityResult(
            pd.DataFrame(set_rows, columns=SET_SUMMARY_COLUMNS),
            pd.DataFrame(edge_rows, columns=EDGE_SUMMARY_COLUMNS),
            pd.DataFrame(self.replicate_rows, columns=REPLICATE_SUMMARY_COLUMNS),
            metadata,
        )


def summarize_stability(
    observed: LensResult,
    bootstrap_results: Iterable[LensResult],
    *,
    significance_alpha: float = 0.05,
    interval_level: float = 0.95,
    core_threshold: float = 0.50,
    min_same_direction: int = 30,
) -> LensStabilityResult:
    """Summarize an iterable of full-pipeline replicates in one pass."""
    accumulator = _StabilityAccumulator(
        observed,
        significance_alpha=significance_alpha,
        interval_level=interval_level,
        core_threshold=core_threshold,
        min_same_direction=min_same_direction,
    )
    for result in bootstrap_results:
        accumulator.add(result)
    return accumulator.finalize()


def _contains_missing(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_contains_missing(item) for item in value)
    return bool(np.asarray(pd.isna(value), dtype=bool).any())


def _object_vector(values: Iterable[Any], *, label: str, expected: int) -> np.ndarray:
    items = list(values)
    if len(items) != expected:
        raise ValueError(f"{label} must contain one value per subject")
    if any(_contains_missing(value) for value in items):
        raise ValueError(f"{label} cannot contain missing values")
    output = np.empty(len(items), dtype=object)
    output[:] = items
    return output


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
    blocks = None if exchangeability_blocks is None else _object_vector(
        exchangeability_blocks,
        label="exchangeability_blocks",
        expected=len(values),
    )
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
    accumulators = {
        name: _StabilityAccumulator(
            observed[name],
            significance_alpha=significance_alpha, interval_level=interval_level,
            core_threshold=core_threshold, min_same_direction=min_same_direction,
        )
        for name in observed.contrast_names
    }
    for name, accumulator in accumulators.items():
        accumulator.add(first[name])
    if n_bootstraps > 1:
        if n_jobs == 1:
            remaining: Iterable[GLMResult] = (
                fit_one(index) for index in range(1, n_bootstraps)
            )
        else:
            remaining = Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(fit_one)(index) for index in range(1, n_bootstraps)
            )
        for result in remaining:
            for name, accumulator in accumulators.items():
                accumulator.add(result[name])
    return {name: accumulator.finalize() for name, accumulator in accumulators.items()}
