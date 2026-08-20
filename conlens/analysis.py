"""High-level, traceable LENS workflows."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .core import lens_enrich
from .data import matrix_to_edges, validate_connectome
from .inference import apply_null_inference, freedman_lane_null, label_permutation_null
from .results import LensResult, LensStabilityResult
from .stability import (
    _validate_compatible_result,
    _validate_complete_result,
    _validate_stability_options,
    bootstrap_lens,
    summarize_bootstrap_stability,
)
from .stats import edge_correlation, glm_statistic, two_group_ttest


class LensAnalysis:
    """Analysis object that retains inputs and delegates to public low-level functions."""

    def __init__(
        self,
        edges: pd.DataFrame,
        edge_sets: Mapping[str, Iterable[str]],
        **defaults: Any,
    ) -> None:
        self.edges = edges.copy()
        self.edge_sets = {name: set(members) for name, members in edge_sets.items()}
        self.defaults = defaults
        self.result_: LensResult | None = None

    def run(self, **options: Any) -> LensResult:
        parameters = {**self.defaults, **options}
        self.result_ = lens_enrich(self.edges, self.edge_sets, **parameters)
        return self.result_

    @classmethod
    def from_subject_connectomes(
        cls,
        connectomes: np.ndarray,
        edge_sets: Mapping[str, Iterable[str]],
        *,
        node_labels: Sequence[Any] | None = None,
        directed: bool = False,
        **defaults: Any,
    ) -> SubjectLensAnalysis:
        return SubjectLensAnalysis(
            connectomes,
            edge_sets,
            node_labels=node_labels,
            directed=directed,
            **defaults,
        )


class SubjectLensAnalysis:
    def __init__(
        self,
        connectomes: np.ndarray,
        edge_sets: Mapping[str, Iterable[str]],
        *,
        node_labels: Sequence[Any] | None = None,
        directed: bool = False,
        **defaults: Any,
    ) -> None:
        array = validate_connectome(connectomes, node_labels=node_labels, directed=directed)
        if array.ndim != 3:
            raise ValueError("subject-level analysis requires shape (subjects, nodes, nodes)")
        long_edges = matrix_to_edges(array, node_labels, directed=directed)
        first = long_edges[long_edges["subject"] == 0]
        self.edge_template = first[["node1", "node2", "edge_id", "canonical_edge_id"]].copy()
        self.data = np.stack(
            [
                long_edges[long_edges["subject"] == subject]["statistic"].to_numpy(float)
                for subject in range(len(array))
            ]
        )
        self.edge_sets = {name: set(members) for name, members in edge_sets.items()}
        self.directed = directed
        self.defaults = defaults
        self.result_: LensResult | None = None

    def bootstrap(
        self,
        statistic_function,
        *,
        n_bootstraps: int = 1000,
        random_state: int | None = None,
        strata: Iterable[Any] | None = None,
        **options: Any,
    ) -> list[LensResult]:
        """Generate descriptive statistic-bootstrap LENS results.

        This low-level method does not rerun subject-level null inference. Use
        :meth:`bootstrap_stability` when each replicate must repeat inference and
        BH adjustment.
        """
        parameters = {**self.defaults, **options, "directed": self.directed}
        return bootstrap_lens(
            self.edge_template,
            self.edge_sets,
            subject_data=self.data,
            statistic_function=statistic_function,
            n_bootstraps=n_bootstraps,
            random_state=random_state,
            strata=strata,
            **parameters,
        )

    def bootstrap_stability(
        self,
        observed: LensResult,
        refit: Callable[[SubjectLensAnalysis, np.ndarray, int], LensResult],
        *,
        n_bootstraps: int = 1000,
        random_state: int | None = None,
        strata: Iterable[Any] | None = None,
        n_jobs: int = 1,
        significance_alpha: float = 0.05,
        interval_level: float = 0.95,
        core_threshold: float = 0.50,
        min_same_direction: int = 30,
        keep_bootstrap_results: bool = False,
    ) -> LensStabilityResult:
        """Run a full subject bootstrap and summarize sampling stability.

        ``refit`` receives a resampled ``SubjectLensAnalysis``, the original row
        indices in that sample, and a replicate-specific random seed. It must rerun
        the complete model, null inference, and BH adjustment and return a
        ``LensResult``. Supplying ``strata`` samples subjects within each stratum;
        combine factors such as site and group before passing them.
        """
        if not callable(refit):
            raise TypeError("refit must be callable")
        if not isinstance(n_bootstraps, int) or n_bootstraps < 1:
            raise ValueError("n_bootstraps must be a positive integer")
        if not isinstance(n_jobs, int) or n_jobs == 0:
            raise ValueError("n_jobs must be a nonzero integer")
        _validate_stability_options(
            significance_alpha,
            interval_level,
            core_threshold,
            min_same_direction,
        )
        observed_sets = _validate_complete_result(observed, "observed result")

        strata_codes = None
        n_strata = None
        if strata is not None:
            labels = list(strata)
            if len(labels) != len(self.data):
                raise ValueError("strata must contain one value per subject")
            if any(_contains_missing(label) for label in labels):
                raise ValueError("strata cannot contain missing values")
            series = pd.Series(labels, dtype=object)
            try:
                strata_codes, unique_strata = pd.factorize(series, sort=False)
            except TypeError as exc:
                raise ValueError("strata values must be hashable") from exc
            n_strata = len(unique_strata)

        rng = np.random.default_rng(random_state)
        all_indices = np.arange(len(self.data))
        draws: list[tuple[int, np.ndarray, int]] = []
        for replicate_index in range(n_bootstraps):
            if strata_codes is None:
                indices = rng.choice(all_indices, size=len(all_indices), replace=True)
            else:
                pieces = []
                for code in range(int(strata_codes.max()) + 1):
                    members = all_indices[strata_codes == code]
                    pieces.append(rng.choice(members, size=len(members), replace=True))
                indices = np.concatenate(pieces)
            fit_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
            draws.append((replicate_index, indices, fit_seed))

        def run_one(task: tuple[int, np.ndarray, int]) -> LensResult:
            replicate_index, indices, fit_seed = task
            sample = self._resampled(indices)
            try:
                result = refit(sample, indices.copy(), fit_seed)
            except Exception as exc:
                raise RuntimeError(f"bootstrap replicate {replicate_index} failed") from exc
            if not isinstance(result, LensResult):
                raise TypeError(f"bootstrap replicate {replicate_index} did not return LensResult")
            return result

        first_result = run_one(draws[0])
        _validate_compatible_result(observed, observed_sets, first_result, 0)
        remaining_draws = draws[1:]
        if n_jobs == 1:
            results = [first_result, *(run_one(task) for task in remaining_draws)]
        else:
            from joblib import Parallel, delayed

            remaining_results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(run_one)(task) for task in remaining_draws
            )
            results = [first_result, *remaining_results]

        stability = summarize_bootstrap_stability(
            observed,
            results,
            significance_alpha=significance_alpha,
            interval_level=interval_level,
            core_threshold=core_threshold,
            min_same_direction=min_same_direction,
            keep_bootstrap_results=keep_bootstrap_results,
        )
        stability.metadata.update(
            {
                "resampling_method": "subject" if strata_codes is None else "stratified_subject",
                "n_subjects": len(self.data),
                "n_strata": n_strata,
                "random_seed": random_state,
                "n_jobs": n_jobs,
            }
        )
        return stability

    def _resampled(self, indices: np.ndarray) -> SubjectLensAnalysis:
        sample = object.__new__(SubjectLensAnalysis)
        sample.edge_template = self.edge_template.copy()
        sample.data = self.data[indices].copy()
        sample.edge_sets = {name: set(members) for name, members in self.edge_sets.items()}
        sample.directed = self.directed
        sample.defaults = self.defaults.copy()
        sample.result_ = None
        return sample

    def two_group(
        self,
        labels: Iterable[Any],
        *,
        null_method: str | None = None,
        n_permutations: int = 1000,
        random_state: int | None = None,
        exchangeability_blocks: Iterable[Any] | None = None,
        **options: Any,
    ) -> LensResult:
        group_labels = np.asarray(list(labels))
        statistics = two_group_ttest(self.data, group_labels)
        edges = self.edge_template.copy()
        edges["statistic"] = statistics
        parameters = {**self.defaults, **options, "directed": self.directed}
        unique_groups = np.unique(group_labels)
        low_group, high_group = unique_groups.tolist()
        parameters.setdefault("positive_direction", f"{high_group!r} > {low_group!r}")
        result = lens_enrich(edges, self.edge_sets, null_method=None, **parameters)
        result.metadata["analysis_signature"] = {
            "kind": "two_group",
            "group_levels": [repr(low_group), repr(high_group)],
        }
        if null_method is None:
            self.result_ = result
            return result
        if null_method != "label_permutation":
            raise ValueError("two_group supports null_method None or 'label_permutation'")
        null = label_permutation_null(
            self.data,
            group_labels,
            self.edge_template["edge_id"],
            self.edge_sets,
            n_permutations=n_permutations,
            random_state=random_state,
            exchangeability_blocks=exchangeability_blocks,
            weight=parameters.get("weight", 1.0),
            score_type=parameters.get("score_type", "standard"),
        )
        apply_null_inference(
            result, null, correction_family_id=parameters.get("correction_family_id", "default")
        )
        result.metadata.update(
            null_method="label_permutation",
            permutation_scheme="shared_subject_label_permutation",
            n_permutations=n_permutations,
            random_seed=random_state,
            inference_status="complete",
            exchangeability_blocks_summary=_block_summary(exchangeability_blocks),
        )
        self.result_ = result
        return result

    def glm(
        self,
        tested_design: np.ndarray,
        nuisance_design: np.ndarray,
        *,
        contrast: np.ndarray | None = None,
        null_method: str | None = None,
        n_permutations: int = 1000,
        random_state: int | None = None,
        exchangeability_blocks: Iterable[Any] | None = None,
        **options: Any,
    ) -> LensResult:
        tested = np.asarray(tested_design, float)
        if tested.ndim == 1:
            tested = tested[:, None]
        nuisance = np.asarray(nuisance_design, float)
        if nuisance.ndim == 1:
            nuisance = nuisance[:, None]
        if not np.any(np.all(np.isclose(nuisance, 1.0), axis=0)):
            raise ValueError("nuisance_design must explicitly contain an intercept column")
        full = np.column_stack([tested, nuisance])
        c = np.zeros(full.shape[1])
        if contrast is None:
            c[0] = 1
        else:
            supplied = np.asarray(contrast, float).reshape(-1)
            if len(supplied) == tested.shape[1]:
                c[: tested.shape[1]] = supplied
            elif len(supplied) == full.shape[1]:
                c = supplied
            else:
                raise ValueError("contrast must match tested or full design columns")
        statistics = glm_statistic(self.data, full, c)
        edges = self.edge_template.copy()
        edges["statistic"] = statistics
        parameters = {**self.defaults, **options, "directed": self.directed}
        parameters.setdefault("positive_direction", "positive tested contrast")
        result = lens_enrich(edges, self.edge_sets, null_method=None, **parameters)
        result.metadata["analysis_signature"] = {
            "kind": "glm",
            "n_tested_columns": tested.shape[1],
            "n_nuisance_columns": nuisance.shape[1],
            "contrast_vector": c.tolist(),
        }
        if null_method is None:
            self.result_ = result
            return result
        if null_method != "freedman_lane":
            raise ValueError("glm supports null_method None or 'freedman_lane'")
        null = freedman_lane_null(
            self.data,
            tested,
            nuisance,
            self.edge_template["edge_id"],
            self.edge_sets,
            contrast=contrast,
            n_permutations=n_permutations,
            random_state=random_state,
            exchangeability_blocks=exchangeability_blocks,
            weight=parameters.get("weight", 1.0),
            score_type=parameters.get("score_type", "standard"),
        )
        apply_null_inference(
            result, null, correction_family_id=parameters.get("correction_family_id", "default")
        )
        result.metadata.update(
            null_method="freedman_lane",
            permutation_scheme="shared_reduced_model_residual_permutation",
            n_permutations=n_permutations,
            random_seed=random_state,
            inference_status="complete",
            exchangeability_blocks_summary=_block_summary(exchangeability_blocks),
        )
        self.result_ = result
        return result

    def phenotype(
        self,
        phenotype: Iterable[float],
        *,
        statistic_function=edge_correlation,
        null_method: str | None = None,
        n_permutations: int = 1000,
        random_state: int | None = None,
        exchangeability_blocks: Iterable[Any] | None = None,
        **options: Any,
    ) -> LensResult:
        """Analyze a simple phenotype design without nuisance covariates."""
        target = np.asarray(list(phenotype), dtype=float)
        statistics = np.asarray(statistic_function(self.data, target), dtype=float)
        if statistics.shape != (self.data.shape[1],) or not np.isfinite(statistics).all():
            raise ValueError("statistic_function must return one finite statistic per edge")
        edges = self.edge_template.copy()
        edges["statistic"] = statistics
        parameters = {**self.defaults, **options, "directed": self.directed}
        parameters.setdefault("positive_direction", "positive phenotype association")
        result = lens_enrich(edges, self.edge_sets, null_method=None, **parameters)
        result.metadata["analysis_signature"] = {
            "kind": "phenotype",
            "statistic_function": _callable_identifier(statistic_function),
        }
        if null_method is None:
            self.result_ = result
            return result
        if null_method != "label_permutation":
            raise ValueError("phenotype supports null_method None or 'label_permutation'")
        null = label_permutation_null(
            self.data,
            target,
            self.edge_template["edge_id"],
            self.edge_sets,
            n_permutations=n_permutations,
            random_state=random_state,
            exchangeability_blocks=exchangeability_blocks,
            weight=parameters.get("weight", 1.0),
            score_type=parameters.get("score_type", "standard"),
            statistic_function=statistic_function,
        )
        apply_null_inference(
            result, null, correction_family_id=parameters.get("correction_family_id", "default")
        )
        result.metadata.update(
            null_method="label_permutation",
            permutation_scheme="shared_subject_phenotype_permutation",
            n_permutations=n_permutations,
            random_seed=random_state,
            inference_status="complete",
            exchangeability_blocks_summary=_block_summary(exchangeability_blocks),
        )
        self.result_ = result
        return result


def _block_summary(blocks: Iterable[Any] | None) -> dict[str, int] | None:
    if blocks is None:
        return None
    values = np.asarray(list(blocks))
    return {"n_blocks": int(len(np.unique(values))), "n_observations": int(len(values))}


def _callable_identifier(function: Callable[..., Any]) -> str:
    module = getattr(function, "__module__", type(function).__module__)
    name = getattr(function, "__qualname__", type(function).__qualname__)
    return f"{module}.{name}"


def _contains_missing(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_contains_missing(component) for component in value)
    return bool(np.asarray(pd.isna(value), dtype=bool).any())
