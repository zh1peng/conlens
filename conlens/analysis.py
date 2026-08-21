"""High-level, traceable LENS workflows."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .core import lens_enrich
from .data import matrix_to_edges, validate_connectome
from .design import Contrast, DesignMatrix
from .inference import _freedman_lane_null, adjust_pvalues, apply_null_inference
from .results import GLMResult, LensResult, LensStabilityResult
from .stability import (
    _validate_compatible_result,
    _validate_complete_result,
    _validate_stability_options,
    bootstrap_lens,
    summarize_bootstrap_stability,
)
from .stats import glm_contrast_statistics


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
        self.result_: GLMResult | None = None

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

    def glm(
        self,
        design: DesignMatrix,
        contrasts: Mapping[str, Contrast],
        *,
        n_permutations: int | None = 1000,
        random_state: int | None = None,
        exchangeability_blocks: Iterable[Any] | None = None,
        correction_family_id: str = "default",
        **options: Any,
    ) -> GLMResult:
        """Fit named GLM contrasts and run contrast-specific Freedman-Lane inference."""
        if not isinstance(design, DesignMatrix):
            raise TypeError("design must be created with make_design()")
        if design.n_observations != len(self.data):
            raise ValueError("design must contain one row per subject")
        if not isinstance(contrasts, Mapping) or not contrasts:
            raise ValueError("contrasts must be a non-empty mapping of names to Contrast objects")
        if any(not isinstance(name, str) or not name for name in contrasts):
            raise ValueError("contrast names must be non-empty strings")
        if any(not isinstance(specification, Contrast) for specification in contrasts.values()):
            raise TypeError("every contrast specification must be a Contrast object")
        if n_permutations is not None and (
            not isinstance(n_permutations, int) or n_permutations < 1
        ):
            raise ValueError("n_permutations must be None or a positive integer")
        if not isinstance(correction_family_id, str) or not correction_family_id:
            raise ValueError("correction_family_id must be a non-empty string")

        block_values = None if exchangeability_blocks is None else list(exchangeability_blocks)
        block_summary = _block_summary(block_values, len(self.data))
        parameters = {**self.defaults, **options, "directed": self.directed}
        for owned in (
            "positive_direction",
            "statistic_name",
            "null_method",
            "n_permutations",
            "random_state",
            "correction_family_id",
        ):
            parameters.pop(owned, None)

        outputs: dict[str, LensResult] = {}
        resolved: dict[str, np.ndarray] = {}
        for contrast_name, specification in contrasts.items():
            weights = specification.resolve(design)
            resolved[contrast_name] = weights
            statistics = glm_contrast_statistics(
                self.data,
                design.values,
                weights,
                effect_size=specification.effect_size,
            )
            edges = self.edge_template.copy()
            edges["statistic"] = statistics.effect_size
            edges["effect_size"] = statistics.effect_size
            edges["contrast_estimate"] = statistics.contrast_estimate
            edges["standard_error"] = statistics.standard_error
            edges["t_statistic"] = statistics.t_statistic
            edges["residual_df"] = statistics.residual_df
            edges["edge_p_value_two_sided"] = statistics.p_value_two_sided
            edges["residual_sd"] = statistics.residual_sd
            statistic_name = (
                "partial correlation"
                if specification.effect_size == "partial_r"
                else "model-adjusted Hedges' g"
            )
            result = lens_enrich(
                edges,
                self.edge_sets,
                null_method=None,
                statistic_name=statistic_name,
                positive_direction=specification.positive_direction,
                correction_family_id=correction_family_id,
                **parameters,
            )
            result.metadata.update(
                {
                    "analysis_signature": {
                        "kind": "glm_contrast",
                        "design": design.signature(),
                        "contrast_name": contrast_name,
                        "contrast_vector": weights.tolist(),
                        "effect_size": specification.effect_size,
                    },
                    "contrast_name": contrast_name,
                    "contrast_vector": weights.tolist(),
                    "effect_size": specification.effect_size,
                    "residual_df": statistics.residual_df,
                    "null_method": None,
                    "permutation_scheme": None,
                    "n_permutations": 0,
                    "random_seed": random_state,
                    "inference_status": "not_requested",
                    "exchangeability_blocks_summary": block_summary,
                }
            )
            outputs[contrast_name] = result

        if n_permutations is not None:
            for contrast_name, specification in contrasts.items():
                null = _freedman_lane_null(
                    self.data,
                    design,
                    specification,
                    self.edge_template["edge_id"],
                    self.edge_sets,
                    n_permutations=n_permutations,
                    random_state=random_state,
                    exchangeability_blocks=block_values,
                    weight=parameters.get("weight", 1.0),
                    score_type=parameters.get("score_type", "standard"),
                    canonical_edge_ids=self.edge_template["canonical_edge_id"],
                )
                result = outputs[contrast_name]
                apply_null_inference(result, null, correction_family_id=correction_family_id)
                result.metadata.update(
                    {
                        "null_method": "freedman_lane",
                        "permutation_scheme": "contrast_specific_freedman_lane",
                        "n_permutations": n_permutations,
                        "random_seed": random_state,
                        "inference_status": "complete",
                        "exchangeability_blocks_summary": block_summary,
                    }
                )

            tested_sets = [
                item
                for result in outputs.values()
                for item in result.sets
                if item.status == "ok"
            ]
            pvalues = [item.p_value for item in tested_sets]
            if any(value is None for value in pvalues):
                raise ArithmeticError("joint inference produced a missing P value")
            qvalues = adjust_pvalues(float(value) for value in pvalues if value is not None)
            for item, qvalue in zip(tested_sets, qvalues, strict=True):
                item.q_value = float(qvalue)
            for result in outputs.values():
                result.metadata.update(
                    {
                        "adjustment_method": "BH",
                        "multiple_testing_method": "BH",
                        "n_sets_tested": len(tested_sets),
                        "correction_family_id": correction_family_id,
                    }
                )

        collection = GLMResult(
            contrasts=outputs,
            metadata={
                "design": design.metadata(),
                "contrasts": {
                    name: {
                        "weights": resolved[name].tolist(),
                        "effect_size": specification.effect_size,
                        "positive_direction": specification.positive_direction,
                    }
                    for name, specification in contrasts.items()
                },
                "n_permutations": 0 if n_permutations is None else n_permutations,
                "permutation_scheme": None
                if n_permutations is None
                else "contrast_specific_freedman_lane",
                "adjustment_method": None if n_permutations is None else "BH",
                "correction_family_id": correction_family_id,
            },
        )
        self.result_ = collection
        return collection


def _block_summary(
    blocks: Iterable[Any] | None, n_observations: int
) -> dict[str, int] | None:
    if blocks is None:
        return None
    values = list(blocks)
    if len(values) != n_observations:
        raise ValueError("exchangeability_blocks must contain one value per subject")
    if any(_contains_missing(value) for value in values):
        raise ValueError("exchangeability_blocks cannot contain missing values")
    try:
        _, unique = pd.factorize(pd.Series(values, dtype=object), sort=False)
    except TypeError as exc:
        raise ValueError("exchangeability_blocks values must be hashable") from exc
    return {"n_blocks": int(len(unique)), "n_observations": len(values)}


def _contains_missing(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_contains_missing(component) for component in value)
    return bool(np.asarray(pd.isna(value), dtype=bool).any())
