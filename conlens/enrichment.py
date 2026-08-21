"""Null inference for precomputed observed and streamed LENS statistics."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .results import GLMResult, LensResult, LensSetResult, LensStatResult


def adjust_pvalues(p_values: list[float]) -> list[float]:
    """Benjamini--Hochberg adjusted p values in original order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("p_values must be a finite one-dimensional sequence")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("p_values must lie between 0 and 1")
    if len(values) == 0:
        return []
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    scaled = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output.tolist()


def _as_mapping(
    value: LensStatResult | Mapping[str, LensStatResult],
) -> tuple[dict[str, LensStatResult], bool]:
    if isinstance(value, LensStatResult):
        return {"__single__": value}, False
    if not isinstance(value, Mapping) or not value:
        raise TypeError("LENS statistics must be a LensStatResult or non-empty mapping")
    output = {str(name): item for name, item in value.items()}
    if any(not isinstance(item, LensStatResult) for item in output.values()):
        raise TypeError("every mapped value must be a LensStatResult")
    return output, True


def _validate_options(
    min_size: int,
    max_size: int | None,
    family_name: str,
) -> None:
    if not isinstance(min_size, int) or min_size < 1:
        raise ValueError("min_size must be a positive integer")
    if max_size is not None and (not isinstance(max_size, int) or max_size < min_size):
        raise ValueError("max_size must be None or an integer >= min_size")
    if not isinstance(family_name, str) or not family_name.strip():
        raise ValueError("family_name must be a non-empty string")


def _filter_sets(
    result: LensStatResult,
    *,
    min_size: int,
    max_size: int | None,
) -> list[LensSetResult]:
    output = copy.deepcopy(result.sets)
    for item in output:
        if item.status != "ok":
            continue
        too_small = item.set_size_effective < min_size
        too_large = max_size is not None and item.set_size_effective > max_size
        if too_small or too_large:
            bounds = f"[{min_size}, {max_size if max_size is not None else 'unbounded'}]"
            item.status = "filtered"
            item.warnings.append(
                f"effective set size {item.set_size_effective} is outside {bounds}"
            )
    return output


_COMPATIBILITY_FIELDS = (
    "edge_universe_hash",
    "edge_mapping_hash",
    "node_identity_hash",
    "set_definition_hash",
    "positive_direction",
    "weight_exponent",
    "score_type",
    "analysis_signature",
)


def _validate_null(observed: LensStatResult, null: LensStatResult, label: str) -> None:
    if [item.set_name for item in null.sets] != [item.set_name for item in observed.sets]:
        raise ValueError(f"{label} contains different edge sets")
    for field in _COMPATIBILITY_FIELDS:
        expected = observed.metadata.get(field)
        actual = null.metadata.get(field)
        if expected is None:
            raise ValueError(f"observed LENS statistics are missing required metadata {field!r}")
        if actual != expected:
            raise ValueError(f"{label} has incompatible {field}")
    expected_design_hash = observed.metadata.get("design_data_hash")
    if (
        expected_design_hash is not None
        and null.metadata.get("design_data_hash") != expected_design_hash
    ):
        raise ValueError(f"{label} has incompatible design_data_hash")


def _apply_null(item: LensSetResult, null_values: np.ndarray) -> None:
    assert item.ES is not None
    positive = null_values[null_values >= 0]
    negative = null_values[null_values <= 0]
    item.n_null_positive = len(positive)
    item.n_null_negative = len(negative)
    item.n_permutations = len(null_values)
    tail = positive if item.ES > 0 else negative
    item.minimum_resolvable_p = 1.0 / (len(tail) + 1)
    item.p_value_method = "same-direction empirical tail with plus-one correction"
    if len(tail) == 0:
        item.NES = None
        item.p_value = 1.0
        item.n_more_extreme = 0
        item.normalization_status = "no same-direction null scores"
        return
    scale = float(np.mean(np.abs(tail)))
    if scale <= 0 or not np.isfinite(scale):
        item.NES = None
        item.normalization_status = "zero same-direction null scale"
    else:
        item.NES = float(item.ES / scale)
        item.normalization_status = "ok"
    if item.ES > 0:
        more_extreme = int(np.count_nonzero(tail >= item.ES))
    else:
        more_extreme = int(np.count_nonzero(tail <= item.ES))
    item.n_more_extreme = more_extreme
    item.p_value = (more_extreme + 1) / (len(tail) + 1)


def lens_enrich(
    observed_lens_stat: LensStatResult | Mapping[str, LensStatResult],
    null_lens_stats: Iterable[LensStatResult | Mapping[str, LensStatResult]] | None = None,
    *,
    min_size: int = 5,
    max_size: int | None = None,
    family_name: str = "default",
) -> LensResult | GLMResult:
    """Add size filtering, streamed null inference, and joint BH correction.

    ``lens_enrich`` never computes or permutes edge statistics. The retained null
    data contain one enrichment score per permutation and tested edge set, which
    is sufficient for observed-versus-null diagnostics without an edge-by-null
    matrix.
    """
    _validate_options(min_size, max_size, family_name)
    observed, was_mapping = _as_mapping(observed_lens_stat)
    sets_by_contrast = {
        name: _filter_sets(item, min_size=min_size, max_size=max_size)
        for name, item in observed.items()
    }
    tested = {
        name: [item.set_name for item in sets if item.status == "ok"]
        for name, sets in sets_by_contrast.items()
    }
    null_rows: dict[str, list[dict[str, float]]] = {name: [] for name in observed}
    null_provenance: dict[str, dict[str, Any]] = {name: {} for name in observed}
    n_permutations = 0
    if null_lens_stats is not None:
        for replicate_index, raw_null in enumerate(null_lens_stats):
            null, null_was_mapping = _as_mapping(raw_null)
            if null_was_mapping != was_mapping or tuple(null) != tuple(observed):
                raise ValueError(
                    f"null replicate {replicate_index} contains different contrasts"
                )
            for name, observed_item in observed.items():
                _validate_null(observed_item, null[name], f"null replicate {replicate_index}")
                provenance = {
                    "permutation_scheme": null[name].metadata.get("permutation_scheme"),
                    "exchangeability_blocks_used": bool(
                        null[name].metadata.get("exchangeability_blocks_used", False)
                    ),
                }
                if replicate_index == 0:
                    null_provenance[name] = provenance
                elif provenance != null_provenance[name]:
                    raise ValueError("null permutation provenance changed across replicates")
                row: dict[str, float] = {}
                for set_name in tested[name]:
                    value = null[name].get(set_name).ES
                    if value is None or not np.isfinite(value):
                        raise ValueError(
                            f"null replicate {replicate_index} has no finite ES for {set_name!r}"
                        )
                    row[set_name] = float(value)
                null_rows[name].append(row)
            n_permutations += 1
        if n_permutations == 0:
            raise ValueError("null_lens_stats must yield at least one replicate")

    results: dict[str, LensResult] = {}
    pvalue_targets: list[LensSetResult] = []
    for name, observed_item in observed.items():
        null_frame = (
            pd.DataFrame(null_rows[name], columns=tested[name])
            if null_lens_stats is not None
            else None
        )
        current_sets = sets_by_contrast[name]
        if null_frame is not None:
            for item in current_sets:
                if item.status != "ok":
                    continue
                _apply_null(item, null_frame[item.set_name].to_numpy(float))
                pvalue_targets.append(item)
        metadata = {
            **copy.deepcopy(observed_item.metadata),
            "family_name": family_name,
            "min_size": min_size,
            "max_size": max_size,
            "n_permutations": n_permutations,
            "inference_status": "complete" if null_frame is not None else "descriptive",
            "adjustment_method": "BH" if null_frame is not None else None,
            **null_provenance[name],
        }
        results[name] = LensResult(
            sets=current_sets,
            metadata=metadata,
            ranked_edges=observed_item.ranked_edges.copy(),
            null_scores=null_frame,
        )

    if pvalue_targets:
        raw_pvalues = []
        for item in pvalue_targets:
            if item.p_value is None:
                raise ArithmeticError("null inference did not produce a p value")
            raw_pvalues.append(item.p_value)
        adjusted = adjust_pvalues(raw_pvalues)
        for item, q_value in zip(pvalue_targets, adjusted, strict=True):
            item.q_value = q_value
    n_tests = len(pvalue_targets)
    for result in results.values():
        result.metadata["n_tests_in_family"] = n_tests
        result.metadata["contrasts_in_family"] = [
            name for name, set_names in tested.items() if set_names
        ]

    if was_mapping:
        return GLMResult(
            results,
            {
                "family_name": family_name,
                "n_tests_in_family": n_tests,
                "n_permutations": n_permutations,
                "adjustment_method": "BH" if null_lens_stats is not None else None,
            },
        )
    return results["__single__"]
