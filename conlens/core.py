"""Deterministic ranking, running-sum, ES, and high-level enrichment."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from ._version import __version__
from .data import validate_edge_table
from .results import LensResult, LensSetResult
from .sets import validate_edge_sets

TOLERANCE = 1e-12


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def rank_edges(edges: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sort statistic descending, then canonical edge ID ascending, stably."""
    if not {"edge_id", "statistic"}.issubset(edges.columns):
        raise ValueError("edges must contain edge_id and statistic")
    if edges["edge_id"].isna().any() or edges["edge_id"].duplicated().any():
        raise ValueError("edge_id values must be present and unique")
    try:
        statistics = pd.to_numeric(edges["statistic"], errors="raise").to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("statistics must be numeric") from exc
    if not np.isfinite(statistics).all():
        raise ValueError("statistics must be finite")
    if np.unique(statistics).size <= 1:
        raise ValueError("all statistics are identical; ranked list is not interpretable")
    sortable = edges.copy()
    sortable["statistic"] = statistics
    tie_column = "canonical_edge_id" if "canonical_edge_id" in edges else "edge_id"
    ranked = sortable.sort_values(
        ["statistic", tie_column], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    tied = ranked["statistic"].duplicated(keep=False)
    metadata = {
        "n_tied_edges": int(tied.sum()),
        "tied_edge_fraction": float(tied.mean()),
        "tie_method": "statistic_desc_then_canonical_edge_id_asc_stable",
    }
    return ranked, metadata


def compute_running_sum(
    statistics: Iterable[float],
    hits: Iterable[bool],
    *,
    weight: float = 1.0,
    tolerance: float = TOLERANCE,
) -> tuple[np.ndarray, bool]:
    values = np.asarray(list(statistics), dtype=float)
    membership = np.asarray(list(hits), dtype=bool)
    if values.ndim != 1 or membership.ndim != 1 or len(values) != len(membership):
        raise ValueError("statistics and hits must be one-dimensional arrays of equal length")
    if not np.isfinite(values).all():
        raise ValueError("statistics must be finite")
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("weight must be a finite number >= 0")
    n_hits = int(membership.sum())
    if n_hits in {0, len(values)}:
        raise ValueError("running sum is undefined for empty or full-universe sets")
    zero_weight_fallback = False
    if weight == 0:
        hit_weights = np.full(n_hits, 1.0 / n_hits)
    else:
        raw = np.abs(values[membership]) ** weight
        total = float(raw.sum())
        if total == 0:
            hit_weights = np.full(n_hits, 1.0 / n_hits)
            zero_weight_fallback = True
        else:
            hit_weights = raw / total
    increments = np.full(len(values), -1.0 / (len(values) - n_hits))
    increments[membership] = hit_weights
    profile = np.concatenate(([0.0], np.cumsum(increments)))
    endpoint_tolerance = max(tolerance, 8 * np.finfo(float).eps * len(values))
    if abs(profile[-1]) > endpoint_tolerance:
        raise ArithmeticError(
            f"running-sum endpoint {profile[-1]} exceeds tolerance {endpoint_tolerance}"
        )
    profile[-1] = 0.0
    return profile, zero_weight_fallback


def compute_enrichment_score(
    running_sum: Iterable[float],
    *,
    score_type: str = "standard",
    tolerance: float = TOLERANCE,
) -> dict[str, Any]:
    profile = np.asarray(list(running_sum), dtype=float)
    if profile.ndim != 1 or len(profile) < 2 or not np.isfinite(profile).all():
        raise ValueError("running_sum must be a finite one-dimensional profile including RS(0)")
    if score_type not in {"standard", "positive", "negative"}:
        raise ValueError("score_type must be 'standard', 'positive', or 'negative'")
    values = profile[1:]
    positive = float(np.max(values))
    negative = float(np.min(values))
    if score_type == "positive":
        score, direction = positive, "positive"
        rank = int(np.flatnonzero(np.isclose(values, positive, atol=tolerance, rtol=0))[0] + 1)
    elif score_type == "negative":
        score, direction = negative, "negative"
        rank = int(np.flatnonzero(np.isclose(values, negative, atol=tolerance, rtol=0))[-1] + 1)
    elif abs(positive - abs(negative)) <= tolerance:
        score, direction, rank = 0.0, "ambiguous", None
    elif positive > abs(negative):
        score, direction = positive, "positive"
        rank = int(np.flatnonzero(np.isclose(values, positive, atol=tolerance, rtol=0))[0] + 1)
    else:
        score, direction = negative, "negative"
        rank = int(np.flatnonzero(np.isclose(values, negative, atol=tolerance, rtol=0))[-1] + 1)
    return {
        "ES": score,
        "ES_positive": positive,
        "ES_negative": negative,
        "direction": direction,
        "peak_rank": rank,
    }


def extract_leading_edges(
    edge_ids: Iterable[str],
    hits: Iterable[bool],
    score: float,
    peak_rank: int | None,
) -> list[str]:
    identifiers = np.asarray(list(edge_ids), dtype=object)
    membership = np.asarray(list(hits), dtype=bool)
    if len(identifiers) != len(membership):
        raise ValueError("edge_ids and hits must have equal length")
    if score == 0 or peak_rank is None:
        return []
    ranks = np.arange(1, len(identifiers) + 1)
    selected = membership & (ranks <= peak_rank if score > 0 else ranks > peak_rank)
    return identifiers[selected].astype(str).tolist()


def _set_result(
    name: str,
    input_size: int,
    members: set[str],
    ranked: pd.DataFrame,
    *,
    weight: float,
    score_type: str,
    store_running_sum: bool,
) -> LensSetResult:
    hits = ranked["edge_id"].isin(members).to_numpy()
    profile, fallback = compute_running_sum(ranked["statistic"], hits, weight=weight)
    score = compute_enrichment_score(profile, score_type=score_type)
    leading_ids = extract_leading_edges(ranked["edge_id"], hits, score["ES"], score["peak_rank"])
    leading_rows = ranked[ranked["edge_id"].isin(leading_ids)]
    leading_nodes = list(
        dict.fromkeys([*leading_rows["node1"].tolist(), *leading_rows["node2"].tolist()])
    )
    return LensSetResult(
        set_name=name,
        set_size_input=input_size,
        set_size_effective=len(members),
        ES=score["ES"],
        ES_positive=score["ES_positive"],
        ES_negative=score["ES_negative"],
        direction=score["direction"],
        peak_rank=score["peak_rank"],
        peak_fraction=None if score["peak_rank"] is None else score["peak_rank"] / len(ranked),
        leading_edge_ids=leading_ids,
        leading_edge_size=len(leading_ids),
        leading_edge_fraction=len(leading_ids) / len(members),
        leading_node_ids=leading_nodes,
        zero_weight_fallback=fallback,
        edge_set_ids=sorted(members),
        running_sum=profile.tolist() if store_running_sum else None,
    )


def lens_enrich(
    edges: pd.DataFrame,
    edge_sets: Mapping[str, Iterable[str]],
    *,
    directed: bool = False,
    diagonal: bool = False,
    nan_policy: str = "raise",
    weight: float = 1.0,
    score_type: str = "standard",
    min_size: int = 5,
    max_size: int | None = None,
    null_method: str | None = None,
    n_permutations: int = 1000,
    random_state: int | None = None,
    provided_null: Mapping[str, Iterable[float]] | np.ndarray | None = None,
    provided_null_kind: str = "es",
    provided_null_edge_ids: Iterable[str] | None = None,
    provided_null_edge_sets: Mapping[str, Iterable[str]] | None = None,
    provided_null_direction: str | None = None,
    store_running_sum: bool = False,
    statistic_name: str = "statistic",
    positive_direction: str | None = None,
    correction_family_id: str = "default",
) -> LensResult:
    """Run descriptive LENS and optional explicitly selected edge/provided-null inference."""
    if min_size < 1:
        raise ValueError("min_size must be >= 1")
    if max_size is not None and (max_size < 1 or max_size < min_size):
        raise ValueError("max_size must be >= min_size")
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("weight must be a finite number >= 0")
    if score_type not in {"standard", "positive", "negative"}:
        raise ValueError("score_type must be 'standard', 'positive', or 'negative'")
    if null_method not in {None, "edge_permutation", "provided_null"}:
        raise ValueError(
            "lens_enrich accepts null_method None, 'edge_permutation', or 'provided_null'; "
            "use label_permutation_null/freedman_lane_null for subject-level inference"
        )
    if null_method == "edge_permutation" and n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    validated = validate_edge_table(
        edges,
        node_order=edges.attrs.get("node_order"),
        directed=directed,
        diagonal=diagonal,
        nan_policy=nan_policy,
    )
    ranked, ranking_meta = rank_edges(validated)
    universe = set(ranked["edge_id"])
    input_sets = {
        str(name): [str(member) for member in members] for name, members in edge_sets.items()
    }
    sets = validate_edge_sets(input_sets, universe)
    maximum = len(ranked) - 1 if max_size is None else max_size
    results: list[LensSetResult] = []
    for name, members in sets.items():
        size = len(members)
        if size in {0, len(ranked)}:
            status, reason = "invalid", "empty set" if size == 0 else "full-universe set"
            results.append(
                LensSetResult(
                    name,
                    len(input_sets[name]),
                    size,
                    None,
                    None,
                    None,
                    status=status,
                    warnings=[reason],
                )
            )
        elif size < min_size or size > maximum:
            reason = f"effective set size {size} is outside [{min_size}, {maximum}]"
            results.append(
                LensSetResult(
                    name,
                    len(input_sets[name]),
                    size,
                    None,
                    None,
                    None,
                    status="filtered",
                    warnings=[reason],
                )
            )
        else:
            results.append(
                _set_result(
                    name,
                    len(input_sets[name]),
                    members,
                    ranked,
                    weight=weight,
                    score_type=score_type,
                    store_running_sum=store_running_sum,
                )
            )
    node_order = validated.attrs.get("node_order", [])
    payload = validated[["node1", "node2", "statistic", "edge_id", "canonical_edge_id"]].to_dict(
        "records"
    )
    metadata = {
        "package_version": __version__,
        "python_version": platform.python_version(),
        "input_hash": _hash_payload(payload),
        "node_order": node_order,
        "edge_universe_hash": _hash_payload(sorted(universe)),
        "edge_universe_size": len(universe),
        "directed": directed,
        "diagonal": diagonal,
        "ranking_statistic_name": statistic_name,
        "analysis_signature": {
            "kind": "edge_statistics",
            "ranking_statistic_name": statistic_name,
        },
        "positive_direction": positive_direction,
        "weight_exponent": weight,
        "score_type": score_type,
        "set_size_filters": {"min_size": min_size, "max_size": maximum},
        "null_method": null_method,
        "permutation_scheme": null_method,
        "exchangeability_blocks_summary": None,
        "n_permutations": n_permutations if null_method else 0,
        "random_seed": random_state,
        "multiple_testing_method": "BH" if null_method else None,
        "correction_family_id": correction_family_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "omitted_edges": validated.attrs.get("omitted_edges", []),
        "inference_status": "not_requested" if null_method is None else "requested",
        **ranking_meta,
    }
    output = LensResult(results, metadata, ranked)
    if null_method is None:
        return output
    from .inference import (
        apply_null_inference,
        edge_permutation_null,
    )
    from .inference import (
        provided_null as validate_provided_null,
    )

    valid_sets = {
        result.set_name: sets[result.set_name] for result in results if result.status == "ok"
    }
    if null_method == "edge_permutation":
        null_scores = edge_permutation_null(
            ranked,
            valid_sets,
            n_permutations=n_permutations,
            random_state=random_state,
            weight=weight,
            score_type=score_type,
        )
        metadata["null_scope"] = "competitive_edge_label"
    elif null_method == "provided_null":
        if provided_null is None:
            raise ValueError("provided_null is required when null_method='provided_null'")
        if (
            provided_null_edge_ids is None
            or provided_null_edge_sets is None
            or provided_null_direction is None
            or positive_direction is None
        ):
            raise ValueError(
                "provided null inference requires edge IDs, edge sets, and both observed "
                "and null statistic directions"
            )
        null_edge_ids = [str(edge_id) for edge_id in provided_null_edge_ids]
        if len(null_edge_ids) != len(set(null_edge_ids)) or set(null_edge_ids) != universe:
            raise ValueError("provided null edge universe does not match observed edges")
        null_set_definitions = {
            str(name): {str(member) for member in members}
            for name, members in provided_null_edge_sets.items()
        }
        if null_set_definitions != valid_sets:
            raise ValueError("provided null set definitions do not match valid observed sets")
        if provided_null_direction != positive_direction:
            raise ValueError("provided null statistic direction does not match observed direction")
        canonical_by_id = dict(zip(ranked["edge_id"], ranked["canonical_edge_id"], strict=True))
        null_scores = validate_provided_null(
            provided_null,
            kind=provided_null_kind,
            edge_ids=null_edge_ids,
            canonical_edge_ids=[canonical_by_id[edge_id] for edge_id in null_edge_ids],
            edge_sets=valid_sets,
            weight=weight,
            score_type=score_type,
        )
        if set(null_scores) != set(valid_sets):
            raise ValueError("provided null set definitions do not match valid observed sets")
        lengths = {len(value) for value in null_scores.values()}
        metadata["n_permutations"] = lengths.pop() if lengths else 0
        metadata["provided_null_kind"] = provided_null_kind
        metadata["provided_null_edge_order"] = null_edge_ids
        metadata["provided_null_direction"] = provided_null_direction
        metadata["provided_null_validation"] = "edge_order_sets_and_direction_match"
    apply_null_inference(output, null_scores, correction_family_id=correction_family_id)
    metadata["inference_status"] = "complete"
    return output
