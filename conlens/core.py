"""Deterministic edge ranking and LENS set statistics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .data import validate_edge_table
from .results import EdgeStatistics, LensSetResult, LensStatResult
from .sets import validate_edge_sets

TOLERANCE = 1e-12


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def rank_edges(edges: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sort statistics descending with deterministic canonical-ID tie breaking."""
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
    return ranked, {
        "n_tied_edges": int(tied.sum()),
        "tied_edge_fraction": float(tied.mean()),
        "tie_method": "statistic_desc_then_canonical_edge_id_asc_stable",
    }


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
    positive, negative = float(np.max(values)), float(np.min(values))
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


def _coerce_edge_statistics(
    edge_statistics: EdgeStatistics | pd.DataFrame,
    *,
    positive_direction: str | None = None,
    directed: bool = False,
    diagonal: bool = False,
    nan_policy: str = "raise",
) -> EdgeStatistics:
    if isinstance(edge_statistics, EdgeStatistics):
        if positive_direction is not None:
            stored = edge_statistics.metadata.get("positive_direction")
            if stored is not None and stored != positive_direction:
                raise ValueError("positive_direction conflicts with edge-statistic metadata")
        frame = edge_statistics.table
        metadata = edge_statistics.metadata.copy()
        directed = bool(metadata.get("directed", directed))
        diagonal = bool(metadata.get("diagonal", diagonal))
    elif isinstance(edge_statistics, pd.DataFrame):
        frame = edge_statistics
        metadata = {}
    else:
        raise TypeError("edge_statistics must be EdgeStatistics or a pandas DataFrame")
    validated = validate_edge_table(
        frame,
        node_order=frame.attrs.get("node_order", metadata.get("node_order")),
        directed=directed,
        diagonal=diagonal,
        nan_policy=nan_policy,
    )
    direction = positive_direction or metadata.get("positive_direction")
    if not isinstance(direction, str) or not direction.strip():
        raise ValueError("positive_direction must be supplied for signed edge statistics")
    metadata.update(
        {
            "positive_direction": direction,
            "statistic_name": metadata.get("statistic_name", "signed edge statistic"),
            "node_order": validated.attrs.get("node_order", []),
            "directed": directed,
            "diagonal": diagonal,
        }
    )
    metadata.setdefault(
        "analysis_signature",
        {"kind": "external_edge_statistics", "name": metadata["statistic_name"]},
    )
    return EdgeStatistics(validated, metadata)


def make_edge_statistics(
    edges: pd.DataFrame,
    *,
    positive_direction: str,
    statistic_name: str = "statistic",
    directed: bool = False,
    diagonal: bool = False,
    nan_policy: str = "raise",
) -> EdgeStatistics:
    """Validate an externally computed signed edge-statistic table."""
    result = _coerce_edge_statistics(
        edges,
        positive_direction=positive_direction,
        directed=directed,
        diagonal=diagonal,
        nan_policy=nan_policy,
    )
    result.metadata.update(
        {
            "statistic_name": statistic_name,
            "analysis_signature": {"kind": "external_edge_statistics", "name": statistic_name},
        }
    )
    return result


def _score_set(
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


def _lens_stat_one(
    edge_statistics: EdgeStatistics | pd.DataFrame,
    edge_sets: Mapping[str, Iterable[str]],
    *,
    positive_direction: str | None,
    weight: float,
    score_type: str,
    store_running_sum: bool,
) -> LensStatResult:
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("weight must be a finite number >= 0")
    if score_type not in {"standard", "positive", "negative"}:
        raise ValueError("score_type must be 'standard', 'positive', or 'negative'")
    prepared = _coerce_edge_statistics(edge_statistics, positive_direction=positive_direction)
    ranked, ranking_metadata = rank_edges(prepared.table)
    universe = set(ranked["edge_id"])
    input_sets = {
        str(name): [str(member) for member in members] for name, members in edge_sets.items()
    }
    sets = validate_edge_sets(input_sets, universe)
    output: list[LensSetResult] = []
    for name, members in sets.items():
        if len(members) in {0, len(ranked)}:
            reason = "empty set" if not members else "full-universe set"
            output.append(
                LensSetResult(
                    set_name=name,
                    set_size_input=len(input_sets[name]),
                    set_size_effective=len(members),
                    ES=None,
                    ES_positive=None,
                    ES_negative=None,
                    status="invalid",
                    warnings=[reason],
                    edge_set_ids=sorted(members),
                )
            )
        else:
            output.append(
                _score_set(
                    name,
                    len(input_sets[name]),
                    members,
                    ranked,
                    weight=weight,
                    score_type=score_type,
                    store_running_sum=store_running_sum,
                )
            )
    canonical_column = "canonical_edge_id" if "canonical_edge_id" in ranked else "edge_id"
    edge_mapping = ranked[["edge_id", canonical_column]].sort_values("edge_id").to_dict("records")
    metadata = {
        **prepared.metadata,
        "edge_universe_hash": _hash_payload(sorted(universe)),
        "edge_mapping_hash": _hash_payload(edge_mapping),
        "edge_universe_size": len(universe),
        "set_definition_hash": _hash_payload(
            {name: sorted(members) for name, members in sorted(sets.items())}
        ),
        "weight_exponent": weight,
        "score_type": score_type,
        **ranking_metadata,
    }
    return LensStatResult(output, metadata, ranked)


def lens_stat(
    edge_statistics: EdgeStatistics | pd.DataFrame | Mapping[str, EdgeStatistics],
    edge_sets: Mapping[str, Iterable[str]],
    *,
    positive_direction: str | None = None,
    weight: float = 1.0,
    score_type: str = "standard",
    store_running_sum: bool = False,
) -> LensStatResult | dict[str, LensStatResult]:
    """Calculate the same deterministic LENS statistics for observed or null edges."""
    if isinstance(edge_statistics, Mapping) and not isinstance(edge_statistics, pd.DataFrame):
        if not edge_statistics:
            raise ValueError("edge_statistics mapping cannot be empty")
        return {
            str(name): _lens_stat_one(
                item,
                edge_sets,
                positive_direction=None,
                weight=weight,
                score_type=score_type,
                store_running_sum=store_running_sum,
            )
            for name, item in edge_statistics.items()
        }
    return _lens_stat_one(
        edge_statistics,
        edge_sets,
        positive_direction=positive_direction,
        weight=weight,
        score_type=score_type,
        store_running_sum=store_running_sum,
    )
