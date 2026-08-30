"""Deterministic edge ranking and LENS set statistics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data import _edge_identity_hash, _edge_universe_hash, validate_edge_table
from .results import (
    EdgeStatistics,
    LensSetResult,
    LensStatResult,
    NullEdgeStatistics,
    _NumericEdgeTemplate,
)
from .sets import validate_edge_sets

TOLERANCE = 1e-12


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _identity_token(value: Any) -> Any:
    """Represent node labels without conflating values such as ``1`` and ``"1"``."""
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_identity_token(item) for item in value]}
    if isinstance(value, np.generic):
        value = value.item()
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": repr(value),
    }


def _node_identity_hash(ranked: pd.DataFrame, metadata: Mapping[str, Any]) -> str:
    node_order = metadata.get("node_order", ranked.attrs.get("node_order", []))
    mapping = [
        {
            "edge_id": str(row.edge_id),
            "node1": _identity_token(row.node1),
            "node2": _identity_token(row.node2),
        }
        for row in ranked[["edge_id", "node1", "node2"]]
        .sort_values("edge_id")
        .itertuples(index=False)
    ]
    return _hash_payload(
        {
            "node_order": [_identity_token(node) for node in node_order],
            "directed": bool(metadata.get("directed", False)),
            "diagonal": bool(metadata.get("diagonal", False)),
            "edge_endpoints": mapping,
        }
    )


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
    values = np.asarray(
        statistics if isinstance(statistics, np.ndarray) else list(statistics), dtype=float
    )
    membership = np.asarray(hits if isinstance(hits, np.ndarray) else list(hits), dtype=bool)
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
    profile = np.asarray(
        running_sum if isinstance(running_sum, np.ndarray) else list(running_sum), dtype=float
    )
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
    identifiers = np.asarray(
        edge_ids if isinstance(edge_ids, np.ndarray) else list(edge_ids), dtype=object
    )
    membership = np.asarray(hits if isinstance(hits, np.ndarray) else list(hits), dtype=bool)
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


def make_null_edge_statistics(
    values: np.ndarray | pd.DataFrame,
    *,
    reference: EdgeStatistics,
    permutation_scheme: str,
    random_state: int | None = None,
    exchangeability_blocks_used: bool = False,
) -> NullEdgeStatistics:
    """Wrap an external edge-by-permutation matrix as a reusable null iterable.

    Rows must follow the ordered edge universe in ``reference``. A DataFrame must
    use edge IDs as its index and is reordered to the reference before its numeric
    values are used. Columns represent null replicates.
    """
    if not isinstance(reference, EdgeStatistics):
        raise TypeError("reference must be an EdgeStatistics object")
    if not isinstance(permutation_scheme, str) or not permutation_scheme.strip():
        raise ValueError("permutation_scheme must be a non-empty string")
    if random_state is not None and not isinstance(random_state, int):
        raise TypeError("random_state must be an integer or None")
    if not isinstance(exchangeability_blocks_used, bool):
        raise TypeError("exchangeability_blocks_used must be boolean")

    prepared_reference = _coerce_edge_statistics(reference)
    reference_table = prepared_reference.table
    reference_ids = reference_table["edge_id"].astype(str).tolist()
    if isinstance(values, pd.DataFrame):
        if not values.index.is_unique:
            raise ValueError("null edge-statistic DataFrame index must contain unique edge IDs")
        if not values.columns.is_unique:
            raise ValueError("null edge-statistic DataFrame columns must be unique")
        normalized = values.copy()
        normalized.index = normalized.index.map(str)
        if not normalized.index.is_unique:
            raise ValueError(
                "null edge-statistic DataFrame index is duplicated after string normalization"
            )
        supplied = set(normalized.index)
        expected = set(reference_ids)
        if supplied != expected:
            missing = sorted(expected - supplied)
            unknown = sorted(supplied - expected)
            raise ValueError(
                "null edge-statistic DataFrame has an incompatible edge universe "
                f"(missing={missing!r}, unknown={unknown!r})"
            )
        try:
            matrix = normalized.loc[reference_ids].apply(
                pd.to_numeric, errors="raise"
            ).to_numpy(dtype=float, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("null edge statistics must be numeric") from exc
    else:
        try:
            matrix = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("null edge statistics must be numeric") from exc

    if matrix.ndim != 2:
        raise ValueError("null edge statistics must have shape (edges, permutations)")
    if matrix.shape[0] != len(reference_ids):
        raise ValueError(
            "null edge-statistic rows must match the reference edge universe "
            f"({matrix.shape[0]} != {len(reference_ids)})"
        )
    if matrix.shape[1] < 1:
        raise ValueError("null edge statistics must contain at least one permutation")
    if not np.isfinite(matrix).all():
        raise ValueError("null edge statistics must contain only finite values")

    identity_columns = ["node1", "node2", "edge_id", "canonical_edge_id"]
    template_frame = reference_table[identity_columns].copy()
    template_frame.attrs.update(reference_table.attrs)
    metadata = {
        **prepared_reference.metadata,
        "source": "external_null_effects",
        "permutation_scheme": permutation_scheme.strip(),
        "random_seed": random_state,
        "exchangeability_blocks_used": exchangeability_blocks_used,
        "n_permutations": int(matrix.shape[1]),
    }
    return NullEdgeStatistics(
        np.asarray(matrix, dtype=float),
        _NumericEdgeTemplate(template_frame),
        metadata,
    )


@dataclass(slots=True)
class _NumericLensPlan:
    input_sets: dict[str, list[str]]
    sets: dict[str, set[str]]
    edge_ids: np.ndarray
    node1: np.ndarray
    node2: np.ndarray
    tie_ranks: np.ndarray
    membership: dict[str, np.ndarray]
    identity_metadata: dict[str, Any]


def _numeric_plan(
    template: _NumericEdgeTemplate,
    edge_sets: Mapping[str, Iterable[str]],
    metadata: Mapping[str, Any],
) -> _NumericLensPlan:
    input_sets = {
        str(name): [str(member) for member in members] for name, members in edge_sets.items()
    }
    cache_key = tuple((name, tuple(members)) for name, members in input_sets.items())
    cached = template.plans.get(cache_key)
    if cached is not None:
        if not isinstance(cached, _NumericLensPlan):
            raise RuntimeError("numeric LENS plan cache is corrupted")
        return cached
    frame = template.frame
    edge_ids = frame["edge_id"].astype(str).to_numpy(copy=True)
    universe = set(edge_ids.tolist())
    sets = validate_edge_sets(input_sets, universe)
    canonical_column = "canonical_edge_id" if "canonical_edge_id" in frame else "edge_id"
    edge_mapping = (
        frame[["edge_id", canonical_column]].sort_values("edge_id").to_dict("records")
    )
    plan = _NumericLensPlan(
        input_sets=input_sets,
        sets=sets,
        edge_ids=edge_ids,
        node1=frame["node1"].to_numpy(dtype=object, copy=True),
        node2=frame["node2"].to_numpy(dtype=object, copy=True),
        tie_ranks=np.argsort(
            np.argsort(
                frame[canonical_column].astype(str).to_numpy(copy=True), kind="stable"
            ),
            kind="stable",
        ),
        membership={
            name: np.isin(edge_ids, np.asarray(sorted(members), dtype=object))
            for name, members in sets.items()
        },
        identity_metadata={
            "edge_universe_hash": _edge_universe_hash(frame),
            "edge_identity_hash": _edge_identity_hash(frame),
            "edge_mapping_hash": _hash_payload(edge_mapping),
            "node_identity_hash": _node_identity_hash(frame, metadata),
            "edge_universe_size": len(universe),
            "set_definition_hash": _hash_payload(
                {name: sorted(members) for name, members in sorted(sets.items())}
            ),
        },
    )
    template.plans[cache_key] = plan
    return plan


def _numeric_tie_metadata(statistics: np.ndarray) -> dict[str, Any]:
    adjacent = statistics[:-1] == statistics[1:]
    tied = np.zeros(len(statistics), dtype=bool)
    tied[:-1] |= adjacent
    tied[1:] |= adjacent
    return {
        "n_tied_edges": int(tied.sum()),
        "tied_edge_fraction": float(tied.mean()),
        "tie_method": "statistic_desc_then_canonical_edge_id_asc_stable",
    }


def _lens_stat_numeric(
    edge_statistics: EdgeStatistics,
    template: _NumericEdgeTemplate,
    statistics: np.ndarray,
    edge_sets: Mapping[str, Iterable[str]],
    *,
    weight: float,
    score_type: str,
    store_running_sum: bool,
) -> LensStatResult:
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("weight must be a finite number >= 0")
    if score_type not in {"standard", "positive", "negative"}:
        raise ValueError("score_type must be 'standard', 'positive', or 'negative'")
    direction = edge_statistics.metadata.get("positive_direction")
    if not isinstance(direction, str) or not direction.strip():
        raise ValueError("positive_direction must be supplied for signed edge statistics")
    values = np.asarray(statistics, dtype=float)
    if values.ndim != 1 or len(values) != len(template.frame):
        raise ValueError("numeric edge statistics have an incompatible shape")
    if not np.isfinite(values).all():
        raise ValueError("statistics must be finite")
    if np.unique(values).size <= 1:
        raise ValueError("all statistics are identical; ranked list is not interpretable")
    plan = _numeric_plan(template, edge_sets, edge_statistics.metadata)
    order = np.lexsort((plan.tie_ranks, -values))
    ranked_statistics = values[order]
    ranked_edge_ids = plan.edge_ids[order]
    output: list[LensSetResult] = []
    for name, members in plan.sets.items():
        if len(members) in {0, len(values)}:
            reason = "empty set" if not members else "full-universe set"
            output.append(
                LensSetResult(
                    set_name=name,
                    set_size_input=len(plan.input_sets[name]),
                    set_size_effective=len(members),
                    ES=None,
                    ES_positive=None,
                    ES_negative=None,
                    status="invalid",
                    warnings=[reason],
                    edge_set_ids=sorted(members),
                )
            )
            continue
        hits = plan.membership[name][order]
        profile, fallback = compute_running_sum(ranked_statistics, hits, weight=weight)
        score = compute_enrichment_score(profile, score_type=score_type)
        if score["ES"] != 0 and score["peak_rank"] is not None:
            ranks = np.arange(1, len(values) + 1)
            selected = hits & (
                ranks <= score["peak_rank"]
                if score["ES"] > 0
                else ranks > score["peak_rank"]
            )
            leading_ids = ranked_edge_ids[selected].astype(str).tolist()
            selected_original = order[selected]
            leading_nodes = list(
                dict.fromkeys(
                    [
                        *plan.node1[selected_original].tolist(),
                        *plan.node2[selected_original].tolist(),
                    ]
                )
            )
        else:
            leading_ids = []
            leading_nodes = []
        output.append(
            LensSetResult(
                set_name=name,
                set_size_input=len(plan.input_sets[name]),
                set_size_effective=len(members),
                ES=score["ES"],
                ES_positive=score["ES_positive"],
                ES_negative=score["ES_negative"],
                direction=score["direction"],
                peak_rank=score["peak_rank"],
                peak_fraction=(
                    None if score["peak_rank"] is None else score["peak_rank"] / len(values)
                ),
                leading_edge_ids=leading_ids,
                leading_edge_size=len(leading_ids),
                leading_edge_fraction=len(leading_ids) / len(members),
                leading_node_ids=leading_nodes,
                zero_weight_fallback=fallback,
                edge_set_ids=sorted(members),
                running_sum=profile.tolist() if store_running_sum else None,
            )
        )
    metadata = {
        **edge_statistics.metadata,
        **plan.identity_metadata,
        "weight_exponent": weight,
        "score_type": score_type,
        **_numeric_tie_metadata(ranked_statistics),
    }
    return LensStatResult._from_numeric(
        output,
        metadata,
        template,
        order,
        ranked_statistics,
    )


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
    if isinstance(edge_statistics, EdgeStatistics):
        numeric = edge_statistics._numeric_parts()
        if numeric is not None:
            stored_direction = edge_statistics.metadata.get("positive_direction")
            if positive_direction is not None and stored_direction not in {
                None,
                positive_direction,
            }:
                raise ValueError("positive_direction conflicts with edge-statistic metadata")
            template, statistics = numeric
            return _lens_stat_numeric(
                edge_statistics,
                template,
                statistics,
                edge_sets,
                weight=weight,
                score_type=score_type,
                store_running_sum=store_running_sum,
            )
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
        "edge_universe_hash": _edge_universe_hash(ranked),
        "edge_identity_hash": _edge_identity_hash(ranked),
        "edge_mapping_hash": _hash_payload(edge_mapping),
        "node_identity_hash": _node_identity_hash(ranked, prepared.metadata),
        "edge_universe_size": len(universe),
        "set_definition_hash": _hash_payload(
            {name: sorted(members) for name, members in sorted(sets.items())}
        ),
        "weight_exponent": weight,
        "score_type": score_type,
        **ranking_metadata,
    }
    return LensStatResult(output, metadata, ranked)


def _attach_edge_set_construction(
    result: LensStatResult,
    edge_set_info: Mapping[str, Any],
) -> None:
    for key in ("edge_universe_hash", "edge_identity_hash"):
        expected = edge_set_info.get(key)
        observed = result.metadata.get(key)
        if expected is not None and expected != observed:
            raise ValueError(
                "edge sets were built for a different edge universe "
                f"({key}: {expected!r} != {observed!r})"
            )
    result.metadata["edge_set_construction"] = deepcopy(dict(edge_set_info))


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
    edge_set_info = getattr(edge_sets, "info", None)
    if isinstance(edge_statistics, Mapping) and not isinstance(edge_statistics, pd.DataFrame):
        if not edge_statistics:
            raise ValueError("edge_statistics mapping cannot be empty")
        results = {
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
        if isinstance(edge_set_info, Mapping):
            for result in results.values():
                _attach_edge_set_construction(result, edge_set_info)
        return results
    result = _lens_stat_one(
        edge_statistics,
        edge_sets,
        positive_direction=positive_direction,
        weight=weight,
        score_type=score_type,
        store_running_sum=store_running_sum,
    )
    if isinstance(edge_set_info, Mapping):
        _attach_edge_set_construction(result, edge_set_info)
    return result
