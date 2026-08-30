"""Construction and validation of edge sets."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    _edge_identity_hash,
    _edge_universe_hash,
    canonicalize_edges,
)
from .resources import NodeMaps


def _jsonable(value: Any) -> Any:
    if value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(init=False, slots=True)
class EdgeSets(Mapping[str, frozenset[str]]):
    """Named edge sets with construction metadata and an optional audit table."""

    members: dict[str, frozenset[str]]
    info: dict[str, Any]
    audit: pd.DataFrame

    def __init__(
        self,
        members: Mapping[str, Iterable[str]],
        info: Mapping[str, Any] | None = None,
        audit: pd.DataFrame | None = None,
    ) -> None:
        normalized: dict[str, frozenset[str]] = {}
        for name, values in members.items():
            set_name = str(name)
            if not set_name:
                raise ValueError("edge-set names must not be empty")
            if isinstance(values, str):
                raise TypeError(f"edge set {set_name!r} members must be an iterable of edge IDs")
            value_list = [str(value) for value in values]
            if len(value_list) != len(set(value_list)):
                raise ValueError(f"edge set {set_name!r} contains duplicate members")
            normalized[set_name] = frozenset(value_list)
        if len(normalized) != len(members):
            raise ValueError("edge-set names must be unique after string normalization")
        self.members = normalized
        self.info = {} if info is None else deepcopy(dict(info))
        self.audit = pd.DataFrame() if audit is None else audit.copy()

    @property
    def names(self) -> list[str]:
        return list(self.members)

    def __getitem__(self, name: str) -> frozenset[str]:
        return self.members[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.members)

    def __len__(self) -> int:
        return len(self.members)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "object_type": "EdgeSets",
            "members": {name: sorted(values) for name, values in self.members.items()},
            "info": _jsonable(self.info),
            "audit": {
                "columns": self.audit.columns.tolist(),
                "records": _jsonable(self.audit.to_dict(orient="records")),
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EdgeSets:
        if payload.get("schema_version") != 1 or payload.get("object_type") != "EdgeSets":
            raise ValueError("unsupported EdgeSets payload")
        audit_payload = payload.get("audit", {})
        if not isinstance(audit_payload, Mapping):
            raise ValueError("EdgeSets audit must be an object")
        audit = pd.DataFrame(
            audit_payload.get("records", []),
            columns=audit_payload.get("columns", []),
        )
        members = payload.get("members")
        if not isinstance(members, Mapping):
            raise ValueError("EdgeSets members must be an object")
        info = payload.get("info", {})
        if not isinstance(info, Mapping):
            raise ValueError("EdgeSets info must be an object")
        return cls(dict(members), dict(info), audit)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> EdgeSets:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("EdgeSets file must contain a JSON object")
        return cls.from_dict(payload)


def _coerce_maps(
    maps: NodeMaps | pd.DataFrame | pd.Series,
    map_names: Sequence[str] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if isinstance(maps, NodeMaps):
        frame = maps.data.copy()
        source_info = dict(maps.info)
    elif isinstance(maps, pd.Series):
        if maps.name is None:
            raise ValueError("a map Series must have a name")
        frame = maps.to_frame()
        source_info = {}
    elif isinstance(maps, pd.DataFrame):
        frame = maps.copy()
        if "node_id" in frame:
            frame = frame.set_index("node_id")
        source_info = {}
    else:
        raise TypeError("maps must be NodeMaps, a pandas DataFrame, or a pandas Series")
    if frame.index.hasnans or not frame.index.is_unique:
        raise ValueError("map node IDs must be present and unique")
    if not frame.columns.is_unique or any(not isinstance(name, str) for name in frame.columns):
        raise ValueError("map names must be unique strings")
    if map_names is None:
        selected_names = frame.columns.tolist()
    else:
        if isinstance(map_names, str):
            raise TypeError("map_names must be a sequence of names, not a string")
        selected_names = list(map_names)
        if len(selected_names) != len(set(selected_names)):
            raise ValueError("map_names must be unique")
        unknown = set(selected_names) - set(frame.columns)
        if unknown:
            raise KeyError(f"unknown maps: {sorted(unknown)!r}")
    if not selected_names:
        raise ValueError("at least one map must be selected")
    return frame[selected_names].copy(), source_info


def _numeric_map(values: pd.Series, *, missing: str) -> tuple[pd.Series, pd.Series]:
    if missing not in {"raise", "omit"}:
        raise ValueError("missing must be 'raise' or 'omit'")
    try:
        numeric = pd.to_numeric(values, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"map {values.name!r} must contain numeric values") from exc
    missing_mask = numeric.isna()
    finite = np.isfinite(numeric[~missing_mask].to_numpy(dtype=float))
    if not finite.all():
        raise ValueError(f"map {values.name!r} contains infinite values")
    if missing_mask.any() and missing == "raise":
        nodes = numeric.index[missing_mask].tolist()
        raise ValueError(f"map {values.name!r} contains missing nodes: {nodes!r}")
    return numeric, missing_mask


def _edge_universe(edges: pd.DataFrame, node_ids: pd.Index) -> pd.DataFrame:
    required = {"node1", "node2", "edge_id"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"edges are missing required columns: {sorted(missing)!r}")
    frame = edges.copy()
    if frame["edge_id"].isna().any():
        raise ValueError("edge_id values must not be missing")
    frame["edge_id"] = frame["edge_id"].astype(str)
    if (frame["edge_id"].str.len() == 0).any() or frame["edge_id"].duplicated().any():
        raise ValueError("edge_id values must be non-empty and unique")
    unknown = (set(frame["node1"]) | set(frame["node2"])) - set(node_ids)
    if unknown:
        raise ValueError(f"edges contain nodes absent from the maps: {sorted(unknown, key=str)!r}")
    return canonicalize_edges(
        frame,
        node_order=node_ids.tolist(),
        directed=bool(edges.attrs.get("directed", False)),
    )


def _map_content_hash(data: pd.DataFrame) -> str:
    values = []
    for row in data.itertuples(index=False, name=None):
        values.append(
            [None if pd.isna(value) else float(value).hex() for value in row]
        )
    payload = {
        "node_ids": [
            {
                "type": f"{type(node).__module__}.{type(node).__qualname__}",
                "value": repr(node),
            }
            for node in data.index
        ],
        "map_names": data.columns.tolist(),
        "values": values,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rank_nodes(values: pd.Series, *, descending: bool) -> pd.DataFrame:
    valid = values.dropna()
    ranked = pd.DataFrame(
        {
            "node_id": valid.index.tolist(),
            "value": valid.to_numpy(dtype=float),
            "_node_order": np.arange(len(valid)),
        }
    ).sort_values(
        ["value", "_node_order"],
        ascending=[not descending, True],
        kind="stable",
    )
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked.drop(columns="_node_order")


def _selection_count(size: int, fraction: float | None, count: int | None) -> int:
    if size < 1:
        raise ValueError("no nodes have finite map values")
    if fraction is not None:
        if not np.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("fraction must be a finite number in (0, 1]")
        return math.ceil(size * fraction)
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= size:
        raise ValueError(f"n_nodes must be an integer between 1 and {size}")
    return count


def _set_name(map_name: str, prefix: str | None) -> str:
    if prefix is None:
        return map_name
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("name_prefix must be a non-empty string or None")
    return f"{prefix}:{map_name}"


def make_node_value_sets(
    edges: pd.DataFrame,
    maps: NodeMaps | pd.DataFrame | pd.Series,
    *,
    map_names: Sequence[str] | None = None,
    keep: str = "highest",
    fraction: float | None = None,
    n_nodes: int | None = None,
    cutoff: float | None = None,
    connect: str = "within",
    missing: str = "raise",
    name_prefix: str | None = None,
) -> EdgeSets:
    """Create edge sets from nodes with high or low values in each map."""
    if keep not in {"highest", "lowest"}:
        raise ValueError("keep must be 'highest' or 'lowest'")
    if connect not in {"within", "touching", "between"}:
        raise ValueError("connect must be 'within', 'touching', or 'between'")
    supplied = sum(item is not None for item in (fraction, n_nodes, cutoff))
    if supplied != 1:
        raise ValueError("provide exactly one of fraction, n_nodes, or cutoff")
    if cutoff is not None and not np.isfinite(cutoff):
        raise ValueError("cutoff must be finite")

    data, source_info = _coerce_maps(maps, map_names)
    universe = _edge_universe(edges, data.index)
    members: dict[str, frozenset[str]] = {}
    audit_parts: list[pd.DataFrame] = []
    set_info: dict[str, Any] = {}
    descending = keep == "highest"

    for map_name in data.columns:
        values, missing_mask = _numeric_map(data[map_name], missing=missing)
        ranked = _rank_nodes(values, descending=descending)
        if cutoff is None:
            count = _selection_count(len(ranked), fraction, n_nodes)
            selected = set(ranked.iloc[:count]["node_id"])
        elif keep == "highest":
            selected = set(values.index[(values >= cutoff) & ~missing_mask])
        else:
            selected = set(values.index[(values <= cutoff) & ~missing_mask])

        first = universe["node1"].isin(selected)
        second = universe["node2"].isin(selected)
        if connect == "within":
            edge_mask = first & second
        elif connect == "touching":
            edge_mask = first | second
        else:
            edge_mask = first ^ second
        set_name = _set_name(map_name, name_prefix)
        members[set_name] = frozenset(universe.loc[edge_mask, "edge_id"])

        rank_lookup = ranked.set_index("node_id")["rank"]
        audit = pd.DataFrame(
            {
                "set_name": set_name,
                "map_name": map_name,
                "node_id": values.index,
                "value": values.to_numpy(),
                "rank": values.index.map(rank_lookup),
                "selected": values.index.isin(selected),
            }
        )
        audit_parts.append(audit)
        set_info[set_name] = {
            "map_name": map_name,
            "selected_nodes": len(selected),
            "selected_edges": int(edge_mask.sum()),
            "missing_nodes": int(missing_mask.sum()),
        }

    if fraction is not None:
        selection = {"type": "fraction", "value": fraction, "rounding": "ceil"}
    elif n_nodes is not None:
        selection = {"type": "n_nodes", "value": n_nodes}
    else:
        selection = {"type": "cutoff", "value": cutoff}
    return EdgeSets(
        members,
        info={
            "method": "node_values",
            "maps": source_info,
            "map_data_sha256": _map_content_hash(data),
            "map_names": data.columns.tolist(),
            "keep": keep,
            "selection": selection,
            "connect": connect,
            "missing": missing,
            "tie_break": "map_node_order",
            "edge_universe_size": len(universe),
            "edge_universe_hash": _edge_universe_hash(universe),
            "edge_identity_hash": _edge_identity_hash(universe),
            "sets": set_info,
        },
        audit=pd.concat(audit_parts, ignore_index=True),
    )


def _select_edges_by_score(
    edges: pd.DataFrame,
    scores: np.ndarray,
    *,
    prefer_larger: bool,
    edge_fraction: float | None,
    cutoff: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(scores)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        raise ValueError("no edges have a finite map-based score")
    if (edge_fraction is None) == (cutoff is None):
        raise ValueError("provide exactly one of edge_fraction or cutoff")
    if cutoff is not None:
        if not np.isfinite(cutoff):
            raise ValueError("cutoff must be finite")
        selected = valid & (scores >= cutoff if prefer_larger else scores <= cutoff)
    else:
        assert edge_fraction is not None
        if not np.isfinite(edge_fraction) or not 0 < edge_fraction <= 1:
            raise ValueError("edge_fraction must be a finite number in (0, 1]")
        count = math.ceil(len(valid_indices) * edge_fraction)
        ranking = pd.DataFrame(
            {
                "row": valid_indices,
                "score": scores[valid],
                "tie": edges.iloc[valid_indices]["canonical_edge_id"].astype(str).to_numpy(),
            }
        ).sort_values(
            ["score", "tie"],
            ascending=[not prefer_larger, True],
            kind="stable",
        )
        selected = np.zeros(len(edges), dtype=bool)
        selected[ranking.iloc[:count]["row"].to_numpy(dtype=int)] = True
    ranks = np.full(len(edges), np.nan)
    ranking = pd.DataFrame(
        {
            "row": valid_indices,
            "score": scores[valid],
            "tie": edges.iloc[valid_indices]["canonical_edge_id"].astype(str).to_numpy(),
        }
    ).sort_values(
        ["score", "tie"],
        ascending=[not prefer_larger, True],
        kind="stable",
    )
    ranks[ranking["row"].to_numpy(dtype=int)] = np.arange(1, len(ranking) + 1)
    return selected, ranks


def make_node_distance_sets(
    edges: pd.DataFrame,
    node_map: pd.Series,
    *,
    keep: str = "closest",
    edge_fraction: float | None = None,
    cutoff: float | None = None,
    value_scale: str = "raw",
    missing: str = "raise",
    name: str | None = None,
) -> EdgeSets:
    """Create an edge set from distances between node values in one map."""
    if not isinstance(node_map, pd.Series):
        raise TypeError("node_map must be a pandas Series")
    if not isinstance(node_map.name, str) or not node_map.name:
        raise ValueError("node_map must have a non-empty string name")
    if keep not in {"closest", "farthest"}:
        raise ValueError("keep must be 'closest' or 'farthest'")
    if value_scale not in {"raw", "rank"}:
        raise ValueError("value_scale must be 'raw' or 'rank'")
    if cutoff is not None and cutoff < 0:
        raise ValueError("distance cutoff must be >= 0")
    values, missing_mask = _numeric_map(node_map, missing=missing)
    universe = _edge_universe(edges, values.index)
    scaled = values.copy()
    if value_scale == "rank":
        scaled.loc[~missing_mask] = values.loc[~missing_mask].rank(method="average", pct=True)
    first = universe["node1"].map(scaled).to_numpy(dtype=float)
    second = universe["node2"].map(scaled).to_numpy(dtype=float)
    distances = np.abs(first - second)
    selected, ranks = _select_edges_by_score(
        universe,
        distances,
        prefer_larger=keep == "farthest",
        edge_fraction=edge_fraction,
        cutoff=cutoff,
    )
    set_name = f"{node_map.name}:{keep}" if name is None else name
    if not isinstance(set_name, str) or not set_name:
        raise ValueError("name must be a non-empty string or None")
    selection = (
        {"type": "edge_fraction", "value": edge_fraction, "rounding": "ceil"}
        if edge_fraction is not None
        else {"type": "cutoff", "value": cutoff}
    )
    audit = universe[["edge_id", "node1", "node2"]].copy()
    audit.insert(0, "set_name", set_name)
    audit["distance"] = distances
    audit["rank"] = ranks
    audit["selected"] = selected
    return EdgeSets(
        {set_name: frozenset(universe.loc[selected, "edge_id"])},
        info={
            "method": "node_distance",
            "map_name": str(node_map.name),
            "map_data_sha256": _map_content_hash(node_map.to_frame()),
            "keep": keep,
            "selection": selection,
            "value_scale": value_scale,
            "distance": "absolute_difference",
            "missing": missing,
            "missing_nodes": int(missing_mask.sum()),
            "candidate_edges": int(np.isfinite(distances).sum()),
            "selected_edges": int(selected.sum()),
            "tie_break": "canonical_edge_id",
            "selection_unit": "directed_edge"
            if bool(universe.attrs.get("directed", False))
            else "undirected_edge",
            "edge_universe_size": len(universe),
            "edge_universe_hash": _edge_universe_hash(universe),
            "edge_identity_hash": _edge_identity_hash(universe),
        },
        audit=audit,
    )


def make_profile_similarity_sets(
    edges: pd.DataFrame,
    maps: NodeMaps | pd.DataFrame,
    *,
    map_names: Sequence[str] | None = None,
    metric: str = "pearson",
    keep: str = "most_similar",
    edge_fraction: float | None = None,
    cutoff: float | None = None,
    map_scaling: str = "none",
    name: str | None = None,
) -> EdgeSets:
    """Create an edge set from similarity between multi-map node profiles."""
    if metric not in {"pearson", "cosine", "euclidean"}:
        raise ValueError("metric must be 'pearson', 'cosine', or 'euclidean'")
    if keep not in {"most_similar", "least_similar"}:
        raise ValueError("keep must be 'most_similar' or 'least_similar'")
    if map_scaling not in {"none", "zscore"}:
        raise ValueError("map_scaling must be 'none' or 'zscore'")
    if cutoff is not None:
        if metric in {"pearson", "cosine"} and not -1 <= cutoff <= 1:
            raise ValueError(f"{metric} cutoff must be between -1 and 1")
        if metric == "euclidean" and cutoff < 0:
            raise ValueError("euclidean cutoff must be >= 0")
    data, source_info = _coerce_maps(maps, map_names)
    if data.shape[1] < 2:
        raise ValueError("profile similarity requires at least two maps")
    numeric = pd.DataFrame(index=data.index)
    for map_name in data.columns:
        values, missing_mask = _numeric_map(data[map_name], missing="raise")
        if missing_mask.any():
            raise ValueError("profile similarity does not support missing values")
        numeric[map_name] = values
    if map_scaling == "zscore":
        standard_deviation = numeric.std(axis=0, ddof=0)
        constant = standard_deviation == 0
        if constant.any():
            names = numeric.columns[constant].tolist()
            raise ValueError(f"cannot z-score constant maps: {names!r}")
        numeric = (numeric - numeric.mean(axis=0)) / standard_deviation

    universe = _edge_universe(edges, numeric.index)
    left = numeric.loc[universe["node1"]].to_numpy(dtype=float)
    right = numeric.loc[universe["node2"]].to_numpy(dtype=float)
    if metric == "pearson":
        left = left - left.mean(axis=1, keepdims=True)
        right = right - right.mean(axis=1, keepdims=True)
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        if (denominator == 0).any():
            bad = universe.loc[denominator == 0, "edge_id"].tolist()
            raise ValueError(f"Pearson similarity is undefined for constant profiles: {bad!r}")
        scores = np.sum(left * right, axis=1) / denominator
        larger_is_similar = True
    elif metric == "cosine":
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        if (denominator == 0).any():
            bad = universe.loc[denominator == 0, "edge_id"].tolist()
            raise ValueError(f"cosine similarity is undefined for zero profiles: {bad!r}")
        scores = np.sum(left * right, axis=1) / denominator
        larger_is_similar = True
    else:
        scores = np.linalg.norm(left - right, axis=1)
        larger_is_similar = False
    prefer_larger = larger_is_similar == (keep == "most_similar")
    selected, ranks = _select_edges_by_score(
        universe,
        scores,
        prefer_larger=prefer_larger,
        edge_fraction=edge_fraction,
        cutoff=cutoff,
    )
    set_name = f"profile:{keep}" if name is None else name
    if not isinstance(set_name, str) or not set_name:
        raise ValueError("name must be a non-empty string or None")
    selection = (
        {"type": "edge_fraction", "value": edge_fraction, "rounding": "ceil"}
        if edge_fraction is not None
        else {"type": "cutoff", "value": cutoff}
    )
    audit = universe[["edge_id", "node1", "node2"]].copy()
    audit.insert(0, "set_name", set_name)
    audit["metric_value"] = scores
    audit["rank"] = ranks
    audit["selected"] = selected
    return EdgeSets(
        {set_name: frozenset(universe.loc[selected, "edge_id"])},
        info={
            "method": "profile_similarity",
            "maps": source_info,
            "map_data_sha256": _map_content_hash(data),
            "map_names": data.columns.tolist(),
            "metric": metric,
            "keep": keep,
            "selection": selection,
            "map_scaling": map_scaling,
            "candidate_edges": len(universe),
            "selected_edges": int(selected.sum()),
            "tie_break": "canonical_edge_id",
            "selection_unit": "directed_edge"
            if bool(universe.attrs.get("directed", False))
            else "undirected_edge",
            "edge_universe_size": len(universe),
            "edge_universe_hash": _edge_universe_hash(universe),
            "edge_identity_hash": _edge_identity_hash(universe),
        },
        audit=audit,
    )


def validate_edge_sets(
    edge_sets: Mapping[str, Iterable[str]],
    universe: Iterable[str],
    *,
    unknown_policy: str = "raise",
) -> dict[str, set[str]]:
    if unknown_policy not in {"raise", "omit"}:
        raise ValueError("unknown_policy must be 'raise' or 'omit'")
    universe_ids = set(universe)
    validated: dict[str, set[str]] = {}
    for name, members in edge_sets.items():
        member_list = list(members)
        member_ids = set(member_list)
        if len(member_ids) != len(member_list):
            raise ValueError(f"edge set {name!r} contains duplicate members")
        unknown = member_ids - universe_ids
        if unknown and unknown_policy == "raise":
            raise ValueError(f"edge set {name!r} contains unknown edges: {sorted(unknown)!r}")
        validated[str(name)] = member_ids & universe_ids
    return validated


def make_custom_edge_sets(
    definitions: Mapping[str, pd.DataFrame | Iterable[str]],
    edges: pd.DataFrame,
    *,
    directed: bool = False,
) -> dict[str, set[str]]:
    universe = set(edges["edge_id"])
    result: dict[str, set[str]] = {}
    node_order = edges.attrs.get("node_order")
    for name, definition in definitions.items():
        if isinstance(definition, pd.DataFrame):
            canonical = canonicalize_edges(
                definition[["node1", "node2"]],
                node_order=node_order,
                directed=directed,
            )
            canonical_column = "canonical_edge_id" if "canonical_edge_id" in edges else "edge_id"
            identity_map = dict(zip(edges[canonical_column], edges["edge_id"], strict=True))
            unknown_canonical = set(canonical["canonical_edge_id"]) - set(identity_map)
            if unknown_canonical:
                raise ValueError(
                    f"custom set {name!r} contains unknown edges: {sorted(unknown_canonical)!r}"
                )
            members = {identity_map[item] for item in canonical["canonical_edge_id"]}
        else:
            members = set(definition)
        unknown = members - universe
        if unknown:
            raise ValueError(f"custom set {name!r} contains unknown edges: {sorted(unknown)!r}")
        result[str(name)] = members
    return result


def make_network_pair_sets(
    edges: pd.DataFrame,
    node_networks: Mapping[Any, str] | pd.Series,
    *,
    directed: bool = False,
) -> dict[str, set[str]]:
    labels = dict(node_networks)
    nodes = set(edges["node1"]) | set(edges["node2"])
    missing = nodes - set(labels)
    if missing:
        raise ValueError(f"missing network labels for nodes: {sorted(missing, key=str)!r}")
    sets: dict[str, set[str]] = {}
    for row in edges[["node1", "node2", "edge_id"]].itertuples(index=False):
        first, second = str(labels[row.node1]), str(labels[row.node2])
        if directed:
            name = f"{first}->{second}"
        else:
            first, second = sorted((first, second))
            name = f"{first}--{second}"
        sets.setdefault(name, set()).add(row.edge_id)
    return sets


def make_within_network_sets(
    edges: pd.DataFrame, node_networks: Mapping[Any, str] | pd.Series
) -> dict[str, set[str]]:
    return {
        name: members
        for name, members in make_network_pair_sets(edges, node_networks).items()
        if name.split("--")[0] == name.split("--")[1]
    }


def make_hemisphere_sets(
    edges: pd.DataFrame,
    node_hemispheres: Mapping[Any, str] | pd.Series,
    *,
    directed: bool = False,
) -> dict[str, set[str]]:
    return make_network_pair_sets(edges, node_hemispheres, directed=directed)
