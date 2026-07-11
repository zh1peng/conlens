"""Construction and validation of edge sets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from .data import canonicalize_edges


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
