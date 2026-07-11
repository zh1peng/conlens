"""Leading-edge graph reconstruction and summaries."""

from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd

from .results import LeadingNetwork, LensResult


def build_leading_network(
    result: LensResult,
    set_name: str,
    *,
    node_metadata: pd.DataFrame | None = None,
) -> LeadingNetwork:
    set_result = result.get(set_name)
    if result.ranked_edges is None:
        raise ValueError("ranked_edges are required to reconstruct a leading network")
    edge_frame = result.ranked_edges[
        result.ranked_edges["edge_id"].isin(set_result.leading_edge_ids)
    ].copy()
    nodes = list(dict.fromkeys([*edge_frame["node1"].tolist(), *edge_frame["node2"].tolist()]))
    if node_metadata is None:
        node_frame = pd.DataFrame({"node_id": nodes})
    else:
        if "node_id" not in node_metadata:
            raise ValueError("node_metadata must contain node_id")
        if node_metadata["node_id"].duplicated().any():
            raise ValueError("node_metadata contains duplicate node_id values")
        missing = set(nodes) - set(node_metadata["node_id"])
        if missing:
            raise ValueError(f"node metadata missing leading nodes: {sorted(missing, key=str)!r}")
        node_frame = node_metadata[node_metadata["node_id"].isin(nodes)].copy()
    return LeadingNetwork(
        node_frame.reset_index(drop=True),
        edge_frame.reset_index(drop=True),
        bool(result.metadata.get("directed")),
    )


def compute_node_participation(network: LeadingNetwork) -> pd.DataFrame:
    graph = network.to_networkx()
    rows = []
    for node in graph.nodes:
        incident = network.edges[
            (network.edges["node1"] == node) | (network.edges["node2"] == node)
        ]
        weight_column = "statistic" if "statistic" in incident else "weight"
        strength = (
            float(incident[weight_column].abs().sum())
            if weight_column in incident
            else float(len(incident))
        )
        row: dict[str, Any] = {
            "node_id": node,
            "degree": int(graph.degree(node)),
            "strength": strength,
        }
        if network.directed:
            inbound = network.edges[network.edges["node2"] == node]
            outbound = network.edges[network.edges["node1"] == node]
            row.update(
                in_degree=int(graph.in_degree(node)),
                out_degree=int(graph.out_degree(node)),
                in_strength=(
                    float(inbound[weight_column].abs().sum())
                    if weight_column in inbound
                    else float(len(inbound))
                ),
                out_strength=(
                    float(outbound[weight_column].abs().sum())
                    if weight_column in outbound
                    else float(len(outbound))
                ),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def identify_leading_hubs(
    network: LeadingNetwork,
    *,
    metric: str = "degree",
    top_n: int | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    if (top_n is None) == (threshold is None):
        raise ValueError("specify exactly one of top_n or threshold")
    participation = compute_node_participation(network)
    if metric not in participation:
        raise ValueError(f"unknown participation metric {metric!r}")
    ranked = participation.sort_values(metric, ascending=False, kind="stable")
    return ranked.head(top_n) if top_n is not None else ranked[ranked[metric] >= threshold]


def summarize_leading_network(network: LeadingNetwork) -> dict[str, Any]:
    graph = network.to_networkx()
    if len(graph) == 0:
        components = 0
        density = 0.0
    else:
        components = (
            nx.number_weakly_connected_components(graph)
            if network.directed
            else nx.number_connected_components(graph)
        )
        density = float(nx.density(graph))
    return {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "density": density,
        "n_components": components,
        "directed": network.directed,
    }
