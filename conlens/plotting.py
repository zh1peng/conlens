"""Publication-oriented Matplotlib visualizations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import compute_running_sum
from .leading import compute_node_participation
from .results import LeadingNetwork, LensResult


def _axes(ax=None):
    return ax if ax is not None else plt.subplots()[1]


def plot_ranked_statistics(result: LensResult, ax=None):
    ax = _axes(ax)
    ranked = result.ranked_edges
    if ranked is None:
        raise ValueError("ranked edges are unavailable")
    ax.plot(np.arange(1, len(ranked) + 1), ranked["statistic"], color="0.2")
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.set(
        xlabel="Edge rank (positive → negative)",
        ylabel=result.metadata.get("ranking_statistic_name", "Statistic"),
    )
    return ax


def plot_running_sum(result: LensResult, set_name: str, ax=None):
    ax = _axes(ax)
    item = result.get(set_name)
    ranked = result.ranked_edges
    if ranked is None:
        raise ValueError("ranked edges are unavailable")
    if item.ES is None:
        raise ValueError("cannot plot an invalid or filtered edge set")
    if item.running_sum is None:
        if not item.edge_set_ids:
            raise ValueError("edge-set membership is unavailable for running-sum reconstruction")
        hits = ranked["edge_id"].isin(item.edge_set_ids)
        profile, _ = compute_running_sum(
            ranked["statistic"],
            hits,
            weight=float(result.metadata["weight_exponent"]),
        )
    else:
        profile = np.asarray(item.running_sum)
    ax.plot(np.arange(len(profile)), profile, color="#2166ac")
    ax.axhline(0, color="0.5", linewidth=0.8)
    if item.peak_rank is not None:
        ax.scatter([item.peak_rank], [profile[item.peak_rank]], color="#b2182b", zorder=3)
        if item.ES > 0:
            ax.axvspan(0, item.peak_rank, alpha=0.08, color="#b2182b")
        else:
            ax.axvspan(item.peak_rank, len(ranked), alpha=0.08, color="#2166ac")
    annotation = (
        f"ES={item.ES:.3g}  NES={item.NES if item.NES is not None else 'NA'}  "
        f"p={item.p_value if item.p_value is not None else 'NA'}  "
        f"q={item.q_value if item.q_value is not None else 'NA'}\n"
        f"set={item.set_size_effective}  leading={item.leading_edge_size}"
    )
    ax.text(0.02, 0.98, annotation, transform=ax.transAxes, va="top", fontsize=8)
    ax.set(xlabel="Edge rank (positive → negative)", ylabel="Running sum", title=set_name)
    return ax


def plot_hit_rug(result: LensResult, edge_set: set[str], ax=None):
    ax = _axes(ax)
    if result.ranked_edges is None:
        raise ValueError("ranked edges are unavailable")
    positions = np.flatnonzero(result.ranked_edges["edge_id"].isin(edge_set)) + 1
    ax.eventplot(positions, orientation="horizontal", colors="black")
    ax.set(xlabel="Edge rank", yticks=[])
    return ax


def plot_nes(result: LensResult, ax=None):
    ax = _axes(ax)
    frame = result.to_frame().dropna(subset=["NES"])
    ax.scatter(
        frame["NES"], np.arange(len(frame)), c=np.where(frame["NES"] >= 0, "#b2182b", "#2166ac")
    )
    ax.axvline(0, color="0.5", linewidth=0.8)
    ax.set(yticks=np.arange(len(frame)), yticklabels=frame["set_name"], xlabel="NES")
    return ax


def plot_network_pair_heatmap(values: pd.Series | Mapping[str, float], ax=None):
    ax = _axes(ax)
    series = pd.Series(values, dtype=float)
    separator = "->" if any("->" in name for name in series.index) else "--"
    pairs = [name.split(separator) for name in series.index]
    labels = sorted({value for pair in pairs for value in pair})
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    for (first, second), value in zip(pairs, series, strict=True):
        matrix.loc[first, second] = value
        if separator == "--":
            matrix.loc[second, first] = value
    image = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=range(len(labels)), xticklabels=labels, yticks=range(len(labels)), yticklabels=labels
    )
    return ax


def plot_leading_adjacency(network: LeadingNetwork, ax=None):
    ax = _axes(ax)
    graph = network.to_networkx()
    import networkx as nx

    nodes = list(graph.nodes)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight="statistic")
    image = ax.imshow(adjacency, cmap="coolwarm", aspect="equal")
    ax.figure.colorbar(image, ax=ax)
    ax.set(xticks=range(len(nodes)), xticklabels=nodes, yticks=range(len(nodes)), yticklabels=nodes)
    return ax


def plot_leading_connectome(
    network: LeadingNetwork, coordinates: Mapping[Any, tuple[float, float]], ax=None
):
    ax = _axes(ax)
    graph = network.to_networkx()
    import networkx as nx

    missing = set(graph) - set(coordinates)
    if missing:
        raise ValueError(f"missing coordinates for nodes: {sorted(missing, key=str)!r}")
    nx.draw_networkx(graph, pos=dict(coordinates), ax=ax, node_size=80, font_size=7)
    ax.set_axis_off()
    return ax


def plot_node_participation(network: LeadingNetwork, ax=None):
    ax = _axes(ax)
    frame = compute_node_participation(network).sort_values("degree", ascending=False)
    ax.bar(frame["node_id"].astype(str), frame["degree"])
    ax.tick_params(axis="x", rotation=90)
    ax.set(ylabel="Leading-edge degree", xlabel="Node")
    return ax


def plot_stability(summary: dict[str, Any], set_name: str, ax=None):
    ax = _axes(ax)
    frequencies = summary["sets"][set_name]["edge_inclusion_frequency"]
    ordered = sorted(frequencies.items(), key=lambda item: item[1], reverse=True)
    ax.bar([item[0] for item in ordered], [item[1] for item in ordered])
    ax.tick_params(axis="x", rotation=90)
    ax.set(ylim=(0, 1), ylabel="Bootstrap inclusion frequency", xlabel="Edge")
    return ax


def plot_enrichment(result: LensResult, set_name: str, edge_set: set[str], axes=None):
    if axes is None:
        _, axes = plt.subplots(3, 1, sharex=True, height_ratios=[2, 0.3, 1])
    plot_running_sum(result, set_name, axes[0])
    plot_hit_rug(result, edge_set, axes[1])
    plot_ranked_statistics(result, axes[2])
    return axes
