"""Matplotlib visualizations for designs, enrichment, and leading networks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path

from .core import compute_running_sum
from .design import Contrast, DesignMatrix
from .leading import compute_node_participation
from .results import LeadingNetwork, LensResult, LensStabilityResult

DEFAULT_NETWORK_COLORS = (
    "#173F67", "#D94A4A", "#E8A33A", "#2E8B57", "#6B5CA5",
    "#2A9DAB", "#A46350", "#7A8796", "#C86B98", "#8C6D31",
)


def _axes(ax=None):
    return ax if ax is not None else plt.subplots()[1]


def _network_palette(labels: Sequence[str]) -> dict[str, str]:
    unique = list(dict.fromkeys(labels))
    return {
        label: DEFAULT_NETWORK_COLORS[index % len(DEFAULT_NETWORK_COLORS)]
        for index, label in enumerate(unique)
    }


def plot_design(
    design: DesignMatrix,
    contrasts: Mapping[str, Contrast],
    axes=None,
):
    """Show a validated design matrix beside its named contrast vectors."""
    if not isinstance(design, DesignMatrix):
        raise TypeError("design must be created with make_design()")
    if not contrasts:
        raise ValueError("contrasts must not be empty")
    if any(not isinstance(item, Contrast) for item in contrasts.values()):
        raise TypeError("every contrast specification must be a Contrast object")
    if axes is None:
        _, axes = plt.subplots(
            1, 2, figsize=(max(7, design.n_columns * 0.8 + 2), 5),
            gridspec_kw={"width_ratios": [2.4, 1]}, constrained_layout=True,
        )
    design_ax, contrast_ax = np.asarray(axes, dtype=object).reshape(-1)
    values = design.values
    scale = np.max(np.abs(values), axis=0)
    display = values / np.where(scale == 0, 1.0, scale)
    image = design_ax.imshow(display, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    design_ax.set(
        title=(f"Design matrix · n={design.n_observations}, rank={design.n_columns}\n"
               f"condition number={design.condition_number:.3g}"),
        xlabel="Regressors", ylabel="Observations",
        xticks=np.arange(design.n_columns), xticklabels=design.columns,
    )
    design_ax.tick_params(axis="x", rotation=45)
    design_ax.figure.colorbar(image, ax=design_ax, shrink=0.72, label="Column-scaled value")
    names = list(contrasts)
    weights = np.vstack([contrasts[name].resolve(design) for name in names])
    maximum = max(float(np.max(np.abs(weights))), 1e-12)
    image = contrast_ax.imshow(
        weights, cmap="RdBu_r", vmin=-maximum, vmax=maximum, aspect="auto"
    )
    contrast_ax.set(
        title="Contrasts", xlabel="Regressors", xticks=np.arange(design.n_columns),
        xticklabels=design.columns, yticks=np.arange(len(names)), yticklabels=names,
    )
    contrast_ax.tick_params(axis="x", rotation=45)
    if weights.size <= 80:
        for row, column in np.ndindex(weights.shape):
            value = weights[row, column]
            contrast_ax.text(
                column, row, f"{value:g}", ha="center", va="center",
                color="white" if abs(value) > maximum * 0.55 else "0.15", fontsize=8,
            )
    contrast_ax.figure.colorbar(image, ax=contrast_ax, shrink=0.72, label="Contrast weight")
    return np.asarray([design_ax, contrast_ax], dtype=object)


def plot_connectome_heatmap(
    matrix: np.ndarray,
    node_networks: Mapping[Any, str] | Sequence[str],
    *,
    node_labels: Sequence[Any] | None = None,
    network_order: Sequence[str] | None = None,
    cmap: str = "RdBu_r",
    ax=None,
):
    """Plot a connectome reordered by network with top and left annotations."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    labels = list(range(len(values))) if node_labels is None else list(node_labels)
    if len(labels) != len(values):
        raise ValueError("node_labels must match matrix dimensions")
    if isinstance(node_networks, Mapping):
        missing = set(labels) - set(node_networks)
        if missing:
            raise ValueError(f"missing network annotations: {sorted(missing, key=str)!r}")
        networks = [str(node_networks[label]) for label in labels]
    else:
        networks = [str(value) for value in node_networks]
        if len(networks) != len(labels):
            raise ValueError("node_networks must contain one value per node")
    order_names = (
        list(dict.fromkeys(networks)) if network_order is None else list(network_order)
    )
    unknown = set(networks) - set(order_names)
    if unknown:
        raise ValueError(f"network_order omits annotations: {sorted(unknown)!r}")
    rank = {name: index for index, name in enumerate(order_names)}
    order = sorted(range(len(labels)), key=lambda index: (rank[networks[index]], index))
    ordered_values = values[np.ix_(order, order)]
    ordered_networks = [networks[index] for index in order]
    palette = _network_palette(order_names)
    if ax is None:
        figure = plt.figure(figsize=(7, 7), constrained_layout=True)
        grid = figure.add_gridspec(2, 2, width_ratios=[0.045, 1], height_ratios=[0.045, 1])
        top_ax = figure.add_subplot(grid[0, 1])
        left_ax = figure.add_subplot(grid[1, 0])
        ax = figure.add_subplot(grid[1, 1])
    else:
        top_ax = ax.inset_axes([0, 1.01, 1, 0.035])
        left_ax = ax.inset_axes([-0.045, 0, 0.035, 1])
    color_index = {name: index for index, name in enumerate(order_names)}
    annotation = np.asarray([color_index[name] for name in ordered_networks])
    annotation_cmap = ListedColormap([palette[name] for name in order_names])
    top_ax.imshow(annotation[None, :], aspect="auto", cmap=annotation_cmap)
    left_ax.imshow(annotation[:, None], aspect="auto", cmap=annotation_cmap)
    top_ax.set_axis_off()
    left_ax.set_axis_off()
    maximum = np.nanmax(np.abs(ordered_values))
    image = ax.imshow(
        ordered_values, cmap=cmap, vmin=-maximum, vmax=maximum,
        interpolation="nearest", aspect="equal",
    )
    boundaries = [
        index - 0.5 for index in range(1, len(order))
        if ordered_networks[index] != ordered_networks[index - 1]
    ]
    for boundary in boundaries:
        ax.axvline(boundary, color="white", linewidth=0.8, alpha=0.8)
        ax.axhline(boundary, color="white", linewidth=0.8, alpha=0.8)
    centers = []
    for name in order_names:
        positions = [i for i, current in enumerate(ordered_networks) if current == name]
        if positions:
            centers.append((name, float(np.mean(positions))))
    ax.set(
        xticks=[center for _, center in centers],
        xticklabels=[name for name, _ in centers],
        yticks=[center for _, center in centers],
        yticklabels=[name for name, _ in centers],
        xlabel="Network", ylabel="Network",
    )
    ax.tick_params(axis="x", labelrotation=45)
    ax.figure.colorbar(image, ax=ax, shrink=0.76, label="Connectivity")
    return ax


def _running_sum(result: LensResult, set_name: str) -> tuple[Any, np.ndarray]:
    item = result.get(set_name)
    if item.ES is None or item.status == "invalid":
        raise ValueError("cannot plot an invalid edge set")
    if item.running_sum is not None:
        return item, np.asarray(item.running_sum, dtype=float)
    profile, _ = compute_running_sum(
        result.ranked_edges["statistic"],
        result.ranked_edges["edge_id"].isin(item.edge_set_ids),
        weight=float(result.metadata["weight_exponent"]),
    )
    return item, profile


def plot_running_sum(result: LensResult, set_name: str, ax=None):
    """Plot the enrichment walk, its extremum, and leading-edge interval."""
    ax = _axes(ax)
    item, profile = _running_sum(result, set_name)
    ranks = np.arange(len(profile))
    ax.plot(ranks, profile, color="#173F67", linewidth=2)
    ax.axhline(0, color="0.68", linewidth=0.8)
    if item.peak_rank is not None:
        peak_color = "#D94A4A" if item.ES > 0 else "#2A6F9E"
        ax.scatter(item.peak_rank, profile[item.peak_rank], color=peak_color, s=32, zorder=3)
        interval = (0, item.peak_rank) if item.ES > 0 else (item.peak_rank, len(profile) - 1)
        ax.axvspan(*interval, color=peak_color, alpha=0.08)
    hits = np.flatnonzero(result.ranked_edges["edge_id"].isin(item.edge_set_ids)) + 1
    lower, upper = ax.get_ylim()
    rug_height = (upper - lower) * 0.035
    ax.vlines(hits, lower, lower + rug_height, color="0.25", linewidth=0.5, alpha=0.55)
    ax.set(
        xlabel="Ranked edges: positive → negative", ylabel="Running sum",
        title=f"{set_name} · ES={item.ES:.3f}" +
        ("" if item.NES is None else f" · NES={item.NES:.3f}"),
    )
    return ax


def plot_null_distribution(result: LensResult, set_name: str, ax=None):
    """Compare the observed set ES with its retained null ES distribution."""
    ax = _axes(ax)
    item = result.get(set_name)
    if item.ES is None:
        raise ValueError("observed ES is unavailable")
    null = result.null_for(set_name).dropna().to_numpy(float)
    ax.hist(null, bins="auto", color="#B8C6D4", edgecolor="white", linewidth=0.5)
    ax.axvline(float(item.ES), color="#D94A4A", linewidth=2, label=f"Observed ES={item.ES:.3f}")
    ax.axvline(0, color="0.45", linewidth=0.8)
    ax.set(xlabel="Null enrichment score", ylabel="Count", title=set_name)
    ax.legend(frameon=False)
    return ax


def plot_enrichment(result: LensResult, set_name: str, axes=None):
    """Show running sum, ranked edge statistics, and the null ES distribution."""
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    running_ax, ranked_ax, null_ax = np.asarray(axes, dtype=object).reshape(-1)
    plot_running_sum(result, set_name, running_ax)
    ranked_ax.plot(
        np.arange(1, len(result.ranked_edges) + 1),
        result.ranked_edges["statistic"], color="#173F67", linewidth=1.3,
    )
    ranked_ax.axhline(0, color="0.65", linewidth=0.8)
    ranked_ax.set(xlabel="Edge rank", ylabel=result.metadata.get("statistic_name", "Statistic"))
    plot_null_distribution(result, set_name, null_ax)
    return np.asarray([running_ax, ranked_ax, null_ax], dtype=object)


def plot_enrichment_heatmap(
    result: LensResult,
    *,
    value: str = "NES",
    significance_alpha: float = 0.05,
    annotate: bool = True,
    ax=None,
):
    """Display network-pair enrichment results as a symmetric or directed heatmap."""
    if value not in {"ES", "NES"}:
        raise ValueError("value must be 'ES' or 'NES'")
    if not 0 < significance_alpha < 1:
        raise ValueError("significance_alpha must be in (0, 1)")
    frame = result.to_frame()
    separator = "->" if any("->" in name for name in frame["set_name"]) else "--"
    pairs = [name.split(separator) for name in frame["set_name"]]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("all set names must be network pairs such as 'A--B' or 'A->B'")
    labels = list(dict.fromkeys(value for pair in pairs for value in pair))
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    q_values = pd.DataFrame(np.nan, index=labels, columns=labels)
    for pair, row in zip(pairs, frame.to_dict("records"), strict=True):
        first, second = pair
        matrix.loc[first, second] = row[value]
        q_values.loc[first, second] = row["q_value"]
        if separator == "--":
            matrix.loc[second, first] = row[value]
            q_values.loc[second, first] = row["q_value"]
    ax = _axes(ax)
    finite = np.asarray(matrix, float)
    maximum = np.nanmax(np.abs(finite)) if np.isfinite(finite).any() else 1.0
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-maximum, vmax=maximum)
    ax.set(
        xticks=np.arange(len(labels)), xticklabels=labels,
        yticks=np.arange(len(labels)), yticklabels=labels,
        xlabel="Network", ylabel="Network",
    )
    ax.tick_params(axis="x", labelrotation=45)
    if annotate:
        for row, column in np.ndindex(matrix.shape):
            cell = matrix.iloc[row, column]
            if pd.isna(cell):
                continue
            significant = q_values.iloc[row, column] <= significance_alpha
            label = f"{cell:.2f}" + ("*" if significant else "")
            ax.text(column, row, label, ha="center", va="center", fontsize=8)
    ax.figure.colorbar(image, ax=ax, shrink=0.78, label=value)
    return ax


def plot_circos(
    network: LeadingNetwork,
    node_networks: Mapping[Any, str],
    *,
    ax=None,
):
    """Plot a leading-edge network as network-grouped circular chords."""
    nodes = network.nodes["node_id"].tolist()
    missing = set(nodes) - set(node_networks)
    if missing:
        raise ValueError(f"missing network annotations: {sorted(missing, key=str)!r}")
    groups = [str(node_networks[node]) for node in nodes]
    group_order = list(dict.fromkeys(groups))
    ordered_nodes = sorted(
        nodes,
        key=lambda node: (group_order.index(str(node_networks[node])), str(node)),
    )
    ordered_groups = [str(node_networks[node]) for node in ordered_nodes]
    palette = _network_palette(group_order)
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(nodes), endpoint=False)
    positions = {
        node: np.asarray([np.cos(angle), np.sin(angle)])
        for node, angle in zip(ordered_nodes, angles, strict=True)
    }
    ax = _axes(ax)
    ax.set_aspect("equal")
    for group in group_order:
        indices = [i for i, current in enumerate(ordered_groups) if current == group]
        start = np.degrees(angles[min(indices)] - np.pi / len(nodes) * 0.8)
        end = np.degrees(angles[max(indices)] + np.pi / len(nodes) * 0.8)
        ax.add_patch(Wedge((0, 0), 1.12, start, end, width=0.075,
                           facecolor=palette[group], edgecolor="none"))
        middle = np.mean(angles[indices])
        ax.text(1.27 * np.cos(middle), 1.27 * np.sin(middle), group,
                ha="center", va="center", fontsize=9, color=palette[group])
    if len(network.edges):
        weights = (
            network.edges["statistic"].abs().to_numpy(float)
            if "statistic" in network.edges else np.ones(len(network.edges))
        )
        norm = Normalize(vmin=float(weights.min()), vmax=float(weights.max()) + 1e-12)
        for edge, magnitude in zip(network.edges.to_dict("records"), weights, strict=True):
            first, second = positions[edge["node1"]], positions[edge["node2"]]
            vertices = np.vstack([first, np.zeros(2), np.zeros(2), second])
            path = Path(
                vertices,
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
            )
            statistic = float(edge.get("statistic", 1.0))
            color = "#D94A4A" if statistic >= 0 else "#2A6F9E"
            ax.add_patch(PathPatch(
                path, fill=False, color=color, alpha=0.12 + 0.42 * norm(magnitude),
                linewidth=0.4 + 1.5 * norm(magnitude),
            ))
    for node in ordered_nodes:
        position = positions[node]
        ax.scatter(*position, s=13, color=palette[str(node_networks[node])], zorder=3)
    ax.set_xlim(-1.38, 1.38)
    ax.set_ylim(-1.38, 1.38)
    ax.set_axis_off()
    return ax


def plot_leading_adjacency(network: LeadingNetwork, ax=None):
    ax = _axes(ax)
    nodes = network.nodes["node_id"].tolist()
    adjacency = np.zeros((len(nodes), len(nodes)), dtype=float)
    index = {node: position for position, node in enumerate(nodes)}
    for edge in network.edges.to_dict("records"):
        value = float(edge.get("statistic", 1.0))
        adjacency[index[edge["node1"]], index[edge["node2"]]] = value
        if not network.directed:
            adjacency[index[edge["node2"]], index[edge["node1"]]] = value
    maximum = max(float(np.max(np.abs(adjacency))), 1e-12)
    image = ax.imshow(adjacency, cmap="RdBu_r", vmin=-maximum, vmax=maximum)
    ax.figure.colorbar(image, ax=ax)
    ax.set(xticks=range(len(nodes)), xticklabels=nodes,
           yticks=range(len(nodes)), yticklabels=nodes)
    return ax


def plot_node_participation(network: LeadingNetwork, ax=None):
    ax = _axes(ax)
    frame = compute_node_participation(network).sort_values("degree", ascending=False)
    ax.bar(frame["node_id"].astype(str), frame["degree"], color="#173F67")
    ax.tick_params(axis="x", rotation=90)
    ax.set(ylabel="Leading-edge degree", xlabel="Node")
    return ax


def plot_stability(summary: LensStabilityResult, set_name: str, ax=None):
    ax = _axes(ax)
    frame = summary.edges_for(set_name).sort_values(
        "full_pipeline_stability", ascending=False
    )
    ax.bar(frame["edge_id"], frame["full_pipeline_stability"], color="#173F67")
    ax.tick_params(axis="x", rotation=90)
    ax.set(ylim=(0, 1), ylabel="Full-pipeline stability", xlabel="Edge")
    return ax
