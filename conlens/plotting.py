"""Matplotlib visualizations for designs, enrichment, and leading networks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

from .core import compute_running_sum
from .design import Contrast, DesignMatrix
from .leading import compute_node_participation
from .results import LeadingNetwork, LensResult, LensStabilityResult

DEFAULT_NETWORK_COLORS = (
    "#1F77B4", "#2CA02C", "#FFB000", "#D62728", "#9467BD",
    "#17BECF", "#8C564B", "#7A8796", "#C86B98", "#8C6D31",
)
POSITIVE_COLOR = "#D33F2F"
NEGATIVE_COLOR = "#2768B7"
PLOT_BACKGROUND = "#FAFAF7"
GRID_COLOR = "#E5E5E1"
TEXT_COLOR = "#202427"


def _axes(ax=None):
    return ax if ax is not None else plt.subplots()[1]


def _network_palette(
    labels: Sequence[str], colors: Mapping[str, str] | None = None
) -> dict[str, str]:
    unique = list(dict.fromkeys(labels))
    if colors is not None:
        missing = set(unique) - set(colors)
        if missing:
            raise ValueError(f"network_colors omits annotations: {sorted(missing)!r}")
        return {label: colors[label] for label in unique}
    return {
        label: DEFAULT_NETWORK_COLORS[index % len(DEFAULT_NETWORK_COLORS)]
        for index, label in enumerate(unique)
    }


def _effect_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "conlens_signed_effect",
        ["#254E78", "#BFD0DE", "#F8F3E9", "#E9B7A6", "#A63A32"],
        N=256,
    )


def _resolve_network_layout(
    labels: Sequence[Any],
    node_networks: Mapping[Any, str] | Sequence[str],
    network_order: Sequence[str] | None,
    network_colors: Mapping[str, str] | None,
) -> tuple[list[int], list[str], list[str], dict[str, str], list[tuple[str, int, int, float]]]:
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
    ordered_networks = [networks[index] for index in order]
    palette = _network_palette(order_names, network_colors)
    blocks = []
    for name in order_names:
        positions = [index for index, current in enumerate(ordered_networks) if current == name]
        if positions:
            blocks.append((name, min(positions), max(positions), float(np.mean(positions))))
    return order, ordered_networks, order_names, palette, blocks


def _decorate_matrix(
    ax,
    blocks: Sequence[tuple[str, int, int, float]],
    palette: Mapping[str, str],
    size: int,
    *,
    show_network_labels: bool,
) -> None:
    strip_offset = 3.25
    strip_width = 2.0
    for name, start, end, _ in blocks:
        length = end - start + 1
        ax.add_patch(Rectangle(
            (start - 0.5, -strip_offset), length, strip_width,
            color=palette[name], clip_on=False, linewidth=0,
        ))
        ax.add_patch(Rectangle(
            (-strip_offset, start - 0.5), strip_width, length,
            color=palette[name], clip_on=False, linewidth=0,
        ))
        if end < size - 1:
            ax.axhline(end + 0.5, color=GRID_COLOR, linewidth=0.45)
            ax.axvline(end + 0.5, color=GRID_COLOR, linewidth=0.45)
    ax.set_xlim(-3.65, size - 0.5)
    ax.set_ylim(size - 0.5, -3.65)
    if show_network_labels:
        centers = [center for _, _, _, center in blocks]
        names = [name for name, *_ in blocks]
        ax.set_xticks(centers, names, rotation=45, ha="right")
        ax.set_yticks(centers, names)
        ax.tick_params(length=0, labelsize=8, colors="#5D6265")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#777773")
        spine.set_linewidth(0.6)


def _symmetric_limit(values: np.ndarray) -> float:
    finite = np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite)]
    return max(float(finite.max()), 1e-12) if finite.size else 1.0


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
    network_colors: Mapping[str, str] | None = None,
    triangle: str = "lower",
    significant_mask: np.ndarray | None = None,
    colorbar_label: str = "Connectivity",
    show_colorbar: bool = True,
    show_network_labels: bool = False,
    vmax: float | None = None,
    cmap: Any = None,
    ax=None,
):
    """Plot a network-ordered matrix with compact annotation strips.

    The lower-triangle default avoids displaying an undirected matrix twice. If
    ``significant_mask`` is supplied, masked discoveries are added to the upper
    triangle without changing the underlying values.
    """
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    if triangle not in {"lower", "full"}:
        raise ValueError("triangle must be 'lower' or 'full'")
    labels = list(range(len(values))) if node_labels is None else list(node_labels)
    if len(labels) != len(values):
        raise ValueError("node_labels must match matrix dimensions")
    order, _, _, palette, blocks = _resolve_network_layout(
        labels, node_networks, network_order, network_colors
    )
    ordered_values = values[np.ix_(order, order)]
    if significant_mask is not None:
        mask = np.asarray(significant_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("significant_mask must match matrix dimensions")
        ordered_mask = mask[np.ix_(order, order)]
    else:
        ordered_mask = None
    display = ordered_values.copy()
    if triangle == "lower":
        display = np.full_like(ordered_values, np.nan)
        lower = np.tril_indices(len(values), -1)
        display[lower] = ordered_values[lower]
        if ordered_mask is not None:
            upper = np.triu(ordered_mask, 1)
            display[upper] = ordered_values[upper]
    ax = _axes(ax)
    maximum = _symmetric_limit(ordered_values) if vmax is None else float(vmax)
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("vmax must be a finite number > 0")
    plot_cmap = _effect_cmap() if cmap is None else plt.get_cmap(cmap)
    if hasattr(plot_cmap, "copy"):
        plot_cmap = plot_cmap.copy()
    plot_cmap.set_bad(PLOT_BACKGROUND)
    image = ax.imshow(
        display, cmap=plot_cmap, vmin=-maximum, vmax=maximum,
        interpolation="nearest", aspect="equal",
    )
    _decorate_matrix(
        ax, blocks, palette, len(values), show_network_labels=show_network_labels
    )
    ax.set_facecolor(PLOT_BACKGROUND)
    if show_colorbar:
        ax.figure.colorbar(
            image, ax=ax, shrink=0.72, pad=0.035, label=colorbar_label,
        )
    return ax


def _network_pair_rows(
    result: LensResult, value: str
) -> tuple[str, list[str], list[tuple[str, str, float, float | None]]]:
    if value not in {"ES", "NES"}:
        raise ValueError("value must be 'ES' or 'NES'")
    frame = result.to_frame()
    names = frame.get("set_name", pd.Series(dtype=str)).astype(str).tolist()
    separator = "->" if any("->" in name for name in names) else "--"
    pairs = [name.split(separator) for name in names]
    if not pairs or any(len(pair) != 2 for pair in pairs):
        raise ValueError("all set names must be network pairs such as 'A--B' or 'A->B'")
    labels = list(dict.fromkeys(label for pair in pairs for label in pair))
    rows = []
    for pair, row in zip(pairs, frame.to_dict("records"), strict=True):
        cell = row[value]
        if cell is None or pd.isna(cell):
            continue
        q_value = row.get("q_value")
        rows.append((pair[0], pair[1], float(cell), None if pd.isna(q_value) else float(q_value)))
    return separator, labels, rows


def _draw_enrichment_bubbles(
    ax,
    rows: Sequence[tuple[str, str, float, float | None]],
    centers: Mapping[str, float],
    ranks: Mapping[str, int],
    *,
    significance_alpha: float,
    maximum: float,
    undirected_upper: bool,
    minimum_size: float,
    size_range: float,
    annotate: bool,
) -> None:
    for first, second, cell, q_value in rows:
        if first not in centers or second not in centers:
            raise ValueError(f"network pair {first!r}, {second!r} lacks node annotations")
        if undirected_upper and ranks[first] > ranks[second]:
            first, second = second, first
        x, y = centers[second], centers[first]
        strength = min(abs(cell) / maximum, 1.0)
        color = POSITIVE_COLOR if cell >= 0 else NEGATIVE_COLOR
        significant = q_value is not None and q_value <= significance_alpha
        ax.scatter(
            x, y,
            s=minimum_size + size_range * strength,
            facecolor=color if significant else PLOT_BACKGROUND,
            edgecolor=color,
            linewidth=0.8,
            alpha=0.92,
            zorder=4,
        )
        if annotate:
            ax.text(
                x, y, f"{cell:.2f}", ha="center", va="center", fontsize=7,
                color="white" if significant else color, zorder=5,
            )


def plot_lens_heatmap(
    result: LensResult,
    node_networks: Mapping[Any, str],
    *,
    node_order: Sequence[Any] | None = None,
    network_order: Sequence[str] | None = None,
    network_colors: Mapping[str, str] | None = None,
    value: str = "NES",
    significance_alpha: float = 0.05,
    edge_vmax: float | None = None,
    enrichment_vmax: float | None = None,
    show_colorbar: bool = True,
    show_network_labels: bool = False,
    ax=None,
):
    """Fuse the edge ranking and network-pair enrichment in one matrix.

    Edge statistics occupy the lower triangle. Network-pair results occupy the
    upper triangle as signed bubbles; filled bubbles pass the requested q-value
    threshold and hollow bubbles do not.
    """
    if not 0 < significance_alpha < 1:
        raise ValueError("significance_alpha must be in (0, 1)")
    separator, _, rows = _network_pair_rows(result, value)
    if separator != "--":
        raise ValueError("plot_lens_heatmap currently requires undirected 'A--B' edge sets")
    if bool(result.metadata.get("directed", False)):
        raise ValueError("plot_lens_heatmap currently requires undirected edge statistics")
    ranked = result.ranked_edges
    required = {"node1", "node2", "statistic"}
    if not required.issubset(ranked):
        raise ValueError(f"ranked_edges must contain {sorted(required)!r}")
    if node_order is None:
        stored = result.metadata.get("node_order")
        node_order = list(stored) if stored else list(dict.fromkeys([
            *ranked["node1"].tolist(), *ranked["node2"].tolist(),
        ]))
    labels = list(node_order)
    if len(labels) != len(set(labels)):
        raise ValueError("node_order must not contain duplicates")
    missing_nodes = (
        set(ranked["node1"]) | set(ranked["node2"])
    ) - set(labels)
    if missing_nodes:
        raise ValueError(f"node_order omits ranked edge nodes: {sorted(missing_nodes, key=str)!r}")
    index = {node: position for position, node in enumerate(labels)}
    matrix = np.full((len(labels), len(labels)), np.nan, dtype=float)
    for edge in ranked[["node1", "node2", "statistic"]].itertuples(index=False):
        first, second = index[edge.node1], index[edge.node2]
        matrix[first, second] = float(edge.statistic)
        matrix[second, first] = float(edge.statistic)
    order, _, order_names, palette, blocks = _resolve_network_layout(
        labels, node_networks, network_order, network_colors
    )
    ordered = matrix[np.ix_(order, order)]
    display = np.full_like(ordered, np.nan)
    lower = np.tril_indices(len(labels), -1)
    display[lower] = ordered[lower]
    edge_limit = _symmetric_limit(ordered) if edge_vmax is None else float(edge_vmax)
    if not np.isfinite(edge_limit) or edge_limit <= 0:
        raise ValueError("edge_vmax must be a finite number > 0")
    enrichment_limit = (
        max((abs(row[2]) for row in rows), default=1.0)
        if enrichment_vmax is None else float(enrichment_vmax)
    )
    if not np.isfinite(enrichment_limit) or enrichment_limit <= 0:
        raise ValueError("enrichment_vmax must be a finite number > 0")
    ax = _axes(ax)
    cmap = _effect_cmap()
    cmap.set_bad(PLOT_BACKGROUND)
    image = ax.imshow(
        display, cmap=cmap, vmin=-edge_limit, vmax=edge_limit,
        interpolation="nearest", aspect="equal",
    )
    centers = {name: center for name, _, _, center in blocks}
    ranks = {name: index for index, name in enumerate(order_names)}
    _draw_enrichment_bubbles(
        ax, rows, centers, ranks,
        significance_alpha=significance_alpha,
        maximum=enrichment_limit,
        undirected_upper=True,
        minimum_size=34,
        size_range=335,
        annotate=False,
    )
    _decorate_matrix(
        ax, blocks, palette, len(labels), show_network_labels=show_network_labels
    )
    ax.set_facecolor(PLOT_BACKGROUND)
    if show_colorbar:
        statistic_name = result.metadata.get("statistic_name", "Edge statistic")
        ax.figure.colorbar(
            image, ax=ax, shrink=0.72, pad=0.035, label=str(statistic_name),
        )
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
    color = POSITIVE_COLOR if item.ES > 0 else NEGATIVE_COLOR
    ax.plot(ranks, profile, color=color, linewidth=1.45)
    ax.axhline(0, color="#9B9B96", linewidth=0.55)
    if item.peak_rank is not None:
        interval = (0, item.peak_rank) if item.ES > 0 else (item.peak_rank, len(profile) - 1)
        ax.axvline(item.peak_rank, color=color, linewidth=0.7, linestyle=":")
        ax.axvspan(*interval, color=color, alpha=0.065)
    ax.margins(x=0, y=0.08)
    hits = np.flatnonzero(result.ranked_edges["edge_id"].isin(item.edge_set_ids)) + 1
    lower, upper = ax.get_ylim()
    rug_height = (upper - lower) * 0.045
    ax.vlines(hits, lower, lower + rug_height, color="#5D6265", linewidth=0.45, alpha=0.5)
    q_text = "" if item.q_value is None else f", q={item.q_value:.3f}"
    nes_text = "" if item.NES is None else f"  NES={item.NES:.2f}{q_text}"
    ax.set_title(
        f"{set_name}{nes_text}", fontsize=9, color=color, weight="bold", pad=5,
    )
    ax.set_xlim(0, max(len(profile) - 1, 1))
    ax.set_xticks([])
    ax.set_ylabel("Running sum", fontsize=8, color="#5D6265")
    ax.tick_params(axis="y", labelsize=7, colors="#666666", width=0.5)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("#C8C8C3")
    ax.spines["left"].set_linewidth(0.55)
    ax.set_facecolor(PLOT_BACKGROUND)
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
    annotate: bool = False,
    network_order: Sequence[str] | None = None,
    network_colors: Mapping[str, str] | None = None,
    vmax: float | None = None,
    ax=None,
):
    """Display network-pair enrichment as a signed significance bubble matrix."""
    if not 0 < significance_alpha < 1:
        raise ValueError("significance_alpha must be in (0, 1)")
    separator, labels, rows = _network_pair_rows(result, value)
    order_names = list(labels) if network_order is None else list(network_order)
    unknown = set(labels) - set(order_names)
    if unknown:
        raise ValueError(f"network_order omits set labels: {sorted(unknown)!r}")
    labels = [label for label in order_names if label in set(labels)]
    palette = _network_palette(labels, network_colors)
    centers = {name: float(index) for index, name in enumerate(labels)}
    ranks = {name: index for index, name in enumerate(labels)}
    maximum = max((abs(row[2]) for row in rows), default=1.0) if vmax is None else float(vmax)
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("vmax must be a finite number > 0")
    ax = _axes(ax)
    ax.set_facecolor(PLOT_BACKGROUND)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_aspect("equal")
    for boundary in np.arange(-0.5, len(labels), 1):
        ax.axhline(boundary, color=GRID_COLOR, linewidth=0.55, zorder=0)
        ax.axvline(boundary, color=GRID_COLOR, linewidth=0.55, zorder=0)
    _draw_enrichment_bubbles(
        ax, rows, centers, ranks,
        significance_alpha=significance_alpha,
        maximum=maximum,
        undirected_upper=separator == "--",
        minimum_size=42,
        size_range=255,
        annotate=annotate,
    )
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.tick_params(length=0, labelsize=8)
    for tick, label in zip(ax.get_xticklabels(), labels, strict=True):
        tick.set_color(palette[label])
    for tick, label in zip(ax.get_yticklabels(), labels, strict=True):
        tick.set_color(palette[label])
    ax.set_title(
        f"Network-pair {value}  ·  filled: q ≤ {significance_alpha:g}",
        fontsize=9, color=TEXT_COLOR, pad=8,
    )
    for spine in ax.spines.values():
        spine.set_color("#777773")
        spine.set_linewidth(0.6)
    return ax


def plot_circos(
    network: LeadingNetwork,
    node_networks: Mapping[Any, str],
    *,
    network_order: Sequence[str] | None = None,
    network_colors: Mapping[str, str] | None = None,
    show_labels: bool = False,
    show_nodes: bool = False,
    empty_message: str = "No leading edges",
    ax=None,
):
    """Plot leading edges as fine chords inside a double network ring."""
    edge_nodes = network.nodes["node_id"].tolist()
    missing = set(edge_nodes) - set(node_networks)
    if missing:
        raise ValueError(f"missing network annotations: {sorted(missing, key=str)!r}")
    nodes = list(node_networks)
    if not nodes:
        raise ValueError("node_networks must not be empty")
    groups = [str(node_networks[node]) for node in nodes]
    group_order = (
        list(dict.fromkeys(groups)) if network_order is None else list(network_order)
    )
    unknown = set(groups) - set(group_order)
    if unknown:
        raise ValueError(f"network_order omits annotations: {sorted(unknown)!r}")
    ordered_nodes = sorted(
        nodes,
        key=lambda node: (group_order.index(str(node_networks[node])), str(node)),
    )
    ordered_groups = [str(node_networks[node]) for node in ordered_nodes]
    palette = _network_palette(group_order, network_colors)
    gap = np.deg2rad(4.2)
    step = (2 * np.pi - gap * len(group_order)) / len(nodes)
    angle = np.deg2rad(92)
    positions: dict[Any, np.ndarray] = {}
    segments = []
    for group in group_order:
        group_nodes = [
            node for node, current in zip(ordered_nodes, ordered_groups, strict=True)
            if current == group
        ]
        if not group_nodes:
            continue
        start = angle
        for node in group_nodes:
            current = angle - step / 2
            positions[node] = np.asarray([np.cos(current), np.sin(current)])
            angle -= step
        segments.append((group, start, angle))
        angle -= gap
    ax = _axes(ax)
    ax.set_aspect("equal")
    ax.set_facecolor("none")
    theta = np.linspace(0, 2 * np.pi, 600)
    ax.plot(np.cos(theta), np.sin(theta), color="#D8D3CC", linewidth=0.55, zorder=1)
    for group, start, end in segments:
        values = np.linspace(start, end, 100)
        ax.plot(
            1.03 * np.cos(values), 1.03 * np.sin(values),
            color=palette[group], linewidth=4.2, solid_capstyle="butt", zorder=3,
        )
        ax.plot(
            1.095 * np.cos(values), 1.095 * np.sin(values),
            color=palette[group], linewidth=1.8, solid_capstyle="butt", zorder=3,
        )
        if show_labels:
            middle = (start + end) / 2
            ax.text(
                1.22 * np.cos(middle), 1.22 * np.sin(middle), group,
                ha="center", va="center", fontsize=8, color=palette[group],
            )
    if len(network.edges):
        weights = (
            network.edges["statistic"].abs().to_numpy(float)
            if "statistic" in network.edges else np.ones(len(network.edges))
        )
        maximum = max(float(weights.max()), 1e-12)
        records = network.edges.assign(_magnitude=weights).sort_values("_magnitude")
        density = 0.65 if len(records) > 300 else 1.0
        for edge in records.to_dict("records"):
            first = 0.93 * positions[edge["node1"]]
            second = 0.93 * positions[edge["node2"]]
            path = Path(
                np.vstack([first, np.zeros(2), second]),
                [Path.MOVETO, Path.CURVE3, Path.CURVE3],
            )
            statistic = float(edge.get("statistic", 1.0))
            strength = min(float(edge["_magnitude"]) / maximum, 1.0)
            color = POSITIVE_COLOR if statistic >= 0 else NEGATIVE_COLOR
            ax.add_patch(PathPatch(
                path, fill=False, color=color,
                alpha=density * (0.10 + 0.30 * strength),
                linewidth=0.10 + 0.65 * strength,
                zorder=2,
            ))
    else:
        ax.text(
            0, 0, empty_message, ha="center", va="center",
            fontsize=8, color="#6F7375",
        )
    if show_nodes:
        for node in ordered_nodes:
            position = 0.97 * positions[node]
            ax.scatter(
                *position, s=5, color=palette[str(node_networks[node])],
                linewidth=0, zorder=4,
            )
    limit = 1.30 if show_labels else 1.17
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
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
