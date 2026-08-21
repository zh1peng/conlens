"""Generate the figures shown in the VitePress visualization guide."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from conlens import (
    build_leading_network,
    lens_edge_permute,
    lens_enrich,
    lens_stat,
    make_edge_statistics,
    make_network_pair_sets,
    matrix_to_edges,
    plot_circos,
    plot_connectome_heatmap,
    plot_enrichment,
    plot_enrichment_heatmap,
    plot_lens_heatmap,
)

output = Path(__file__).parents[1] / "website" / "public" / "figures"
output.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(12)
network_order = ["VIS", "SMN", "DAN", "VAN", "FPN", "DMN"]
labels = [f"N{index + 1:02d}" for index in range(48)]
node_networks = {
    label: network
    for label, network in zip(labels, np.repeat(network_order, 8), strict=True)
}
raw = rng.normal(0, 0.18, size=(48, 48))
matrix = (raw + raw.T) / 2
np.fill_diagonal(matrix, 0)
matrix[32:40, 40:48] += 0.36
matrix[40:48, 32:40] = matrix[32:40, 40:48].T
matrix[24:32, 40:48] -= 0.28
matrix[40:48, 24:32] = matrix[24:32, 40:48].T
matrix[0:8, 8:16] += 0.18
matrix[8:16, 0:8] = matrix[0:8, 8:16].T
edges = matrix_to_edges(matrix, labels)
edge_sets = make_network_pair_sets(edges, node_networks)
true_edges = make_edge_statistics(
    edges,
    positive_direction="stronger association",
    statistic_name="signed edge effect",
)
observed = lens_stat(true_edges, edge_sets, store_running_sum=True)
null_edges = lens_edge_permute(true_edges, n_permutations=499, random_state=9)
result = lens_enrich(
    observed,
    (lens_stat(item, edge_sets) for item in null_edges),
    min_size=5,
    family_name="visual-demo",
)

figure, ax = plt.subplots(figsize=(5.7, 5.2))
plot_connectome_heatmap(
    matrix, node_networks, node_labels=labels, network_order=network_order,
    colorbar_label="Signed connectivity", ax=ax,
)
figure.savefig(
    output / "connectome-heatmap.png", dpi=180, bbox_inches="tight", facecolor="#FAFAF7",
)
plt.close()
figure, ax = plt.subplots(figsize=(5.4, 5.1))
plot_enrichment_heatmap(result, network_order=network_order, ax=ax)
figure.savefig(
    output / "enrichment-heatmap.png", dpi=180, bbox_inches="tight", facecolor="#FAFAF7",
)
plt.close()
figure, ax = plt.subplots(figsize=(5.7, 5.2))
plot_lens_heatmap(result, node_networks, network_order=network_order, ax=ax)
figure.savefig(
    output / "lens-heatmap.png", dpi=180, bbox_inches="tight", facecolor="#FAFAF7",
)
plt.close()
plot_enrichment(result, "DMN--FPN")
plt.gcf().savefig(
    output / "enrichment-profile.png", dpi=180, bbox_inches="tight", facecolor="#FAFAF7",
)
plt.close()
leading = build_leading_network(result, "DMN--FPN")
figure, ax = plt.subplots(figsize=(5.2, 5.2))
plot_circos(leading, node_networks, network_order=network_order, ax=ax)
figure.savefig(output / "leading-circos.png", dpi=180, bbox_inches="tight", transparent=True)
plt.close()
