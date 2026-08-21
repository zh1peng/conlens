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
)

output = Path(__file__).parents[1] / "website" / "public" / "figures"
output.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(12)
labels = [f"N{index + 1}" for index in range(12)]
node_networks = {
    label: network
    for label, network in zip(labels, np.repeat(["DMN", "FPN", "VIS"], 4), strict=True)
}
raw = rng.normal(0, 0.22, size=(12, 12))
matrix = (raw + raw.T) / 2
np.fill_diagonal(matrix, 0)
matrix[:4, 4:8] += 0.65
matrix[4:8, :4] = matrix[:4, 4:8].T
edges = matrix_to_edges(matrix, labels)
edge_sets = make_network_pair_sets(edges, node_networks)
true_edges = make_edge_statistics(
    edges,
    positive_direction="stronger association",
    statistic_name="signed edge effect",
)
observed = lens_stat(true_edges, edge_sets, store_running_sum=True)
null_edges = lens_edge_permute(true_edges, n_permutations=199, random_state=9)
result = lens_enrich(
    observed,
    (lens_stat(item, edge_sets) for item in null_edges),
    min_size=5,
    family_name="visual-demo",
)

plot_connectome_heatmap(matrix, node_networks, node_labels=labels)
plt.gcf().savefig(output / "connectome-heatmap.png", dpi=170, bbox_inches="tight")
plt.close()
plot_enrichment_heatmap(result)
plt.gcf().savefig(output / "enrichment-heatmap.png", dpi=170, bbox_inches="tight")
plt.close()
plot_enrichment(result, "DMN--FPN")
plt.gcf().savefig(output / "enrichment-profile.png", dpi=170, bbox_inches="tight")
plt.close()
leading = build_leading_network(result, "DMN--FPN")
plot_circos(leading, node_networks)
plt.gcf().savefig(output / "leading-circos.png", dpi=170, bbox_inches="tight", transparent=True)
plt.close()
