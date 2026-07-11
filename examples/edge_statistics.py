"""Reproducible edge-table, edge-set, inference, network, plot, and I/O tutorial."""

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import pandas as pd

from conlens import (
    LensResult,
    build_leading_network,
    lens_enrich,
    make_network_pair_sets,
    validate_edge_table,
)
from conlens.plotting import plot_enrichment

edges = pd.DataFrame(
    {
        "node1": ["A", "A", "A", "B", "B", "C"],
        "node2": ["B", "C", "D", "C", "D", "D"],
        "statistic": [3.0, 2.0, 1.0, -0.5, -1.5, -2.5],
    }
)
edges = validate_edge_table(edges, node_order=["A", "B", "C", "D"])
networks = {"A": "X", "B": "X", "C": "Y", "D": "Y"}
edge_sets = make_network_pair_sets(edges, networks)

result = lens_enrich(
    edges,
    edge_sets,
    min_size=1,
    null_method="edge_permutation",
    n_permutations=50,
    random_state=7,
    positive_direction="case > control",
)
network = build_leading_network(result, "X--Y")
axes = plot_enrichment(result, "X--Y", edge_sets["X--Y"])
assert len(axes) == 3
assert network.directed is False

with TemporaryDirectory() as directory:
    result_path = result.save(Path(directory) / "result.json")
    network.save(Path(directory) / "leading.graphml")
    restored = LensResult.load(result_path)
    assert restored.get("X--Y").ES == result.get("X--Y").ES

plt.close("all")
