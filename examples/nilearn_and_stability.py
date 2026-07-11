"""Reproducible Nilearn-adapter and statistic-replicate stability tutorial."""

import numpy as np
import pandas as pd

from conlens import bootstrap_lens, consensus_network, summarize_stability
from conlens.interfaces.nilearn import from_nilearn_connectivity

rng = np.random.default_rng(21)
connectomes = rng.normal(size=(8, 3, 3))
connectomes = (connectomes + connectomes.transpose(0, 2, 1)) / 2
for matrix in connectomes:
    np.fill_diagonal(matrix, 0)

labels = ["A", "B", "C"]
metadata = pd.DataFrame({"node_id": labels, "network": ["X", "X", "Y"]})
subject_edges = from_nilearn_connectivity(
    connectomes,
    labels,
    coordinates=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
    node_metadata=metadata,
)
template = subject_edges[subject_edges["subject"] == 0].copy()
template["statistic"] = [2.0, 0.5, -1.0]
edge_sets = {"example": {"0--1", "0--2"}}
replicates = np.vstack(
    [template["statistic"].to_numpy() + rng.normal(0, 0.1, len(template)) for _ in range(5)]
)
results = bootstrap_lens(
    template,
    edge_sets,
    statistic_replicates=replicates,
    min_size=1,
)
summary = summarize_stability(results)
consensus = consensus_network(results, "example", threshold=0.6)
assert summary["n_replicates"] == 5
assert "inclusion_frequency" in consensus
