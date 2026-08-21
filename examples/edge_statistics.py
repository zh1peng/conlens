"""LENS analysis when signed edge statistics already exist."""

import pandas as pd

from conlens import lens_edge_permute, lens_enrich, lens_stat, make_edge_statistics

edges = pd.DataFrame({
    "node1": [0, 0, 0, 1, 1, 2],
    "node2": [1, 2, 3, 2, 3, 3],
    "statistic": [0.61, 0.43, 0.12, -0.08, -0.37, -0.55],
})
edge_sets = {
    "DMN--DMN": {"0--1", "0--2", "0--3"},
    "DMN--VIS": {"1--2", "1--3", "2--3"},
}

true_edges = make_edge_statistics(
    edges,
    positive_direction="connectivity increases with age",
    statistic_name="partial correlation",
)
observed = lens_stat(true_edges, edge_sets, store_running_sum=True)
null_edges = lens_edge_permute(true_edges, n_permutations=199, random_state=42)
null_stats = (lens_stat(item, edge_sets) for item in null_edges)
result = lens_enrich(
    observed,
    null_stats,
    min_size=1,
    family_name="age-network-pairs",
)

assert result.null_scores is not None
assert result.null_scores.shape == (199, 2)
print(result.to_frame()[["set_name", "ES", "NES", "p_value", "q_value"]])
