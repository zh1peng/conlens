"""Small full-pipeline bootstrap example."""

import numpy as np

from conlens import Contrast, lens_bootstrap, make_design

rng = np.random.default_rng(11)
n_subjects, n_nodes = 30, 5
group = np.repeat([0, 1], n_subjects // 2)
raw = rng.normal(size=(n_subjects, n_nodes, n_nodes))
connectomes = (raw + raw.transpose(0, 2, 1)) / 2
for matrix in connectomes:
    np.fill_diagonal(matrix, 0)
design = make_design(groups={"control": group == 0, "case": group == 1})
contrasts = {
    "case_vs_control": Contrast(
        {"case": 1, "control": -1}, "hedges_g", "case > control"
    )
}
edge_sets = {
    "A--A": {"0--1", "0--2", "1--2"},
    "A--B": {"0--3", "0--4", "1--3", "1--4"},
}

stability = lens_bootstrap(
    connectomes,
    edge_sets,
    design=design,
    contrasts=contrasts,
    n_bootstraps=3,
    n_permutations=9,
    strata=group,
    random_state=42,
    min_size=2,
    min_same_direction=1,
)
print(stability["case_vs_control"].set_summary)
