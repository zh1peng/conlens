"""Unified subject-level GLM, FL permutation, and LENS enrichment."""

import numpy as np

from conlens import (
    Contrast,
    lens_enrich,
    lens_fl_permute,
    lens_glm,
    lens_stat,
    make_design,
)

rng = np.random.default_rng(7)
n_subjects, n_nodes = 36, 6
diagnosis = np.repeat(["control", "g1", "g2"], 12)
age = rng.normal(45, 9, n_subjects)
sex = rng.integers(0, 2, n_subjects)
raw = rng.normal(size=(n_subjects, n_nodes, n_nodes))
connectomes = (raw + raw.transpose(0, 2, 1)) / 2
for matrix in connectomes:
    np.fill_diagonal(matrix, 0)
connectomes[diagnosis == "g1", 0, 1] += 0.9
connectomes[:, 1, 0] = connectomes[:, 0, 1]

design = make_design(
    groups={
        "control": diagnosis == "control",
        "g1": diagnosis == "g1",
        "g2": diagnosis == "g2",
    },
    indicators={"sex": sex},
    continuous={"age": age},
)
contrasts = {
    "g1_vs_control": Contrast(
        {"g1": 1, "control": -1},
        effect_size="hedges_g",
        positive_direction="g1 > control",
    ),
    "g2_vs_control": Contrast(
        {"g2": 1, "control": -1},
        effect_size="hedges_g",
        positive_direction="g2 > control",
    ),
    "age": Contrast(
        {"age": 1},
        effect_size="partial_r",
        positive_direction="connectivity increases with age",
    ),
}
edge_sets = {
    "A--A": {"0--1", "0--2", "1--2"},
    "A--B": {"0--3", "0--4", "1--3", "1--4", "2--3", "2--4"},
    "B--B": {"3--4", "3--5", "4--5"},
}

true_edges = lens_glm(connectomes, design=design, contrasts=contrasts)
observed = lens_stat(true_edges, edge_sets, store_running_sum=True)
null_edges = lens_fl_permute(
    connectomes,
    design=design,
    contrasts=contrasts,
    n_permutations=99,
    random_state=42,
)
null_stats = (lens_stat(item, edge_sets) for item in null_edges)
result = lens_enrich(
    observed,
    null_stats,
    min_size=2,
    family_name="primary-model",
)

print(result.to_frame()[
    ["contrast_name", "set_name", "ES", "NES", "p_value", "q_value"]
])
