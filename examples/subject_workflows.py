"""Reproducible unified GLM and full-pipeline bootstrap stability workflow."""

import numpy as np

from conlens import Contrast, LensAnalysis, make_design

rng = np.random.default_rng(12)
n_subjects, n_nodes = 24, 6
groups = np.repeat([0, 1], n_subjects // 2)
age = np.linspace(20, 60, n_subjects)
upper = np.triu_indices(n_nodes, 1)
edge_data = rng.normal(size=(n_subjects, len(upper[0])))
edge_data[groups == 1, :5] += 5.0

connectomes = np.zeros((n_subjects, n_nodes, n_nodes))
connectomes[:, upper[0], upper[1]] = edge_data
connectomes[:, upper[1], upper[0]] = edge_data
edge_ids = [f"{node1}--{node2}" for node1, node2 in zip(*upper, strict=True)]
edge_sets = {
    "front": set(edge_ids[:5]),
    "back": set(edge_ids[5:]),
}
analysis = LensAnalysis.from_subject_connectomes(connectomes, edge_sets)
design = make_design(
    indicators={
        "control": groups == 0,
        "case": groups == 1,
    },
    continuous={
        "age": age,
    },
    add_intercept=False,
)
contrasts = {
    "case_vs_control": Contrast(
        {"case": 1, "control": -1},
        effect_size="hedges_g",
        positive_direction="case > control",
    ),
    "age": Contrast(
        {"age": 1},
        effect_size="partial_r",
        positive_direction="increases with age",
    ),
}

glm = analysis.glm(
    design,
    contrasts,
    n_permutations=199,
    random_state=4,
    min_size=1,
)
observed = glm["case_vs_control"]


def refit_case_contrast(sample, indices, fit_seed):
    """Repeat the complete joint contrast family on one bootstrap sample."""
    return sample.glm(
        design.take(indices),
        contrasts,
        n_permutations=199,
        random_state=fit_seed,
        min_size=1,
    )["case_vs_control"]


stability = analysis.bootstrap_stability(
    observed,
    refit_case_contrast,
    strata=groups,
    n_bootstraps=5,
    random_state=4,
)
assert glm.metadata["adjustment_method"] == "BH"
assert observed.metadata["effect_size"] == "hedges_g"
assert glm["age"].metadata["effect_size"] == "partial_r"
assert stability.metadata["resampling_method"] == "stratified_subject"
assert stability.metadata["n_bootstraps"] == 5
