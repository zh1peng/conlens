"""Reproducible two-group, phenotype, GLM, and subject-bootstrap tutorial."""

import numpy as np

from conlens import LensAnalysis
from conlens.stats import two_group_ttest

rng = np.random.default_rng(12)
n_subjects, n_nodes = 24, 4
groups = np.repeat([0, 1], n_subjects // 2)
edge_data = rng.normal(size=(n_subjects, 6))
edge_data[groups == 1, :2] += 0.8

connectomes = np.zeros((n_subjects, n_nodes, n_nodes))
upper = np.triu_indices(n_nodes, 1)
connectomes[:, upper[0], upper[1]] = edge_data
connectomes[:, upper[1], upper[0]] = edge_data
edge_sets = {
    "front": {"0--1", "0--2", "0--3"},
    "back": {"1--2", "1--3", "2--3"},
}
analysis = LensAnalysis.from_subject_connectomes(connectomes, edge_sets)

two_group = analysis.two_group(
    groups,
    null_method="label_permutation",
    n_permutations=30,
    random_state=4,
    min_size=1,
)
phenotype = analysis.phenotype(
    np.linspace(-1, 1, n_subjects),
    null_method="label_permutation",
    n_permutations=30,
    random_state=4,
    min_size=1,
)
nuisance = np.column_stack([np.ones(n_subjects), rng.normal(size=n_subjects)])
glm = analysis.glm(
    groups,
    nuisance,
    null_method="freedman_lane",
    n_permutations=30,
    random_state=4,
    min_size=1,
)


def bootstrap_statistic(sample, indices):
    return two_group_ttest(sample, groups[indices])


bootstrap_results = analysis.bootstrap(
    bootstrap_statistic,
    strata=groups,
    n_bootstraps=5,
    random_state=4,
    min_size=1,
)
assert two_group.metadata["null_method"] == "label_permutation"
assert phenotype.metadata["permutation_scheme"] == "shared_subject_phenotype_permutation"
assert glm.metadata["null_method"] == "freedman_lane"
assert len(bootstrap_results) == 5
