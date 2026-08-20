"""Reproducible subject models and both bootstrap stability workflows."""

import numpy as np

from conlens import LensAnalysis, summarize_stability
from conlens.stats import two_group_ttest

rng = np.random.default_rng(12)
n_subjects, n_nodes = 24, 6
groups = np.repeat([0, 1], n_subjects // 2)
upper = np.triu_indices(n_nodes, 1)
edge_data = rng.normal(size=(n_subjects, len(upper[0])))
edge_data[groups == 1, :5] += 2.0

connectomes = np.zeros((n_subjects, n_nodes, n_nodes))
connectomes[:, upper[0], upper[1]] = edge_data
connectomes[:, upper[1], upper[0]] = edge_data
edge_ids = [f"{node1}--{node2}" for node1, node2 in zip(*upper, strict=True)]
edge_sets = {
    "front": set(edge_ids[:5]),
    "back": set(edge_ids[5:]),
}
analysis = LensAnalysis.from_subject_connectomes(connectomes, edge_sets)

two_group = analysis.two_group(
    groups,
    null_method="label_permutation",
    n_permutations=99,
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


descriptive_results = analysis.bootstrap(
    bootstrap_statistic,
    strata=groups,
    n_bootstraps=5,
    random_state=4,
    min_size=1,
)
descriptive_summary = summarize_stability(descriptive_results)


def refit_two_group(sample, indices, fit_seed):
    """Repeat the edge model, subject-level null, LENS, and BH adjustment."""
    return sample.two_group(
        groups[indices],
        null_method="label_permutation",
        n_permutations=99,
        random_state=fit_seed,
        min_size=1,
    )


stability = analysis.bootstrap_stability(
    two_group,
    refit_two_group,
    strata=groups,
    n_bootstraps=5,
    random_state=4,
)
assert two_group.metadata["null_method"] == "label_permutation"
assert phenotype.metadata["permutation_scheme"] == "shared_subject_phenotype_permutation"
assert glm.metadata["null_method"] == "freedman_lane"
assert len(descriptive_results) == 5
assert descriptive_summary["n_replicates"] == 5
assert stability.metadata["resampling_method"] == "stratified_subject"
assert not stability.set_summary.empty
assert {"set_stability", "full_pipeline_core_size"}.issubset(stability.set_summary)
