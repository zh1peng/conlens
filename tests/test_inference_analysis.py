import numpy as np
import pytest

from conlens import LensAnalysis, adjust_pvalues, lens_enrich
from conlens.inference import (
    apply_null_inference,
    edge_permutation_null,
    empirical_pvalue,
    freedman_lane_null,
    label_permutation_null,
    normalize_enrichment_scores,
    provided_null,
)
from conlens.stats import edge_correlation, glm_statistic, two_group_ttest


def test_empirical_pvalues_and_nes():
    null = [-0.8, -0.2, 0.1, 0.4, 0.7]
    positive = empirical_pvalue(0.5, null)
    assert positive["p_value"] == 2 / 4
    assert positive["minimum_resolvable_p"] == 1 / 4
    assert positive["n_more_extreme"] == 1
    negative = empirical_pvalue(-0.5, null)
    assert negative["p_value"] == 2 / 3
    assert negative["minimum_resolvable_p"] == 1 / 3
    assert empirical_pvalue(0, null)["p_value"] == 1
    nes, status = normalize_enrichment_scores(0.5, null)
    assert nes == pytest.approx(0.5 / 0.4)
    assert status == "defined"
    nes, _ = normalize_enrichment_scores(-0.5, null)
    assert nes == pytest.approx(-1)
    assert normalize_enrichment_scores(1, [-1, -2]) == (None, "undefined")
    assert normalize_enrichment_scores(0, null) == (0.0, "defined")


def test_bh_known_values_and_validation():
    np.testing.assert_allclose(adjust_pvalues([0.01, 0.04, 0.03, 0.002]), [0.02, 0.04, 0.04, 0.008])
    assert adjust_pvalues([]).size == 0
    with pytest.raises(ValueError):
        adjust_pvalues([1.2])
    with pytest.raises(ValueError):
        adjust_pvalues([0.1], method="bonferroni")


def test_edge_null_reproducible_across_jobs(example_edges, example_sets):
    from conlens import validate_edge_table

    edges = validate_edge_table(example_edges)
    one = edge_permutation_null(edges, example_sets, n_permutations=20, random_state=8, n_jobs=1)
    two = edge_permutation_null(edges, example_sets, n_permutations=20, random_state=8, n_jobs=2)
    for name in example_sets:
        np.testing.assert_array_equal(one[name], two[name])


def test_lens_inference_and_provided_null(example_edges, example_sets):
    result = lens_enrich(
        example_edges,
        example_sets,
        min_size=1,
        null_method="edge_permutation",
        n_permutations=30,
        random_state=2,
    )
    assert result.metadata["null_scope"] == "competitive_edge_label"
    assert result.metadata["n_sets_tested"] == 2
    assert all(item.p_value is not None and item.q_value is not None for item in result.sets)
    supplied = {name: [-0.5, 0.2, 0.8] for name in example_sets}
    result2 = lens_enrich(
        example_edges,
        example_sets,
        min_size=1,
        null_method="provided_null",
        provided_null=supplied,
        provided_null_edge_ids=["0--1", "0--2", "0--3", "1--2", "1--3", "2--3"],
        provided_null_edge_sets=example_sets,
        positive_direction="case > control",
        provided_null_direction="case > control",
    )
    assert result2.metadata["n_permutations"] == 3
    with pytest.raises(ValueError, match="required"):
        lens_enrich(example_edges, example_sets, min_size=1, null_method="provided_null")
    with pytest.raises(ValueError, match="accepts"):
        lens_enrich(example_edges, example_sets, min_size=1, null_method="label_permutation")


def _subject_data():
    rng = np.random.default_rng(10)
    data = rng.normal(size=(16, 6))
    labels = np.repeat([0, 1], 8)
    data[labels == 1, :2] += 1.2
    ids = np.array(["0--1", "0--2", "0--3", "1--2", "1--3", "2--3"])
    sets = {"first": set(ids[:3]), "last": set(ids[3:])}
    return data, labels, ids, sets


def test_subject_statistics():
    data, labels, _, _ = _subject_data()
    statistic = two_group_ttest(data, labels)
    assert statistic.shape == (6,)
    design = np.column_stack([labels, np.ones(len(labels))])
    glm = glm_statistic(data, design, np.array([1, 0]))
    assert glm.shape == (6,)
    correlations = edge_correlation(data, np.arange(len(data)))
    assert correlations.shape == (6,)
    with pytest.raises(ValueError):
        two_group_ttest(data, np.zeros(len(data)))
    with pytest.raises(ValueError):
        glm_statistic(data, np.ones((len(data), 2)), np.ones(2))
    with pytest.raises(ValueError):
        edge_correlation(data, np.ones(len(data)))


def test_label_and_freedman_lane_reproducible():
    data, labels, ids, sets = _subject_data()
    blocks = np.repeat(np.arange(4), 4)
    first = label_permutation_null(
        data, labels, ids, sets, n_permutations=12, random_state=3, exchangeability_blocks=blocks
    )
    second = label_permutation_null(
        data, labels, ids, sets, n_permutations=12, random_state=3, exchangeability_blocks=blocks
    )
    np.testing.assert_array_equal(first["first"], second["first"])
    tested = labels[:, None]
    nuisance = np.ones((len(labels), 1))
    fl1 = freedman_lane_null(data, tested, nuisance, ids, sets, n_permutations=10, random_state=4)
    fl2 = freedman_lane_null(data, tested, nuisance, ids, sets, n_permutations=10, random_state=4)
    np.testing.assert_array_equal(fl1["last"], fl2["last"])
    with pytest.raises(ValueError, match="intercept"):
        freedman_lane_null(
            data, tested, np.arange(len(labels))[:, None], ids, sets, n_permutations=2
        )
    with pytest.raises(ValueError, match="n_permutations"):
        label_permutation_null(data, labels, ids, sets, n_permutations=0)
    with pytest.raises(ValueError, match="n_permutations"):
        freedman_lane_null(data, labels, np.ones((len(labels), 1)), ids, sets, n_permutations=0)


def test_high_level_subject_two_group_and_glm():
    data, labels, _, sets = _subject_data()
    matrices = np.zeros((len(data), 4, 4))
    tri = np.triu_indices(4, 1)
    matrices[:, tri[0], tri[1]] = data
    matrices[:, tri[1], tri[0]] = data
    analysis = LensAnalysis.from_subject_connectomes(matrices, sets)
    descriptive = analysis.two_group(labels, min_size=1)
    assert descriptive.metadata["inference_status"] == "not_requested"
    assert descriptive.metadata["positive_direction"] == "1 > 0"
    inferred = analysis.two_group(
        labels, null_method="label_permutation", n_permutations=10, random_state=5, min_size=1
    )
    assert inferred.metadata["null_method"] == "label_permutation"
    nuisance = np.ones((len(labels), 1))
    glm = analysis.glm(
        labels,
        nuisance,
        null_method="freedman_lane",
        n_permutations=10,
        random_state=5,
        min_size=1,
    )
    assert glm.metadata["null_method"] == "freedman_lane"
    assert glm.metadata["positive_direction"] == "positive tested contrast"
    with pytest.raises(ValueError, match="intercept"):
        analysis.glm(labels, np.arange(len(labels)), min_size=1)
    with pytest.raises(ValueError, match="contrast"):
        analysis.glm(labels, nuisance, contrast=np.ones(4), min_size=1)
    phenotype = np.linspace(-1, 1, len(labels))
    phenotype_result = analysis.phenotype(
        phenotype,
        null_method="label_permutation",
        n_permutations=10,
        random_state=7,
        min_size=1,
    )
    assert phenotype_result.metadata["permutation_scheme"] == (
        "shared_subject_phenotype_permutation"
    )
    assert phenotype_result.metadata["positive_direction"] == ("positive phenotype association")


def test_provided_null_validation_and_apply(example_edges, example_sets):
    arrays = provided_null({"a": [1, -1], "b": [0.5, -0.5]})
    assert arrays["a"].shape == (2,)
    with pytest.raises(ValueError):
        provided_null({"a": [1], "b": [1, 2]})
    with pytest.raises(ValueError):
        provided_null({"a": []})
    identifiers = ["0--1", "0--2", "0--3", "1--2", "1--3", "2--3"]
    statistic_matrix = np.array([[3, 2, 1, -1, -2, -3], [-2, 1, 3, -3, 2, -1]], dtype=float)
    scored_statistics = provided_null(
        statistic_matrix,
        kind="statistics",
        edge_ids=identifiers,
        edge_sets=example_sets,
    )
    assert scored_statistics["positive"].shape == (2,)
    rank_matrix = np.array([np.arange(6), np.arange(5, -1, -1)])
    scored_ranks = provided_null(
        rank_matrix,
        kind="ranks",
        edge_ids=identifiers,
        edge_sets=example_sets,
        weight=0,
    )
    assert scored_ranks["negative"].shape == (2,)
    with pytest.raises(ValueError, match="weight=0"):
        provided_null(
            rank_matrix,
            kind="ranks",
            edge_ids=identifiers,
            edge_sets=example_sets,
        )
    with pytest.raises(ValueError, match="complete rank"):
        provided_null(
            np.array([[0, 0, 1, 2, 3, 4]]),
            kind="ranks",
            edge_ids=identifiers,
            edge_sets=example_sets,
            weight=0,
        )
    with pytest.raises(ValueError, match="kind"):
        provided_null({"a": [1]}, kind="bad")
    matrix_result = lens_enrich(
        example_edges,
        example_sets,
        min_size=1,
        null_method="provided_null",
        provided_null=statistic_matrix,
        provided_null_kind="statistics",
        provided_null_edge_ids=identifiers,
        provided_null_edge_sets=example_sets,
        positive_direction="case > control",
        provided_null_direction="case > control",
    )
    assert matrix_result.metadata["provided_null_kind"] == "statistics"
    with pytest.raises(ValueError, match="direction"):
        lens_enrich(
            example_edges,
            example_sets,
            min_size=1,
            null_method="provided_null",
            provided_null=statistic_matrix,
            provided_null_kind="statistics",
            provided_null_edge_ids=identifiers,
            provided_null_edge_sets=example_sets,
            positive_direction="case > control",
            provided_null_direction="control > case",
        )
    result = lens_enrich(example_edges, example_sets, min_size=1)
    with pytest.raises(ValueError, match="missing null"):
        apply_null_inference(result, {"positive": [1, -1]})
