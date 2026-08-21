import numpy as np
import pandas as pd
import pytest

from conlens import (
    adjust_pvalues,
    compute_enrichment_score,
    compute_running_sum,
    extract_leading_edges,
    lens_edge_permute,
    lens_enrich,
    lens_stat,
    make_edge_statistics,
    rank_edges,
)


def prepared(example_edges):
    return make_edge_statistics(example_edges, positive_direction="higher values")


def test_hand_calculated_stat_and_filtering(example_edges, example_sets):
    statistic = lens_stat(prepared(example_edges), example_sets, store_running_sum=True)
    positive = statistic.get("positive")
    assert positive.ES == pytest.approx(0.7692307692307693)
    assert positive.leading_edge_ids == ["0--1", "0--2"]
    assert positive.running_sum[-1] == 0
    assert statistic.get("negative").ES == pytest.approx(-2 / 3)
    result = lens_enrich(statistic, min_size=4, max_size=4, family_name="primary")
    assert all(item.status == "filtered" for item in result.sets)
    assert result.metadata["family_name"] == "primary"
    assert result.metadata["inference_status"] == "descriptive"


def test_streamed_edge_null_is_reproducible_and_retained(example_edges, example_sets):
    edges = prepared(example_edges)
    observed = lens_stat(edges, example_sets, store_running_sum=True)

    def fit(seed):
        null = (
            lens_stat(item, example_sets)
            for item in lens_edge_permute(edges, n_permutations=19, random_state=seed)
        )
        return lens_enrich(observed, null, min_size=1, family_name="network-pairs")

    first, second = fit(42), fit(42)
    pd.testing.assert_frame_equal(first.null_scores, second.null_scores)
    assert first.null_scores.shape == (19, 2)
    assert first.metadata["permutation_scheme"] == "edge_label_permutation"
    assert first.metadata["adjustment_method"] == "BH"
    assert all(item.q_value is not None for item in first.sets)
    assert first.null_for("positive").equals(first.null_scores["positive"])
    with pytest.raises(KeyError):
        first.null_for("missing")


def test_mapping_uses_joint_bh_family(example_edges, example_sets):
    one = prepared(example_edges)
    two = make_edge_statistics(
        example_edges.assign(statistic=-example_edges["statistic"]),
        positive_direction="lower values",
    )
    observed = lens_stat({"one": one, "two": two}, example_sets)
    null_edges = lens_edge_permute(
        {"one": one, "two": two}, n_permutations=9, random_state=4
    )
    result = lens_enrich(
        observed,
        (lens_stat(item, example_sets) for item in null_edges),
        min_size=1,
        family_name="two-contrasts",
    )
    assert result.metadata["n_tests_in_family"] == 4
    assert result.contrast_names == ("one", "two")
    assert len(result.to_frame()) == 4


def test_running_sum_scoring_ranking_and_errors():
    profile, fallback = compute_running_sum(
        [3, 2, 1, 0], [True, False, True, False], weight=0
    )
    np.testing.assert_allclose(profile, [0, 0.5, 0, 0.5, 0])
    assert not fallback
    profile, fallback = compute_running_sum(
        [0, 2, 0, -1], [True, False, True, False], weight=1
    )
    assert fallback and profile[-1] == 0
    score = compute_enrichment_score([0, 0.5, 0, -0.5, 0])
    assert score["direction"] == "ambiguous"
    assert compute_enrichment_score([0, -0.5, -0.5, 0], score_type="negative")["peak_rank"] == 2
    assert extract_leading_edges(["a", "b", "c"], [True, False, True], 0.5, 2) == ["a"]
    ranked, metadata = rank_edges(
        pd.DataFrame({"edge_id": ["b", "a", "c"], "statistic": [1, 1, 0]})
    )
    assert ranked.edge_id.tolist() == ["a", "b", "c"]
    assert metadata["n_tied_edges"] == 2
    with pytest.raises(ValueError):
        compute_running_sum([1, 2], [True, False], weight=-1)
    with pytest.raises(ValueError):
        compute_enrichment_score([0, 1], score_type="bad")
    with pytest.raises(ValueError, match="identical"):
        rank_edges(pd.DataFrame({"edge_id": ["a", "b"], "statistic": [1, 1]}))


def test_validation_and_bh(example_edges, example_sets):
    assert adjust_pvalues([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    with pytest.raises(ValueError):
        adjust_pvalues([1.1])
    with pytest.raises(ValueError, match="positive_direction"):
        lens_stat(example_edges, example_sets)
    with pytest.raises(ValueError):
        list(lens_edge_permute(prepared(example_edges), n_permutations=0))
    observed = lens_stat(prepared(example_edges), example_sets)
    with pytest.raises(ValueError):
        lens_enrich(observed, min_size=0)
    with pytest.raises(TypeError):
        lens_enrich({})
    with pytest.raises(TypeError):
        lens_enrich({"bad": object()})
    wrong = lens_stat(prepared(example_edges), {"other": {"0--1"}})
    with pytest.raises(ValueError, match="different edge sets"):
        lens_enrich(observed, [wrong], min_size=1)
