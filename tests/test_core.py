import numpy as np
import pandas as pd
import pytest

from conlens import (
    compute_enrichment_score,
    compute_running_sum,
    extract_leading_edges,
    lens_enrich,
    rank_edges,
)


def test_hand_calculated_positive_and_negative(example_edges, example_sets):
    result = lens_enrich(example_edges, example_sets, min_size=1, store_running_sum=True)
    positive = result.get("positive")
    assert positive.ES == pytest.approx(0.7692307692307693, abs=1e-12)
    assert positive.leading_edge_ids == ["0--1", "0--2"]
    assert positive.running_sum[-1] == 0
    negative = result.get("negative")
    assert negative.ES == pytest.approx(-2 / 3, abs=1e-12)
    assert negative.leading_edge_ids == ["1--3", "2--3"]
    assert result.metadata["inference_status"] == "not_requested"
    assert negative.NES is negative.p_value is negative.q_value is None


def test_running_sum_unweighted_and_zero_weight_fallback():
    profile, fallback = compute_running_sum([3, 2, 1, 0], [True, False, True, False], weight=0)
    np.testing.assert_allclose(profile, [0, 0.5, 0, 0.5, 0], atol=1e-12)
    assert not fallback
    profile, fallback = compute_running_sum([0, 2, 0, -1], [True, False, True, False], weight=1)
    np.testing.assert_allclose(profile, [0, 0.5, 0, 0.5, 0], atol=1e-12)
    assert fallback


@pytest.mark.parametrize(
    ("statistics", "hits", "weight"),
    [([1, 2], [True], 1), ([1, 2], [True, False], -1), ([1, np.nan], [True, False], 1)],
)
def test_running_sum_invalid(statistics, hits, weight):
    with pytest.raises(ValueError):
        compute_running_sum(statistics, hits, weight=weight)


def test_running_sum_rejects_empty_and_full_sets():
    with pytest.raises(ValueError, match="empty or full"):
        compute_running_sum([2, 1], [False, False])
    with pytest.raises(ValueError, match="empty or full"):
        compute_running_sum([2, 1], [True, True])


def test_score_types_ambiguity_and_plateau():
    profile = [0, 0.5, 0, -0.5, 0]
    standard = compute_enrichment_score(profile)
    assert standard == {
        "ES": 0.0,
        "ES_positive": 0.5,
        "ES_negative": -0.5,
        "direction": "ambiguous",
        "peak_rank": None,
    }
    assert compute_enrichment_score(profile, score_type="positive")["peak_rank"] == 1
    assert compute_enrichment_score(profile, score_type="negative")["peak_rank"] == 3
    plateau = compute_enrichment_score([0, -0.5, -0.5, 0.2, 0], score_type="negative")
    assert plateau["peak_rank"] == 2
    with pytest.raises(ValueError):
        compute_enrichment_score(profile, score_type="bad")


def test_extract_leading_edges_rules():
    ids = ["a", "b", "c", "d"]
    hits = [True, False, True, True]
    assert extract_leading_edges(ids, hits, 0.5, 2) == ["a"]
    assert extract_leading_edges(ids, hits, -0.5, 2) == ["c", "d"]
    assert extract_leading_edges(ids, hits, 0, None) == []


def test_ranking_ties_are_deterministic():
    edges = pd.DataFrame({"edge_id": ["b", "a", "c"], "statistic": [1.0, 1.0, 0.0]})
    ranked, metadata = rank_edges(edges)
    assert ranked["edge_id"].tolist() == ["a", "b", "c"]
    assert metadata["n_tied_edges"] == 2
    assert metadata["tied_edge_fraction"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="identical"):
        rank_edges(pd.DataFrame({"edge_id": ["a", "b"], "statistic": [1, 1]}))


def test_set_filtering_and_invalid_status(example_edges):
    result = lens_enrich(
        example_edges,
        {"small": {"0--1"}, "full": {"0--1", "0--2", "0--3", "1--2", "1--3", "2--3"}},
    )
    assert result.get("small").status == "filtered"
    assert result.get("full").status == "invalid"


@pytest.mark.parametrize(
    "options",
    [
        {"weight": -1},
        {"score_type": "bad"},
        {"min_size": 3, "max_size": 2},
        {"null_method": "bad"},
        {"null_method": "edge_permutation", "n_permutations": 0},
    ],
)
def test_high_level_parameter_validation(example_edges, options):
    with pytest.raises(ValueError):
        lens_enrich(example_edges, {"small": {"0--1"}}, **options)


def test_ranking_rejects_invalid_statistics_and_ids():
    with pytest.raises(ValueError, match="finite"):
        rank_edges(pd.DataFrame({"edge_id": ["a", "b"], "statistic": [1, np.nan]}))
    with pytest.raises(ValueError, match="unique"):
        rank_edges(pd.DataFrame({"edge_id": ["a", "a"], "statistic": [2, 1]}))
