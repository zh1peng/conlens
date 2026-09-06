import json
from pathlib import Path

import numpy as np

from conlens import compute_enrichment_score, compute_running_sum, extract_leading_edges


def test_fixed_gsea_reference_fixture():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "gsea_reference.json").read_text())
    hits = [edge in fixture["members"] for edge in fixture["edge_ids"]]
    profile, fallback = compute_running_sum(fixture["statistics"], hits, weight=fixture["weight"])
    np.testing.assert_allclose(profile, fixture["expected_running_sum"], atol=1e-12)
    score = compute_enrichment_score(profile)
    assert score["ES"] == fixture["expected_es"]
    assert (
        extract_leading_edges(fixture["edge_ids"], hits, score["ES"], score["peak_rank"])
        == fixture["expected_leading_edges"]
    )
    assert not fallback


def test_large_universe_running_sum():
    n_edges = 100_000
    statistics = np.linspace(5, -5, n_edges)
    hits = np.zeros(n_edges, dtype=bool)
    hits[::20] = True
    profile, _ = compute_running_sum(statistics, hits)
    assert profile.shape == (n_edges + 1,)
    assert np.isfinite(profile).all()
    assert profile[0] == 0
    assert profile[-1] == 0
