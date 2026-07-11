import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import stats

from conlens import (
    LeadingNetwork,
    LensResult,
    bootstrap_lens,
    build_leading_network,
    compare_leading_edges,
    compare_lens_results,
    consensus_network,
    identify_leading_hubs,
    lens_enrich,
    summarize_leading_network,
    summarize_stability,
)
from conlens.leading import compute_node_participation
from conlens.plotting import (
    plot_enrichment,
    plot_leading_adjacency,
    plot_leading_connectome,
    plot_nes,
    plot_network_pair_heatmap,
    plot_node_participation,
    plot_stability,
)


def _results(example_edges, example_sets):
    first = lens_enrich(
        example_edges,
        example_sets,
        min_size=1,
        store_running_sum=True,
        null_method="edge_permutation",
        n_permutations=12,
        random_state=1,
    )
    changed = example_edges.copy()
    changed["statistic"] = [2, 3, 1, -1.5, -0.5, -2.5]
    second = lens_enrich(changed, example_sets, min_size=1, store_running_sum=True)
    return first, second


def test_serialization_roundtrip(tmp_path, example_edges, example_sets):
    result, _ = _results(example_edges, example_sets)
    path = result.save(tmp_path / "result.json")
    restored = LensResult.load(path)
    assert restored.to_frame().to_dict("records") == result.to_frame().to_dict("records")
    assert restored.metadata == result.metadata
    assert restored.ranked_edges.equals(result.ranked_edges)
    assert json.loads(path.read_text())["metadata"]["package_version"] == "1.0.0"
    assert restored.get("positive").set_name == "positive"
    with pytest.raises(KeyError):
        restored.get("missing")


def test_serialization_preserves_optional_nonfinite_metadata(tmp_path, example_edges, example_sets):
    edges = example_edges.copy()
    edges["optional"] = [np.nan, np.inf, -np.inf, 1, 2, 3]
    result = lens_enrich(edges, example_sets, min_size=1)
    restored = LensResult.load(result.save(tmp_path / "nonfinite.json"))
    values = restored.ranked_edges.sort_values("canonical_edge_id")["optional"].tolist()
    assert np.isnan(values[0])
    assert values[1:3] == [np.inf, -np.inf]


def test_leading_network_and_comparison(example_edges, example_sets):
    first, second = _results(example_edges, example_sets)
    network = build_leading_network(first, "positive")
    assert set(network.nodes["node_id"]) == {0, 1, 2}
    assert len(network.edges) == 2
    graph = network.to_networkx()
    assert graph.number_of_edges() == 2
    summary = summarize_leading_network(network)
    assert summary["n_nodes"] == 3
    participation = compute_node_participation(network)
    assert set(participation.columns) >= {"node_id", "degree", "strength"}
    hubs = identify_leading_hubs(network, top_n=1)
    assert len(hubs) == 1
    with pytest.raises(ValueError):
        identify_leading_hubs(network)
    overlap = compare_leading_edges(first, second, "positive")
    assert 0 <= overlap["jaccard"] <= 1
    compared = compare_lens_results(first, second)
    assert set(compared["set_name"]) == set(example_sets)


def test_leading_network_export(tmp_path, example_edges, example_sets):
    result, _ = _results(example_edges, example_sets)
    network = build_leading_network(result, "positive")
    json_path = network.save(tmp_path / "leading.json")
    restored = LeadingNetwork.load(json_path)
    assert restored.nodes.equals(network.nodes)
    assert restored.edges.equals(network.edges)
    assert network.save(tmp_path / "leading.graphml").exists()
    with pytest.raises(ValueError, match="must end"):
        network.save(tmp_path / "leading.txt")


def test_stability_and_consensus(example_edges, example_sets):
    rng = np.random.default_rng(2)
    replicates = np.vstack([example_edges["statistic"] + rng.normal(0, 0.2, 6) for _ in range(5)])
    results = bootstrap_lens(
        example_edges, example_sets, statistic_replicates=replicates, min_size=1
    )
    summary = summarize_stability(results)
    assert summary["n_replicates"] == 5
    assert len(summary["sets"]["positive"]["pairwise_jaccard"]) == 10
    assert set(summary["sets"]["positive"]["edge_inclusion_frequency"]) == example_sets["positive"]
    assert summary["sets"]["positive"]["significance_alpha"] == 0.05
    consensus = consensus_network(results, "positive", threshold=0.5)
    assert "inclusion_frequency" in consensus
    assert bootstrap_lens(example_edges, example_sets, results=results) == results
    with pytest.raises(ValueError, match="exactly one"):
        bootstrap_lens(example_edges, example_sets)
    with pytest.raises(ValueError):
        consensus_network(results, "positive", threshold=1.1)
    with pytest.raises(ValueError, match="significance_alpha"):
        summarize_stability(results, significance_alpha=1)


def test_subject_level_bootstrap_is_explicit_and_reproducible(example_edges, example_sets):
    rng = np.random.default_rng(9)
    subject_data = rng.normal(size=(20, len(example_edges)))
    groups = np.repeat([0, 1], 10)

    def statistic(sample, indices):
        sampled_groups = groups[indices]
        return stats.ttest_ind(
            sample[sampled_groups == 1],
            sample[sampled_groups == 0],
            axis=0,
            equal_var=False,
        ).statistic

    first = bootstrap_lens(
        example_edges,
        example_sets,
        subject_data=subject_data,
        statistic_function=statistic,
        strata=groups,
        n_bootstraps=4,
        random_state=3,
        min_size=1,
    )
    second = bootstrap_lens(
        example_edges,
        example_sets,
        subject_data=subject_data,
        statistic_function=statistic,
        strata=groups,
        n_bootstraps=4,
        random_state=3,
        min_size=1,
    )
    assert [item.get("positive").ES for item in first] == [
        item.get("positive").ES for item in second
    ]
    with pytest.raises(ValueError, match="statistic_function"):
        bootstrap_lens(example_edges, example_sets, subject_data=subject_data)
    with pytest.raises(ValueError, match="strata"):
        bootstrap_lens(
            example_edges,
            example_sets,
            subject_data=subject_data,
            statistic_function=statistic,
            strata=[0],
        )


def test_all_plots(example_edges, example_sets):
    first, _ = _results(example_edges, example_sets)
    network = build_leading_network(first, "positive")
    axes = plot_enrichment(first, "positive", example_sets["positive"])
    assert len(axes) == 3
    assert plot_nes(first).get_xlabel() == "NES"
    assert plot_network_pair_heatmap({"A--A": 1, "A--B": -1}).images
    assert plot_leading_adjacency(network).images
    coordinates = {0: (0, 0), 1: (1, 0), 2: (0, 1)}
    assert plot_leading_connectome(network, coordinates).axison is False
    assert plot_node_participation(network).get_ylabel() == "Leading-edge degree"
    results = [first, first]
    summary = summarize_stability(results)
    assert plot_stability(summary, "positive").get_ylim() == (0.0, 1.0)
    plt.close("all")
