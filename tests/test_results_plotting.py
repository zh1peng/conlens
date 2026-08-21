from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from conlens import (
    EdgeStatistics,
    GLMResult,
    LeadingNetwork,
    LensResult,
    LensStabilityResult,
    build_leading_network,
    compare_leading_edges,
    compare_lens_results,
    compute_node_participation,
    identify_leading_hubs,
    lens_edge_permute,
    lens_enrich,
    lens_stat,
    make_edge_statistics,
    plot_circos,
    plot_connectome_heatmap,
    plot_enrichment,
    plot_enrichment_heatmap,
    plot_leading_adjacency,
    plot_node_participation,
    plot_null_distribution,
    plot_running_sum,
    plot_stability,
    summarize_leading_network,
)


def inferred_result(example_edges):
    sets = {
        "A--A": {"0--1", "0--2"},
        "A--B": {"0--3", "1--2"},
        "B--B": {"1--3", "2--3"},
    }
    edges = make_edge_statistics(example_edges, positive_direction="higher")
    observed = lens_stat(edges, sets, store_running_sum=True)
    null = (
        lens_stat(item, sets)
        for item in lens_edge_permute(edges, n_permutations=11, random_state=3)
    )
    return lens_enrich(observed, null, min_size=1)


def test_result_roundtrip_and_leading_network(example_edges, tmp_path: Path):
    result = inferred_result(example_edges)
    path = tmp_path / "result.json"
    result.save(path)
    restored = LensResult.load(path)
    pd.testing.assert_frame_equal(restored.null_scores, result.null_scores)
    assert restored.metadata == result.metadata
    network = build_leading_network(restored, "A--A")
    summary = summarize_leading_network(network)
    assert summary["n_edges"] == len(network.edges)
    participation = compute_node_participation(network)
    assert set(participation) >= {"node_id", "degree", "strength"}
    assert len(identify_leading_hubs(network, top_n=1)) <= 1
    assert len(identify_leading_hubs(network, metric="strength", threshold=0)) == len(
        participation
    )
    comparison = compare_leading_edges(restored, result, "A--A")
    assert comparison["jaccard"] == 1
    assert compare_lens_results(restored, result).leading_edge_dice.eq(1).all()
    with pytest.raises(ValueError):
        identify_leading_hubs(network)


def test_visualizations_render(example_edges):
    result = inferred_result(example_edges)
    matrix = np.arange(16.0).reshape(4, 4)
    annotations = {0: "A", 1: "A", 2: "B", 3: "B"}
    assert plot_connectome_heatmap(matrix, annotations) is not None
    assert plot_running_sum(result, "A--A") is not None
    assert plot_null_distribution(result, "A--A") is not None
    assert len(plot_enrichment(result, "A--A")) == 3
    assert plot_enrichment_heatmap(result) is not None
    network = build_leading_network(result, "A--A")
    assert plot_circos(network, annotations) is not None
    assert plot_leading_adjacency(network) is not None
    assert plot_node_participation(network) is not None
    plt.close("all")


def test_leading_network_serialization_and_plot_errors(tmp_path: Path):
    network = LeadingNetwork(
        pd.DataFrame({"node_id": ["a", "b"]}),
        pd.DataFrame({"node1": ["a"], "node2": ["b"], "statistic": [1.0]}),
    )
    assert network.to_networkx().number_of_edges() == 1
    assert network.save(tmp_path / "network.json").exists()
    assert network.save(tmp_path / "network.graphml").exists()
    with pytest.raises(ValueError, match="missing network"):
        plot_circos(network, {"a": "A"})
    with pytest.raises(ValueError):
        plot_connectome_heatmap(np.ones((2, 3)), ["A", "B"])


def test_leading_metadata_directed_and_stability_plot(example_edges):
    result = inferred_result(example_edges)
    metadata = pd.DataFrame({
        "node_id": [0, 1, 2, 3],
        "network": ["A", "A", "B", "B"],
    })
    network = build_leading_network(result, "A--A", node_metadata=metadata)
    assert "network" in network.nodes
    with pytest.raises(ValueError, match="duplicate"):
        build_leading_network(result, "A--A", node_metadata=pd.concat([metadata, metadata]))
    with pytest.raises(ValueError, match="missing leading"):
        build_leading_network(result, "A--A", node_metadata=metadata.iloc[2:])
    directed = LeadingNetwork(
        pd.DataFrame({"node_id": ["a", "b"]}),
        pd.DataFrame({"node1": ["a"], "node2": ["b"], "statistic": [-2.0]}),
        directed=True,
    )
    participation = compute_node_participation(directed)
    assert set(participation) >= {"in_degree", "out_degree", "in_strength", "out_strength"}
    assert summarize_leading_network(
        LeadingNetwork(pd.DataFrame({"node_id": []}), pd.DataFrame(columns=["node1", "node2"]))
    )["n_components"] == 0

    stability = LensStabilityResult(
        pd.DataFrame({"set_name": ["x"]}),
        pd.DataFrame({
            "set_name": ["x", "x"], "edge_id": ["a", "b"],
            "full_pipeline_stability": [0.7, 0.2],
        }),
        pd.DataFrame(),
        {},
    )
    assert plot_stability(stability, "x") is not None
    _, ax = plt.subplots()
    assert plot_connectome_heatmap(
        np.eye(4), {0: "A", 1: "A", 2: "B", 3: "B"}, ax=ax
    ) is ax
    plt.close("all")


def test_serializable_edge_and_glm_results(example_edges, tmp_path: Path):
    edge = make_edge_statistics(example_edges, positive_direction="higher")
    edge_path = tmp_path / "edge.json"
    edge.save(edge_path)
    restored_edge = EdgeStatistics.load(edge_path)
    pd.testing.assert_frame_equal(restored_edge.table, edge.table)
    result = inferred_result(example_edges)
    glm = GLMResult({"demo": result}, {"family_name": "demo"})
    glm_path = tmp_path / "glm.json"
    glm.save(glm_path)
    restored_glm = GLMResult.load(glm_path)
    assert restored_glm.contrast_names == ("demo",)
    pd.testing.assert_frame_equal(restored_glm["demo"].null_scores, result.null_scores)
