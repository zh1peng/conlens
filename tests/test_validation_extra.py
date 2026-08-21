import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conlens import (
    Contrast,
    compute_enrichment_score,
    compute_running_sum,
    lens_edge_permute,
    lens_enrich,
    lens_fl_permute,
    lens_glm,
    lens_stat,
    make_design,
    make_edge_statistics,
    plot_enrichment_heatmap,
    rank_edges,
    summarize_stability,
)
from conlens.interfaces.nilearn import from_nilearn_connectivity, plot_nilearn_connectome


def symmetric_connectomes(n=8, nodes=4):
    rng = np.random.default_rng(3)
    raw = rng.normal(size=(n, nodes, nodes))
    values = (raw + raw.transpose(0, 2, 1)) / 2
    for matrix in values:
        np.fill_diagonal(matrix, 0)
    return values


def test_nilearn_adapter_and_optional_plot(monkeypatch):
    values = symmetric_connectomes(n=2, nodes=3)
    labels = ["a", "b", "c"]
    metadata = pd.DataFrame({"node_id": labels, "network": ["A", "A", "B"]})
    edges = from_nilearn_connectivity(
        values, labels, coordinates=np.zeros((3, 3)), node_metadata=metadata
    )
    assert edges.attrs["atlas_labels"] == labels
    with pytest.raises(ValueError, match="coordinates"):
        from_nilearn_connectivity(values, labels, coordinates=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="one row"):
        from_nilearn_connectivity(values, labels, node_metadata=metadata.iloc[:2])
    with pytest.raises(ValueError, match="order"):
        from_nilearn_connectivity(values, labels, node_metadata=metadata.iloc[::-1])
    fake = SimpleNamespace(plot_connectome=lambda adjacency, coordinates, **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "nilearn", SimpleNamespace(plotting=fake))
    assert plot_nilearn_connectome(np.eye(2), np.zeros((2, 3)), color="red") == {
        "color": "red"
    }


def test_core_and_design_validation_paths(example_edges, example_sets):
    with pytest.raises(ValueError, match="equal length"):
        compute_running_sum([1, 2], [True])
    with pytest.raises(ValueError, match="finite"):
        compute_enrichment_score([0, np.nan])
    with pytest.raises(ValueError, match="finite"):
        rank_edges(pd.DataFrame({"edge_id": ["a", "b"], "statistic": [1, np.nan]}))
    with pytest.raises(ValueError, match="unique"):
        rank_edges(pd.DataFrame({"edge_id": ["a", "a"], "statistic": [1, 0]}))
    with pytest.raises(ValueError, match="combined"):
        make_design(matrix=np.eye(4), column_names=list("abcd"), continuous={"x": range(4)})
    with pytest.raises(ValueError, match="column_names"):
        make_design(continuous={"x": range(4)}, column_names=["x"])
    with pytest.raises(ValueError, match="provide"):
        make_design()
    with pytest.raises(TypeError, match="boolean"):
        make_design(continuous={"x": range(4)}, center_continuous=1)
    with pytest.raises(ValueError, match="0/1"):
        make_design(indicators={"bad": [0, 2, 0, 1]})
    with pytest.raises(ValueError, match="unknown columns"):
        make_design(
            continuous={"x": range(6)}, interactions={"x_y": ("x", "y")}
        )
    with pytest.raises(ValueError, match="pair"):
        make_design(continuous={"x": range(6)}, interactions={"bad": ("x",)})
    with pytest.raises(ValueError, match="positive_direction"):
        make_edge_statistics(example_edges, positive_direction="")


def test_glm_permutation_and_enrichment_validation(example_edges, example_sets):
    values = symmetric_connectomes()
    design = make_design(continuous={"age": range(8)})
    contrast = {"age": Contrast({"age": 1}, "partial_r", "positive")}
    with pytest.raises(ValueError, match="positive integer"):
        list(lens_fl_permute(
            values, design=design, contrasts=contrast, n_permutations=0
        ))
    with pytest.raises(ValueError, match="one value"):
        list(lens_fl_permute(
            values, design=design, contrasts=contrast, n_permutations=1,
            exchangeability_blocks=[1, 2],
        ))
    with pytest.raises(ValueError, match="one row"):
        lens_glm(values, design=make_design(continuous={"age": range(7)}), contrasts=contrast)
    edge = make_edge_statistics(example_edges, positive_direction="higher")
    with pytest.raises(ValueError, match="same ordered"):
        list(lens_edge_permute(
            {"a": edge, "b": make_edge_statistics(
                example_edges.iloc[::-1], positive_direction="higher"
            )}, n_permutations=1,
        ))
    observed = lens_stat(edge, example_sets)
    with pytest.raises(ValueError, match="at least one"):
        lens_enrich(observed, iter(()), min_size=1)
    with pytest.raises(ValueError, match="family_name"):
        lens_enrich(observed, min_size=1, family_name="")
    descriptive = lens_enrich(observed, min_size=1)
    with pytest.raises(KeyError):
        descriptive.null_for("positive")
    with pytest.raises(ValueError, match="network pairs"):
        plot_enrichment_heatmap(descriptive)


@pytest.mark.parametrize(
    "options",
    [
        {"significance_alpha": 0},
        {"interval_level": 1},
        {"core_threshold": 0},
        {"min_same_direction": 0},
    ],
)
def test_stability_option_validation(example_edges, options):
    edge = make_edge_statistics(example_edges, positive_direction="higher")
    stat = lens_stat(edge, {"target": {"0--1", "0--2"}})
    null = (lens_stat(item, {"target": {"0--1", "0--2"}})
            for item in lens_edge_permute(edge, n_permutations=3, random_state=1))
    result = lens_enrich(stat, null, min_size=1)
    with pytest.raises(ValueError):
        summarize_stability(result, [result], **options)
