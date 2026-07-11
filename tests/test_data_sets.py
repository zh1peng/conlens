import numpy as np
import pandas as pd
import pytest

from conlens import (
    canonicalize_edges,
    edges_to_matrix,
    make_custom_edge_sets,
    make_hemisphere_sets,
    make_network_pair_sets,
    make_within_network_sets,
    matrix_to_edges,
    rank_edges,
    validate_connectome,
    validate_edge_sets,
    validate_edge_table,
)


def test_matrix_roundtrip_undirected():
    matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0.0]])
    edges = matrix_to_edges(matrix, ["z", "a", "m"])
    assert edges["edge_id"].tolist() == ["0--1", "0--2", "1--2"]
    np.testing.assert_allclose(edges_to_matrix(edges, ["z", "a", "m"]), matrix)


def test_directed_and_subject_matrix_conversion():
    matrix = np.array([[0, 1], [2, 0.0]])
    edges = matrix_to_edges(matrix, ["a", "b"], directed=True)
    assert set(edges["edge_id"]) == {"0->1", "1->0"}
    np.testing.assert_allclose(edges_to_matrix(edges, ["a", "b"], directed=True), matrix)
    long = matrix_to_edges(np.stack([matrix, matrix * 2]), ["a", "b"], directed=True)
    assert long.shape == (4, 6)
    assert set(long["subject"]) == {0, 1}
    assert set(long["edge_id"]) == {"0->1", "1->0"}


def test_canonicalization_uses_node_order_not_lexical():
    frame = pd.DataFrame({"node1": ["a"], "node2": ["z"]})
    result = canonicalize_edges(frame, node_order=["z", "a"])
    assert result.loc[0, ["node1", "node2", "edge_id"]].tolist() == ["z", "a", "0--1"]


def test_user_edge_ids_are_preserved_and_canonical_ids_drive_ties(example_edges):
    edges = example_edges.copy()
    edges["edge_id"] = ["custom-z", "custom-y", "custom-x", "custom-w", "custom-v", "custom-u"]
    edges.loc[0, "statistic"] = edges.loc[1, "statistic"]
    validated = validate_edge_table(edges)
    assert validated["edge_id"].tolist() == edges["edge_id"].tolist()
    assert validated["canonical_edge_id"].tolist()[0] == "0--1"
    ranked, _ = rank_edges(validated)
    assert ranked["edge_id"].tolist()[:2] == ["custom-z", "custom-y"]
    custom = make_custom_edge_sets({"x": pd.DataFrame({"node1": [1], "node2": [0]})}, validated)
    assert custom == {"x": {"custom-z"}}
    with pytest.raises(ValueError, match="duplicate edge_id"):
        validate_edge_table(edges.assign(edge_id="same"))


def test_edges_to_matrix_uses_requested_value_column(example_edges):
    edges = example_edges.copy()
    edges["effect"] = np.arange(len(edges), dtype=float)
    matrix = edges_to_matrix(edges, [0, 1, 2, 3], value_column="effect")
    assert matrix[0, 1] == 0
    assert matrix[2, 3] == 5
    with pytest.raises(ValueError, match="value column"):
        edges_to_matrix(edges, value_column="missing")
    labels = np.array([0, 1, 2, 3])
    assert edges_to_matrix(edges, labels).shape == (4, 4)


def test_mixed_type_node_labels_are_not_coerced():
    matrix = np.array([[0, 2], [2, 0]], dtype=float)
    edges = matrix_to_edges(matrix, [1, "two"])
    assert edges.loc[0, "node1"] == 1
    assert edges.loc[0, "node2"] == "two"


@pytest.mark.parametrize(
    "matrix",
    [np.ones((2, 3)), np.array([[0, 1], [2, 0.0]]), np.array([[0, np.nan], [np.nan, 0]])],
)
def test_invalid_connectomes(matrix):
    with pytest.raises(ValueError):
        validate_connectome(matrix)


def test_duplicate_nodes_edges_unknown_and_diagonal(example_edges):
    with pytest.raises(ValueError, match="unique"):
        validate_edge_table(example_edges, node_order=[0, 0, 1, 2, 3])
    duplicate = pd.concat([example_edges, example_edges.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_edge_table(duplicate)
    with pytest.raises(ValueError, match="unknown"):
        validate_edge_table(example_edges, node_order=[0, 1, 2])
    diagonal = pd.DataFrame({"node1": [0, 0], "node2": [0, 1], "statistic": [2, 1]})
    with pytest.raises(ValueError, match="diagonal"):
        validate_edge_table(diagonal)


def test_nonfinite_policy_and_missing_columns(example_edges):
    broken = example_edges.copy()
    broken.loc[1, "statistic"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_edge_table(broken)
    omitted = validate_edge_table(broken, nan_policy="omit")
    assert len(omitted) == 5
    assert omitted.attrs["omitted_edges"] == [
        {"row": 1, "node1": 0.0, "node2": 2.0, "reason": "non-finite statistic"}
    ]
    with pytest.raises(ValueError, match="missing"):
        validate_edge_table(example_edges.drop(columns="node1"))


def test_network_pair_and_custom_sets(example_edges):
    edges = validate_edge_table(example_edges)
    labels = {0: "B", 1: "A", 2: "B", 3: "A"}
    pairs = make_network_pair_sets(edges, labels)
    assert set(pairs) == {"A--A", "A--B", "B--B"}
    assert sum(map(len, pairs.values())) == len(edges)
    assert set(make_within_network_sets(edges, labels)) == {"A--A", "B--B"}
    assert make_hemisphere_sets(edges, labels) == pairs
    directed = make_network_pair_sets(edges, labels, directed=True)
    assert "B->A" in directed
    custom = make_custom_edge_sets({"x": pd.DataFrame({"node1": [1], "node2": [0]})}, edges)
    assert custom == {"x": {"0--1"}}
    with pytest.raises(ValueError, match="missing network"):
        make_network_pair_sets(edges, {0: "A"})


def test_validate_edge_sets_errors():
    with pytest.raises(ValueError, match="unknown"):
        validate_edge_sets({"x": ["a", "z"]}, ["a"])
    assert validate_edge_sets({"x": ["a", "z"]}, ["a"], unknown_policy="omit") == {"x": {"a"}}
    with pytest.raises(ValueError, match="duplicate"):
        validate_edge_sets({"x": ["a", "a"]}, ["a"])
