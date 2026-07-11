import builtins
import json

import numpy as np
import pandas as pd
import pytest

from conlens import LensAnalysis, LensResult, lens_enrich
from conlens.cli import build_parser, main
from conlens.interfaces.nilearn import from_nilearn_connectivity, plot_nilearn_connectome
from conlens.leading import build_leading_network, identify_leading_hubs
from conlens.plotting import plot_leading_connectome, plot_running_sum
from conlens.results import LeadingNetwork


def test_cli_descriptive_roundtrip(tmp_path, example_edges, example_sets):
    edge_path = tmp_path / "edges.csv"
    set_path = tmp_path / "sets.json"
    output_path = tmp_path / "result.json"
    example_edges.to_csv(edge_path, index=False)
    set_path.write_text(
        json.dumps({name: sorted(values) for name, values in example_sets.items()}),
        encoding="utf-8",
    )
    assert (
        main(
            [
                str(edge_path),
                str(set_path),
                str(output_path),
                "--min-size",
                "1",
                "--store-running-sum",
            ]
        )
        == 0
    )
    result = LensResult.load(output_path)
    assert result.get("positive").ES == pytest.approx(10 / 13)
    assert build_parser().prog == "conlens"


def test_cli_provided_null_requires_path(tmp_path, example_edges, example_sets):
    edge_path = tmp_path / "edges.csv"
    set_path = tmp_path / "sets.json"
    example_edges.to_csv(edge_path, index=False)
    set_path.write_text(json.dumps({"positive": sorted(example_sets["positive"])}))
    with pytest.raises(SystemExit, match="provided-null"):
        main(
            [
                str(edge_path),
                str(set_path),
                str(tmp_path / "out.json"),
                "--min-size",
                "1",
                "--null-method",
                "provided_null",
            ]
        )


def test_cli_structured_provided_null(tmp_path, example_edges, example_sets):
    edge_path = tmp_path / "edges.csv"
    set_path = tmp_path / "sets.json"
    null_path = tmp_path / "null.json"
    output_path = tmp_path / "result.json"
    example_edges.to_csv(edge_path, index=False)
    serial_sets = {name: sorted(members) for name, members in example_sets.items()}
    set_path.write_text(json.dumps(serial_sets), encoding="utf-8")
    null_path.write_text(
        json.dumps(
            {
                "data": {name: [-0.5, 0.2, 0.8] for name in example_sets},
                "edge_ids": ["0--1", "0--2", "0--3", "1--2", "1--3", "2--3"],
                "edge_sets": serial_sets,
                "positive_direction": "case > control",
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                str(edge_path),
                str(set_path),
                str(output_path),
                "--min-size",
                "1",
                "--null-method",
                "provided_null",
                "--provided-null",
                str(null_path),
            ]
        )
        == 0
    )
    result = LensResult.load(output_path)
    assert result.metadata["provided_null_validation"] == ("edge_order_sets_and_direction_match")


def test_nilearn_conversion_and_optional_import(monkeypatch):
    matrices = np.array([[[0, 1], [1, 0]], [[0, 2], [2, 0]]], dtype=float)
    metadata = pd.DataFrame({"node_id": ["a", "b"], "network": ["X", "Y"]})
    coordinates = np.array([[0, 0, 0], [1, 2, 3]])
    converted = from_nilearn_connectivity(
        matrices, ["a", "b"], coordinates=coordinates, node_metadata=metadata
    )
    assert converted["statistic"].tolist() == [1, 2]
    assert converted.attrs["atlas_labels"] == ["a", "b"]
    assert converted.attrs["node_coordinates"] == coordinates.tolist()
    with pytest.raises(ValueError, match="coordinates"):
        from_nilearn_connectivity(matrices, ["a", "b"], coordinates=np.zeros((2, 2)))
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "nilearn" or name.startswith("nilearn."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"conlens\[nilearn\]"):
        plot_nilearn_connectome(np.eye(2), np.zeros((2, 3)))


def test_analysis_object_and_leading_errors(example_edges, example_sets):
    analysis = LensAnalysis(example_edges, example_sets, min_size=1)
    assert analysis.run().get("negative").ES == pytest.approx(-2 / 3)
    result = lens_enrich(example_edges, example_sets, min_size=1)
    metadata = pd.DataFrame({"node_id": [0, 1, 2, 3], "label": list("abcd")})
    network = build_leading_network(result, "positive", node_metadata=metadata)
    assert "label" in network.nodes
    with pytest.raises(ValueError, match="node_id"):
        build_leading_network(result, "positive", node_metadata=pd.DataFrame({"node": [0]}))
    with pytest.raises(ValueError, match="unknown participation"):
        identify_leading_hubs(network, top_n=1, metric="bad")
    reconstructed = plot_running_sum(result, "positive")
    assert reconstructed.get_ylabel() == "Running sum"
    with pytest.raises(ValueError, match="missing coordinates"):
        plot_leading_connectome(network, {})


def test_empty_leading_network_summary():
    from conlens import summarize_leading_network

    network = LeadingNetwork(
        pd.DataFrame({"node_id": []}),
        pd.DataFrame({"node1": [], "node2": []}),
    )
    assert summarize_leading_network(network)["n_components"] == 0
