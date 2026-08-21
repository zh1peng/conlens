import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conlens import (
    Contrast,
    LensStabilityResult,
    lens_bootstrap,
    lens_edge_permute,
    lens_enrich,
    lens_stat,
    make_design,
    make_edge_statistics,
    summarize_stability,
)
from conlens.cli import main


def base_result(example_edges):
    sets = {"target": {"0--1", "0--2", "0--3"}}
    edges = make_edge_statistics(example_edges, positive_direction="higher")
    observed = lens_stat(edges, sets)
    null = (
        lens_stat(item, sets)
        for item in lens_edge_permute(edges, n_permutations=9, random_state=1)
    )
    return lens_enrich(observed, null, min_size=1, family_name="primary")


def test_observed_anchored_stability_formulas(example_edges, tmp_path: Path):
    observed = base_result(example_edges)
    observed.get("target").q_value = 0.01
    same = copy.deepcopy(observed)
    same.get("target").q_value = 0.02
    opposite = copy.deepcopy(observed)
    opposite.get("target").q_value = 0.03
    opposite.get("target").direction = "negative"
    undetected = copy.deepcopy(observed)
    undetected.get("target").q_value = 0.4
    summary = summarize_stability(
        observed, [same, opposite, undetected], min_same_direction=1
    )
    row = summary.get_set("target")
    assert row.detection_count == 2
    assert row.same_direction_count == 1
    assert row.set_stability == pytest.approx(1 / 3)
    edges = summary.edges_for("target")
    np.testing.assert_allclose(
        edges.full_pipeline_stability,
        row.set_stability * edges.conditional_stability,
    )
    path = tmp_path / "stability.json"
    summary.save(path)
    restored = LensStabilityResult.load(path)
    pd.testing.assert_frame_equal(restored.set_summary, summary.set_summary)
    wrong = copy.deepcopy(same)
    wrong.metadata["family_name"] = "other"
    with pytest.raises(ValueError, match="family_name"):
        summarize_stability(observed, [wrong])
    wrong_identity = copy.deepcopy(same)
    wrong_identity.metadata["node_identity_hash"] = "different"
    with pytest.raises(ValueError, match="node_identity_hash"):
        summarize_stability(observed, iter([wrong_identity]))


def test_full_pipeline_bootstrap_smoke():
    rng = np.random.default_rng(4)
    n, nodes = 24, 5
    raw = rng.normal(size=(n, nodes, nodes))
    values = (raw + raw.transpose(0, 2, 1)) / 2
    for matrix in values:
        np.fill_diagonal(matrix, 0)
    group = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    design = make_design(groups={"control": group == 0, "g1": group == 1})
    contrast = {
        "g1_vs_control": Contrast(
            {"g1": 1, "control": -1}, "hedges_g", "g1 > control"
        )
    }
    edge_sets = {"first": {"0--1", "0--2"}, "second": {"1--2", "1--3"}}
    options = dict(
        design=design,
        contrasts=contrast,
        n_bootstraps=2,
        n_permutations=3,
        strata=group,
        random_state=8,
        min_size=1,
        min_same_direction=1,
    )
    result = lens_bootstrap(
        values,
        edge_sets,
        n_jobs=1,
        **options,
    )
    parallel = lens_bootstrap(values, edge_sets, n_jobs=2, **options)
    assert set(result) == {"g1_vs_control"}
    assert result["g1_vs_control"].metadata["n_bootstraps"] == 2
    pd.testing.assert_frame_equal(
        result["g1_vs_control"].set_summary,
        parallel["g1_vs_control"].set_summary,
    )
    blocked = lens_bootstrap(
        values,
        edge_sets,
        n_jobs=1,
        exchangeability_blocks=[(int(index % 3), "site") for index in range(n)],
        **options,
    )
    assert blocked["g1_vs_control"].metadata["n_bootstraps"] == 2


def test_cli_edge_workflow(example_edges, tmp_path: Path):
    edge_path, set_path, output_path = (
        tmp_path / "edges.csv", tmp_path / "sets.json", tmp_path / "result.json"
    )
    example_edges.to_csv(edge_path, index=False)
    set_path.write_text(json.dumps({"target": ["0--1", "0--2"]}), encoding="utf-8")
    assert main([
        str(edge_path), str(set_path), str(output_path),
        "--positive-direction", "higher", "--min-size", "1",
        "--n-permutations", "5", "--random-state", "2",
    ]) == 0
    assert output_path.exists()
