import json

import numpy as np
import pytest
from pandas.testing import assert_frame_equal
from scipy.stats import beta

from conlens import (
    LensAnalysis,
    LensResult,
    LensStabilityResult,
    lens_enrich,
    summarize_bootstrap_stability,
)


def _template_result(example_edges):
    edge_sets = {
        "target": {"0--1", "0--2", "0--3"},
        "untracked": {"1--2", "1--3", "2--3"},
    }
    result = lens_enrich(
        example_edges,
        edge_sets,
        min_size=1,
        null_method="edge_permutation",
        n_permutations=20,
        random_state=1,
        positive_direction="case > control",
    )
    result.get("target").q_value = 0.01
    result.get("target").p_value = 0.005
    result.get("target").direction = "positive"
    result.get("target").leading_edge_ids = ["0--1", "0--2"]
    result.get("target").leading_edge_size = 2
    result.get("untracked").q_value = 0.50
    return result


def _replicate(template, *, q_value, direction, leading_edges):
    result = LensResult.from_dict(template.to_dict())
    target = result.get("target")
    target.q_value = q_value
    target.direction = direction
    target.leading_edge_ids = list(leading_edges)
    target.leading_edge_size = len(leading_edges)
    return result


def test_full_pipeline_counts_rates_and_intervals(example_edges):
    observed = _template_result(example_edges)
    results = [
        _replicate(
            observed, q_value=0.01, direction="positive", leading_edges=["0--1", "0--3"]
        ),
        _replicate(observed, q_value=0.01, direction="negative", leading_edges=["0--2"]),
        _replicate(
            observed, q_value=0.06, direction="positive", leading_edges=["0--1", "0--2"]
        ),
        _replicate(
            observed, q_value=0.05, direction="positive", leading_edges=["0--2", "0--3"]
        ),
    ]
    before_observed = observed.to_dict()
    before_results = [result.to_dict() for result in results]

    stability = summarize_bootstrap_stability(
        observed, results, min_same_direction=2, core_threshold=0.50
    )
    set_result = stability.get_set("target")
    assert set_result["detection_count"] == 3
    assert set_result["same_direction_count"] == 2
    assert set_result["different_direction_count"] == 1
    assert set_result["detection_rate"] == 0.75
    assert set_result["direction_consistency"] == pytest.approx(2 / 3)
    assert set_result["set_stability"] == 0.50
    expected_lower = beta.ppf(0.025, 2.5, 2.5)
    assert set_result["set_stability_lower"] == pytest.approx(expected_lower)
    assert set_result["conditional_localization_supported"]
    assert not set_result["conditional_core_reportable"]

    edges = stability.edges_for("target").set_index("edge_id")
    assert edges.loc["0--1", "same_direction_inclusion_count"] == 1
    assert edges.loc["0--2", "same_direction_inclusion_count"] == 1
    assert edges.loc["0--3", "same_direction_inclusion_count"] == 2
    np.testing.assert_allclose(edges["conditional_stability"], [0.5, 0.5, 1.0])
    np.testing.assert_allclose(edges["full_pipeline_stability"], [0.25, 0.25, 0.50])
    np.testing.assert_allclose(
        edges["full_pipeline_stability"],
        set_result["set_stability"] * edges["conditional_stability"],
    )
    assert set(stability.set_summary["set_name"]) == {"target"}
    assert len(stability.replicate_summary) == 4
    assert stability.replicate_summary["same_direction"].tolist() == [True, False, False, True]
    assert observed.to_dict() == before_observed
    assert [result.to_dict() for result in results] == before_results


def test_zero_denominators_and_conditional_gate(example_edges):
    observed = _template_result(example_edges)
    undetected = [
        _replicate(observed, q_value=0.50, direction="positive", leading_edges=["0--1"])
        for _ in range(3)
    ]
    stability = summarize_bootstrap_stability(observed, undetected)
    set_result = stability.get_set("target")
    assert set_result["detection_count"] == 0
    assert set_result["same_direction_count"] == 0
    assert np.isnan(set_result["direction_consistency"])
    assert set_result["set_stability"] == 0
    assert set_result["conditional_status"] == "insufficient same-direction detections"
    assert stability.edges_for("target")["conditional_stability"].isna().all()
    assert not stability.edges_for("target")["conditional_core"].any()

    opposite = [
        _replicate(observed, q_value=0.01, direction="negative", leading_edges=["0--1"])
        for _ in range(3)
    ]
    opposite_summary = summarize_bootstrap_stability(observed, opposite)
    opposite_set = opposite_summary.get_set("target")
    assert opposite_set["detection_count"] == 3
    assert opposite_set["direction_consistency"] == 0
    assert opposite_summary.edges_for("target")["full_pipeline_stability"].eq(0).all()


def test_full_and_conditional_cores_are_gated(example_edges):
    observed = _template_result(example_edges)
    results = [
        _replicate(
            observed,
            q_value=0.01,
            direction="positive",
            leading_edges=["0--1", "0--2"],
        )
        for _ in range(40)
    ]
    stability = summarize_bootstrap_stability(observed, results)
    set_result = stability.get_set("target")
    assert set_result["set_reproducibility_supported"]
    assert set_result["conditional_core_reportable"]
    edges = stability.edges_for("target").set_index("edge_id")
    assert edges.loc["0--1", "full_pipeline_core"]
    assert edges.loc["0--1", "conditional_core"]
    assert not edges.loc["0--3", "full_pipeline_core"]
    assert set_result["full_pipeline_core_size"] == 2
    assert set_result["conditional_core_size"] == 2


def test_stability_validation_and_empty_observed_sets(example_edges):
    observed = _template_result(example_edges)
    replicate = _replicate(
        observed, q_value=0.01, direction="positive", leading_edges=["0--1"]
    )

    incomplete = LensResult.from_dict(observed.to_dict())
    incomplete.get("target").q_value = None
    with pytest.raises(ValueError, match="finite q value"):
        summarize_bootstrap_stability(incomplete, [replicate])

    wrong_family = LensResult.from_dict(replicate.to_dict())
    wrong_family.metadata["correction_family_id"] = "other"
    with pytest.raises(ValueError, match="correction_family_id"):
        summarize_bootstrap_stability(observed, [wrong_family])

    wrong_definition = LensResult.from_dict(replicate.to_dict())
    wrong_definition.get("target").edge_set_ids = ["0--1", "0--2"]
    with pytest.raises(ValueError, match="definition differs"):
        summarize_bootstrap_stability(observed, [wrong_definition])

    wrong_node_order = LensResult.from_dict(replicate.to_dict())
    wrong_node_order.metadata["node_order"] = ["W", "X", "Y", "Z"]
    with pytest.raises(ValueError, match="node_order"):
        summarize_bootstrap_stability(observed, [wrong_node_order])

    wrong_edge_mapping = LensResult.from_dict(replicate.to_dict())
    assert wrong_edge_mapping.ranked_edges is not None
    canonical = wrong_edge_mapping.ranked_edges["canonical_edge_id"].copy()
    wrong_edge_mapping.ranked_edges.loc[:1, "canonical_edge_id"] = (
        canonical.iloc[:2][::-1].to_numpy()
    )
    with pytest.raises(ValueError, match="edge ID mapping"):
        summarize_bootstrap_stability(observed, [wrong_edge_mapping])

    wrong_signature = LensResult.from_dict(replicate.to_dict())
    wrong_signature.metadata["analysis_signature"] = {
        "kind": "glm",
        "contrast_vector": [0.0, 1.0],
    }
    with pytest.raises(ValueError, match="analysis_signature"):
        summarize_bootstrap_stability(observed, [wrong_signature])

    wrong_blocks = LensResult.from_dict(replicate.to_dict())
    wrong_blocks.metadata["exchangeability_blocks_summary"] = {
        "n_blocks": 2,
        "n_observations": 4,
    }
    with pytest.raises(ValueError, match="exchangeability_blocks_used"):
        summarize_bootstrap_stability(observed, [wrong_blocks])

    wrong_size = LensResult.from_dict(replicate.to_dict())
    wrong_size.get("target").leading_edge_size += 1
    with pytest.raises(ValueError, match="leading-edge size"):
        summarize_bootstrap_stability(observed, [wrong_size])

    missing_metadata = LensResult.from_dict(replicate.to_dict())
    del missing_metadata.metadata["edge_universe_hash"]
    with pytest.raises(ValueError, match="missing required metadata"):
        summarize_bootstrap_stability(observed, [missing_metadata])

    missing_direction = LensResult.from_dict(replicate.to_dict())
    missing_direction.metadata["positive_direction"] = None
    with pytest.raises(ValueError, match="incomplete analysis metadata"):
        summarize_bootstrap_stability(observed, [missing_direction])

    with pytest.raises(ValueError, match="core_threshold"):
        summarize_bootstrap_stability(observed, [replicate], core_threshold=0)

    nonsignificant = LensResult.from_dict(observed.to_dict())
    nonsignificant.get("target").q_value = 0.50
    empty = summarize_bootstrap_stability(nonsignificant, [replicate])
    assert empty.set_summary.empty
    assert empty.edge_summary.empty
    assert empty.replicate_summary.empty


def test_multiple_sets_align_by_name_not_position(example_edges):
    observed = _template_result(example_edges)
    untracked = observed.get("untracked")
    untracked.q_value = 0.01
    untracked.direction = "negative"
    untracked.leading_edge_ids = ["1--3", "2--3"]
    untracked.leading_edge_size = 2
    replicate = _replicate(
        observed, q_value=0.01, direction="positive", leading_edges=["0--1"]
    )
    replicate_untracked = replicate.get("untracked")
    replicate_untracked.q_value = 0.01
    replicate_untracked.direction = "negative"
    replicate_untracked.leading_edge_ids = ["1--3"]
    replicate_untracked.leading_edge_size = 1
    replicate.sets.reverse()

    stability = summarize_bootstrap_stability(
        observed, [replicate], min_same_direction=1, interval_level=0.50
    )
    assert stability.set_summary["set_name"].tolist() == ["target", "untracked"]
    assert stability.get_set("target")["same_direction_count"] == 1
    assert stability.get_set("untracked")["same_direction_count"] == 1
    assert stability.edges_for("target")["set_name"].eq("target").all()
    assert stability.edges_for("untracked")["set_name"].eq("untracked").all()


def test_stability_serialization_roundtrip(tmp_path, example_edges):
    observed = _template_result(example_edges)
    results = [
        _replicate(observed, q_value=0.50, direction="positive", leading_edges=["0--1"])
    ]
    stability = summarize_bootstrap_stability(
        observed, results, keep_bootstrap_results=True
    )
    path = stability.save(tmp_path / "stability.json")
    restored = LensStabilityResult.load(path)
    assert_frame_equal(restored.set_summary, stability.set_summary, check_dtype=False)
    assert_frame_equal(restored.edge_summary, stability.edge_summary, check_dtype=False)
    assert_frame_equal(restored.replicate_summary, stability.replicate_summary, check_dtype=False)
    assert restored.metadata == stability.metadata
    assert restored.bootstrap_results is not None
    assert len(restored.bootstrap_results) == 1
    assert json.loads(path.read_text(encoding="utf-8"))["metadata"]["n_bootstraps"] == 1
    with pytest.raises(KeyError):
        restored.edges_for("missing")

    nonsignificant = LensResult.from_dict(observed.to_dict())
    nonsignificant.get("target").q_value = 0.50
    empty = summarize_bootstrap_stability(nonsignificant, results)
    restored_empty = LensStabilityResult.load(empty.save(tmp_path / "empty_stability.json"))
    assert restored_empty.set_summary.columns.tolist() == empty.set_summary.columns.tolist()
    assert restored_empty.edge_summary.columns.tolist() == empty.edge_summary.columns.tolist()
    assert restored_empty.replicate_summary.columns.tolist() == (
        empty.replicate_summary.columns.tolist()
    )


def test_subject_executor_is_reproducible_across_jobs():
    rng = np.random.default_rng(24)
    n_subjects, n_nodes = 16, 4
    groups = np.repeat([0, 1], n_subjects // 2)
    site = np.tile(np.repeat(["A", "B"], 4), 2)
    edge_data = rng.normal(size=(n_subjects, 6))
    edge_data[groups == 1, :2] += 1.0
    connectomes = np.zeros((n_subjects, n_nodes, n_nodes))
    upper = np.triu_indices(n_nodes, 1)
    connectomes[:, upper[0], upper[1]] = edge_data
    connectomes[:, upper[1], upper[0]] = edge_data
    edge_sets = {
        "front": {"0--1", "0--2", "0--3"},
        "back": {"1--2", "1--3", "2--3"},
    }
    analysis = LensAnalysis.from_subject_connectomes(connectomes, edge_sets)
    observed = analysis.two_group(
        groups,
        null_method="label_permutation",
        n_permutations=8,
        random_state=7,
        min_size=1,
    )

    def refit(sample, indices, fit_seed):
        return sample.two_group(
            groups[indices],
            null_method="label_permutation",
            n_permutations=8,
            random_state=fit_seed,
            min_size=1,
        )

    options = {
        "n_bootstraps": 4,
        "random_state": 12,
        "strata": list(zip(site, groups, strict=True)),
        "keep_bootstrap_results": True,
    }
    serial = analysis.bootstrap_stability(observed, refit, n_jobs=1, **options)
    parallel = analysis.bootstrap_stability(observed, refit, n_jobs=2, **options)
    assert_frame_equal(serial.set_summary, parallel.set_summary)
    assert_frame_equal(serial.edge_summary, parallel.edge_summary)
    assert_frame_equal(serial.replicate_summary, parallel.replicate_summary)
    assert serial.bootstrap_results is not None
    assert parallel.bootstrap_results is not None
    serial_values = [
        (result.metadata["random_seed"], result.to_frame()["q_value"].tolist())
        for result in serial.bootstrap_results
    ]
    parallel_values = [
        (result.metadata["random_seed"], result.to_frame()["q_value"].tolist())
        for result in parallel.bootstrap_results
    ]
    assert serial_values == parallel_values
    assert serial.metadata["resampling_method"] == "stratified_subject"
    assert serial.metadata["n_strata"] == 4


def test_subject_executor_rejects_missing_strata_and_failed_refits():
    matrices = np.repeat(np.eye(3)[None, :, :], 4, axis=0)
    analysis = LensAnalysis.from_subject_connectomes(matrices, {"set": {"0--1"}})
    observed = lens_enrich(
        analysis.edge_template.assign(statistic=[1.0, 0.0, -1.0]),
        {"set": {"0--1"}},
        min_size=1,
        null_method="edge_permutation",
        n_permutations=2,
        random_state=1,
        positive_direction="higher statistic",
    )
    with pytest.raises(ValueError, match="missing"):
        analysis.bootstrap_stability(
            observed,
            lambda sample, indices, seed: observed,
            n_bootstraps=1,
            strata=["A", "A", None, "B"],
        )
    with pytest.raises(ValueError, match="missing"):
        analysis.bootstrap_stability(
            observed,
            lambda sample, indices, seed: observed,
            n_bootstraps=1,
            strata=[("A", 0), ("A", 1), ("B", None), ("B", 1)],
        )
    with pytest.raises(RuntimeError, match="replicate 0"):
        analysis.bootstrap_stability(
            observed,
            lambda sample, indices, seed: (_ for _ in ()).throw(ValueError("bad design")),
            n_bootstraps=1,
            random_state=1,
        )


def test_subject_executor_preflights_first_result_before_remaining_refits():
    matrices = np.repeat(np.eye(3)[None, :, :], 4, axis=0)
    analysis = LensAnalysis.from_subject_connectomes(matrices, {"set": {"0--1"}})
    observed = lens_enrich(
        analysis.edge_template.assign(statistic=[1.0, 0.0, -1.0]),
        {"set": {"0--1"}},
        min_size=1,
        null_method="edge_permutation",
        n_permutations=2,
        random_state=1,
        positive_direction="higher statistic",
    )
    calls = []

    def wrong_refit(sample, indices, seed):
        calls.append(seed)
        result = LensResult.from_dict(observed.to_dict())
        result.metadata["correction_family_id"] = "wrong"
        return result

    with pytest.raises(ValueError, match="correction_family_id"):
        analysis.bootstrap_stability(
            observed,
            wrong_refit,
            n_bootstraps=3,
            n_jobs=2,
            random_state=1,
        )
    assert len(calls) == 1
