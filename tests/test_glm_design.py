import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import conlens
from conlens import Contrast, GLMResult, LensAnalysis, make_design, plot_design
from conlens.inference import adjust_pvalues
from conlens.stats import glm_contrast_statistics


def _subject_analysis():
    rng = np.random.default_rng(42)
    n_subjects, n_nodes = 24, 4
    groups = np.repeat(["control", "g1", "g2"], 8)
    age = np.linspace(20, 65, n_subjects)
    edge_data = rng.normal(scale=0.7, size=(n_subjects, 6))
    edge_data[groups == "g1", :3] += 0.9
    edge_data[groups == "g2", 3:] -= 0.7
    edge_data += (age - age.mean())[:, None] * np.array([0.01, 0.02, 0, -0.01, 0, 0.015])
    connectomes = np.zeros((n_subjects, n_nodes, n_nodes))
    upper = np.triu_indices(n_nodes, 1)
    connectomes[:, upper[0], upper[1]] = edge_data
    connectomes[:, upper[1], upper[0]] = edge_data
    sets = {
        "front": {"0--1", "0--2", "0--3"},
        "back": {"1--2", "1--3", "2--3"},
    }
    analysis = LensAnalysis.from_subject_connectomes(connectomes, sets)
    design = make_design(
        indicators={
            "control": groups == "control",
            "g1": groups == "g1",
            "g2": groups == "g2",
        },
        continuous={
            "age": age,
        },
        add_intercept=False,
    )
    contrasts = {
        "g1_vs_control": Contrast(
            {"g1": 1, "control": -1},
            effect_size="hedges_g",
            positive_direction="g1 > control",
        ),
        "g2_vs_control": Contrast(
            {"g2": 1, "control": -1},
            effect_size="hedges_g",
            positive_direction="g2 > control",
        ),
        "age": Contrast(
            {"age": 1},
            effect_size="partial_r",
            positive_direction="increases with age",
        ),
    }
    return analysis, design, contrasts


def test_make_design_is_required_and_records_centering():
    raw = pd.DataFrame({"age": [20, 30, 40, 50], "sex": [0, 1, 0, 1]})
    design = make_design(
        indicators={"sex": raw["sex"]},
        continuous={"age": raw["age"]},
    )
    assert design.columns == ("intercept", "sex", "age")
    assert design.centering == {"age": 35.0}
    assert design.frame["age"].mean() == pytest.approx(0)
    assert design.metadata()["n_observations"] == 4
    with pytest.raises(TypeError, match="make_design"):
        conlens.DesignMatrix(raw, {}, 1.0)
    with pytest.raises(ValueError, match="full column rank"):
        make_design(continuous={"a": [1, 2, 3, 4], "duplicate": [2, 4, 6, 8]})


def test_design_builder_preserves_names_and_builds_interactions_after_centering():
    age = np.array([20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    g1 = np.array([0, 0, 0, 1, 1, 1])
    design = make_design(
        indicators={"control": 1 - g1, "g1": g1},
        continuous={"age": age},
        interactions={"g1_age": ("g1", "age")},
        add_intercept=False,
    )
    assert design.columns == ("control", "g1", "age", "g1_age")
    np.testing.assert_allclose(design.frame["age"], age - age.mean())
    np.testing.assert_allclose(design.frame["g1_age"], g1 * (age - age.mean()))
    assert design.metadata()["interactions"] == {"g1_age": ["g1", "age"]}


def test_raw_matrix_is_used_without_centering_or_intercept():
    matrix = np.array([[1.0, 10.0], [1.0, 20.0], [1.0, 35.0], [1.0, 50.0]])
    design = make_design(matrix=matrix, column_names=["constant", "age"])
    np.testing.assert_array_equal(design.values, matrix)
    assert design.columns == ("constant", "age")
    assert design.centering == {}
    assert design.metadata()["input_mode"] == "matrix"
    with pytest.raises(ValueError, match="cannot be combined"):
        make_design(matrix=matrix, column_names=["a", "b"], continuous={"age": range(4)})


def test_design_and_contrast_validation_is_explicit():
    design = make_design(continuous={"age": [20, 30, 40, 50]})
    with pytest.raises(ValueError, match="unknown"):
        Contrast({"missing": 1}, "partial_r", "positive").resolve(design)
    with pytest.raises(ValueError, match="match"):
        Contrast([1], "partial_r", "positive").resolve(design)
    with pytest.raises(ValueError, match="not all zero"):
        Contrast({"age": 0}, "partial_r", "positive").resolve(design)
    with pytest.raises(ValueError, match="effect_size"):
        Contrast({"age": 1}, "invalid", "positive")
    with pytest.raises(ValueError, match="positive_direction"):
        Contrast({"age": 1}, "partial_r", "")

    with pytest.raises(ValueError, match="0/1"):
        make_design(indicators={"group": [0, 1, 2, 0]})
    with pytest.raises(ValueError, match="must not overlap"):
        make_design(indicators={"age": [0, 1, 0, 1]}, continuous={"age": range(4)})
    with pytest.raises(ValueError, match="same number"):
        make_design(indicators={"group": [0, 1]}, continuous={"age": range(4)})
    with pytest.raises(ValueError, match="unknown columns"):
        make_design(
            continuous={"age": range(5)},
            interactions={"bad": ("age", "missing")},
        )
    with pytest.raises(ValueError, match="pair"):
        make_design(continuous={"age": range(5)}, interactions={"bad": ["age"]})
    with pytest.raises(ValueError, match="overlap"):
        make_design(
            continuous={"age": range(5)},
            interactions={"age": ("age", "age")},
        )


def test_raw_matrix_validation_and_dataframe_names():
    frame = pd.DataFrame({0: [1, 1, 1, 1], 1: [2, 3, 5, 7]})
    design = make_design(matrix=frame)
    assert design.columns == ("0", "1")
    renamed = make_design(matrix=frame, column_names=["constant", "age"])
    assert renamed.columns == ("constant", "age")
    with pytest.raises(ValueError, match="two-dimensional"):
        make_design(matrix=np.arange(4), column_names=["x"])
    with pytest.raises(ValueError, match="required"):
        make_design(matrix=np.ones((4, 1)))
    with pytest.raises(ValueError, match="number of matrix columns"):
        make_design(matrix=np.ones((4, 2)), column_names=["x"])
    with pytest.raises(ValueError, match="does not add"):
        make_design(matrix=np.ones((4, 1)), column_names=["x"], add_intercept=True)
    with pytest.raises(ValueError, match="only used with matrix"):
        make_design(continuous={"age": range(4)}, column_names=["age"])
    with pytest.raises(ValueError, match="provide"):
        make_design()


def test_all_documented_design_patterns_are_full_rank_and_resolve_contrasts():
    rng = np.random.default_rng(91)
    n = 36
    diagnosis = np.repeat(["control", "g1", "g2"], n // 3)
    age = rng.normal(45, 11, n)
    motion = rng.uniform(0.05, 0.25, n)
    male = rng.integers(0, 2, n)
    site_b = np.tile([0, 1], n // 2)
    two_group = diagnosis != "g2"

    continuous_contrast = Contrast({"age": 1}, "partial_r", "increases with age")
    g1_contrast = Contrast({"g1": 1, "control": -1}, "hedges_g", "g1 > control")
    g2_contrast = Contrast({"g2": 1, "control": -1}, "hedges_g", "g2 > control")

    designs_and_contrasts = [
        (make_design(continuous={"age": age}), [continuous_contrast]),
        (
            make_design(
                indicators={"male": male, "site_B": site_b},
                continuous={"age": age, "motion": motion},
            ),
            [continuous_contrast],
        ),
        (
            make_design(
                indicators={
                    "control": diagnosis[two_group] == "control",
                    "g1": diagnosis[two_group] == "g1",
                },
                add_intercept=False,
            ),
            [g1_contrast],
        ),
        (
            make_design(
                indicators={
                    "control": diagnosis[two_group] == "control",
                    "g1": diagnosis[two_group] == "g1",
                    "male": male[two_group],
                    "site_B": site_b[two_group],
                },
                continuous={"age": age[two_group], "motion": motion[two_group]},
                add_intercept=False,
            ),
            [g1_contrast],
        ),
        (
            make_design(
                indicators={
                    "control": diagnosis == "control",
                    "g1": diagnosis == "g1",
                    "g2": diagnosis == "g2",
                },
                add_intercept=False,
            ),
            [g1_contrast, g2_contrast],
        ),
        (
            make_design(
                indicators={
                    "control": diagnosis == "control",
                    "g1": diagnosis == "g1",
                    "g2": diagnosis == "g2",
                    "male": male,
                    "site_B": site_b,
                },
                continuous={"age": age, "motion": motion},
                add_intercept=False,
            ),
            [g1_contrast, g2_contrast],
        ),
    ]
    for design, contrast_specs in designs_and_contrasts:
        assert np.linalg.matrix_rank(design.values) == design.n_columns
        for contrast in contrast_specs:
            assert contrast.resolve(design).shape == (design.n_columns,)


def test_effect_sizes_match_frozen_definitions():
    rng = np.random.default_rng(2)
    group = np.repeat([0, 1], 10)
    age = np.linspace(-1, 1, 20)
    x = np.column_stack([1 - group, group, age])
    y = rng.normal(size=(20, 3)) + group[:, None] * np.array([0.8, -0.2, 0.4])
    c = np.array([-1.0, 1.0, 0.0])
    output = glm_contrast_statistics(y, x, c, effect_size="hedges_g")
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    residual_df = len(x) - x.shape[1]
    full_model_sd = np.sqrt(np.sum(residual**2, axis=0) / residual_df)
    correction = 1 - 3 / (4 * residual_df - 1)
    np.testing.assert_allclose(output.effect_size, correction * (c @ beta) / full_model_sd)
    np.testing.assert_allclose(output.residual_sd, full_model_sd)

    target = np.linspace(-2, 2, 20)
    simple_x = np.column_stack([np.ones(20), target])
    partial = glm_contrast_statistics(
        y,
        simple_x,
        np.array([0.0, 1.0]),
        effect_size="partial_r",
    )
    expected = np.array([np.corrcoef(target, y[:, edge])[0, 1] for edge in range(y.shape[1])])
    np.testing.assert_allclose(partial.effect_size, expected)


def test_hedges_contrast_requires_mean_difference_normalization():
    design = make_design(
        indicators={"control": [1, 1, 0, 0], "case": [0, 0, 1, 1]},
        add_intercept=False,
    )
    valid = Contrast(
        {"case": 1, "control": -1},
        effect_size="hedges_g",
        positive_direction="case > control",
    )
    np.testing.assert_array_equal(valid.resolve(design), [-1, 1])
    invalid = Contrast(
        {"case": 2, "control": -1},
        effect_size="hedges_g",
        positive_direction="case > control",
    )
    with pytest.raises(ValueError, match="sum to 1"):
        invalid.resolve(design)


def test_multi_contrast_glm_uses_joint_bh_and_retains_edge_statistics(tmp_path):
    analysis, design, contrasts = _subject_analysis()
    result = analysis.glm(
        design,
        contrasts,
        n_permutations=12,
        random_state=8,
        min_size=1,
        correction_family_id="primary",
    )
    assert isinstance(result, GLMResult)
    assert result.contrast_names == ("g1_vs_control", "g2_vs_control", "age")
    pvalues = result.to_frame()["p_value"].to_numpy(float)
    np.testing.assert_allclose(result.to_frame()["q_value"], adjust_pvalues(pvalues))
    for lens_result in result.contrasts.values():
        assert lens_result.metadata["n_sets_tested"] == 6
        assert lens_result.metadata["permutation_scheme"] == "contrast_specific_freedman_lane"
        assert lens_result.metadata["correction_family_id"] == "primary"
        assert {
            "effect_size",
            "contrast_estimate",
            "standard_error",
            "t_statistic",
            "residual_df",
            "edge_p_value_two_sided",
            "residual_sd",
        }.issubset(lens_result.ranked_edges.columns)

    restored = GLMResult.load(result.save(tmp_path / "glm.json"))
    assert restored.contrast_names == result.contrast_names
    pd.testing.assert_frame_equal(restored.to_frame(), result.to_frame(), check_dtype=False)


def test_glm_reproducible_and_rejects_unvalidated_design():
    analysis, design, contrasts = _subject_analysis()
    first = analysis.glm(design, contrasts, n_permutations=8, random_state=3, min_size=1)
    second = analysis.glm(design, contrasts, n_permutations=8, random_state=3, min_size=1)
    pd.testing.assert_frame_equal(first.to_frame(), second.to_frame())
    with pytest.raises(TypeError, match="make_design"):
        analysis.glm(design.values, contrasts, n_permutations=None, min_size=1)
    assert not hasattr(analysis, "two_group")
    assert not hasattr(analysis, "phenotype")
    assert not hasattr(conlens, "label_permutation_null")


def test_plot_design_shows_named_matrix_and_contrasts():
    _, design, contrasts = _subject_analysis()
    axes = plot_design(design, contrasts)
    assert axes.shape == (2,)
    assert axes[0].get_title().startswith("Design matrix")
    assert [tick.get_text() for tick in axes[1].get_yticklabels()] == list(contrasts)
    plt.close(axes[0].figure)
