import numpy as np
import pandas as pd
import pytest

from conlens import Contrast, lens_fl_permute, lens_glm, make_design, plot_design


def connectomes(seed=1, n=24, nodes=5):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, nodes, nodes))
    values = (raw + raw.transpose(0, 2, 1)) / 2
    for matrix in values:
        np.fill_diagonal(matrix, 0)
    return values


def test_design_semantics_centering_interactions_and_raw_mode():
    group = np.array([0, 0, 1, 1, 0, 1], dtype=bool)
    age = np.arange(6.0)
    design = make_design(
        groups={"control": ~group, "g1": group},
        continuous={"age": age},
        interactions={"g1_x_age": ("g1", "age")},
    )
    assert design.columns == ("control", "g1", "age", "g1_x_age")
    assert design.frame["age"].mean() == pytest.approx(0)
    assert design.metadata()["intercept_added"] is False
    raw = make_design(matrix=pd.DataFrame({"constant": np.ones(6), "age": age}))
    assert raw.frame["age"].tolist() == age.tolist()
    assert raw.metadata()["intercept_added"] is False
    with pytest.raises(ValueError, match="exactly one group"):
        make_design(groups={"a": [1, 0, 0, 0], "b": [0, 1, 0, 0]})
    with pytest.raises(ValueError, match="full column rank"):
        make_design(matrix=np.ones((5, 2)), column_names=["a", "b"])


def test_glm_effect_sizes_use_full_model_residuals_and_fl_is_reproducible():
    n = 30
    values = connectomes(n=n)
    group = np.r_[np.zeros(n // 3), np.ones(n // 3), np.full(n // 3, 2)]
    age = np.linspace(20, 70, n)
    values[group == 1, 0, 1] += 1.0
    values[group == 2, 0, 1] += 0.4
    values[:, 1, 0] = values[:, 0, 1]
    design = make_design(
        groups={"control": group == 0, "g1": group == 1, "g2": group == 2},
        continuous={"age": age},
    )
    contrasts = {
        "g1_vs_control": Contrast(
            {"g1": 1, "control": -1}, "hedges_g", "g1 > control"
        ),
        "g2_vs_control": Contrast(
            {"g2": 1, "control": -1}, "hedges_g", "g2 > control"
        ),
        "age": Contrast({"age": 1}, "partial_r", "increases with age"),
    }
    fitted = lens_glm(values, design=design, contrasts=contrasts)
    assert set(fitted) == set(contrasts)
    edge = fitted["g1_vs_control"].table.iloc[0]
    correction = 1 - 3 / (4 * edge.residual_df - 1)
    assert edge.effect_size == pytest.approx(
        correction * edge.contrast_estimate / edge.residual_sd
    )
    age_edge = fitted["age"].table.iloc[0]
    expected_r = age_edge.t_statistic / np.sqrt(
        age_edge.t_statistic**2 + age_edge.residual_df
    )
    assert age_edge.effect_size == pytest.approx(expected_r)
    first = list(lens_fl_permute(
        values, design=design, contrasts=contrasts, n_permutations=3, random_state=9
    ))
    second = list(lens_fl_permute(
        values, design=design, contrasts=contrasts, n_permutations=3, random_state=9
    ))
    for left, right in zip(first, second, strict=True):
        for name in contrasts:
            np.testing.assert_allclose(left[name].table.statistic, right[name].table.statistic)
            assert list(left[name].table) == [
                "node1", "node2", "edge_id", "canonical_edge_id", "statistic"
            ]


def test_contrasts_and_design_plot_validation():
    design = make_design(continuous={"age": np.arange(8.0)})
    contrast = Contrast({"age": 1}, "partial_r", "older > younger")
    assert contrast.resolve(design).tolist() == [0, 1]
    axes = plot_design(design, {"age": contrast})
    assert len(axes) == 2
    with pytest.raises(ValueError, match="mean-difference"):
        Contrast({"age": 1}, "hedges_g", "positive").resolve(design)
    with pytest.raises(ValueError, match="unknown"):
        Contrast({"missing": 1}, "partial_r", "positive").resolve(design)
    with pytest.raises(TypeError):
        lens_glm(connectomes(n=8), design=np.ones((8, 1)), contrasts={"x": contrast})


def test_block_validation():
    values = connectomes(n=10)
    design = make_design(continuous={"age": np.arange(10.0)})
    contrasts = {"age": Contrast({"age": 1}, "partial_r", "positive")}
    with pytest.raises(ValueError, match="missing"):
        list(lens_fl_permute(
            values, design=design, contrasts=contrasts, n_permutations=1,
            exchangeability_blocks=[("site", i) if i else ("site", None) for i in range(10)],
        ))
