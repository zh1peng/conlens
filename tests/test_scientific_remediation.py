"""Independent numerical and discrete-null regression checks."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import solve_triangular
from scipy.stats import t

from conlens import (
    Contrast,
    LensStabilityResult,
    compute_enrichment_score,
    compute_running_sum,
    lens_bootstrap,
    lens_enrich,
    lens_fl_permute,
    lens_glm,
    lens_stat,
    make_design,
)
from conlens.enrichment import _apply_null
from conlens.results import LensSetResult
from conlens.stats import glm_contrast_statistics


def test_large_offset_design_against_independent_qr_and_units():
    rng = np.random.default_rng(206)
    x, u, noise = rng.normal(size=(3, 120))
    v = 1_500_000 + 30_000 * u
    y = (0.4 * x + 0.5 * (v - v.mean()) / 30_000 + noise)[:, None]
    design = np.column_stack([np.ones(120), x, v])
    q, r = np.linalg.qr(design)
    beta = solve_triangular(r, q.T @ y)
    residual_sd = np.linalg.norm(y - design @ beta, axis=0) / np.sqrt(117)
    se = residual_sd * np.linalg.norm(solve_triangular(r.T, [0, 0, 1], lower=True))
    expected_t = beta[2] / se
    assert expected_t[0] == pytest.approx(7.01717, abs=5e-6)
    for predictor in (v, v / 1e6, v - v.mean(), (v - v.mean()) / 30_000):
        for response_scale in (1, 1e-20, 1e20):
            result = glm_contrast_statistics(
                y * response_scale, np.column_stack([np.ones(120), x, predictor]),
                [0, 0, 1], effect_size="partial_r",
            )
            assert result.estimable.all()
            np.testing.assert_allclose(result.t_statistic, expected_t, rtol=1e-8)
            np.testing.assert_allclose(
                result.p_value_two_sided, 2 * t.sf(expected_t, 117), rtol=1e-7,
            )
            np.testing.assert_allclose(
                result.effect_size, expected_t / np.sqrt(expected_t**2 + 117), rtol=1e-8,
            )


@pytest.mark.parametrize("effect_size", ["partial_r", "hedges_g"])
def test_group_and_interaction_against_qr(effect_size):
    rng = np.random.default_rng(83)
    group = np.repeat([0, 1], 40)
    age = rng.normal(size=80)
    x = np.column_stack([1 - group, group, age, group * age])
    y = rng.normal(size=(80, 4)) + (group + 0.3 * group * age)[:, None]
    c = np.array([-1, 1, 0, 0]) if effect_size == "hedges_g" else np.array([0, 0, 0, 1])
    q, r = np.linalg.qr(x)
    beta = solve_triangular(r, q.T @ y)
    sd = np.linalg.norm(y - x @ beta, axis=0) / np.sqrt(76)
    expected_t = (c @ beta) / (sd * np.linalg.norm(solve_triangular(r.T, c, lower=True)))
    result = glm_contrast_statistics(y, x, c, effect_size=effect_size)
    np.testing.assert_allclose(result.t_statistic, expected_t, rtol=1e-12)
    expected = ((1 - 3 / 303) * (c @ beta) / sd if effect_size == "hedges_g"
                else expected_t / np.sqrt(expected_t**2 + 76))
    np.testing.assert_allclose(result.effect_size, expected, rtol=1e-12)


def score_item(es):
    return LensSetResult("target", 7, 7, es, max(es, 0), min(es, 0))


def test_exact_14_choose_7_null():
    scores = []
    for members in combinations(range(14), 7):
        hits = np.zeros(14, dtype=bool)
        hits[list(members)] = True
        running, _ = compute_running_sum(np.arange(14, 0, -1), hits, weight=0)
        es = compute_enrichment_score(running)["ES"]
        # Independent integer random walk: no floating-point tie ambiguity.
        walk = np.r_[0, np.cumsum(np.where(hits, 1, -1))]
        expected = (walk.max() if walk.max() > -walk.min() else walk.min()) / 7
        if walk.max() == -walk.min():
            expected = 0
        assert es == pytest.approx(expected, abs=1e-14)
        scores.append(es)
    null = np.asarray(scores)
    assert [sum(null > 0), sum(null < 0), sum(null == 0)] == [1458, 1458, 516]
    # Use actual numerical values at the boundary to avoid inventing an epsilon tail rule.
    positive = np.unique(null[np.isclose(null, 5 / 7)])
    for es in positive:
        item = score_item(es)
        _apply_null(item, null)
        assert item.n_null_zero == 516
        assert item.n_null_tail == 1458
        assert item.p_value == pytest.approx(92 / 1459)
        assert item.NES == pytest.approx(es / null[null > 0].mean())
    rejected = 0
    for es in null:
        item = score_item(es)
        _apply_null(item, null)
        rejected += item.p_value <= 0.05
    assert rejected / len(null) <= 0.05


@pytest.mark.parametrize("mode,es,expected", [
    ("standard", 0.5, 2 / 3), ("standard", -0.5, 2 / 3),
    ("positive", 0.5, 2 / 7), ("negative", -0.5, 2 / 7),
    ("positive", 0, 1), ("negative", 0, 1), ("standard", 0, 1),
])
def test_score_type_zero_rules(mode, es, expected):
    null = np.array([0, 0, 0.25, 0.75, -0.25, -0.75])
    if mode == "positive":
        null = np.maximum(null, 0)
    elif mode == "negative":
        null = np.minimum(null, 0)
    item = score_item(es)
    _apply_null(item, null, mode)
    assert item.p_value == pytest.approx(expected)
    if mode == "standard" and es == 0:
        assert item.NES is None and item.n_null_tail == 0
    else:
        assert item.n_null_tail == (2 if mode == "standard" else 6)
        assert item.minimum_resolvable_p == pytest.approx(1 / (item.n_null_tail + 1))


def test_zero_null_and_missing_tail():
    for mode in ("standard", "positive", "negative"):
        item = score_item(0)
        _apply_null(item, np.zeros(5), mode)
        assert item.p_value == 1 and item.NES is None
    item = score_item(0.5)
    _apply_null(item, np.array([-0.5, 0]))
    assert item.p_value == 1 and item.NES is None and item.n_null_tail == 0


def test_nonzero_null_preserves_previous_directional_formula():
    null = np.array([-0.7, -0.2, 0.1, 0.4])
    for es in (-0.3, 0.3):
        item = score_item(es)
        _apply_null(item, null)
        assert item.p_value == pytest.approx(2 / 3)
        tail = null[null >= 0] if es > 0 else null[null <= 0]
        assert item.NES == pytest.approx(es / np.abs(tail).mean())
        assert item.n_null_zero == 0


@pytest.mark.parametrize("mode", ["standard", "positive", "negative"])
def test_public_inference_uses_score_mode(example_edges, mode):
    from conlens import lens_edge_permute, make_edge_statistics

    edges = make_edge_statistics(example_edges, positive_direction="higher")
    sets = {"target": edges.table["edge_id"].tolist()[:3]}
    observed = lens_stat(edges, sets, weight=0, score_type=mode)
    null_edges = lens_edge_permute(edges, n_permutations=31, random_state=2)
    null = [lens_stat(item, sets, weight=0, score_type=mode) for item in null_edges]
    result = lens_enrich(observed, null, min_size=1)
    item = result.get("target")
    expected = score_item(item.ES)
    _apply_null(expected, result.null_for("target").to_numpy(), mode)
    assert item.n_null_tail == expected.n_null_tail
    assert item.NES == expected.NES
    assert item.p_value == expected.p_value


def subject_fixture():
    rng = np.random.default_rng(18)
    raw = rng.normal(size=(24, 4, 4))
    values = raw + raw.transpose(0, 2, 1)
    design = make_design(continuous={"age": np.arange(24.0)})
    contrasts = {"age": Contrast({"age": 1}, "partial_r", "increases with age")}
    return values, design, contrasts


@pytest.mark.parametrize("kind", ["constant", "zero", "perfect_fit"])
def test_degenerate_measurements_are_auditable_but_cannot_be_ranked(kind):
    values, design, contrasts = subject_fixture()
    values[:, 0, 1] = np.arange(24.0) if kind == "perfect_fit" else (0 if kind == "zero" else 2)
    values[:, 1, 0] = values[:, 0, 1]
    observed = lens_glm(values, design=design, contrasts=contrasts)["age"]
    assert observed.metadata["nonestimable_edge_ids"] == ["0--1"]
    for edges in (observed, observed.table):
        with pytest.raises(ValueError, match="nonestimable"):
            lens_stat(edges, {"a": ["0--1", "0--2"]}, positive_direction="increases with age")


def test_nonestimable_null_and_singleton_blocks():
    values, design, contrasts = subject_fixture()
    options = dict(design=design, contrasts=contrasts, n_permutations=1)
    with pytest.raises(ValueError, match="singletons"):
        next(lens_fl_permute(values, **options, exchangeability_blocks=range(24)))
    values[:, 0, 1] = values[:, 1, 0] = 0
    null = next(lens_fl_permute(values, **options))
    with pytest.raises(ValueError, match="nonestimable"):
        lens_stat(null, {"a": ["0--1", "0--2"]})


def test_many_structural_zeros_are_not_neutral_background():
    values, design, contrasts = subject_fixture()
    values[:, 0, 1:] = 0
    values[:, 1:, 0] = 0
    observed = lens_glm(values, design=design, contrasts=contrasts)
    assert observed["age"].metadata["n_nonestimable_edges"] == 3
    with pytest.raises(ValueError, match="nonestimable"):
        lens_stat(observed, {"a": ["1--2", "1--3"]})


def test_observed_null_connection_data_must_match():
    values, design, contrasts = subject_fixture()
    sets = {"a": ["0--1", "0--2"]}
    observed = lens_stat(lens_glm(values, design=design, contrasts=contrasts), sets)
    # Counts, graph and design match; the data row order does not.
    null = lens_fl_permute(values[::-1], design=design, contrasts=contrasts, n_permutations=1)
    with pytest.raises(ValueError, match="connectome_data_hash"):
        lens_enrich(observed, (lens_stat(item, sets) for item in null), min_size=1)


def test_blocks_cannot_leave_tested_contrast_invariant():
    values, _, _ = subject_fixture()
    site = np.repeat([0, 1], 12)
    design = make_design(groups={"a": site == 0, "b": site == 1})
    contrasts = {"site": Contrast({"b": 1, "a": -1}, "hedges_g", "b > a")}
    with pytest.raises(ValueError, match="invariant"):
        next(lens_fl_permute(
            values, design=design, contrasts=contrasts,
            n_permutations=2, exchangeability_blocks=site,
        ))


def test_bootstrap_reference_roundtrip_and_seed_replay(tmp_path):
    values, design, contrasts = subject_fixture()
    labels = ["A", "B", "C", "D"]
    sets = {"a": ["0--1", "0--2"]}
    result = lens_bootstrap(
        values, sets, design=design, contrasts=contrasts, node_labels=labels,
        n_bootstraps=2, n_permutations=9, min_size=1, random_state=71,
    )["age"]
    path = result.save(tmp_path / "stability.json")
    loaded = LensStabilityResult.load(path)
    assert loaded.observed_reference is not None
    seed = loaded.metadata["observed_permutation_seed"]
    observed = lens_stat(
        lens_glm(values, design=design, contrasts=contrasts, node_labels=labels), sets,
    )
    null = lens_fl_permute(
        values, design=design, contrasts=contrasts, node_labels=labels,
        n_permutations=9, random_state=seed,
    )
    replay = lens_enrich(observed, (lens_stat(item, sets) for item in null), min_size=1)["age"]
    pd.testing.assert_frame_equal(loaded.observed_reference.to_frame(), replay.to_frame())
    assert len(loaded.metadata["bootstrap_draw_indices"]) == 2
    assert len(loaded.metadata["bootstrap_permutation_seeds"]) == 2
    legacy = loaded.to_dict()
    legacy.pop("observed_reference")
    assert LensStabilityResult.from_dict(legacy).observed_reference is None
