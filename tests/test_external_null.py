from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from conlens import (
    EdgeStatistics,
    NullEdgeStatistics,
    lens_edge_permute,
    lens_enrich,
    lens_stat,
    make_edge_statistics,
    make_null_edge_statistics,
)


def _reference(example_edges) -> EdgeStatistics:
    return make_edge_statistics(
        example_edges,
        positive_direction="higher values",
        statistic_name="external score",
    )


def _edge_permutation_matrix(
    reference: EdgeStatistics,
    *,
    n_permutations: int,
    random_state: int,
) -> tuple[np.ndarray, list[EdgeStatistics]]:
    generated = list(
        lens_edge_permute(
            reference,
            n_permutations=n_permutations,
            random_state=random_state,
        )
    )
    columns = []
    for item in generated:
        numeric = item._numeric_parts()
        assert numeric is not None
        columns.append(numeric[1])
    return np.column_stack(columns), generated


def test_external_null_matrix_is_reiterable_and_uses_column_views(example_edges):
    reference = _reference(example_edges)
    matrix = np.arange(30, dtype=float).reshape(6, 5)
    null_edges = make_null_edge_statistics(
        matrix,
        reference=reference,
        permutation_scheme="external_sign_flip",
        random_state=23,
        exchangeability_blocks_used=True,
    )

    assert isinstance(null_edges, NullEdgeStatistics)
    assert null_edges.shape == (6, 5)
    assert null_edges.n_edges == 6
    assert null_edges.n_permutations == 5
    assert len(null_edges) == 5

    first_pass = list(null_edges)
    second_pass = list(null_edges)
    assert len(first_pass) == len(second_pass) == 5
    for index, (first, second) in enumerate(zip(first_pass, second_pass, strict=True)):
        first_numeric = first._numeric_parts()
        second_numeric = second._numeric_parts()
        assert first_numeric is not None
        assert second_numeric is not None
        np.testing.assert_array_equal(first_numeric[1], matrix[:, index])
        np.testing.assert_array_equal(second_numeric[1], matrix[:, index])
        assert np.shares_memory(first_numeric[1], matrix)
        assert np.shares_memory(second_numeric[1], matrix)
        assert first.metadata["permutation_index"] == index
        assert first.metadata["permutation_scheme"] == "external_sign_flip"
        assert first.metadata["random_seed"] == 23
        assert first.metadata["exchangeability_blocks_used"] is True
        assert first.metadata["positive_direction"] == "higher values"
        assert first.metadata["analysis_signature"] == reference.metadata["analysis_signature"]


def test_external_null_path_exactly_matches_current_on_the_fly_path(
    example_edges, example_sets
):
    reference = _reference(example_edges)
    observed = lens_stat(reference, example_sets, store_running_sum=True)
    matrix, generated = _edge_permutation_matrix(
        reference,
        n_permutations=31,
        random_state=17,
    )
    external = make_null_edge_statistics(
        matrix,
        reference=reference,
        permutation_scheme="edge_label_permutation",
        random_state=17,
    )

    current = lens_enrich(
        observed,
        (lens_stat(item, example_sets) for item in generated),
        min_size=1,
        family_name="external-equivalence",
    )
    from_matrix = lens_enrich(
        observed,
        (lens_stat(item, example_sets) for item in external),
        min_size=1,
        family_name="external-equivalence",
    )

    assert [asdict(item) for item in from_matrix.sets] == [
        asdict(item) for item in current.sets
    ]
    assert from_matrix.metadata == current.metadata
    pd.testing.assert_frame_equal(
        from_matrix.null_scores,
        current.null_scores,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        from_matrix.ranked_edges,
        current.ranked_edges,
        check_exact=True,
    )


def test_external_null_dataframe_matches_edges_by_identity(example_edges):
    reference = _reference(example_edges)
    edge_ids = reference.table["edge_id"].tolist()
    ordered = np.arange(18, dtype=float).reshape(6, 3)
    shuffled = pd.DataFrame(ordered, index=edge_ids, columns=["p0", "p1", "p2"]).iloc[
        [3, 0, 5, 1, 4, 2]
    ]

    null_edges = make_null_edge_statistics(
        shuffled,
        reference=reference,
        permutation_scheme="external",
    )

    for index, item in enumerate(null_edges):
        numeric = item._numeric_parts()
        assert numeric is not None
        np.testing.assert_array_equal(numeric[1], ordered[:, index])


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.ones(6), "shape"),
        (np.ones((5, 2)), "reference edge universe"),
        (np.ones((6, 0)), "at least one permutation"),
        (np.full((6, 2), np.nan), "finite"),
        (np.array([["not numeric"]] * 6, dtype=object), "numeric"),
    ],
)
def test_external_null_matrix_rejects_invalid_shape_or_values(
    example_edges, values, message
):
    with pytest.raises(ValueError, match=message):
        make_null_edge_statistics(
            values,
            reference=_reference(example_edges),
            permutation_scheme="external",
        )


def test_external_null_dataframe_rejects_incompatible_edge_identity(example_edges):
    reference = _reference(example_edges)
    edge_ids = reference.table["edge_id"].tolist()
    frame = pd.DataFrame(np.ones((6, 2)), index=edge_ids)

    with pytest.raises(ValueError, match="unique edge IDs"):
        make_null_edge_statistics(
            frame.set_axis([*edge_ids[:-1], edge_ids[0]]),
            reference=reference,
            permutation_scheme="external",
        )
    with pytest.raises(ValueError, match="incompatible edge universe"):
        make_null_edge_statistics(
            frame.rename(index={edge_ids[-1]: "unknown--edge"}),
            reference=reference,
            permutation_scheme="external",
        )
    with pytest.raises(ValueError, match="columns must be unique"):
        make_null_edge_statistics(
            frame.set_axis(["same", "same"], axis="columns"),
            reference=reference,
            permutation_scheme="external",
        )


def test_external_null_validates_reference_and_metadata(example_edges):
    reference = _reference(example_edges)
    values = np.ones((6, 2))

    with pytest.raises(TypeError, match="reference"):
        make_null_edge_statistics(
            values,
            reference=example_edges,
            permutation_scheme="external",
        )
    with pytest.raises(ValueError, match="permutation_scheme"):
        make_null_edge_statistics(values, reference=reference, permutation_scheme="  ")
    with pytest.raises(TypeError, match="random_state"):
        make_null_edge_statistics(
            values,
            reference=reference,
            permutation_scheme="external",
            random_state="42",
        )
    with pytest.raises(TypeError, match="exchangeability_blocks_used"):
        make_null_edge_statistics(
            values,
            reference=reference,
            permutation_scheme="external",
            exchangeability_blocks_used=1,
        )
    with pytest.raises(ValueError, match="positive_direction"):
        make_null_edge_statistics(
            values,
            reference=EdgeStatistics(reference.table.copy(), {}),
            permutation_scheme="external",
        )
