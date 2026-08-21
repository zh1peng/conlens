"""Validated GLM design matrices and named one-degree-of-freedom contrasts."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

EffectSize = Literal["partial_r", "hedges_g"]

_DESIGN_TOKEN = object()


class DesignMatrix:
    """Immutable-by-interface design created by :func:`make_design`."""

    __slots__ = ("_centering", "_condition_number", "_frame", "_provenance")

    def __init__(
        self,
        frame: pd.DataFrame,
        centering: Mapping[str, float],
        condition_number: float,
        provenance: Mapping[str, Any] | None = None,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _DESIGN_TOKEN:
            raise TypeError("DesignMatrix objects must be created with make_design()")
        self._frame = frame.copy()
        self._centering = dict(centering)
        self._condition_number = float(condition_number)
        self._provenance = dict(provenance or {})

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self._frame.columns)

    @property
    def values(self) -> np.ndarray:
        return self._frame.to_numpy(float, copy=True)

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy()

    @property
    def centering(self) -> dict[str, float]:
        return self._centering.copy()

    @property
    def condition_number(self) -> float:
        return self._condition_number

    @property
    def n_observations(self) -> int:
        return len(self._frame)

    @property
    def n_columns(self) -> int:
        return self._frame.shape[1]

    def take(self, indices: Sequence[int] | np.ndarray) -> DesignMatrix:
        """Select rows while preserving the original design specification."""
        selected = self._frame.iloc[np.asarray(indices, dtype=int)].reset_index(drop=True)
        _validate_numeric_design(selected)
        condition_number = _validate_rank(selected)
        return DesignMatrix(
            selected,
            self._centering,
            condition_number,
            self._provenance,
            _token=_DESIGN_TOKEN,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            **self._provenance,
            "columns": list(self.columns),
            "centering": self.centering,
            "condition_number": self.condition_number,
            "n_observations": self.n_observations,
            "n_columns": self.n_columns,
        }

    def signature(self) -> dict[str, Any]:
        """Return the resampling-stable specification of the design."""
        metadata = self.metadata()
        metadata.pop("condition_number")
        metadata.pop("n_observations")
        metadata.pop("n_columns")
        return metadata


@dataclass(frozen=True, slots=True)
class Contrast:
    """A named GLM contrast specification resolved against a :class:`DesignMatrix`."""

    weights: Mapping[str, float] | Sequence[float]
    effect_size: EffectSize
    positive_direction: str

    def __post_init__(self) -> None:
        if self.effect_size not in {"partial_r", "hedges_g"}:
            raise ValueError("effect_size must be 'partial_r' or 'hedges_g'")
        if not isinstance(self.positive_direction, str) or not self.positive_direction.strip():
            raise ValueError("positive_direction must be a non-empty string")

    def resolve(self, design: DesignMatrix) -> np.ndarray:
        if isinstance(self.weights, Mapping):
            unknown = set(self.weights) - set(design.columns)
            if unknown:
                raise ValueError(f"contrast contains unknown design columns: {sorted(unknown)!r}")
            values = np.asarray([self.weights.get(column, 0.0) for column in design.columns], float)
        else:
            values = np.asarray(list(self.weights), dtype=float)
            if values.ndim != 1 or len(values) != design.n_columns:
                raise ValueError("contrast weights must match the design columns")
        if not np.isfinite(values).all() or np.allclose(values, 0.0):
            raise ValueError("contrast weights must be finite and not all zero")
        if self.effect_size == "hedges_g":
            positive = float(values[values > 0].sum())
            negative = float(values[values < 0].sum())
            if not np.isclose(positive, 1.0) or not np.isclose(negative, -1.0):
                raise ValueError(
                    "hedges_g contrasts must be mean-difference normalized: positive "
                    "weights sum to 1 and negative weights sum to -1"
                )
        return values


def _validate_names(names: Iterable[str], label: str) -> list[str]:
    output = list(names)
    if any(not isinstance(name, str) or not name for name in output):
        raise ValueError(f"{label} names must be non-empty strings")
    if len(output) != len(set(output)):
        raise ValueError(f"{label} names must be unique")
    return output


def _validate_numeric_design(frame: pd.DataFrame) -> None:
    if frame.ndim != 2 or frame.shape[1] == 0:
        raise ValueError("design must contain at least one column")
    if len(frame) == 0:
        raise ValueError("design must contain at least one observation")
    if not frame.columns.is_unique:
        raise ValueError("design column names must be unique")
    _validate_names(frame.columns, "design column")
    try:
        values = frame.apply(pd.to_numeric, errors="raise").to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("design columns must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("design must contain only finite values")


def _validate_rank(frame: pd.DataFrame) -> float:
    values = frame.to_numpy(float)
    if len(values) <= values.shape[1]:
        raise ValueError("design must have positive residual degrees of freedom")
    if np.linalg.matrix_rank(values) != values.shape[1]:
        raise ValueError("design matrix must have full column rank")
    condition_number = float(np.linalg.cond(values))
    if condition_number > 1e8:
        warnings.warn(
            f"design matrix is ill-conditioned (condition number={condition_number:.3g})",
            RuntimeWarning,
            stacklevel=3,
        )
    return condition_number


def _mapping_frame(
    groups: Mapping[str, Iterable[float | bool]],
    indicators: Mapping[str, Iterable[float | bool]],
    continuous: Mapping[str, Iterable[float]],
) -> pd.DataFrame:
    group_names = _validate_names(groups, "group column")
    indicator_names = _validate_names(indicators, "indicator column")
    continuous_names = _validate_names(continuous, "continuous column")
    all_names = [*group_names, *indicator_names, *continuous_names]
    if len(all_names) != len(set(all_names)):
        raise ValueError("group, indicator, and continuous column names must not overlap")
    try:
        frame = pd.DataFrame({**groups, **indicators, **continuous}).reset_index(drop=True)
    except ValueError as exc:
        raise ValueError("all design columns must contain the same number of observations") from exc
    _validate_numeric_design(frame)
    return frame.apply(pd.to_numeric, errors="raise").astype(float)


def _raw_frame(
    matrix: np.ndarray | pd.DataFrame,
    column_names: Sequence[str] | None,
) -> pd.DataFrame:
    if isinstance(matrix, pd.DataFrame):
        frame = matrix.copy().reset_index(drop=True)
        if column_names is not None:
            names = _validate_names(column_names, "column")
            if len(names) != frame.shape[1]:
                raise ValueError("column_names must match the number of matrix columns")
            frame.columns = names
        else:
            frame.columns = [str(column) for column in frame.columns]
    else:
        values = np.asarray(matrix)
        if values.ndim != 2:
            raise ValueError("matrix must be two-dimensional")
        if column_names is None:
            raise ValueError("column_names are required for a NumPy design matrix")
        names = _validate_names(column_names, "column")
        if len(names) != values.shape[1]:
            raise ValueError("column_names must match the number of matrix columns")
        frame = pd.DataFrame(values, columns=names)
    _validate_numeric_design(frame)
    return frame.apply(pd.to_numeric, errors="raise").astype(float)


def make_design(
    *,
    groups: Mapping[str, Iterable[float | bool]] | None = None,
    indicators: Mapping[str, Iterable[float | bool]] | None = None,
    continuous: Mapping[str, Iterable[float]] | None = None,
    interactions: Mapping[str, tuple[str, str]] | None = None,
    matrix: np.ndarray | pd.DataFrame | None = None,
    column_names: Sequence[str] | None = None,
    center_continuous: bool = True,
) -> DesignMatrix:
    """Build a validated design without hidden recoding or orthogonalization.

    ``groups`` defines exhaustive cell-means columns: every row must belong to
    exactly one named group, and no separate intercept is added. Without
    ``groups``, semantic mode adds an intercept automatically. Continuous
    columns are mean-centered by default; indicator columns are never centered.
    Named interactions are constructed only after centering.

    Raw-matrix mode uses ``matrix`` exactly as supplied: it never centers columns,
    creates interactions, or adds an intercept.
    """
    group_columns = {} if groups is None else dict(groups)
    indicator_columns = {} if indicators is None else dict(indicators)
    continuous_columns = {} if continuous is None else dict(continuous)
    interaction_specs = {} if interactions is None else dict(interactions)
    semantic_requested = bool(
        group_columns or indicator_columns or continuous_columns or interaction_specs
    )

    if matrix is not None:
        if semantic_requested:
            raise ValueError(
                "matrix mode cannot be combined with groups, indicators, continuous, "
                "or interactions"
            )
        frame = _raw_frame(matrix, column_names)
        centering: dict[str, float] = {}
        provenance: dict[str, Any] = {
            "input_mode": "matrix",
            "group_columns": [],
            "indicator_columns": [],
            "continuous_columns": [],
            "center_continuous": False,
            "interactions": {},
            "intercept_added": False,
        }
    else:
        if column_names is not None:
            raise ValueError("column_names are only used with matrix")
        if not group_columns and not indicator_columns and not continuous_columns:
            raise ValueError("provide groups/indicators/continuous or a raw matrix")
        if not isinstance(center_continuous, bool):
            raise TypeError("center_continuous must be boolean")
        frame = _mapping_frame(group_columns, indicator_columns, continuous_columns)
        for name in [*group_columns, *indicator_columns]:
            values = frame[name].to_numpy(float)
            if not np.isin(values, [0.0, 1.0]).all():
                raise ValueError(f"binary column {name!r} must contain only 0/1 values")
        if group_columns:
            membership = frame[list(group_columns)].sum(axis=1).to_numpy(float)
            if not np.allclose(membership, 1.0):
                raise ValueError("every row must belong to exactly one group")

        centering = {}
        if center_continuous:
            for name in continuous_columns:
                value = float(frame[name].mean())
                frame[name] = frame[name] - value
                centering[name] = value

        base_names = set(frame.columns)
        interaction_names = _validate_names(interaction_specs, "interaction")
        overlap = set(interaction_names) & base_names
        if overlap:
            raise ValueError(f"interaction names overlap design columns: {sorted(overlap)!r}")
        normalized_interactions: dict[str, list[str]] = {}
        for name, sources in interaction_specs.items():
            if not isinstance(sources, tuple) or len(sources) != 2:
                raise ValueError("each interaction must be a pair of source column names")
            first, second = sources
            missing = {first, second} - base_names
            if missing:
                raise ValueError(
                    f"interaction {name!r} contains unknown columns: {sorted(missing)!r}"
                )
            frame[name] = frame[first] * frame[second]
            normalized_interactions[name] = [first, second]

        use_intercept = not group_columns
        if use_intercept:
            if "intercept" in frame.columns:
                raise ValueError("design already contains an 'intercept' column")
            frame.insert(0, "intercept", 1.0)
        provenance = {
            "input_mode": "semantic",
            "group_columns": list(group_columns),
            "indicator_columns": list(indicator_columns),
            "continuous_columns": list(continuous_columns),
            "center_continuous": center_continuous,
            "interactions": normalized_interactions,
            "intercept_added": use_intercept,
        }

    _validate_numeric_design(frame)
    condition_number = _validate_rank(frame)
    return DesignMatrix(
        frame,
        centering,
        condition_number,
        provenance,
        _token=_DESIGN_TOKEN,
    )
