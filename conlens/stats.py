"""Vectorized edge-wise GLM statistics for subject-level connectomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

from .design import EffectSize


@dataclass(frozen=True, slots=True)
class GLMEdgeStatistics:
    """Complete one-degree-of-freedom GLM contrast statistics for every edge."""

    effect_size: np.ndarray
    contrast_estimate: np.ndarray
    standard_error: np.ndarray
    t_statistic: np.ndarray
    residual_df: int
    p_value_two_sided: np.ndarray
    residual_sd: np.ndarray
    estimable: np.ndarray


@dataclass(frozen=True, slots=True)
class _PreparedGLMDesign:
    pseudoinverse: np.ndarray
    xtx_inverse: np.ndarray
    residual_df: int


def _prepare_glm_design(design: np.ndarray) -> _PreparedGLMDesign:
    x = np.asarray(design, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("design must be a finite two-dimensional array")
    rank = int(np.linalg.matrix_rank(x))
    residual_df = len(x) - rank
    if rank < x.shape[1] or residual_df <= 0:
        raise ValueError("design matrix must have full column rank and positive residual df")
    return _PreparedGLMDesign(
        pseudoinverse=np.linalg.pinv(x),
        xtx_inverse=np.linalg.pinv(x.T @ x, hermitian=True),
        residual_df=residual_df,
    )


def glm_contrast_statistics(
    data: np.ndarray,
    design: np.ndarray,
    contrast: np.ndarray,
    *,
    effect_size: EffectSize,
    _prepared: _PreparedGLMDesign | None = None,
) -> GLMEdgeStatistics:
    """Fit one full OLS model and calculate a signed standardized effect.

    ``partial_r`` is the partial correlation implied by the contrast t statistic.
    ``hedges_g`` is ``J * (c @ beta) / residual_sd``, where residual SD and
    residual degrees of freedom come from the full model.
    """
    y = np.asarray(data, dtype=float)
    x = np.asarray(design, dtype=float)
    c = np.asarray(contrast, dtype=float).reshape(-1)
    if y.ndim != 2 or x.ndim != 2 or len(y) != len(x):
        raise ValueError("data and design must be 2D with matching observation counts")
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("data and design must contain only finite values")
    if len(c) != x.shape[1] or not np.isfinite(c).all() or np.allclose(c, 0.0):
        raise ValueError("contrast must be finite, nonzero, and match the design columns")
    if effect_size not in {"partial_r", "hedges_g"}:
        raise ValueError("effect_size must be 'partial_r' or 'hedges_g'")

    prepared = _prepare_glm_design(x) if _prepared is None else _prepared
    residual_df = prepared.residual_df
    beta = prepared.pseudoinverse @ y
    residual = y - x @ beta
    residual_variance = np.sum(residual**2, axis=0) / residual_df
    residual_sd = np.sqrt(residual_variance)
    contrast_scale = float(c @ prepared.xtx_inverse @ c)
    estimates = c @ beta
    standard_error = residual_sd * np.sqrt(contrast_scale)
    estimable = np.isfinite(standard_error) & (standard_error > np.finfo(float).eps)
    t_statistic = np.divide(
        estimates,
        standard_error,
        out=np.zeros_like(estimates, dtype=float),
        where=estimable,
    )

    p_value = 2 * scipy_stats.t.sf(np.abs(t_statistic), residual_df)
    if effect_size == "partial_r":
        effect = t_statistic / np.sqrt(t_statistic**2 + residual_df)
    else:
        correction = 1 - 3 / (4 * residual_df - 1)
        effect = np.divide(
            correction * estimates,
            residual_sd,
            out=np.zeros_like(estimates, dtype=float),
            where=estimable,
        )

    return GLMEdgeStatistics(
        effect_size=np.asarray(effect, float),
        contrast_estimate=np.asarray(estimates, float),
        standard_error=np.asarray(standard_error, float),
        t_statistic=np.asarray(t_statistic, float),
        residual_df=residual_df,
        p_value_two_sided=np.asarray(p_value, float),
        residual_sd=np.asarray(residual_sd, float),
        estimable=np.asarray(estimable, bool),
    )
