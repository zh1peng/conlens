"""Vectorized edge-wise statistics for subject-level connectomes."""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats


def two_group_ttest(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Welch signed t statistic (group 1 minus group 0) for every edge."""
    values = np.asarray(data, dtype=float)
    groups = np.asarray(labels)
    if values.ndim != 2:
        raise ValueError("data must have shape (subjects, edges)")
    if groups.ndim != 1 or len(groups) != len(values):
        raise ValueError("labels must contain one value per subject")
    unique = np.unique(groups)
    if len(unique) != 2:
        raise ValueError("two_group_ttest requires exactly two groups")
    first, second = values[groups == unique[0]], values[groups == unique[1]]
    if min(len(first), len(second)) < 2:
        raise ValueError("each group must contain at least two observations")
    statistic = scipy_stats.ttest_ind(second, first, axis=0, equal_var=False).statistic
    if not np.isfinite(statistic).all():
        raise ValueError("edge-wise t statistics are non-finite; check zero-variance edges")
    return np.asarray(statistic, float)


def glm_statistic(
    data: np.ndarray,
    design: np.ndarray,
    contrast: np.ndarray,
) -> np.ndarray:
    """OLS t statistic for a single contrast across all edge columns."""
    y = np.asarray(data, dtype=float)
    x = np.asarray(design, dtype=float)
    c = np.asarray(contrast, dtype=float).reshape(-1)
    if y.ndim != 2 or x.ndim != 2 or len(y) != len(x):
        raise ValueError("data and design must be 2D with matching observation counts")
    if len(c) != x.shape[1]:
        raise ValueError("contrast length must equal number of design columns")
    rank = np.linalg.matrix_rank(x)
    dof = len(x) - rank
    if rank < x.shape[1] or dof <= 0:
        raise ValueError("design matrix must have full column rank and positive residual df")
    pinv = np.linalg.pinv(x)
    beta = pinv @ y
    residual = y - x @ beta
    variance = np.sum(residual**2, axis=0) / dof
    contrast_variance = float(c @ np.linalg.inv(x.T @ x) @ c)
    standard_error = np.sqrt(variance * contrast_variance)
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = (c @ beta) / standard_error
    if not np.isfinite(statistic).all():
        raise ValueError("GLM produced non-finite statistics; check zero-variance edges")
    return statistic


def edge_correlation(data: np.ndarray, phenotype: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    target = np.asarray(phenotype, dtype=float)
    if values.ndim != 2 or target.ndim != 1 or len(values) != len(target):
        raise ValueError("data must be subjects-by-edges and phenotype one-dimensional")
    centered_data = values - values.mean(axis=0)
    centered_target = target - target.mean()
    denominator = np.sqrt(np.sum(centered_data**2, axis=0) * np.sum(centered_target**2))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = centered_target @ centered_data / denominator
    if not np.isfinite(correlations).all():
        raise ValueError("correlation is undefined for a constant phenotype or edge")
    return correlations
