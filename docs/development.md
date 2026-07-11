# Development

## Standards

- Python 3.10+, typed public signatures, NumPy-style numerical behavior.
- Statistical definitions are changed only with specification, test, and changelog
  updates in the same change.
- Core code has no Nilearn dependency; plotting and interfaces are separate.
- New behavior requires unit and numerical tests. Stable sorting and seeded null
  sequences must remain platform and worker-scheduling independent.

## Decisions

- Canonical undirected identity is based on the supplied/derived node index, never
  lexical endpoint order.
- The stored running-sum profile includes `RS(0)` and therefore has length `N+1`.
- Subject connectomes are converted once to a shared subject-by-edge matrix.
- Parallel edge permutation spawns one seed per replicate before dispatch, making
  results independent of worker scheduling.
- Consensus stability thresholds are mandatory user inputs.

## Package checks

Run `ruff check .`, `pytest --cov=conlens --cov-fail-under=90`, `python -m build`,
and `twine check dist/*`. CI repeats linting and tests on Linux, macOS, and Windows.

PyPI, TestPyPI, Zenodo, GitHub Release/tag, and public documentation deployment are
out of scope unless the user provides a new explicit authorization. Package checks
must not upload artifacts or request publishing credentials.
