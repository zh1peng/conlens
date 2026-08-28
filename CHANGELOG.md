# Changelog

## 2.1.0 - 2026-08-28

- Added `NullEdgeStatistics` and `make_null_edge_statistics` for externally
  computed edge-by-permutation matrices.
- External null matrices use lazy NumPy column views, can be iterated more than
  once, and plug into the existing `lens_stat` → `lens_enrich` workflow.
- Added validation for matrix shape, finite numeric values, ordered edge identity,
  and permutation provenance metadata.
- Kept the existing on-the-fly GLM, Freedman–Lane, and edge-label permutation APIs
  unchanged, with exact-equivalence tests between streamed and matrix-backed nulls.

## 2.0.1 - 2026-08-27

- Replaced the repeated null LENS DataFrame path with a lazy, NumPy-based
  implementation using precompiled integer edge-set membership and tie ranks.
- Kept the public Pandas tables, result schemas, statistical definitions, random
  streams, and serialized outputs unchanged.
- Added exact equivalence tests against the materialized 2.0.0 Pandas path for
  edge permutation and Freedman–Lane workflows, including ties, all score types,
  multiple weights, and invalid edge sets.
- Added a reproducible benchmark that verifies exact ES equality before reporting
  speedup.

## 2.0.0 - 2026-08-21

- Rebuilt the public workflow around `lens_glm`, `lens_stat`, streaming null
  generators, and `lens_enrich`.
- Added named multi-contrast GLM designs, partial correlation, and model-adjusted
  Hedges' g based on the full-model residual standard deviation.
- Added contrast-specific Freedman–Lane and edge-label permutation iterators.
- Retained compact permutation × edge-set null ES tables for audit and plotting,
  without storing edge × permutation arrays.
- Rebuilt subject bootstrap as a complete GLM → FL → LENS → joint-BH refit.
- Added annotated connectome heatmaps, network-enrichment heatmaps, running-sum
  diagnostics, null distributions, and leading-edge circos plots.
- Replaced the former public analysis objects and inference entry points; 2.0 is
  intentionally not API-compatible with 1.0.

## 1.0.0 - 2026-07-11

- Initial public development release.
