# Changelog

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
