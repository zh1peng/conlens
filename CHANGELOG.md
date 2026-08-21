# Changelog

## Unreleased

- Replaced separate subject-level group/phenotype entry points with one validated
  design-matrix and named-contrast GLM workflow.
- Added explicit indicator/continuous/interaction design construction, design and
  contrast visualization, signed partial r, model-adjusted Hedges' g, and joint
  contrast-by-set BH adjustment.
- Added observed-anchored, full-pipeline subject-bootstrap stability summaries.
- Added conditional and full-pipeline edge stability, Monte Carlo bounds, gated
  reproducible cores, and serializable `LensStabilityResult` tables.
- Added deterministic stratified-subject refitting through
  `SubjectLensAnalysis.bootstrap_stability`.

## 1.0.0 - 2026-07-11

- Initial stable API for deterministic LENS enrichment.
- Explicit edge, contrast-specific Freedman–Lane, and provided-null inference.
- Leading-network, stability, comparison, visualization, serialization, CLI, and Nilearn adapters.
