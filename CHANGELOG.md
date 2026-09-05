# Changelog

## 2.3.0 - 2026-09-05

- Fixed GLM contrast variance to use the same SVD pseudoinverse as coefficients,
  with the rank-check tolerance and a response-relative residual degeneracy check.
  This corrects t, P and standardized effects for affected scaled/offset raw designs;
  weighted ES may also change. The reviewed example now gives t=7.01717, not 338.19.
- Fixed standard-score directional null tails to exclude ambiguous zero ES. Zero
  observed ES yields P=1 and undefined NES. Prespecified positive/negative scores
  retain all null draws, including zeros. P and NES use the same tail definition;
  added mutually exclusive sign counts, `n_null_zero` and actual `n_null_tail`.
  Affected standard-mode P/NES can change, particularly with unweighted discrete
  rankings; this is a scientific correction, not a performance change.
- Nonestimable GLM edges remain in audit output with IDs and placeholder 0/1 values,
  but `lens_stat` now rejects these inputs instead of treating them as no association.
  This intentionally stops previously accepted degenerate inference. All-singleton
  blocks and row permutations that cannot change the tested contrast are rejected.
- Bootstrap results serialize the complete observed reference, actual seed derivation,
  per-fit seeds and subject draw indices. Replicate tables include null-tail precision.
  Older stability JSON files load with `observed_reference=None`.
- Inference metadata retains input-data and block fingerprints, permutation seeds and
  numeric dependency versions. Fingerprints do not establish semantic subject alignment.
- Reworked Chinese onboarding, model/null selection, result interpretation, subject
  alignment and stability tutorials. Examples share tested executable sources and
  generated outputs; fixed string-node-label propagation and removed arbitrary size caps.
- Added independent QR and exhaustive discrete regression checks, targeted calibration,
  fixed-target/background-signal checks and bootstrap inner-seed/budget sensitivity.
  Version-bound records and limitations are in `benchmarks/results/remediation.json`
  and `website/guide/validation.md`. No general validation of clustered designs is claimed.

## 2.2.0 - 2026-08-30

- Added verified local/remote loading of atlas-aligned node maps with `load_maps`.
- Added `make_node_value_sets`, `make_node_distance_sets`, and
  `make_profile_similarity_sets` for three explicit map-based edge-set definitions.
- Added serializable, Mapping-compatible `EdgeSets` with construction provenance and
  selection audits.
- Pinned GitHub map sources to resolved commits, hardened checksum-backed cache paths,
  made score ties independent of custom edge IDs, and reject cross-universe `EdgeSets`.

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
