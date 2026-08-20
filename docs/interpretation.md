# Interpretation guide

- ES is a running-sum statistic, not an effect size.
- NES is null-normalized ES, not a standardized beta.
- Significant enrichment does not mean every set edge is significant.
- Leading edges are running-sum drivers, not individually significant edges.
- A negative NES means accumulation toward low ranking statistics; its phenotype
  meaning depends on `positive_direction`.
- Edge permutation does not preserve connectivity dependence, topology, shared-node
  structure, or spatial structure.
- Label and Freedman–Lane inference are valid only under the supplied exchangeability
  assumptions. Blocks must represent restricted exchangeability correctly.
- Results from distinct contrasts, cohorts, modalities, or confirmatory families
  should not share an FDR correction unless that family was explicitly intended.

## Bootstrap stability

- `SubjectLensAnalysis.bootstrap_stability` and `summarize_bootstrap_stability`
  report empirical sampling stability around an observed, BH-adjusted analysis.
  They do not estimate the probability that a set or edge is true, control
  edge-level FDP, or provide a precise probability of replication in a future
  study.
- `set_stability` is the fraction of all bootstrap samples in which the set again
  has `q_value <= significance_alpha` with the observed direction. `detection_rate`
  also includes detections in the other direction; `direction_consistency` is
  conditional on detection.
- `conditional_stability` is edge inclusion among same-direction detected samples.
  `full_pipeline_stability` uses all bootstrap samples as the denominator and is
  therefore `set_stability * conditional_stability` when the conditional quantity
  is defined.
- Stability bounds are Jeffreys bootstrap-frequency Monte Carlo intervals. They are
  not population confidence intervals, edge-confidence intervals, or posterior
  probabilities.
- A `full_pipeline_core` is a reproducible core under the chosen bootstrap,
  significance threshold, interval level, and `core_threshold`; it is not a set of
  confirmed true or individually significant edges.
- Conditional cores are reportable only when the set-level lower bound exceeds the
  built-in 0.50 gate and at least `min_same_direction` eligible samples are
  available. Otherwise conditional localization point estimates, when defined,
  remain descriptive.
- `bootstrap_lens` with `summarize_stability` reports ungated leading-edge
  localization sensitivity. Do not relabel its inclusion frequencies or consensus
  network as full-pipeline stability.
- The implemented resampling unit is the independent subject, optionally within
  strata. `strata` and permutation `exchangeability_blocks` serve different roles.
  Cluster bootstrap and checkpoint/resume are currently unsupported.
