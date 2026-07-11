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
