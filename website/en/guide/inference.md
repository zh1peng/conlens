# Permutation and inference

The English guide is being expanded. Since 2.0.1, repeated null calculations use a precompiled
integer edge-set representation and NumPy arrays internally. DataFrames are materialized only for
public inspection, serialization, and observed outputs. This optimization does not change ranking,
running sums, enrichment scores, leading edges, NES, P values, or Q values.

The current remediation changes standard-score null tails to strictly positive/negative
scores, excluding ambiguous zeros. Prespecified one-sided modes retain zero scores.
GLM uses one SVD for coefficients and contrast variance, and nonestimable edges cannot
enter LENS ranking. These scientific corrections are distinct from the historical
performance optimization above. See [validation and limitations](/guide/validation).
