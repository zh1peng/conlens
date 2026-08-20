# API

The stable top-level API is exported by `conlens`. Core enrichment includes
`lens_enrich`, `compute_running_sum`, `compute_enrichment_score`, and
`extract_leading_edges`. Data functions include `validate_connectome`,
`validate_edge_table`, `canonicalize_edges`, `matrix_to_edges`, and
`edges_to_matrix`. Set, inference, leading-network, stability, and comparison
functions follow the names in the development specification.

`LensAnalysis` handles edge tables. `LensAnalysis.from_subject_connectomes` returns
a `SubjectLensAnalysis` whose `two_group` and `glm` methods expose label permutation
and Freedman–Lane explicitly. Every high-level result is a `LensResult` traceable to
the same public low-level functions.

## Bootstrap stability APIs

`SubjectLensAnalysis.bootstrap_stability` is the high-level full-pipeline subject
bootstrap. Its `refit` callback receives the resampled analysis, the sampled
original-row indices, and a replicate-specific seed. The callback must return a
`LensResult` after repeating the edge model, null inference, ranking, LENS tests,
and BH adjustment over the same correction family.

Formal summaries require a recorded `positive_direction` and reject changes in
the analysis signature, edge/node mapping, correction family, null scheme, or
whether exchangeability blocks were used. The first bootstrap result is validated
before the remaining parallel jobs start. Built-in signatures record the
two-group levels, GLM contrast, or phenotype callable; the callback remains
responsible for preserving the scientific meaning of all subject-aligned design
columns.

`summarize_bootstrap_stability` applies the same observed-result-anchored summary to
completed bootstrap `LensResult` objects generated elsewhere. Both routes return a
`LensStabilityResult` with `set_summary`, `edge_summary`, `replicate_summary`, and
metadata. Use `get_set(set_name)` and `edges_for(set_name)` for one tested set;
`save` and `load` provide JSON round trips. As in `LensResult`, JSON converts tuple
node labels to JSON arrays and reloads them as lists; use scalar string or integer
node labels when exact Python label types must be retained.

`bootstrap_lens` and `summarize_stability` remain the lower-level descriptive
workflow. They summarize leading-edge localization across statistic replicates and
do not, by themselves, require repeated subject-level inference, BH significance,
or agreement with an observed direction. Their output must not be described as
full-pipeline stability.

## Public reference

::: conlens

## Plotting reference

::: conlens.plotting
