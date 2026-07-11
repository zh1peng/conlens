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

## Public reference

::: conlens

## Plotting reference

::: conlens.plotting
