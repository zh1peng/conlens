# Python API

## Design and edge-wise model

| API | Purpose |
| --- | --- |
| `make_design` / `DesignMatrix` | Build and validate a named design matrix |
| `Contrast` | Define a named one-degree-of-freedom contrast and effect size |
| `lens_glm` | Fit observed edge-wise effects from subject connectomes |
| `lens_fl_permute` | Stream contrast-specific Freedman–Lane null edge effects |
| `plot_design` | Inspect the design matrix and contrast vectors |

`partial_r` and model-adjusted `hedges_g` are supported. Zero-residual-variance edges remain in
the edge universe with neutral statistics and `estimable=False`.

## Data and edge sets

`validate_connectome`, `matrix_to_edges`, `edges_to_matrix`, `validate_edge_table`, and
`canonicalize_edges` define the edge universe. Use `make_network_pair_sets`,
`make_within_network_sets`, `make_hemisphere_sets`, or `make_custom_edge_sets` to build sets;
`validate_edge_sets` validates an existing mapping.

## LENS and inference

`make_edge_statistics` validates external signed statistics. `lens_edge_permute` streams
edge-label nulls, `lens_stat` computes the same deterministic statistic for observed and null
inputs, and `lens_enrich` applies size filters, normalization, empirical P values, and joint BH.
The low-level pure functions are `rank_edges`, `compute_running_sum`,
`compute_enrichment_score`, `extract_leading_edges`, and `adjust_pvalues`.

## Results, comparison, and stability

The main values are `EdgeStatistics`, `LensStatResult`, `LensResult`, `GLMResult`,
`LensStabilityResult`, and `LeadingNetwork`. JSON payloads carry a schema version and object type;
serializable results provide matching save/load methods. `build_leading_network`,
`compute_node_participation`, `identify_leading_hubs`, and `summarize_leading_network` reconstruct
and summarize leading networks. `compare_leading_edges` and `compare_lens_results` reject
incompatible node/edge identities.

`lens_bootstrap` reruns the complete GLM → Freedman–Lane → LENS → BH pipeline.
`summarize_stability` accepts externally produced completed replicates. Both consume bootstrap
results incrementally rather than retaining every full replicate.

## Plotting

The plotting API includes `plot_connectome_heatmap`, `plot_lens_heatmap`,
`plot_enrichment_heatmap`, `plot_running_sum`, `plot_null_distribution`, `plot_enrichment`,
`plot_circos`, `plot_leading_adjacency`, `plot_node_participation`, and `plot_stability`.
