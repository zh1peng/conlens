<img class="conlens-doc-logo" src="/conlens-logo.png" alt="ConLens logo">

# Ranked connectome enrichment

ConLens tests predefined edge sets against a complete signed connectome-wide ranking and
reconstructs the leading-edge networks that drive enrichment extrema.

The refactored API separates four stages: `lens_glm` (or external edge statistics),
`lens_stat`, a streaming null generator (`lens_fl_permute` or `lens_edge_permute`), and
`lens_enrich`. The final result retains set-level null enrichment scores without storing an
edge × permutation matrix.

The full English guide is being prepared. The [Chinese documentation](/) contains the
current tutorials, formulas, bootstrap workflow, and visualization examples.
