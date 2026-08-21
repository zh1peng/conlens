"""conlens: connectome-wide ranked enrichment and leading-edge networks."""

from ._version import __version__
from .compare import compare_leading_edges, compare_lens_results
from .core import (
    compute_enrichment_score,
    compute_running_sum,
    extract_leading_edges,
    lens_stat,
    make_edge_statistics,
    rank_edges,
)
from .data import (
    canonicalize_edges,
    edges_to_matrix,
    matrix_to_edges,
    validate_connectome,
    validate_edge_table,
)
from .design import Contrast, DesignMatrix, make_design
from .enrichment import adjust_pvalues, lens_enrich
from .glm import lens_fl_permute, lens_glm
from .leading import (
    build_leading_network,
    compute_node_participation,
    identify_leading_hubs,
    summarize_leading_network,
)
from .permutation import lens_edge_permute
from .plotting import (
    plot_circos,
    plot_connectome_heatmap,
    plot_design,
    plot_enrichment,
    plot_enrichment_heatmap,
    plot_leading_adjacency,
    plot_lens_heatmap,
    plot_node_participation,
    plot_null_distribution,
    plot_running_sum,
    plot_stability,
)
from .results import (
    EdgeStatistics,
    GLMResult,
    LeadingNetwork,
    LensResult,
    LensSetResult,
    LensStabilityResult,
    LensStatResult,
)
from .sets import (
    make_custom_edge_sets,
    make_hemisphere_sets,
    make_network_pair_sets,
    make_within_network_sets,
    validate_edge_sets,
)
from .stability import lens_bootstrap, summarize_stability

__all__ = [
    "Contrast",
    "DesignMatrix",
    "EdgeStatistics",
    "GLMResult",
    "LeadingNetwork",
    "LensResult",
    "LensSetResult",
    "LensStabilityResult",
    "LensStatResult",
    "__version__",
    "adjust_pvalues",
    "build_leading_network",
    "canonicalize_edges",
    "compare_leading_edges",
    "compare_lens_results",
    "compute_enrichment_score",
    "compute_node_participation",
    "compute_running_sum",
    "edges_to_matrix",
    "extract_leading_edges",
    "identify_leading_hubs",
    "lens_bootstrap",
    "lens_edge_permute",
    "lens_enrich",
    "lens_fl_permute",
    "lens_glm",
    "lens_stat",
    "make_custom_edge_sets",
    "make_design",
    "make_edge_statistics",
    "make_hemisphere_sets",
    "make_network_pair_sets",
    "make_within_network_sets",
    "matrix_to_edges",
    "plot_circos",
    "plot_connectome_heatmap",
    "plot_design",
    "plot_enrichment",
    "plot_enrichment_heatmap",
    "plot_leading_adjacency",
    "plot_lens_heatmap",
    "plot_node_participation",
    "plot_null_distribution",
    "plot_running_sum",
    "plot_stability",
    "rank_edges",
    "summarize_leading_network",
    "summarize_stability",
    "validate_connectome",
    "validate_edge_sets",
    "validate_edge_table",
]
