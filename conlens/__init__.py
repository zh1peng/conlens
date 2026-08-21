"""conlens: connectome-wide ranked enrichment and leading-edge networks."""

from ._version import __version__
from .analysis import LensAnalysis, SubjectLensAnalysis
from .compare import compare_leading_edges, compare_lens_results
from .core import (
    compute_enrichment_score,
    compute_running_sum,
    extract_leading_edges,
    lens_enrich,
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
from .inference import (
    adjust_pvalues,
    edge_permutation_null,
    empirical_pvalue,
    normalize_enrichment_scores,
    permutation_test,
    provided_null,
)
from .leading import (
    build_leading_network,
    compute_node_participation,
    identify_leading_hubs,
    summarize_leading_network,
)
from .plotting import plot_design
from .results import GLMResult, LeadingNetwork, LensResult, LensSetResult, LensStabilityResult
from .sets import (
    make_custom_edge_sets,
    make_hemisphere_sets,
    make_network_pair_sets,
    make_within_network_sets,
    validate_edge_sets,
)
from .stability import (
    bootstrap_lens,
    consensus_network,
    summarize_bootstrap_stability,
    summarize_stability,
)

__all__ = [
    "Contrast",
    "DesignMatrix",
    "GLMResult",
    "LeadingNetwork",
    "LensAnalysis",
    "LensResult",
    "LensSetResult",
    "LensStabilityResult",
    "SubjectLensAnalysis",
    "__version__",
    "adjust_pvalues",
    "bootstrap_lens",
    "build_leading_network",
    "canonicalize_edges",
    "compare_leading_edges",
    "compare_lens_results",
    "compute_enrichment_score",
    "compute_node_participation",
    "compute_running_sum",
    "consensus_network",
    "edge_permutation_null",
    "edges_to_matrix",
    "empirical_pvalue",
    "extract_leading_edges",
    "identify_leading_hubs",
    "lens_enrich",
    "make_custom_edge_sets",
    "make_design",
    "make_hemisphere_sets",
    "make_network_pair_sets",
    "make_within_network_sets",
    "matrix_to_edges",
    "normalize_enrichment_scores",
    "permutation_test",
    "plot_design",
    "provided_null",
    "rank_edges",
    "summarize_leading_network",
    "summarize_bootstrap_stability",
    "summarize_stability",
    "validate_connectome",
    "validate_edge_sets",
    "validate_edge_table",
]
