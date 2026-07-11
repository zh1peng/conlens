# conlens

`conlens` is a transparent, reproducible, modality-agnostic implementation of LENS
(Leading-edge Network Set enrichment) for ranked connectome-wide statistics.

It uses every valid edge in a signed ranked list. It does not threshold edge-wise
statistics, silently choose a null model, or treat leading edges as individually
significant edges.

## Install

```bash
pip install conlens
```

Nilearn integration is optional:

```bash
pip install "conlens[nilearn]"
```

## Minimal analysis

```python
import pandas as pd
from conlens import lens_enrich

edges = pd.DataFrame({
    "node1": ["A", "A", "A", "B", "B", "C"],
    "node2": ["B", "C", "D", "C", "D", "D"],
    "statistic": [3.0, 2.0, 1.0, -0.5, -1.5, -2.5],
})

# Validate once to discover canonical edge IDs, or construct sets from metadata.
from conlens import validate_edge_table
validated = validate_edge_table(edges)
edge_sets = {"example": set(validated.loc[[0, 1, 4], "edge_id"])}

result = lens_enrich(
    edges,
    edge_sets,
    min_size=1,
    positive_direction="case > control",
    store_running_sum=True,
)
print(result.to_frame())
```

Without `null_method`, the result is descriptive: `NES`, `p_value`, and `q_value`
remain `None`. Inference is always explicit:

```python
inferred = lens_enrich(
    edges,
    edge_sets,
    min_size=1,
    null_method="edge_permutation",
    n_permutations=10_000,
    random_state=42,
)
```

The edge-label null is competitive and does **not** preserve shared-node,
topological, spatial, or edge-covariance dependence. Use subject-level label
permutation or Freedman–Lane where the design permits it.

See [Concepts](docs/concepts.md), [How LENS works](docs/algorithm.md),
[Tutorials](docs/tutorials.md), and the [Interpretation guide](docs/interpretation.md).

## CLI

```bash
conlens edges.csv sets.json result.json \
  --null-method edge_permutation --n-permutations 10000 --random-state 42
```

## Development

```bash
python -m pip install -e ".[dev]"
pytest --cov=conlens
ruff check .
python -m build
```

The package supports Python 3.10+ on Linux, macOS, and Windows and is MIT licensed.
