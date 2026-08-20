<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/conlens-logo-on-white.png">
    <img src="docs/assets/brand/conlens-logo.png" alt="ConLens logo" width="420">
  </picture>
</p>

<p align="center">
  <a href="https://zh1peng.github.io/conlens/">中文文档</a>
  ·
  <a href="https://zh1peng.github.io/conlens/en/">English</a>
  ·
  <a href="https://github.com/zh1peng/conlens">GitHub</a>
</p>

# ConLens

`conlens` is a transparent, reproducible, modality-agnostic implementation of LENS
(Leading-edge Network Set enrichment) for ranked connectome-wide statistics.

It uses every valid edge in a signed ranked list. It does not threshold edge-wise
statistics, silently choose a null model, or treat leading edges as individually
significant edges.

## Install

```bash
git clone https://github.com/zh1peng/conlens.git
cd conlens
pip install .
```

Nilearn integration is optional:

```bash
pip install ".[nilearn]"
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
    positive_direction="case > control",
)
```

The edge-label null is competitive and does **not** preserve shared-node,
topological, spatial, or edge-covariance dependence. Use subject-level label
permutation or Freedman–Lane where the design permits it.

## Full-pipeline bootstrap stability

For subject-level analyses, `SubjectLensAnalysis.bootstrap_stability` resamples
subjects and calls a user-supplied `refit` function for every replicate. That
function must repeat the edge model, null inference, LENS tests, and BH adjustment;
the returned `LensStabilityResult` separates set detection/direction stability from
conditional and full-pipeline leading-edge stability. The older
`bootstrap_lens`/`summarize_stability` workflow remains available for descriptive,
ungated localization sensitivity.

Bootstrap frequencies and their Monte Carlo intervals are sampling-sensitivity
summaries, not edge-truth probabilities, FDP guarantees, or exact future-study
replication probabilities. See the [stability tutorial](docs/tutorials.md#8-bootstrap-stability)
for the complete refit callback and supported resampling schemes.

See [Concepts](docs/concepts.md), [How LENS works](docs/algorithm.md),
[Tutorials](docs/tutorials.md), and the [Interpretation guide](docs/interpretation.md).
The Chinese-first VitePress documentation is available at
[zh1peng.github.io/conlens](https://zh1peng.github.io/conlens/).

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
