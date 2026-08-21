<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/conlens-logo-on-white.png">
    <img src="docs/assets/brand/conlens-logo.png" alt="ConLens logo" width="420">
  </picture>
</p>

<p align="center">
  <a href="https://zh1peng.github.io/conlens/">中文文档</a> ·
  <a href="https://zh1peng.github.io/conlens/en/">English</a> ·
  <a href="https://github.com/zh1peng/conlens">GitHub</a>
</p>

# ConLens

ConLens performs ranked enrichment of predefined connectome edge sets and reconstructs
the leading-edge networks that drive each enrichment result. It uses the complete signed
edge ranking; edge-wise significance filtering is not part of the method.

The public workflow has four explicit stages:

```text
lens_glm / external statistics
             ↓
          lens_stat
             ↓
lens_fl_permute / lens_edge_permute → lens_stat (streamed)
             ↓
         lens_enrich
```

`lens_enrich` never fits or permutes edge models. It consumes observed LENS statistics and
a stream of null LENS statistics, then performs normalization, empirical inference, and a
joint Benjamini–Hochberg correction. The result keeps one null enrichment score per
permutation and edge set—not the much larger edge × permutation matrix.

## Install

```bash
git clone https://github.com/zh1peng/conlens.git
cd conlens
python -m pip install .
```

## Subject-level example

```python
from conlens import (
    Contrast, lens_enrich, lens_fl_permute, lens_glm,
    lens_stat, make_design,
)

design = make_design(
    groups={
        "control": diagnosis == "control",
        "g1": diagnosis == "g1",
        "g2": diagnosis == "g2",
    },
    continuous={"age": age},
    indicators={"sex": sex},
)
contrasts = {
    "g1_vs_control": Contrast(
        {"g1": 1, "control": -1}, "hedges_g", "g1 > control"
    ),
    "age": Contrast(
        {"age": 1}, "partial_r", "connectivity increases with age"
    ),
}

true_edges = lens_glm(connectomes, design=design, contrasts=contrasts)
observed = lens_stat(true_edges, edge_sets, store_running_sum=True)
null_edges = lens_fl_permute(
    connectomes,
    design=design,
    contrasts=contrasts,
    n_permutations=10_000,
    random_state=42,
)
null_stats = (lens_stat(item, edge_sets) for item in null_edges)
result = lens_enrich(
    observed,
    null_stats,
    family_name="primary-model",
)
```

For a continuous contrast, the ranked edge statistic is partial correlation. For a group
contrast, it is model-adjusted Hedges' g using the full model residual standard deviation.
See the [Chinese tutorial](https://zh1peng.github.io/conlens/tutorials/design-and-contrasts)
for formulas, multi-group examples, permutation details, bootstrap stability, and figures.

## Development

```bash
python -m pip install -e ".[dev]"
pytest --cov=conlens --cov-fail-under=90
ruff check .
python -m mypy conlens
python -m build
```

Python 3.10+ · Linux, macOS, and Windows · MIT license
