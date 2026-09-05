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

Choose the null model before interpreting significance. Subject-level GLM/FL requires
appropriate residual exchangeability. External observed and permutation statistics inherit
the external model's assumptions. With observed statistics alone, start with descriptive
ES and leading edges. Edge-label permutation requires a distinct edge-exchangeability
null and cannot replace subject-level inference.

A positive ES means a set ranks relatively high against the background; it does not mean
every member edge has a positive effect. Enrichment significance applies to sets, not to
individual leading edges.

## Install

```bash
git clone https://github.com/zh1peng/conlens.git
cd conlens
python -m pip install .
```

## Run a complete example

From the cloned repository root:

```bash
python -m examples.teaching_workflow
```

This deterministic simulation supplies connectomes, phenotypes, string node labels, and
network sets. It runs GLM/FL, descriptive analysis, ID-aligned external null import,
annotation-based set construction, and a small bootstrap. Executed output and JSON files
are written to `website/generated`. The low resampling counts demonstrate execution,
not statistical calibration. See the [first-analysis tutorial](https://zh1peng.github.io/conlens/guide/quick-start)
for the shared executable source and actual output.

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

## Scientific remediation

ConLens 2.3.0 unifies GLM coefficient/variance computation through one SVD,
exclude ambiguous zero ES from standard-mode directional tails, and retain zeros for
prespecified one-sided scores. Nonestimable GLM edges remain auditable but cannot enter
LENS ranking. Bootstrap exports its complete observed reference, actual seeds, and draws.
See `CHANGELOG.md` and `benchmarks/README.md` for affected behavior and validation scope.
