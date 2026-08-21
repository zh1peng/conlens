# Tutorials

The Chinese VitePress tutorial contains the complete, rendered walkthrough:
[Design matrix 与 contrasts](https://zh1peng.github.io/conlens/tutorials/design-and-contrasts).

## Continuous association

Age only:

```python
design = make_design(continuous={"age": age})
contrasts = {
    "age": Contrast({"age": 1}, "partial_r", "increases with age"),
}
fit = analysis.glm(design, contrasts, n_permutations=10_000, random_state=1)
```

Age with covariates:

```python
design = make_design(
    indicators={"male": sex == "male", "site_B": site == "B"},
    continuous={"age": age, "motion": mean_fd},
)
fit = analysis.glm(design, contrasts, n_permutations=10_000, random_state=1)
```

The ranking effect is Pearson r without covariates and signed partial r with
covariates.

## One group contrast

Without covariates:

```python
design = make_design(
    indicators={"control": diagnosis == "control", "g1": diagnosis == "g1"},
    add_intercept=False,
)
contrasts = {
    "g1_vs_control": Contrast(
        {"g1": 1, "control": -1}, "hedges_g", "g1 > control"
    ),
}
```

With covariates, add them to the same full model:

```python
design = make_design(
    indicators={
        "control": diagnosis == "control",
        "g1": diagnosis == "g1",
        "male": sex == "male",
        "site_B": site == "B",
    },
    continuous={"age": age, "motion": mean_fd},
    add_intercept=False,
)
```

## Two group contrasts in one family

```python
design = make_design(
    indicators={
        "control": diagnosis == "control",
        "g1": diagnosis == "g1",
        "g2": diagnosis == "g2",
    },
    add_intercept=False,
)
contrasts = {
    "g1_vs_control": Contrast(
        {"g1": 1, "control": -1}, "hedges_g", "g1 > control"
    ),
    "g2_vs_control": Contrast(
        {"g2": 1, "control": -1}, "hedges_g", "g2 > control"
    ),
}
fit = analysis.glm(design, contrasts, n_permutations=10_000, random_state=1)
```

For the covariate-adjusted version, add indicator and continuous nuisance columns
exactly as in the previous section. Both contrasts use the residual SD from this
same full three-group model, and BH is joint across both contrasts and all valid
edge sets.

## Design visualization

```python
from conlens import plot_design
plot_design(design, contrasts)
```

## Full-pipeline bootstrap

Use the same complete contrast family inside each refit:

```python
observed = fit["g1_vs_control"]

def refit(sample, indices, fit_seed):
    return sample.glm(
        design.take(indices),
        contrasts,
        n_permutations=10_000,
        random_state=fit_seed,
    )["g1_vs_control"]

stability = analysis.bootstrap_stability(
    observed,
    refit,
    n_bootstraps=1_000,
    random_state=2,
    strata=diagnosis,
)
```
