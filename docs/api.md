# API

The subject-level API is a single named-contrast GLM workflow:

```python
from conlens import Contrast, LensAnalysis, make_design, plot_design

design = make_design(
    indicators={"control": diagnosis == "control", "g1": diagnosis == "g1"},
    continuous={"age": age, "motion": mean_fd},
    add_intercept=False,
)
contrasts = {
    "g1_vs_control": Contrast(
        {"g1": 1, "control": -1},
        effect_size="hedges_g",
        positive_direction="g1 > control",
    ),
    "age": Contrast(
        {"age": 1},
        effect_size="partial_r",
        positive_direction="increases with age",
    ),
}
fit = analysis.glm(design, contrasts, n_permutations=10_000, random_state=1)
```

`make_design` semantic mode mean-centers continuous columns by default, never
centers 0/1 indicators, constructs explicitly named interactions after centering,
and adds an intercept by default. Cell-means group designs use
`add_intercept=False`. Raw `matrix=` input is used exactly as supplied.

`analysis.glm` returns `GLMResult`; indexing it by contrast name returns a
`LensResult`. Each contrast receives its own constrained reduced model for
Freedman–Lane, and all valid contrast-by-set P values are adjusted together by BH.

## Bootstrap stability APIs

`SubjectLensAnalysis.bootstrap_stability` receives one observed `LensResult` and a
`refit` callback. In each callback use `design.take(indices)`, rerun the same full
contrast family, and select the observed contrast from the resulting `GLMResult`.

`summarize_bootstrap_stability` summarizes compatible completed results generated
elsewhere. `bootstrap_lens` and `summarize_stability` remain lower-level,
descriptive localization tools and do not replace full subject-level refitting.

## Public reference

::: conlens

## Plotting reference

::: conlens.plotting
