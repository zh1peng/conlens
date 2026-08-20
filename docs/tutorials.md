# Tutorials

These compact recipes cover the ten supported workflows. All stochastic examples
set a seed; publication analyses should ordinarily use at least 10,000 permutations.
Complete executable versions live in `examples/` and are run by the test suite on
every supported platform.

## 1. Edge-statistics input

Create a DataFrame with `node1`, `node2`, and a finite signed `statistic`; call
`validate_edge_table`, construct sets from its canonical `edge_id`, then call
`lens_enrich`. Record `positive_direction`.

## 2. Network-pair construction

```python
from conlens import make_network_pair_sets, validate_edge_table
edges = validate_edge_table(edges)
sets = make_network_pair_sets(edges, {"A": "DMN", "B": "VIS", "C": "DMN"})
```

## 3. Edge-permutation analysis

```python
result = lens_enrich(edges, sets, null_method="edge_permutation",
                     n_permutations=10_000, random_state=1,
                     positive_direction="case > control")
```

This competitive null does not preserve connectome dependence.

## 4. Subject-level two-group analysis

```python
from conlens import LensAnalysis
analysis = LensAnalysis.from_subject_connectomes(connectomes, sets, node_labels=labels)
result = analysis.two_group(group, null_method="label_permutation",
                            exchangeability_blocks=site, random_state=1)
```

## 5. GLM with nuisance covariates

```python
tested = age[:, None]
nuisance = np.column_stack([np.ones(len(age)), sex, site_dummies])
result = analysis.glm(tested, nuisance, null_method="freedman_lane",
                      exchangeability_blocks=family, random_state=1)
```

The nuisance design must explicitly contain an intercept.

## 6. Nilearn workflow

Pass `ConnectivityMeasure.fit_transform(...)` output to
`LensAnalysis.from_subject_connectomes`. The core package never imports Nilearn;
`conlens.interfaces.nilearn.plot_nilearn_connectome` is optional.

## 7. Leading-edge visualization

```python
from conlens import build_leading_network
from conlens.plotting import plot_leading_adjacency
network = build_leading_network(result, "DMN--VIS")
plot_leading_adjacency(network)
```

## 8. Bootstrap stability

### Full-pipeline subject bootstrap

First run the observed analysis with inference and BH adjustment. Then supply a
small `refit` callback that repeats that same complete analysis in each subject
bootstrap sample:

```python
import numpy as np
from conlens import LensAnalysis

group = np.asarray(group)
analysis = LensAnalysis.from_subject_connectomes(connectomes, sets, node_labels=labels)
observed = analysis.two_group(
    group,
    null_method="label_permutation",
    n_permutations=10_000,
    random_state=1,
)


def refit(sample, indices, fit_seed):
    return sample.two_group(
        group[indices],
        null_method="label_permutation",
        n_permutations=10_000,
        random_state=fit_seed,
    )

stability = analysis.bootstrap_stability(
    observed,
    refit,
    n_bootstraps=1_000,
    random_state=2,
    strata=group,
)
print(stability.set_summary)
print(stability.edges_for("DMN--VIS"))
```

The callback receives a `SubjectLensAnalysis` whose data contain the resampled
subjects, the corresponding original-row `indices`, and a seed reserved for that
replicate's inference. Index every subject-aligned label, design row, covariate, or
exchangeability block with `indices`. For example, use
`exchangeability_blocks=site[indices]` inside the callback. Family or repeated-
measure clusters require a cluster bootstrap, which this executor does not yet
support.

`strata` controls the outer bootstrap, not the inner permutation scheme. For a
multi-site two-group analysis, pass a one-dimensional combined stratum such as
`list(zip(site, group, strict=True))` to preserve every site-by-group count. An
invalid bootstrap design stops with the failing replicate number; samples are not
silently discarded or redrawn. The observed result is checked before sampling,
and the first refit is checked for compatibility before the remaining jobs start.

The result contains:

- `set_summary`: detection rate, direction consistency, same-direction set
  stability, interval bounds, localization diagnostics, and core sizes;
- `edge_summary`: conditional and full-pipeline edge stability with their interval
  bounds and core indicators;
- `replicate_summary`: one diagnostic row per tracked set and replicate;
- `metadata`: bootstrap, interval, correction-family, and interpretation details.

Only sets with `q_value <= significance_alpha` in `observed` are tracked. A
bootstrap leading edge counts toward edge stability only when that set again has
`q_value <= significance_alpha` with the observed direction. Full-pipeline edge
stability uses all bootstrap samples as its denominator; conditional stability
uses only same-direction detected samples.
By default, a core requires its interval lower bound to exceed
`core_threshold=0.50`. A conditional core is reportable only after
`min_same_direction=30` eligible detections and when the set-stability lower bound
exceeds the fixed 0.50 gate.

If complete, compatible bootstrap `LensResult` objects already exist, summarize
them directly:

```python
from conlens import summarize_bootstrap_stability
stability = summarize_bootstrap_stability(observed, bootstrap_results)
stability.save("stability.json")
```

The current executor supports independent-subject and stratified-subject sampling.
Precomputed draws and seeds support reproducible parallel refits through `n_jobs`
when all callback randomness uses `fit_seed`. Cluster resampling and
checkpoint/resume are not supported. Bootstrap results are omitted from the saved
object unless `keep_bootstrap_results=True`.

### Descriptive localization sensitivity

`bootstrap_lens` followed by `summarize_stability` is the legacy, lower-level
workflow for genuine statistic replicates or subject data plus a statistic
callback. It recalculates rankings and leading edges, but the summary does not gate
edge inclusion on observed BH significance and direction. Use it for descriptive
localization sensitivity, not full-pipeline stability. `consensus_network` belongs
to this descriptive workflow and requires an explicit threshold.

## 9. Custom edge sets

Use `make_custom_edge_sets({"hypothesis": endpoint_frame}, validated_edges)`. Unknown
or duplicate edges raise rather than being silently repaired.

## 10. Save and load

```python
from conlens import LensResult, LensStabilityResult
result.save("result.json")
restored = LensResult.load("result.json")
stability.save("stability.json")
restored_stability = LensStabilityResult.load("stability.json")
```

Metadata includes hashes, node order, ranking/tie rules, analysis parameters, seed,
null method, package/Python versions, and timestamp.
