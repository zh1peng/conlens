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
                     n_permutations=10_000, random_state=1)
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

## 8. Stability analysis

Supply genuine bootstrap statistic replicates, existing results, or subject-by-edge
data plus an explicit statistic callback to `bootstrap_lens`, then call
`summarize_stability`. The callback receives the resampled data and original-row
indices so labels and covariates can use the identical draw. Optional `strata`
enforces within-stratum sampling. Call `consensus_network(..., threshold=...)`; the
threshold is deliberately never inferred by the package.

## 9. Custom edge sets

Use `make_custom_edge_sets({"hypothesis": endpoint_frame}, validated_edges)`. Unknown
or duplicate edges raise rather than being silently repaired.

## 10. Save and load

```python
from conlens import LensResult
result.save("result.json")
restored = LensResult.load("result.json")
```

Metadata includes hashes, node order, ranking/tie rules, analysis parameters, seed,
null method, package/Python versions, and timestamp.
