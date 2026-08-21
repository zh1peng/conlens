# Concepts

## What is LENS?

LENS is a ranked enrichment framework whose analysis unit is an edge. All valid
edges are ordered by a signed statistic. A predefined edge set is enriched when
its members cluster near the positive or negative extreme of that ranking.

## Why use the full ranked edge list?

Pre-thresholding discards graded evidence and makes results depend on an arbitrary
edge-wise cutoff. LENS instead compares hit accumulation with uniform misses across
the frozen edge universe.

## What is an edge set?

An edge set is a hypothesis defined before testing: for example all connections
between two canonical networks. Custom sets may overlap. Network-pair sets from a
single-label parcellation partition an undirected universe.

## Positive and negative enrichment

Positive enrichment means set edges accumulate toward high signed statistics;
negative enrichment means they accumulate toward low signed statistics. Biological
meaning comes entirely from the recorded `positive_direction` of the statistic.

## Leading edge and leading-edge network

The leading edge contains set members up to the first positive running-sum maximum,
or after the last negative minimum. It is not an edge-wise significance list. The
leading-edge network contains exactly those edges and their incident nodes.

## ES versus NES

ES is the deterministic extreme of the observed running sum. NES divides ES by the
mean magnitude of same-sign null scores. NES therefore requires a declared null
model and is neither an effect size nor a standardized regression coefficient.

## Why the null model matters

Different nulls answer different questions and preserve different dependencies.
Edge-label permutation is competitive but breaks connectome dependence. Subject-level
GLM inference uses a contrast-specific Freedman–Lane reduced model and applies the
same legal residual-row permutation across every edge.
