# How LENS works

This page fixes the ranking, running-sum, score, leading-edge, null-inference,
normalization, and multiple-testing definitions used by the implementation.

## 1. Edge ranking

Let the frozen valid edge universe be
\(\mathcal E=\{e_1,\ldots,e_N\}\), with finite signed statistics \(r_e\), and let
\(S\subset\mathcal E\) have \(N_H\) members. Sort the complete universe so that

\[
r_{(1)}\ge r_{(2)}\ge\cdots\ge r_{(N)}.
\]

Ranking uses signed values, never absolute values or edge-wise significance filters.
Ties are resolved by ascending canonical edge ID with a stable sort; no random
jitter is added. Results report the number and fraction of tied edges and the tie
method. A list in which every statistic is identical is rejected.

## 2. Weighted running sum

For exponent \(p\ge0\), define

\[
N_R=\sum_{e_{(j)}\in S}|r_{(j)}|^p.
\]

A hit increases the running sum by \(|r_{(j)}|^p/N_R\), while a miss decreases it
by \(1/(N-N_H)\). Ranking remains signed even though hit weights use magnitudes.
For \(p=0\), or when \(p>0\) and every set-member statistic is zero, each hit is
\(1/N_H\); the latter case records `zero_weight_fallback=True`. Thus

\[
RS(i)=RS(i-1)+
\begin{cases}
|r_{(i)}|^p/N_R,&e_{(i)}\in S, N_R>0,\\
1/N_H,&e_{(i)}\in S, N_R=0,\\
-1/(N-N_H),&e_{(i)}\notin S.
\end{cases}
\]

Empty and full-universe sets have no defined running sum. The endpoint must satisfy
\(|RS(N)|\le10^{-12}\); exceeding that tolerance is an implementation error.

## 3. Enrichment score

Let \(ES^+=\max_i RS(i)\ge0\) and \(ES^-=\min_i RS(i)\le0\). Standard signed ES is

\[
ES=\begin{cases}
ES^+,&ES^+>|ES^-|,\\
ES^-,&ES^+<|ES^-|,\\
0,&|ES^+-|ES^-||\le10^{-12}.
\end{cases}
\]

The equal-extreme case has ambiguous direction and no leading edge. Positive-only
and negative-only scoring return \(ES^+\) and \(ES^-\), respectively; standard is
the default.

## 4. Leading-edge connections and network

For positive ES, define the first peak
\(i^*=\min\{i:RS(i)=ES^+\}\). The leading edge is
\(\{e_{(j)}\in S:j\le i^*\}\). For negative ES, define the last trough
\(i^*=\max\{i:RS(i)=ES^-\}\); the leading edge is
\(\{e_{(j)}\in S:j>i^*\}\). Using the last minimum makes a downstream core
deterministic under plateaus. Zero ES has an empty leading edge.

The leading-edge graph is \(G_{LE}=(V_{LE},E_{LE})\), where \(E_{LE}\) is exactly
the leading-edge set and \(V_{LE}\) contains exactly its incident nodes. No unused
node from the original edge set is added.

## 5. Hand-calculated checks

For ranked statistics \([3,2,1,-0.5,-1.5,-2.5]\) and hits at ranks 1, 2, and 5,
the weighted profile is
\([0,0.461538,0.769231,0.435897,0.102564,0.333333,0]\). Therefore
\(ES=0.769231\) and ranks 1–2 form the leading edge. With hits at ranks 2, 5, and
6, the profile is
\([0,-0.333333,0,-0.333333,-0.666667,-0.416667,0]\), so
\(ES=-0.666667\) and ranks 5–6 form the leading edge. Both are fixed numerical
tests with absolute tolerance \(10^{-12}\).

## 6. Null inference

Inference is absent unless explicitly requested; descriptive output has `NES`, P,
and q equal to `None`. The four supported null inputs are:

- **Edge permutation:** one global permutation of statistics against edge IDs is
  shared by every set in a replicate. Set sizes and overlap are retained, but
  shared-node, topological, spatial, and edge-covariance dependence are not. Its
  scope is `competitive_edge_label`.
- **Label permutation:** only for a simple phenotype/two-group design without
  nuisance covariates under exchangeability. One legal subject-label permutation
  is shared across all edges, after which all statistics, ranks, and ES values are
  recomputed. Exchangeability blocks restrict permutations when supplied.
- **Freedman–Lane:** for a tested design \(X\) and nuisance design \(Z\), where \(Z\)
  explicitly contains an intercept. Fit the reduced model
  \(Y=Z\gamma+\varepsilon\), obtain \(\hat Y_0\) and residual matrix \(R_0\), form
  \(Y_b^*=\hat Y_0+P_bR_0\), then refit the full model. The same legal \(P_b\)
  acts on every edge column.
- **Provided null:** accepts per-set null ES arrays, replicate-by-edge statistic
  matrices, or complete rank matrices in observed edge order. Statistic matrices
  recompute weighted ES normally. Rank-only input is accepted only with `weight=0`,
  because ranks do not encode the magnitudes required for weighted hit increments.
  Set definitions, finite values, edge count/order, and replicate counts are checked.
  High-level analysis requires the supplied edge-ID order, complete set definitions,
  and positive-direction label; all must match the observed analysis. The CLI uses a
  structured JSON object with `data`, `edge_ids`, `edge_sets`, and
  `positive_direction` fields.

Permutation inference defaults to 1,000 replicates when selected; publication
guidance recommends at least 10,000. Every stochastic path accepts a seed. Replicate
seeds are allocated before parallel dispatch so worker scheduling cannot change the
sequence.

## 7. Sign-specific nominal P value

For positive observed ES, the sign-specific add-one empirical probability is

\[
p={1+\#\{ES_b^0\ge ES_{obs}\}\over1+\#\{ES_b^0\ge0\}}.
\]

For negative observed ES,

\[
p={1+\#\{ES_b^0\le ES_{obs}\}\over1+\#\{ES_b^0\le0\}}.
\]

Zero ES has \(p=1\). Add-one correction prevents zero P values. Results include
same-sign null counts, extreme count, replicate count, method name, and the exact
minimum resolvable same-sign P value.

## 8. Normalized enrichment score

Let \(\mu^+\) be the mean of nonnegative null scores and \(\mu^-\) the mean
magnitude of nonpositive null scores. Then

\[
NES=ES_{obs}/\mu^+\quad\text{for }ES_{obs}>0,
\qquad
NES=ES_{obs}/\mu^-\quad\text{for }ES_{obs}<0.
\]

Negative NES retains its sign. If the required side has no null scores or zero mean,
NES is undefined; the opposite side is never substituted.

## 9. Multiple testing

Valid P values from one declared analysis/contrast/set family are adjusted with
Benjamini–Hochberg FDR. Distinct phenotypes, cohorts, modalities, contrasts, or
exploratory/confirmatory families are not mixed unless the user explicitly defines
that combined family. Output records the adjustment method, number of tested sets,
and correction-family ID.
