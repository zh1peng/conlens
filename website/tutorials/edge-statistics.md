# 只有 edge statistics 时

如果没有个体 connectomes，但有一张完整的 signed edge-statistic 表，先明确统计量的方向和名称：

```python
from conlens import make_edge_statistics, lens_stat, lens_enrich

true_edges = make_edge_statistics(
    edges,
    positive_direction="connectivity increases with age",
    statistic_name="partial correlation",
)
observed = lens_stat(
    true_edges,
    edge_sets,
    store_running_sum=True,
)

# 只做描述性 LENS
descriptive = lens_enrich(observed, min_size=5, max_size=500)
```

## Edge-label permutation

```python
from conlens import lens_edge_permute

null_edges = lens_edge_permute(
    true_edges,
    n_permutations=10_000,
    random_state=42,
)
null_stats = (lens_stat(item, edge_sets) for item in null_edges)
fit = lens_enrich(
    observed,
    null_stats,
    min_size=5,
    max_size=500,
    family_name="age-network-pairs",
)
```

`fit.null_scores` 保存的是每个 null 的网络集合 ES，而不是逐边 permutation matrix。

::: warning 这个 null 回答的问题更窄
Edge-label permutation 检验的是“这些预定义边相对于同一 edge universe 中其他边是否异常靠近
排序两端”。它不保留共享节点、拓扑、空间和跨边协方差，也不能恢复没有提供的受试者层信息。
:::

若研究设计需要 covariate adjustment、exchangeability blocks 或复杂 repeated measures，而你只有
一列汇总 edge statistics，应把这些限制写进报告，不要把 edge-label null 称为 Freedman–Lane。
