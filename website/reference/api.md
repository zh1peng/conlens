# Python API

## Design 与 edge-wise model

| 函数 | 返回 | 用途 |
| --- | --- | --- |
| `make_design(...)` | `DesignMatrix` | 构建并验证 design；连续变量默认中心化 |
| `Contrast(weights, effect_size, positive_direction)` | `Contrast` | 定义一自由度 contrast 与效应量 |
| `plot_design(design, contrasts)` | `Axes[]` | 查看 design matrix 和 contrast vectors |
| `lens_glm(connectomes, design=, contrasts=, ...)` | `dict[str, EdgeStatistics]` | 计算 observed edge effects |
| `lens_fl_permute(..., n_permutations=, ...)` | iterator | 流式产生 FL null edge effects |

`effect_size` 只接受 `"partial_r"` 或 `"hedges_g"`。

## LENS 与推断

```python
lens_stat(
    edge_statistics,
    edge_sets,
    weight=1.0,
    score_type="standard",
    store_running_sum=False,
)

lens_enrich(
    observed_lens_stat,
    null_lens_stats=None,
    min_size=5,
    max_size=None,
    family_name="default",
)
```

| 函数 | 说明 |
| --- | --- |
| `make_edge_statistics` | 验证外部 signed edge-statistic table |
| `lens_edge_permute` | 流式产生 edge-label null statistics |
| `lens_stat` | 对 observed/null 使用同一定义计算 ES 和 leading edge |
| `lens_enrich` | set-size filter、NES、经验 P、joint BH，并保留 set-level null ES |

`lens_enrich` 不接受 connectomes、raw edge statistics、`n_permutations` 或 `random_state`；这些属于
上游模型或 permutation generator。

## 结果与 leading network

| 对象/函数 | 说明 |
| --- | --- |
| `EdgeStatistics` | 边表 + 模型与方向 metadata |
| `LensStatResult` | 确定性 LENS statistics |
| `LensResult` | 单 contrast 的 observed + null inference |
| `GLMResult` | 多 contrast 的联合 BH family |
| `build_leading_network` | 从一个 set 的 leading edges 建图 |
| `compute_node_participation` | degree/strength 和有向版本 |
| `identify_leading_hubs` | 显式阈值或 top-n 节点摘要 |

## Stability

```python
lens_bootstrap(
    connectomes,
    edge_sets,
    design=design,
    contrasts=contrasts,
    n_bootstraps=1000,
    n_permutations=10000,
    strata=None,
    exchangeability_blocks=None,
    random_state=42,
    n_jobs=1,
)
```

返回 `dict[str, LensStabilityResult]`。低层 `summarize_stability(observed, replicates)` 可汇总外部
生成、已经完成 joint BH 且 metadata 兼容的 `LensResult`。

## Plotting

- `plot_connectome_heatmap`
- `plot_enrichment_heatmap`
- `plot_running_sum`
- `plot_null_distribution`
- `plot_enrichment`
- `plot_circos`
- `plot_leading_adjacency`
- `plot_node_participation`
- `plot_stability`
