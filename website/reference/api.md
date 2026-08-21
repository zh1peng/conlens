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
`lens_glm` 直接从三维 connectome 提取边矩阵，不会构造 subjects × edges 的长表。若某条边的
残差方差为零，该边保留在统一 edge universe 中，但返回中性统计量（effect/t 为 0、P 为 1），
且 audit 列 `estimable=False`。

## 数据与 edge sets

| 函数 | 说明 |
| --- | --- |
| `validate_connectome` | 检查 2D/3D connectome、有限值和对称性 |
| `matrix_to_edges` / `edges_to_matrix` | 矩阵与 canonical edge table 互转 |
| `validate_edge_table` / `canonicalize_edges` | 校验并固定 node order、端点和 edge ID |
| `make_network_pair_sets` | 构建 network-pair sets |
| `make_within_network_sets` | 只构建 within-network sets |
| `make_hemisphere_sets` | 按半球标签构建 sets |
| `make_custom_edge_sets` / `validate_edge_sets` | 构建或校验自定义 sets |

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

需要检查方法细节时，可直接调用低层纯函数 `rank_edges`、`compute_running_sum`、
`compute_enrichment_score`、`extract_leading_edges` 和 `adjust_pvalues`。常规分析不需要手动拼接
这些步骤。

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
| `summarize_leading_network` | 节点数、边数、密度与连通分量 |
| `compare_leading_edges` | 同一 edge universe 上比较两个 leading edge 集合 |
| `compare_lens_results` | 对兼容结果逐 set 比较 ES/NES、方向与 overlap |

所有正式结果都带 `schema_version` 和 `object_type`，支持 `to_dict()`、`save()` 与对应的
`from_dict()`、`load()`；`LeadingNetwork` 的 JSON 和 GraphML 也都可以重新载入。结果比较、null
推断和 stability 汇总会核对 node labels/order、edge→endpoint mapping 与模型 metadata，避免把
两个 atlas 中碰巧相同的 edge ID 当成同一条边。

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
生成、已经完成 joint BH 且 metadata 兼容的 `LensResult`。两个入口都逐 replicate 更新计数；
不会在内存中保留全部完整 bootstrap result。`n_jobs>1` 时同样逐个收集有序结果。

## Plotting

- `plot_connectome_heatmap`
- `plot_lens_heatmap`
- `plot_enrichment_heatmap`
- `plot_running_sum`
- `plot_null_distribution`
- `plot_enrichment`
- `plot_circos`
- `plot_leading_adjacency`
- `plot_node_participation`
- `plot_stability`
