# Python API

## 设计矩阵与逐边模型

| 函数 | 返回 | 用途 |
| --- | --- | --- |
| `make_design(...)` | `DesignMatrix` | 构建并验证 design；连续变量默认中心化 |
| `Contrast(weights, effect_size, positive_direction)` | `Contrast` | 定义一自由度 contrast 与效应量 |
| `plot_design(design, contrasts)` | `Axes[]` | 查看 design matrix 和 contrast vectors |
| `lens_glm(connectomes, design=, contrasts=, ...)` | `dict[str, EdgeStatistics]` | 计算 observed edge effects |
| `lens_fl_permute(..., n_permutations=, ...)` | iterator | 流式产生 FL null edge effects |

`effect_size` 只接受 `"partial_r"` 或 `"hedges_g"`。
`lens_glm` 返回完整审计表。恒定、全零或数值上完全拟合的边标记 `estimable=False`，metadata 的 `nonestimable_edge_ids` 列出具体边。effect/t=0、P=1 仅作占位，`lens_stat` 会拒绝将带这些标记的输入用于排序。有效边范围政策见[推断说明](/guide/inference)。

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
| `load_maps` / `NodeMaps` | 加载并验证 atlas-aligned node maps |
| `make_node_value_sets` | 从高值/低值 nodes 及其连接关系构建 sets |
| `make_node_distance_sets` | 从单一 map 上的 node-value distance 构建 sets |
| `make_profile_similarity_sets` | 从多 map node-profile similarity 构建 sets |
| `EdgeSets` | 可按字典读取的边集合，保存构建参数、来源与选择记录 |

Map-based builders 的完整签名、参数语义和选择方向见
[从 node maps 构建 edge sets](/tutorials/map-based-edge-sets)。特别注意：

- `make_node_value_sets` 在 `fraction`、`n_nodes`、`cutoff` 中只接受一个；`connect` 可为
  `"within"`、`"touching"` 或 `"between"`；
- `make_node_distance_sets` 在 `edge_fraction`、`cutoff` 中只接受一个，支持
  `keep="closest" / "farthest"` 和 `value_scale="raw" / "rank"`；
- `make_profile_similarity_sets` 在 `edge_fraction`、`cutoff` 中只接受一个，支持 Pearson、cosine
  和 Euclidean metric，以及 `map_scaling="none" / "zscore"`。

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
| `make_null_edge_statistics` | 验证外部 edge × permutation matrix，并返回可重复迭代的 lazy nulls |
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
| `NullEdgeStatistics` | 外部 null matrix 的可重复迭代 column-view 包装 |
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

`LensSetResult` 的 `n_null_zero` 和 `n_null_tail` 分别记录零 ES 数和实际检验尾部数量。`LensStabilityResult.observed_reference` 保存完整观测参照并随 JSON 导出；metadata 保存实际种子与抽样索引，详见[稳定性](/tutorials/stability)。旧稳定性文件可读取，缺少的参照字段为 `None`。
