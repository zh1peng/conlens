# 导入外部模型与置换结果

已有完整的观测边统计量和符合研究设计的外部置换统计量时，可以直接计算集合富集。外部模型负责协变量、误差结构和重采样有效性；ConLens 检查输入对齐，不会因为接收了一个矩阵就验证了零模型。

观测表需要边端点、完整边 ID 和有符号 statistic；零分布矩阵形状为 `n_edges × n_permutations`，每列是一次完整重复，使用相同统计量、方向和固定边范围。优先用边 ID 作 DataFrame 索引，让接口检查遗漏、重复和未知边并重排；NumPy 数组必须由调用者保证行顺序。

仓库的 `examples/teaching_workflow.py` 中 `advanced_analysis()` 导出内置 FL 结果、颠倒行顺序，再按 ID 导入；这只演示格式与对齐，继承原来的 FL 零模型。完整运行命令见[快速开始](/guide/quick-start)。以下参数片段供替换为自己的外部模型结果，`permutation_scheme` 是来源说明，不会执行该方案。

## 观测统计量

Observed 表每行是一条 edge，`statistic` 必须是有符号的数值。方向要写成统计意义明确的短句：

```python
from conlens import make_edge_statistics, lens_stat

observed_edges = make_edge_statistics(
    observed_table,
    positive_direction="post > pre",
    statistic_name="paired t statistic",
)

observed_stats = lens_stat(
    observed_edges,
    edge_sets,
    store_running_sum=True,
)
```

`make_edge_statistics()` 在这里检查边身份并记录 canonical edge IDs、node order、统计量方向
和 analysis signature。若 `observed_edges` 已经来自 `lens_glm()`，则不需要再调用它。

## 置换统计量矩阵

Null matrix 的形状是 `n_edges × n_permutations`：行与 `observed_edges` 是同一组 edge，列是一
次完整 null replicate。NumPy array 必须已经按 observed edge 顺序排列。

```python
from conlens import make_null_edge_statistics, lens_enrich

null_edges = make_null_edge_statistics(
    null_matrix,
    reference=observed_edges,
    permutation_scheme="within-subject sign flip",
    random_state=42,
    exchangeability_blocks_used=True,
)

null_stats = (
    lens_stat(edges, edge_sets)
    for edges in null_edges
)

fit = lens_enrich(
    observed_stats,
    null_stats,
    min_size=5,
    max_size=None,
    family_name="pre-post-network-pairs",
)
```

`NullEdgeStatistics` 不会预先生成数千张 DataFrame。迭代时，每一列以 NumPy view 的形式包装成
现有的 lazy `EdgeStatistics`，随后仍走原来的 `lens_stat()`。它可以重复迭代，因此同一批 null
effects 可以直接换一套 edge sets 再算：

```python
fit_network_pairs = lens_enrich(
    lens_stat(observed_edges, network_pair_sets),
    (lens_stat(edges, network_pair_sets) for edges in null_edges),
    min_size=5,
    family_name="network-pairs",
)

fit_sc_classes = lens_enrich(
    lens_stat(observed_edges, sc_class_sets),
    (lens_stat(edges, sc_class_sets) for edges in null_edges),
    min_size=5,
    family_name="sc-classes",
)
```

若矩阵来自 Pandas，可将 `edge_id` 放在 index。ConLens 会核对 edge universe，并按 observed
顺序重排：

```python
null_frame = null_frame.set_index("edge_id")

null_edges = make_null_edge_statistics(
    null_frame,
    reference=observed_edges,
    permutation_scheme="within-subject sign flip",
)
```

## 使用前要确认什么

ConLens 能检查 shape、finite values、edge identity，以及 null metadata 是否在迭代过程中一致；
它无法判断外部模型和 permutation 是否统计上成立。至少要保证：

- observed 与每一列 null 使用同一种 signed statistic；
- 正值始终对应 `positive_direction` 所写的方向；
- 每列包含完整且顺序一致的 edge universe；
- permutation 保留研究设计要求的 exchangeability；
- 一次 `lens_enrich()` 中比较的 observed/null 来自同一个分析定义。

300 nodes 的无向上三角共有 44,850 条边。5,000 次 `float64` permutations 约占 1.67 GiB，
通常还能接受，但默认 on-the-fly 路径更省内存。只有需要复用 null effects，或外部模型无法由
ConLens 直接产生时，才值得保留这张矩阵。
