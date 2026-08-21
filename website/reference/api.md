---
title: Python API
description: ConLens 公共 API 与统一 subject-level GLM
---

# Python API

公共 API 从 `conlens` 顶层导出。ConLens 当前处于开发阶段，本页描述当前实现，
不保留已经移除的 subject-level 旧入口。

## Design 与 contrasts

### `make_design`

```python
make_design(
    *,
    indicators=None,
    continuous=None,
    interactions=None,
    matrix=None,
    column_names=None,
    center_continuous=True,
    add_intercept=None,
) -> DesignMatrix
```

两种输入模式互斥：

- semantic mode：使用 `indicators`、`continuous` 和可选 `interactions`；
- raw-matrix mode：使用 `matrix` 与可选 `column_names`，输入原样保留。

Semantic mode 中连续变量默认 mean-center，indicator 不中心化，interaction 在中心化后构造；
默认加入 `intercept`。Cell-means group design 必须设置 `add_intercept=False`。

### `DesignMatrix`

| 属性/方法 | 返回内容 |
| --- | --- |
| `columns` | 最终实际列名 |
| `values` | 实际拟合矩阵的副本 |
| `frame` | 带列名的 DataFrame 副本 |
| `centering` | 每个已中心化连续变量的原始均值 |
| `condition_number` | 实际 design 的 condition number |
| `metadata()` | 完整 construction provenance 与数值诊断 |
| `take(indices)` | bootstrap 对齐的 design row 子样本，保留原始 provenance |

`DesignMatrix` 不能直接构造，只能由 `make_design()` 创建。

### `Contrast`

```python
Contrast(
    weights={"g1": 1, "control": -1},
    effect_size="hedges_g",
    positive_direction="g1 > control",
)
```

`weights` 可以是按 design 名称索引的 mapping，也可以是与 `design.columns` 等长的向量。
推荐 mapping，以便可视化和审计。`effect_size` 只能是：

- `"partial_r"`：连续变量的一自由度关联；
- `"hedges_g"`：model-adjusted signed Hedges' $g$。

Hedges' $g$ contrast 的正权重必须合计为 1，负权重必须合计为 $-1$。

## 高层分析

| API | 用途 |
| --- | --- |
| `LensAnalysis` | 从边统计量表运行 LENS，或创建 subject analysis |
| `LensAnalysis.from_subject_connectomes` | 从 `(subjects, nodes, nodes)` 创建 `SubjectLensAnalysis` |
| `SubjectLensAnalysis.glm` | 统一 GLM、多 contrast、Freedman–Lane 与联合 BH |
| `SubjectLensAnalysis.bootstrap_stability` | observed-aware full-pipeline subject bootstrap |

### `SubjectLensAnalysis.glm`

```python
analysis.glm(
    design,
    contrasts,
    *,
    n_permutations=1000,
    random_state=None,
    exchangeability_blocks=None,
    correction_family_id="default",
    **lens_options,
) -> GLMResult
```

- `design` 必须是 `make_design()` 创建的 `DesignMatrix`；
- `contrasts` 必须是非空的 `{name: Contrast}` mapping；
- `n_permutations=None` 只计算描述性 LENS，不执行推断；
- 否则每个 contrast 执行 contrast-specific Freedman–Lane；
- 所有 contrast × 有效 edge sets 的 nominal P 值共同进行一次 BH。

### `GLMResult`

```python
fit = analysis.glm(design, contrasts)
age_result = fit["age"]
summary = fit.to_frame()
fit.save("glm-results.json")
```

| 属性/方法 | 用途 |
| --- | --- |
| `contrast_names` | 保持用户定义顺序的 contrast 名称 |
| `contrasts` | `{contrast_name: LensResult}` |
| `get(name)` / `[name]` | 取得一个 contrast 的 `LensResult` |
| `to_frame()` | 合并所有 contrast × set 结果 |
| `save()` / `load()` | JSON round trip |

## 核心富集

| API | 用途 |
| --- | --- |
| `lens_enrich` | 从一张 signed edge-statistics 表运行 LENS |
| `rank_edges` | 冻结并排序完整 edge universe |
| `compute_running_sum` | 计算加权 running-sum profile |
| `compute_enrichment_score` | 从 profile 得到 signed ES |
| `extract_leading_edges` | 根据方向和极值位置提取 leading edge |

`lens_enrich` 的 subject-independent 推断只支持 edge permutation 或 provided null。
Subject-level inference 统一由 `SubjectLensAnalysis.glm` 提供。

## 数据与集合

- `validate_connectome`
- `validate_edge_table`
- `canonicalize_edges`
- `matrix_to_edges`
- `edges_to_matrix`
- `make_network_pair_sets`
- `make_within_network_sets`
- `make_hemisphere_sets`
- `make_custom_edge_sets`
- `validate_edge_sets`

## 推断工具

- `edge_permutation_null`
- `provided_null`
- `permutation_test`
- `empirical_pvalue`
- `normalize_enrichment_scores`
- `adjust_pvalues`

Subject-level reduced models 是 `analysis.glm()` 的实现细节，不再作为独立 public API 暴露。

## 结果与 leading networks

| 类型/API | 用途 |
| --- | --- |
| `LensResult` | 一个 contrast 的集合结果、完整排序边表和分析元数据 |
| `LensSetResult` | 单个 set 的 ES、NES、方向与 leading edge |
| `GLMResult` | 同一 design 和联合 BH family 的具名 contrast results |
| `LeadingNetwork` | leading-edge 节点和边，可导出 JSON/GraphML |
| `build_leading_network` | 从 `LensResult` 重建 leading network |
| `compute_node_participation` | 汇总节点参与 |
| `identify_leading_hubs` | 根据 participation 标记 hub |
| `summarize_leading_network` | 网络级摘要 |

## 稳定性

| API | 语义 |
| --- | --- |
| `SubjectLensAnalysis.bootstrap_stability` | 重抽 subjects 并用 callback 重跑完整 GLM family |
| `summarize_bootstrap_stability` | 汇总外部生成的完整、兼容 `LensResult` replicates |
| `LensStabilityResult` | set、edge、replicate 三张稳定性表与元数据 |
| `bootstrap_lens` | 低层描述性 statistic bootstrap |
| `summarize_stability` | 不按 observed significance/direction gate 的 localization summary |
| `consensus_network` | 描述性 workflow 的显式阈值 consensus |

## 绘图

`plot_design` 从顶层导出：

```python
from conlens import plot_design
axes = plot_design(design, contrasts)
```

其他绘图位于 `conlens.plotting`：

- `plot_ranked_statistics`
- `plot_running_sum`
- `plot_hit_rug`
- `plot_enrichment`
- `plot_nes`
- `plot_network_pair_heatmap`
- `plot_leading_adjacency`
- `plot_leading_connectome`
- `plot_node_participation`

源码与 docstrings：[`conlens/`](https://github.com/zh1peng/conlens/tree/main/conlens)
