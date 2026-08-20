---
title: Python API
description: ConLens 稳定顶层 API 导航
---

# Python API

稳定的公共 API 从 `conlens` 顶层导出。下面按任务列出入口；完整参数与类型以源码 docstring 为准。

## 高层分析

| API | 用途 |
| --- | --- |
| `LensAnalysis` | 从边表运行分析，或创建 subject-level analysis |
| `SubjectLensAnalysis.two_group` | 两组统计与 label permutation |
| `SubjectLensAnalysis.phenotype` | 单表型统计与 label permutation |
| `SubjectLensAnalysis.glm` | tested + nuisance design 与 Freedman–Lane |
| `SubjectLensAnalysis.bootstrap_stability` | observed-aware full-pipeline subject bootstrap |

## 核心富集

| API | 用途 |
| --- | --- |
| `lens_enrich` | 运行描述性 LENS 与可选显式推断 |
| `rank_edges` | 冻结并排序完整 edge universe |
| `compute_running_sum` | 计算加权 running-sum profile |
| `compute_enrichment_score` | 从 profile 得到 signed ES |
| `extract_leading_edges` | 根据方向和极值位置提取 leading edge |

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

## 推断

- `edge_permutation_null`
- `label_permutation_null`
- `freedman_lane_null`
- `provided_null`
- `permutation_test`
- `empirical_pvalue`
- `normalize_enrichment_scores`
- `adjust_pvalues`

## 结果与 leading networks

| 类型/API | 用途 |
| --- | --- |
| `LensResult` | 集合结果、完整排序边表和分析元数据 |
| `LensSetResult` | 单个 set 的 ES、NES、方向与 leading edge |
| `LeadingNetwork` | leading-edge 节点和边，可导出 JSON/GraphML |
| `build_leading_network` | 从 `LensResult` 重建 leading network |
| `compute_node_participation` | 汇总节点在 leading network 中的参与 |
| `identify_leading_hubs` | 根据 participation 标记 hub |
| `summarize_leading_network` | 网络级摘要 |

## 稳定性

| API | 语义 |
| --- | --- |
| `summarize_bootstrap_stability` | 汇总外部生成的完整、兼容 `LensResult` replicates |
| `LensStabilityResult` | `set_summary`、`edge_summary`、`replicate_summary` 与元数据 |
| `bootstrap_lens` | 低层描述性 statistic bootstrap |
| `summarize_stability` | 不按 observed significance/direction gate 的 localization summary |
| `consensus_network` | 旧描述性 workflow 的显式阈值 consensus |

## 绘图

绘图函数位于 `conlens.plotting`：

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

