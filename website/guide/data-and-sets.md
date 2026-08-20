---
title: 数据与 edge sets
description: ConLens 的边表、connectome 与集合定义
---

# 数据与 edge sets

## 边统计量表

最小输入包含三列：

| 列 | 含义 |
| --- | --- |
| `node1` | 第一端点 |
| `node2` | 第二端点 |
| `statistic` | 有符号边统计量 |

`validate_edge_table` 会检查重复边、缺失值、非有限统计量、对角线和有向性，并生成稳定的 `edge_id` 与 `canonical_edge_id`。

```python
from conlens import validate_edge_table

validated = validate_edge_table(
    edges,
    directed=False,
    diagonal=False,
    nan_policy="raise",
)
```

## Connectome 数组

`LensAnalysis.from_subject_connectomes` 接受形状为：

```text
(subjects, nodes, nodes)
```

的 NumPy 数组。无向连接组默认要求矩阵对称；`node_labels` 的顺序会成为 edge universe 身份的一部分。

## 网络对集合

```python
from conlens import make_network_pair_sets, matrix_to_edges

template = matrix_to_edges(connectomes[0], node_labels=labels)
node_networks = {
    "A": "DMN",
    "B": "DMN",
    "C": "VIS",
    "D": "VIS",
}
edge_sets = make_network_pair_sets(template, node_networks)
```

对于无向数据，集合名称使用排序后的 `NETWORK_A--NETWORK_B`。来自单标签 parcellation 的网络对集合会划分整个无向 edge universe。

## 自定义集合

```python
from conlens import make_custom_edge_sets

custom = make_custom_edge_sets(
    {"hypothesis": endpoint_frame},
    validated,
)
```

未知边或重复边会直接报错，而不会被静默修复。自定义集合可以重叠，但重叠会影响集合间依赖，因此解释多重检验 family 时必须保持明确。

## 集合大小过滤

`lens_enrich` 默认 `min_size=5`。太小、覆盖完整 universe 或超过 `max_size` 的集合不会获得有效 running sum。过滤状态会保留在结果中，不会静默丢失。

## 冻结 edge universe

以下内容在同一次分析及 bootstrap 重拟合中必须保持一致：

- 节点顺序；
- `directed` 与 `diagonal` 设置；
- edge ID 到 canonical edge ID 的映射；
- 集合名称及成员；
- ranking statistic、weight、score type 和过滤规则。

