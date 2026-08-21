# 数据与 edge sets

## Connectomes

`lens_glm` 接受 `(subjects, nodes, nodes)` 数组。无向矩阵必须对称；默认排除对角线。
`node_labels` 决定稳定的节点顺序和 canonical edge IDs，例如 `0--1`。外部表也至少需要
`node1`、`node2`、`statistic` 三列。

```python
from conlens import matrix_to_edges

edges = matrix_to_edges(connectomes.mean(axis=0), node_labels)
```

若自定义 `edge_id`，ConLens 会保留它，同时记录 `edge_id → canonical_edge_id` 映射；这能防止
bootstrap 或 null 结果把同名边错误地映射到不同端点。

## Network-pair sets

```python
from conlens import make_network_pair_sets

edge_sets = make_network_pair_sets(edges, node_networks)
```

无向集合名形如 `DMN--FPN`，有向集合名形如 `DMN->FPN`。也可用
`make_custom_edge_sets`、`make_within_network_sets` 或 `make_hemisphere_sets`。

Set size filter 不在 `lens_stat` 中执行。Observed 和每个 null 先按完全相同的集合定义计算 ES，
再由 `lens_enrich(min_size=..., max_size=...)` 决定哪些集合进入推断和 BH family。
