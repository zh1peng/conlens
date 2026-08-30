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

ConLens 还会把 node labels/order、directed/diagonal 设定和实际端点写入 identity hash。
null inference、结果比较和 stability 汇总都会核对它，而不是只比较看起来相同的 `0--1`。

## Network-pair sets

```python
from conlens import make_network_pair_sets

edge_sets = make_network_pair_sets(edges, node_networks)
```

无向集合名形如 `DMN--FPN`，有向集合名形如 `DMN->FPN`。也可用
`make_custom_edge_sets`、`make_within_network_sets` 或 `make_hemisphere_sets`。

## 从 node maps 构建 edge sets

`load_maps` 从本地目录或远程 ConLens resource repository 读取 node maps，并验证
resource checksum、atlas node count 和 canonical node order。GitHub branch/tag 会在内部解析为
具体 commit SHA 并写入 provenance，不需要用户额外提供 client 或 revision 参数。返回的
`NodeMaps` 可按 map 名称索引：

```python
from conlens import load_maps

pet = load_maps(
    "pet/receptor-react",
    atlas="schaefer200-7net",
    source=r"E:\03_tools\conlens-resources",
)

d1 = pet["D1"]
```

三种 builder 对应三种不同问题。

```python
from conlens import (
    make_node_distance_sets,
    make_node_value_sets,
    make_profile_similarity_sets,
)

# 1. annotation 最高的 20% nodes 之间的 edges
d1_high = make_node_value_sets(
    edges,
    pet,
    map_names=["D1"],
    keep="highest",
    fraction=0.20,
    connect="within",
)

# 2. 单个 map 上 node values 最接近的 10% edges
d1_close = make_node_distance_sets(
    edges,
    pet["D1"],
    keep="closest",
    edge_fraction=0.10,
    value_scale="rank",
)

# 3. 多个 maps 构成的 node profiles 最相似的 5% edges
pet_profile = make_profile_similarity_sets(
    edges,
    pet,
    map_names=pet.names,
    metric="pearson",
    keep="most_similar",
    edge_fraction=0.05,
)
```

`fraction` 选择 nodes，`edge_fraction` 直接选择 edges。Fraction-based selection 使用向上取整；
相同数值按 atlas node order 或 canonical edge ID 确定性打破 ties。所有 builder 都只在输入的
实际 edge universe 上选取成员，因此不会为稀疏 structural connectome 补造不存在的 edges。

返回的 `EdgeSets` 实现普通 Mapping 接口，可直接传给 `lens_stat`，并额外保存构建参数、map
版本、checksum、edge-universe hash 和选择 audit：

```python
observed = lens_stat(edge_statistics, d1_high)
d1_high.save("d1-high.edge-sets.json")
```

`lens_stat` 会自动核对 `EdgeSets` 的 edge universe 与分析统计量；若同一个集合被误用于另一个
universe，会直接报错。该检查和 canonical tie-breaking 都在内部完成，不增加调用参数。

Set size filter 不在 `lens_stat` 中执行。Observed 和每个 null 先按完全相同的集合定义计算 ES，
再由 `lens_enrich(min_size=..., max_size=...)` 决定哪些集合进入推断和 BH family。
