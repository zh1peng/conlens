# 准备连接矩阵、表型与边集合

先确定受试者和图谱顺序，再建立设计矩阵与边集合。样本量相等不能证明连接矩阵第 i 行与表型第 i 行属于同一人。

## 以受试者 ID 对齐

准备影像清单 `manifest.csv`（`subject_id,matrix_path,included,exclusion_reason`）、表型表 `phenotypes.csv`（`subject_id` 与模型变量），以及具有固定顺序的图谱节点表。`included` 必须是明确的布尔值；排除受试者要记录理由。ID 在读取时按字符串保存，避免丢失前导零。

以下为仓库中经测试的对齐函数；从影像清单的同一顺序加载矩阵并选择表型，不分别排序后只比较行数。

```python
import numpy as np
import pandas as pd
```

<<< ../../examples/align_subjects.py#alignment

```python
from conlens import make_design

manifest = pd.read_csv("manifest.csv", dtype={"subject_id": str})
phenotypes = pd.read_csv("phenotypes.csv", dtype={"subject_id": str})
nodes = pd.read_csv("atlas_nodes.csv", dtype={"node_id": str})
node_labels = nodes["node_id"].tolist()
connectomes, aligned, exclusions, unused = align_subjects(manifest, phenotypes, node_labels)
if aligned[["age", "motion"]].isna().any().any():
    raise ValueError("resolve missing model variables before analysis")
design = make_design(continuous={"age": aligned["age"], "motion": aligned["motion"]})
aligned[["subject_id"]].to_csv("analysis-subject-order.csv", index=False)
exclusions.to_csv("analysis-exclusions.csv", index=False)
unused.to_csv("unused-phenotype-rows.csv", index=False)
```

矩阵文件还必须来自相同图谱、相同节点顺序。数组自身无法识别标签语义错配，节点数检查不能替代上游图谱核对。不要把缺失或测量失败编码成零；恒定或退化边的政策见[推断说明](/guide/inference)。

## 最小分析记录

保存最终去标识化受试者顺序或其指纹、排除理由、图谱节点顺序和集合定义；记录模型变量、对比、中心化、固定边范围、重采样分组及种子、评分和大小限制、检验家族、源码提交和软件版本。结果 metadata 已包含设计与连接数据指纹、节点和边集合身份、数值依赖版本及推断设置；这些指纹用于复核输入一致性，不能证明受试者配对本身正确。源码提交请在分析时另存 `git rev-parse HEAD`，有本地修改时一并保存修改内容。

## 连接矩阵

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

## 网络内与网络间集合

```python
from conlens import make_network_pair_sets

edge_sets = make_network_pair_sets(edges, node_networks)
```

无向集合名形如 `DMN--FPN`，有向集合名形如 `DMN->FPN`。也可用
`make_custom_edge_sets`、`make_within_network_sets` 或 `make_hemisphere_sets`。

## 根据脑区注释构建边集合

完整的参数解释、`connect="within" / "touching" / "between"` 图解、PET top-10% / top-20%
实例，以及 node distance 和 profile similarity 两种方法，见
[从 node maps 构建 edge sets](/tutorials/map-based-edge-sets)。

`load_maps` 从本地目录或远程 ConLens resource repository 读取 node maps，并验证
resource checksum、atlas node count 和 canonical node order。GitHub branch/tag 会在内部解析为
具体 commit SHA 并写入 provenance，不需要用户额外提供 client 或 revision 参数。返回的
`NodeMaps` 可按 map 名称索引：

```python
from conlens import load_maps

pet = load_maps(
    "pet/receptor-react",
    atlas="schaefer200-7net",
    source="./conlens-resources",
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
