# 根据脑区注释构建边集合

ConLens 自 2.2.0 起提供三种 map-based builder。它们回答的是三个不同问题：

| 研究问题 | API | 先选择什么 |
| --- | --- | --- |
| 哪些边连接高值或低值节点？ | `make_node_value_sets` | 节点 |
| 哪些边的两个端点在单个 map 上最接近或相差最远？ | `make_node_distance_sets` | 边 |
| 哪些边的两个端点具有相似或不同的多 map profile？ | `make_profile_similarity_sets` | 边 |

PET abundance 的 top-10% / top-20% 定义属于第一种。后两种方法并不先把节点分成“选中”和
“未选中”，而是直接给 edge universe 中的每条边计算分数。

::: tip 先区分两个 fraction
`fraction` 是要保留的**节点比例**；`edge_fraction` 是要保留的**候选边比例**。两者不能互换。
:::

## 共同准备：maps 与 edge universe

### 加载 atlas-aligned maps

下面用 REACT-inspired PET resource 和 Schaefer200 演示：

```python
from conlens import load_maps

pet_maps = load_maps(
    "pet/receptor-react",
    atlas="schaefer200-7net",
    source="./conlens-resources",
)
```

`source` 可以是本地 resource repository，也可以是兼容的 HTTP(S) 地址。省略它时使用 ConLens
默认 resource repository；使用 GitHub branch 或 tag 时，loader 会先解析到具体 commit SHA，
再把它写入 provenance。

加载过程会验证 resource manifest、文件 checksum、atlas 节点数与节点顺序，以及 map 名称和顺序。
返回的 `NodeMaps` 包含：

```python
pet_maps.data       # index 是 node_id，columns 是 map 名称
pet_maps.nodes      # 与 data 顺序一致的 atlas 节点表
pet_maps.map_info   # 每个 map 的元信息
pet_maps.info       # resource、atlas、checksum、source 等 provenance
pet_maps.names      # map 名称列表
pet_maps["D1"]      # 名为 D1 的 pandas Series
```

loader 只读取并校验 resource 中已经准备好的数值，不会再次 z-score 或 0–1 scale。是否以及如何
预处理 PET maps，应由 resource 的方法与 provenance 说明，而不是由 edge-set builder 隐式决定。

本例 `pet/receptor-react` 的 resource 定义为：从冻结的 43 条 neuromaps annotation inventory
开始，把同一数据集的 MNI152 与 fsaverage 表示视为重复空间表示并优先保留 fsaverage；将
MNI-only maps 投影到 fsaverage 10k；把 vertex 负值截断为 0；分区到目标 atlas；在每个 atlas
resolution 内把每张独立研究图分别 min-max scale 到 [0, 1]；最后按报告的研究样本量，在同一
biological target 内加权平均。筛选后共有 37 张独立研究图，聚合为 19 个 receptor / transporter
targets，聚合后不再做第二次 scaling。这个流程属于 resource 的科学定义，并不是
`make_node_value_sets` 的隐式行为。

### 建立候选边空间

所有 builder 都只从传入的 `edges` 中选择成员。若研究问题允许 atlas 中任意两个节点成边，可建立
完整无向、无自连接的 edge universe：

```python
import numpy as np
from conlens import matrix_to_edges

node_ids = pet_maps.data.index.tolist()
n_nodes = len(node_ids)

edges = matrix_to_edges(
    np.zeros((n_nodes, n_nodes)),
    node_labels=node_ids,
    directed=False,
    diagonal=False,
)
```

这里的零矩阵只用于定义“哪些边可能进入集合”，数值 0 不参与 map-based selection。对于完整无向
图，候选边数为 $N(N-1)/2$：

| Atlas | 节点数 | 候选边数 |
| --- | ---: | ---: |
| DK68 | 68 | 2,278 |
| Schaefer100 | 100 | 4,950 |
| Schaefer200 | 200 | 19,900 |
| Schaefer300 | 300 | 44,850 |

也可以传入稀疏 structural connectome 或其他预先限定的 edge table。此时 builder 只会保留实际存在
的边，不会把不存在的边补成完整图。`edges` 中至少要有 `node1`、`node2` 和唯一的 `edge_id`，且
所有端点必须出现在 maps 的 node index 中。

## 方法一：从高值或低值节点构建集合

### PET top-20% 完整例子

```python
from conlens import make_node_value_sets

pet_top20 = make_node_value_sets(
    edges,
    pet_maps,
    map_names=pet_maps.names,
    keep="highest",
    fraction=0.20,
    connect="within",
)
```

一次调用会对每个 PET map 分别排序、选择节点并创建一个 edge set。因此 19 个 maps 会返回 19 个
集合；不同集合可能重叠，检验结果也可能相关，而不是合并为一个集合：

```python
pet_top20.names
pet_top20["D1"]       # D1 集合中的 edge IDs
len(pet_top20["D1"])  # Schaefer200 完整无向 universe 中为 780
```

Top-10% 只需修改节点比例：

```python
pet_top10 = make_node_value_sets(
    edges,
    pet_maps,
    keep="highest",
    fraction=0.10,
    connect="within",
)
```

### 如何选择节点

`keep` 决定排序方向：

- `"highest"`：选择值最高的节点；
- `"lowest"`：选择值最低的节点。

必须在以下三种选法中提供且只提供一种：

| 参数 | 含义 | 例子 |
| --- | --- | --- |
| `fraction` | 保留有效节点的一定比例，数量用 `ceil` 向上取整 | `fraction=0.20` |
| `n_nodes` | 保留固定数量的节点 | `n_nodes=40` |
| `cutoff` | 按注释值的指定阈值选择脑区 | `cutoff=0.75` |

使用 `cutoff` 时，`keep="highest"` 保留 $x_i\ge c$，`keep="lowest"` 保留
$x_i\le c$。Fraction 或固定数量在边界出现并列时，按 map 中固定的 atlas node order
按预先固定的节点顺序选择；cutoff 则会保留所有满足阈值的节点，因此可能保留整个并列组。

`map_names` 控制要处理哪些 columns；省略时处理全部 maps。`name_prefix="pet-top20"` 可把返回的
集合命名为 `pet-top20:D1` 等，适合同时保存多个定义。

`missing="raise"` 是默认值，遇到缺失 map value 会报错。`missing="omit"` 会把缺失节点排除在
排序与 node selection 之外；`fraction` 的分母也只计算有效节点。要注意，这不会从 edge universe
中删除该节点：在 `within` 定义中，缺失节点相关边自然不会入选；在 `touching` 或 `between`
定义中，selected-to-missing edge 仍可能满足端点规则。若研究设计要求彻底排除缺失节点，应先过滤
edge universe。

### `connect` 到底控制什么

先令 $S$ 为选中的节点集合。`connect` 决定一条候选边的端点与 $S$ 具有何种关系时才进入
edge set：

| `connect` | 端点规则 | 直观含义 |
| --- | --- | --- |
| `"within"` | 两个端点都在 $S$ | 选中节点内部的边 |
| `"touching"` | 至少一个端点在 $S$ | 所有接触选中节点的边 |
| `"between"` | 恰好一个端点在 $S$ | 选中与未选中节点之间的边 |

例如节点为 `A, B, C, D`，选中的节点为 `A, B`：

| 边 | `within` | `touching` | `between` |
| --- | :---: | :---: | :---: |
| A–B | ✓ | ✓ |  |
| A–C、A–D、B–C、B–D |  | ✓ | ✓ |
| C–D |  |  |  |

因此 `touching` 是 `within` 与 `between` 的并集。对于 $N$ 个节点、选中 $k$ 个节点的
完整无向、无自连接 universe：

$$
|E_{within}|=\frac{k(k-1)}{2},\qquad
|E_{between}|=k(N-k)
$$

$$
|E_{touching}|=\frac{N(N-1)}{2}-\frac{(N-k)(N-k-1)}{2}
$$

这些公式只适用于完整无向 universe。输入是稀疏、有向或包含 diagonal 的 edge table 时，应直接
检查返回集合的长度，而不要套用公式。

PET abundance 通常使用 `connect="within"`，因为问题是“高 abundance parcels 彼此之间的
连接是否富集”。如果问题是“所有与高 abundance parcels 相连的连接”，则 `touching` 更合适；
如果只关心其向其余 cortex 延伸的边，则使用 `between`。

### PET top-10% / top-20% 的集合大小

在完整无向、无自连接 universe 中，`fraction` 使用 `ceil`，`connect="within"` 时：

| Atlas | 阈值 | 选中节点数 | 每个 map 的边数 |
| --- | --- | ---: | ---: |
| DK68 | top 10% | 7 | 21 |
| DK68 | top 20% | 14 | 91 |
| Schaefer100 | top 10% | 10 | 45 |
| Schaefer100 | top 20% | 20 | 190 |
| Schaefer200 | top 10% | 20 | 190 |
| Schaefer200 | top 20% | 40 | 780 |
| Schaefer300 | top 10% | 30 | 435 |
| Schaefer300 | top 20% | 60 | 1,770 |

建议在正式分析脚本中保留以下可执行检查，核对节点数、边数与集合成员：

```python
assert len(pet_maps.names) == 19
assert len(edges) == 19_900
assert pet_top20.info["sets"]["D1"]["selected_nodes"] == 40
assert len(pet_top20["D1"]) == 780

# 用端点而不是 edge ID 字符串比较新旧定义
d1_endpoints = edges.loc[
    edges["edge_id"].isin(pet_top20["D1"]),
    ["node1", "node2"],
]
```

最后两个断言只适用于 Schaefer200、top-20%、完整无向且无 diagonal 的本例。若输入的是稀疏
universe，节点数仍为 40，但实际入选边数可能小于 780。

## 方法二：按单个 map 上的 node distance 选边

有时问题不是“哪些节点的值高”，而是“一条边两端的 annotation 是否接近”。对于单个 scalar map，
ConLens 为每条候选边计算绝对差：

$$
d_{ij}=|x_i-x_j|
$$

例如选择 D1 abundance 最接近的 10% edges：

```python
from conlens import make_node_distance_sets

d1_close = make_node_distance_sets(
    edges,
    pet_maps["D1"],
    keep="closest",
    edge_fraction=0.10,
    value_scale="raw",
    name="D1:closest-10pct",
)
```

选择相差最远的 edges：

```python
d1_far = make_node_distance_sets(
    edges,
    pet_maps["D1"],
    keep="farthest",
    edge_fraction=0.10,
    value_scale="rank",
)
```

关键参数为：

| 参数 | 可选值与含义 |
| --- | --- |
| `keep` | `"closest"` 选择较小距离；`"farthest"` 选择较大距离 |
| `edge_fraction` | 保留候选边的一定比例，数量用 `ceil` |
| `cutoff` | 按距离的指定阈值选择边；与 `edge_fraction` 二选一 |
| `value_scale` | `"raw"` 用原始值；`"rank"` 先转成 percentile rank 再计算差值 |
| `missing` | `"raise"` 遇到缺失值报错；`"omit"` 排除端点含缺失值的边 |
| `name` | 覆盖默认集合名 `<map_name>:<keep>` |

用 cutoff 时，`closest` 保留 $d_{ij}\le c$，`farthest` 保留 $d_{ij}\ge c$。
`value_scale="rank"` 关注节点在 map 中的相对次序，减少原始单位和极端值的影响，但它改变了
科学定义，不应仅为了得到更多显著结果而切换。

这种方法直接给边打分，所以没有 `connect` 参数。它可以选出两个中等值但彼此非常接近的节点之间
的边，而这条边不会出现在 top-abundance 的 `within` 集合中。

## 方法三：按多 map node profile 的 similarity 选边

当每个节点都有多个 annotation 时，可把节点 $i$ 表示为 profile
$\mathbf{x}_i=(x_{i1},\ldots,x_{im})$，再比较每条边两端的 profile：

```python
from conlens import make_profile_similarity_sets

pet_profile = make_profile_similarity_sets(
    edges,
    pet_maps,
    map_names=pet_maps.names,
    metric="pearson",
    keep="most_similar",
    edge_fraction=0.05,
    map_scaling="none",
    name="PET-profile:most-similar-5pct",
)
```

这会返回一个集合：profile 最相似的 5% 候选边。它与方法一不同，不会为每个 PET receptor
分别返回一个集合，因为所有指定 maps 共同定义了一个 node profile。

### metric 与方向

| `metric` | 分数 | `most_similar` | `least_similar` |
| --- | --- | --- | --- |
| `"pearson"` | profile Pearson $r$ | 分数大；cutoff 时 $r\ge c$ | 分数小；$r\le c$ |
| `"cosine"` | cosine similarity | 分数大；$s\ge c$ | 分数小；$s\le c$ |
| `"euclidean"` | Euclidean distance | 距离小；$d\le c$ | 距离大；$d\ge c$ |

必须在 `edge_fraction` 与 `cutoff` 中提供且只提供一个。`pearson` 和 `cosine` cutoff 必须位于
[-1, 1]；Euclidean cutoff 必须非负。Profile similarity 至少需要两个 maps，且目前不接受缺失值。

`map_scaling="zscore"` 会在比较节点 profile 前，把每个 map 在节点间分别标准化。这样每个 map
以标准差单位参与距离或 similarity，适合不同量纲的 annotations；`"none"` 保留 resource 中的
原始相对尺度。若 maps 已经按研究目的统一缩放，通常先使用 `"none"`。

::: warning 不是 annotation–annotation correlation
这里计算的是**每条候选边两端的 node–node profile similarity**，所以结果仍是 edge set。
如果把两个 annotation columns 在所有节点上做 correlation，得到的是 map–map 关系，不会直接
定义 connectome 的 node–node edges。
:::

## 选择哪一种方法

| 如果研究假设是…… | 推荐方法 |
| --- | --- |
| 高（或低）annotation 节点内部/周边的连接形成一个集合 | node values + `connect` |
| 单个 annotation 值相似（homophily）或形成强梯度的节点对构成集合 | node distance |
| 多个 annotations 的整体组成相似或差异较大的节点对构成集合 | profile similarity |

这三种定义没有默认的优劣。应在查看 LENS 结果之前，根据研究假设预先指定方法、方向、比例或阈值、
scaling，以及 edge universe。对多个阈值做敏感性分析时，应把每个定义清楚命名并报告完整 family。

## 返回值、audit 与 provenance

三个 builder 都返回 `EdgeSets`。它实现 Mapping 接口，因此可像字典一样使用，也可直接传给
`lens_stat`：

```python
from conlens import lens_stat

observed = lens_stat(edge_statistics, pet_top20)

pet_top20.members   # set name -> frozenset(edge IDs)
pet_top20.audit     # 完整选择记录
pet_top20.info      # 方法、参数、hash 与 provenance

pet_top20.save("pet-top20.edge-sets.json")
```

`make_node_value_sets` 的 audit 每行是一个 node，包含 value、rank 和 selected；distance 与 profile
similarity 的 audit 每行是一条候选 edge，包含端点、distance 或 metric value、rank 和 selected。

`info` 会记录 builder 方法、map 内容 hash、selection 参数、tie-breaking、edge-universe identity
和 resource provenance。`lens_stat` 会核对该 universe 与输入的 edge statistics，防止把一个
atlas 或稀疏度下建立的集合误用于另一个 universe。

若 `edges` 由 `matrix_to_edges` 创建，默认 edge IDs 是基于 node order 的 canonical IDs，例如
`4--38`；端点标签仍保存在 `node1` 和 `node2` 中。若输入 edge table 已有自定义 `edge_id`，ConLens
会保留它并另外维护 canonical endpoint identity。因此不要通过解析 edge ID 猜端点，应始终使用
edge table 的端点 columns。

## 参数速查

### `load_maps`

```python
load_maps(
    resource,
    *,
    atlas,
    source=DEFAULT_MAP_SOURCE,
    cache_dir=None,
)
```

`resource` 是 registry 中的 resource ID，`atlas` 是目标 atlas ID。`source` 指向本地或远程
resource repository；`cache_dir` 可覆盖远程文件的默认 checksum-keyed cache 目录，本地读取
不需要 cache。

### `make_node_value_sets`

```python
make_node_value_sets(
    edges,
    maps,
    *,
    map_names=None,
    keep="highest",
    fraction=None,
    n_nodes=None,
    cutoff=None,
    connect="within",
    missing="raise",
    name_prefix=None,
)
```

### `make_node_distance_sets`

```python
make_node_distance_sets(
    edges,
    node_map,
    *,
    keep="closest",
    edge_fraction=None,
    cutoff=None,
    value_scale="raw",
    missing="raise",
    name=None,
)
```

### `make_profile_similarity_sets`

```python
make_profile_similarity_sets(
    edges,
    maps,
    *,
    map_names=None,
    metric="pearson",
    keep="most_similar",
    edge_fraction=None,
    cutoff=None,
    map_scaling="none",
    name=None,
)
```

所有 fraction-based selection 都使用 `ceil`。Node selection 的 tie 按 map node order 打破；
edge selection 的 tie 按 canonical edge ID 打破。这些规则和 provenance 都由 API 内部保存，
不需要增加额外参数。

## 集合大小与推断

Schaefer200 的最高 20% 节点有 40 个，它们之间的完整无向连接有 780 条。`max_size=None`（默认）不会因此过滤该集合；若设为 500，则集合状态为 filtered，不能解释为未显著。不同注释分别构建的集合可能重叠，检验结果也可能相关。D1 高注释脑区间的富集不证明 D1 直接导致了连接变化。

无需下载 PET 的运行检查见 `examples/teaching_workflow.py` 的模拟 annotation：8 个节点选择最高一半，产生 6 条边并核对进入检验。该注释是模拟数值，不能用于分子科学结论。
