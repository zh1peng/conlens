---
title: 边统计量输入
description: 从完整 edge-statistics 表运行 ConLens
---

# 教程：从边统计量开始

这个工作流适合已经在其他软件中计算好每条连接统计量的情况。

## 完整示例

```python
import pandas as pd

from conlens import (
    build_leading_network,
    lens_enrich,
    make_network_pair_sets,
    validate_edge_table,
)
from conlens.plotting import plot_enrichment, plot_nes

# 1. 每条有效边必须出现一次。
edges = pd.DataFrame({
    "node1": ["A", "A", "A", "B", "B", "C"],
    "node2": ["B", "C", "D", "C", "D", "D"],
    "statistic": [3.0, 2.0, 1.0, -0.5, -1.5, -2.5],
})

# 2. 验证 edge universe，并获得 canonical edge IDs。
validated = validate_edge_table(edges)

# 3. 根据预先定义的节点网络标签建立网络对集合。
node_networks = {
    "A": "DMN",
    "B": "DMN",
    "C": "VIS",
    "D": "VIS",
}
edge_sets = make_network_pair_sets(validated, node_networks)

# 示例只有 4 个节点，因此降低 min_size。
result = lens_enrich(
    edges,
    edge_sets,
    min_size=1,
    null_method="edge_permutation",
    n_permutations=10_000,
    random_state=42,
    positive_direction="case > control",
    correction_family_id="primary-network-pairs",
    store_running_sum=True,
)

print(result.to_frame()[
    ["set_name", "set_size", "ES", "NES", "p_value", "q_value", "direction"]
])

# 4. 查看一个集合的富集轨迹。
set_name = "DMN--VIS"
plot_enrichment(result, set_name, edge_sets[set_name])
plot_nes(result)

# 5. 构建并保存 leading-edge network。
network = build_leading_network(result, set_name)
network.save("dmn-vis-leading.graphml")
result.save("edge-statistics-result.json")
```

## 描述性分析与推断分析

如果暂时只需要检查排序与 leading edge，可以省略 `null_method`：

```python
descriptive = lens_enrich(
    edges,
    edge_sets,
    min_size=1,
    positive_direction="case > control",
)
```

此时 `NES`、`p_value` 和 `q_value` 为 `None`。不要把描述性 ES 当作经过推断检验的结果。

## 何时不应使用 edge permutation？

若原始数据包含受试者级 connectome，edge permutation 通常不是最贴合依赖结构的方案。优先转入 [受试者两组分析](/tutorials/subject-two-group) 或 [含协变量的 GLM](/tutorials/glm)。

