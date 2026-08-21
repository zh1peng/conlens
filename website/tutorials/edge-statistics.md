---
title: 只有 edge statistics 时
description: 从完整边统计量表运行描述性 LENS、edge permutation 或 provided null
---

# 只有 edge statistics 时

这条路线适合已经在其他软件中计算好每条边统计量、但拿不到个体 connectome 的情况。
LENS 仍然可以计算 running sum、ES 和 leading edge。区别在于：ConLens 已经无法重跑
subject-level model。

## 准备完整边表

```python
import pandas as pd

from conlens import (
    build_leading_network,
    lens_enrich,
    make_network_pair_sets,
    validate_edge_table,
)
from conlens.plotting import plot_enrichment, plot_nes

edges = pd.DataFrame({
    "node1": ["A", "A", "A", "B", "B", "C"],
    "node2": ["B", "C", "D", "C", "D", "D"],
    "statistic": [3.0, 2.0, 1.0, -0.5, -1.5, -2.5],
})

validated = validate_edge_table(edges)
node_networks = {"A": "DMN", "B": "DMN", "C": "VIS", "D": "VIS"}
edge_sets = make_network_pair_sets(validated, node_networks)
```

每条有效边必须恰好出现一次。`statistic` 要有符号，且所有值共享同一个方向定义。

## 先看描述性结果

```python
result = lens_enrich(
    edges,
    edge_sets,
    min_size=1,  # 这里只是 4-node toy example
    positive_direction="case > control",
    store_running_sum=True,
)

print(result.to_frame()[["set_name", "ES", "direction"]])
```

没有 `null_method` 时，`NES`、`p_value` 和 `q_value` 为 `None`。ES 和 leading edge 仍可
用于描述排序，但没有推断性 P 值。

## Edge permutation

```python
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
```

每次 permutation 都把 observed statistics 随机分配给 edge IDs，再计算全部 sets 的 ES。
它保留集合大小和重叠，不保留 connectome 的 subject-level dependence、共享节点、网络拓扑
或空间结构。

这不是 Freedman–Lane 的简化版。两者检验的零假设不同。有个体 connectome 时，回到
[从个体数据到 LENS](/tutorials/design-and-contrasts)，让 permutation 重跑 edge model。

## Provided null

外部流程如果已经生成了符合研究设计的 null，可以传给
`lens_enrich(..., null_method="provided_null")`。支持三种输入：

- 每个 set 的 null ES；
- 每个 replicate 的完整 edge-statistic vector；
- 每个 replicate 的完整 rank vector，限 `weight=0`。

Observed 与 null 必须使用同一个 edge order、edge sets 和 `positive_direction`。保存外部
null 的生成代码和参数；单凭 null matrix，ConLens 无法判断它是否保留了正确的
exchangeability 或 dependence structure。

## 查看和保存结果

```python
set_name = "DMN--VIS"
plot_enrichment(result, set_name, edge_sets[set_name])
plot_nes(result)

network = build_leading_network(result, set_name)
network.save("dmn-vis-leading.graphml")
result.save("edge-statistics-result.json")
```

若没有可信的 null，停在描述性结果是合理的。不要把 ES 或 leading edge 写成“经过
显著性检验的边”，也不要把 edge permutation 的 P 值解释成 subject-level model 的结果。
