---
title: 快速开始
description: 使用边统计量完成一次最小 ConLens 分析
---

# 五分钟快速开始

下面从一张边统计量表出发，完成验证、集合定义、富集推断和结果查看。

## 1. 准备完整边表

```python
import pandas as pd

edges = pd.DataFrame({
    "node1": ["A", "A", "A", "B", "B", "C"],
    "node2": ["B", "C", "D", "C", "D", "D"],
    "statistic": [3.0, 2.0, 1.0, -0.5, -1.5, -2.5],
})
```

`statistic` 必须是有限的有符号数值。排序始终使用原始符号，而不是绝对值。

## 2. 验证并定义 edge set

```python
from conlens import validate_edge_table

validated = validate_edge_table(edges)
edge_sets = {
    "example": set(validated.loc[[0, 1, 4], "edge_id"]),
}
```

先验证一次可以获得稳定的 canonical `edge_id`。不要从端点字符串自行猜测 ID。

## 3. 运行富集分析

```python
from conlens import lens_enrich

result = lens_enrich(
    edges,
    edge_sets,
    min_size=1,
    null_method="edge_permutation",
    n_permutations=10_000,
    random_state=42,
    positive_direction="case > control",
    store_running_sum=True,
)
```

`positive_direction` 是结果解释的一部分。正富集表示集合边集中在“case > control”的一端；负富集表示集中在相反一端。

::: warning 关于 edge permutation
该零模型会打乱统计量与 edge ID 的对应关系，因此不保留共享节点、拓扑、空间结构或边间协方差。若有受试者级数据并且设计允许，优先考虑 label permutation 或 Freedman–Lane。
:::

## 4. 查看结果

```python
summary = result.to_frame()
print(summary[["set_name", "ES", "NES", "p_value", "q_value", "direction"]])

item = result.get("example")
print(item.leading_edge_ids)
```

## 5. 绘制 running sum

```python
from conlens.plotting import plot_enrichment

plot_enrichment(result, "example", edge_sets["example"])
```

## 6. 保存可追溯结果

```python
result.save("conlens-result.json")
```

JSON 中同时保存集合结果、完整排序边表和关键元数据，包括排序规则、零模型、随机种子、方向标签和软件版本。

下一步可阅读 [数据与 edge sets](/guide/data-and-sets) 或进入 [边统计量完整教程](/tutorials/edge-statistics)。

