---
title: 结果与 leading edge
description: LensResult、leading-edge network 与序列化
---

# 结果与 leading edge

## LensResult

`lens_enrich` 返回一个 `LensResult`。`analysis.glm()` 返回 `GLMResult`，其中每个具名
contrast 对应一个 `LensResult`。

```python
result = glm_results["g1_vs_control"]
frame = result.to_frame()
item = result.get("DMN--VIS")
```

常用集合级字段包括：

- `ES`：observed running-sum 极值；
- `NES`：同符号 null 归一化分数；
- `p_value` 与 `q_value`：名义 P 值和 BH 校正结果；
- `direction`：`positive` 或 `negative`；
- `peak_rank`：首次正峰或最后负谷的位置；
- `leading_edge_ids`：推动极值的集合成员。

## Leading edge 的精确定义

- 正 ES：从排序起点到**第一次**达到正峰之间的集合成员；
- 负 ES：从**最后一次**达到负谷之后到排序末端的集合成员；
- ES 为零或方向不明确：leading edge 为空。

::: warning 不要过度解释
Leading edge 是 running-sum driver，不是逐边显著性列表。网络级 q 值不能转移给其中每条边。
:::

## 构建 leading-edge network

```python
from conlens import build_leading_network

network = build_leading_network(result, "DMN--VIS")
graph = network.to_networkx()
network.save("dmn-vis-leading.graphml")
```

网络只包含 leading-edge 边及其实际端点，不会补入未使用节点。

## 绘图

```python
from conlens.plotting import (
    plot_enrichment,
    plot_leading_adjacency,
    plot_nes,
)

plot_enrichment(result, "DMN--VIS", edge_sets["DMN--VIS"])
plot_nes(result)
plot_leading_adjacency(network)
```

## 保存与恢复

```python
from conlens import LensResult

result.save("result.json")
restored = LensResult.load("result.json")
```

JSON 包含结果表、排序边表与分析元数据，适合长期保存和审计。
