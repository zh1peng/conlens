---
title: LENS 的输入是什么
description: 先看 LENS 实际读取的数据，再选择有个体数据或只有边统计量的分析路线
---

# LENS 的输入是什么

LENS 分析的对象是一张完整的边排序。对一次 analysis，它需要：

- edge universe：本次分析包含哪些边；
- signed edge statistic：每条边在同一方向定义下的一个数值；
- edge sets：准备检验的连接集合。

假设 `positive_direction="g1 > control"`。正数表示连接在 g1 中更高，负数表示在
control 中更高。LENS 将所有边从大到小排列，再检查某个 edge set 的成员是否集中在
排序的正端或负端。

它不会先挑出 edge-wise P 值小于 0.05 的边。每一条有效边都会进入 running sum。

## 两条输入路线

| 手头的数据 | 排序量从哪里来 | 合适的推断 |
| --- | --- | --- |
| 个体 connectomes | `analysis.glm()` 为每条边计算 partial $r$ 或 model-adjusted Hedges' $g$ | contrast-specific Freedman–Lane |
| 已算好的 edge statistics | 直接使用 `statistic` 列 | 描述性 LENS、edge permutation，或外部 provided null |

有个体数据时，优先走第一条路线。ConLens 会在每次 permutation 中重新拟合所有边、
重新排序，再计算每个 edge set 的 ES。这样置换分布针对的是 LENS 结果本身。

只有 edge statistics 时，ConLens 看不到受试者、design 和协方差结构。内置 edge
permutation 仍可用，但它回答的是另一个零假设，不能替代 subject-level permutation。

## LENS 从排序中算什么

沿排序从左到右扫描：遇到 set 内的边，running sum 上升或按权重变化；遇到 set 外的
边，running sum 下降。曲线离零最远的位置给出 enrichment score（ES）。正 ES 表示集合
成员偏向排序正端，负 ES 表示偏向负端。

Leading edge 是推动曲线到达该极值的 set members。它是网络级结果的定位，不是一张
edge-wise significance list。

## 接下来怎么读

第一次使用可以先跑[五分钟快速开始](/guide/quick-start)，再读[从个体数据到 LENS](/tutorials/design-and-contrasts)。后者包含 design、contrasts、效应量和 subject-level
permutation 的完整关系。

如果没有个体数据，直接看[只有 edge statistics 时](/tutorials/edge-statistics)，并留意
其中的零模型限制。
