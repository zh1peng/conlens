---
title: 认识 ConLens
description: ConLens 的分析目标、适用场景与核心边界
---

# 认识 ConLens

ConLens 是 LENS（Leading-edge Network Set enrichment）的透明、可复现、模态无关实现。它回答的问题不是“哪一条连接通过了边水平阈值”，而是：

> 一个预先定义的连接集合，是否在完整的有符号边排序中集中出现在正端或负端？哪些连接推动了这次富集？

## 核心工作流

1. 为每条有效连接计算一个**有方向的统计量**。
2. 冻结完整 edge universe，并按统计量从高到低排序。
3. 对预先定义的 edge set 计算加权 running sum 和 enrichment score（ES）。
4. 通过明确声明的零模型获得 NES、P 值与 BH 校正后的 q 值。
5. 提取 leading edge，并构建对应的 leading-edge network。
6. 如有受试者级数据，通过完整重拟合 bootstrap 检查集合和边的采样稳定性。

## 为什么不先筛选边？

边水平阈值会丢弃连续证据，并使网络结果依赖一个常常任意的 cutoff。ConLens 使用每一条有效边，只把排序、集合定义和零模型固定下来。这样可以检验“集合成员是否系统性聚集”，而不是把若干单边结果事后拼成网络。

## 三个必须区分的层次

| 层次 | ConLens 输出 | 不能声称什么 |
| --- | --- | --- |
| 网络集合 | ES、NES、P、q、方向 | 不能据此说集合内每条边都显著 |
| Leading edge | 驱动 running-sum 极值的连接 | 不是 edge-wise significance list |
| Bootstrap 稳定性 | 采样下的重复检出与纳入频率 | 不是“边为真”的概率，也不是未来研究的精确复制概率 |

## 适用场景

- 已有一张完整的连接统计量表，需要检验预定义网络对或自定义连接集合。
- 有受试者 × 节点 × 节点的 connectome，需要进行两组、表型或 GLM 分析。
- 希望在保留连接依赖结构的前提下使用 subject-level label permutation 或 Freedman–Lane。
- 希望提取 leading-edge network，并评估完整分析流程的 bootstrap 稳定性。

## 不适合直接使用的场景

- edge set 是看过当前统计结果后临时定义的，却仍被当作确认性假设。
- 没有明确统计量正方向，无法解释正/负富集的生物学含义。
- 设计包含家庭或重复测量 cluster，却把每一行当作独立受试者 bootstrap。
- 希望用 leading edge 代替边水平多重检验。

## 下一步

- [安装 ConLens](/guide/installation)
- [五分钟快速开始](/guide/quick-start)
- [理解推断与零模型](/guide/inference)

