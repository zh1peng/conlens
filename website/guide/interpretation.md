---
title: 结果解释
description: 正确报告 ConLens 富集、leading edge 与稳定性结果
---

# 结果解释

## 推荐的表述

> 在预先定义的 DMN--VIS 连接集合中，边统计量显著聚集于排序正端（NES = …，BH q = …）。`positive_direction` 定义为 case > control。Leading-edge 连接是推动该集合 running-sum 峰值的成员，不作逐边显著性解释。

## 常见误读

| 输出 | 正确含义 | 错误说法 |
| --- | --- | --- |
| ES | observed running-sum 极值 | 效应量 |
| NES | 同符号 null 标准化的 ES | 标准化 beta |
| q 值 | 集合 family 的 BH 结果 | 集合中所有边都显著 |
| Leading edge | 推动集合富集的排序成员 | 独立验证的显著边 |
| 负 NES | 集合聚集在低统计量一端 | 自动等于“连接降低” |

负方向的生物学含义完全依赖 `positive_direction`。例如统计量定义为 `case > control` 时，负富集才对应集合边更集中于 control 较高的一端。

## 零模型必须与问题匹配

- edge permutation 不保留 connectome 的共享节点、空间、拓扑和边协方差结构；
- label permutation 依赖受试者标签的可交换性；
- Freedman–Lane 依赖 reduced-model residual 的合法置换；
- exchangeability blocks 必须代表真实的受限交换结构。

## Bootstrap stability

- `set_stability`：所有 bootstrap 样本中，集合再次通过 q 阈值且方向一致的比例；
- `conditional_stability`：只在同方向检出样本中，某条边进入 leading edge 的比例；
- `full_pipeline_stability`：所有 bootstrap 样本中，该边在同方向检出的集合里进入 leading edge 的比例；
- 当 conditional quantity 有定义时，`full_pipeline_stability = set_stability × conditional_stability`。

这些频率描述给定分析流程对采样的敏感性，不是 posterior probability、edge-level confidence interval 或精确未来复制概率。

## 最小报告清单

1. edge universe、节点顺序、有向性与对角线规则；
2. ranking statistic 名称及其 `positive_direction`；
3. edge set 的预定义来源和大小过滤；
4. weight exponent、score type 和 tie rule；
5. null method、置换次数、seed 与 exchangeability blocks；
6. BH correction family；
7. ES、NES、P、q、方向和 leading-edge 大小；
8. 若报告稳定性，说明 outer resampling、inner inference、阈值和分母。

