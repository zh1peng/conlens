---
title: 推断与零模型
description: ConLens 支持的零模型、P 值与 BH 校正
---

# 推断与零模型

ConLens 将确定性的排序富集统计量与推断严格分离。如果不传 `null_method`，结果只有 ES 和 leading edge；NES、P 与 q 保持 `None`。

## 支持的推断路径

| 路径 | 适用输入 | 保留的结构 | 主要限制 |
| --- | --- | --- | --- |
| `edge_permutation` | 一张边统计量表 | 集合大小与重叠 | 破坏连接依赖、拓扑与空间结构 |
| `label_permutation` | 简单两组或单表型受试者设计 | 每名受试者的完整连接向量 | 必须满足可交换性；不处理 nuisance covariates |
| `freedman_lane` | tested design + nuisance design | 共享的受试者残差置换 | nuisance design 必须显式含截距；设计需满足可交换性 |
| provided null | 外部生成的 ES、统计量矩阵或 rank 矩阵 | 取决于外部生成流程 | 必须提供完整身份信息并与 observed 分析一致 |

## Edge permutation

```python
result = lens_enrich(
    edges,
    edge_sets,
    null_method="edge_permutation",
    n_permutations=10_000,
    random_state=1,
    positive_direction="case > control",
)
```

它回答的是 competitive edge-label 问题，不是保留真实 connectome 依赖结构的 subject-level 推断。

## Label permutation

```python
result = analysis.two_group(
    group,
    null_method="label_permutation",
    exchangeability_blocks=site,
    n_permutations=10_000,
    random_state=1,
)
```

同一次 permutation 会作用于所有边，然后重新计算统计量、排序和每个集合的 ES。`exchangeability_blocks` 限制允许的置换范围。

## Freedman–Lane

```python
result = analysis.glm(
    tested_design,
    nuisance_design,
    contrast=contrast,
    null_method="freedman_lane",
    exchangeability_blocks=site,
    n_permutations=10_000,
    random_state=1,
)
```

ConLens 先拟合 reduced model，再置换其残差行并重新拟合 full model。相同的合法置换作用于每一条边。

## P 值与 NES

- P 值采用**同符号、add-one empirical probability**，因此不会得到零 P 值。
- 正观察 ES 只与非负 null ES 比较；负观察 ES 只与非正 null ES 比较。
- NES 用同符号 null ES 的平均绝对幅度标准化；它不是效应量或标准化回归系数。

## BH 多重检验

同一个分析、contrast 和预先声明的 set family 中的有效 P 值进行一次 Benjamini–Hochberg 校正。不同 phenotype、cohort、modality 或确认性 family 不应仅因为方便而混在一次校正中。

## 置换次数

调试时可以使用较少 replicate，但正式分析通常至少使用 10,000 次。所有随机路径都应明确设置 `random_state`，并在报告中记录可解析的最小 P 值。

