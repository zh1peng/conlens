---
title: 推断与零模型
description: Edge permutation、contrast-specific Freedman–Lane、P 值与联合 BH
---

# 推断与零模型

ConLens 将确定性的排序富集统计量与推断严格分离。连续变量关联和分类组间比较统一使用
`analysis.glm(design, contrasts)`，不再维护不同的 subject-level 拟合入口。

## 支持的推断路径

| 路径 | 适用输入 | 保留的结构 | 主要限制 |
| --- | --- | --- | --- |
| `edge_permutation` | 一张边统计量表 | 集合大小与重叠 | 破坏连接依赖、拓扑、空间与边间协方差 |
| contrast-specific Freedman–Lane | subject connectomes + validated design | 同一 residual-row permutation 作用于全部边 | design 与 exchangeability 必须正确 |
| provided null | 外部 ES、统计量矩阵或 rank 矩阵 | 取决于外部生成流程 | 身份信息必须与 observed 分析一致 |

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

它回答 competitive edge-label 问题，不保留真实 connectome 的 subject-level 依赖。

## Subject-level Freedman–Lane

```python
results = analysis.glm(
    design,
    contrasts,
    exchangeability_blocks=site,
    n_permutations=10_000,
    random_state=1,
)
```

对于每个一自由度 contrast $\mathbf c$，检验

$$
H_0:\mathbf c^\mathsf{T}\boldsymbol\beta_e=0.
$$

ConLens 为该约束构造 contrast-specific reduced design，拟合 reduced model，置换合法的
residual rows，然后重新拟合 full model。每次 replicate 都重新计算 edge effect、完整排序、
全部 set 的 ES 与 leading edge。不同 contrast 不会错误共享同一个 reduced model。

`exchangeability_blocks` 只限制允许的 residual permutation，不等同于把 site、family 或
其他 covariate 加入 design。需要调整的变量仍必须出现在 `make_design()` 中。

## P 值与 NES

- P 值是同符号、add-one empirical probability，因此不会为零；
- 正 observed ES 只与非负 null ES 比较，负 observed ES 只与非正 null ES 比较；
- NES 使用同符号 null ES 的平均绝对幅度归一化；
- NES 不是 edge effect size，也不是标准化回归系数。

## 多 contrast 联合 BH

一次 `analysis.glm()` 调用定义一个 correction family。若包含 $C$ 个 contrasts 和 $S$ 个
有效 edge sets，ConLens 对最多 $C\times S$ 个 nominal set-level P 值执行一次
Benjamini–Hochberg 校正。返回的每个 `LensResult` 都记录相同的
`correction_family_id` 与 `n_sets_tested`。

不同 cohort、modality 或科学上独立的 confirmatory families 不应仅为了方便放入同一次调用。

## 置换次数

调试可以使用较少 replicate；正式分析通常至少使用 10,000 次。始终设置
`random_state`，并报告同符号 null 数量与最小可解析 P 值。
