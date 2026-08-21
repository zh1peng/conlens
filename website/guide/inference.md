---
title: Permutation 与零模型
description: Subject-level Freedman–Lane、edge permutation、provided null 和 BH 校正
---

# Permutation 与零模型

ES 是 observed 排序的描述。P 值、NES 和 q 值还需要一个零模型。选择哪种零模型，取决于
ConLens 能看到哪一层数据。

## 有个体数据：Freedman–Lane

```python
fit = analysis.glm(
    design,
    contrasts,
    exchangeability_blocks=site,
    n_permutations=10_000,
    random_state=1,
)
```

对每个一自由度 contrast $\mathbf c$，ConLens 检验

$$
H_0:\mathbf c^\mathsf{T}\boldsymbol\beta_e=0.
$$

Reduced model 由当前 contrast 的约束决定。ConLens 置换 reduced-model residual rows，
然后用完整 design 重新拟合每条边。同一个 row permutation 同时作用于所有边，因此不会
拆散同一受试者内的 edge covariance。

每个 replicate 都会产生一张新的完整边排序，随后重算所有 edge sets 的 ES。最终 P 值
比较的是 observed ES 与 permuted ES，不是把 edge-wise P 值送进 LENS。

`exchangeability_blocks` 限制 row permutation 的范围。需要调整的 site、motion 或其他
covariates 仍要写进 design；两者用途不同。

## 只有边统计量：edge permutation

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

这个方法打乱 statistic 与 edge ID 的对应关系。它保留 edge-set 大小和重叠，却不保留
共享节点、拓扑、空间关系或 edge covariance。它检验 competitive edge-label null，
不是 subject-level model null。

## 外部生成的 null

`null_method="provided_null"` 可接收 null ES、edge-statistic matrices 或 rank matrices。
Observed 与 null 的 edge 顺序、sets 和方向定义必须一致。Rank matrices 没有原始统计量
幅度，只能用于 `weight=0` 的 unweighted enrichment。

这种方式是否保留研究设计和依赖结构，取决于外部 null 的生成过程。ConLens 只能验证
传入对象的身份信息，不能从结果矩阵反推出置换是否设计正确。

## P 值、NES 和 BH

P 值使用 add-one empirical estimate，并按 observed ES 的符号选择同侧 null。NES 用
同符号 null ES 的平均绝对值做归一化。NES 是网络集合统计量，不是 edge effect size。

一次 `analysis.glm()` 调用构成一个 correction family。若有 $C$ 个 contrasts 和 $S$ 个
有效 edge sets，BH 覆盖最多 $C\times S$ 个 nominal P 值。不同 cohort 或互不相关的
confirmatory families 不应为了省一次调用而合并。

调试时可以减少 permutations。正式分析应根据所需的 P 值分辨率选择次数，并固定
`random_state`。ConLens 会记录 permutation 数量、scheme 和 correction family。
