---
title: 含协变量的 GLM
description: 使用 Freedman–Lane 处理 nuisance covariates
---

# 教程：含协变量的 GLM

当分析包含年龄、性别、site 或其他 nuisance covariates 时，使用 `SubjectLensAnalysis.glm` 和 Freedman–Lane。

## 设计矩阵

```python
import numpy as np

tested = age[:, None]
nuisance = np.column_stack([
    np.ones(len(age)),  # 必须显式包含截距
    sex,
    site_dummy_1,
    site_dummy_2,
])
contrast = np.array([1.0])
```

`tested_design` 可以包含一个或多个待检验列。`nuisance_design` 必须显式包含全 1 截距列；函数不会静默补入截距。

## 运行 Freedman–Lane

```python
result = analysis.glm(
    tested,
    nuisance,
    contrast=contrast,
    null_method="freedman_lane",
    exchangeability_blocks=site,
    n_permutations=10_000,
    random_state=42,
    positive_direction="positive age coefficient",
)
```

每次 replicate 会：

1. 拟合只含 nuisance design 的 reduced model；
2. 在允许的 block 内置换 reduced-model residual rows；
3. 构造新的 response matrix；
4. 对每条边重拟合 full model；
5. 重新排序并计算全部 LENS set；
6. 将 observed P 值进行 BH 校正。

## Contrast 长度

- 若 `contrast` 长度等于 tested design 列数，它只作用于 tested 部分；
- 若长度等于 full design 列数，它可以显式指定完整 contrast；
- 省略时默认检验第一个 tested column。

## 设计检查

正式分析前至少确认：

- tested 和 nuisance 行数等于受试者数；
- nuisance 中有明确截距；
- full design 满秩；
- exchangeability blocks 与采样/研究设计一致；
- `positive_direction` 准确描述 contrast 的正方向；
- correction family 只包含计划共同校正的 sets。

::: warning 多 contrast 分析
每个 `LensResult` 表示一次明确的分析/contrast/set family。当前不要仅凭相同 `correction_family_id` 假装多个独立结果已经做过跨 contrast 的联合 BH 校正。
:::

