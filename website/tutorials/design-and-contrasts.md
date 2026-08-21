---
title: Design matrix 与 contrasts
description: 用统一 GLM 表达连续变量关联、组间比较和协变量调整
---

# 教程：构建设计矩阵与 contrasts

ConLens 的受试者级分析只有一个入口：

```python
results = analysis.glm(design, contrasts)
```

连续变量关联和组间比较使用同一个 GLM 与同一套 contrast-specific
Freedman–Lane 推断。两者的区别不是拟合器，而是进入 LENS 排序的效应量：

- 连续变量 contrast：signed Pearson/partial $r$；
- 分类组间 contrast：model-adjusted signed Hedges' $g$。

## 1. Builder 的明确规则

`make_design()` 不猜测变量类型。用户通过不同参数明确声明：

```python
from conlens import Contrast, make_design, plot_design

design = make_design(
    indicators={
        "control": diagnosis == "control",
        "g1": diagnosis == "g1",
        "g2": diagnosis == "g2",
        "male": sex == "male",
    },
    continuous={
        "age": age,
        "motion": mean_fd,
    },
    add_intercept=False,
)
```

| 输入 | Builder 行为 |
| --- | --- |
| `continuous` | 默认 mean-center；名称保持不变 |
| `indicators` | 只接受 0/1；不中心化；名称保持不变 |
| `interactions` | 在连续变量中心化后构造；输出名由用户指定 |
| `matrix` | 原样使用；不中心化、不增加截距、不构造 interaction |

Semantic mode 默认增加 `intercept`。使用一列对应一个组均值的 cell-means
design 时，组 indicator 已经共同承担截距，因此必须传 `add_intercept=False`。
如果误把截距与全部组 indicators 同时加入，ConLens 会因为 design 不满秩而报错。

关闭连续变量中心化：

```python
design = make_design(
    continuous={"age": age},
    center_continuous=False,
)
```

如果已经在其他软件中构造并检查好完整 design matrix，可使用 raw-matrix mode：

```python
design = make_design(
    matrix=X,
    column_names=["control", "g1", "g2", "age", "motion"],
)
```

该模式逐值使用 `X`，不会静默中心化、增加截距、生成 interaction 或正交化列。

中心化均值、实际列名、interaction 来源、condition number 与是否加入截距均保存在：

```python
print(design.metadata())
print(design.frame.head())
```

ConLens 不进行 sequential regressor orthogonalization。内部数值分解不会修改 design，
也不会改变 estimand。design 不满秩会立即报错；condition number 大于 $10^8$ 时发出警告。

## 2. Contrast 只引用实际列名

`design.columns` 是 contrast 名称的唯一来源：

```python
print(design.columns)
# ('control', 'g1', 'g2', 'male', 'age', 'motion')
```

未写出的列自动获得权重 0；未知名称会报错。

```python
g1_vs_control = Contrast(
    {"g1": 1, "control": -1},
    effect_size="hedges_g",
    positive_direction="g1 > control",
)
```

Hedges' $g$ 会随 contrast 缩放，因此 ConLens 要求正权重之和为 1、负权重之和为
$-1$。例如平均两个治疗组对 control 可以写成
`{"g1": 0.5, "g2": 0.5, "control": -1}`。

## 3. Age

只有一个连续预测变量时，semantic mode 自动加入截距，`age` 默认中心化：

```python
design = make_design(
    continuous={"age": age},
)

contrasts = {
    "age": Contrast(
        {"age": 1},
        effect_size="partial_r",
        positive_direction="connectivity increases with age",
    ),
}

results = analysis.glm(
    design,
    contrasts,
    n_permutations=10_000,
    random_state=42,
)
age_result = results["age"]
```

没有 covariates 时，该效应量等于每条边与 age 的 signed Pearson $r$。

## 4. Age + covariates

```python
design = make_design(
    indicators={
        "male": sex == "male",       # female 为 reference
        "site_B": site == "B",       # site A 为 reference
        "site_C": site == "C",
    },
    continuous={
        "age": age,
        "motion": mean_fd,
    },
)

contrasts = {
    "age": Contrast(
        {"age": 1},
        effect_size="partial_r",
        positive_direction="connectivity increases with age",
    ),
}

results = analysis.glm(design, contrasts, n_permutations=10_000, random_state=42)
```

这里进入 LENS 排序的是控制 sex、site 和 motion 后的 signed partial $r$。
中心化只改变截距的解释，不改变没有 interaction 时的 age slope、t 或 partial $r$。

## 5. G1 vs. control

```python
design = make_design(
    indicators={
        "control": diagnosis == "control",
        "g1": diagnosis == "g1",
    },
    add_intercept=False,
)

contrasts = {
    "g1_vs_control": Contrast(
        {"g1": 1, "control": -1},
        effect_size="hedges_g",
        positive_direction="g1 > control",
    ),
}

results = analysis.glm(design, contrasts, n_permutations=10_000, random_state=42)
```

没有 covariates 时，这是基于两组共同 full-model residual SD 的 signed Hedges' $g$。
在只有两个组的同方差 OLS 中，它对应常规 pooled-residual standardized mean difference，
并带 residual-df small-sample correction。

## 6. G1 vs. control + covariates

```python
design = make_design(
    indicators={
        "control": diagnosis == "control",
        "g1": diagnosis == "g1",
        "male": sex == "male",
        "site_B": site == "B",
        "site_C": site == "C",
    },
    continuous={
        "age": age,
        "motion": mean_fd,
    },
    add_intercept=False,
)

contrasts = {
    "g1_vs_control": Contrast(
        {"g1": 1, "control": -1},
        effect_size="hedges_g",
        positive_direction="g1 > control",
    ),
}

results = analysis.glm(design, contrasts, n_permutations=10_000, random_state=42)
```

`g1 - control` 是在平均 age、平均 motion、female 和 site A 处的调整后组均值差。
进入 LENS 的 $g$ 使用这个调整后差值和整个 full model 的 residual SD。

## 7. G1 vs. control 与 G2 vs. control

```python
design = make_design(
    indicators={
        "control": diagnosis == "control",
        "g1": diagnosis == "g1",
        "g2": diagnosis == "g2",
    },
    add_intercept=False,
)

contrasts = {
    "g1_vs_control": Contrast(
        {"g1": 1, "control": -1},
        effect_size="hedges_g",
        positive_direction="g1 > control",
    ),
    "g2_vs_control": Contrast(
        {"g2": 1, "control": -1},
        effect_size="hedges_g",
        positive_direction="g2 > control",
    ),
}

results = analysis.glm(design, contrasts, n_permutations=10_000, random_state=42)
```

两个 contrast 使用同一个三组 full model。因此每条边上两个 Hedges' $g$ 的
$s_{\mathrm{res}}$ 都来自包含 control、g1 和 g2 的完整模型，而不是各自两组子样本。

## 8. G1 vs. control 与 G2 vs. control + covariates

```python
design = make_design(
    indicators={
        "control": diagnosis == "control",
        "g1": diagnosis == "g1",
        "g2": diagnosis == "g2",
        "male": sex == "male",
        "site_B": site == "B",
        "site_C": site == "C",
    },
    continuous={
        "age": age,
        "motion": mean_fd,
    },
    add_intercept=False,
)

contrasts = {
    "g1_vs_control": Contrast(
        {"g1": 1, "control": -1},
        effect_size="hedges_g",
        positive_direction="g1 > control",
    ),
    "g2_vs_control": Contrast(
        {"g2": 1, "control": -1},
        effect_size="hedges_g",
        positive_direction="g2 > control",
    ),
}

results = analysis.glm(
    design,
    contrasts,
    exchangeability_blocks=family_id,
    n_permutations=10_000,
    random_state=42,
    correction_family_id="primary-diagnosis-contrasts",
)

print(results.to_frame().sort_values(["contrast_name", "q_value"]))
```

ConLens 对每个 contrast 构造自己的 reduced model，并在所有
`contrast × valid edge sets` 的 nominal P 值上执行一次联合 BH。

## 9. Design 与 contrast 可视化

在运行耗时的 permutation 前先检查实际 design：

```python
axes = plot_design(design, contrasts)
axes[0].figure.savefig("design-and-contrasts.png", dpi=200)
```

左图显示最终进入 GLM 的 design（仅为显示而按列缩放）；右图显示具名 contrast。
图标题同时报告 rank 和 condition number。可视化不会改变实际拟合矩阵。

## 10. Interaction

Interaction 的最终名称和来源都必须显式声明：

```python
design = make_design(
    indicators={
        "control": diagnosis == "control",
        "g1": diagnosis == "g1",
    },
    continuous={"age": age},
    interactions={"g1_age": ("g1", "age")},
    add_intercept=False,
)
```

`g1_age` 使用已经中心化的 `age` 构造。此参数化中：

- `{"age": 1}`：control 的 age slope；
- `{"g1_age": 1}`：g1 相对 control 的 slope difference；
- `{"age": 1, "g1_age": 1}`：g1 的 age slope。

## 11. 进入 LENS 的效应量

对边 $e$ 拟合完整模型

$$
\mathbf y_e = \mathbf X\boldsymbol\beta_e + \boldsymbol\varepsilon_e,
\qquad
df_{\mathrm{res}} = n - \operatorname{rank}(\mathbf X).
$$

给定一自由度 contrast $\mathbf c$，ConLens 保存

$$
\hat\beta_{c,e}=\mathbf c^\mathsf{T}\hat{\boldsymbol\beta}_e,
$$

$$
SE_{c,e}=s_{\mathrm{res},e}
\sqrt{\mathbf c^\mathsf{T}(\mathbf X^\mathsf{T}\mathbf X)^{-1}\mathbf c},
\qquad
t_{c,e}=\frac{\hat\beta_{c,e}}{SE_{c,e}},
$$

以及 residual df 和双侧 edge-wise P 值。edge-wise P 值仅用于审计，不用于预筛边。

### 连续变量：signed partial r

$$
r_{\mathrm{partial},e}
=
\frac{t_{c,e}}{\sqrt{t_{c,e}^{2}+df_{\mathrm{res}}}}.
$$

没有 covariates 时它等于 Pearson $r$；有 covariates 时是该一自由度 contrast 的
signed partial correlation。

### 组间 contrast：model-adjusted signed Hedges' g

$$
g_e
=
J\frac{\hat\beta_{c,e}}{s_{\mathrm{res},e}},
\qquad
J=1-\frac{3}{4df_{\mathrm{res}}-1},
$$

其中

$$
s_{\mathrm{res},e}
=
\sqrt{
\frac{
\lVert\mathbf y_e-\mathbf X\hat{\boldsymbol\beta}_e\rVert_2^2
}{df_{\mathrm{res}}}
}.
$$

这里的 $s_{\mathrm{res},e}$ 始终来自完整模型，包含所有组、covariates 和 interaction；
它不是从当前 contrast 涉及的两个组重新计算的 SD。

ConLens 按上述 signed effect 从大到小排列全部边，然后计算 running sum、ES 和
leading edge。它不按 edge-wise P 值筛选边。

## 12. Contrast-specific Freedman–Lane

每个 contrast 检验 $H_0:\mathbf c^\mathsf{T}\boldsymbol\beta_e=0$。ConLens 使用该
约束对应的 reduced design 拟合 $\hat{\mathbf y}_{0,e}$ 与 residual matrix
$\mathbf R_0$，并构造

$$
\mathbf Y_b^* = \hat{\mathbf Y}_0 + \mathbf P_b\mathbf R_0.
$$

每次 replicate 重新拟合 full model、重新计算效应量、重新排序全部边并重新计算全部
edge sets。不同 contrast 的 reduced model 不会被错误地共享。
