# Design、contrast 与效应量

这一页从 LENS 真正使用的输入开始：每条边需要一个 signed effect。`lens_glm` 负责把个体
connectomes、design matrix 和 contrast 转成这些 edge statistics；`lens_stat` 再对完整边排序
计算网络集合的 ES。

## 统一的模型入口

无论研究问题是年龄关联还是组间差异，底层都是同一个 OLS：

$$
\mathbf Y_e = \mathbf X\boldsymbol\beta_e + \boldsymbol\varepsilon_e,
$$

其中 $\mathbf Y_e$ 是第 $e$ 条边在所有受试者中的取值，$\mathbf X$ 是同一个 design，
$\mathbf c$ 定义要检验的 contrast。不同问题只改变 $\mathbf X$、$\mathbf c$ 和进入边排序的效应量。

## `make_design` 做什么

- 连续变量默认先减去样本均值；可用 `center_continuous=False` 关闭，并会记录在 metadata 中。
- 只有连续变量或独立 indicator 时，自动加入 intercept。
- `groups=` 是互斥且穷尽的 cell-means columns；此时不另加 intercept。
- categorical/group indicators 不中心化。
- interaction 在连续变量中心化后构造。
- 用户用 `matrix=` 直接给矩阵时原样使用，不中心化、不加 intercept、不重编码。
- 不做 sequential orthogonalization。QR/SVD 只可作为数值计算，不改变 estimand。
- rank deficiency 直接报错；condition number 大于 $10^8$ 时给出警告并写入结果。

```python
from conlens import Contrast, make_design, plot_design

design = make_design(
    groups={
        "control": diagnosis == "control",
        "g1": diagnosis == "g1",
        "g2": diagnosis == "g2",
    },
    continuous={"age": age, "motion": mean_fd},
    indicators={"sex": sex},
)

contrasts = {
    "g1_vs_control": Contrast(
        {"g1": 1, "control": -1}, "hedges_g", "g1 > control"
    ),
    "age": Contrast(
        {"age": 1}, "partial_r", "connectivity increases with age"
    ),
}

plot_design(design, contrasts)
```

Contrast 的 key 直接对应 `design.columns`。不需要写 `diagnosis[g1]` 之类的二次命名。

## 连续变量：partial correlation

对一自由度 contrast，先得到 full-model t statistic。进入 LENS 的边效应量是：

$$
r_{partial,e} = \frac{t_e}{\sqrt{t_e^2 + df_{res}}}.
$$

### 只有 age

```python
design = make_design(continuous={"age": age})
contrasts = {
    "age": Contrast(
        {"age": 1},
        effect_size="partial_r",
        positive_direction="connectivity increases with age",
    )
}
```

这里 design 自动包含 `intercept`；age 默认 mean-centered。中心化改变 intercept 的含义，不改变
age slope、t 或 partial $r$。

### age + covariates

```python
design = make_design(
    continuous={"age": age, "mean_fd": mean_fd},
    indicators={"sex": sex},
)
contrasts = {
    "age": Contrast(
        {"age": 1},
        effect_size="partial_r",
        positive_direction="connectivity increases with age, adjusted for sex and motion",
    )
}
```

此时 partial $r$ 对应控制了 design 中其余列后的 age 关联。如果只传 age，没有协变量，软件不会
凭空“自动调整”其他变量；它只调整你明确放进 design 的列。

## 分类变量：model-adjusted Hedges' g

对于组间 contrast，ConLens 保留 adjusted contrast estimate $\hat\beta_e$、标准误、t、残差自由度、
双侧 edge-wise P 和 full-model residual SD。进入 LENS 的效应量定义为：

$$
g_e = J\frac{\hat\beta_e}{s_{res,e}},
\qquad
J = 1 - \frac{3}{4df_{res}-1}.
$$

$s_{res,e}$ 来自包含所有组和协变量的 full model。多组设计里，它不是只用 contrast 两组重新计算的
pooled SD；这样所有 contrasts 共用同一误差模型，也与 adjusted coefficient 的定义一致。

### g1 vs control

```python
design = make_design(
    groups={
        "control": diagnosis == "control",
        "g1": diagnosis == "g1",
    }
)
contrasts = {
    "g1_vs_control": Contrast(
        {"g1": 1, "control": -1},
        effect_size="hedges_g",
        positive_direction="g1 > control",
    )
}
```

### g1 vs control + covariates

```python
design = make_design(
    groups={
        "control": diagnosis == "control",
        "g1": diagnosis == "g1",
    },
    continuous={"age": age, "mean_fd": mean_fd},
    indicators={"sex": sex},
)
contrasts = {
    "g1_vs_control": Contrast(
        {"g1": 1, "control": -1},
        effect_size="hedges_g",
        positive_direction="g1 > control, adjusted for age, sex and motion",
    )
}
```

### g1 vs control 与 g2 vs control

```python
design = make_design(
    groups={
        "control": diagnosis == "control",
        "g1": diagnosis == "g1",
        "g2": diagnosis == "g2",
    }
)
contrasts = {
    "g1_vs_control": Contrast(
        {"g1": 1, "control": -1}, "hedges_g", "g1 > control"
    ),
    "g2_vs_control": Contrast(
        {"g2": 1, "control": -1}, "hedges_g", "g2 > control"
    ),
}
```

这是一次模型拟合入口，不需要为两个比较分别调用两次 API。两个 contrast 的网络集合 P 值还会在
同一个 `lens_enrich` 调用中联合做 BH。

### 两个组 contrast + covariates

```python
design = make_design(
    groups={
        "control": diagnosis == "control",
        "g1": diagnosis == "g1",
        "g2": diagnosis == "g2",
    },
    continuous={"age": age, "mean_fd": mean_fd},
    indicators={"sex": sex},
)
contrasts = {
    "g1_vs_control": Contrast(
        {"g1": 1, "control": -1}, "hedges_g", "g1 > control"
    ),
    "g2_vs_control": Contrast(
        {"g2": 1, "control": -1}, "hedges_g", "g2 > control"
    ),
}
```

## 从 edge-wise effect 到正式 LENS 推断

上面的 design/contrast 只定义了 observed 边效应。正式分析还要让每个 FL null 都走同一条
`lens_stat`：

```python
from conlens import lens_enrich, lens_fl_permute, lens_glm, lens_stat

true_edges = lens_glm(
    connectomes,
    design=design,
    contrasts=contrasts,
    node_labels=node_labels,
)
observed = lens_stat(true_edges, edge_sets, store_running_sum=True)

null_edges = lens_fl_permute(
    connectomes,
    design=design,
    contrasts=contrasts,
    n_permutations=10_000,
    exchangeability_blocks=site,
    random_state=42,
)
null_stats = (lens_stat(item, edge_sets) for item in null_edges)

fit = lens_enrich(
    observed,
    null_stats,
    min_size=5,
    max_size=500,
    family_name="primary-model",
)
```

`lens_fl_permute` 对每个 contrast 构造对应的 reduced model。Observed 与 null 都用相同的效应量、
edge universe、edge sets、权重和 ES 定义。`lens_enrich` 只消费 set-level LENS statistics，因而
无需保存巨大的 edge × permutation 数组。

设计看清楚后，可以继续看[Permutation 与推断](/guide/inference)和[可视化](/tutorials/visualization)。
