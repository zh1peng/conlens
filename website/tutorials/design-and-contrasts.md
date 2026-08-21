---
title: 从个体数据到 LENS
description: 从 connectome、design 和 contrast 生成边排序，并在置换中重跑整个 LENS 分析
---

# 从个体数据到 LENS

先把分析对象说清楚。LENS 不直接读取 `age`、`diagnosis` 或 design matrix。对某一个
contrast，它真正使用的是下面三样东西：

1. 一套固定的边，也就是 edge universe；
2. 每条边的一个有符号数值，用来从大到小排序；
3. 事先定义好的 edge sets，例如网络内和网络间连接。

有个体 connectome 时，ConLens 先用 GLM 为每条边计算排序量，再把同一套计算放进
subject-level permutation。这样得到的不只是 edge-wise 回归结果，置换分布对应的也是
最终的 set-level LENS 统计量。

```text
subject connectomes + design + contrast
                  │
                  ├─ observed：逐边效应量 → 全边排序 → running sum → ES
                  │
                  └─ permutation：重排 reduced-model residual rows
                                  → 重新拟合每条边
                                  → 重新排序并计算每个 set 的 ES
```

如果手头只有一张已经算好的 edge-statistics 表，也可以运行 LENS，但可用的零模型和
结论范围会更窄。本页最后再讲这种情况。

## 1. 准备 connectome 和 edge sets

个体数据是形状为 `(subjects, nodes, nodes)` 的 NumPy 数组。一套分析中的节点顺序、
边 ID 和 edge-set 成员需要保持不变。

```python
import numpy as np
import pandas as pd

from conlens import (
    Contrast,
    LensAnalysis,
    make_design,
    make_network_pair_sets,
    matrix_to_edges,
    plot_design,
)

connectomes = np.load("connectomes.npy")
participants = pd.read_csv("participants.tsv", sep="\t")
labels = np.load("node-labels.npy", allow_pickle=True).tolist()

node_table = pd.read_csv("node-networks.tsv", sep="\t")
node_networks = dict(zip(node_table["node"], node_table["network"], strict=True))

edge_template = matrix_to_edges(connectomes[0], node_labels=labels)
edge_sets = make_network_pair_sets(edge_template, node_networks)

analysis = LensAnalysis.from_subject_connectomes(
    connectomes,
    edge_sets,
    node_labels=labels,
    min_size=5,
    store_running_sum=True,
)
```

`edge_sets` 也可以是研究前定义的自定义集合。不要根据当前样本的 edge-wise 结果选边，
然后再把这些边当作待检验集合。

## 2. Design 和 contrast 分别做什么

Design matrix 描述每个受试者。Contrast 从 design 的列中取出当前要检验的一维效应。

```python
design = make_design(
    indicators={
        "control": participants["diagnosis"] == "control",
        "g1": participants["diagnosis"] == "g1",
        "g2": participants["diagnosis"] == "g2",
        "male": participants["sex"] == "male",
    },
    continuous={
        "age": participants["age"],
        "motion": participants["mean_fd"],
    },
    add_intercept=False,
)

contrasts = {
    "g1_vs_control": Contrast(
        {"g1": 1, "control": -1},
        effect_size="hedges_g",
        positive_direction="g1 > control",
    ),
    "age": Contrast(
        {"age": 1},
        effect_size="partial_r",
        positive_direction="connectivity increases with age",
    ),
}
```

Contrast 只引用 `design.columns` 中实际存在的名称。没有写出的列权重为 0；拼错列名会
直接报错。

`make_design()` 的处理规则如下。

| 输入 | 实际处理 |
| --- | --- |
| `continuous` | 默认减去样本均值；可用 `center_continuous=False` 关闭 |
| `indicators` | 只接受 0/1，不做中心化 |
| `interactions` | 连续变量中心化后再相乘 |
| `matrix` | 原样使用，不加截距，也不改列 |

Semantic mode 默认添加截距。上面的 cell-means design 已经为每个组各放一列，所以用了
`add_intercept=False`。若同时加入截距和全部组 indicators，矩阵不满秩，ConLens 会停止。

ConLens 不做 sequential orthogonalization。内部会使用数值分解求解模型，但不会改写
design 或改变 contrast 的含义。矩阵不满秩时报错；condition number 大于 $10^8$ 时给出
警告。

在开始大量置换前，可以直接看实际送入模型的矩阵和 contrasts：

```python
axes = plot_design(design, contrasts)
axes[0].figure.savefig("design-and-contrasts.png", dpi=200)
```

左图是最终 design，显示时按列缩放；右图是 contrast 权重。图中的 rank 和 condition
number 也值得检查。绘图本身不改动拟合矩阵。

## 3. 哪个数值进入边排序

对每条边 $e$，完整模型为

$$
\mathbf y_e = \mathbf X\boldsymbol\beta_e + \boldsymbol\varepsilon_e,
\qquad
df_{\mathrm{res}} = n - \operatorname{rank}(\mathbf X).
$$

对一自由度 contrast $\mathbf c$，ConLens 保存

$$
\hat\beta_{c,e}=\mathbf c^\mathsf{T}\hat{\boldsymbol\beta}_e,
$$

$$
SE_{c,e}=s_{\mathrm{res},e}
\sqrt{\mathbf c^\mathsf{T}(\mathbf X^\mathsf{T}\mathbf X)^{-1}\mathbf c},
\qquad
t_{c,e}=\frac{\hat\beta_{c,e}}{SE_{c,e}}.
$$

边的双侧 P 值会保存在结果中，便于复核拟合，但 LENS 不用它筛边。排序使用下面的
signed effect size。

### 连续变量：Pearson r 或 partial r

```python
Contrast(
    {"age": 1},
    effect_size="partial_r",
    positive_direction="connectivity increases with age",
)
```

$$
r_{\mathrm{partial},e}
=
\frac{t_{c,e}}{\sqrt{t_{c,e}^{2}+df_{\mathrm{res}}}}.
$$

模型只有 `age` 和截距时，它等于 Pearson $r$。加入 motion、sex 或 site 后，它是控制
这些列后的 signed partial $r$。

### 组间 contrast：model-adjusted Hedges' g

```python
Contrast(
    {"g1": 1, "control": -1},
    effect_size="hedges_g",
    positive_direction="g1 > control",
)
```

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

$s_{\mathrm{res},e}$ 来自完整模型。三组模型中的 `g1 - control` 和 `g2 - control` 都使用
三组共同拟合后的 residual SD；加了 covariates 时，这些列也属于同一个完整模型。ConLens
不会为了某一个 contrast 另取两个组重算 SD。

Hedges' $g$ 会随 contrast 的整体缩放而改变，因此正权重之和必须是 1，负权重之和必须
是 $-1$。例如两治疗组均值对 control 可写成
`{"g1": 0.5, "g2": 0.5, "control": -1}`。

## 4. Permutation 检验的是 LENS 结果

`analysis.glm()` 使用 contrast-specific Freedman–Lane。对 contrast $\mathbf c$，零假设为

$$
H_0:\mathbf c^\mathsf{T}\boldsymbol\beta_e=0.
$$

ConLens 先在这个约束下拟合 reduced model，得到 $\hat{\mathbf Y}_0$ 和 residual matrix
$\mathbf R_0$。第 $b$ 次置换的数据是

$$
\mathbf Y_b^*=\hat{\mathbf Y}_0+\mathbf P_b\mathbf R_0.
$$

这里的 $\mathbf P_b$ 对所有边相同，所以同一受试者内的边间依赖没有被拆开。每次置换
随后会：

1. 用完整 design 重新拟合所有边；
2. 重新计算该 contrast 的 partial $r$ 或 Hedges' $g$；
3. 重新排序整个 edge universe；
4. 为每个 edge set 重新计算 running sum 和 ES。

Observed ES 最终与这些 permuted ES 比较，得到 NES 和 set-level P 值。如果一次
`analysis.glm()` 中有多个 contrasts，ConLens 对所有 `contrast × valid edge set` 的
P 值做一次 BH 校正。

`exchangeability_blocks` 只约束哪些 residual rows 可以互换。它不会替代 design 中的
site、family 或其他 covariates。若数据包含 family、重复测量或其他 cluster，置换单位和
design 都要与采样结构相符；仅把变量传给 `exchangeability_blocks` 并不能自动修正模型。

## 5. 六种常见写法

下面只列出 design 和 contrast 的变化。`analysis` 和 `edge_sets` 沿用本页开头的对象。

### Age

```python
design = make_design(continuous={"age": participants["age"]})
contrasts = {
    "age": Contrast(
        {"age": 1},
        effect_size="partial_r",
        positive_direction="connectivity increases with age",
    )
}

fit = analysis.glm(
    design,
    contrasts,
    n_permutations=10_000,
    random_state=42,
)
age_result = fit["age"]
```

进入排序的是 Pearson $r$。

### Age + covariates

```python
design = make_design(
    indicators={
        "male": participants["sex"] == "male",
        "site_B": participants["site"] == "B",
        "site_C": participants["site"] == "C",
    },
    continuous={
        "age": participants["age"],
        "motion": participants["mean_fd"],
    },
)

contrasts = {
    "age": Contrast(
        {"age": 1},
        effect_size="partial_r",
        positive_direction="connectivity increases with age",
    )
}

fit = analysis.glm(design, contrasts, n_permutations=10_000, random_state=42)
```

进入排序的是控制 sex、site 和 motion 后的 partial $r$。没有 interaction 时，连续变量
中心化只改变截距的解释，不改变 age 的 slope、t 或 partial $r$。

### G1 vs. control

```python
design = make_design(
    indicators={
        "control": participants["diagnosis"] == "control",
        "g1": participants["diagnosis"] == "g1",
    },
    add_intercept=False,
)

contrasts = {
    "g1_vs_control": Contrast(
        {"g1": 1, "control": -1},
        effect_size="hedges_g",
        positive_direction="g1 > control",
    )
}

fit = analysis.glm(design, contrasts, n_permutations=10_000, random_state=42)
```

这是两组共同 residual SD 标准化后的 Hedges' $g$。

### G1 vs. control + covariates

```python
design = make_design(
    indicators={
        "control": participants["diagnosis"] == "control",
        "g1": participants["diagnosis"] == "g1",
        "male": participants["sex"] == "male",
        "site_B": participants["site"] == "B",
        "site_C": participants["site"] == "C",
    },
    continuous={
        "age": participants["age"],
        "motion": participants["mean_fd"],
    },
    add_intercept=False,
)

contrasts = {
    "g1_vs_control": Contrast(
        {"g1": 1, "control": -1},
        effect_size="hedges_g",
        positive_direction="g1 > control",
    )
}

fit = analysis.glm(design, contrasts, n_permutations=10_000, random_state=42)
```

这里的 $g$ 使用调整后的 `g1 - control` 和完整模型的 residual SD。

### G1 vs. control 与 G2 vs. control

```python
design = make_design(
    indicators={
        "control": participants["diagnosis"] == "control",
        "g1": participants["diagnosis"] == "g1",
        "g2": participants["diagnosis"] == "g2",
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

fit = analysis.glm(design, contrasts, n_permutations=10_000, random_state=42)
```

两个 contrasts 来自同一个三组模型，residual SD 也来自这一个模型。

### G1 vs. control 与 G2 vs. control + covariates

```python
design = make_design(
    indicators={
        "control": participants["diagnosis"] == "control",
        "g1": participants["diagnosis"] == "g1",
        "g2": participants["diagnosis"] == "g2",
        "male": participants["sex"] == "male",
        "site_B": participants["site"] == "B",
        "site_C": participants["site"] == "C",
    },
    continuous={
        "age": participants["age"],
        "motion": participants["mean_fd"],
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

fit = analysis.glm(
    design,
    contrasts,
    n_permutations=10_000,
    random_state=42,
    correction_family_id="primary-diagnosis-contrasts",
)

print(fit.to_frame().sort_values(["contrast_name", "q_value"]))
```

这里有两个 contrasts，但只需拟合一次分析。两组结果属于同一个 correction family，BH
覆盖两个 contrasts 和全部有效 edge sets。

## 6. 如果只有 edge statistics

没有个体 connectome 时，LENS 仍可从一张完整的有符号边表计算 ES 和 leading edge：

```python
from conlens import lens_enrich

result = lens_enrich(
    edges,
    edge_sets,
    positive_direction="case > control",
    store_running_sum=True,
)
```

这时结果是描述性的，`NES`、`p_value` 和 `q_value` 都是 `None`。

若使用内置 edge permutation：

```python
result = lens_enrich(
    edges,
    edge_sets,
    null_method="edge_permutation",
    n_permutations=10_000,
    random_state=42,
    positive_direction="case > control",
)
```

它随机打乱 statistic 与 edge ID 的对应关系。集合大小和集合重叠仍在，但共享节点、网络
拓扑、空间结构和 edge covariance 都被破坏。因此它回答的是 competitive edge-label
问题，不能当作 subject-level GLM permutation 的近似替代。

如果外部软件已经按正确研究设计生成了 null ES、edge-statistic matrices 或 rank
matrices，可以使用 `null_method="provided_null"`。Observed 与 null 必须有相同的 edge
顺序、edge sets 和方向定义；rank matrix 只支持 unweighted enrichment。外部 null 保留
了什么结构，完全取决于它的生成过程。

只有汇总 edge statistics、又无法建立合理 null 时，报告 ES、方向和 leading edge 即可，
不要给描述性结果补上推断性表述。完整代码见[只有 edge statistics 时](/tutorials/edge-statistics)。
