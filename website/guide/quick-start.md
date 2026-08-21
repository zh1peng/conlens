---
title: 快速开始
description: 用个体 connectome、age 和 covariates 跑通一次 subject-level LENS 分析
---

# 五分钟快速开始

这个例子检验 age 与连接强度的关系。它假设磁盘上已有个体 connectomes、受试者表和
节点所属网络。

## 1. 读入数据并定义 edge sets

```python
import numpy as np
import pandas as pd

from conlens import (
    Contrast,
    LensAnalysis,
    make_design,
    make_network_pair_sets,
    matrix_to_edges,
)

connectomes = np.load("connectomes.npy")          # subjects × nodes × nodes
participants = pd.read_csv("participants.tsv", sep="\t")
labels = np.load("node-labels.npy", allow_pickle=True).tolist()

node_table = pd.read_csv("node-networks.tsv", sep="\t")
node_networks = dict(zip(node_table["node"], node_table["network"], strict=True))

template = matrix_to_edges(connectomes[0], node_labels=labels)
edge_sets = make_network_pair_sets(template, node_networks)

analysis = LensAnalysis.from_subject_connectomes(
    connectomes,
    edge_sets,
    node_labels=labels,
    min_size=5,
    store_running_sum=True,
)
```

`connectomes` 的第一维必须与 `participants` 的行一一对应。

## 2. 写出模型和 contrast

```python
design = make_design(
    indicators={
        "male": participants["sex"] == "male",
        "site_B": participants["site"] == "B",
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
```

这里每条边的排序量是控制 sex、site 和 motion 后的 signed partial $r$。

## 3. 拟合并做 permutation

```python
fit = analysis.glm(
    design,
    contrasts,
    n_permutations=10_000,
    random_state=42,
    correction_family_id="age-primary",
)

age_result = fit["age"]
print(age_result.to_frame()[
    ["set_name", "ES", "NES", "p_value", "q_value", "direction"]
])
```

每次 Freedman–Lane permutation 都会重新计算全部边的 partial $r$，然后重跑排序、
running sum 和每个 set 的 ES。`q_value` 是网络集合层面的 BH 结果，不是单边校正结果。

## 4. 看一条富集曲线

```python
from conlens.plotting import plot_enrichment

set_name = age_result.to_frame().sort_values("q_value").iloc[0]["set_name"]
plot_enrichment(age_result, set_name, edge_sets[set_name])
```

继续阅读[从个体数据到 LENS](/tutorials/design-and-contrasts)，可以看到 age、组间 contrasts
及 covariate 调整的六种写法。只有汇总边统计量时，使用[另一条输入路线](/tutorials/edge-statistics)。
