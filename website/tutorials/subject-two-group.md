---
title: 受试者两组分析
description: 使用 label permutation 分析 subject-level connectomes
---

# 教程：受试者两组分析

本教程展示如何从 `(subjects, nodes, nodes)` 的 connectome 数组开始，并使用共享的 subject-label permutation。

## 1. 建立 edge sets

```python
import numpy as np

from conlens import LensAnalysis, make_network_pair_sets, matrix_to_edges

# connectomes.shape == (n_subjects, n_nodes, n_nodes)
labels = ["A", "B", "C", "D"]
node_networks = {
    "A": "DMN",
    "B": "DMN",
    "C": "VIS",
    "D": "VIS",
}

template = matrix_to_edges(connectomes[0], node_labels=labels)
edge_sets = make_network_pair_sets(template, node_networks)
```

集合在看当前组间统计结果之前定义。`node_labels` 顺序之后必须保持不变。

## 2. 创建分析对象

```python
analysis = LensAnalysis.from_subject_connectomes(
    connectomes,
    edge_sets,
    node_labels=labels,
    min_size=1,
    correction_family_id="primary-network-pairs",
)
```

## 3. 运行两组检验

```python
group = np.asarray(group)
site = np.asarray(site)

result = analysis.two_group(
    group,
    null_method="label_permutation",
    exchangeability_blocks=site,
    n_permutations=10_000,
    random_state=42,
)

print(result.to_frame().sort_values("q_value"))
```

ConLens 在每次合法 label permutation 中使用同一组重排标签计算所有边统计量，然后重新排序、计算每个 set 的 ES，并对 observed family 做 BH 校正。

## `positive_direction`

对于两个可排序的 group level，默认方向记录为较高 level 大于较低 level。若这个默认字符串不够符合研究语义，可显式覆盖：

```python
result = analysis.two_group(
    group,
    null_method="label_permutation",
    n_permutations=10_000,
    random_state=42,
    positive_direction="patients > controls",
)
```

## Exchangeability blocks

`exchangeability_blocks=site` 表示标签只能在合法 block 内交换。它不是“控制 site covariate”的替代品；如需估计并控制 nuisance effects，应使用 GLM 和 Freedman–Lane。

## 可视化与保存

```python
from conlens import build_leading_network
from conlens.plotting import plot_enrichment

set_name = "DMN--VIS"
plot_enrichment(result, set_name, edge_sets[set_name])

leading = build_leading_network(result, set_name)
leading.save("dmn-vis-leading.json")
result.save("two-group-result.json")
```

若要评估采样稳定性，继续阅读 [Bootstrap 稳定性教程](/tutorials/stability)。

