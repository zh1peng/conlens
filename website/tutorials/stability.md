---
title: Bootstrap 稳定性
description: 运行 observed-aware full-pipeline subject bootstrap
---

# 教程：Full-pipeline bootstrap 稳定性

正式稳定性分析必须以一个完成推断和 BH 校正的 observed `LensResult` 为锚点。每次 bootstrap 都要重复边模型、内部零假设推断、排序、LENS 和 BH。

## 1. 运行 observed 分析

```python
import numpy as np
from conlens import Contrast, make_design

diagnosis = np.asarray(diagnosis)
site = np.asarray(site)

design = make_design(
    indicators={
        "control": diagnosis == "control",
        "g1": diagnosis == "g1",
        "g2": diagnosis == "g2",
    },
    continuous={"age": age},
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

observed_family = analysis.glm(
    design,
    contrasts,
    exchangeability_blocks=site,
    n_permutations=10_000,
    random_state=1,
)
observed = observed_family["g1_vs_control"]
```

Observed result 必须包含 q 值、明确的 `positive_direction` 和完整分析身份元数据。

## 2. 定义完整重拟合 callback

```python
def refit(sample, indices, fit_seed):
    # 必须重跑同一个 joint contrast family，再取出 observed contrast。
    return sample.glm(
        design.take(indices),
        contrasts,
        exchangeability_blocks=site[indices],
        n_permutations=10_000,
        random_state=fit_seed,
    )["g1_vs_control"]
```

Callback 收到：

- `sample`：数据已按 bootstrap indices 重抽的 `SubjectLensAnalysis`；
- `indices`：对应原始受试者行的索引；
- `fit_seed`：专用于该 replicate 内部推断的 seed。

`design.take(indices)` 同步索引实际 design rows，同时保留原始中心化与 construction
provenance。每个 subject-aligned permutation block 也必须使用相同 `indices`。

## 3. 运行 outer bootstrap

```python
strata = list(zip(site, diagnosis, strict=True))

stability = analysis.bootstrap_stability(
    observed,
    refit,
    n_bootstraps=1_000,
    random_state=2,
    strata=strata,
    n_jobs=-1,
    significance_alpha=0.05,
    interval_level=0.95,
    core_threshold=0.50,
    min_same_direction=30,
)
```

`strata` 控制 outer subject bootstrap；`exchangeability_blocks` 控制 inner permutation。它们承担不同任务。对于多 site 分组设计，组合 `site × diagnosis` strata 可以在每次重抽中保持各层样本数。

::: warning 当前 resampling 边界
当前 executor 支持独立受试者和分层受试者 bootstrap，不支持 family/repeated-measure cluster bootstrap，也不支持 checkpoint/resume。不要把 cluster 中的每一行当成独立 bootstrap 单元。
:::

## 4. 查看结果

```python
print(stability.set_summary)
print(stability.edges_for("DMN--VIS"))
print(stability.replicate_summary.head())

stability.save("stability.json")
```

三张表分别回答：

| 表 | 问题 |
| --- | --- |
| `set_summary` | 该 set 多常再次检出？方向是否一致？ |
| `edge_summary` | 某条边在条件分母或全部 bootstrap 分母下多常进入 leading edge？ |
| `replicate_summary` | 每个 set × replicate 的检出、方向、q、leading size 和 observed Jaccard 如何？ |

## 5. 分母定义

- `detection_rate = 任一方向 q <= alpha 的次数 / B`；
- `set_stability = 同方向且 q <= alpha 的次数 / B`；
- `direction_consistency = 同方向检出次数 / 全部检出次数`；
- `conditional_stability = 同方向检出且边进入 leading edge 的次数 / 同方向检出次数`；
- `full_pipeline_stability = 同一分子 / B`。

Observed 中未通过 `q_value <= significance_alpha` 的 set 默认不进入正式稳定性追踪。

## 6. 失败策略

第一份 refit 会先做 compatibility preflight，再运行剩余 replicate。若某次重抽导致组样本不足或 GLM 降秩，executor 会报告 replicate 编号并停止；它不会静默丢弃或重抽失败样本，因为那会改变 bootstrap 分布。

## 7. 不要混淆 legacy workflow

`bootstrap_lens` + `summarize_stability` 仍用于描述性的 leading-edge localization sensitivity。它不会自动重复高层 subject null inference，也不会按 observed BH 显著性与方向 gate，因此不能改名为 full-pipeline stability。
