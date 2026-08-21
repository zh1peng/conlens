---
title: Bootstrap 稳定性
description: 运行 observed-aware full-pipeline subject bootstrap
---

# Bootstrap 稳定性

某个 set 在 observed 样本中通过了 BH，换一批受试者后还会检出吗？方向会不会改变？
Leading edges 又有多稳定？这里的 bootstrap 用来回答这些采样敏感性问题。

ConLens 所说的 full-pipeline bootstrap，是指每次重抽受试者后都重跑 edge model、
permutation、LENS 和 BH。只拿 observed leading edges 做重抽不属于这条流程。

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

`observed` 需要有 q 值、`positive_direction` 和分析身份元数据。缺少这些信息时，ConLens
无法判断 bootstrap replicate 是否来自同一个分析。

## 2. 写 refit callback

```python
def refit(sample, indices, fit_seed):
    # 先重跑同一个 joint contrast family，再取出 observed contrast。
    return sample.glm(
        design.take(indices),
        contrasts,
        exchangeability_blocks=site[indices],
        n_permutations=10_000,
        random_state=fit_seed,
    )["g1_vs_control"]
```

Callback 有三个参数：

- `sample`：数据已按 bootstrap indices 重抽的 `SubjectLensAnalysis`；
- `indices`：对应原始受试者行的索引；
- `fit_seed`：专用于该 replicate 内部推断的 seed。

`design.take(indices)` 选择相同的 design rows，并保留原始中心化信息。与受试者对齐的
permutation blocks 也要使用同一组 `indices`。

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

`strata` 用于 outer subject bootstrap；`exchangeability_blocks` 用于 inner permutation。
多 site 分组设计可以用 `site × diagnosis` 作为 strata，让每次重抽保留各层样本数。

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

结果分成三张表：

| 表 | 问题 |
| --- | --- |
| `set_summary` | set 再次检出的频率，以及方向是否一致 |
| `edge_summary` | 一条边在条件分母或全部 bootstrap 分母下进入 leading edge 的频率 |
| `replicate_summary` | 每个 set × replicate 的检出、方向、q、leading size 和 observed Jaccard |

## 5. 分母定义

- `detection_rate = 任一方向 q <= alpha 的次数 / B`；
- `set_stability = 同方向且 q <= alpha 的次数 / B`；
- `direction_consistency = 同方向检出次数 / 全部检出次数`；
- `conditional_stability = 同方向检出且边进入 leading edge 的次数 / 同方向检出次数`；
- `full_pipeline_stability = 同一分子 / B`。

Observed 中未通过 `q_value <= significance_alpha` 的 set 默认不进入正式稳定性追踪。

## 6. 失败策略

ConLens 先运行第一份 refit 并检查 compatibility，通过后才继续。某次重抽若导致组样本不足
或 GLM 降秩，程序会给出 replicate 编号并停止。失败样本不会被丢弃或重新抽取，因为那会
改变 bootstrap 分布。

## 7. 不要混淆 legacy workflow

`bootstrap_lens` + `summarize_stability` 计算的是描述性 leading-edge localization
sensitivity。它不重跑 subject-level null inference，也不按 observed BH 结果和方向筛选
replicates，所以不要把这组输出写成 full-pipeline stability。
