# 五分钟快速开始

下面用一个含诊断组、年龄和性别的模型跑完整流程。假定 `connectomes` 的形状是
`(subjects, nodes, nodes)`，`edge_sets` 是 `{set_name: edge_id 集合}`。

```python
from conlens import (
    Contrast,
    lens_enrich,
    lens_fl_permute,
    lens_glm,
    lens_stat,
    make_design,
)

design = make_design(
    groups={
        "control": diagnosis == "control",
        "g1": diagnosis == "g1",
        "g2": diagnosis == "g2",
    },
    indicators={"sex": sex},
    continuous={"age": age},
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

true_edges = lens_glm(
    connectomes,
    design=design,
    contrasts=contrasts,
    node_labels=node_labels,
)
observed = lens_stat(
    true_edges,
    edge_sets,
    store_running_sum=True,
)

null_edges = lens_fl_permute(
    connectomes,
    design=design,
    contrasts=contrasts,
    n_permutations=10_000,
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
print(fit.to_frame())
```

`fit["age"]` 是一个 `LensResult`。其中 `null_scores` 只有
`n_permutations × n_tested_sets` 大小，可直接用于 observed ES 与 null ES 的比较；逐边 null
统计量在每次 `lens_stat` 后即被释放。

`family_name` 是这次联合检验的可读标签。所有 contrast × 合法 edge set 的 P 值在同一
family 内做一次 BH，不要用相同名字暗示两次独立运行曾经联合校正。
