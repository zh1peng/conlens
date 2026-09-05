# 分析已有观测统计量

只有完整观测边统计量时，可以描述集合富集及其连接位置。表中需要全部预定有效边，而不是仅逐边显著的边。`statistic` 应有正负方向，节点顺序与集合成员必须一致。

## 最小可运行的描述性分析

```python
import pandas as pd
from conlens import make_edge_statistics, lens_stat, lens_enrich

edges = pd.DataFrame({
    "node1": [0, 0, 0, 1, 1, 2],
    "node2": [1, 2, 3, 2, 3, 3],
    "statistic": [0.61, 0.43, 0.12, -0.08, -0.37, -0.55],
})
edge_sets = {"target": {"0--1", "0--2", "0--3"}}
observed_edges = make_edge_statistics(
    edges, positive_direction="connectivity increases with age",
    statistic_name="partial correlation",
)
observed = lens_stat(observed_edges, edge_sets, store_running_sum=True)
descriptive = lens_enrich(observed, min_size=1, max_size=None)
print(descriptive.metadata["inference_status"])  # descriptive
print(descriptive.to_frame()["q_value"].isna().all())  # True
```

这里 min_size=1 是为了展示仅含 3 条边的教学集合；软件默认 min_size=5。实际分析应预先设定大小规则，max_size 默认不设上限。没有 P/q 是尚未进行推断，不是统计上不显著。

## 显著性需要适当零分布

有外部模型和符合设计的重采样结果时，优先使用[外部零分布接口](/tutorials/external-effects)。ConLens 不会从一列观测统计量恢复受试者依赖结构。

边标签置换随机打乱边统计量与边身份，要求这一标签随机化零模型符合科学问题。它不保留共享节点、空间、拓扑与跨边协方差，不能替代受试者残差置换。明确选择这个模型后，才执行：

```python
from conlens import lens_edge_permute

null_edges = lens_edge_permute(observed_edges, n_permutations=399, random_state=42)
fit = lens_enrich(
    observed, (lens_stat(item, edge_sets) for item in null_edges),
    min_size=1, max_size=None, family_name="edge-label-example",
)
```

结果应注明“边标签随机化下的富集”，不要将其称为受试者层面重现检验。评分尾部的定义见[推断说明](/guide/inference)。
