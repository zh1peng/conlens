# 结果对象

## `EdgeStatistics`

保存边表和产生该统计量的模型元数据。`lens_glm` 返回 `{contrast_name: EdgeStatistics}`。

## `LensStatResult`

这是 `lens_stat` 的确定性输出：完整排序、每个 edge set 的 ES、方向、peak、leading edges，
以及可选的 running sum。它不含 P 值，因为此时还没有把 observed 与 null 比较。

## `LensResult` 与 `GLMResult`

`lens_enrich` 对单一输入返回 `LensResult`；对 contrast mapping 返回 `GLMResult`。常用入口：

```python
fit.to_frame()
fit["age"].get("DMN--FPN")
fit["age"].null_for("DMN--FPN")
fit.save("conlens-result.json")
```

`null_scores` 是紧凑的 set-level null ES 表。它可用来复查 observed/null 分离、画直方图，或检查
同方向 null 的数量；它不包含逐边 null 轨迹。

## Leading network

```python
from conlens import build_leading_network

network = build_leading_network(fit["age"], "DMN--FPN")
network.edges
network.nodes
```

Leading edge 是产生当前 ES 极值的集合成员，不是另一次 edge-wise hypothesis test。
