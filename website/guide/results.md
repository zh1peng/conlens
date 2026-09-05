# 结果表与连接定位

先确认集合是否进入检验，再看校正后的 P 值、ES/NES 方向与 leading-edge 大小。承接[快速开始](/guide/quick-start)：

```python
table = example["fit"].to_frame()
print(table[["contrast_name", "set_name", "status", "ES", "NES", "p_value", "q_value"]])
```

## 区分结果状态

| 状态 | 识别方法 | 允许的解释 |
| --- | --- | --- |
| 完成推断且显著 | status=ok、inference_status=complete，q 达到预设阈值 | 在所设零模型及检验家族下集合富集显著 |
| 完成推断但不显著 | 同上，但 q 未达到阈值 | 未获得足够的集合富集证据 |
| 大小过滤 | status=filtered，warnings 写明有效大小和边界 | 集合未进入检验，不能写成“不显著” |
| 集合无效 | status=invalid，例如空集或整个边范围 | 不具备当前评分定义所需的集合背景 |
| 描述性分析 | inference_status=descriptive，P/q/NES 为空 | 可描述 ES 和 leading edge，没有推断性显著性 |

`inference_status` 位于单对比结果的 metadata 中，`status` 位于每个集合结果中。即使 status=ok，也要区分完成推断与描述性分析。

以下直接使用同一可执行示例，生成描述性结果和过滤结果：

```python
from conlens import lens_enrich

descriptive = lens_enrich(example["observed"], **example["inference_options"])
filtered = lens_enrich(example["observed"], min_size=17, max_size=None)
print(descriptive["age"].metadata["inference_status"])  # descriptive
print(descriptive.to_frame()["q_value"].isna().all())  # True
print(filtered.to_frame()["status"].unique())          # ['filtered']
```

标准模式的 ES=0 表示方向不明确；完成推断时 P=1、NES 为空。它与没有提供零分布不同。

## 关键列

- ES/NES：原始富集分数与零分布归一化分数；表示相对排序方向。
- q_value：一次调用内全部纳入对比与集合联合 BH 后的 P 值。
- leading_edge_size：形成该次 ES 极值的连接数量，非逐边显著连接数。
- n_null_positive / negative / zero：严格正、严格负和零 ES 的互斥计数。
- n_null_tail、minimum_resolvable_p：实际检验尾部数量和可分辨的最小 P。
- normalization_status：NES 是否有可解释的归一化尺度。

## 查看与保存连接网络

```python
from conlens import build_leading_network

result = example["fit"]["patient_vs_control"]
network = build_leading_network(result, "A--A")
print(network.edges)
print(network.nodes)
example["fit"].save("conlens-result.json")
```

Leading edge 定位形成当前富集的集合成员，不是另一次逐边检验。没有显著集合时，若展示描述性网络，应明确标注。下一步阅读[如何解释和报告结果](/guide/interpretation)。

对象层次与序列化方法见 [API](/reference/api)；`null_scores` 保留集合层面的零分布，供核对推断和绘图。
