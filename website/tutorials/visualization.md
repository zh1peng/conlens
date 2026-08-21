# 可视化

绘图函数直接读取 ConLens 结果对象，不会重新计算统计量。下面的函数都返回
Matplotlib `Axes`，所以可以照常规 Matplotlib 用法继续改标题、字体和版式。

## 一张图同时看 edge 和 network 结果

```python
from conlens import plot_lens_heatmap

plot_lens_heatmap(
    fit["age"],
    node_networks,
    network_order=["VIS", "SMN", "DAN", "VAN", "FPN", "DMN"],
)
```

![edge statistic 与 network enrichment 融合图](/figures/lens-heatmap.png)

下三角是完整的 signed edge statistic，上三角是 network-pair NES。气泡大小表示
`|NES|`，红蓝表示方向；`q_value <= significance_alpha` 时填充，否则留空。
这样可以在同一坐标系里核对：下面的 network-level enrichment 是由哪些 edge 模式推动的。

## 单独查看 connectome / edge statistic matrix

```python
from conlens import plot_connectome_heatmap

plot_connectome_heatmap(
    group_mean_connectome,
    node_networks,
    node_labels=node_labels,
    network_order=network_order,
)
```

![带 network annotation 的 connectome heatmap](/figures/connectome-heatmap.png)

矩阵会按 network 重排，顶部与左侧色带记录 node membership。无向矩阵默认只画
下三角，避免把同一组数值显示两次。如果确实需要完整矩阵，传入 `triangle="full"`。

## Network enrichment heatmap

```python
from conlens import plot_enrichment_heatmap

plot_enrichment_heatmap(fit["age"], value="NES")
```

![网络富集 bubble matrix](/figures/enrichment-heatmap.png)

函数识别 `A--B` 或 `A->B` set names。这张图只表达 set-level NES/q，不是
edge-wise statistic heatmap。需要在气泡上同时显示数值时，可传入 `annotate=True`。

## Running sum、边排序与 null ES

```python
from conlens import plot_enrichment, plot_running_sum, plot_null_distribution

plot_running_sum(fit["age"], "DMN--FPN")
plot_null_distribution(fit["age"], "DMN--FPN")
plot_enrichment(fit["age"], "DMN--FPN")
```

![running sum、edge ranking 与 null ES](/figures/enrichment-profile.png)

左图是 running-sum walk 和 leading-edge 区间；中图是进入 LENS 的完整 signed edge ranking；
右图直接用 `LensResult.null_scores` 比较 observed ES 与 set-level null ES。

## Leading-edge circos

```python
from conlens import build_leading_network, plot_circos

leading = build_leading_network(fit["age"], "DMN--FPN")
plot_circos(leading, node_networks)
```

![leading-edge circos](/figures/leading-circos.png)

双层外圈保留全部 network，弦只画 leading edges；红、蓝分别对应正负 edge
statistic，透明度和线宽随绝对效应变化。默认不标 node 或 network 文字，因为颜色顺序已经和
heatmap 一致；独立出图时可传入 `show_labels=True`。Circos 只回答 leading network
的拓扑结构，不替代 network-level ES、P 和 q。
