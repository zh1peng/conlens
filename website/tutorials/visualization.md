# 可视化

绘图函数直接读取 ConLens 结果对象，不会重新计算统计量。默认配色沿用 logo 的深蓝、珊瑚红和
暖黄色；所有函数都返回 Matplotlib `Axes`，便于继续改标题、字体和版式。

## Connectome heatmap + network annotations

```python
from conlens import plot_connectome_heatmap

plot_connectome_heatmap(
    group_mean_connectome,
    node_networks,
    node_labels=node_labels,
)
```

![带 network annotation 的 connectome heatmap](/figures/connectome-heatmap.png)

矩阵会按 network 重排；顶部与左侧色带记录 node membership，白线标记 network boundaries。

## Network enrichment heatmap

```python
from conlens import plot_enrichment_heatmap

plot_enrichment_heatmap(fit["age"], value="NES")
```

![网络富集 heatmap](/figures/enrichment-heatmap.png)

函数识别 `A--B` 或 `A->B` set names。星号表示 `q_value <= significance_alpha`，格内数值默认是
NES；它不是 edge-wise statistic heatmap。

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

外圈按 network 分组，弦表示 leading edges；红、蓝分别对应正负 edge statistic，透明度和线宽随
绝对效应变化。Circos 展示的是 leading network 的结构，不替代 network-level ES、P 和 q。
