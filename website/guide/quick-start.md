# 运行第一个分析

本例用明确标注的模拟数据，完成调整年龄的组间分析和调整组别的年龄关联分析。数据包含受试者连接矩阵、表型、字符串节点标签和两个网络，不需要下载影像数据。受试者相互独立；内置 Freedman–Lane 流程要求所设约化模型的误差满足相应可交换性条件。

## 运行

按[安装说明](/guide/installation)克隆仓库并安装后，在仓库根目录执行：

```bash
python -m examples.teaching_workflow
```

脚本完成 GLM、399 次置换与联合 BH，还运行描述性、外部零分布对齐、模拟脑区注释集合和 4 次 bootstrap 示例。次数用于演示执行流程，不能作为正式分析精度建议或统计校准证据。

## 完整的主分析

以下代码直接引用实际执行的脚本。导入后调用 `first_analysis()`，用 `example["fit"].to_frame()` 查看结果。

```python
import numpy as np
from conlens import (
    Contrast, lens_enrich, lens_fl_permute, lens_glm, lens_stat,
    make_design, make_network_pair_sets, matrix_to_edges,
)
```

<<< ../../examples/teaching_workflow.py#first-analysis

```python
example = first_analysis()
fit = example["fit"]
print(fit.to_frame())
```

观测、置换和 bootstrap 复用相同的节点标签、方向和对角线设置。边 ID 来自边表；即使标签是字符串，默认 ID 仍按固定节点顺序编码，例如 `0--1`，不要自行用标签拼接 ID。

## 实际输出

下表由本仓库示例实际生成；重新运行脚本会同步更新该文件。数值末位可能随数值依赖版本变化。

<<< ../generated/first-analysis.txt

以 q≤0.05 为教学阈值时，患者相对对照的 A--A 集合显著，年龄对比的 A--A 集合不显著。下图来自同一数据和设置；上行为组间对比，下行为年龄对比。左侧显示排序累积曲线，右侧显示观测 ES 与置换分布。图中 A--A 的连接定位不是逐边显著性检验。

![模拟数据中组间与年龄对比的累积曲线和零分布](/figures/teaching-analysis.png)

图由 `python -m examples.teaching_plot` 生成；本例的数值用于解释结果表，不代表真实研究发现。

先查看 `status`，再查看校正后的 `q_value`、ES/NES 和 leading-edge 大小。`ok` 表示集合符合评分和大小要求；是否完成推断还要查看 `inference_status`。本次所有纳入的“统计对比 × 边集合”一起进行 BH 校正。

`family_name="teaching"` 只给本次检验家族命名。分开运行的两次分析不会因同名而自动联合校正。结果保存在 `website/generated/first-analysis.json`，可用 `GLMResult.load()` 读取；bootstrap 文件含自身的完整观测参照。

## 换入自己的数据

先按[数据准备](/guide/data-and-sets)对齐受试者 ID，再替换矩阵、表型、节点标签和网络归属。不要先按逐边 P 值筛边。默认 `max_size=None` 不限制集合上限；只有研究设计确实要求时才设置上限。

没有集合显著也可正常报告阴性结果；被大小限制过滤的集合和只有描述性结果的集合需要分别标注。下一步阅读[结果表与连接定位](/guide/results)。
