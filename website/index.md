<img class="conlens-doc-logo" src="/conlens-logo.png" alt="ConLens 标志">

# 连接组全排序富集与 leading-edge 网络

`ConLens` 面向 connectome-wide 数据，在不预先阈值化边统计量的前提下，从完整有符号排序中检验预先定义的网络连接集合，并提取真正驱动富集峰值或谷值的连接。

它将确定性的 LENS 统计量、零假设推断、BH 校正、leading-edge 网络和全流程 bootstrap 稳定性放在一条可审计工作流中。

## 你现在拥有什么数据？

- **已经计算好的边统计量**：从[边统计量输入教程](/tutorials/edge-statistics)开始。
- **受试者 × 边数据与两组标签**：进入[受试者两组分析](/tutorials/subject-two-group)。
- **需要控制协变量的受试者数据**：进入[含协变量的 GLM](/tutorials/glm)。
- **已经完成正式推断并希望评估复现性**：进入[全流程 bootstrap 稳定性](/tutorials/stability)。

第一次使用建议先完成[安装](/guide/installation)和[五分钟快速开始](/guide/quick-start)。

## ConLens 会做什么？

- 使用全部有效连接的有符号统计量进行网络级排序富集；
- 显式区分 edge permutation、label permutation、Freedman–Lane 和 provided null；
- 在同一校正 family 中对集合结果执行 BH 校正；
- 提取并导出 leading-edge 边、节点和分析元数据；
- 在受试者 bootstrap 中重新拟合边模型、内部零假设推断、LENS 与 BH。

## 必须保留的统计边界

::: warning 网络级结果不是单边显著性
显著的网络集合表示该集合的边在完整排序中出现系统性聚集，不表示 leading-edge 中每条边都具有 edge-wise 显著性，也不应被解释为逐边因果证据。
:::

正式推断必须明确选择与研究设计匹配的零模型。需要报告或复核结果时，请同时保留 `positive_direction`、校正 family、edge universe 和排列方案。参见[推断与零模型](/guide/inference)及[如何解释结果](/guide/interpretation)。

## 选择文档路线

- [认识 ConLens](/guide/introduction)
- [数据与 edge sets](/guide/data-and-sets)
- [结果与 leading edge](/guide/results)
- [Python API](/reference/api)
- [命令行工具](/reference/cli)
- [English documentation](/en/)
