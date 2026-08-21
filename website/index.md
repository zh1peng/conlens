<img class="conlens-doc-logo" src="/conlens-logo.png" alt="ConLens 标志">

# 连接组全排序富集与 leading-edge 网络

`ConLens` 从完整的有符号边排序中检验预先定义的连接集合，并找出推动富集峰值或谷值的 leading edges。分析前不按 edge-wise P 值筛边。

有个体 connectome 时，它先用 GLM 计算每条边的 partial $r$ 或 model-adjusted Hedges' $g$，再用 subject-level permutation 重跑边排序和 LENS。

## 先看 LENS 需要什么

LENS 使用一套固定的边、每条边的 signed statistic，以及事先定义的 edge sets。详见
[LENS 的输入是什么](/guide/introduction)。

如果有个体 connectomes，从[从个体数据到 LENS](/tutorials/design-and-contrasts)开始。这里
会把 design、contrast、edge-wise effect 和 Freedman–Lane permutation 连成一条分析流程。

如果只有已经计算好的边统计量，再看[只有 edge statistics 时](/tutorials/edge-statistics)。
这一条路线可做描述性 LENS、edge permutation 或 provided-null inference，但不再拥有
subject-level model 的信息。

已经完成正式推断、准备评估采样稳定性时，再进入[全流程 bootstrap 稳定性](/tutorials/stability)。

第一次使用建议先完成[安装](/guide/installation)和[五分钟快速开始](/guide/quick-start)。

## 输出包括什么

- 每个 edge set 的 ES、NES、P、q 和方向；
- running-sum 曲线及 leading edges；
- 完整边排序、效应量和模型元数据；
- 可选的 subject-bootstrap 稳定性结果。

## 结果读到哪一层

::: warning 网络级结果不是单边显著性
显著的网络集合表示该集合的边在完整排序中出现系统性聚集，不表示 leading-edge 中每条边都具有 edge-wise 显著性，也不应被解释为逐边因果证据。
:::

零模型需要与数据层级和研究设计对应。报告结果时保留 `positive_direction`、correction
family、edge universe 和 permutation scheme。参见[Permutation 与零模型](/guide/inference)
及[如何解释结果](/guide/interpretation)。

## 选择文档路线

- [LENS 的输入是什么](/guide/introduction)
- [数据与 edge sets](/guide/data-and-sets)
- [结果与 leading edge](/guide/results)
- [Python API](/reference/api)
- [命令行工具](/reference/cli)
- [English documentation](/en/)
