---
layout: home

hero:
  name: ConLens
  text: 连接组全排序富集与 leading-edge 网络
  tagline: 不预先阈值化边统计量；明确选择零模型；从完整有符号排序中识别网络级富集，并追踪真正驱动富集的连接。
  image:
    src: /conlens-logo.png
    alt: ConLens 标志
  actions:
    - theme: brand
      text: 开始使用
      link: /guide/introduction
    - theme: alt
      text: 中文教程
      link: /tutorials/edge-statistics
    - theme: alt
      text: GitHub
      link: https://github.com/zh1peng/conlens

features:
  - title: 完整边排序
    details: 使用所有有效连接的有符号统计量，不以任意 edge-wise 阈值切断渐进证据。
  - title: 显式统计推断
    details: 将确定性的 LENS 统计量与 edge permutation、label permutation、Freedman–Lane 或 provided null 清晰分离。
  - title: Leading-edge 网络
    details: 提取推动富集峰值或谷值的连接，并保留节点、边及分析元数据以便复核。
  - title: 全流程稳定性
    details: 在受试者 bootstrap 中重新拟合边模型、零假设推断、LENS 与 BH，区分集合稳定性和边稳定性。
---

## ConLens 解决什么问题？

连接组研究经常先对数千条边逐一检验，再用阈值筛选“显著连接”。这种流程容易丢失网络层面的渐进证据，也会让结果强烈依赖阈值。ConLens 将每条边视为排序单元，检验预先定义的网络连接集合是否聚集在完整排序的正端或负端。

::: tip 核心边界
显著的网络级富集不意味着 leading-edge 中的每条边都具有 edge-wise 显著性。ConLens 明确保留这一区别。
:::

从 [认识 ConLens](/guide/introduction) 开始，或直接进入 [五分钟快速开始](/guide/quick-start)。
