<img class="conlens-doc-logo" src="/conlens-logo.png" alt="ConLens 标志">

# 从边排序看到网络结构

ConLens 用完整的有符号边排序检验预先定义的 edge sets，并把推动富集峰值或谷值的
leading edges 还原为网络。它不先按 edge-wise P 值筛边，也不把 leading edge 当作逐边显著结果。

## 一条清楚的分析链

```text
个体 connectomes ── lens_glm ── true edge statistics
                                         │
                                         ├── lens_stat ── observed LENS statistics
                                         │
                                         └── lens_fl_permute ── lens_stat ── null LENS statistics
                                                                               │
                                                                               ▼
                                                                          lens_enrich
```

如果手里只有已经算好的边统计量，把 `lens_glm` 换成 `make_edge_statistics`，把
`lens_fl_permute` 换成 `lens_edge_permute`。两个入口最终都进入同一个 `lens_stat` 和
`lens_enrich`，统计定义不会分叉。

## 从哪里开始

- 第一次使用：先读[五分钟快速开始](/guide/quick-start)。
- 要建立 age、诊断组和协变量模型：看[Design、contrast 与效应量](/tutorials/design-and-contrasts)。
- 想知道 null 如何生成和为何不会占满内存：看[Permutation 与推断](/guide/inference)。
- 要画 connectome heatmap、富集 heatmap、running sum 或 circos：看[可视化](/tutorials/visualization)。
- 要评估结果对受试者抽样的敏感性：看[Bootstrap 稳定性](/tutorials/stability)。

::: warning 网络级结果不是单边显著性
网络集合显著，表示集合中的边在完整排序里出现系统性聚集；它不意味着 leading-edge
里的每条边都通过了逐边检验。
:::
