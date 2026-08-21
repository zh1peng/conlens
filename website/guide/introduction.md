# LENS 吃进去的是什么

LENS 本身只需要三样东西：固定的 edge universe、每条边的有符号统计量，以及事先定义的
edge sets。统计量越大，边越靠近排序顶部；越负，越靠近底部。`positive_direction` 必须写清楚，
否则正负方向没有科学含义。

## 三层数据，不要混在一起

1. `EdgeStatistics`：一行一条边，保存 signed statistic、端点和模型来源。
2. `LensStatResult`：对某个完整边排序计算 ES、running sum 和 leading edge；observed 与 null
   使用完全相同的 `lens_stat`。
3. `LensResult`：`lens_enrich` 加入 set-size filter、null normalization、经验 P 值和 BH q 值。

这三层的分工是刻意的。`lens_glm` 不知道 edge sets；`lens_stat` 不知道某次输入是 observed
还是 null；`lens_enrich` 不拟合模型，也不生成 permutation。

## 两种起点

有个体 connectomes 时，用 `make_design`、`Contrast` 和 `lens_glm` 得到边统计量；对应的 null
由 `lens_fl_permute` 生成。只有汇总边统计量时，用 `make_edge_statistics`，必要时采用
`lens_edge_permute`。后一种 null 不保留受试者层面的协方差、空间和拓扑依赖，因此解释更受限。

下一步可直接进入[快速开始](/guide/quick-start)，或者先了解[数据与 edge sets](/guide/data-and-sets)。
