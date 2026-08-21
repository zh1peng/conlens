# 如何解释结果

先读方向，再读显著性。`positive_direction` 定义了正 statistic 的科学含义；ES/NES 的正负沿用
这个定义。P 和 q 回答的是 edge set 在对应 null 下是否出现异常聚集，不回答单条边是否显著。

## 应当报告

- edge statistic 的定义：partial $r$ 或 model-adjusted Hedges' $g$；
- design columns、contrast weights、连续变量是否中心化；
- `family_name`、tested sets 数量和 BH；
- permutation scheme、次数、随机种子与 exchangeability blocks；
- ES、NES、P、q、方向与 leading-edge size；
- set-size filter、edge universe、权重指数和 score type。

## 不应声称

- leading edges 是逐边校正后的显著边；
- edge-label permutation 保留了 connectome 的拓扑或空间依赖；
- bootstrap stability 是“真边概率”或未来样本的精确复现概率；
- 相同 `family_name` 能把分开运行的检验自动组成同一个 BH family。

当模型包含家系、重复测量或其他 cluster 时，当前 subject bootstrap 不是合适的 outer resampling
单位。ConLens 目前只实现独立受试者或分层受试者 bootstrap。
