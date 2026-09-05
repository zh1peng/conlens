## 本次执行结果

数值直接生成自保存的逐次模拟记录。各情景均为 400 个数据集；区间为 95% Wilson 区间。

| 情景 | 任一 BH 拒绝次数 | 全局零假设 FDR | 95% 区间 |
| --- | ---: | ---: | --- |
| independent | 16 | 4.00% | 2.48%–6.40% |
| shared_nodes | 12 | 3.00% | 1.72%–5.17% |
| within_site | 12 | 3.00% | 1.72%–5.17% |
| heteroscedastic_unbalanced | 14 | 3.50% | 2.10%–5.79% |

### 逐检验的未校正拒绝率

| 情景 | 对比与集合 | 拒绝率 | 95% 区间 |
| --- | --- | ---: | --- |
| independent | group:first | 5.50% | 3.66%–8.19% |
| independent | group:last | 5.50% | 3.66%–8.19% |
| independent | group:overlap | 5.25% | 3.46%–7.89% |
| independent | age:first | 3.75% | 2.29%–6.09% |
| independent | age:last | 5.75% | 3.86%–8.48% |
| independent | age:overlap | 4.25% | 2.67%–6.70% |
| shared_nodes | group:first | 5.25% | 3.46%–7.89% |
| shared_nodes | group:last | 3.00% | 1.72%–5.17% |
| shared_nodes | group:overlap | 4.50% | 2.87%–7.00% |
| shared_nodes | age:first | 5.50% | 3.66%–8.19% |
| shared_nodes | age:last | 4.25% | 2.67%–6.70% |
| shared_nodes | age:overlap | 5.50% | 3.66%–8.19% |
| within_site | group:first | 5.50% | 3.66%–8.19% |
| within_site | group:last | 4.00% | 2.48%–6.40% |
| within_site | group:overlap | 5.00% | 3.26%–7.60% |
| within_site | age:first | 4.00% | 2.48%–6.40% |
| within_site | age:last | 4.75% | 3.06%–7.30% |
| within_site | age:overlap | 4.25% | 2.67%–6.70% |
| heteroscedastic_unbalanced | group:first | 4.25% | 2.67%–6.70% |
| heteroscedastic_unbalanced | group:last | 6.50% | 4.47%–9.35% |
| heteroscedastic_unbalanced | group:overlap | 3.75% | 2.29%–6.09% |
| heteroscedastic_unbalanced | age:first | 7.50% | 5.30%–10.50% |
| heteroscedastic_unbalanced | age:last | 5.50% | 3.66%–8.19% |
| heteroscedastic_unbalanced | age:overlap | 4.75% | 3.06%–7.30% |

异方差压力情景的部分逐检验拒绝率高于名义 5%，不能用较低的总体 BH 拒绝率掩盖。这些估计包含有限重复的不确定性，且同时查看了多项检验；它们既不证明任意异方差都适用，也不支持声称所有现有分析均失效。

### 固定目标集合、改变排序背景

| 集合外年龄信号系数 | 目标集合 ES | P | q |
| ---: | ---: | ---: | ---: |
| 0.0 | 0.681578 | 0.240566 | 0.721698 |
| 0.4 | -0.600000 | 0.369565 | 0.554348 |
| 0.8 | -1.000000 | 0.042105 | 0.126316 |

目标集合的数据完全相同，但 ES 改变方向，P 也随背景改变。这是相对排序检验的性质，不能将这里的 P<0.05 自动定义为目标集合假阳性。

### 固定抽样、改变内层种子和预算

| 内层置换数 | 种子基数 | 对比与集合 | 集合稳定性 | 全流程/条件 core 边数 |
| ---: | ---: | --- | ---: | --- |
| 399 | 89000 | patient_vs_control:A--A | 95.0% | 6 / 6 |
| 399 | 99000 | patient_vs_control:A--A | 90.0% | 6 / 6 |
| 399 | 109000 | patient_vs_control:A--A | 90.0% | 6 / 6 |
| 1599 | 89000 | patient_vs_control:A--A | 97.5% | 6 / 6 |
| 1599 | 99000 | patient_vs_control:A--A | 100.0% | 6 / 6 |
| 1599 | 109000 | patient_vs_control:A--A | 97.5% | 6 / 6 |

同一预算内三组种子使用相同观测参照；两个预算分别重算其设置一致的参照。两个预算使用完全相同的 40 组抽样索引。该强信号例中 core 数量相同，再检出频率仍受内层随机性影响，不能将其全部归因于受试者抽样。

本次依赖版本：conlens 2.2.0，numpy 2.4.4，scipy 1.17.1，pandas 3.0.3，python 3.12.10。
