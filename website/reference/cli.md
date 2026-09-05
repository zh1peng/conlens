# 命令行工具

当前命令行接收完整的观测边统计量表。默认 `--n-permutations 0` 只计算描述性结果；正数显式选择边标签置换，依赖边标签可交换的零假设，不是受试者置换。需要个体 GLM/FL 或外部设计匹配零分布时使用 Python API。下面是明确选择边标签零模型后的命令（多行续行写法适用于 Bash；PowerShell 可写成一行）：

```bash
conlens edges.csv sets.json result.json \
  --positive-direction "connectivity increases with age" \
  --n-permutations 10000 \
  --random-state 42 \
  --family-name age-network-pairs \
  --min-size 5 \
  --store-running-sum
```

`edges.csv` 至少包含 `node1,node2,statistic`；`sets.json` 是 set name 到 edge IDs 数组的映射。
`--n-permutations 0` 只输出描述性 ES/leading edge。CLI 使用 edge-label permutation；需要
subject-level GLM/FL 时应使用 Python API。
