# 命令行工具

CLI 面向“已有 signed edge statistics”的路线：

```bash
conlens edges.csv sets.json result.json \
  --positive-direction "connectivity increases with age" \
  --n-permutations 10000 \
  --random-state 42 \
  --family-name age-network-pairs \
  --min-size 5 --max-size 500 \
  --store-running-sum
```

`edges.csv` 至少包含 `node1,node2,statistic`；`sets.json` 是 set name 到 edge IDs 数组的映射。
`--n-permutations 0` 只输出描述性 ES/leading edge。CLI 使用 edge-label permutation；需要
subject-level GLM/FL 时应使用 Python API。
