---
title: 命令行工具
description: 使用 conlens CLI 分析边表
---

# 命令行工具

安装后可使用 `conlens` 命令：

```bash
conlens edges.csv sets.json result.json \
  --null-method edge_permutation \
  --n-permutations 10000 \
  --random-state 42
```

## 输入文件

`edges.csv` 至少包含：

```csv
node1,node2,statistic
A,B,3.0
A,C,2.0
B,C,-1.5
```

`sets.json` 将集合名称映射到 edge IDs：

```json
{
  "DMN--VIS": ["0--2", "0--3", "1--2", "1--3"]
}
```

建议先在 Python 中调用 `validate_edge_table` 生成 canonical IDs，再写出 `sets.json`，避免手工猜测端点到 ID 的映射。

## 输出

`result.json` 可以通过 Python 恢复：

```python
from conlens import LensResult

result = LensResult.load("result.json")
print(result.to_frame())
```

CLI 适合已经准备好 edge-statistics 和 set definitions 的批处理。需要统一 subject-level
GLM、contrast-specific Freedman–Lane 或 full-pipeline bootstrap 时，使用 Python API。
