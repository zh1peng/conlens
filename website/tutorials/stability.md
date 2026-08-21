# Bootstrap 稳定性

`lens_bootstrap` 不是在最终 ES 上单独抖动几次。每个 outer replicate 都会：

1. 重抽受试者；
2. 重拟合全部 edge-wise GLM contrasts；
3. 重跑 Freedman–Lane null；
4. 重新计算 observed/null LENS statistics；
5. 在同一个 `family_name` 下重新做联合 BH；
6. 与原始样本结果比较 set detection、方向与 leading-edge inclusion。

```python
from conlens import lens_bootstrap

stability = lens_bootstrap(
    connectomes,
    edge_sets,
    design=design,
    contrasts=contrasts,
    n_bootstraps=1_000,
    n_permutations=10_000,
    strata=diagnosis,
    exchangeability_blocks=site,
    random_state=42,
    n_jobs=-1,
    family_name="primary-model",
)

age_stability = stability["age"]
age_stability.set_summary
age_stability.edge_summary
age_stability.replicate_summary
```

`strata` 控制 outer subject bootstrap，例如让各诊断组样本量保持不变；
`exchangeability_blocks` 控制每个 replicate 内的 FL permutation。两者不是同一个概念。

Replicate 会边生成边汇总。ConLens 只保留 set/edge 计数和用于报告的逐 replicate set-level
记录，不会同时保存 1,000 份带完整 ranked edges 与 null scores 的结果。

## 三个不同的频率

设总 bootstrap 次数为 $B$，某 observed-significant set 在 $M$ 次 replicate 中再次 BH 显著且方向
相同。Set stability 为：

$$
\text{set stability}=\frac{M}{B}.
$$

若 edge $e$ 在其中 $K_e$ 次进入 replicate leading edge：

$$
\text{conditional stability}_e=\frac{K_e}{M},
\qquad
\text{full-pipeline stability}_e=\frac{K_e}{B}.
$$

因此 $M>0$ 时，full-pipeline stability 等于 set stability × conditional stability。反方向显著和
未检出的 replicate 对 full-pipeline edge inclusion 都贡献 0。

ConLens 报告 Jeffreys Monte Carlo bounds 来说明有限 bootstrap 次数下的频率误差。它们不是
edge truth 的置信区间。当前只实现独立或分层 subject bootstrap；cluster bootstrap、checkpoint
和 resume 尚未实现。
