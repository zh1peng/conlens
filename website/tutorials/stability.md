# 受试者重抽样与稳定性

每次重新抽取受试者，并重新完成模型拟合、置换检验、集合富集和联合 BH，能够评估完整分析对抽样的敏感性。当前内置独立或分层受试者 bootstrap；家系和重复测量需要适当抽样单位，strata 不提供通用 cluster bootstrap。

## 复用完整设置

在仓库根目录运行以下代码；数据和设置与[快速开始](/guide/quick-start)完全相同：

```python
from examples.teaching_workflow import first_analysis
from conlens import lens_bootstrap

example = first_analysis()
stability = lens_bootstrap(
    example["connectomes"], example["edge_sets"],
    **example["model_options"], **example["inference_options"],
    strata=example["diagnosis"], n_bootstraps=4,
    n_permutations=example["n_permutations"], random_state=42,
)
result = stability["patient_vs_control"]
print(result.set_summary)
print(result.edge_summary)
result.save("patient-stability.json")
```

4 次重复仅用于运行演示，不足以评估 core。strata 在组内抽取受试者以保持组别样本量；exchangeability_blocks 则限制每次拟合内部的残差重排，两者作用不同。

## 核对真正的观测参照

lens_bootstrap 内部重跑观测样本并派生置换种子，所以参照可能与先前 fit 不同，尤其在显著性阈值附近。返回的 observed_reference 是本次稳定性分析实际采用的完整结果，包含所有集合、零分布、排序和分析设置；不是只包含显著集合的摘要。

```python
reference = result.observed_reference
print(reference.to_frame())
reference.save("bootstrap-observed-reference.json")
print(result.metadata["observed_permutation_seed"])
print(result.metadata["bootstrap_permutation_seeds"])
print(reference.metadata["min_size"], reference.metadata["max_size"])
```

结果还保存主种子、SeedSequence 的实际 entropy（主种子为 None 时也可追溯）、派生规则和每次 bootstrap 抽样索引。重新计算观测参照时，将 observed_permutation_seed 直接传给 lens_fl_permute。失败重复会报错，不会静默删除并缩小分母。

## 三种频率

设共完成 $B$ 次重抽样，其中 $M$ 次某集合再次显著且方向与观测一致，边 $e$ 在这些重复中 $K_e$ 次进入 leading edge：

$$
S_{set}=M/B,\qquad S_{conditional,e}=K_e/M,\qquad S_{full,e}=K_e/B.
$$

以下是教学数字，不是包的实际结果：1,000 次抽样中集合同方向再检出 600 次，某边其中 420 次进入 leading edge，则集合稳定性为 60%，条件稳定性为 70%，全流程稳定性为 42%。当 $M>0$ 时，$S_{full}=S_{set}S_{conditional}$；没有同方向检出时，条件频率无定义。

只追踪观测显著集合。Jeffreys 区间表示有限重复次数带来的频率不确定性，不是边为真的概率区间，也不是新队列复现概率区间。

Core 使用频率区间的下界，而不是单看点估计。条件 core 还要求达到 min_same_direction（默认 30）及集合稳定性下界门槛 0.50；因此上例 70% 不能直接判为条件 core。具体门槛见返回 metadata。

## 内层置换精度

每次重抽样还含有限置换的随机性，阈值附近的再检出频率不能全部归因于受试者抽样。replicate_summary 记录实际 n_null_tail 和 minimum_resolvable_p；观测参照也提供这些字段。

检查敏感性时，固定 metadata 中的 bootstrap_draw_indices 和观测参照，仅改变内层置换种子或置换数，再用 summarize_stability 汇总完整重复结果。低层汇总要求同一组重复拥有一致的置换数；比较不同预算时，应分别建立设置一致的观测参照与重复结果，不能混合分母。示例验证脚本的同预算种子敏感性使用同一参照，见[验证记录](/guide/validation)。
