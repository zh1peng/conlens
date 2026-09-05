"""Render the checked-in simulation evidence without hand-maintained numeric tables."""

import json
from pathlib import Path


def main():
    evidence = json.loads(Path("benchmarks/results/remediation.json").read_text(encoding="utf-8"))
    rows = [
        "## 本次执行结果", "",
        f"数值直接生成自保存的逐次模拟记录。各情景均为 {evidence['repetitions']} 个数据集；"
        "区间为 95% Wilson 区间。", "",
        "| 情景 | 任一 BH 拒绝次数 | 全局零假设 FDR | 95% 区间 |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, result in evidence["calibration"].items():
        lo, hi = result["interval"]
        rows.append(f"| {name} | {result['any_bh_rejection_count']} | "
                    f"{result['global_null_fdr']:.2%} | {lo:.2%}–{hi:.2%} |")
    rows += ["", "### 逐检验的未校正拒绝率", "",
             "| 情景 | 对比与集合 | 拒绝率 | 95% 区间 |", "| --- | --- | ---: | --- |"]
    for name, result in evidence["calibration"].items():
        for test in result["single_tests"]:
            lo, hi = test["interval"]
            rows.append(f"| {name} | {test['test']} | {test['rate']:.2%} | {lo:.2%}–{hi:.2%} |")
    rows += [
        "", "异方差压力情景的部分逐检验拒绝率高于名义 5%，不能用较低的总体 BH 拒绝率掩盖。"
        "这些估计包含有限重复的不确定性，且同时查看了多项检验；它们既不证明任意异方差都适用，"
        "也不支持声称所有现有分析均失效。", "",
        "### 固定目标集合、改变排序背景", "",
        "| 集合外年龄信号系数 | 目标集合 ES | P | q |", "| ---: | ---: | ---: | ---: |",
    ]
    for item in evidence["background"]:
        rows.append(f"| {item['outside_signal']:.1f} | {item['ES']:.6f} | "
                    f"{item['p_value']:.6f} | {item['q_value']:.6f} |")
    rows += ["", "目标集合的数据完全相同，但 ES 改变方向，P 也随背景改变。"
             "这是相对排序检验的性质，不能将这里的 P<0.05 自动定义为目标集合假阳性。", "",
             "### 固定抽样、改变内层种子和预算", "",
             "| 内层置换数 | 种子基数 | 对比与集合 | 集合稳定性 | 全流程/条件 core 边数 |",
             "| ---: | ---: | --- | ---: | --- |"]
    for budget, result in evidence["inner_seed_sensitivity"].items():
        for item in result["results"]:
            rows.append(f"| {budget} | {item['seed_base']} | "
                        f"{item['contrast']}:{item['set_name']} | "
                        f"{item['set_stability']:.1%} | {item['full_pipeline_core_size']} / "
                        f"{item['conditional_core_size']} |")
    rows += ["", "同一预算内三组种子使用相同观测参照；两个预算分别重算其设置一致的参照。"
             "两个预算使用完全相同的 40 组抽样索引。该强信号例中 core 数量相同，"
             "再检出频率仍受内层随机性影响，不能将其全部归因于受试者抽样。", "",
             "本次依赖版本：" + "，".join(
                 f"{k} {v}" for k, v in evidence["versions"].items()
             ) + "。", ""]
    Path("website/generated/calibration-results.md").write_text("\n".join(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
