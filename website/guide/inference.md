# Permutation 与推断

## 有个体数据：Freedman–Lane

对每个 contrast $\mathbf c$，`lens_fl_permute` 先拟合与 $H_0:\mathbf c^\top\boldsymbol\beta=0$
对应的 reduced model，置换其残差，再与 reduced fitted values 相加并重拟合 full model：

$$
\mathbf Y^{*(b)} = \widehat{\mathbf Y}_0 + \mathbf P_b\widehat{\mathbf E}_0.
$$

每个 permutation 都重新计算 edge-wise effect、完整边排序和 LENS ES。多个 contrasts 共用同一
行置换，但各自有符合该 contrast 的 reduced model。

```python
null_edges = lens_fl_permute(
    connectomes,
    design=design,
    contrasts=contrasts,
    n_permutations=10_000,
    exchangeability_blocks=site,
    random_state=42,
)
null_stats = (lens_stat(item, edge_sets) for item in null_edges)
```

`exchangeability_blocks` 约束内层 permutation；它不是 bootstrap 的 `strata`，也不能替代
cluster bootstrap。

若个别边没有残差方差，ConLens 不会因此中止整个 GLM。该边继续留在 edge universe，effect
和 t 记为 0、双侧 P 记为 1，并在 observed audit table 中标记 `estimable=False`。Observed 与
每个 null replicate 因而始终使用同一组边。

## On the fly，而不是保存 edge × permutation

`lens_fl_permute` 和 `lens_edge_permute` 都返回迭代器。推荐直接把它们接到 `lens_stat`，然后交给
`lens_enrich`：

```python
fit = lens_enrich(observed, null_stats, family_name="primary-model")
```

每次 null 的边统计量在算完 ES 后即可释放。最终 `LensResult.null_scores` 只保留每个
permutation × tested set 的 ES。这正是计算 NES、经验 P 值和画 null 分布所需的信息。

## P、NES 与联合 BH

Observed ES 为正时只使用非负 null tail；为负时只使用非正 tail。经验 P 值采用 plus-one：

$$
p = \frac{1 + \#\{ES_0\text{ 至少与 }ES_{obs}\text{ 一样极端}\}}
{1 + \#\{ES_0\text{ 与 }ES_{obs}\text{ 同方向}\}}.
$$

NES 用同方向 null ES 的平均绝对值归一化。随后，`lens_enrich` 对同一次调用中的全部
`contrast × tested set` P 值做一次 Benjamini–Hochberg 校正。`family_name` 只是可读的审计标签，
不会把分开运行的结果魔法般合并为一个 family。

## 只有 edge statistics 时

`lens_edge_permute` 随机打乱 statistic 与 edge label 的对应关系。它是 competitive null，
不会保留共享节点、空间邻近、拓扑结构或跨边协方差。没有个体数据时可以用它做受限推断，
但不能把它描述成 subject-level reproduction test。完整写法见[汇总边统计量教程](/tutorials/edge-statistics)。
