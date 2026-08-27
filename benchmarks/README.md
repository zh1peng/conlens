# Null-path benchmark

`benchmark_null_path.py` compares the NumPy fast path with the materialized Pandas path that
implements the ConLens 2.0.0 calculation. Both paths receive the same generated data and random
seed. The script aborts unless their complete ES sequences are exactly equal according to
`numpy.array_equal`.

Run the representative benchmark with:

```bash
python benchmarks/benchmark_null_path.py \
  --mode both --nodes 120 --subjects 60 --sets 12 \
  --set-size 400 --permutations 50
```

On the Windows development host used for ConLens 2.0.1, one run produced:

| Path | NumPy | Materialized Pandas | Speedup |
| --- | ---: | ---: | ---: |
| Edge permutation + LENS | 0.422 s | 7.478 s | 17.73× |
| Freedman–Lane + LENS | 0.935 s | 8.058 s | 8.61× |

These timings characterize this workload and machine; they are not a universal performance
guarantee. The public result objects remain Pandas-backed when inspected or serialized. The
optimization removes Pandas only from the repeated null calculation.
