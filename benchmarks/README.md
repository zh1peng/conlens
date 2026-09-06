# Null-path benchmark

## Scientific validation

Run `python -m benchmarks.validate_inference` from the repository root for the
remediation validation. Defaults are 400 independent datasets per scenario, 399
permutations, four workers, and 40 fixed bootstrap draws evaluated with three seed
streams at both 399 and 1,599 inner permutations. The JSON record contains all raw
calibration P/q values, seeds, Wilson intervals, background-signal results, stability
summaries, dependency versions, Git HEAD/dirty state, and source-file SHA-256 hashes.
The report in `website/guide/validation.md` distinguishes limited simulation evidence
from software correctness and does not claim validity for arbitrary dependence or
clustered study designs. These statistical runs are separate from fast CI tests.

After the statistical run, `python -m benchmarks.render_validation` generates the
website numeric tables directly from the recorded results.

## Performance only

Run `python -m benchmarks.benchmark_running_sum` to check time and peak memory for
100,000 edges. The existing 5-second and 50-MiB limits are machine-dependent and
are kept outside the default test suite; CI retains the large-array correctness test.

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
