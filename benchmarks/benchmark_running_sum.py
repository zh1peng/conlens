"""Check running-sum time and memory outside the default test suite.

Run: python -m benchmarks.benchmark_running_sum
"""

import time
import tracemalloc

import numpy as np

from conlens import compute_running_sum


def main():
    n_edges = 100_000
    statistics = np.linspace(5, -5, n_edges)
    hits = np.zeros(n_edges, dtype=bool)
    hits[::20] = True
    tracemalloc.start()
    started = time.perf_counter()
    compute_running_sum(statistics, hits)
    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"{n_edges:,} edges: {elapsed:.3f} s, {peak_bytes / 1024**2:.2f} MiB peak")
    assert elapsed < 5
    assert peak_bytes < 50 * 1024 * 1024


if __name__ == "__main__":
    main()
