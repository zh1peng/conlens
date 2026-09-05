"""Version-bound targeted calibration, ranking-background and inner-seed checks.

Run: python -m benchmarks.validate_inference --repetitions 400 --permutations 399 --jobs 4
This estimates operating characteristics under specified simulations, not universal validity.
"""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed

from conlens import (
    Contrast,
    __version__,
    lens_enrich,
    lens_fl_permute,
    lens_glm,
    lens_stat,
    make_design,
    summarize_stability,
)
from examples.teaching_workflow import first_analysis

SCENARIOS = ("independent", "shared_nodes", "within_site", "heteroscedastic_unbalanced")


def wilson(successes, total):
    p = successes / total
    z = 1.959963984540054
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / (1 + z * z / total)
    return [float(center - half), float(center + half)]


def fit(values, design, contrasts, sets, permutations, seed, blocks=None):
    observed = lens_stat(lens_glm(values, design=design, contrasts=contrasts), sets)
    null = lens_fl_permute(
        values, design=design, contrasts=contrasts, n_permutations=permutations,
        random_state=seed, exchangeability_blocks=blocks,
    )
    return lens_enrich(
        observed, (lens_stat(item, sets) for item in null), family_name="validation",
    )


def simulate(scenario, seed):
    rng = np.random.default_rng(seed)
    n, nodes = 48, 6
    row, col = np.triu_indices(nodes, 1)
    group = np.arange(n) < (12 if scenario == "heteroscedastic_unbalanced" else 24)
    age, nuisance = rng.normal(size=(2, n))
    site = np.tile([0, 1], n // 2)
    y = rng.normal(size=(n, len(row)))
    if scenario == "shared_nodes":
        shared = rng.normal(size=(n, nodes))
        y = (y + 0.6 * shared[:, row] + 0.6 * shared[:, col]) / np.sqrt(1.72)
    elif scenario == "within_site":
        y *= (1 + site[:, None])
        y += site[:, None]
    elif scenario == "heteroscedastic_unbalanced":
        y *= np.where(group, 3.0, 1.0)[:, None]
    y += 0.4 * nuisance[:, None]
    values = np.zeros((n, nodes, nodes))
    values[:, row, col] = values[:, col, row] = y
    continuous = {"age": age, "nuisance": nuisance}
    indicators = {"site": site} if scenario == "within_site" else None
    design = make_design(
        groups={"control": ~group, "patient": group}, continuous=continuous,
        indicators=indicators,
    )
    contrasts = {
        "group": Contrast({"patient": 1, "control": -1}, "hedges_g", "patient > control"),
        "age": Contrast({"age": 1}, "partial_r", "increases with age"),
    }
    ids = [f"{i}--{j}" for i, j in zip(row, col, strict=True)]
    sets = {"first": ids[:5], "last": ids[-5:], "overlap": ids[3:8]}
    return values, design, contrasts, sets, site if scenario == "within_site" else None


def calibration_one(scenario, seed, permutations):
    values, design, contrasts, sets, blocks = simulate(scenario, seed)
    result = fit(values, design, contrasts, sets, permutations, seed + 1, blocks)
    table = result.to_frame()
    return {
        "seed": seed, "p": table["p_value"].tolist(), "q": table["q_value"].tolist(),
        "labels": (table["contrast_name"] + ":" + table["set_name"]).tolist(),
        "minimum_resolvable_p": table["minimum_resolvable_p"].tolist(),
    }


def background_check(permutations):
    values, design, contrasts, sets, _ = simulate("shared_nodes", 77001)
    row, col = np.triu_indices(6, 1)
    age = design.frame["age"].to_numpy()
    records = []
    for outside_signal in (0, 0.4, 0.8):
        current = values.copy()
        for i, j in zip(row[5:], col[5:], strict=True):
            current[:, i, j] += outside_signal * age
            current[:, j, i] = current[:, i, j]
        np.testing.assert_array_equal(current[:, row[:5], col[:5]], values[:, row[:5], col[:5]])
        result = fit(current, design, {"age": contrasts["age"]}, sets, permutations, 77002)
        item = result["age"].get("first")
        records.append(dict(outside_signal=outside_signal, ES=item.ES, NES=item.NES,
                            p_value=item.p_value, q_value=item.q_value))
    return records


def inner_seed_check(permutations, n_draws, jobs):
    example = first_analysis(permutations)
    reference = example["fit"]
    rng = np.random.default_rng(88001)
    groups = [np.flatnonzero(example["diagnosis"] == value) for value in ("control", "patient")]
    draws = [np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
             for _ in range(n_draws)]

    def fit_draw(draw, seed):
        options = {**example["model_options"],
                   "design": example["model_options"]["design"].take(draw)}
        values = example["connectomes"][draw]
        observed = lens_stat(lens_glm(values, **options), example["edge_sets"])
        null = lens_fl_permute(values, **options, n_permutations=permutations, random_state=seed)
        return lens_enrich(
            observed, (lens_stat(item, example["edge_sets"]) for item in null),
            **example["inference_options"],
        )

    records = []
    for seed_base in (89000, 99000, 109000):
        repeats = Parallel(n_jobs=jobs)(
            delayed(fit_draw)(draw, seed_base + i) for i, draw in enumerate(draws)
        )
        for name in reference.contrast_names:
            result = summarize_stability(reference[name], [item[name] for item in repeats])
            for row in json.loads(result.set_summary.to_json(orient="records")):
                records.append({"seed_base": seed_base, "contrast": name, **row})
    return {"draw_seed": 88001, "n_draws": n_draws,
            "draw_indices": [draw.tolist() for draw in draws], "results": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=400)
    parser.add_argument("--permutations", type=int, default=399)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--bootstrap-draws", type=int, default=40)
    parser.add_argument("--sensitivity-permutations", type=int, default=1599)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/remediation.json"))
    args = parser.parse_args()
    if min(args.repetitions, args.permutations, args.bootstrap_draws,
           args.sensitivity_permutations) < 1:
        parser.error("counts must be positive")
    evidence = {
        "scope": "targeted simulation; all tested group/age coefficients zero in calibration",
        "seed_base": 61000, "alpha": 0.05,
        "repetitions": args.repetitions, "permutations": args.permutations,
        "interval": "95% Wilson binomial intervals across independent datasets",
        "design": "48 subjects, 6 nodes, group and age contrasts adjusted for nuisance",
        "family": "2 contrasts x 3 sets; first/last disjoint, third overlaps first",
        "versions": {"conlens": __version__, "numpy": np.__version__, "scipy": scipy.__version__,
                     "pandas": pd.__version__, "python": platform.python_version()},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "source_sha256": {
            str(path).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(list(Path("conlens").rglob("*.py")) +
                               [Path("benchmarks/validate_inference.py"),
                                Path("examples/teaching_workflow.py")])
        },
        "calibration": {},
    }
    for index, scenario in enumerate(SCENARIOS):
        records = Parallel(n_jobs=args.jobs)(
            delayed(calibration_one)(scenario, 61000 + index * 10000 + i * 2, args.permutations)
            for i in range(args.repetitions)
        )
        pvalues = np.array([row["p"] for row in records])
        qvalues = np.array([row["q"] for row in records])
        family_count = int((qvalues <= 0.05).any(axis=1).sum())
        single = []
        for i, label in enumerate(records[0]["labels"]):
            count = int((pvalues[:, i] <= 0.05).sum())
            single.append({"test": label, "rejections": count, "rate": count / args.repetitions,
                           "interval": wilson(count, args.repetitions),
                           "p_quantiles": np.quantile(pvalues[:, i], [0.05, 0.5, 0.95]).tolist()})
        evidence["calibration"][scenario] = {
            "single_tests": single, "any_bh_rejection_count": family_count,
            "global_null_fdr": family_count / args.repetitions,
            "interval": wilson(family_count, args.repetitions), "replicates": records,
        }
        print(f"{scenario}: global-null BH FDR={family_count / args.repetitions:.4f}", flush=True)
    evidence["background"] = background_check(args.permutations)
    evidence["inner_seed_sensitivity"] = {
        str(budget): inner_seed_check(budget, args.bootstrap_draws, args.jobs)
        for budget in dict.fromkeys([args.permutations, args.sensitivity_permutations])
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
