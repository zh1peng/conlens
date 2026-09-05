"""Deterministic simulated teaching data; run from the repository root."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from conlens import (
    Contrast,
    lens_bootstrap,
    lens_enrich,
    lens_fl_permute,
    lens_glm,
    lens_stat,
    make_design,
    make_network_pair_sets,
    make_node_value_sets,
    make_null_edge_statistics,
    matrix_to_edges,
)


# region first-analysis
def first_analysis(n_permutations=399):
    # Simulated independent subjects, eight labelled nodes, two networks.
    rng = np.random.default_rng(2026)
    subject_ids = [f"sim-{i:03d}" for i in range(120)]
    diagnosis = np.repeat(["control", "patient"], 60)
    age = rng.normal(45, 8, 120)
    node_labels = [f"LH-A{i}" for i in range(1, 5)] + [f"RH-B{i}" for i in range(1, 5)]
    raw = rng.normal(size=(120, 8, 8))
    connectomes = (raw + raw.transpose(0, 2, 1)) / 2
    for i in range(8):
        connectomes[:, i, i] = 0
    for i, j in zip(*np.triu_indices(4, k=1), strict=True):
        connectomes[:, i, j] += 0.8 * (diagnosis == "patient")
        connectomes[:, j, i] = connectomes[:, i, j]
    edges = matrix_to_edges(connectomes.mean(axis=0), node_labels)
    networks = dict(zip(node_labels, ["A"] * 4 + ["B"] * 4, strict=True))
    edge_sets = make_network_pair_sets(edges, networks)
    design = make_design(
        groups={"control": diagnosis == "control", "patient": diagnosis == "patient"},
        continuous={"age": age},
    )
    contrasts = {
        "patient_vs_control": Contrast(
            {"patient": 1, "control": -1}, "hedges_g", "patient > control"
        ),
        "age": Contrast({"age": 1}, "partial_r", "connectivity increases with age"),
    }
    model_options = dict(design=design, contrasts=contrasts, node_labels=node_labels,
                         directed=False, diagonal=False)
    # Sets contain six or sixteen edges; retain the default size limits.
    inference_options = dict(min_size=5, max_size=None, family_name="teaching")
    observed_edges = lens_glm(connectomes, **model_options)
    observed = lens_stat(observed_edges, edge_sets, store_running_sum=True)
    null_edges = lens_fl_permute(
        connectomes, **model_options, n_permutations=n_permutations, random_state=42,
    )
    fit = lens_enrich(
        observed, (lens_stat(item, edge_sets) for item in null_edges), **inference_options,
    )
    return dict(
        fit=fit, observed=observed, observed_edges=observed_edges, connectomes=connectomes,
        edge_sets=edge_sets, edges=edges, model_options=model_options,
        inference_options=inference_options, diagnosis=diagnosis, subject_ids=subject_ids,
        n_permutations=n_permutations,
    )
# endregion first-analysis


# region advanced-analysis
def advanced_analysis(example, n_bootstraps=4):
    observed = example["observed"]
    edge_sets = example["edge_sets"]
    inference_options = example["inference_options"]
    descriptive = lens_enrich(observed, **inference_options)
    filtered = lens_enrich(observed, min_size=17, max_size=None)

    # Demonstrate ID alignment by exporting FL results and reversing the rows.
    # These columns inherit the FL null; the wrapper does not invent a null model.
    reference = example["observed_edges"]["patient_vs_control"]
    null = lens_fl_permute(
        example["connectomes"], **example["model_options"],
        n_permutations=example["n_permutations"], random_state=42,
    )
    null_frame = pd.DataFrame({
        i: item["patient_vs_control"].table.set_index("edge_id")["statistic"]
        for i, item in enumerate(null)
    }).iloc[::-1]
    external_null = make_null_edge_statistics(
        null_frame, reference=reference, permutation_scheme="contrast_specific_freedman_lane",
        random_state=42,
    )
    external = lens_enrich(
        observed["patient_vs_control"],
        (lens_stat(item, edge_sets) for item in external_null), **inference_options,
    )

    # Synthetic annotation values, not PET measurements.
    annotation = pd.Series(
        list(range(8, 0, -1)), index=example["model_options"]["node_labels"], name="synthetic",
    )
    molecular_sets = make_node_value_sets(
        example["edges"], annotation, keep="highest", fraction=0.5, connect="within",
    )
    molecular = lens_enrich(lens_stat(reference, molecular_sets), **inference_options)

    stability = lens_bootstrap(
        example["connectomes"], edge_sets, **example["model_options"], **inference_options,
        strata=example["diagnosis"], n_bootstraps=n_bootstraps,
        n_permutations=example["n_permutations"], random_state=42,
    )
    return dict(descriptive=descriptive, filtered=filtered, external=external,
                molecular=molecular, stability=stability)
# endregion advanced-analysis


def write_output(directory):
    example = first_analysis()
    advanced = advanced_analysis(example)
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    columns = ["contrast_name", "set_name", "status", "ES", "NES", "p_value", "q_value"]
    text = example["fit"].to_frame()[columns].to_string(index=False, float_format="%.6f")
    (output / "first-analysis.txt").write_text(text + "\n", encoding="utf-8")
    example["fit"].save(output / "first-analysis.json")
    for name, result in advanced["stability"].items():
        result.save(output / f"{name}-stability.json")
    record = {
        "data_source": "simulated, seed 2026; no real subjects",
        "subject_order_sha256": hashlib.sha256(
            json.dumps(example["subject_ids"]).encode()
        ).hexdigest(),
        "exclusions": "none in the synthetic example",
        "software_versions": example["fit"]["age"].metadata["software_versions"],
    }
    (output / "analysis-record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    write_output("website/generated")
