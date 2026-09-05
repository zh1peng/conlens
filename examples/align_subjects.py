"""Join a deidentified image manifest and phenotype table without silent exclusions."""

import numpy as np


# region alignment
def align_subjects(manifest, phenotypes, node_labels):
    # manifest: subject_id, matrix_path, included, exclusion_reason
    # phenotypes: subject_id plus the covariates used in the model
    for name, frame in [("manifest", manifest), ("phenotypes", phenotypes)]:
        if frame["subject_id"].isna().any() or frame["subject_id"].duplicated().any():
            raise ValueError(f"{name}: subject IDs must be present and unique")
    if not manifest["included"].isin([True, False]).all():
        raise ValueError("included must contain explicit boolean decisions")
    excluded = manifest.loc[~manifest["included"]].copy()
    if excluded["exclusion_reason"].fillna("").str.strip().eq("").any():
        raise ValueError("record a reason for every excluded subject")
    selected = manifest.loc[manifest["included"]].copy()
    aligned = selected.merge(
        phenotypes, on="subject_id", how="left", validate="one_to_one", sort=False,
        indicator=True,
    )
    missing = aligned.loc[aligned["_merge"] != "both", "subject_id"].tolist()
    if missing:
        raise ValueError(f"selected subjects missing phenotypes: {missing}")
    aligned = aligned.drop(columns="_merge")
    matrices = [np.load(path, allow_pickle=False) for path in aligned["matrix_path"]]
    expected_shape = (len(node_labels), len(node_labels))
    if not matrices or any(matrix.shape != expected_shape for matrix in matrices):
        raise ValueError("every selected matrix must match the atlas node count")
    # Build both the array and model from this one table, in this exact order.
    connectomes = np.stack(matrices)
    unused = phenotypes.loc[~phenotypes["subject_id"].isin(selected["subject_id"])].copy()
    return connectomes, aligned, excluded, unused
# endregion alignment
