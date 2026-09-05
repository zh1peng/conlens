import numpy as np
import pandas as pd
import pytest

from examples.align_subjects import align_subjects


def test_alignment_uses_manifest_order_and_records_exclusions(tmp_path):
    paths = []
    for i in range(3):
        path = tmp_path / f"{i}.npy"
        np.save(path, np.full((2, 2), i))
        paths.append(str(path))
    manifest = pd.DataFrame({
        "subject_id": ["c", "a", "b"], "matrix_path": paths,
        "included": [True, False, True], "exclusion_reason": ["", "motion", ""],
    })
    phenotypes = pd.DataFrame({"subject_id": ["a", "b", "c"], "age": [30, 40, 50]})
    values, aligned, excluded, unused = align_subjects(manifest, phenotypes, ["L", "R"])
    assert aligned["subject_id"].tolist() == ["c", "b"]
    assert aligned["age"].tolist() == [50, 40]
    np.testing.assert_array_equal(values[:, 0, 0], [0, 2])
    assert excluded["subject_id"].tolist() == unused["subject_id"].tolist() == ["a"]
    with pytest.raises(ValueError, match="unique"):
        align_subjects(manifest, pd.concat([phenotypes, phenotypes]), ["L", "R"])
    with pytest.raises(ValueError, match="missing phenotypes"):
        align_subjects(manifest, phenotypes.iloc[:2], ["L", "R"])
    with pytest.raises(ValueError, match="reason"):
        align_subjects(manifest.assign(exclusion_reason=""), phenotypes, ["L", "R"])
    with pytest.raises(ValueError, match="node count"):
        align_subjects(manifest, phenotypes, ["L"])
