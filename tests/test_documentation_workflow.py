import numpy as np
import pytest

from conlens import lens_enrich, lens_fl_permute, lens_stat
from examples.teaching_workflow import advanced_analysis, first_analysis


def test_shared_teaching_workflows():
    example = first_analysis(n_permutations=19)
    advanced = advanced_analysis(example, n_bootstraps=2)
    observed_fit = example["fit"]["patient_vs_control"]
    assert observed_fit.metadata["node_order"][0] == "LH-A1"
    np.testing.assert_allclose(observed_fit.null_scores, advanced["external"].null_scores)
    np.testing.assert_allclose(
        observed_fit.to_frame()["p_value"], advanced["external"].to_frame()["p_value"],
    )
    assert advanced["descriptive"]["age"].metadata["inference_status"] == "descriptive"
    assert advanced["descriptive"].to_frame()["q_value"].isna().all()
    assert advanced["filtered"].to_frame()["status"].eq("filtered").all()
    assert advanced["molecular"].sets[0].set_size_effective == 6
    assert advanced["molecular"].sets[0].status == "ok"
    reference = advanced["stability"]["age"].observed_reference
    assert reference.metadata["min_size"] == 5
    assert reference.metadata["max_size"] is None
    assert reference.metadata["family_name"] == "teaching"
    # Regression for omitted string node labels: retain the compatibility guard.
    wrong_options = {k: v for k, v in example["model_options"].items() if k != "node_labels"}
    null = lens_fl_permute(example["connectomes"], **wrong_options, n_permutations=1)
    with pytest.raises(ValueError, match="node_identity_hash"):
        lens_enrich(
            example["observed"], (lens_stat(item, example["edge_sets"]) for item in null),
            **example["inference_options"],
        )
