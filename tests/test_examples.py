import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "script",
    ["edge_statistics.py", "subject_workflows.py", "nilearn_and_stability.py"],
)
def test_reproducible_example(script):
    runpy.run_path(Path(__file__).parents[1] / "examples" / script)
