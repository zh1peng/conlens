import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")


@pytest.fixture
def example_edges():
    return pd.DataFrame(
        {
            "node1": [0, 0, 0, 1, 1, 2],
            "node2": [1, 2, 3, 2, 3, 3],
            "statistic": [3.0, 2.0, 1.0, -0.5, -1.5, -2.5],
        }
    )


@pytest.fixture
def example_sets():
    return {
        "positive": {"0--1", "0--2", "1--3"},
        "negative": {"0--2", "1--3", "2--3"},
    }
