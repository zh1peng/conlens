"""Comparison of LENS results and leading-edge membership."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .results import LensResult


def compare_leading_edges(
    first: LensResult,
    second: LensResult,
    set_name: str,
) -> dict[str, Any]:
    left = set(first.get(set_name).leading_edge_ids)
    right = set(second.get(set_name).leading_edge_ids)
    union = left | right
    total = len(left) + len(right)
    return {
        "set_name": set_name,
        "intersection": sorted(left & right),
        "only_first": sorted(left - right),
        "only_second": sorted(right - left),
        "jaccard": 1.0 if not union else len(left & right) / len(union),
        "dice": 1.0 if total == 0 else 2 * len(left & right) / total,
    }


def compare_lens_results(first: LensResult, second: LensResult) -> pd.DataFrame:
    common = sorted(
        {item.set_name for item in first.sets} & {item.set_name for item in second.sets}
    )
    rows = []
    for name in common:
        left, right = first.get(name), second.get(name)
        overlap = compare_leading_edges(first, second, name)
        rows.append(
            {
                "set_name": name,
                "ES_first": left.ES,
                "ES_second": right.ES,
                "NES_first": left.NES,
                "NES_second": right.NES,
                "direction_agreement": left.direction == right.direction,
                "leading_edge_jaccard": overlap["jaccard"],
                "leading_edge_dice": overlap["dice"],
            }
        )
    return pd.DataFrame(rows)
