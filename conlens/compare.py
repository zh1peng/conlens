"""Comparison of LENS results and leading-edge membership."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .results import LensResult

_IDENTITY_FIELDS = ("edge_universe_hash", "edge_mapping_hash", "node_identity_hash")


def _validate_comparable(first: LensResult, second: LensResult) -> None:
    for field in _IDENTITY_FIELDS:
        left = first.metadata.get(field)
        right = second.metadata.get(field)
        if left is None or right is None:
            raise ValueError(f"both results must contain {field!r} metadata")
        if left != right:
            raise ValueError(f"results have incompatible {field}")


def compare_leading_edges(
    first: LensResult,
    second: LensResult,
    set_name: str,
) -> dict[str, Any]:
    _validate_comparable(first, second)
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
    _validate_comparable(first, second)
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
