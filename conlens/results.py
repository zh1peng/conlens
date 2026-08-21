"""Serializable value objects returned by ConLens functions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _jsonable(value.item())
    if value is pd.NA:
        return {"__missing__": True}
    if isinstance(value, float) and not np.isfinite(value):
        label = "nan" if np.isnan(value) else ("inf" if value > 0 else "-inf")
        return {"__nonfinite__": label}
    if isinstance(value, pd.DataFrame):
        return {
            "__dataframe__": {
                "columns": _jsonable(list(value.columns)),
                "records": _jsonable(value.to_dict(orient="records")),
            }
        }
    if isinstance(value, tuple):
        return {"__tuple__": [_jsonable(item) for item in value]}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, set)):
        return [_jsonable(item) for item in value]
    return value


def _restore(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"__nonfinite__"}:
        return {"nan": np.nan, "inf": np.inf, "-inf": -np.inf}[value["__nonfinite__"]]
    if isinstance(value, dict) and set(value) == {"__missing__"}:
        return pd.NA
    if isinstance(value, dict) and set(value) == {"__tuple__"}:
        return tuple(_restore(item) for item in value["__tuple__"])
    if isinstance(value, dict) and set(value) == {"__dataframe__"}:
        payload = value["__dataframe__"]
        if isinstance(payload, list):
            return pd.DataFrame(_restore(payload))
        return pd.DataFrame(_restore(payload["records"]), columns=_restore(payload["columns"]))
    if isinstance(value, dict):
        return {key: _restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore(item) for item in value]
    return value


def _save_payload(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination


def _load_payload(path: str | Path) -> dict[str, Any]:
    return _restore(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(slots=True)
class EdgeStatistics:
    """One signed edge-statistic table plus its model and identity metadata."""

    table: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({"table": self.table, "metadata": self.metadata})

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EdgeStatistics:
        data = _restore(payload)
        return cls(table=data["table"], metadata=data.get("metadata", {}))

    def save(self, path: str | Path) -> Path:
        return _save_payload({"table": self.table, "metadata": self.metadata}, path)

    @classmethod
    def load(cls, path: str | Path) -> EdgeStatistics:
        return cls.from_dict(_load_payload(path))


@dataclass(slots=True)
class LensSetResult:
    set_name: str
    set_size_input: int
    set_size_effective: int
    ES: float | None
    ES_positive: float | None
    ES_negative: float | None
    direction: str | None = None
    peak_rank: int | None = None
    peak_fraction: float | None = None
    leading_edge_ids: list[str] = field(default_factory=list)
    leading_edge_size: int = 0
    leading_edge_fraction: float = 0.0
    leading_node_ids: list[Any] = field(default_factory=list)
    zero_weight_fallback: bool = False
    edge_set_ids: list[str] = field(default_factory=list)
    running_sum: list[float] | None = None
    status: str = "ok"
    warnings: list[str] = field(default_factory=list)
    NES: float | None = None
    p_value: float | None = None
    q_value: float | None = None
    n_more_extreme: int | None = None
    n_null_positive: int | None = None
    n_null_negative: int | None = None
    n_permutations: int | None = None
    minimum_resolvable_p: float | None = None
    p_value_method: str | None = None
    normalization_status: str | None = None


def _sets_frame(sets: list[LensSetResult]) -> pd.DataFrame:
    rows = []
    for item in sets:
        row = asdict(item)
        row.pop("running_sum", None)
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass(slots=True)
class LensStatResult:
    """Deterministic set statistics for one signed edge ranking."""

    sets: list[LensSetResult]
    metadata: dict[str, Any]
    ranked_edges: pd.DataFrame

    def get(self, set_name: str) -> LensSetResult:
        for item in self.sets:
            if item.set_name == set_name:
                return item
        raise KeyError(set_name)

    def to_frame(self) -> pd.DataFrame:
        return _sets_frame(self.sets)


@dataclass(slots=True)
class LensResult:
    """Observed LENS statistics with optional null inference."""

    sets: list[LensSetResult]
    metadata: dict[str, Any]
    ranked_edges: pd.DataFrame
    null_scores: pd.DataFrame | None = None

    def get(self, set_name: str) -> LensSetResult:
        for item in self.sets:
            if item.set_name == set_name:
                return item
        raise KeyError(set_name)

    def to_frame(self) -> pd.DataFrame:
        return _sets_frame(self.sets)

    def null_for(self, set_name: str) -> pd.Series:
        if self.null_scores is None or set_name not in self.null_scores:
            raise KeyError(f"no null scores stored for {set_name!r}")
        return self.null_scores[set_name].copy()

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "sets": [asdict(item) for item in self.sets],
                "metadata": self.metadata,
                "ranked_edges": self.ranked_edges,
                "null_scores": self.null_scores,
            }
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LensResult:
        data = _restore(payload)
        return cls(
            sets=[LensSetResult(**item) for item in data["sets"]],
            metadata=data["metadata"],
            ranked_edges=data["ranked_edges"],
            null_scores=data.get("null_scores"),
        )

    def save(self, path: str | Path) -> Path:
        return _save_payload(
            {
                "sets": [asdict(item) for item in self.sets],
                "metadata": self.metadata,
                "ranked_edges": self.ranked_edges,
                "null_scores": self.null_scores,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> LensResult:
        return cls.from_dict(_load_payload(path))


@dataclass(slots=True)
class GLMResult:
    """Jointly corrected LENS results indexed by contrast name."""

    contrasts: dict[str, LensResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def contrast_names(self) -> tuple[str, ...]:
        return tuple(self.contrasts)

    def get(self, contrast_name: str) -> LensResult:
        return self.contrasts[contrast_name]

    def __getitem__(self, contrast_name: str) -> LensResult:
        return self.get(contrast_name)

    def to_frame(self) -> pd.DataFrame:
        frames = []
        for name, result in self.contrasts.items():
            frame = result.to_frame()
            frame.insert(0, "contrast_name", name)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "contrasts": {name: result.to_dict() for name, result in self.contrasts.items()},
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GLMResult:
        data = _restore(payload)
        return cls(
            contrasts={
                name: LensResult.from_dict(item)
                for name, item in data["contrasts"].items()
            },
            metadata=data.get("metadata", {}),
        )

    def save(self, path: str | Path) -> Path:
        return _save_payload(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> GLMResult:
        return cls.from_dict(_load_payload(path))


@dataclass(slots=True)
class LensStabilityResult:
    set_summary: pd.DataFrame
    edge_summary: pd.DataFrame
    replicate_summary: pd.DataFrame
    metadata: dict[str, Any]

    def get_set(self, set_name: str) -> pd.Series:
        selected = self.set_summary[self.set_summary["set_name"] == set_name]
        if len(selected) != 1:
            raise KeyError(set_name)
        return selected.iloc[0].copy()

    def edges_for(self, set_name: str) -> pd.DataFrame:
        return self.edge_summary[self.edge_summary["set_name"] == set_name].copy()

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "set_summary": self.set_summary,
                "edge_summary": self.edge_summary,
                "replicate_summary": self.replicate_summary,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LensStabilityResult:
        data = _restore(payload)
        return cls(
            set_summary=data["set_summary"],
            edge_summary=data["edge_summary"],
            replicate_summary=data["replicate_summary"],
            metadata=data["metadata"],
        )

    def save(self, path: str | Path) -> Path:
        return _save_payload(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> LensStabilityResult:
        return cls.from_dict(_load_payload(path))


@dataclass(slots=True)
class LeadingNetwork:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    directed: bool = False

    def to_networkx(self) -> nx.Graph:
        graph: nx.Graph = nx.DiGraph() if self.directed else nx.Graph()
        for row in self.nodes.to_dict("records"):
            node_id = row.pop("node_id")
            graph.add_node(node_id, **row)
        for row in self.edges.to_dict("records"):
            first, second = row.pop("node1"), row.pop("node2")
            graph.add_edge(first, second, **row)
        return graph

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        if destination.suffix.lower() == ".graphml":
            nx.write_graphml(self.to_networkx(), destination)
            return destination
        return _save_payload(
            {"nodes": self.nodes, "edges": self.edges, "directed": self.directed},
            destination,
        )
