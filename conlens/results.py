"""Serializable result models used by the public API."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

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
        return {"__dataframe__": _jsonable(value.to_dict(orient="records"))}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _restore(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"__nonfinite__"}:
        return {"nan": np.nan, "inf": np.inf, "-inf": -np.inf}[value["__nonfinite__"]]
    if isinstance(value, dict) and set(value) == {"__missing__"}:
        return pd.NA
    if isinstance(value, dict) and set(value) == {"__dataframe__"}:
        return pd.DataFrame(_restore(value["__dataframe__"]))
    if isinstance(value, dict):
        return {k: _restore(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore(v) for v in value]
    return value


@dataclass(slots=True)
class LensSetResult:
    set_name: str
    set_size_input: int
    set_size_effective: int
    ES: float | None
    ES_positive: float | None
    ES_negative: float | None
    NES: float | None = None
    direction: str | None = None
    peak_rank: int | None = None
    peak_fraction: float | None = None
    p_value: float | None = None
    q_value: float | None = None
    n_more_extreme: int | None = None
    leading_edge_ids: list[str] = field(default_factory=list)
    leading_edge_size: int = 0
    leading_edge_fraction: float = 0.0
    leading_node_ids: list[Any] = field(default_factory=list)
    zero_weight_fallback: bool = False
    status: str = "ok"
    warnings: list[str] = field(default_factory=list)
    edge_set_ids: list[str] = field(default_factory=list)
    running_sum: list[float] | None = None
    n_null_positive: int | None = None
    n_null_negative: int | None = None
    n_permutations: int | None = None
    minimum_resolvable_p: float | None = None
    p_value_method: str | None = None
    normalization_status: str | None = None


@dataclass(slots=True)
class LensResult:
    sets: list[LensSetResult]
    metadata: dict[str, Any]
    ranked_edges: pd.DataFrame | None = None

    def get(self, set_name: str) -> LensSetResult:
        for result in self.sets:
            if result.set_name == set_name:
                return result
        raise KeyError(set_name)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for result in self.sets:
            row = asdict(result)
            row.pop("running_sum", None)
            rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "sets": [asdict(result) for result in self.sets],
                "metadata": self.metadata,
                "ranked_edges": self.ranked_edges,
            }
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LensResult:
        data = _restore(payload)
        return cls(
            sets=[LensSetResult(**item) for item in data["sets"]],
            metadata=data["metadata"],
            ranked_edges=data.get("ranked_edges"),
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> LensResult:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(slots=True)
class GLMResult:
    """Named LENS results from one jointly adjusted multi-contrast GLM."""

    contrasts: dict[str, LensResult]
    metadata: dict[str, Any]

    def __getitem__(self, contrast_name: str) -> LensResult:
        return self.contrasts[contrast_name]

    def get(self, contrast_name: str) -> LensResult:
        try:
            return self.contrasts[contrast_name]
        except KeyError:
            raise KeyError(contrast_name) from None

    @property
    def contrast_names(self) -> tuple[str, ...]:
        return tuple(self.contrasts)

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
                name: LensResult.from_dict(result) for name, result in data["contrasts"].items()
            },
            metadata=data["metadata"],
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> GLMResult:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _frame_payload(frame: pd.DataFrame) -> dict[str, Any]:
    return {"columns": frame.columns.tolist(), "records": frame.to_dict(orient="records")}


def _frame_from_payload(payload: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(payload["records"], columns=payload["columns"])


@dataclass(slots=True)
class LensStabilityResult:
    """Set- and edge-level summaries from a full-pipeline bootstrap."""

    set_summary: pd.DataFrame
    edge_summary: pd.DataFrame
    replicate_summary: pd.DataFrame
    metadata: dict[str, Any]
    bootstrap_results: list[LensResult] | None = None

    def get_set(self, set_name: str) -> pd.Series:
        matches = self.set_summary[self.set_summary["set_name"] == set_name]
        if len(matches) != 1:
            raise KeyError(set_name)
        return matches.iloc[0].copy()

    def edges_for(self, set_name: str) -> pd.DataFrame:
        if set_name not in set(self.set_summary.get("set_name", [])):
            raise KeyError(set_name)
        return self.edge_summary[self.edge_summary["set_name"] == set_name].reset_index(
            drop=True
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "set_summary": _frame_payload(self.set_summary),
                "edge_summary": _frame_payload(self.edge_summary),
                "replicate_summary": _frame_payload(self.replicate_summary),
                "metadata": self.metadata,
                "bootstrap_results": None
                if self.bootstrap_results is None
                else [result.to_dict() for result in self.bootstrap_results],
            }
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LensStabilityResult:
        data = _restore(payload)
        stored_results = data.get("bootstrap_results")
        return cls(
            set_summary=_frame_from_payload(data["set_summary"]),
            edge_summary=_frame_from_payload(data["edge_summary"]),
            replicate_summary=_frame_from_payload(data["replicate_summary"]),
            metadata=data["metadata"],
            bootstrap_results=None
            if stored_results is None
            else [LensResult.from_dict(item) for item in stored_results],
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> LensStabilityResult:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(slots=True)
class LeadingNetwork:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    directed: bool = False

    def to_networkx(self):
        import networkx as nx

        graph = nx.DiGraph() if self.directed else nx.Graph()
        for row in self.nodes.to_dict(orient="records"):
            node = row.pop("node_id")
            graph.add_node(node, **row)
        for row in self.edges.to_dict(orient="records"):
            row = row.copy()
            graph.add_edge(row.pop("node1"), row.pop("node2"), **row)
        return graph

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({"nodes": self.nodes, "edges": self.edges, "directed": self.directed})

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LeadingNetwork:
        restored = _restore(payload)
        return cls(
            nodes=restored["nodes"],
            edges=restored["edges"],
            directed=bool(restored["directed"]),
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        suffix = destination.suffix.lower()
        if suffix == ".json":
            destination.write_text(
                json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
        elif suffix == ".graphml":
            import networkx as nx

            nx.write_graphml(self.to_networkx(), destination)
        else:
            raise ValueError("leading network output must end in .json or .graphml")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> LeadingNetwork:
        source = Path(path)
        if source.suffix.lower() != ".json":
            raise ValueError("only JSON leading networks can be loaded losslessly")
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))
