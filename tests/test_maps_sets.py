import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import conlens.resources as resource_module
from conlens import (
    EdgeSets,
    NodeMaps,
    lens_stat,
    load_maps,
    make_edge_statistics,
    make_node_distance_sets,
    make_node_value_sets,
    make_profile_similarity_sets,
    matrix_to_edges,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_resource(root: Path, *, reverse_data: bool = False) -> Path:
    atlas_dir = root / "atlases" / "tiny"
    resource_dir = root / "resources" / "example" / "maps"
    data_dir = resource_dir / "data"
    atlas_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    node_ids = ["a", "b", "c"]
    nodes = pd.DataFrame(
        {
            "node_id": node_ids,
            "label": ["A", "B", "C"],
            "hemisphere": ["LH", "LH", "RH"],
            "network": ["X", "X", "Y"],
        }
    )
    nodes.to_csv(atlas_dir / "nodes.tsv", sep="\t", index=False, lineterminator="\n")
    node_hash = hashlib.sha256(("\n".join(node_ids) + "\n").encode()).hexdigest()
    (atlas_dir / "atlas.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "atlas_id": "tiny",
                "atlas_version": "1.0.0",
                "n_nodes": 3,
                "nodes_file": "nodes.tsv",
                "node_order_sha256": node_hash,
                "default_edge_universe": {
                    "construction": "complete",
                    "directed": False,
                    "diagonal": False,
                },
            }
        ),
        encoding="utf-8",
    )

    features = pd.DataFrame(
        {
            "feature_id": ["map_a", "map_b"],
            "display_name": ["Map A", "Map B"],
        }
    )
    features.to_csv(resource_dir / "features.tsv", sep="\t", index=False, lineterminator="\n")
    data = pd.DataFrame(
        {
            "node_id": node_ids,
            "map_a": [0.0, 0.5, 1.0],
            "map_b": [1.0, 0.5, 0.0],
        }
    )
    if reverse_data:
        data = data.iloc[::-1]
    data.to_csv(data_dir / "tiny.node_features.csv", index=False, lineterminator="\n")

    manifest = {
        "schema_version": "1.0.0",
        "resource_id": "example/maps",
        "resource_version": "1.0.0",
        "kind": "node_features",
        "title": "Example maps",
        "description": "Fixture",
        "features_file": "features.tsv",
        "atlases": {"tiny": "data/tiny.node_features.csv"},
        "license": "MIT",
        "checksums": {
            "features.tsv": _sha256(resource_dir / "features.tsv"),
            "data/tiny.node_features.csv": _sha256(data_dir / "tiny.node_features.csv"),
        },
    }
    (resource_dir / "resource.json").write_text(json.dumps(manifest), encoding="utf-8")
    registry = {
        "schema_version": "1.0.0",
        "registry_version": "1.0.0",
        "resources": [
            {
                "resource_id": "example/maps",
                "resource_version": "1.0.0",
                "kind": "node_features",
                "path": "resources/example/maps/resource.json",
                "atlases": ["tiny"],
            }
        ],
    }
    (root / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
    return root


def _maps(values: dict[str, list[float]]) -> NodeMaps:
    node_ids = list("abcd")
    data = pd.DataFrame(values, index=pd.Index(node_ids, name="node_id"))
    nodes = pd.DataFrame({"node_id": node_ids})
    map_info = pd.DataFrame({"feature_id": list(values)})
    return NodeMaps(data, nodes, map_info, {"resource_id": "fixture"})


def _complete_edges(node_ids: list[str]) -> pd.DataFrame:
    return matrix_to_edges(np.zeros((len(node_ids), len(node_ids))), node_ids)


def test_load_maps_validates_resource_and_atlas(tmp_path: Path):
    root = _write_resource(tmp_path / "resources")
    maps = load_maps("example/maps", atlas="tiny", source=root)
    assert maps.names == ["map_a", "map_b"]
    assert maps.nodes["node_id"].tolist() == ["a", "b", "c"]
    assert maps["map_a"].tolist() == [0.0, 0.5, 1.0]
    assert maps.info["resource_version"] == "1.0.0"
    assert maps.info["data_sha256"] == _sha256(
        root / "resources" / "example" / "maps" / "data" / "tiny.node_features.csv"
    )
    with pytest.raises(KeyError, match="unknown resource"):
        load_maps("missing", atlas="tiny", source=root)
    with pytest.raises(ValueError, match="not available"):
        load_maps("example/maps", atlas="other", source=root)


def test_load_maps_rejects_checksum_and_order_errors(tmp_path: Path):
    corrupted = _write_resource(tmp_path / "corrupted")
    data_path = (
        corrupted
        / "resources"
        / "example"
        / "maps"
        / "data"
        / "tiny.node_features.csv"
    )
    data_path.write_text(data_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_maps("example/maps", atlas="tiny", source=corrupted)

    reversed_root = _write_resource(tmp_path / "reversed", reverse_data=True)
    with pytest.raises(ValueError, match="node order"):
        load_maps("example/maps", atlas="tiny", source=reversed_root)


@pytest.mark.parametrize("node_count, expected", [(68, 91), (200, 780)])
def test_node_value_top_twenty_percent_counts(node_count: int, expected: int):
    node_ids = [f"n{index}" for index in range(node_count)]
    values = pd.Series(np.arange(node_count), index=node_ids, name="abundance")
    sets = make_node_value_sets(
        _complete_edges(node_ids),
        values,
        keep="highest",
        fraction=0.20,
    )
    assert len(sets["abundance"]) == expected
    assert sets.info["sets"]["abundance"]["selected_nodes"] == int(np.ceil(node_count * 0.2))


def test_node_value_sets_use_map_order_and_actual_edge_universe():
    maps = _maps({"target": [3.0, 3.0, 2.0, 1.0]})
    sparse = pd.DataFrame(
        {
            "node1": ["a", "a", "b"],
            "node2": ["b", "c", "d"],
            "edge_id": ["ab", "ac", "bd"],
        }
    )
    exact = make_node_value_sets(sparse, maps, keep="highest", fraction=0.25)
    assert exact.info["sets"]["target"]["selected_nodes"] == 1
    assert exact["target"] == frozenset()
    within = make_node_value_sets(sparse, maps, keep="highest", n_nodes=2)
    assert within["target"] == {"ab"}
    touching = make_node_value_sets(
        sparse,
        maps,
        keep="highest",
        cutoff=3.0,
        connect="touching",
    )
    assert touching["target"] == {"ab", "ac", "bd"}
    between = make_node_value_sets(
        sparse,
        maps,
        keep="highest",
        cutoff=3.0,
        connect="between",
    )
    assert between["target"] == {"ac", "bd"}


def test_node_value_missing_policy_and_unknown_nodes():
    values = pd.Series([3.0, np.nan, 1.0], index=list("abc"), name="map")
    edges = pd.DataFrame({"node1": ["a"], "node2": ["c"], "edge_id": ["ac"]})
    with pytest.raises(ValueError, match="missing nodes"):
        make_node_value_sets(edges, values, fraction=0.5)
    omitted = make_node_value_sets(edges, values, fraction=1.0, missing="omit")
    assert omitted["map"] == {"ac"}
    unknown = edges.assign(node2="z")
    with pytest.raises(ValueError, match="absent from the maps"):
        make_node_value_sets(unknown, values.fillna(0), fraction=0.5)


def test_node_distance_sets_find_close_and_far_edges():
    values = pd.Series([0.0, 0.1, 0.9, 1.0], index=list("abcd"), name="gradient")
    edges = _complete_edges(list("abcd"))
    closest = make_node_distance_sets(edges, values, keep="closest", cutoff=0.11)
    assert closest["gradient:closest"] == {"0--1", "2--3"}
    farthest = make_node_distance_sets(edges, values, keep="farthest", cutoff=0.9)
    assert farthest["gradient:farthest"] == {"0--2", "0--3", "1--3"}
    ranked = make_node_distance_sets(
        edges,
        values,
        keep="closest",
        edge_fraction=0.34,
        value_scale="rank",
    )
    assert len(ranked["gradient:closest"]) == 3


def test_profile_similarity_sets_and_edge_set_roundtrip(tmp_path: Path):
    maps = _maps(
        {
            "x": [1.0, 2.0, 0.0, 0.0],
            "y": [0.0, 0.0, 1.0, 2.0],
            "z": [0.0, 0.0, 0.0, 0.0],
        }
    )
    edges = _complete_edges(list("abcd"))
    sets = make_profile_similarity_sets(
        edges,
        maps,
        metric="pearson",
        keep="most_similar",
        cutoff=0.99,
    )
    assert sets["profile:most_similar"] == {"0--1", "2--3"}
    path = sets.save(tmp_path / "sets.json")
    restored = EdgeSets.load(path)
    assert restored.members == sets.members
    assert restored.info == sets.info
    assert restored.audit.to_dict("records") == sets.audit.to_dict("records")

    statistics = edges.assign(statistic=np.arange(len(edges), dtype=float))
    result = lens_stat(
        make_edge_statistics(statistics, positive_direction="larger"),
        sets,
    )
    assert result.metadata["edge_set_construction"] == sets.info


def test_profile_similarity_rejects_undefined_profiles():
    maps = _maps({"x": [1.0, 1.0, 0.0, 0.0], "y": [1.0, 1.0, 1.0, 2.0]})
    edges = _complete_edges(list("abcd"))
    with pytest.raises(ValueError, match="constant profiles"):
        make_profile_similarity_sets(
            edges,
            maps,
            metric="pearson",
            keep="most_similar",
            edge_fraction=0.5,
        )


def test_edge_sets_validation_paths(tmp_path: Path):
    empty = EdgeSets({"x": []})
    assert empty.names == ["x"]
    assert len(empty) == 1
    assert list(empty) == ["x"]
    assert empty.info == {}
    assert empty.audit.empty
    payload = EdgeSets(
        {"x": [np.int64(1)]},
        {"missing": pd.NA, "nested": [np.float64(np.nan)]},
    ).to_dict()
    assert payload["info"] == {"missing": None, "nested": [None]}

    with pytest.raises(ValueError, match="names"):
        EdgeSets({"": []})
    with pytest.raises(TypeError, match="iterable"):
        EdgeSets({"x": "edge"})
    with pytest.raises(ValueError, match="duplicate members"):
        EdgeSets({"x": [1, "1"]})
    with pytest.raises(ValueError, match="unique"):
        EdgeSets({1: [], "1": []})
    with pytest.raises(ValueError, match="unsupported"):
        EdgeSets.from_dict({})
    base = {"schema_version": 1, "object_type": "EdgeSets", "members": {}}
    with pytest.raises(ValueError, match="audit"):
        EdgeSets.from_dict({**base, "audit": []})
    with pytest.raises(ValueError, match="members"):
        EdgeSets.from_dict({**base, "members": []})
    with pytest.raises(ValueError, match="info"):
        EdgeSets.from_dict({**base, "info": []})
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        EdgeSets.load(bad_file)


def test_node_maps_and_map_input_validation():
    valid = _maps({"x": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="index"):
        NodeMaps(valid.data.rename_axis("parcel"), valid.nodes, valid.map_info, {})
    duplicated = valid.data.copy()
    duplicated.index = pd.Index(["a", "a", "c", "d"], name="node_id")
    with pytest.raises(ValueError, match="unique"):
        NodeMaps(duplicated, valid.nodes, valid.map_info, {})
    numeric_columns = valid.data.copy()
    numeric_columns.columns = [1]
    with pytest.raises(ValueError, match="map names"):
        NodeMaps(numeric_columns, valid.nodes, pd.DataFrame({"feature_id": [1]}), {})
    with pytest.raises(ValueError, match="same ordered"):
        NodeMaps(valid.data, valid.nodes.iloc[::-1], valid.map_info, {})
    with pytest.raises(ValueError, match="feature_id"):
        NodeMaps(valid.data, valid.nodes, pd.DataFrame(), {})
    with pytest.raises(ValueError, match="feature order"):
        NodeMaps(valid.data, valid.nodes, pd.DataFrame({"feature_id": ["y"]}), {})

    edges = _complete_edges(list("abcd"))
    with pytest.raises(TypeError, match="maps must"):
        make_node_value_sets(edges, [1, 2, 3, 4], fraction=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must have a name"):
        make_node_value_sets(edges, pd.Series([1, 2, 3, 4], index=list("abcd")), fraction=0.5)
    with pytest.raises(TypeError, match="sequence"):
        make_node_value_sets(edges, valid, map_names="x", fraction=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique"):
        make_node_value_sets(edges, valid, map_names=["x", "x"], fraction=0.5)
    with pytest.raises(KeyError, match="unknown maps"):
        make_node_value_sets(edges, valid, map_names=["y"], fraction=0.5)
    with pytest.raises(ValueError, match="at least one"):
        make_node_value_sets(edges, valid.data.iloc[:, :0], fraction=0.5)


def test_node_value_builder_validation_paths():
    values = pd.Series([4.0, 3.0, 2.0, 1.0], index=list("abcd"), name="map")
    edges = _complete_edges(list("abcd"))
    low = make_node_value_sets(
        edges,
        values,
        keep="lowest",
        cutoff=2.0,
        name_prefix="low",
    )
    assert low["low:map"] == {"2--3"}
    with pytest.raises(ValueError, match="keep"):
        make_node_value_sets(edges, values, keep="middle", fraction=0.5)
    with pytest.raises(ValueError, match="connect"):
        make_node_value_sets(edges, values, connect="all", fraction=0.5)
    with pytest.raises(ValueError, match="exactly one"):
        make_node_value_sets(edges, values)
    with pytest.raises(ValueError, match="exactly one"):
        make_node_value_sets(edges, values, fraction=0.5, n_nodes=2)
    with pytest.raises(ValueError, match="cutoff"):
        make_node_value_sets(edges, values, cutoff=np.inf)
    with pytest.raises(ValueError, match="fraction"):
        make_node_value_sets(edges, values, fraction=0)
    with pytest.raises(ValueError, match="n_nodes"):
        make_node_value_sets(edges, values, n_nodes=0)
    with pytest.raises(ValueError, match="name_prefix"):
        make_node_value_sets(edges, values, n_nodes=2, name_prefix="")
    with pytest.raises(ValueError, match="numeric"):
        make_node_value_sets(edges, values.astype(str).replace("4.0", "bad"), n_nodes=2)
    with pytest.raises(ValueError, match="infinite"):
        make_node_value_sets(edges, values.replace(4.0, np.inf), n_nodes=2)
    with pytest.raises(ValueError, match="missing must"):
        make_node_value_sets(edges, values, n_nodes=2, missing="drop")
    with pytest.raises(ValueError, match="required columns"):
        make_node_value_sets(edges.drop(columns="edge_id"), values, n_nodes=2)
    with pytest.raises(ValueError, match="non-empty and unique"):
        make_node_value_sets(edges.assign(edge_id="same"), values, n_nodes=2)


def test_distance_builder_validation_paths():
    values = pd.Series([0.0, 0.2, 0.8, 1.0], index=list("abcd"), name="gradient")
    edges = _complete_edges(list("abcd"))
    with pytest.raises(TypeError, match="Series"):
        make_node_distance_sets(edges, values.to_frame(), edge_fraction=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="string name"):
        make_node_distance_sets(edges, values.rename(None), edge_fraction=0.5)
    with pytest.raises(ValueError, match="keep"):
        make_node_distance_sets(edges, values, keep="middle", edge_fraction=0.5)
    with pytest.raises(ValueError, match="value_scale"):
        make_node_distance_sets(edges, values, value_scale="zscore", edge_fraction=0.5)
    with pytest.raises(ValueError, match="exactly one"):
        make_node_distance_sets(edges, values)
    with pytest.raises(ValueError, match="edge_fraction"):
        make_node_distance_sets(edges, values, edge_fraction=0)
    with pytest.raises(ValueError, match="cutoff"):
        make_node_distance_sets(edges, values, cutoff=np.nan)
    with pytest.raises(ValueError, match="name"):
        make_node_distance_sets(edges, values, cutoff=0.5, name="")
    missing = values.copy()
    missing.iloc[1] = np.nan
    only_missing_edge = pd.DataFrame({"node1": ["a"], "node2": ["b"], "edge_id": ["ab"]})
    with pytest.raises(ValueError, match="no edges"):
        make_node_distance_sets(
            only_missing_edge,
            missing,
            edge_fraction=0.5,
            missing="omit",
        )


def test_profile_metrics_scaling_and_validation():
    maps = _maps(
        {
            "x": [1.0, 2.0, 0.0, 0.0],
            "y": [0.0, 0.0, 1.0, 2.0],
            "z": [0.5, 1.0, 0.5, 1.0],
        }
    )
    edges = _complete_edges(list("abcd"))
    cosine = make_profile_similarity_sets(
        edges,
        maps,
        metric="cosine",
        keep="least_similar",
        edge_fraction=0.5,
        map_scaling="zscore",
        name="cosine",
    )
    assert len(cosine["cosine"]) == 3
    euclidean = make_profile_similarity_sets(
        edges,
        maps,
        metric="euclidean",
        keep="least_similar",
        cutoff=1.0,
    )
    assert euclidean.info["metric"] == "euclidean"
    for option, value, match in [
        ("metric", "spearman", "metric"),
        ("keep", "closest", "keep"),
        ("map_scaling", "rank", "map_scaling"),
    ]:
        kwargs = {option: value, "edge_fraction": 0.5}
        with pytest.raises(ValueError, match=match):
            make_profile_similarity_sets(edges, maps, **kwargs)
    with pytest.raises(ValueError, match="at least two"):
        make_profile_similarity_sets(edges, maps, map_names=["x"], edge_fraction=0.5)
    constant_map = _maps({"x": [1, 1, 1, 1], "y": [0, 1, 2, 3]})
    with pytest.raises(ValueError, match="constant maps"):
        make_profile_similarity_sets(
            edges,
            constant_map,
            map_scaling="zscore",
            edge_fraction=0.5,
        )
    zero_profile = _maps({"x": [0, 1, 2, 3], "y": [0, 1, 2, 3]})
    with pytest.raises(ValueError, match="zero profiles"):
        make_profile_similarity_sets(
            edges,
            zero_profile,
            metric="cosine",
            edge_fraction=0.5,
        )
    with pytest.raises(ValueError, match="name"):
        make_profile_similarity_sets(edges, maps, edge_fraction=0.5, name="")


def test_remote_resource_reads_are_verified_and_cached(monkeypatch, tmp_path: Path):
    payload = b"verified payload"
    checksum = hashlib.sha256(payload).hexdigest()
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return payload

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return Response()

    monkeypatch.setattr(resource_module, "urlopen", fake_urlopen)
    first, origin = resource_module._read_file(
        "https://example.test/root",
        "data/maps.csv",
        expected_sha256=checksum,
        cache_dir=tmp_path,
    )
    second, _ = resource_module._read_file(
        "https://example.test/root",
        "data/maps.csv",
        expected_sha256=checksum,
        cache_dir=tmp_path,
    )
    assert first == second == payload
    assert origin == "https://example.test/root/data/maps.csv"
    assert len(calls) == 1
    with pytest.raises(ValueError, match="checksum mismatch"):
        resource_module._read_file(
            "https://example.test/root",
            "data/other.csv",
            expected_sha256="0" * 64,
            cache_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="safe relative"):
        resource_module._read_file(tmp_path, "../outside")


def test_malicious_checksum_cannot_escape_or_delete_cache(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="64-character"):
        resource_module._read_file(
            "https://example.test/root",
            "data/maps.csv",
            expected_sha256="aa/../../victim.txt",
            cache_dir=cache,
        )
    assert victim.read_text(encoding="utf-8") == "keep"


def test_github_source_is_pinned_without_changing_public_api(monkeypatch):
    revision = "a" * 40

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"sha": revision}).encode()

    monkeypatch.setattr(resource_module, "urlopen", lambda request, timeout: Response())
    pinned, resolved = resource_module._resolve_source(
        "https://raw.githubusercontent.com/owner/repository/main"
    )
    assert pinned == f"https://raw.githubusercontent.com/owner/repository/{revision}"
    assert resolved == revision
    already_pinned, same_revision = resource_module._resolve_source(pinned)
    assert already_pinned == pinned
    assert same_revision == revision


def test_edge_sets_reject_a_different_analysis_universe():
    values = pd.Series([0.0, 0.1, 0.9, 1.0], index=list("abcd"), name="gradient")
    full_edges = _complete_edges(list("abcd"))
    small_edges = full_edges.iloc[:2].copy()
    small_edges.attrs.update(full_edges.attrs)
    sets = make_node_distance_sets(
        small_edges,
        values,
        keep="closest",
        edge_fraction=0.5,
    )
    statistics = make_edge_statistics(
        full_edges.assign(statistic=np.arange(len(full_edges), dtype=float)),
        positive_direction="larger",
    )
    with pytest.raises(ValueError, match="different edge universe"):
        lens_stat(statistics, sets)

    raw = pd.DataFrame(
        {
            "node1": ["b", "a"],
            "node2": ["c", "b"],
            "edge_id": ["bc", "ab"],
            "statistic": [1.0, -1.0],
        }
    )
    same_universe_sets = make_node_distance_sets(
        raw,
        values,
        keep="closest",
        edge_fraction=0.5,
    )
    same_result = lens_stat(
        make_edge_statistics(raw, positive_direction="larger"),
        same_universe_sets,
    )
    assert same_result.metadata["edge_set_construction"]["edge_universe_size"] == 2


def test_score_ties_are_invariant_to_custom_edge_id_names():
    values = pd.Series([0.0, 1.0, 2.0], index=list("abc"), name="gradient")
    first = pd.DataFrame(
        {"node1": ["a", "b"], "node2": ["b", "c"], "edge_id": ["z", "a"]}
    )
    second = first.assign(edge_id=["a", "z"])
    first_sets = make_node_distance_sets(first, values, edge_fraction=0.5)
    second_sets = make_node_distance_sets(second, values, edge_fraction=0.5)

    def selected_pair(sets):
        row = sets.audit.loc[sets.audit["selected"]].iloc[0]
        return row["node1"], row["node2"]

    assert selected_pair(first_sets) == selected_pair(second_sets) == ("a", "b")


def test_map_content_hash_and_provenance_are_snapshots():
    maps = _maps({"target": [4.0, 3.0, 2.0, 1.0]})
    maps.info["data_sha256"] = "ORIGINAL"
    edges = _complete_edges(list("abcd"))
    first = make_node_value_sets(edges, maps, fraction=0.5)
    original_resource_hash = maps.info.get("data_sha256")
    maps.data.loc["a", "target"] = -10.0
    second = make_node_value_sets(edges, maps, fraction=0.5)
    assert first.info["map_data_sha256"] != second.info["map_data_sha256"]
    assert maps.info.get("data_sha256") == original_resource_hash

    statistics = make_edge_statistics(
        edges.assign(statistic=np.arange(len(edges), dtype=float)),
        positive_direction="larger",
    )
    result = lens_stat(statistics, second)
    saved_value = result.metadata["edge_set_construction"]["selection"]["value"]
    second.info["selection"]["value"] = 0.75
    assert result.metadata["edge_set_construction"]["selection"]["value"] == saved_value
