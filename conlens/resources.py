"""Load versioned node maps from a ConLens resource repository."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen

import pandas as pd

DEFAULT_MAP_SOURCE = "https://raw.githubusercontent.com/zh1peng/conlens-resources/main"
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validated_sha256(value: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("checksum must be a 64-character hexadecimal SHA-256 digest")
    return value.lower()


def _relative_path(value: str) -> str:
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"resource path must be a safe relative path: {value!r}")
    return path.as_posix()


def _default_cache_dir() -> Path:
    configured = os.environ.get("CONLENS_CACHE_DIR")
    if configured:
        return Path(configured)
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "conlens" / "cache"
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "conlens"


def _is_url(source: str | Path) -> bool:
    if isinstance(source, Path):
        return False
    return urlparse(str(source)).scheme in {"http", "https"}


def _remote_url(source: str, relative: str) -> str:
    base = source.rstrip("/") + "/"
    encoded = "/".join(quote(part) for part in PurePosixPath(relative).parts)
    return urljoin(base, encoded)


@lru_cache(maxsize=32)
def _resolve_github_revision(owner: str, repository: str, revision: str) -> str:
    api_url = (
        f"https://api.github.com/repos/{quote(owner)}/{quote(repository)}"
        f"/commits/{quote(revision)}"
    )
    request = Request(
        api_url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "conlens"},
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS API
        payload = json.loads(response.read().decode("utf-8"))
    resolved = str(payload.get("sha", "")).lower()
    if _GIT_COMMIT_PATTERN.fullmatch(resolved) is None:
        raise ValueError("GitHub did not return a valid commit SHA for the resource source")
    return resolved


def _resolve_source(source: str | Path) -> tuple[str | Path, str | None]:
    if not _is_url(source):
        return source, None
    parsed = urlparse(str(source))
    if parsed.hostname != "raw.githubusercontent.com":
        return source, None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 3:
        raise ValueError("GitHub raw source must include owner, repository, and revision")
    owner, repository, revision, *relative = parts
    normalized_revision = revision.lower()
    if _GIT_COMMIT_PATTERN.fullmatch(normalized_revision) is None:
        normalized_revision = _resolve_github_revision(owner, repository, revision)
    pinned_path = "/" + "/".join([owner, repository, normalized_revision, *relative])
    return urlunparse(parsed._replace(path=pinned_path)), normalized_revision


def _read_remote(
    source: str,
    relative: str,
    *,
    expected_sha256: str | None,
    cache_dir: Path,
) -> tuple[bytes, str]:
    url = _remote_url(source, relative)
    expected = None if expected_sha256 is None else _validated_sha256(expected_sha256)
    cache_key = expected or _sha256(url.encode("utf-8"))
    cache_root = cache_dir.resolve()
    cached = (cache_root / cache_key[:2] / cache_key).resolve()
    try:
        cached.relative_to(cache_root)
    except ValueError as exc:
        raise ValueError("cache entry escapes cache root") from exc
    if expected is not None and cached.is_file():
        payload = cached.read_bytes()
        if _sha256(payload) == expected:
            return payload, url
        cached.unlink()

    with urlopen(url, timeout=30) as response:  # noqa: S310 - URL scheme is restricted
        payload = response.read()
    if expected is not None and _sha256(payload) != expected:
        raise ValueError(f"checksum mismatch for {relative!r}")

    if expected is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix="download-", dir=cached.parent)
        try:
            with os.fdopen(handle, "wb") as output:
                output.write(payload)
            Path(temporary).replace(cached)
        finally:
            temporary_path = Path(temporary)
            if temporary_path.exists():
                temporary_path.unlink()
    return payload, url


def _read_file(
    source: str | Path,
    relative: str,
    *,
    expected_sha256: str | None = None,
    cache_dir: Path | None = None,
) -> tuple[bytes, str]:
    safe_relative = _relative_path(relative)
    expected = None if expected_sha256 is None else _validated_sha256(expected_sha256)
    if _is_url(source):
        return _read_remote(
            str(source),
            safe_relative,
            expected_sha256=expected,
            cache_dir=_default_cache_dir() if cache_dir is None else cache_dir,
        )

    root = Path(source).resolve()
    path = (root / Path(*PurePosixPath(safe_relative).parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"resource path escapes source root: {relative!r}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    if expected is not None and _sha256(payload) != expected:
        raise ValueError(f"checksum mismatch for {safe_relative!r}")
    return payload, str(path)


def _read_json(source: str | Path, relative: str, *, cache_dir: Path | None) -> dict[str, Any]:
    payload, _ = _read_file(source, relative, cache_dir=cache_dir)
    value = json.loads(payload.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {relative!r}")
    return value


def _read_table(payload: bytes, *, separator: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(payload), sep=separator, encoding="utf-8-sig")


def _node_order_sha256(node_ids: list[str]) -> str:
    return _sha256(("\n".join(node_ids) + "\n").encode("utf-8"))


@dataclass(slots=True)
class NodeMaps:
    """Atlas-aligned node maps plus their resource metadata."""

    data: pd.DataFrame
    nodes: pd.DataFrame
    map_info: pd.DataFrame
    info: dict[str, Any]

    def __post_init__(self) -> None:
        if self.data.index.name != "node_id":
            raise ValueError("map data index must be named 'node_id'")
        if self.data.index.hasnans or not self.data.index.is_unique:
            raise ValueError("map node IDs must be present and unique")
        if not self.data.columns.is_unique or any(
            not isinstance(item, str) for item in self.data.columns
        ):
            raise ValueError("map names must be unique strings")
        if (
            "node_id" not in self.nodes
            or self.nodes["node_id"].tolist() != self.data.index.tolist()
        ):
            raise ValueError("nodes and map data must use the same ordered node IDs")
        if "feature_id" not in self.map_info:
            raise ValueError("map_info must contain feature_id")
        if self.map_info["feature_id"].tolist() != self.names:
            raise ValueError("map_info feature order must match map data columns")

    @property
    def names(self) -> list[str]:
        return self.data.columns.tolist()

    def __getitem__(self, key: str | list[str]) -> pd.Series | pd.DataFrame:
        return self.data[key]


def load_maps(
    resource: str,
    *,
    atlas: str,
    source: str | Path = DEFAULT_MAP_SOURCE,
    cache_dir: str | Path | None = None,
) -> NodeMaps:
    """Load and verify one atlas-specific collection of node maps."""
    if not isinstance(resource, str) or not resource.strip():
        raise ValueError("resource must be a non-empty string")
    if not isinstance(atlas, str) or not atlas.strip():
        raise ValueError("atlas must be a non-empty string")
    resource_id = resource.strip()
    atlas_id = atlas.strip()
    cache_path = None if cache_dir is None else Path(cache_dir)
    resolved_source, source_revision = _resolve_source(source)

    registry = _read_json(resolved_source, "registry.json", cache_dir=cache_path)
    entries = [
        item
        for item in registry.get("resources", [])
        if isinstance(item, dict) and item.get("resource_id") == resource_id
    ]
    if len(entries) != 1:
        if not entries:
            raise KeyError(f"unknown resource {resource_id!r}")
        raise ValueError(f"registry contains duplicate resource {resource_id!r}")
    entry = entries[0]
    if atlas_id not in entry.get("atlases", []):
        raise ValueError(f"resource {resource_id!r} is not available for atlas {atlas_id!r}")

    manifest_path = _relative_path(entry.get("path", ""))
    manifest = _read_json(resolved_source, manifest_path, cache_dir=cache_path)
    if manifest.get("resource_id") != resource_id:
        raise ValueError("resource manifest ID does not match registry")
    if manifest.get("resource_version") != entry.get("resource_version"):
        raise ValueError("resource version does not match registry")
    if manifest.get("kind") != "node_features":
        raise ValueError(f"resource {resource_id!r} does not contain node maps")

    atlas_files = manifest.get("atlases", {})
    atlas_entry = atlas_files.get(atlas_id) if isinstance(atlas_files, dict) else None
    if isinstance(atlas_entry, str):
        data_relative = atlas_entry
    elif isinstance(atlas_entry, dict):
        data_relative = atlas_entry.get("data_file", "")
    else:
        raise ValueError(f"resource manifest has no data file for atlas {atlas_id!r}")

    manifest_dir = PurePosixPath(manifest_path).parent
    data_path = _relative_path((manifest_dir / _relative_path(data_relative)).as_posix())
    features_relative = manifest.get("features_file")
    if not isinstance(features_relative, str) or not features_relative:
        raise ValueError("node-map resource must declare features_file")
    features_path = _relative_path(
        (manifest_dir / _relative_path(features_relative)).as_posix()
    )
    checksums = manifest.get("checksums", {})
    if not isinstance(checksums, dict):
        raise ValueError("resource manifest checksums must be an object")

    data_checksum_key = _relative_path(data_relative)
    feature_checksum_key = _relative_path(features_relative)
    if data_checksum_key not in checksums or feature_checksum_key not in checksums:
        raise ValueError("resource manifest must checksum its map data and features table")
    data_payload, data_origin = _read_file(
        resolved_source,
        data_path,
        expected_sha256=checksums[data_checksum_key],
        cache_dir=cache_path,
    )
    features_payload, _ = _read_file(
        resolved_source,
        features_path,
        expected_sha256=checksums[feature_checksum_key],
        cache_dir=cache_path,
    )

    atlas_manifest_path = f"atlases/{_relative_path(atlas_id)}/atlas.json"
    atlas_manifest = _read_json(resolved_source, atlas_manifest_path, cache_dir=cache_path)
    if atlas_manifest.get("atlas_id") != atlas_id:
        raise ValueError("atlas manifest ID does not match requested atlas")
    nodes_relative = atlas_manifest.get("nodes_file")
    if not isinstance(nodes_relative, str) or not nodes_relative:
        raise ValueError("atlas manifest must declare nodes_file")
    nodes_path = _relative_path(
        (PurePosixPath(atlas_manifest_path).parent / _relative_path(nodes_relative)).as_posix()
    )
    nodes_payload, _ = _read_file(resolved_source, nodes_path, cache_dir=cache_path)
    if isinstance(atlas_entry, dict):
        expected_atlas_version = atlas_entry.get("atlas_version")
        expected_node_order = atlas_entry.get("node_order_sha256")
        if (
            expected_atlas_version is not None
            and expected_atlas_version != atlas_manifest.get("atlas_version")
        ):
            raise ValueError("resource atlas version does not match atlas manifest")
        if (
            expected_node_order is not None
            and expected_node_order != atlas_manifest.get("node_order_sha256")
        ):
            raise ValueError("resource node order does not match atlas manifest")

    nodes = _read_table(nodes_payload, separator="\t")
    if (
        "node_id" not in nodes
        or nodes["node_id"].isna().any()
        or nodes["node_id"].duplicated().any()
    ):
        raise ValueError("atlas nodes table must contain unique, non-missing node_id values")
    node_ids = nodes["node_id"].astype(str).tolist()
    if len(node_ids) != atlas_manifest.get("n_nodes"):
        raise ValueError("atlas node count does not match atlas manifest")
    if _node_order_sha256(node_ids) != atlas_manifest.get("node_order_sha256"):
        raise ValueError("atlas node order checksum mismatch")
    nodes = nodes.copy()
    nodes["node_id"] = node_ids

    map_info = _read_table(features_payload, separator="\t")
    if (
        "feature_id" not in map_info
        or map_info["feature_id"].isna().any()
        or map_info["feature_id"].duplicated().any()
    ):
        raise ValueError("features table must contain unique, non-missing feature_id values")
    map_info = map_info.copy()
    map_info["feature_id"] = map_info["feature_id"].astype(str)

    table = _read_table(data_payload, separator=",")
    if "node_id" not in table:
        raise ValueError("map data must contain node_id")
    data_node_ids = table["node_id"].astype(str).tolist()
    if data_node_ids != node_ids:
        raise ValueError("map data node order does not match the atlas")
    expected_names = map_info["feature_id"].tolist()
    if table.columns.tolist() != ["node_id", *expected_names]:
        raise ValueError("map data columns do not match the features table")
    data = table.set_index("node_id")
    try:
        data = data.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("node maps must contain numeric values") from exc
    data.index = pd.Index(node_ids, name="node_id")

    return NodeMaps(
        data=data,
        nodes=nodes,
        map_info=map_info,
        info={
            "resource_id": resource_id,
            "resource_version": manifest.get("resource_version"),
            "registry_version": registry.get("registry_version"),
            "atlas_id": atlas_id,
            "atlas_version": atlas_manifest.get("atlas_version"),
            "node_order_sha256": atlas_manifest.get("node_order_sha256"),
            "data_sha256": checksums[data_checksum_key],
            "source": str(resolved_source),
            "source_revision": source_revision,
            "data_origin": data_origin,
        },
    )
