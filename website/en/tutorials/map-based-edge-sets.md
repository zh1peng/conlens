# Building edge sets from node maps

ConLens 2.2.0 provides three map-based builders. They encode different scientific questions:

| Question | API | Selection unit |
| --- | --- | --- |
| Which edges connect nodes with high or low values? | `make_node_value_sets` | Nodes, then edges |
| Which edges join nodes that are close or far apart on one map? | `make_node_distance_sets` | Edges |
| Which edges join nodes with similar or dissimilar multi-map profiles? | `make_profile_similarity_sets` | Edges |

The common PET top-10% or top-20% abundance definition is the first method. The other two methods
score each candidate edge directly.

::: tip Two different fractions
`fraction` is a fraction of valid **nodes**. `edge_fraction` is a fraction of candidate **edges**.
:::

## Prepare maps and the edge universe

Load atlas-aligned maps from a local resource repository:

```python
from conlens import load_maps

pet_maps = load_maps(
    "pet/receptor-react",
    atlas="schaefer200-7net",
    source="/path/to/conlens-resources",
)
```

The loader verifies the resource manifest, file checksums, atlas node count and order, and map names
and order. `NodeMaps` exposes:

```python
pet_maps.data       # node_id index × map columns
pet_maps.nodes      # atlas node table in the same order
pet_maps.map_info   # map-level metadata
pet_maps.info       # resource, atlas, checksums, and source provenance
pet_maps.names      # ordered map names
pet_maps["D1"]      # one map as a pandas Series
```

The loader does not rescale map values. It reads the processed values declared by the resource.
If a GitHub branch or tag is used as the source, it is resolved to a commit SHA and recorded in
provenance.

For this example, the `pet/receptor-react` resource starts from a frozen 43-entry neuromaps
annotation inventory. Alternate MNI152 and fsaverage representations of the same dataset are
deduplicated in favor of fsaverage; MNI-only maps are projected to fsaverage 10k; negative vertex
values are clipped to zero; maps are parcellated; each independent study map is min-max scaled to
[0, 1] within each atlas resolution; and maps assigned to the same biological target are averaged
with reported sample size as the weight. The retained 37 independent maps produce 19 receptor or
transporter targets, with no second scaling after aggregation. This is part of the resource's
scientific definition, not hidden behavior of `make_node_value_sets`.

Create a complete undirected universe without self-edges when any atlas node pair may be selected:

```python
import numpy as np
from conlens import matrix_to_edges

node_ids = pet_maps.data.index.tolist()
n_nodes = len(node_ids)

edges = matrix_to_edges(
    np.zeros((n_nodes, n_nodes)),
    node_labels=node_ids,
    directed=False,
    diagonal=False,
)
```

The zero matrix defines which edges can be selected; its values are not used by the builders. A
complete undirected universe contains \(N(N-1)/2\) edges: 2,278 for DK68, 4,950 for Schaefer100,
19,900 for Schaefer200, and 44,850 for Schaefer300.

You may instead supply a sparse structural or otherwise restricted edge table. Builders only select
from the supplied universe and never invent missing edges. The table must contain `node1`, `node2`,
and unique `edge_id` columns, and all endpoints must occur in the map index.

## Method 1: high- or low-value nodes

Build one top-20% within-node edge set for every PET map:

```python
from conlens import make_node_value_sets

pet_top20 = make_node_value_sets(
    edges,
    pet_maps,
    map_names=pet_maps.names,
    keep="highest",
    fraction=0.20,
    connect="within",
)
```

The maps are processed independently. Nineteen input maps produce 19 separate sets rather than one
merged set:

```python
pet_top20.names
pet_top20["D1"]
len(pet_top20["D1"])  # 780 in a complete Schaefer200 universe
```

`keep="highest"` selects high values and `keep="lowest"` selects low values. Supply exactly one
node-selection rule:

| Parameter | Meaning |
| --- | --- |
| `fraction` | Fraction of valid nodes; count uses ceiling rounding |
| `n_nodes` | Fixed number of nodes |
| `cutoff` | Threshold in map-value units |

With a cutoff, `highest` keeps \(x_i\ge c\) and `lowest` keeps \(x_i\le c\). Fractional and
fixed-count ties are resolved by atlas map order; cutoff selection retains every node satisfying the
threshold. `map_names` selects columns, while `name_prefix` can distinguish multiple definitions.

`missing="raise"` rejects missing map values. `missing="omit"` excludes missing nodes from ranking
and node selection, and `fraction` is then based only on valid nodes. It does not remove those nodes
from the edge universe: their incident edges cannot enter a `within` set, but a selected-to-missing
edge can still satisfy `touching` or `between`. Filter the edge universe first if missing nodes must
be removed entirely.

### The meaning of `connect`

Let \(S\) be the selected nodes:

| Value | Endpoint rule | Interpretation |
| --- | --- | --- |
| `"within"` | Both endpoints are in \(S\) | Edges inside selected nodes |
| `"touching"` | At least one endpoint is in \(S\) | Every edge incident to selected nodes |
| `"between"` | Exactly one endpoint is in \(S\) | Selected-to-unselected edges |

If `A` and `B` are selected from `A, B, C, D`, `within` contains A–B, `between` contains A–C,
A–D, B–C, and B–D, and `touching` contains both groups. C–D is in none of them.

For a complete undirected universe with \(N\) nodes and \(k\) selected nodes:

\[
|E_{within}|=\frac{k(k-1)}{2},\quad
|E_{between}|=k(N-k),\quad
|E_{touching}|=\frac{N(N-1)}{2}-\frac{(N-k)(N-k-1)}{2}.
\]

These formulas do not apply unchanged to sparse, directed, or diagonal-inclusive universes. PET
abundance studies commonly use `within` to ask about edges among receptor-rich parcels. Use
`touching` for all edges incident to those parcels and `between` for their outward connections only.

For complete undirected universes and `connect="within"`, the expected sizes are:

| Atlas | Threshold | Selected nodes | Edges per map |
| --- | --- | ---: | ---: |
| DK68 | top 10% / 20% | 7 / 14 | 21 / 91 |
| Schaefer100 | top 10% / 20% | 10 / 20 | 45 / 190 |
| Schaefer200 | top 10% / 20% | 20 / 40 | 190 / 780 |
| Schaefer300 | top 10% / 20% | 30 / 60 | 435 / 1,770 |

Keep executable sanity checks in the analysis script:

```python
assert len(pet_maps.names) == 19
assert len(edges) == 19_900
assert pet_top20.info["sets"]["D1"]["selected_nodes"] == 40
assert len(pet_top20["D1"]) == 780

# Compare endpoint columns rather than edge-ID strings across implementations.
d1_endpoints = edges.loc[
    edges["edge_id"].isin(pet_top20["D1"]),
    ["node1", "node2"],
]
```

The last two assertions are specific to this complete, undirected, diagonal-free Schaefer200
top-20% example. A sparse universe still selects 40 nodes but can contain fewer than 780 edges.

## Method 2: distance on one scalar map

This method scores each candidate edge by absolute endpoint difference:

\[
d_{ij}=|x_i-x_j|.
\]

Select edges whose D1 values are closest:

```python
from conlens import make_node_distance_sets

d1_close = make_node_distance_sets(
    edges,
    pet_maps["D1"],
    keep="closest",
    edge_fraction=0.10,
    value_scale="raw",
    name="D1:closest-10pct",
)
```

Use `keep="farthest"` for large differences. Supply exactly one of `edge_fraction` and `cutoff`.
For cutoff selection, `closest` keeps \(d_{ij}\le c\) and `farthest` keeps \(d_{ij}\ge c\).

`value_scale="raw"` uses map values. `value_scale="rank"` first converts valid node values to
percentile ranks, emphasizing relative order and reducing sensitivity to units and outliers. This
changes the scientific definition and should be chosen a priori. `missing="raise"` rejects missing
values; `missing="omit"` excludes edges incident to them. `name` overrides the default
`<map_name>:<keep>` name.

There is no `connect` parameter because nodes are not divided into selected and unselected groups.
An edge between two medium-valued but very similar nodes can be selected here even though it would
not occur in a top-abundance within set.

## Method 3: multi-map node-profile similarity

Represent node \(i\) by \(\mathbf{x}_i=(x_{i1},\ldots,x_{im})\), then compare endpoint profiles
for every candidate edge:

```python
from conlens import make_profile_similarity_sets

pet_profile = make_profile_similarity_sets(
    edges,
    pet_maps,
    map_names=pet_maps.names,
    metric="pearson",
    keep="most_similar",
    edge_fraction=0.05,
    map_scaling="none",
    name="PET-profile:most-similar-5pct",
)
```

All selected maps jointly define one profile and therefore one edge set. They do not produce one set
per receptor.

| Metric | `most_similar` | `least_similar` |
| --- | --- | --- |
| `"pearson"` | Large correlation; cutoff keeps \(r\ge c\) | Small correlation; \(r\le c\) |
| `"cosine"` | Large similarity; \(s\ge c\) | Small similarity; \(s\le c\) |
| `"euclidean"` | Small distance; \(d\le c\) | Large distance; \(d\ge c\) |

Supply exactly one of `edge_fraction` and `cutoff`. Pearson and cosine cutoffs must be in [-1, 1];
Euclidean cutoffs must be non-negative. At least two maps and complete values are required.

`map_scaling="zscore"` standardizes each map across nodes before profiles are compared. It gives
each map a standard-deviation scale and can be appropriate for mixed units. `"none"` preserves the
resource scales and is a natural default when the maps were intentionally normalized together.

::: warning Node–node, not annotation–annotation, correlation
This method calculates node-profile similarity across the endpoints of each candidate edge. A
correlation between two annotation columns across all nodes is a map–map relationship and does not
directly define connectome node–node edges.
:::

## Choosing a method

| Hypothesis | Builder |
| --- | --- |
| Connections within or around high/low annotation nodes form a set | Node values + `connect` |
| Nodes similar on one scalar annotation, or spanning a strong gradient, form a set | Node distance |
| Nodes with similar or complementary multi-annotation composition form a set | Profile similarity |

Choose the method, direction, threshold or fraction, scaling, and edge universe before inspecting
LENS results. When reporting sensitivity analyses over several definitions, name every definition
clearly and account for the full inferential family.

## Output, audit, and provenance

All three builders return `EdgeSets`, which behaves as a mapping and can be passed directly to
`lens_stat`:

```python
from conlens import lens_stat

observed = lens_stat(edge_statistics, pet_top20)

pet_top20.members
pet_top20.audit
pet_top20.info
pet_top20.save("pet-top20.edge-sets.json")
```

The node-value audit has one row per node with value, rank, and selection status. Distance and
profile audits have one row per candidate edge with endpoints, distance or metric value, rank, and
selection status. `info` stores construction parameters, map-content hash, tie-breaking rules,
resource provenance, and edge-universe identity. `lens_stat` rejects an `EdgeSets` object built for
an incompatible universe.

`matrix_to_edges` normally creates node-order-based canonical IDs such as `4--38`, while endpoint
labels remain in `node1` and `node2`. Existing custom edge IDs are preserved and tracked alongside
canonical endpoint identity. Use the endpoint columns rather than parsing edge IDs.

## Signature reference

```python
load_maps(
    resource, *, atlas,
    source=DEFAULT_MAP_SOURCE, cache_dir=None,
)
```

`resource` and `atlas` are registry IDs. `source` is a local or remote resource repository, and
`cache_dir` overrides the default checksum-keyed cache for remote files. Local reads do not need a
cache.

```python
make_node_value_sets(
    edges, maps, *, map_names=None, keep="highest",
    fraction=None, n_nodes=None, cutoff=None,
    connect="within", missing="raise", name_prefix=None,
)

make_node_distance_sets(
    edges, node_map, *, keep="closest",
    edge_fraction=None, cutoff=None, value_scale="raw",
    missing="raise", name=None,
)

make_profile_similarity_sets(
    edges, maps, *, map_names=None, metric="pearson",
    keep="most_similar", edge_fraction=None, cutoff=None,
    map_scaling="none", name=None,
)
```

Fractional counts use ceiling rounding. Node-selection ties use map node order; edge-selection ties
use canonical edge ID. These rules and provenance are recorded internally without extra parameters.
