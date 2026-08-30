# Data and edge sets

`matrix_to_edges` converts a connectome to the canonical edge table consumed by ConLens. Custom
edge IDs are preserved, while canonical IDs retain the node-order-based endpoint identity.

Use `load_maps` to load atlas-aligned node maps from a local or remote ConLens resource repository.
The loader verifies resource checksums and exact atlas node order. GitHub branches and tags are
resolved internally to a commit SHA and recorded in provenance; no client or revision argument is
required.

```python
from conlens import load_maps

pet = load_maps(
    "pet/receptor-react",
    atlas="schaefer200-7net",
    source="/path/to/conlens-resources",
)
```

Three builders answer distinct map-based questions:

```python
from conlens import (
    make_node_distance_sets,
    make_node_value_sets,
    make_profile_similarity_sets,
)

# Edges within the 20% of nodes with the highest D1 values.
d1_high = make_node_value_sets(
    edges, pet, map_names=["D1"], keep="highest", fraction=0.20
)

# The 10% of edges whose endpoints are closest on one map.
d1_close = make_node_distance_sets(
    edges, pet["D1"], keep="closest", edge_fraction=0.10, value_scale="rank"
)

# The 5% of edges with the most similar multi-map node profiles.
pet_profile = make_profile_similarity_sets(
    edges,
    pet,
    metric="pearson",
    keep="most_similar",
    edge_fraction=0.05,
)
```

`fraction` selects nodes; `edge_fraction` selects edges. Fractional counts use ceiling rounding,
and ties are resolved deterministically by atlas node order or canonical edge ID. Builders always
intersect their definitions with the supplied edge universe, so they do not invent missing edges in
a sparse connectome.

Each builder returns serializable `EdgeSets`, which behaves as an ordinary mapping for `lens_stat`
while retaining the map version, checksums, construction settings, universe hash, and audit table.
`lens_stat` automatically rejects an `EdgeSets` object built for another edge universe. Universe
checks and canonical tie-breaking require no additional user parameters.
