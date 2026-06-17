# OCSD LiveMap protocol

The LiveMap layer is an interpretability and demo component. It visualizes query-conditioned retrieval probability over a two-dimensional projection of the document graph.

The map should be used to inspect retrieval dynamics and failure modes. It is not a substitute for qrels-based metrics such as nDCG, Recall, MRR, and MAP.

Visual labels:

- green: judged relevant for the active query;
- red: judged non-relevant for the active query;
- blue: unjudged document;
- white ring: current top-ranked attractor;
- node size: normalized OCSD probability;
- edges: semantic adjacency in the embedding graph.

Recommended figure sequence:

1. raw document graph;
2. cosine retrieval distribution;
3. OCSD distribution after diffusion and origin coupling;
4. qrels-aware final map.
