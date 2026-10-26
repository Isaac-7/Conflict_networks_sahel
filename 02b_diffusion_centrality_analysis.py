"""
Centrality analysis - Event-Location Diffusion Network
Isaac Okai - MSc Dissertation - Conflict Information Networks in the Sahel

This network is DIRECTED, unlike the actor co-occurrence network, so
centrality measures are adapted accordingly:
  - out-strength / in-strength: already computed at construction time
    (weighted out/in degree) - re-reported here alongside the rest.
  - betweenness centrality: computed on the directed graph directly
    (networkx supports directed betweenness natively).
  - eigenvector centrality: computed using LEFT eigenvector convention
    (nx.eigenvector_centrality on a DiGraph reflects incoming-edge
    influence, i.e. "how central is this region as a DESTINATION of
    diffusion from already-central regions"). This is reported clearly
    as such, since directed eigenvector centrality is easy to
    mis-describe.
  - closeness centrality: computed using the convention that distance
    is 1/weight (more episodes = "closer"), consistent with the
    weighted-betweenness treatment used in the co-occurrence network.

The graph is already a single weakly connected component (100% of nodes),
so no component-trimming is needed here, unlike the actor network.
"""

import networkx as nx
import pandas as pd
import json

OUT_DIR = "/home/claude/networks"

G = nx.read_graphml(f"{OUT_DIR}/event_location_diffusion_network.graphml")
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (directed)")
print(f"Weakly connected: {nx.is_weakly_connected(G)}")
print()

# --- Strength (already in node attributes from construction, recomputed
#     here directly from the graph for a single source of truth) ---
out_strength = dict(G.out_degree(weight="weight"))
in_strength = dict(G.in_degree(weight="weight"))
out_degree = dict(G.out_degree())
in_degree = dict(G.in_degree())

# --- Betweenness centrality (directed, unweighted and weighted) ---
betw_unweighted = nx.betweenness_centrality(G, normalized=True)

G_dist = G.copy()
for u, v, data in G_dist.edges(data=True):
    data["distance"] = 1.0 / data["weight"]
betw_weighted = nx.betweenness_centrality(G_dist, normalized=True, weight="distance")

# --- Closeness centrality (directed, using distance weight) ---
close_weighted = nx.closeness_centrality(G_dist, distance="distance")

# --- Eigenvector centrality (directed) ---
# Note: for a DiGraph, nx.eigenvector_centrality measures centrality based
# on INCOMING edges from central nodes (i.e. "this region receives
# diffusion from other important regions"). This is reported explicitly as
# "eigenvector_in" to avoid ambiguity, and we also compute it on the
# reversed graph ("eigenvector_out": central as a SOURCE feeding into
# other central regions).
eigen_in = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
eigen_out = nx.eigenvector_centrality(G.reverse(), max_iter=1000, weight="weight")

# --- Assemble results ---
rows = []
for n in G.nodes():
    rows.append({
        "region": n,
        "country": G.nodes[n].get("country", ""),
        "out_strength": out_strength.get(n, 0),
        "in_strength": in_strength.get(n, 0),
        "out_degree": out_degree.get(n, 0),
        "in_degree": in_degree.get(n, 0),
        "betweenness": betw_unweighted.get(n, 0),
        "betweenness_weighted": betw_weighted.get(n, 0),
        "closeness_weighted": close_weighted.get(n, 0),
        "eigenvector_in": eigen_in.get(n, 0),
        "eigenvector_out": eigen_out.get(n, 0),
    })

df = pd.DataFrame(rows)
df.to_csv(f"{OUT_DIR}/diffusion_centrality_full.csv", index=False)
print(f"Saved: {OUT_DIR}/diffusion_centrality_full.csv")


def top(col, n=12):
    return df.sort_values(col, ascending=False).head(n)[["region", "country", col]]


print()
print("--- Top 12 by OUT-STRENGTH (most frequent diffusion SOURCE) ---")
print(top("out_strength").to_string(index=False))

print()
print("--- Top 12 by IN-STRENGTH (most frequent diffusion DESTINATION) ---")
print(top("in_strength").to_string(index=False))

print()
print("--- Top 12 by BETWEENNESS unweighted (bridging / broker regions) ---")
print(top("betweenness").to_string(index=False))

print()
print("--- Top 12 by BETWEENNESS weighted ---")
print(top("betweenness_weighted").to_string(index=False))

print()
print("--- Top 12 by EIGENVECTOR-IN (receives diffusion from central regions) ---")
print(top("eigenvector_in").to_string(index=False))

print()
print("--- Top 12 by EIGENVECTOR-OUT (feeds diffusion into central regions) ---")
print(top("eigenvector_out").to_string(index=False))

print()
print("DONE.")
