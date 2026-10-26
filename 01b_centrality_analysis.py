"""
Centrality analysis - Actor Co-occurrence Network
Isaac Okai - MSc Dissertation - Conflict Information Networks in the Sahel

Computes degree, betweenness, eigenvector, and closeness centrality on both
the full and filtered (Civilians/Protesters/Rioters excluded) versions of the
actor co-occurrence network. Reports both unweighted and weighted variants
where the measure supports it.

Centrality is computed on each network's largest connected component, since
betweenness and closeness are undefined (or require special handling) across
disconnected components. This is reported explicitly, not silently assumed.
"""

import networkx as nx
import pandas as pd
import json

OUT_DIR = "/home/claude/networks"


def analyse(graphml_path, label, out_prefix):
    print("=" * 70)
    print(f"NETWORK: {label}")
    print("=" * 70)

    G = nx.read_graphml(graphml_path)
    print(f"Full graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges, "
          f"{nx.number_connected_components(G)} components")

    # Centrality measures that require connectivity (betweenness, closeness,
    # eigenvector) are computed on the largest connected component (LCC).
    # Degree centrality is computed on the full graph since it does not
    # require connectivity.
    largest_cc_nodes = max(nx.connected_components(G), key=len)
    Gc = G.subgraph(largest_cc_nodes).copy()
    print(f"Largest connected component used for betweenness/closeness/eigenvector: "
          f"{Gc.number_of_nodes():,} nodes ({Gc.number_of_nodes()/G.number_of_nodes()*100:.1f}% of all nodes)")
    print()

    # --- Degree centrality (full graph, both weighted and unweighted) ---
    deg_unweighted = nx.degree_centrality(G)  # normalised by (n-1)
    deg_weighted_raw = dict(G.degree(weight="weight"))  # raw weighted degree, not normalised

    # --- Betweenness centrality (LCC only) ---
    betw_unweighted = nx.betweenness_centrality(Gc, normalized=True)
    # weighted betweenness: networkx treats 'weight' as distance/cost by
    # default, but our weight = co-occurrence COUNT (higher = stronger tie,
    # not higher = farther apart). We invert it so frequent co-occurrence
    # pairs are treated as "closer" (shorter path), which is the standard
    # approach for converting an affinity weight into a distance weight.
    Gc_dist = Gc.copy()
    for u, v, data in Gc_dist.edges(data=True):
        data["distance"] = 1.0 / data["weight"]
    betw_weighted = nx.betweenness_centrality(Gc_dist, normalized=True, weight="distance")

    # --- Closeness centrality (LCC only) ---
    close_unweighted = nx.closeness_centrality(Gc)
    close_weighted = nx.closeness_centrality(Gc_dist, distance="distance")

    # --- Eigenvector centrality (LCC only) ---
    eigen_unweighted = nx.eigenvector_centrality(Gc, max_iter=1000)
    eigen_weighted = nx.eigenvector_centrality(Gc, max_iter=1000, weight="weight")

    # --- Assemble results table ---
    rows = []
    for n in G.nodes():
        in_lcc = n in largest_cc_nodes
        rows.append({
            "actor": n,
            "country": G.nodes[n].get("country", ""),
            "actor_type": G.nodes[n].get("actor_type", ""),
            "in_largest_component": in_lcc,
            "degree_centrality": deg_unweighted[n],
            "weighted_degree": deg_weighted_raw[n],
            "betweenness": betw_unweighted.get(n, None),
            "betweenness_weighted": betw_weighted.get(n, None),
            "closeness": close_unweighted.get(n, None),
            "closeness_weighted": close_weighted.get(n, None),
            "eigenvector": eigen_unweighted.get(n, None),
            "eigenvector_weighted": eigen_weighted.get(n, None),
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/{out_prefix}_centrality_full.csv", index=False)
    print(f"\nSaved full centrality table: {OUT_DIR}/{out_prefix}_centrality_full.csv")

    # --- Print top 15 per measure for quick inspection ---
    def top(col, n=15):
        return df.sort_values(col, ascending=False).head(n)[["actor", "country", col]]

    print()
    print(f"--- Top 15 by WEIGHTED DEGREE ({label}) ---")
    print(top("weighted_degree").to_string(index=False))

    print()
    print(f"--- Top 15 by BETWEENNESS (unweighted, {label}) ---")
    print(top("betweenness").to_string(index=False))

    print()
    print(f"--- Top 15 by BETWEENNESS (weighted, {label}) ---")
    print(top("betweenness_weighted").to_string(index=False))

    print()
    print(f"--- Top 15 by EIGENVECTOR (unweighted, {label}) ---")
    print(top("eigenvector").to_string(index=False))

    print()
    print(f"--- Top 15 by EIGENVECTOR (weighted, {label}) ---")
    print(top("eigenvector_weighted").to_string(index=False))

    print()
    print(f"--- Top 15 by CLOSENESS (unweighted, {label}) ---")
    print(top("closeness").to_string(index=False))

    return df


df_full = analyse(
    f"{OUT_DIR}/actor_cooccurrence_network.graphml",
    "FULL (all actor types)",
    "full"
)

print()
print()

df_filtered = analyse(
    f"{OUT_DIR}/actor_cooccurrence_network_filtered.graphml",
    "FILTERED (Civilians/Protesters/Rioters excluded)",
    "filtered"
)

print()
print("=" * 70)
print("DONE.")
print("=" * 70)
