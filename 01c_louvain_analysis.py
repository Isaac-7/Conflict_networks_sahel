"""
Louvain Community Detection - Actor Co-occurrence Network
Isaac Okai - MSc Dissertation - Conflict Information Networks in the Sahel

Runs Louvain community detection (Blondel et al., 2008) via NetworkX's
implementation on both the full and filtered actor co-occurrence networks.
Communities are weighted by co-occurrence count (higher weight = stronger
pull toward the same community, the correct modularity convention).

Run on the FULL graph (including all 182 / 230 disconnected components),
since isolated dyads are legitimate trivial communities, not data to discard.
A fixed random_state is used throughout for reproducibility, and is reported
explicitly so the result can be replicated.
"""

import networkx as nx
import pandas as pd
import json

OUT_DIR = "/home/claude/networks"
SEED = 42


def run_louvain(graphml_path, label, out_prefix):
    print("=" * 70)
    print(f"NETWORK: {label}")
    print("=" * 70)

    G = nx.read_graphml(graphml_path)
    print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    communities = nx.algorithms.community.louvain_communities(
        G, weight="weight", seed=SEED, resolution=1.0
    )
    modularity = nx.algorithms.community.modularity(G, communities, weight="weight")

    print(f"Number of communities detected: {len(communities)}")
    print(f"Modularity score: {modularity:.4f}")
    print(f"(Resolution parameter = 1.0, random seed = {SEED})")
    print()

    # Map node -> community id (ranked by size, largest = community 0)
    sized = sorted(communities, key=len, reverse=True)
    node_to_comm = {}
    for cid, comm in enumerate(sized):
        for node in comm:
            node_to_comm[node] = cid

    comm_sizes = [len(c) for c in sized]
    print("Community size distribution:")
    print(f"  Largest community: {comm_sizes[0]} nodes")
    print(f"  Number of singleton communities (size 1): {sum(1 for s in comm_sizes if s == 1)}")
    print(f"  Number of communities with >= 5 nodes: {sum(1 for s in comm_sizes if s >= 5)}")
    print(f"  Number of communities with >= 20 nodes: {sum(1 for s in comm_sizes if s >= 20)}")
    print()

    # Build results table
    rows = []
    for n in G.nodes():
        rows.append({
            "actor": n,
            "country": G.nodes[n].get("country", ""),
            "actor_type": G.nodes[n].get("actor_type", ""),
            "community_id": node_to_comm[n],
            "community_size": comm_sizes[node_to_comm[n]],
        })
    df = pd.DataFrame(rows).sort_values(["community_size", "community_id"], ascending=[False, True])
    df.to_csv(f"{OUT_DIR}/{out_prefix}_louvain_communities.csv", index=False)
    print(f"Saved: {OUT_DIR}/{out_prefix}_louvain_communities.csv")

    # For the top 8 largest communities, show country composition and top
    # actors by weighted degree within that community - this is what lets
    # you describe each community substantively in Chapter 4 (e.g.
    # "Community 0 is JNIM-centred, spanning Mali/Burkina Faso/Niger").
    print()
    print(f"--- Top 8 largest communities: composition ---")
    weighted_degree = dict(G.degree(weight="weight"))

    summary_rows = []
    for cid in range(min(8, len(sized))):
        comm_nodes = sized[cid]
        comm_df = df[df["community_id"] == cid]
        country_counts = comm_df["country"].value_counts().to_dict()
        top_actors = sorted(
            [(n, weighted_degree.get(n, 0)) for n in comm_nodes],
            key=lambda x: x[1], reverse=True
        )[:5]
        print(f"\nCommunity {cid} (size {len(comm_nodes)}):")
        print(f"  Country composition: {country_counts}")
        print(f"  Top actors by weighted degree:")
        for actor, wd in top_actors:
            print(f"    {wd:>6,}  {actor}")
        summary_rows.append({
            "community_id": cid,
            "size": len(comm_nodes),
            "country_composition": json.dumps(country_counts),
            "top_actor": top_actors[0][0] if top_actors else None,
        })

    pd.DataFrame(summary_rows).to_csv(f"{OUT_DIR}/{out_prefix}_louvain_top_communities_summary.csv", index=False)

    return {
        "n_communities": len(communities),
        "modularity": modularity,
        "largest_community_size": comm_sizes[0],
        "singleton_communities": sum(1 for s in comm_sizes if s == 1),
        "communities_5plus": sum(1 for s in comm_sizes if s >= 5),
        "communities_20plus": sum(1 for s in comm_sizes if s >= 20),
        "seed": SEED,
        "resolution": 1.0,
    }


results = {}
results["full"] = run_louvain(
    f"{OUT_DIR}/actor_cooccurrence_network.graphml",
    "FULL (all actor types)",
    "full"
)

print()
print()

results["filtered"] = run_louvain(
    f"{OUT_DIR}/actor_cooccurrence_network_filtered.graphml",
    "FILTERED (Civilians/Protesters/Rioters excluded)",
    "filtered"
)

with open(f"{OUT_DIR}/louvain_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print()
print("=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"{'Metric':<30}{'Full':>15}{'Filtered':>15}")
print(f"{'Communities detected':<30}{results['full']['n_communities']:>15}{results['filtered']['n_communities']:>15}")
print(f"{'Modularity':<30}{results['full']['modularity']:>15.4f}{results['filtered']['modularity']:>15.4f}")
print(f"{'Largest community size':<30}{results['full']['largest_community_size']:>15}{results['filtered']['largest_community_size']:>15}")
print(f"{'Singleton communities':<30}{results['full']['singleton_communities']:>15}{results['filtered']['singleton_communities']:>15}")

print()
print("DONE.")
