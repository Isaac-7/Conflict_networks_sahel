"""
Temporal Sequence Network construction
Isaac Okai - MSc Dissertation - Conflict Information Networks in the Sahel

This is NOT a single graph - it is three actor co-occurrence networks, one
per historical period (matching Chapter 1's periodisation exactly), built
with IDENTICAL methodology to the main Actor Co-occurrence Network
(filtered version: Civilians/Protesters/Rioters excluded as nodes), so that
structural change across periods can be directly and fairly compared.

Periods (Section 1.5):
  P1 - Pre-crisis    : 1997-01-01 to 2011-12-31
  P2 - Early-crisis   : 2012-01-01 to 2018-12-31
  P3 - Escalation     : 2019-01-01 to 2025-06-22 (end of available data)

For each period: nodes, edges, density, largest component, top actors by
weighted degree, and modularity/community count (Louvain, same parameters
as the main analysis) are reported.

Cross-period comparison: for each pair of CONSECUTIVE periods, actors and
edges are classified as:
  - PERSISTENT : present in both periods
  - EMERGING   : present in the later period only
  - DISAPPEARING : present in the earlier period only
This directly operationalises Subsidiary RQ1 (how network topology has
evolved across historical periods of escalation/de-escalation).
"""

import pandas as pd
import networkx as nx
from collections import Counter
import json

RAW_PATH = "/mnt/user-data/uploads/ACLED_Data_2026-06-22.csv"
OUT_DIR = "/home/claude/networks"
NON_STRATEGIC_TYPES = {"Civilians", "Protesters", "Rioters"}

PERIODS = {
    "P1_precrisis_1997_2011": (1997, 2011),
    "P2_earlycrisis_2012_2018": (2012, 2018),
    "P3_escalation_2019_2025": (2019, 2025),
}

print("=" * 70)
print("STEP 1: Load and split by period")
print("=" * 70)

df = pd.read_csv(RAW_PATH, low_memory=False)
df["event_date"] = pd.to_datetime(df["event_date"])
df["year"] = df["event_date"].dt.year

dyadic = df.dropna(subset=["actor1", "actor2"]).copy()
dyadic = dyadic[dyadic["actor1"] != dyadic["actor2"]]  # drop self-loops, as in main network
dyadic = dyadic[
    (~dyadic["inter1"].isin(NON_STRATEGIC_TYPES)) &
    (~dyadic["inter2"].isin(NON_STRATEGIC_TYPES))
].copy()
print(f"Total strategic-actor dyadic events (post-filter, all periods): {len(dyadic):,}")


def build_period_network(period_df):
    """Build a co-occurrence graph from a period's event subset, using the
    same weighting/attribute logic as the main co-occurrence network."""
    G = nx.Graph()
    pair_counts = Counter()
    for _, row in period_df.iterrows():
        pair = tuple(sorted([row["actor1"], row["actor2"]]))
        pair_counts[pair] += 1
    for (a1, a2), w in pair_counts.items():
        G.add_edge(a1, a2, weight=w)

    lookup_rows = pd.concat([
        period_df[["actor1", "country"]].rename(columns={"actor1": "actor"}),
        period_df[["actor2", "country"]].rename(columns={"actor2": "actor"}),
    ])
    actor_country = {a: g["country"].mode().iloc[0] for a, g in lookup_rows.groupby("actor")}
    nx.set_node_attributes(G, actor_country, "country")
    return G


period_graphs = {}
period_stats = {}

print()
print("=" * 70)
print("STEP 2: Build one network per period")
print("=" * 70)

for label, (y0, y1) in PERIODS.items():
    sub = dyadic[(dyadic["year"] >= y0) & (dyadic["year"] <= y1)]
    G = build_period_network(sub)
    period_graphs[label] = G

    largest_cc = max(nx.connected_components(G), key=len) if G.number_of_nodes() > 0 else set()
    wd = dict(G.degree(weight="weight"))
    top5 = sorted(wd.items(), key=lambda x: -x[1])[:5]

    comms = nx.algorithms.community.louvain_communities(G, weight="weight", seed=42)
    modularity = nx.algorithms.community.modularity(G, comms, weight="weight")

    stats = {
        "events": len(sub),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "components": nx.number_connected_components(G),
        "largest_component_size": len(largest_cc),
        "largest_component_pct": len(largest_cc) / G.number_of_nodes() * 100 if G.number_of_nodes() else 0,
        "n_communities": len(comms),
        "modularity": modularity,
        "top5_actors": top5,
    }
    period_stats[label] = stats

    print(f"\n--- {label} ({y0}-{y1}) ---")
    print(f"  Events: {stats['events']:,}")
    print(f"  Nodes: {stats['nodes']:,}  Edges: {stats['edges']:,}  Density: {stats['density']:.5f}")
    print(f"  Components: {stats['components']}  Largest component: {stats['largest_component_size']} "
          f"({stats['largest_component_pct']:.1f}%)")
    print(f"  Louvain communities: {stats['n_communities']}  Modularity: {stats['modularity']:.4f}")
    print(f"  Top 5 actors by weighted degree:")
    for actor, w in top5:
        print(f"    {w:>5,}  {actor}")

    nx.write_graphml(G, f"{OUT_DIR}/temporal_{label}.graphml")

print()
print("=" * 70)
print("STEP 3: Cross-period comparison (persistent / emerging / disappearing)")
print("=" * 70)

period_keys = list(PERIODS.keys())
comparisons = []

for i in range(len(period_keys) - 1):
    early_key, late_key = period_keys[i], period_keys[i + 1]
    G_early, G_late = period_graphs[early_key], period_graphs[late_key]

    nodes_early, nodes_late = set(G_early.nodes()), set(G_late.nodes())
    persistent_nodes = nodes_early & nodes_late
    emerging_nodes = nodes_late - nodes_early
    disappearing_nodes = nodes_early - nodes_late

    edges_early = set(tuple(sorted(e)) for e in G_early.edges())
    edges_late = set(tuple(sorted(e)) for e in G_late.edges())
    persistent_edges = edges_early & edges_late
    emerging_edges = edges_late - edges_early
    disappearing_edges = edges_early - edges_late

    print(f"\n--- {early_key}  ->  {late_key} ---")
    print(f"  Actors: {len(nodes_early)} -> {len(nodes_late)}")
    print(f"    Persistent: {len(persistent_nodes)}  "
          f"({len(persistent_nodes)/len(nodes_early)*100:.1f}% of earlier period's actors carried forward)")
    print(f"    Emerging (new in later period): {len(emerging_nodes)}")
    print(f"    Disappearing (not seen again): {len(disappearing_nodes)}")
    print(f"  Edges (actor pairs): {len(edges_early)} -> {len(edges_late)}")
    print(f"    Persistent: {len(persistent_edges)}")
    print(f"    Emerging: {len(emerging_edges)}")
    print(f"    Disappearing: {len(disappearing_edges)}")

    # Which emerging actors in the later period had the highest weighted
    # degree - i.e. which NEW actors immediately became major players
    wd_late = dict(G_late.degree(weight="weight"))
    top_emerging = sorted(
        [(a, wd_late[a]) for a in emerging_nodes], key=lambda x: -x[1]
    )[:8]
    print(f"  Top 8 EMERGING actors by weighted degree in {late_key}:")
    for actor, w in top_emerging:
        print(f"    {w:>5,}  {actor}")

    # Which persistent actors grew the most in weighted degree
    wd_early = dict(G_early.degree(weight="weight"))
    growth = sorted(
        [(a, wd_late.get(a, 0) - wd_early.get(a, 0)) for a in persistent_nodes],
        key=lambda x: -x[1]
    )[:8]
    print(f"  Top 8 PERSISTENT actors by growth in weighted degree ({early_key} -> {late_key}):")
    for actor, growth_amt in growth:
        print(f"    +{growth_amt:>5,}  {actor}  ({wd_early.get(actor,0)} -> {wd_late.get(actor,0)})")

    comparisons.append({
        "from": early_key, "to": late_key,
        "actors_early": len(nodes_early), "actors_late": len(nodes_late),
        "persistent_actors": len(persistent_nodes),
        "emerging_actors": len(emerging_nodes),
        "disappearing_actors": len(disappearing_nodes),
        "edges_early": len(edges_early), "edges_late": len(edges_late),
        "persistent_edges": len(persistent_edges),
        "emerging_edges": len(emerging_edges),
        "disappearing_edges": len(disappearing_edges),
        "top_emerging_actors": top_emerging,
        "top_growth_actors": growth,
    })

print()
print("=" * 70)
print("STEP 4: Save outputs")
print("=" * 70)

with open(f"{OUT_DIR}/temporal_period_stats.json", "w") as f:
    json.dump(period_stats, f, indent=2, default=str)
print(f"Saved: {OUT_DIR}/temporal_period_stats.json")

with open(f"{OUT_DIR}/temporal_comparisons.json", "w") as f:
    json.dump(comparisons, f, indent=2, default=str)
print(f"Saved: {OUT_DIR}/temporal_comparisons.json")

print("Saved per-period graphs: temporal_P1_*.graphml, temporal_P2_*.graphml, temporal_P3_*.graphml")
print()
print("DONE.")
