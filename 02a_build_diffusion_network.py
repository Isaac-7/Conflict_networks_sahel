"""
Event-Location Diffusion Network construction
Isaac Okai - MSc Dissertation - Conflict Information Networks in the Sahel

Nodes: admin1 regions (labelled "Region, Country" to keep cross-country
       regions with identical names distinct).

Edges: DIRECTED edge Region_A -> Region_B if an event in Region_A involving
       actor X is followed, within a 90-day window, by an event in
       Region_B that ALSO involves actor X (shared-actor criterion). This
       operationalises actor-mediated diffusion rather than mere geographic
       adjacency or coincidental temporal clustering (cf. Buhaug & Gleditsch
       2008; Chapter 2, Section 2.3.3).

Edge weight: number of distinct qualifying (event_A, event_B) episodes
             linking that ordered region pair across the full dataset.
             For each shared actor, only the chronologically NEAREST
             subsequent qualifying event in the destination region is
             counted per source event, to avoid combinatorial explosion
             from busy regions.

Excluded: events with no actor1 (none in this dataset - actor1 always
          present). assoc_actor fields not used as the "shared actor",
          consistent with the actor1/actor2-only convention already
          adopted for the co-occurrence network.
"""

import pandas as pd
import networkx as nx
from collections import defaultdict
import json

RAW_PATH = "/mnt/user-data/uploads/ACLED_Data_2026-06-22.csv"
OUT_DIR = "/home/claude/networks"
WINDOW_DAYS = 90

print("=" * 70)
print("STEP 1: Load and prepare")
print("=" * 70)

df = pd.read_csv(RAW_PATH, low_memory=False)
df["event_date"] = pd.to_datetime(df["event_date"])
df["region_node"] = df["admin1"] + ", " + df["country"]

print(f"Total events: {len(df):,}")
print(f"Unique region nodes (admin1 x country): {df['region_node'].nunique()}")

# Build a long table of (actor, event_date, region, row_index) for every
# actor mention - actor1 and actor2 both count as "an actor present at this
# event", consistent with the co-occurrence network's primary-dyad scope.
print()
print("=" * 70)
print("STEP 2: Build actor-event index")
print("=" * 70)

a1 = df[["actor1", "event_date", "region_node"]].rename(columns={"actor1": "actor"})
a2 = df[["actor2", "event_date", "region_node"]].rename(columns={"actor2": "actor"}).dropna(subset=["actor"])
actor_events = pd.concat([a1, a2], ignore_index=True)
actor_events = actor_events.sort_values(["actor", "event_date"]).reset_index(drop=True)

print(f"Total actor-event mentions (actor1 + actor2): {len(actor_events):,}")
print(f"Unique actors in this index: {actor_events['actor'].nunique():,}")

print()
print("=" * 70)
print(f"STEP 3: Identify diffusion episodes (shared actor, <= {WINDOW_DAYS} days, source != dest)")
print("=" * 70)

# For each actor, walk through their chronologically sorted events. For each
# event i, look ahead to find the NEXT event in a DIFFERENT region within
# the window. That is one diffusion episode: region(i) -> region(next),
# with the actor as the mediating link. We deliberately take only the next
# qualifying event per source event (not every later event in the window),
# which is the standard way to avoid one busy actor generating a quadratic
# number of edges from a single timeline.

episode_counts = defaultdict(int)        # (region_from, region_to) -> count
episode_actors = defaultdict(set)        # (region_from, region_to) -> set of actors involved
episode_examples = defaultdict(list)     # (region_from, region_to) -> list of (actor, date_from, date_to)

n_actors_processed = 0
n_episodes_total = 0

for actor, group in actor_events.groupby("actor"):
    n_actors_processed += 1
    records = group[["event_date", "region_node"]].values
    n = len(records)
    if n < 2:
        continue
    for i in range(n - 1):
        date_i, region_i = records[i]
        # look ahead for the next event in a DIFFERENT region within window
        for j in range(i + 1, n):
            date_j, region_j = records[j]
            delta = (date_j - date_i).days
            if delta > WINDOW_DAYS:
                break  # records are sorted by date, so no further j can qualify
            if region_j == region_i:
                continue  # same region = not a diffusion move, keep looking
            # found the nearest subsequent event in a different region
            episode_counts[(region_i, region_j)] += 1
            episode_actors[(region_i, region_j)].add(actor)
            if len(episode_examples[(region_i, region_j)]) < 3:
                episode_examples[(region_i, region_j)].append((actor, str(date_i.date()), str(date_j.date())))
            n_episodes_total += 1
            break  # only the nearest qualifying event per source event

print(f"Actors processed: {n_actors_processed:,}")
print(f"Total diffusion episodes identified: {n_episodes_total:,}")
print(f"Unique directed region pairs with at least one episode: {len(episode_counts):,}")

print()
print("=" * 70)
print("STEP 4: Build directed weighted graph")
print("=" * 70)

G = nx.DiGraph()
for (r_from, r_to), weight in episode_counts.items():
    G.add_edge(
        r_from, r_to,
        weight=weight,
        n_actors=len(episode_actors[(r_from, r_to)]),
        example_actors=json.dumps(list(episode_actors[(r_from, r_to)])[:5]),
    )

# Attach country attribute to each region node
region_country = df.drop_duplicates("region_node").set_index("region_node")["country"].to_dict()
nx.set_node_attributes(G, region_country, "country")

# Make sure every region that appears anywhere in the data is a node, even
# if it has zero diffusion edges (isolated regions are informative - they
# tell us where violence does NOT propagate out from/into).
for region in df["region_node"].unique():
    if region not in G:
        G.add_node(region, country=region_country.get(region, ""))

print(f"Graph nodes (regions): {G.number_of_nodes()}")
print(f"Graph edges (directed diffusion links): {G.number_of_edges()}")
print(f"Density: {nx.density(G):.4f}")
print(f"Isolated nodes (no diffusion in or out): {sum(1 for n in G.nodes() if G.degree(n) == 0)}")
print(f"Weakly connected components: {nx.number_weakly_connected_components(G)}")
largest_wcc = max(nx.weakly_connected_components(G), key=len)
print(f"Largest weakly connected component: {len(largest_wcc)} nodes "
      f"({len(largest_wcc)/G.number_of_nodes()*100:.1f}%)")

print()
print("=" * 70)
print("STEP 5: Top diffusion links and top source/sink regions")
print("=" * 70)

edge_rows = []
for u, v, data in G.edges(data=True):
    edge_rows.append({
        "region_from": u, "region_to": v,
        "weight": data["weight"], "n_distinct_actors": data["n_actors"],
        "example_actors": data["example_actors"],
    })
edge_df = pd.DataFrame(edge_rows).sort_values("weight", ascending=False)

print("Top 15 diffusion links by weight (episode count):")
print(edge_df.head(15)[["region_from", "region_to", "weight", "n_distinct_actors"]].to_string(index=False))

out_strength = dict(G.out_degree(weight="weight"))
in_strength = dict(G.in_degree(weight="weight"))

print()
print("Top 10 regions by OUT-strength (most frequent SOURCE of diffusion):")
for region, s in sorted(out_strength.items(), key=lambda x: -x[1])[:10]:
    print(f"  {s:>5}  {region}")

print()
print("Top 10 regions by IN-strength (most frequent DESTINATION of diffusion):")
for region, s in sorted(in_strength.items(), key=lambda x: -x[1])[:10]:
    print(f"  {s:>5}  {region}")

print()
print("=" * 70)
print("STEP 6: Save outputs")
print("=" * 70)

nx.write_graphml(G, f"{OUT_DIR}/event_location_diffusion_network.graphml")
print(f"Saved: {OUT_DIR}/event_location_diffusion_network.graphml")

edge_df.to_csv(f"{OUT_DIR}/diffusion_edgelist.csv", index=False)
print(f"Saved: {OUT_DIR}/diffusion_edgelist.csv ({len(edge_df):,} edges)")

node_rows = []
for n in G.nodes():
    node_rows.append({
        "region": n,
        "country": region_country.get(n, ""),
        "out_strength": out_strength.get(n, 0),
        "in_strength": in_strength.get(n, 0),
        "out_degree": G.out_degree(n),
        "in_degree": G.in_degree(n),
    })
node_df = pd.DataFrame(node_rows).sort_values("out_strength", ascending=False)
node_df.to_csv(f"{OUT_DIR}/diffusion_nodelist.csv", index=False)
print(f"Saved: {OUT_DIR}/diffusion_nodelist.csv ({len(node_df):,} nodes)")

summary = {
    "window_days": WINDOW_DAYS,
    "granularity": "admin1",
    "total_episodes": n_episodes_total,
    "nodes": G.number_of_nodes(),
    "edges": G.number_of_edges(),
    "density": nx.density(G),
    "isolated_nodes": sum(1 for n in G.nodes() if G.degree(n) == 0),
    "weakly_connected_components": nx.number_weakly_connected_components(G),
    "largest_wcc_size": len(largest_wcc),
    "largest_wcc_pct": len(largest_wcc) / G.number_of_nodes() * 100,
}
with open(f"{OUT_DIR}/diffusion_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"Saved: {OUT_DIR}/diffusion_summary.json")

print()
print("DONE.")
