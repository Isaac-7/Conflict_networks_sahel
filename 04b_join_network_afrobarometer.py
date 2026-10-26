"""
Network <-> Afrobarometer linkage - Research Objective 4
Isaac Okai - MSc Dissertation - Conflict Information Networks in the Sahel

Joins region-level pooled Afrobarometer security-perception measures to the
region-level structural centrality measures from the Event-Location
Diffusion Network (admin1 granularity matches exactly).

Chad is excluded from this join (never surveyed by Afrobarometer). All
other regions in the diffusion network for the 5 remaining countries are
checked for a match; any non-matches are reported explicitly rather than
silently dropped, so that naming mismatches can be fixed rather than
treated as missing data.
"""

import pandas as pd
import json

OUT_DIR = "/home/claude/afrobarometer"

diffusion = pd.read_csv("/home/claude/networks/diffusion_centrality_full.csv")
afro = pd.read_csv(f"{OUT_DIR}/afrobarometer_region_level_pooled.csv")

diffusion_5 = diffusion[diffusion["country"] != "Chad"].copy()
print(f"Diffusion network regions (excluding Chad): {len(diffusion_5)}")
print(f"Afrobarometer pooled regions: {len(afro)}")
print()

merged = afro.merge(
    diffusion_5,
    left_on="region_country_key",
    right_on="region",
    how="left",
    suffixes=("_afro", "_net"),
)

matched = merged[merged["region_net"].notna()]
unmatched_afro = merged[merged["region_net"].isna()]

print(f"Matched: {len(matched)} / {len(afro)} Afrobarometer regions found a network counterpart")
print()
if len(unmatched_afro) > 0:
    print("UNMATCHED Afrobarometer regions (no network match found):")
    print(unmatched_afro[["country_afro", "region_country_key"]].to_string(index=False))
else:
    print("All Afrobarometer regions matched successfully.")

# Also check the reverse: which network regions have NO Afrobarometer data
# (expected for Chad, but check if any non-Chad region is also missing)
afro_keys = set(afro["region_country_key"])
net_only = diffusion_5[~diffusion_5["region"].isin(afro_keys)]
print()
print(f"Network regions (non-Chad) with NO Afrobarometer match: {len(net_only)}")
if len(net_only) > 0:
    print(net_only[["country", "region"]].to_string(index=False))

print()
print("=" * 70)
print("Saving the joined dataset")
print("=" * 70)

final = matched.drop(columns=["region_net", "country_net"]).rename(
    columns={"country_afro": "country", "region_afro": "region"}
)
final = final[[
    "country", "region", "region_country_key",
    "n_respondents_total", "n_valid_unsafe", "mean_unsafe_walking", "pct_felt_unsafe_ever",
    "n_valid_extremist_fear", "mean_extremist_fear", "pct_feared_extremist_violence",
    "low_n_flag",
    "out_strength", "in_strength", "out_degree", "in_degree",
    "betweenness", "betweenness_weighted", "closeness_weighted",
    "eigenvector_in", "eigenvector_out",
]]
final.to_csv(f"{OUT_DIR}/network_afrobarometer_joined.csv", index=False)
print(f"Saved: {OUT_DIR}/network_afrobarometer_joined.csv ({len(final)} regions)")

print()
print("=" * 70)
print("Quick correlation check (descriptive only - not a causal test)")
print("=" * 70)

corr_vars = ["mean_unsafe_walking", "pct_felt_unsafe_ever", "mean_extremist_fear",
             "out_strength", "in_strength", "betweenness_weighted", "eigenvector_in", "eigenvector_out"]
corr_df = final[corr_vars].dropna()
print(f"N regions with complete data for correlation check: {len(corr_df)}")
corr = corr_df.corr()
print()
print("Correlation of perception measures with network structural measures:")
print(corr.loc[["mean_unsafe_walking", "pct_felt_unsafe_ever", "mean_extremist_fear"],
               ["out_strength", "in_strength", "betweenness_weighted", "eigenvector_in", "eigenvector_out"]].round(3))

summary = {
    "afrobarometer_regions": len(afro),
    "matched_to_network": len(matched),
    "unmatched_afrobarometer_regions": len(unmatched_afro),
    "network_regions_without_afrobarometer": len(net_only),
}
with open(f"{OUT_DIR}/network_afrobarometer_join_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print()
print(f"Saved: {OUT_DIR}/network_afrobarometer_join_summary.json")
print()
print("DONE.")
