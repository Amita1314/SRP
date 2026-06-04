import pandas as pd
import sqlite3

AOWL1   = "/root/.claude/uploads/7fa1371a-acc6-488d-ae44-4086ff5c099e/f6065243-aowl_20250916_part1.txt"
AOWL2   = "/root/.claude/uploads/7fa1371a-acc6-488d-ae44-4086ff5c099e/4cdf6a3c-aowl_20250916_part2.txt"
SPOKEN  = "/root/.claude/uploads/7fa1371a-acc6-488d-ae44-4086ff5c099e/bf3c7bbb-coml_spoken_download_20180104.txt"
VOYAGES = "/root/.claude/uploads/7fa1371a-acc6-488d-ae44-4086ff5c099e/b9953f3b-voyages_20240911.txt"
DB_PATH = "/home/user/SRP/whaling.db"

# ── 1. ENCOUNTERS ──────────────────────────────────────────────────────────────
print("Loading AOWL data...")
aowl = pd.concat([
    pd.read_csv(AOWL1, sep="\t", dtype=str),
    pd.read_csv(AOWL2, sep="\t", dtype=str),
], ignore_index=True)

aowl_enc = pd.DataFrame({
    "VoyageID": aowl["VoyageID"],
    "Lat":      pd.to_numeric(aowl["Lat"],   errors="coerce"),
    "Lon":      pd.to_numeric(aowl["Lon"],   errors="coerce"),
    "Day":      pd.to_numeric(aowl["Day"],   errors="coerce"),
    "Month":    pd.to_numeric(aowl["Month"], errors="coerce"),
    "Year":     pd.to_numeric(aowl["Year"],  errors="coerce"),
    "Encounter": aowl["Encounter"],
    "SpokeVesselID":   pd.NA,
    "SpokeVesselName": pd.NA,
})
print(f"  AOWL rows: {len(aowl_enc):,}")

print("Loading CoML spoken data...")
spoken = pd.read_csv(SPOKEN, sep="\t", dtype=str)

spoke_enc = pd.DataFrame({
    "VoyageID": spoken["VoyageID"],
    "Lat":      pd.to_numeric(spoken["Lat"],   errors="coerce"),
    "Lon":      pd.to_numeric(spoken["Lon"],   errors="coerce"),
    "Day":      pd.to_numeric(spoken["Day"],   errors="coerce"),
    "Month":    pd.to_numeric(spoken["Month"], errors="coerce"),
    "Year":     pd.to_numeric(spoken["Year"],  errors="coerce"),
    "Encounter": "Spoke",
    "SpokeVesselID":   spoken["VIDSpo"].replace("", pd.NA),
    "SpokeVesselName": spoken["VesselSpo"].replace("", pd.NA),
})
print(f"  Spoke rows: {len(spoke_enc):,}")

encounters = pd.concat([aowl_enc, spoke_enc], ignore_index=True)
print(f"  Total encounter rows: {len(encounters):,}")

# ── 2. VOYAGES — keep rank-1 (primary master) per VoyageID ───────────────────
print("Loading voyages data...")
voy_raw = pd.read_csv(VOYAGES, sep="\t", dtype=str)

voy_cols = [
    "voyageID", "voyageRank",
    "port", "sailingFrom", "ground",
    "yearOut", "dayOut", "yearIn", "dayIn",
    "agentID", "agent",
    "masterID", "master",
    "vesselID", "vessel", "rig", "tonnage",
]
voy = (
    voy_raw[voy_cols]
    .copy()
    .assign(voyageRank=lambda d: pd.to_numeric(d["voyageRank"], errors="coerce"))
    .sort_values("voyageRank")
    .drop_duplicates(subset="voyageID", keep="first")   # keep primary master
    .drop(columns="voyageRank")
    .rename(columns={"voyageID": "VoyageID"})
)
print(f"  Unique voyages: {len(voy):,}")

# ── 3. JOIN ────────────────────────────────────────────────────────────────────
print("Joining encounters with voyage metadata...")
merged = encounters.merge(voy, on="VoyageID", how="left")
merged.sort_values(["VoyageID", "Year", "Month", "Day"], inplace=True)
merged.reset_index(drop=True, inplace=True)

# Reorder columns: voyage context first, then observation
col_order = [
    "VoyageID",
    "port", "sailingFrom", "ground",
    "yearOut", "dayOut", "yearIn", "dayIn",
    "agentID", "agent",
    "masterID", "master",
    "vesselID", "vessel", "rig", "tonnage",
    "Lat", "Lon", "Day", "Month", "Year",
    "Encounter",
    "SpokeVesselID", "SpokeVesselName",
]
merged = merged[col_order]
print(f"  Merged rows: {len(merged):,}")

# ── 4. WRITE TO SQLITE ─────────────────────────────────────────────────────────
print(f"Writing to {DB_PATH} ...")
con = sqlite3.connect(DB_PATH)
merged.to_sql("observations", con, if_exists="replace", index=True, index_label="id")
con.execute("CREATE INDEX IF NOT EXISTS idx_voyageid  ON observations(VoyageID)")
con.execute("CREATE INDEX IF NOT EXISTS idx_year      ON observations(Year)")
con.execute("CREATE INDEX IF NOT EXISTS idx_encounter ON observations(Encounter)")
con.execute("CREATE INDEX IF NOT EXISTS idx_ground    ON observations(ground)")
con.commit()

print("\nEncounter type counts:")
for row in con.execute("SELECT Encounter, COUNT(*) n FROM observations GROUP BY Encounter ORDER BY n DESC"):
    print(f"  {row[0]:<10} {row[1]:>10,}")

unmatched = con.execute(
    "SELECT COUNT(*) FROM observations WHERE port IS NULL"
).fetchone()[0]
print(f"\nRows with no voyage match: {unmatched:,}")
print(f"Total rows: {len(merged):,}")
con.close()
print("Done.")
