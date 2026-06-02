import pandas as pd
import sqlite3

AOWL1 = "/root/.claude/uploads/f63912a1-c890-44b8-929f-fc612e2612cf/ccb7e463-aowl_20250916_part1.txt"
AOWL2 = "/root/.claude/uploads/f63912a1-c890-44b8-929f-fc612e2612cf/3c96cfef-aowl_20250916_part2.txt"
SPOKEN = "/root/.claude/uploads/f63912a1-c890-44b8-929f-fc612e2612cf/ff389797-coml_spoken_download_20180104.txt"
DB_PATH = "/home/user/SRP/whaling.db"

# --- Load AOWL (part 1 + 2) ---
print("Loading AOWL data...")
aowl = pd.concat([
    pd.read_csv(AOWL1, sep="\t", dtype=str),
    pd.read_csv(AOWL2, sep="\t", dtype=str),
], ignore_index=True)

aowl_enc = pd.DataFrame({
    "VoyageID":      aowl["VoyageID"],
    "Lat":           pd.to_numeric(aowl["Lat"],  errors="coerce"),
    "Lon":           pd.to_numeric(aowl["Lon"],  errors="coerce"),
    "Day":           pd.to_numeric(aowl["Day"],  errors="coerce"),
    "Month":         pd.to_numeric(aowl["Month"],errors="coerce"),
    "Year":          pd.to_numeric(aowl["Year"], errors="coerce"),
    "Encounter":     aowl["Encounter"],
    "Species":       aowl["Species"].replace("NULL", pd.NA),
    "NStruck":       pd.to_numeric(aowl["NStruck"], errors="coerce"),
    "NTried":        pd.to_numeric(aowl["NTried"],  errors="coerce"),
    "Place":         aowl["Place"].replace("NULL", pd.NA),
    "Source":        aowl["Source"],
    "Remarks":       aowl["Remarks"].replace("NULL", pd.NA),
    "SpokeVesselID": pd.NA,
    "SpokeVesselName": pd.NA,
})
print(f"  AOWL rows: {len(aowl_enc):,}")

# --- Load CoML Spoken, reshape as Encounter = "Spoke" ---
print("Loading CoML spoken data...")
spoken = pd.read_csv(SPOKEN, sep="\t", dtype=str)

spoke_enc = pd.DataFrame({
    "VoyageID":      spoken["VoyageID"],
    "Lat":           pd.to_numeric(spoken["Lat"], errors="coerce"),
    "Lon":           pd.to_numeric(spoken["Lon"], errors="coerce"),
    "Day":           pd.to_numeric(spoken["Day"],   errors="coerce"),
    "Month":         pd.to_numeric(spoken["Month"], errors="coerce"),
    "Year":          pd.to_numeric(spoken["Year"],  errors="coerce"),
    "Encounter":     "Spoke",
    "Species":       pd.NA,
    "NStruck":       pd.NA,
    "NTried":        pd.NA,
    "Place":         pd.NA,
    "Source":        "CoML",
    "Remarks":       spoken["Commentx"].replace("", pd.NA),
    "SpokeVesselID":   spoken["VIDSpo"].replace("", pd.NA),
    "SpokeVesselName": spoken["VesselSpo"].replace("", pd.NA),
})
print(f"  Spoke rows: {len(spoke_enc):,}")

# --- Combine ---
print("Combining...")
encounters = pd.concat([aowl_enc, spoke_enc], ignore_index=True)
encounters.sort_values(["VoyageID", "Year", "Month", "Day"], inplace=True)
encounters.reset_index(drop=True, inplace=True)
print(f"  Total rows: {len(encounters):,}")

# --- Write to SQLite ---
print(f"Writing to {DB_PATH} ...")
con = sqlite3.connect(DB_PATH)
encounters.to_sql("encounters", con, if_exists="replace", index=True, index_label="id")
con.execute("CREATE INDEX IF NOT EXISTS idx_voyageid ON encounters(VoyageID)")
con.execute("CREATE INDEX IF NOT EXISTS idx_year     ON encounters(Year)")
con.execute("CREATE INDEX IF NOT EXISTS idx_encounter ON encounters(Encounter)")
con.commit()

# Quick summary
summary = encounters["Encounter"].value_counts()
print("\nEncounter type counts:")
print(summary.to_string())

row_count = con.execute("SELECT COUNT(*) FROM encounters").fetchone()[0]
print(f"\nRows in DB: {row_count:,}")
con.close()
print("Done.")
