"""
enrich_spoke.py
---------------
Two-pass enrichment of the observations table:

  Pass 1 – Resolve missing SpokeVesselID where SpokeVesselName is known.
            Normalises vessel names (lower, strip, drop punctuation, & -> and)
            and filters candidates by whether the encounter year falls inside
            the candidate voyage's [yearOut, yearIn] window.

  Pass 2 – Insert reciprocal Spoke rows so that BOTH vessels record the
            encounter.  Only added when the target VoyageID exists in the DB
            and no matching reciprocal is already present.
"""

import re
import sqlite3
from collections import defaultdict

DB_PATH = "/home/user/SRP/whaling.db"


def norm(name):
    if not name:
        return ""
    s = name.lower().strip()
    s = re.sub(r"[.,';\-]", "", s)
    s = s.replace(" & ", " and ")
    s = re.sub(r"\s+", " ", s)
    return s


con = sqlite3.connect(DB_PATH)

# ── 1. Build voyage lookup ────────────────────────────────────────────────────
print("Building voyage lookup...")
voy_rows = con.execute("""
    SELECT DISTINCT
        VoyageID, vesselID, vessel,
        port, sailingFrom, ground,
        yearOut, dayOut, yearIn, dayIn,
        agentID, agent, masterID, master, rig, tonnage
    FROM observations
    WHERE VoyageID IS NOT NULL AND vessel IS NOT NULL
""").fetchall()

# normalised_name -> list of voyage dicts
name_to_voyages = defaultdict(list)
for r in voy_rows:
    (VoyageID, vesselID, vessel,
     port, sailingFrom, ground,
     yearOut, dayOut, yearIn, dayIn,
     agentID, agent, masterID, master, rig, tonnage) = r
    try:
        yo = int(yearOut) if yearOut else None
        yi = int(yearIn)  if yearIn  else None
    except ValueError:
        yo = yi = None
    name_to_voyages[norm(vessel)].append({
        "VoyageID": VoyageID, "vesselID": vesselID, "vessel": vessel,
        "port": port, "sailingFrom": sailingFrom, "ground": ground,
        "yearOut": yearOut, "dayOut": dayOut, "yearIn": yearIn, "dayIn": dayIn,
        "agentID": agentID, "agent": agent,
        "masterID": masterID, "master": master,
        "rig": rig, "tonnage": tonnage,
        "yo": yo, "yi": yi,
    })

print(f"  {len(name_to_voyages):,} unique normalised vessel names in voyage table")

# ── 2. Resolve missing SpokeVesselIDs ────────────────────────────────────────
print("\nResolving missing SpokeVesselIDs by name + year...")
no_id_rows = con.execute("""
    SELECT id, VoyageID, Year, SpokeVesselName
    FROM observations
    WHERE Encounter='Spoke' AND SpokeVesselID IS NULL AND SpokeVesselName IS NOT NULL
""").fetchall()

resolved_ids = []   # (id, resolved_VoyageID)
ambiguous    = 0
not_found    = 0

for row_id, VoyageID, Year, SpokeVesselName in no_id_rows:
    key       = norm(SpokeVesselName)
    candidates = name_to_voyages.get(key, [])
    yr        = int(Year) if Year else None

    if not candidates:
        not_found += 1
        continue

    # Filter by year window (strict, then ±1 year fallback)
    if yr:
        matched = [c for c in candidates
                   if c["yo"] and c["yi"] and c["yo"] <= yr <= c["yi"]]
        if not matched:
            matched = [c for c in candidates
                       if c["yo"] and c["yi"] and c["yo"] - 1 <= yr <= c["yi"] + 1]
        if not matched:
            matched = candidates  # no date info – use all
    else:
        matched = candidates

    if len(matched) == 1:
        resolved_ids.append((matched[0]["VoyageID"], row_id))
    else:
        # Multiple candidates – pick the voyage whose yearOut is closest to yr
        if yr:
            matched.sort(key=lambda c: abs((c["yo"] or 9999) - yr))
        resolved_ids.append((matched[0]["VoyageID"], row_id))
        if len(matched) > 1:
            ambiguous += 1

print(f"  Resolved : {len(resolved_ids):,}")
print(f"  Ambiguous (best-guess used): {ambiguous:,}")
print(f"  Not found in voyage table  : {not_found:,}")

con.executemany(
    "UPDATE observations SET SpokeVesselID=? WHERE id=?",
    resolved_ids,
)
con.commit()
print("  SpokeVesselIDs updated.")

# ── 3. Build voyage-metadata cache keyed by VoyageID ────────────────────────
voy_meta = {}
for r in voy_rows:
    voy_meta[r[0]] = r   # VoyageID -> full tuple

# ── 4. Insert reciprocal Spoke rows ─────────────────────────────────────────
print("\nInserting reciprocal Spoke rows...")

# Fetch all Spoke rows whose SpokeVesselID is a known VoyageID
spoke_with_id = con.execute("""
    SELECT o.id,
           o.VoyageID, o.vesselID, o.vessel,
           o.Year, o.Month, o.Day, o.Lat, o.Lon,
           o.SpokeVesselID, o.SpokeVesselName
    FROM   observations o
    WHERE  o.Encounter = 'Spoke'
      AND  o.SpokeVesselID IS NOT NULL
      AND  o.SpokeVesselID IN (SELECT DISTINCT VoyageID FROM observations)
""").fetchall()

print(f"  Spoke rows with a resolvable SpokeVesselID: {len(spoke_with_id):,}")

# Check for existing reciprocals in one query for speed
existing = set(con.execute("""
    SELECT VoyageID, SpokeVesselID, Year, Month, Day
    FROM   observations
    WHERE  Encounter = 'Spoke' AND SpokeVesselID IS NOT NULL
""").fetchall())

new_rows = []
for (row_id,
     VoyageID, vesselID, vessel,
     Year, Month, Day, Lat, Lon,
     SpokeVesselID, SpokeVesselName) in spoke_with_id:

    key = (SpokeVesselID, VoyageID, Year, Month, Day)
    if key in existing:
        continue  # reciprocal already present

    meta = voy_meta.get(SpokeVesselID)
    if not meta:
        continue

    (_, svesselID, svessel,
     sport, ssailingFrom, sground,
     syearOut, sdayOut, syearIn, sdayIn,
     sagentID, sagent, smasterID, smaster, srig, stonnage, *_) = meta

    new_rows.append((
        SpokeVesselID,
        sport, ssailingFrom, sground,
        syearOut, sdayOut, syearIn, sdayIn,
        sagentID, sagent,
        smasterID, smaster,
        svesselID, svessel, srig, stonnage,
        Lat, Lon, Day, Month, Year,
        "Spoke",
        VoyageID, vessel,   # reverse the spoke pointers
    ))
    existing.add(key)  # prevent mirror from being added twice in same batch

print(f"  New reciprocal rows to insert: {len(new_rows):,}")

if new_rows:
    con.executemany("""
        INSERT INTO observations (
            VoyageID,
            port, sailingFrom, ground,
            yearOut, dayOut, yearIn, dayIn,
            agentID, agent,
            masterID, master,
            vesselID, vessel, rig, tonnage,
            Lat, Lon, Day, Month, Year,
            Encounter,
            SpokeVesselID, SpokeVesselName
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, new_rows)
    con.commit()

# ── 5. Summary ────────────────────────────────────────────────────────────────
print("\n=== Final encounter counts ===")
for row in con.execute(
    "SELECT Encounter, COUNT(*) n FROM observations GROUP BY Encounter ORDER BY n DESC"
):
    print(f"  {row[0]:<12} {row[1]:>10,}")

print("\n=== Spoke rows by ID / Name availability ===")
for row in con.execute("""
    SELECT
        CASE WHEN SpokeVesselID IS NOT NULL THEN 'has ID'   ELSE 'no ID'   END,
        CASE WHEN SpokeVesselName IS NOT NULL THEN 'has Name' ELSE 'no Name' END,
        COUNT(*) n
    FROM observations WHERE Encounter='Spoke'
    GROUP BY 1,2 ORDER BY n DESC
"""):
    print(f"  {row[0]:<10} {row[1]:<10} {row[2]:>8,}")

print("\n=== Mutual coverage (reciprocal pairs, same day) ===")
for row in con.execute("""
    SELECT COUNT(*) mutual_rows
    FROM observations a
    JOIN observations b
      ON  a.Encounter='Spoke' AND b.Encounter='Spoke'
      AND a.VoyageID       = b.SpokeVesselID
      AND a.SpokeVesselID  = b.VoyageID
      AND a.Year=b.Year AND a.Month=b.Month AND a.Day=b.Day
"""):
    print(f"  {row[0]:,}")

con.close()
print("\nDone.")
