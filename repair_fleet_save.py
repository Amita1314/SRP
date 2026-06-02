#!/usr/bin/env python3
"""
repair_fleet_save.py

Fixes the crashed fleet_cluster.py DB save:
  1. Indexes _tmp_fleet(id) so the UPDATE is O(n log n) not O(n^2)
  2. Runs the UPDATE to populate observations.cluster_fleet
  3. Drops _tmp_fleet
  4. Rebuilds fleet_events, fleet_convoys, leader_follower from the saved labels
"""
import sqlite3
import sys
import time

import pandas as pd

DB_PATH = "whaling.db"


def step1_update_cluster_fleet(con):
    print("Step 1: Populating observations.cluster_fleet from _tmp_fleet ...")
    n_tmp = con.execute("SELECT COUNT(*) FROM _tmp_fleet").fetchone()[0]
    n_labels = con.execute(
        "SELECT COUNT(*) FROM _tmp_fleet WHERE cluster_fleet != -1"
    ).fetchone()[0]
    print(f"  _tmp_fleet rows: {n_tmp:,}  (non-noise: {n_labels:,})")

    print("  Creating index on _tmp_fleet(id) ...")
    con.execute("CREATE INDEX IF NOT EXISTS idx_tmp_id ON _tmp_fleet(id)")
    con.commit()

    print("  Running UPDATE ...")
    t0 = time.time()
    con.execute("""
        UPDATE observations
        SET cluster_fleet = (
            SELECT cluster_fleet FROM _tmp_fleet
            WHERE _tmp_fleet.id = observations.id
        )
        WHERE observations.id IN (SELECT id FROM _tmp_fleet)
    """)
    con.commit()
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    n_set = con.execute(
        "SELECT COUNT(*) FROM observations WHERE cluster_fleet IS NOT NULL"
    ).fetchone()[0]
    print(f"  Rows with cluster_fleet set: {n_set:,}")
    assert n_set == n_tmp, f"Expected {n_tmp}, got {n_set}"

    print("  Dropping _tmp_fleet ...")
    con.execute("DROP TABLE IF EXISTS _tmp_fleet")
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_cluster_fleet ON observations(cluster_fleet)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_fleet_voyage ON observations(VoyageID, cluster_fleet)"
    )
    con.commit()
    print("  Step 1 complete.\n")


def step2_rebuild_tables(con):
    print("Step 2: Rebuilding fleet_events, fleet_convoys, leader_follower ...")

    print("  Loading observations with cluster_fleet ...")
    df = pd.read_sql_query(
        """SELECT id, VoyageID, vesselID, vessel, Lat, Lon, Day, Month, Year,
                  Encounter, SpokeVesselID, SpokeVesselName, cluster_fleet
           FROM observations
           WHERE Lat IS NOT NULL AND Lon IS NOT NULL AND Year IS NOT NULL
             AND Lat BETWEEN -90 AND 90""",
        con,
    )
    df.loc[df["Lon"] < -180, "Lon"] += 360
    df = df[df["Lon"].between(-180, 180)].reset_index(drop=True)
    labels = df["cluster_fleet"].fillna(-1).astype(int).values
    print(f"  {len(df):,} valid rows, labels loaded\n")

    # import the build functions from fleet_cluster.py
    sys.path.insert(0, "/home/user/SRP")
    from fleet_cluster import build_fleet_events, build_convoys, build_leader_follower

    print("Building fleet_events ...")
    fleet_events = build_fleet_events(df, labels)

    print("Building fleet_convoys ...")
    convoys = build_convoys(df, labels)

    print("Building leader_follower ...")
    leader_follower = build_leader_follower(df, labels)

    print("\nSaving tables to DB ...")
    if not fleet_events.empty:
        fleet_events.to_sql("fleet_events", con, if_exists="replace", index=False)
        print(f"  fleet_events: {len(fleet_events):,} rows")
    if not convoys.empty:
        convoys.to_sql("fleet_convoys", con, if_exists="replace", index=False)
        print(f"  fleet_convoys: {len(convoys):,} rows")
    if not leader_follower.empty:
        leader_follower.to_sql("leader_follower", con, if_exists="replace", index=False)
        print(f"  leader_follower: {len(leader_follower):,} rows")
    con.commit()

    print("\nTop 10 fleet events by vessel count:")
    cols = ["cluster_fleet", "n_vessels", "n_obs", "duration_days",
            "centroid_lat", "centroid_lon", "leader_voyageID"]
    print(fleet_events.nlargest(10, "n_vessels")[cols].to_string(index=False))

    print("\nTop 10 convoys by consecutive days:")
    print(
        convoys.head(10)[["vessel_a", "vessel_b", "n_consecutive_days", "n_shared_clusters"]]
        .to_string(index=False)
    )

    print("\nStep 2 complete.")


def main():
    con = sqlite3.connect(DB_PATH)

    tables = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}

    if "_tmp_fleet" in tables:
        step1_update_cluster_fleet(con)
    else:
        n_set = con.execute(
            "SELECT COUNT(*) FROM observations WHERE cluster_fleet IS NOT NULL"
        ).fetchone()[0]
        print(f"_tmp_fleet already gone; observations.cluster_fleet set: {n_set:,}")
        if n_set == 0:
            print("ERROR: no labels in DB and no _tmp_fleet to restore from.")
            con.close()
            sys.exit(1)

    step2_rebuild_tables(con)
    con.close()
    print("\nAll done.")


if __name__ == "__main__":
    main()
