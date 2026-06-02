#!/usr/bin/env python3
"""
fleet_cluster.py — Fleet-in-Company Detection via HDBSCAN (4D Space-Time)

Finds vessels at the same place at the same time: convoys, groups,
leader/follower patterns.

Usage:
  python fleet_cluster.py
  python fleet_cluster.py --tw-multiplier 2 --visualize
  python fleet_cluster.py --nm-threshold 10 --min-cluster-size 4
"""
import argparse
import json
import os
import sqlite3
import time

import hdbscan
import numpy as np
import pandas as pd

DB_PATH    = "whaling.db"
OUTPUT_DIR = "output"
EARTH_R_KM = 6371.0
KM_PER_NM  = 1.852


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(db_path):
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """SELECT id, VoyageID, vesselID, vessel, Lat, Lon, Day, Month, Year,
                  Encounter, SpokeVesselID, SpokeVesselName
           FROM observations
           WHERE Lat IS NOT NULL AND Lon IS NOT NULL AND Year IS NOT NULL
             AND Lat BETWEEN -90 AND 90""",
        con
    )
    con.close()
    df.loc[df['Lon'] < -180, 'Lon'] += 360
    df = df[df['Lon'].between(-180, 180)].reset_index(drop=True)
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def compute_t_days(df):
    """Absolute fractional-day timeline: Year * 365.25 + DOY."""
    day_clean = df['Day'].clip(lower=1)
    doy = np.zeros(len(df), dtype=np.float32)
    try:
        dates = pd.to_datetime(
            dict(year=df['Year'], month=df['Month'], day=day_clean),
            errors='coerce'
        )
        valid = dates.notna()
        doy[valid]  = dates[valid].dt.dayofyear.values
        doy[~valid] = ((df.loc[~valid, 'Month'] - 1) * 30.44 + 15).values
    except Exception:
        doy = (df['Month'] - 1) * 30.44 + 15
    return df['Year'].values * 365.25 + doy


def engineer_features(df, temporal_weight):
    """4D feature matrix (x, y, z on unit sphere, t_days_scaled). float32."""
    lat_rad = np.radians(df['Lat'].values)
    lon_rad = np.radians(df['Lon'].values)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    t = compute_t_days(df) * temporal_weight
    return np.column_stack([x, y, z, t]).astype(np.float32)


# ── Algorithmic min_cluster_size ──────────────────────────────────────────────

def probe_min_cluster_size(X, sample_frac=0.05, random_state=42):
    """
    Fit HDBSCAN with mcs=2 on a 5% sample, extract condensed-tree cluster
    sizes, and find the natural break via gap statistic on log-sizes.
    Returns derived mcs (floor = 3).
    """
    rng = np.random.default_rng(random_state)
    n_sample = max(500, int(len(X) * sample_frac))
    idx = rng.choice(len(X), n_sample, replace=False)
    X_sample = X[idx]

    print(f"  Probe: {n_sample:,} samples (mcs=2)...")
    probe = hdbscan.HDBSCAN(
        min_cluster_size=2, min_samples=2,
        algorithm='prims_balltree', metric='euclidean',
    ).fit(X_sample)

    tree = probe.condensed_tree_.to_pandas()
    cluster_sizes = tree[tree['child_size'] > 1]['child_size'].values

    if len(cluster_sizes) < 3:
        print("  Probe: too few condensed clusters — using mcs=3")
        return 3

    log_sizes = np.log1p(np.sort(cluster_sizes))
    gaps = np.diff(log_sizes)
    knee_idx = int(np.argmax(gaps))
    natural_mcs = int(np.expm1(log_sizes[knee_idx])) + 1
    mcs = max(3, natural_mcs)
    print(f"  Probe: knee at size {natural_mcs} → mcs = {mcs}")
    return mcs


# ── HDBSCAN ───────────────────────────────────────────────────────────────────

def run_hdbscan(X, mcs, min_samples=2):
    print(f"  HDBSCAN: mcs={mcs}, min_samples={min_samples}, n={len(X):,}...")
    t0 = time.time()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=min_samples,
        algorithm='prims_balltree',
        metric='euclidean',
        core_dist_n_jobs=-1,
        cluster_selection_method='leaf',
        cluster_selection_epsilon=0.0,
        prediction_data=True,
    )
    clusterer.fit(X)
    elapsed = time.time() - t0
    labels = clusterer.labels_
    n_clusters = int(labels.max()) + 1
    n_noise    = int((labels == -1).sum())
    print(f"  Done in {elapsed:.1f}s — {n_clusters:,} clusters, "
          f"{n_noise:,} noise ({n_noise / len(labels):.1%})")
    return clusterer, labels


# ── Post-Clustering Tables ────────────────────────────────────────────────────

def build_fleet_events(df, labels):
    """
    One row per cluster. Filters to n_vessels >= 2.
    Leader = VoyageID with earliest t_days in the cluster.
    """
    df = df.copy()
    df['cluster_fleet'] = labels
    df['t_days'] = compute_t_days(df)

    clustered = df[df['cluster_fleet'] >= 0]
    if clustered.empty:
        return pd.DataFrame()

    rows = []
    for cid, grp in clustered.groupby('cluster_fleet'):
        n_vessels = grp['VoyageID'].nunique()

        first_per_voyage = (grp.sort_values('t_days')
                              .drop_duplicates('VoyageID', keep='first')
                              .sort_values('t_days'))
        leader      = first_per_voyage.iloc[0]['VoyageID']
        vessel_list = first_per_voyage['VoyageID'].tolist()
        enc_counts  = grp['Encounter'].value_counts().to_dict()

        # Spoke confirmation
        cluster_voyages = set(grp['VoyageID'].unique())
        spoke_grp = grp[grp['Encounter'] == 'Spoke']
        validated = 0
        if not spoke_grp.empty:
            spoke_vids = spoke_grp['SpokeVesselID'].dropna().unique()
            if any(v in cluster_voyages for v in spoke_vids):
                validated = 1

        rows.append({
            'cluster_fleet':      int(cid),
            'n_obs':              len(grp),
            'n_vessels':          n_vessels,
            'date_start':         float(grp['t_days'].min()),
            'date_end':           float(grp['t_days'].max()),
            'duration_days':      float(grp['t_days'].max() - grp['t_days'].min()),
            'centroid_lat':       float(grp['Lat'].mean()),
            'centroid_lon':       float(grp['Lon'].mean()),
            'leader_voyageID':    leader,
            'vessel_list':        json.dumps(vessel_list),
            'encounter_types':    json.dumps(enc_counts),
            'validated_by_spoke': validated,
        })

    events = pd.DataFrame(rows)
    fleet  = events[events['n_vessels'] >= 2].reset_index(drop=True)
    print(f"  Fleet events (n_vessels>=2): {len(fleet):,} of {len(events):,} clusters")
    return fleet


def build_convoys(df, labels):
    """
    VoyageID pairs sharing a cluster on >= 3 consecutive days = convoy.
    """
    df = df.copy()
    df['cluster_fleet'] = labels
    df['t_days']  = compute_t_days(df)
    df['day_int'] = df['t_days'].astype(int)

    clustered = df[df['cluster_fleet'] >= 0]

    # Per (cluster, VoyageID): set of integer days present
    cg = (clustered.groupby(['cluster_fleet', 'VoyageID'])
          .agg(vessel   =('vessel',   'first'),
               days     =('day_int',   list),
               lat_first=('Lat',      'first'),
               lon_first=('Lon',      'first'))
          .reset_index())

    pair_days: dict = {}
    pair_meta: dict = {}

    for cid, grp in cg.groupby('cluster_fleet'):
        if len(grp) < 2:
            continue
        voyages = grp.to_dict('records')
        for i in range(len(voyages)):
            for j in range(i + 1, len(voyages)):
                va, vb = voyages[i], voyages[j]
                shared = set(va['days']) & set(vb['days'])
                if not shared:
                    continue
                key = tuple(sorted([va['VoyageID'], vb['VoyageID']]))
                if key not in pair_days:
                    pair_days[key] = set()
                    pair_meta[key] = {
                        'vessel_a': va['vessel'] if key[0] == va['VoyageID'] else vb['vessel'],
                        'vessel_b': vb['vessel'] if key[1] == vb['VoyageID'] else va['vessel'],
                        'lat_first': va['lat_first'],
                        'lon_first': va['lon_first'],
                    }
                pair_days[key] |= shared

    def longest_run(days):
        s = sorted(days)
        best = cur = 1
        for k in range(1, len(s)):
            cur = cur + 1 if s[k] == s[k - 1] + 1 else 1
            best = max(best, cur)
        return best

    rows = []
    for (va, vb), days in pair_days.items():
        consec = longest_run(days)
        if consec < 3:
            continue
        meta = pair_meta[(va, vb)]
        rows.append({
            'voyage_a':           va,
            'voyage_b':           vb,
            'vessel_a':           meta['vessel_a'],
            'vessel_b':           meta['vessel_b'],
            'n_shared_clusters':  len(days),
            'n_consecutive_days': consec,
            'convoy_start_date':  float(min(days)),
            'convoy_end_date':    float(max(days)),
            'convoy_start_lat':   meta['lat_first'],
            'convoy_start_lon':   meta['lon_first'],
        })

    convoys = (pd.DataFrame(rows)
               .sort_values('n_consecutive_days', ascending=False)
               .reset_index(drop=True))
    print(f"  Convoys (>= 3 consecutive shared days): {len(convoys):,}")
    return convoys


def build_leader_follower(df, labels):
    """
    For clusters with >= 3 distinct vessels, rank by arrival time.
    Rank 1 = leader (first arrival).
    """
    df = df.copy()
    df['cluster_fleet'] = labels
    df['t_days'] = compute_t_days(df)

    clustered = df[df['cluster_fleet'] >= 0]

    rows = []
    for cid, grp in clustered.groupby('cluster_fleet'):
        if grp['VoyageID'].nunique() < 3:
            continue

        first_per = (grp.sort_values('t_days')
                       .drop_duplicates('VoyageID', keep='first')
                       .sort_values('t_days')
                       .reset_index(drop=True))
        last_per  = (grp.sort_values('t_days')
                       .drop_duplicates('VoyageID', keep='last')
                       .set_index('VoyageID'))

        for rank, row in first_per.iterrows():
            vid = row['VoyageID']
            dep = last_per.loc[vid, 't_days'] if vid in last_per.index else row['t_days']
            rows.append({
                'cluster_fleet':    int(cid),
                'voyageID':         vid,
                'vessel':           row['vessel'],
                'arrival_t_days':   float(row['t_days']),
                'departure_t_days': float(dep),
                'arrival_rank':     rank + 1,
                'preceded_by':      (first_per.iloc[rank - 1]['VoyageID']
                                     if rank > 0 else None),
            })

    lf = pd.DataFrame(rows)
    print(f"  Leader/follower records (clusters with >=3 vessels): {len(lf):,}")
    return lf


# ── Save Results ──────────────────────────────────────────────────────────────

def save_results(db_path, df, labels, fleet_events, convoys, leader_follower):
    con = sqlite3.connect(db_path)

    try:
        con.execute("ALTER TABLE observations ADD COLUMN cluster_fleet INTEGER")
    except Exception:
        pass  # column already exists

    label_df = pd.DataFrame({'id': df['id'].values,
                              'cluster_fleet': labels.astype(int)})
    label_df.to_sql('_tmp_fleet', con, if_exists='replace', index=False)
    con.execute("""
        UPDATE observations SET cluster_fleet = (
            SELECT cluster_fleet FROM _tmp_fleet
            WHERE _tmp_fleet.id = observations.id
        ) WHERE observations.id IN (SELECT id FROM _tmp_fleet)
    """)
    con.execute("DROP TABLE IF EXISTS _tmp_fleet")
    con.execute("CREATE INDEX IF NOT EXISTS idx_cluster_fleet "
                "ON observations(cluster_fleet)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_fleet_voyage "
                "ON observations(VoyageID, cluster_fleet)")

    if not fleet_events.empty:
        fleet_events.to_sql('fleet_events', con, if_exists='replace', index=False)
    if not convoys.empty:
        convoys.to_sql('fleet_convoys', con, if_exists='replace', index=False)
    if not leader_follower.empty:
        leader_follower.to_sql('leader_follower', con, if_exists='replace', index=False)

    con.commit()
    con.close()
    print(f"  Saved to {db_path}")


# ── Visualization ─────────────────────────────────────────────────────────────

def make_fleet_map(df, labels, fleet_events, outfile):
    try:
        import colorsys
        import folium
    except ImportError:
        print("  folium not installed — skipping map")
        return

    df = df.copy()
    df['cluster_fleet'] = labels

    def ccolor(cid):
        if cid < 0:
            return '#cccccc'
        h = (cid * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.85)
        return '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))

    m = folium.Map(location=[20, -40], zoom_start=3, tiles='CartoDB positron')

    plot_df = df[df['cluster_fleet'] >= 0]
    if len(plot_df) > 30000:
        plot_df = plot_df.sample(30000, random_state=42)

    for _, row in plot_df.iterrows():
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=2,
            color=ccolor(int(row['cluster_fleet'])),
            fill=True, fill_opacity=0.6, weight=0,
            popup=f"C{row['cluster_fleet']}: {row['vessel']} ({row['Year']})"
        ).add_to(m)

    if not fleet_events.empty:
        for _, ev in fleet_events.iterrows():
            folium.Marker(
                location=[ev['centroid_lat'], ev['centroid_lon']],
                popup=folium.Popup(
                    f"<b>Fleet C{ev['cluster_fleet']}</b><br>"
                    f"Vessels: {ev['n_vessels']}<br>"
                    f"Duration: {ev['duration_days']:.1f} days<br>"
                    f"Leader: {ev['leader_voyageID']}<br>"
                    f"Spoke confirmed: {'yes' if ev['validated_by_spoke'] else 'no'}",
                    max_width=220
                ),
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)

    m.save(outfile)
    print(f"  Map saved to {outfile}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fleet-in-company HDBSCAN detection")
    parser.add_argument('--db',               default=DB_PATH)
    parser.add_argument('--nm-threshold',     type=float, default=15.0,
                        help='Encounter radius in nautical miles (default 15)')
    parser.add_argument('--tw-multiplier',    type=float, default=2.0,
                        help='Temporal weight multiplier: 1=exact 15nm, 2=stricter (default 2)')
    parser.add_argument('--min-cluster-size', type=int,   default=None,
                        help='Override probe-derived mcs')
    parser.add_argument('--min-samples',      type=int,   default=2)
    parser.add_argument('--no-probe',         action='store_true')
    parser.add_argument('--no-save',          action='store_true')
    parser.add_argument('--visualize',        action='store_true')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    spatial_chord   = (args.nm_threshold * KM_PER_NM) / EARTH_R_KM
    temporal_weight = spatial_chord * args.tw_multiplier

    print(f"\n=== Fleet-in-Company Clustering ===")
    print(f"  NM threshold  : {args.nm_threshold} nm  → chord = {spatial_chord:.6f}")
    print(f"  TW multiplier : {args.tw_multiplier}×   → temporal_weight = {temporal_weight:.6f}")
    print(f"  1 day penalty : {args.nm_threshold * args.tw_multiplier:.1f} nm equivalent\n")

    print("Loading data...")
    df = load_data(args.db)
    print(f"  {len(df):,} valid observations\n")

    print("Engineering features...")
    X = engineer_features(df, temporal_weight)

    if args.min_cluster_size is not None or args.no_probe:
        mcs = args.min_cluster_size or 4
        print(f"  mcs = {mcs} (manual)\n")
    else:
        print("Probing for min_cluster_size...")
        mcs = probe_min_cluster_size(X)
        print()

    print("Clustering...")
    clusterer, labels = run_hdbscan(X, mcs, args.min_samples)

    print("\nBuilding analysis tables...")
    fleet_events    = build_fleet_events(df, labels)
    convoys         = build_convoys(df, labels)
    leader_follower = build_leader_follower(df, labels)

    if not fleet_events.empty:
        print(f"\nTop 10 fleet events by vessel count:")
        cols = ['cluster_fleet', 'n_vessels', 'n_obs', 'duration_days',
                'centroid_lat', 'centroid_lon', 'leader_voyageID']
        print(fleet_events.nlargest(10, 'n_vessels')[cols].to_string(index=False))

    if not convoys.empty:
        print(f"\nTop 10 convoys by consecutive days:")
        print(convoys.head(10)[['vessel_a', 'vessel_b',
                                 'n_consecutive_days', 'n_shared_clusters']]
              .to_string(index=False))

    if not args.no_save:
        print("\nSaving to DB...")
        save_results(args.db, df, labels, fleet_events, convoys, leader_follower)

    meta = {
        'nm_threshold':    args.nm_threshold,
        'tw_multiplier':   args.tw_multiplier,
        'temporal_weight': float(temporal_weight),
        'spatial_chord':   float(spatial_chord),
        'min_cluster_size': int(mcs),
        'min_samples':     args.min_samples,
        'n_rows':          len(df),
        'n_clusters':      int(labels.max()) + 1,
        'noise_fraction':  float((labels == -1).mean()),
        'n_fleet_events':  len(fleet_events),
        'n_convoys':       len(convoys),
    }
    meta_path = os.path.join(OUTPUT_DIR, 'fleet_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata → {meta_path}")

    if args.visualize:
        print("Creating fleet map...")
        make_fleet_map(df, labels, fleet_events,
                       os.path.join(OUTPUT_DIR, 'fleet_map.html'))

    print("Done.")


if __name__ == '__main__':
    main()
