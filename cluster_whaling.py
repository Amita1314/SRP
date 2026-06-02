"""
Spatio-temporal HDBSCAN clustering of whaling vessel observations.

Feature space (6D, float32):
  x, y, z   — Cartesian coordinates on unit sphere (handles spherical geometry + antimeridian)
  year_t     — Year scaled by YEAR_W (era-level temporal axis)
  doy_sin    — sin(2π·DOY/365) × SEASON_W (circular seasonal axis)
  doy_cos    — cos(2π·DOY/365) × SEASON_W (circular seasonal axis)

Modes:
  active  — Sight + Strike + Spoke only (~85k rows); primary output
  all     — all observations (~477k rows); vessel routing baseline

Results written back to whaling.db:
  observations.cluster_active / .cluster_all
  cluster_summary_active / cluster_summary_all tables
"""

import argparse
import json
import os
import sqlite3
import time
from datetime import date

import numpy as np
import pandas as pd

# ── Tunable constants ──────────────────────────────────────────────────────────
DB_PATH      = os.path.join(os.path.dirname(__file__), "whaling.db")
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "output")
EARTH_R_KM   = 6371.0

# Default scaling: 1 year ≈ 25 km of arc distance on the unit sphere
DEFAULT_YEAR_KM   = 25.0
# Seasonal axis half as influential as yearly drift by default
DEFAULT_SEASON_KM = 12.5

ACTIVE_ENCOUNTERS = ("Sight", "Strike", "Spoke")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(db_path: str, mode: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    if mode == "active":
        placeholders = ",".join("?" * len(ACTIVE_ENCOUNTERS))
        query = (
            f"SELECT id, VoyageID, Lat, Lon, Day, Month, Year, "
            f"Encounter, vessel, vesselID, ground "
            f"FROM observations "
            f"WHERE Encounter IN ({placeholders}) "
            f"AND Lat IS NOT NULL AND Lon IS NOT NULL AND Year IS NOT NULL"
        )
        df = pd.read_sql_query(query, con, params=list(ACTIVE_ENCOUNTERS))
    else:
        query = (
            "SELECT id, VoyageID, Lat, Lon, Day, Month, Year, "
            "Encounter, vessel, vesselID, ground "
            "FROM observations "
            "WHERE Lat IS NOT NULL AND Lon IS NOT NULL AND Year IS NOT NULL"
        )
        df = pd.read_sql_query(query, con)
    con.close()

    # Fix 4 rows with Lon ≈ -181 (antimeridian artefact)
    df.loc[df["Lon"] < -180, "Lon"] += 360

    # Drop any remaining out-of-range coords
    df = df[df["Lat"].between(-90, 90) & df["Lon"].between(-180, 180)].reset_index(drop=True)

    print(f"  Loaded {len(df):,} rows (mode={mode})")
    return df


# ── Feature engineering ────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, year_km: float, season_km: float) -> np.ndarray:
    lat_rad = np.radians(df["Lat"].values)
    lon_rad = np.radians(df["Lon"].values)

    # Cartesian on unit sphere
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    # DOY: clip Day=0 → 1, then compute proper day-of-year
    day_safe = df["Day"].clip(lower=1).astype(int)
    month    = df["Month"].astype(int)
    year     = df["Year"].astype(int)
    doy = np.array([
        date(yr, mo, min(dy, 28 if mo == 2 else 30 if mo in (4, 6, 9, 11) else 31)).timetuple().tm_yday
        for yr, mo, dy in zip(year, month, day_safe)
    ])

    # Temporal axes — convert km to chord-distance units (km / R_earth)
    year_w   = year_km   / EARTH_R_KM
    season_w = season_km / EARTH_R_KM

    year_t   = df["Year"].values * year_w
    doy_sin  = np.sin(2 * np.pi * doy / 365) * season_w
    doy_cos  = np.cos(2 * np.pi * doy / 365) * season_w

    X = np.column_stack([x, y, z, year_t, doy_sin, doy_cos]).astype(np.float32)
    print(f"  Feature matrix: {X.shape}, dtype={X.dtype}, ~{X.nbytes / 1e6:.1f} MB")
    return X


# ── HDBSCAN ────────────────────────────────────────────────────────────────────

def run_hdbscan(X: np.ndarray, min_cluster_size: int, min_samples: int) -> "hdbscan.HDBSCAN":
    import hdbscan  # imported here so the module is importable even without hdbscan installed

    print(f"  Running HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples}) ...")
    t0 = time.time()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        algorithm="boruvka_balltree",
        metric="euclidean",
        approx_min_span_tree=True,
        core_dist_n_jobs=-1,
        cluster_selection_method="eom",
        prediction_data=True,
    )
    clusterer.fit(X)
    elapsed = time.time() - t0

    labels = clusterer.labels_
    n_clusters = int(labels.max()) + 1
    n_noise    = int((labels == -1).sum())
    print(f"  Done in {elapsed:.1f}s — {n_clusters} clusters, {n_noise:,} noise ({n_noise/len(labels):.1%})")
    return clusterer


# ── Cluster summary ────────────────────────────────────────────────────────────

def compute_summary(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["cluster"] = labels

    clustered = df[df["cluster"] >= 0]

    summary = (
        clustered.groupby("cluster")
        .agg(
            n_obs        =("cluster",   "size"),
            lat_centroid =("Lat",       "mean"),
            lon_centroid =("Lon",       "mean"),
            lat_std      =("Lat",       "std"),
            lon_std      =("Lon",       "std"),
            year_min     =("Year",      "min"),
            year_max     =("Year",      "max"),
            year_mean    =("Year",      "mean"),
            year_std     =("Year",      "std"),
            n_voyages    =("VoyageID",  "nunique"),
            n_vessels    =("vesselID",  "nunique"),
        )
        .reset_index()
    )

    def mode_val(s):
        v = s.dropna()
        return v.value_counts().index[0] if len(v) else None

    enc_mode    = clustered.groupby("cluster")["Encounter"].agg(mode_val).rename("dominant_encounter")
    ground_mode = clustered.groupby("cluster")["ground"].agg(mode_val).rename("dominant_ground")

    summary = summary.join(enc_mode, on="cluster").join(ground_mode, on="cluster")
    return summary


# ── Save results to DB ─────────────────────────────────────────────────────────

def save_results(db_path: str, df: pd.DataFrame, labels: np.ndarray,
                 summary: pd.DataFrame, suffix: str) -> None:
    label_col = f"cluster_{suffix}"
    con = sqlite3.connect(db_path)

    # Add column if missing
    existing = [r[1] for r in con.execute("PRAGMA table_info(observations)")]
    if label_col not in existing:
        con.execute(f"ALTER TABLE observations ADD COLUMN {label_col} INTEGER")

    # Batch update via temp table (much faster than executemany for 477k rows)
    label_df = pd.DataFrame({"id": df["id"].values, label_col: labels.astype(int)})
    label_df.to_sql("_tmp_labels", con, if_exists="replace", index=False)
    con.execute(
        f"UPDATE observations "
        f"SET {label_col} = (SELECT {label_col} FROM _tmp_labels WHERE _tmp_labels.id = observations.id) "
        f"WHERE id IN (SELECT id FROM _tmp_labels)"
    )
    con.execute("DROP TABLE IF EXISTS _tmp_labels")
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{label_col} ON observations({label_col})")

    # Summary table
    summary.to_sql(f"cluster_summary_{suffix}", con, if_exists="replace", index=False)

    con.commit()
    con.close()
    print(f"  Saved cluster labels to observations.{label_col} and cluster_summary_{suffix}")


# ── Validation ─────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame, labels: np.ndarray, X: np.ndarray) -> dict:
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    n_clusters = int(labels.max()) + 1
    n_noise    = int((labels == -1).sum())
    noise_frac = n_noise / len(labels)

    print(f"\n── Validation ──────────────────────────────")
    print(f"  Clusters:      {n_clusters}")
    print(f"  Noise:         {n_noise:,}  ({noise_frac:.1%})")

    # Silhouette — sample 10k clustered points (full run is too slow)
    clustered_idx = np.where(labels >= 0)[0]
    if len(clustered_idx) >= 2:
        sample_n  = min(10_000, len(clustered_idx))
        rng       = np.random.default_rng(42)
        sample    = rng.choice(clustered_idx, sample_n, replace=False)
        sil       = silhouette_score(X[sample], labels[sample], metric="euclidean")
        db_score  = davies_bouldin_score(X[clustered_idx[:10_000]], labels[clustered_idx[:10_000]])
        print(f"  Silhouette:    {sil:.4f}  (sampled {sample_n:,}; good ≈ 0.3–0.7)")
        print(f"  Davies-Bouldin:{db_score:.4f}  (lower is better)")
    else:
        sil, db_score = float("nan"), float("nan")

    # Cross-tab: clusters vs known ground labels (top 10 × top 10)
    df2 = df.copy()
    df2["cluster"] = labels
    top_clusters = df2[df2["cluster"] >= 0]["cluster"].value_counts().head(10).index
    top_grounds  = df2["ground"].value_counts().head(10).index
    cross = pd.crosstab(
        df2.loc[df2["cluster"].isin(top_clusters), "cluster"],
        df2.loc[df2["cluster"].isin(top_clusters), "ground"].fillna("Unknown"),
    )[top_grounds.tolist()].sort_index()
    print(f"\n  Cross-tab (top-10 clusters × top-10 grounds):\n{cross.to_string()}")

    return {"n_clusters": n_clusters, "noise_fraction": noise_frac,
            "silhouette": sil, "davies_bouldin": db_score}


# ── Visualisation ──────────────────────────────────────────────────────────────

def plot_spatial(df: pd.DataFrame, labels: np.ndarray, outfile: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    n_clusters = int(labels.max()) + 1
    colors     = cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))

    fig, ax = plt.subplots(figsize=(18, 9))

    noise_mask = labels == -1
    ax.scatter(df.loc[noise_mask, "Lon"], df.loc[noise_mask, "Lat"],
               c="lightgray", s=0.2, alpha=0.15, rasterized=True, label="Noise")

    for cid in range(n_clusters):
        mask = labels == cid
        ax.scatter(df.loc[mask, "Lon"], df.loc[mask, "Lat"],
                   c=[colors[cid % 20]], s=0.4, alpha=0.4, rasterized=True)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"HDBSCAN: {n_clusters} clusters, {noise_mask.mean():.1%} noise  [{os.path.basename(outfile)}]")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  Saved: {outfile}")


def plot_temporal(df: pd.DataFrame, labels: np.ndarray, summary: pd.DataFrame,
                  outfile: str) -> None:
    import matplotlib.pyplot as plt

    df2 = df.copy()
    df2["cluster"] = labels

    fig, ax = plt.subplots(figsize=(16, 6))
    top_clusters = summary.nlargest(20, "n_obs")["cluster"]
    for cid in top_clusters:
        year_counts = df2[df2["cluster"] == cid].groupby("Year").size()
        ground = summary.loc[summary["cluster"] == cid, "dominant_ground"].values[0]
        ax.plot(year_counts.index, year_counts.values, alpha=0.6, linewidth=0.9,
                label=f"C{cid} ({ground or '?'})")

    ax.set_xlabel("Year")
    ax.set_ylabel("Observations")
    ax.set_title("Cluster activity over time (top-20 clusters by size)")
    ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  Saved: {outfile}")


def make_folium_map(df: pd.DataFrame, labels: np.ndarray, summary: pd.DataFrame,
                    outfile: str) -> None:
    import colorsys
    import folium

    n_clusters = int(labels.max()) + 1
    palette    = ["#aaaaaa"] + [
        "#%02x%02x%02x" % tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / max(n_clusters, 1), 0.75, 0.80))
        for i in range(n_clusters)
    ]

    df2 = df.copy()
    df2["cluster"] = labels

    # Sample for Folium performance
    MAX_POINTS = 30_000
    plot_df = df2.sample(min(MAX_POINTS, len(df2)), random_state=42)

    m = folium.Map(location=[20, -30], zoom_start=2, tiles="CartoDB positron")

    for _, row in plot_df.iterrows():
        cid   = int(row["cluster"])
        color = palette[cid + 1] if cid >= 0 else palette[0]
        folium.CircleMarker(
            location=[row["Lat"], row["Lon"]],
            radius=2,
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=f"C{cid} | {row['Encounter']} | {int(row['Year'])}",
        ).add_to(m)

    for _, row in summary.iterrows():
        cid = int(row["cluster"])
        folium.Marker(
            location=[row["lat_centroid"], row["lon_centroid"]],
            popup=folium.Popup(
                f"<b>Cluster {cid}</b><br>"
                f"Ground: {row['dominant_ground']}<br>"
                f"Years: {row['year_min']:.0f}–{row['year_max']:.0f}<br>"
                f"Obs: {int(row['n_obs']):,}<br>"
                f"Voyages: {int(row['n_voyages'])}",
                max_width=220,
            ),
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

    m.save(outfile)
    print(f"  Saved: {outfile}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Spatio-temporal HDBSCAN clustering of whaling data")
    parser.add_argument("--mode",              choices=["active", "all"], default="active")
    parser.add_argument("--min-cluster-size",  type=int,   default=150)
    parser.add_argument("--min-samples",       type=int,   default=None,
                        help="Defaults to min_cluster_size // 5")
    parser.add_argument("--year-km",           type=float, default=DEFAULT_YEAR_KM,
                        help="km per year of temporal weight (default 25)")
    parser.add_argument("--season-km",         type=float, default=DEFAULT_SEASON_KM,
                        help="km per full seasonal cycle (default 12.5)")
    parser.add_argument("--db",                type=str,   default=DB_PATH)
    parser.add_argument("--no-save",           action="store_true",
                        help="Skip writing results to DB (useful for parameter sweeps)")
    parser.add_argument("--visualize",         action="store_true",
                        help="Produce spatial PNG, temporal PNG, and Folium HTML map")
    args = parser.parse_args()

    min_samples = args.min_samples if args.min_samples else max(10, args.min_cluster_size // 5)
    suffix      = args.mode

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n── Load ────────────────────────────────────")
    df = load_data(args.db, args.mode)

    print(f"\n── Features ────────────────────────────────")
    X  = engineer_features(df, year_km=args.year_km, season_km=args.season_km)

    print(f"\n── HDBSCAN ─────────────────────────────────")
    clusterer = run_hdbscan(X, args.min_cluster_size, min_samples)
    labels    = clusterer.labels_

    print(f"\n── Summary ─────────────────────────────────")
    summary = compute_summary(df, labels)
    print(summary.sort_values("n_obs", ascending=False).head(20).to_string(index=False))

    metrics = validate(df, labels, X)

    if not args.no_save:
        print(f"\n── Save ────────────────────────────────────")
        save_results(args.db, df, labels, summary, suffix)

    # Log parameters
    meta = {
        "mode":             args.mode,
        "min_cluster_size": args.min_cluster_size,
        "min_samples":      min_samples,
        "year_km":          args.year_km,
        "season_km":        args.season_km,
        "n_rows":           len(df),
        **metrics,
    }
    meta_path = os.path.join(OUTPUT_DIR, f"cluster_metadata_{suffix}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metadata: {meta_path}")

    if args.visualize:
        print(f"\n── Visualise ───────────────────────────────")
        plot_spatial(df, labels, os.path.join(OUTPUT_DIR, f"clusters_spatial_{suffix}.png"))
        plot_temporal(df, labels, summary, os.path.join(OUTPUT_DIR, f"clusters_temporal_{suffix}.png"))
        make_folium_map(df, labels, summary, os.path.join(OUTPUT_DIR, f"clusters_map_{suffix}.html"))

    print(f"\nDone.")


if __name__ == "__main__":
    main()
