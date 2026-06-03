#!/usr/bin/env python3
"""
rf_productivity.py — Random Forest: Inductive Exploration (Co-Duction Step 2)

The RF is an exploration engine, not a prediction tool. Feature importance
is computed on the 80% training partition only.
The 20% holdout is saved pristine for regression hypothesis testing (Step 4).

Features (63 total = 11 base + 52 ground dummies):
  Time:    Year, Month
  Vessel:  rig_enc, tonnage_num
  Social:  in_fleet_event, fleet_n_vessels, fleet_duration_days,
           voyage_has_spoke, n_clusters_visited, pct_days_in_fleet,
           days_outside_cluster
  Location: 52 gnd_* dummies
"""
import json
import os
import pickle
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

DB_PATH    = "whaling.db"
OUTPUT_DIR = "output"

GROUND_LABELS = {
    0:  "Hudson Bay",           1:  "Baja California",
    2:  "Japan / Okhotsk",      3:  "Solomon Islands",
    4:  "Gulf of Guinea",       5:  "S Atlantic (Tristan)",
    6:  "Angola Atlantic",      7:  "Mid-Atlantic",
    8:  "Hawaii Pacific",       9:  "Japan Sea",
    10: "Brazil Atlantic",      11: "Equatorial Pacific",
    12: "Caribbean",            13: "Gulf of Alaska",
    14: "Madagascar",           15: "E Africa Indian Ocean",
    16: "Gulf of Mexico",       17: "US East Coast",
    18: "N Atlantic (Azores)",  19: "Equatorial Atlantic",
    20: "N Pacific (Kamchatka)",21: "Bering Sea (West)",
    22: "Western Arctic",       23: "Bering Strait",
    24: "Cape Verde (S)",       25: "Cape Verde (N)",
    26: "Kamchatka / Okhotsk",  27: "Okhotsk Sea",
    28: "Japan Ground (S)",     29: "Japan Ground (Main)",
    30: "Indonesia Pacific",    31: "S Brazil Atlantic",
    32: "Crozet / Kerguelen",   33: "Indian Ocean (Central)",
    34: "S Indian Ocean (W)",   35: "Crozet Islands",
    36: "S Atlantic (Mid-East)",37: "S Atlantic (Main)",
    38: "Tasman Sea / NZ",      39: "Peru Pacific",
    40: "S Indian Ocean",       41: "SW Australian Coast",
    42: "Tonga",                43: "NW Australia",
    44: "Brazil Coast (N)",     45: "Argentine Atlantic",
    46: "Chilean Coast (S)",    47: "Chilean Coast (N)",
    48: "SE Pacific",           49: "Remote S Pacific",
    50: "W Australian Coast",   51: "Great Australian Bight",
}

RIG_MAP = {"Bark": 0, "Ship": 1, "Brig": 2, "Schr": 3, "Other": 4}


# ── Feature helpers ───────────────────────────────────────────────────────────

def parse_tonnage(s):
    if pd.isna(s):
        return np.nan
    try:
        return float(str(s).split("/")[0].strip())
    except ValueError:
        return np.nan


def clean_rig(s):
    if pd.isna(s):
        return "Other"
    first = str(s).split("/")[0].split()[0].strip().title()
    return first if first in RIG_MAP else "Other"


def nearest_centroid(lats, lons, c_lats, c_lons):
    lat1 = np.radians(lats[:, None])
    lon1 = np.radians(lons[:, None])
    lat2 = np.radians(c_lats[None, :])
    lon2 = np.radians(c_lons[None, :])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return np.argmin(a, axis=1)


# ── Feature building ──────────────────────────────────────────────────────────

def build_features(con):
    print("Loading all valid observations ...")
    df = pd.read_sql_query(
        """SELECT id, VoyageID, vessel, cluster_active, cluster_fleet,
                  Encounter, Year, Month, Lat, Lon, rig, tonnage
           FROM observations
           WHERE Lat IS NOT NULL AND Lon IS NOT NULL AND Year IS NOT NULL
             AND Lat BETWEEN -90 AND 90""",
        con,
    ).reset_index(drop=True)
    df.loc[df["Lon"] < -180, "Lon"] += 360
    df = df[df["Lon"].between(-180, 180)].reset_index(drop=True)
    n0 = len(df)
    print(f"  {n0:,} rows")

    # ── Ground assignment ─────────────────────────────────────────────────────
    print("  Assigning grounds ...")
    centroids = pd.read_sql_query(
        "SELECT cluster, lat_centroid, lon_centroid FROM cluster_summary_active ORDER BY cluster",
        con,
    )
    c_lats = centroids["lat_centroid"].values
    c_lons = centroids["lon_centroid"].values
    c_ids  = centroids["cluster"].values

    ground_id     = df["cluster_active"].fillna(-1).astype(int).values.copy()
    need_centroid = ground_id < 0
    nc_idx = nearest_centroid(
        df.loc[need_centroid, "Lat"].values,
        df.loc[need_centroid, "Lon"].values,
        c_lats, c_lons,
    )
    ground_id[need_centroid] = c_ids[nc_idx]
    df["ground_id"]    = ground_id
    df["ground_label"] = df["ground_id"].map(GROUND_LABELS)
    print(f"    Direct: {(~need_centroid).sum():,}  Nearest centroid: {need_centroid.sum():,}")

    ground_dummies = pd.get_dummies(df["ground_label"], prefix="gnd")
    for lbl in GROUND_LABELS.values():
        col = f"gnd_{lbl}"
        if col not in ground_dummies.columns:
            ground_dummies[col] = 0
    ground_dummies = ground_dummies.reindex(sorted(ground_dummies.columns), axis=1)

    # ── Fleet events ──────────────────────────────────────────────────────────
    print("  Joining fleet features ...")
    fe = pd.read_sql_query(
        "SELECT cluster_fleet, n_vessels, duration_days FROM fleet_events",
        con,
    )
    fe = fe.rename(columns={"n_vessels": "fleet_n_vessels",
                             "duration_days": "fleet_duration_days"})
    df = df.merge(fe, on="cluster_fleet", how="left")
    assert len(df) == n0
    df["fleet_n_vessels"]     = df["fleet_n_vessels"].fillna(0)
    df["fleet_duration_days"] = df["fleet_duration_days"].fillna(0)
    df["in_fleet_event"]      = (df["fleet_n_vessels"] >= 2).astype(int)

    # ── Vessel-level fleet metrics ────────────────────────────────────────────
    voyage_fleet = df.groupby("VoyageID").agg(
        total_days=("cluster_fleet", "count"),
        days_in_fleet=("cluster_fleet", lambda x: (x != -1).sum()),
        n_clusters_visited=("cluster_fleet", lambda x: x[x != -1].nunique()),
    ).reset_index()
    voyage_fleet["days_outside_cluster"] = (
        voyage_fleet["total_days"] - voyage_fleet["days_in_fleet"])
    voyage_fleet["pct_days_in_fleet"] = (
        voyage_fleet["days_in_fleet"] / voyage_fleet["total_days"])
    df = df.merge(
        voyage_fleet[["VoyageID", "n_clusters_visited",
                       "pct_days_in_fleet", "days_outside_cluster"]],
        on="VoyageID", how="left",
    )
    assert len(df) == n0
    df["n_clusters_visited"]   = df["n_clusters_visited"].fillna(0).astype(int)
    df["pct_days_in_fleet"]    = df["pct_days_in_fleet"].fillna(0.0)
    df["days_outside_cluster"] = df["days_outside_cluster"].fillna(0).astype(int)

    # ── Voyage has spoke ──────────────────────────────────────────────────────
    spoke_vids = set(pd.read_sql_query(
        "SELECT DISTINCT VoyageID FROM observations WHERE Encounter='Spoke'", con
    )["VoyageID"])
    df["voyage_has_spoke"] = df["VoyageID"].isin(spoke_vids).astype(int)

    # ── Vessel features ───────────────────────────────────────────────────────
    df["tonnage_num"] = df["tonnage"].apply(parse_tonnage)
    df["tonnage_num"] = df["tonnage_num"].fillna(df["tonnage_num"].median())
    df["rig_enc"]     = df["rig"].apply(clean_rig).map(RIG_MAP)

    # ── Target ───────────────────────────────────────────────────────────────
    df["y"] = (df["Encounter"] == "Strike").astype(int)

    # ── Assemble X ───────────────────────────────────────────────────────────
    base_cols = [
        "Year", "Month",
        "rig_enc", "tonnage_num",
        "in_fleet_event", "fleet_n_vessels", "fleet_duration_days",
        "voyage_has_spoke",
        "n_clusters_visited", "pct_days_in_fleet", "days_outside_cluster",
    ]
    X = pd.concat(
        [df[base_cols].reset_index(drop=True),
         ground_dummies.reset_index(drop=True)],
        axis=1,
    ).astype(float)

    y      = df["y"].values
    groups = df["VoyageID"].values
    print(f"  Feature matrix: {X.shape[0]:,} × {X.shape[1]} columns  "
          f"(Strike rate: {y.mean():.1%})")
    return df, X, y, groups, base_cols


# ── Split and save partitions ─────────────────────────────────────────────────

def split_and_save(df, X, y, groups, base_cols):
    print("\nSplitting 80/20 by VoyageID ...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    print(f"  Train: {len(train_idx):,}  Test: {len(test_idx):,}")

    export_cols = (
        ["VoyageID", "vessel", "Encounter", "Lat", "Lon", "ground_label", "rig", "tonnage"]
        + base_cols + ["y"]
    )

    train_csv = os.path.join(OUTPUT_DIR, "rf_train_80pct.csv")
    test_csv  = os.path.join(OUTPUT_DIR, "rf_test_20pct.csv")

    df.iloc[train_idx][export_cols].to_csv(train_csv, index=False)
    df.iloc[test_idx][export_cols].to_csv(test_csv,  index=False)
    print(f"  Train partition → {train_csv}")
    print(f"  Test partition  → {test_csv}  (reserved for regression)")

    return train_idx, test_idx


# ── Train ─────────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(OUTPUT_DIR, "rf_model.pkl")

def train(X, y, train_idx):
    X_train = X.iloc[train_idx]
    y_train = y[train_idx]

    if os.path.exists(MODEL_PATH):
        print(f"\nLoading saved RF model from {MODEL_PATH} ...")
        with open(MODEL_PATH, "rb") as f:
            rf = pickle.load(f)
    else:
        print("\nTraining RandomForest on 80% partition ...")
        rf = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=30,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(X_train, y_train)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(rf, f)
        print(f"  Model saved → {MODEL_PATH}")

    train_auroc = roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1])
    print(f"  Train AUROC (sanity): {train_auroc:.4f}")
    return rf, X_train, y_train, train_auroc


# ── Feature importance ────────────────────────────────────────────────────────

def plot_feature_importance(rf, X_train, y_train, outfile):
    cache = outfile.replace(".png", "_values.json")
    if os.path.exists(cache):
        print(f"  Loading cached FI values ({cache})")
        with open(cache) as f:
            return json.load(f)
    print("  Computing permutation importance on training set ...")
    result = permutation_importance(
        rf, X_train, y_train,
        n_repeats=5, random_state=42, n_jobs=-1, scoring="roc_auc",
    )
    names    = list(X_train.columns)
    imp_mean = result.importances_mean
    imp_std  = result.importances_std

    top_idx   = np.argsort(imp_mean)[::-1][:25]
    top_names = [names[i] for i in top_idx]
    top_mean  = imp_mean[top_idx]
    top_std   = imp_std[top_idx]

    fig, ax = plt.subplots(figsize=(8, 8))
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_mean[::-1], xerr=top_std[::-1],
            align="center", color="steelblue", ecolor="gray", capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Permutation importance (AUROC drop)")
    ax.set_title("Feature Importance — Training Set")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    fi_dict = dict(zip(names, imp_mean.tolist()))
    with open(cache, "w") as f:
        json.dump(fi_dict, f, indent=2)
    print(f"  FI plot → {outfile}")
    return fi_dict


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    df, X, y, groups, base_cols = build_features(con)
    con.close()

    train_idx, test_idx = split_and_save(df, X, y, groups, base_cols)

    rf, X_train, y_train, train_auroc = train(X, y, train_idx)

    print("\nGenerating exploration outputs ...")
    feat_imp = plot_feature_importance(
        rf, X_train, y_train,
        os.path.join(OUTPUT_DIR, "feature_importance.png"),
    )

    meta = {
        "n_total":       int(len(df)),
        "n_train":       int(len(train_idx)),
        "n_test":        int(len(test_idx)),
        "positive_rate": float(y.mean()),
        "train_auroc":   float(train_auroc),
        "top_features_fi": {k: round(v, 6) for k, v in
                             sorted(feat_imp.items(), key=lambda x: -x[1])[:30]},
    }
    meta_path = os.path.join(OUTPUT_DIR, "rf_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata → {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
