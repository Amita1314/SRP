#!/usr/bin/env python3
"""
rf_productivity.py — Random Forest: Whale Strike Productivity

Predicts whether any ship-day observation results in a Strike.
Full 477k dataset. Grounds assigned by direct HDBSCAN label (68k rows)
or nearest centroid (408k rows). 80/20 VoyageID-grouped split.
"""
import json
import os
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

DB_PATH    = "whaling.db"
OUTPUT_DIR = "output"

GROUND_LABELS = {
    0:  "Hudson Bay",
    1:  "Baja California",
    2:  "Japan / Okhotsk",
    3:  "Solomon Islands",
    4:  "Gulf of Guinea",
    5:  "S Atlantic (Tristan)",
    6:  "Angola Atlantic",
    7:  "Mid-Atlantic",
    8:  "Hawaii Pacific",
    9:  "Japan Sea",
    10: "Brazil Atlantic",
    11: "Equatorial Pacific",
    12: "Caribbean",
    13: "Gulf of Alaska",
    14: "Madagascar",
    15: "E Africa Indian Ocean",
    16: "Gulf of Mexico",
    17: "US East Coast",
    18: "N Atlantic (Azores)",
    19: "Equatorial Atlantic",
    20: "N Pacific (Kamchatka)",
    21: "Bering Sea (West)",
    22: "Western Arctic",
    23: "Bering Strait",
    24: "Cape Verde (S)",
    25: "Cape Verde (N)",
    26: "Kamchatka / Okhotsk",
    27: "Okhotsk Sea",
    28: "Japan Ground (S)",
    29: "Japan Ground (Main)",
    30: "Indonesia Pacific",
    31: "S Brazil Atlantic",
    32: "Crozet / Kerguelen",
    33: "Indian Ocean (Central)",
    34: "S Indian Ocean (W)",
    35: "Crozet Islands",
    36: "S Atlantic (Mid-East)",
    37: "S Atlantic (Main)",
    38: "Tasman Sea / NZ",
    39: "Peru Pacific",
    40: "S Indian Ocean",
    41: "SW Australian Coast",
    42: "Tonga",
    43: "NW Australia",
    44: "Brazil Coast (N)",
    45: "Argentine Atlantic",
    46: "Chilean Coast (S)",
    47: "Chilean Coast (N)",
    48: "SE Pacific",
    49: "Remote S Pacific",
    50: "W Australian Coast",
    51: "Great Australian Bight",
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


def parse_enc_count(json_str, key):
    if pd.isna(json_str):
        return 0
    try:
        return int(json.loads(json_str).get(key, 0))
    except (ValueError, TypeError):
        return 0


def nearest_centroid(lats, lons, c_lats, c_lons):
    """
    Vectorized nearest-centroid by haversine proxy.
    Returns integer index (0..51) of nearest centroid for each row.
    Shape: lats/lons (n,), c_lats/c_lons (52,) → returns (n,) int array.
    """
    lat1 = np.radians(lats[:, None])
    lon1 = np.radians(lons[:, None])
    lat2 = np.radians(c_lats[None, :])
    lon2 = np.radians(c_lons[None, :])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
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
    # Fix antimeridian
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
    c_ids  = centroids["cluster"].values      # should be 0..51

    ground_id = df["cluster_active"].fillna(-1).astype(int).values.copy()
    need_centroid = ground_id < 0
    print(f"    Direct ground: {(~need_centroid).sum():,}  "
          f"Nearest centroid: {need_centroid.sum():,}")

    # Nearest centroid for unassigned rows
    nc_idx = nearest_centroid(
        df.loc[need_centroid, "Lat"].values,
        df.loc[need_centroid, "Lon"].values,
        c_lats, c_lons,
    )
    ground_id[need_centroid] = c_ids[nc_idx]
    df["ground_id"] = ground_id
    df["ground_label"] = df["ground_id"].map(GROUND_LABELS)

    ground_dummies = pd.get_dummies(df["ground_label"], prefix="gnd")
    # Ensure all 52 columns present (some may be absent if split unlucky)
    for lbl in GROUND_LABELS.values():
        col = f"gnd_{lbl}"
        if col not in ground_dummies.columns:
            ground_dummies[col] = 0
    ground_dummies = ground_dummies.reindex(
        sorted(ground_dummies.columns), axis=1
    )

    # ── Fleet events ──────────────────────────────────────────────────────────
    print("  Joining fleet features ...")
    fe = pd.read_sql_query(
        "SELECT cluster_fleet, n_vessels, duration_days, encounter_types FROM fleet_events",
        con,
    )
    fe["fleet_sight_count"] = fe["encounter_types"].apply(
        lambda x: parse_enc_count(x, "Sight"))
    fe["fleet_spoke_count"] = fe["encounter_types"].apply(
        lambda x: parse_enc_count(x, "Spoke"))
    fe = fe.rename(columns={"n_vessels": "fleet_n_vessels",
                             "duration_days": "fleet_duration_days"})
    fe = fe[["cluster_fleet", "fleet_n_vessels", "fleet_duration_days",
             "fleet_sight_count", "fleet_spoke_count"]]

    df = df.merge(fe, on="cluster_fleet", how="left")
    assert len(df) == n0
    for col in ["fleet_n_vessels", "fleet_duration_days",
                "fleet_sight_count", "fleet_spoke_count"]:
        df[col] = df[col].fillna(0)
    df["in_fleet_event"] = (df["fleet_n_vessels"] >= 2).astype(int)

    # ── Leader / follower ─────────────────────────────────────────────────────
    lf = pd.read_sql_query(
        "SELECT cluster_fleet, voyageID AS VoyageID, arrival_rank FROM leader_follower",
        con,
    )
    df = df.merge(lf, on=["cluster_fleet", "VoyageID"], how="left")
    assert len(df) == n0
    df["arrival_rank"] = df["arrival_rank"].fillna(0).astype(int)
    df["is_leader"]    = (df["arrival_rank"] == 1).astype(int)

    # ── Convoys ───────────────────────────────────────────────────────────────
    conv = pd.read_sql_query(
        """SELECT voyage_a AS VoyageID, n_consecutive_days FROM fleet_convoys
           UNION ALL
           SELECT voyage_b, n_consecutive_days FROM fleet_convoys""",
        con,
    )
    conv_agg = (conv.groupby("VoyageID")["n_consecutive_days"]
                .max().reset_index()
                .rename(columns={"n_consecutive_days": "convoy_max_days"}))
    conv_agg["convoy_member"] = 1
    df = df.merge(conv_agg, on="VoyageID", how="left")
    assert len(df) == n0
    df["convoy_member"]   = df["convoy_member"].fillna(0).astype(int)
    df["convoy_max_days"] = df["convoy_max_days"].fillna(0).astype(int)

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
        "fleet_sight_count", "fleet_spoke_count",
        "arrival_rank", "is_leader",
        "convoy_member", "convoy_max_days",
        "voyage_has_spoke",
    ]
    X = pd.concat(
        [df[base_cols].reset_index(drop=True),
         ground_dummies.reset_index(drop=True)],
        axis=1,
    ).astype(float)

    y      = df["y"].values
    groups = df["VoyageID"].values

    print(f"  Feature matrix: {X.shape[0]:,} × {X.shape[1]} columns")
    print(f"  Positive rate (Strike): {y.mean():.1%}")
    return df, X, y, groups


# ── Train / evaluate ──────────────────────────────────────────────────────────

def train_evaluate(df, X, y, groups):
    print("\nSplitting 80/20 by VoyageID ...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx],      y[test_idx]
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")
    print(f"  Train +rate: {y_train.mean():.1%}  Test +rate: {y_test.mean():.1%}")

    print("\nTraining RandomForest (500 trees, class_weight=balanced) ...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=30,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    y_prob = rf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auroc = roc_auc_score(y_test, y_prob)
    f1    = f1_score(y_test, y_pred)
    print(f"\n  AUROC : {auroc:.4f}")
    print(f"  F1    : {f1:.4f}")

    return rf, X_train, X_test, y_train, y_test, y_prob, train_idx, test_idx, auroc, f1


def auroc_by_decade(df, test_idx, y_test, y_prob):
    years   = df.iloc[test_idx]["Year"].values
    decades = (years // 10) * 10
    results = {}
    for dec in sorted(set(decades)):
        mask = decades == dec
        if mask.sum() < 50 or y_test[mask].sum() < 5:
            continue
        try:
            results[int(dec)] = roc_auc_score(y_test[mask], y_prob[mask])
        except Exception:
            pass
    print("\n  AUROC by decade:")
    for dec, auc in results.items():
        print(f"    {dec}s: {auc:.3f}  (n={int((decades == dec).sum())})")
    return results


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_precision_recall(y_test, y_prob, outfile):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=1.5, color="steelblue")
    ax.axhline(y_test.mean(), color="gray", linestyle="--",
               label=f"Baseline ({y_test.mean():.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve (Strike class)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"  PR curve → {outfile}")


def plot_feature_importance(rf, X_test, y_test, outfile):
    print("  Computing permutation importance (n_repeats=5) ...")
    result = permutation_importance(
        rf, X_test, y_test,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
        scoring="roc_auc",
    )
    names    = list(X_test.columns)
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
    ax.set_title("Top 25 Feature Importances")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"  Feature importance → {outfile}")

    return dict(zip(names, imp_mean.tolist()))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH)

    df, X, y, groups = build_features(con)
    con.close()

    rf, X_train, X_test, y_train, y_test, y_prob, \
        train_idx, test_idx, auroc, f1 = train_evaluate(df, X, y, groups)

    decade_auroc = auroc_by_decade(df, test_idx, y_test, y_prob)

    print("\nGenerating plots ...")
    plot_precision_recall(y_test, y_prob,
                          os.path.join(OUTPUT_DIR, "pr_curve.png"))
    feat_imp = plot_feature_importance(
        rf, X_test, y_test,
        os.path.join(OUTPUT_DIR, "feature_importance.png"),
    )

    # ── Save test set predictions ─────────────────────────────────────────────
    print("\nSaving test set CSV ...")
    test_df = df.iloc[test_idx][
        ["VoyageID", "vessel", "Year", "Month", "Encounter",
         "Lat", "Lon", "ground_label", "rig", "tonnage"]
    ].copy()
    test_df["y_true"] = y_test
    test_df["y_prob"] = y_prob.round(4)
    test_df["y_pred"] = (y_prob >= 0.5).astype(int)
    test_csv = os.path.join(OUTPUT_DIR, "rf_test_predictions.csv")
    test_df.to_csv(test_csv, index=False)
    print(f"  Test predictions → {test_csv}  ({len(test_df):,} rows)")

    meta = {
        "n_rows":        int(len(df)),
        "n_train":       int(len(X_train)),
        "n_test":        int(len(X_test)),
        "positive_rate": float(y.mean()),
        "auroc":         float(auroc),
        "f1":            float(f1),
        "decade_auroc":  {str(k): round(float(v), 4) for k, v in decade_auroc.items()},
        "top_features":  {k: round(v, 6) for k, v in
                          sorted(feat_imp.items(), key=lambda x: -x[1])[:30]},
    }
    meta_path = os.path.join(OUTPUT_DIR, "rf_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata → {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
