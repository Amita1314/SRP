#!/usr/bin/env python3
"""
regression_step4.py — Co-Duction Step 4: Deductive Hypothesis Testing

Logistic regression on the 20% holdout (never seen by the RF).

Model:
  Pr(strike=1) = logit(β₀ + β₁·voyage_has_spoke + β₂·tonnage_num
                        + ground_dummies + rig_dummies + year_fe + u)

Standard errors clustered by VoyageID (voyage_has_spoke is voyage-level,
so observations within the same voyage are not independent).
"""
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

TEST_CSV  = "output/rf_test_20pct.csv"
OUT_JSON  = "output/regression_results.json"

RIG_CATS = {"Bark", "Ship", "Brig", "Schr"}


def clean_rig(s):
    if pd.isna(s):
        return "Other"
    first = str(s).split("/")[0].split()[0].strip().title()
    return first if first in RIG_CATS else "Other"


def build_matrix(df):
    # Main IV + continuous control
    parts = [df[["voyage_has_spoke", "tonnage_num"]].astype(float)]

    # Ground dummies — drop most common as reference
    ref_ground = df["ground_label"].value_counts().idxmax()
    gnd = pd.get_dummies(df["ground_label"], prefix="gnd").astype(float)
    gnd = gnd.drop(columns=[f"gnd_{ref_ground}"], errors="ignore")
    parts.append(gnd)

    # Rig dummies — drop "Other" as reference
    df["rig_clean"] = df["rig"].apply(clean_rig)
    rig = pd.get_dummies(df["rig_clean"], prefix="rig").astype(float)
    rig = rig.drop(columns=["rig_Other"], errors="ignore")
    parts.append(rig)

    # Year fixed effects — drop earliest year as reference
    yr = pd.get_dummies(df["Year"].astype(int), prefix="yr").astype(float)
    yr = yr.drop(columns=[yr.columns[0]], errors="ignore")
    parts.append(yr)

    X = pd.concat(parts, axis=1)
    X = sm.add_constant(X)
    return X, ref_ground


def main():
    print(f"Loading test partition: {TEST_CSV}")
    df = pd.read_csv(TEST_CSV)
    df = df[df["Encounter"] != "Spoke"].reset_index(drop=True)
    print(f"  {len(df):,} rows (Spoke rows excluded)  |  Strike rate: {df['y'].mean():.3f}")
    print(f"  Voyages: {df['VoyageID'].nunique():,}")

    X, ref_ground = build_matrix(df)
    y = df["y"]
    voyage_ids = df["VoyageID"]

    print(f"\nFeature matrix: {X.shape[0]:,} × {X.shape[1]} columns")
    print(f"  Reference ground: {ref_ground}")
    print(f"  Reference rig:    Other")
    print(f"  Reference year:   {df['Year'].min():.0f}")

    print("\nFitting logistic regression (clustered SEs by VoyageID) ...")
    model  = sm.Logit(y, X)
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": voyage_ids},
        maxiter=300,
        method="bfgs",
        disp=False,
    )

    # ── Key coefficients ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("KEY RESULTS")
    print("="*60)

    focus = ["const", "voyage_has_spoke", "tonnage_num",
             "rig_Bark", "rig_Brig", "rig_Ship", "rig_Schr"]
    tbl = result.summary2().tables[1]
    key = tbl.loc[tbl.index.isin(focus), ["Coef.", "Std.Err.", "z", "P>|z|",
                                           "[0.025", "0.975]"]]
    print(key.to_string())

    # Top 5 ground coefficients by absolute value
    gnd_rows = tbl[tbl.index.str.startswith("gnd_")]
    top_gnd  = gnd_rows.reindex(gnd_rows["Coef."].abs().sort_values(ascending=False).index).head(5)
    print(f"\nTop 5 ground effects (vs {ref_ground}):")
    print(top_gnd[["Coef.", "Std.Err.", "P>|z|"]].to_string())

    print("\n" + "="*60)
    print(f"Pseudo R²  (McFadden): {result.prsquared:.4f}")
    print(f"Log-likelihood:        {result.llf:.1f}")
    print(f"AIC:                   {result.aic:.1f}")
    print(f"N observations:        {int(result.nobs):,}")
    print("="*60)

    # ── Save results ──────────────────────────────────────────────────────────
    tbl_full = result.summary2().tables[1]
    out = {
        "n_obs":          int(result.nobs),
        "pseudo_r2":      round(result.prsquared, 6),
        "log_likelihood": round(result.llf, 3),
        "aic":            round(result.aic, 3),
        "reference_ground": ref_ground,
        "coefficients": {
            var: {
                "coef":   round(float(tbl_full.loc[var, "Coef."]), 6),
                "se":     round(float(tbl_full.loc[var, "Std.Err."]), 6),
                "z":      round(float(tbl_full.loc[var, "z"]), 4),
                "pvalue": round(float(tbl_full.loc[var, "P>|z|"]), 6),
                "ci_low": round(float(tbl_full.loc[var, "[0.025"]), 6),
                "ci_high":round(float(tbl_full.loc[var, "0.975]"]), 6),
            }
            for var in tbl_full.index
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull results → {OUT_JSON}")


if __name__ == "__main__":
    main()
