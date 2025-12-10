"""
XGBoost model training for CrimeHotspotSim

Author: Julian Fennema

This script:
1. Loads the processed weekly dataset (frames.csv).
2. Builds a binary hotspot label (top 20% of crime_count).
3. Trains an XGBoostClassifier using *structural* features
   (lags, neighbor crime, poverty, nightlight, etc.)
4. Reports validation AP (PR-AUC) on the last 4 weeks.
5. Saves:
   - xgb_model.pkl (trained XGBoost model)
   - xgb_latest_scores_after.csv (latest week with:
       - risk_score_before (no intervention)
       - risk_score_after (with patrol/ped intervention)
     )

Run example:
    python models/xgboost_modeling.py \
        --frames data/processed/frames.csv \
        --models_dir models \
        --pred_dir predictions
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier


def logistic(x):
    """Sigmoid function to go from log-odds back to probability."""
    return 1.0 / (1.0 + np.exp(-x))


def logit(p, eps=1e-6):
    """Logit transform with clipping to avoid infinities."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main(frames_csv: str, models_dir: str, pred_dir: str) -> None:
    # ---------------------------------------------------------------
    # 1. Load processed weekly dataset
    # ---------------------------------------------------------------
    panel = pd.read_csv(frames_csv, parse_dates=["week_start"])

    # ---------------------------------------------------------------
    # 2. Define binary hotspot label (top 20% of crime_count)
    # ---------------------------------------------------------------
    q80 = panel["crime_count"].quantile(0.8)
    panel["hotspot"] = (panel["crime_count"] >= q80).astype(int)

    # ---------------------------------------------------------------
    # 3. FEATURES FOR TRAINING (STRUCTURAL ONLY)
    #    We intentionally DO NOT include police_patrols or ped_activity
    #    here. They will be used in a separate simulation layer.
    # ---------------------------------------------------------------
    features = [
        # Temporal crime history
        "lag1", "lag2", "lag3",

        # Spatial neighbor crime features
        "neighbor_crime_1wk",
        "neighbor_crime_4wk",
        "neighbor_crime_8wk",

        # Context / environment
        "poi_density",
        "nightlight",
        "pop_density",
        "poverty_rate",
        "weather_temp",
        "events_count",
        # NOTE: no police_patrols, no ped_activity here on purpose
    ]

    # Safety check: make sure all required columns exist
    missing = [f for f in features if f not in panel.columns]
    if missing:
        raise ValueError(f"Missing feature columns in frames.csv: {missing}")

    # We also expect these columns for the simulation layer later:
    for col in ["police_patrols", "ped_activity"]:
        if col not in panel.columns:
            raise ValueError(
                f"Expected column '{col}' in frames.csv for simulation, but it is missing."
            )

    # ---------------------------------------------------------------
    # 4. Train / validation split (time-aware: last 4 weeks as val)
    # ---------------------------------------------------------------
    latest_week = panel["week_start"].max()
    train_df = panel[panel["week_start"] < latest_week]
    val_df = panel[panel["week_start"] >= latest_week - pd.Timedelta(days=28)]

    X_train = train_df[features]
    y_train = train_df["hotspot"]

    X_val = val_df[features]
    y_val = val_df["hotspot"]

    # ---------------------------------------------------------------
    # 5. Build and train XGBoost model (baseline risk model)
    # ---------------------------------------------------------------
    xgb = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )

    xgb.fit(X_train, y_train)

    # ---------------------------------------------------------------
    # Validation metric (PR-AUC / Average Precision)
    # ---------------------------------------------------------------
    val_pred = xgb.predict_proba(X_val)[:, 1]
    ap = average_precision_score(y_val, val_pred)
    print(f"XGBoost Validation AP (baseline model): {ap:.4f}")

    # ---------------------------------------------------------------
    # FEATURE IMPORTANCE CHECK (XGBoost)
    # ---------------------------------------------------------------
    print("\n=== XGBoost Feature Importances (baseline model) ===")
    importances = xgb.feature_importances_

    for name, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
        print(f"{name:20s} {imp:.5f}")
    print("================================================\n")

    # ---------------------------------------------------------------
    # 6. Save model
    # ---------------------------------------------------------------
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    model_path = models_path / "xgb_model.pkl"
    joblib.dump(xgb, model_path)
    print(f"Saved XGBoost model to: {model_path}")

    # ---------------------------------------------------------------
    # 7. RISK SCORE LATEST WEEK (BASELINE + SIMULATION)
    # ---------------------------------------------------------------
    latest_df = panel[panel["week_start"] == latest_week].copy()

    # Base risk from XGBoost (NO intervention) – probability of hotspot
    latest_df["risk_score_before"] = xgb.predict_proba(latest_df[features])[:, 1]

    # ----------------------------------------------------------------
    # Policy Simulation Layer (multiplicative adjustment on risk)
    #
    # Assumptions:
    # - Increased pedestrians DECREASE crime by up to 30% at +100% increase.
    # - Increased police patrols DECREASE crime by up to:
    #     * 20% in NON-HOTSPOT areas at +100% increase
    #     * 40% in HOTSPOT areas at +100% increase
    #
    # We:
    # 1) Ask the user for % change in pedestrians and patrols.
    # 2) Compute reduction factors.
    # 3) Apply those factors to risk_score_before.
    #
    # Example (pedestrians):
    #   foot_pct = 50  → factor_ped = 1 - 0.30 * 0.5 = 0.85
    #   → 15% reduction in risk from pedestrians.
    # ----------------------------------------------------------------

    # Get user inputs (percent change)
    print("=== Intervention Setup ===")
    print("Enter percent change (can be negative for decreases, 0 for no change):")

    try:
        foot_pct = float(input("Percent change in pedestrian traffic (e.g. 20 or -10): "))
    except Exception:
        foot_pct = 0.0
        print("Invalid input for pedestrian traffic. Defaulting to 0% change.")

    try:
        police_pct = float(input("Percent change in police patrols (e.g. 30 or -15): "))
    except Exception:
        police_pct = 0.0
        print("Invalid input for police patrols. Defaulting to 0% change.")

    print()
    print(f"Using interventions: foot_pct={foot_pct}%, police_pct={police_pct}%")
    print(
        "Assumptions:\n"
        "  +100% pedestrians → -30% crime risk (everywhere)\n"
        "  +100% police → -20% in non-hotspots, -40% in hotspots"
    )
    print()

    # Convert percentage to fraction
    foot_scale = foot_pct / 100.0
    police_scale = police_pct / 100.0

    # Pedestrian effect: up to 30% risk reduction at +100%
    # factor_ped < 1 → risk goes down; >1 → risk goes up (if foot_pct < 0)
    factor_ped = 1.0 - 0.30 * foot_scale
    factor_ped = max(factor_ped, 0.0)  # avoid negative risk multipliers

    # Police effect: depends on hotspot vs non-hotspot
    # non-hotspot: up to 20% reduction at +100%
    # hotspot:     up to 40% reduction at +100%
    non_hot_factor = 1.0 - 0.20 * police_scale
    hot_factor = 1.0 - 0.40 * police_scale
    non_hot_factor = max(non_hot_factor, 0.0)
    hot_factor = max(hot_factor, 0.0)

    # Build per-row police factor based on hotspot label
    is_hotspot = latest_df["hotspot"] == 1
    police_factor = np.where(is_hotspot, hot_factor, non_hot_factor)

    # Total factor = pedestrian effect * police effect
    total_factor = factor_ped * police_factor

    # Apply multiplicative adjustment to probabilities
    risk_score_before = latest_df["risk_score_before"].values.astype(float)
    risk_score_after = risk_score_before * total_factor

    # Clip to [0, 1] to keep them valid probabilities
    risk_score_after = np.clip(risk_score_after, 0.0, 1.0)

    latest_df["risk_score_after"] = risk_score_after

    # ---------------------------------------------------------------
    # 8. Save latest-week risk_scores WITH simulation
    # ---------------------------------------------------------------
    pred_path = Path(pred_dir)
    pred_path.mkdir(parents=True, exist_ok=True)
    out_csv = pred_path / "xgb_latest_scores_after.csv"
    latest_df.to_csv(out_csv, index=False)

    print(f"Saved latest-week baseline + simulated scores to: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True,
                        help="Path to processed frames.csv")
    parser.add_argument("--models_dir", default="models",
                        help="Directory to save xgb_model.pkl")
    parser.add_argument("--pred_dir", default="predictions",
                        help="Directory to save latest scores CSV")
    args = parser.parse_args()

    main(args.frames, args.models_dir, args.pred_dir)

# Run: python models/xgboost_modeling.py --frames data/processed/frames.csv --models_dir models
