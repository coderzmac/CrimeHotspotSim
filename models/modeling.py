"""
Machine Learning model training for CrimeHotspotSim
Author: Emmanuel Bautista

This script:
1. Loads the processed dataset (frames.csv).
2. Builds a binary hotspot label (top 20% by weekly crime count).
3. Trains a GradientBoostingClassifier (simple but strong baseline).
4. Scores the latest week BEFORE intervention.
5. Simulates an intervention (user-specified change in police patrols
   and pedestrian activity) and re-scores AFTER intervention using a
   multiplicative risk adjustment (same assumptions as XGBoost script).
6. Saves:
   - baseline_ml.pkl (trained model)
   - latest_scores_before.csv
   - latest_scores_after.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import joblib

def main(frames_csv, models_dir, pred_dir):

    # ---------------------------------------------------------------
    # 1. Load processed weekly dataset
    # ---------------------------------------------------------------
    panel = pd.read_csv(frames_csv, parse_dates=["week_start"])

    # ---------------------------------------------------------------
    # 2. Define binary hotspot label:
    # hotspot = 1 if crime_count >= 80th percentile overall
    # ---------------------------------------------------------------
    q80 = panel["crime_count"].quantile(0.8)
    panel["hotspot"] = (panel["crime_count"] >= q80).astype(int)

    # ---------------------------------------------------------------
    # Model feature list including spatial neighbor predictors
    # ---------------------------------------------------------------
    features = [
        # Temporal lags
        "lag1", "lag2", "lag3",

        # Spatial neighbor crime features
        "neighbor_crime_1wk",
        "neighbor_crime_4wk",
        "neighbor_crime_8wk",

        # Context + demographics
        "poi_density", "nightlight", "poverty_rate",
        "pop_density", "weather_temp", "events_count",

        # Interventions / levers
        "police_patrols", "ped_activity"
    ]

    # Confirm features exist
    print("\n--- MODEL FEATURES ---")
    for f in features:
        print(f"{f}: {'OK' if f in panel.columns else 'MISSING'}")


    # Automatically include ACS features if present
    extra = [
        f for f in ["pop_total", "median_income", "unemployment_rate"]
        if f in panel.columns
    ]



    # Latest week is used ONLY for inference, not training
    latest_week = panel["week_start"].max()


    # ---------------------------------------------------------------
    # 3. Split training / validation (avoid leakage!)
    # ---------------------------------------------------------------
    train_df = panel[panel["week_start"] < latest_week]
    val_df = panel[panel["week_start"] >= latest_week - pd.Timedelta(days=28)]

    X_train, y_train = train_df[features], train_df["hotspot"]
    X_val, y_val = val_df[features], val_df["hotspot"]


    # ---------------------------------------------------------------
    # 4. Build ML pipeline (scaler + gradient boosting)
    # ---------------------------------------------------------------
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(random_state=42))
    ])

    # Train model
    pipe.fit(X_train, y_train)

    # Quick validation metric (PR-AUC)
    val_pred = pipe.predict_proba(X_val)[:, 1]
    ap = average_precision_score(y_val, val_pred)
    print(f"Validation AP: {ap:.4f}")



    # Validation prediction
    y_val_pred = pipe.predict_proba(X_val)[:, 1]
    ap = average_precision_score(y_val, y_val_pred)
    print(f"Validation AP: {ap:.4f}")

    # ---------------------------------------------------------------
    # FEATURE IMPORTANCE CHECK
    # ---------------------------------------------------------------
    print("\n=== Feature Importances (GradientBoosting) ===")
    clf = pipe.named_steps["clf"]  # GradientBoostingClassifier inside the pipeline

    for name, imp in sorted(zip(features, clf.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"{name:20s} {imp:.5f}")
    print("================================================\n")





    # ---------------------------------------------------------------
    # 5. Save trained model
    # ---------------------------------------------------------------
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    model_path = models_path / "baseline_ml.pkl"
    joblib.dump(pipe, model_path)


    # ---------------------------------------------------------------
    # 6. Score latest week BEFORE intervention
    # ---------------------------------------------------------------
    latest_df = panel[panel["week_start"] == latest_week].copy()
    latest_df["score_before"] = pipe.predict_proba(latest_df[features])[:, 1]


    # ---------------------------------------------------------------
    # 7. Apply intervention (same assumptions as XGBoost script)
    #
    # Assumptions:
    # - Increased pedestrians DECREASE crime by up to 30% at +100% increase.
    # - Increased police patrols DECREASE crime by up to:
    #     * 20% in NON-HOTSPOT areas at +100% increase
    #     * 40% in HOTSPOT areas at +100% increase
    #
    # 1) Ask the user for % change in pedestrians and patrols.
    # 2) Compute reduction factors.
    # 3) Apply those factors multiplicatively to score_before.
    #
    #       Not re-running the model with changed features here;
    #       we are directly adjusting the risk scores based on policy
    #       assumptions, just like in xgboost_modeling.py.
    # ---------------------------------------------------------------
    after_df = latest_df.copy()

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
    factor_ped = 1.0 - 0.30 * foot_scale
    factor_ped = max(factor_ped, 0.0)  # avoid negative risk multipliers

    # Police effect: depends on hotspot vs non-hotspot
    non_hot_factor = 1.0 - 0.20 * police_scale
    hot_factor = 1.0 - 0.40 * police_scale
    non_hot_factor = max(non_hot_factor, 0.0)
    hot_factor = max(hot_factor, 0.0)

    # Build per-row police factor based on hotspot label
    is_hotspot = after_df["hotspot"] == 1
    police_factor = np.where(is_hotspot, hot_factor, non_hot_factor)

    # Total factor = pedestrian effect * police effect
    total_factor = factor_ped * police_factor

    # Apply multiplicative adjustment to baseline probabilities
    score_base = after_df["score_before"].values.astype(float)
    score_after = score_base * total_factor

    # Clip to [0, 1] to keep them valid probabilities
    score_after = np.clip(score_after, 0.0, 1.0)

    after_df["score_after"] = score_after

    # Optional sanity check: how much did scores move?
    diff = after_df["score_after"] - after_df["score_before"]
    print("Simulation: max |Δscore|:", float(np.abs(diff).max()))
    print("Simulation: mean Δscore:", float(diff.mean()))


    # ---------------------------------------------------------------
    # 8. Save predictions (for mapping in Kepler.gl or Folium)
    # ---------------------------------------------------------------
    pred_path = Path(pred_dir)
    pred_path.mkdir(parents=True, exist_ok=True)

    before_csv = pred_path / "latest_scores_before.csv"
    after_csv = pred_path / "latest_scores_after.csv"

    latest_df.to_csv(before_csv, index=False)
    after_df.to_csv(after_csv, index=False)

    print(f"Saved predictions:\n- {before_csv}\n- {after_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True,
                        help="Path to processed frames.csv")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--pred_dir", default="predictions")
    args = parser.parse_args()

    main(args.frames, args.models_dir, args.pred_dir)

# Run: python models/modeling.py --frames data/processed/frames.csv --models_dir models --pred_dir predictions
