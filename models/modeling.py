"""
Machine Learning model training for CrimeHotspotSim
Author: Emmanuel Bautista

This script:
1. Loads the processed dataset (frames.csv).
2. Builds a binary hotspot label (top 20% by weekly crime count).
3. Trains a GradientBoostingClassifier (simple but strong baseline).
4. Scores the latest week BEFORE intervention.
5. Simulates an intervention (add 10 patrol cars, +5% pedestrian activity)
   and re-scores AFTER intervention.
6. Saves:
   - baseline_ml.pkl (trained model)
   - latest_scores_before.csv
   - latest_scores_after.csv
"""

import argparse
import pandas as pd
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

    # Model features
    features = [
        "lag1", "lag2", "lag3",
        "poi_density", "nightlight", "income",
        "pop_density", "weather_temp", "events_count",
        "patrol_cars", "ped_activity"
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
    # 7. Apply simple intervention (example)
    # ---------------------------------------------------------------
    after_df = latest_df.copy()

    # Simulate "add 10 patrol units" (counterfactual)
    after_df["patrol_cars"] = after_df["patrol_cars"] + 10.0

    # Simulate "+5% pedestrian activity"
    after_df["ped_activity"] = after_df["ped_activity"] * 1.05

    # Re-predict AFTER intervention
    after_df["score_after"] = pipe.predict_proba(after_df[features])[:, 1]


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