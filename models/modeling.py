import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_ml(frames_csv: str, models_dir: str, pred_dir: str):
    """
    Train a simple ML hotspot model and generate scores for the latest week.
    """

    # 1. Load the preprocessed panel data
    panel = pd.read_csv(frames_csv, parse_dates=["week_start"])


    # 2. Define a binary "hotspot" label
    q80 = panel["crime_count"].quantile(0.8)
    panel["hotspot"] = (panel["crime_count"] >= q80).astype(int)

    # Feature columns we will use in the model.
    # Right now, lags + simple placeholders.
    feature_cols = [
        "lag1",
        "lag2",
        "lag3",
        "poi_density",
        "nightlight",
        "income",
        "pop_density",
        "weather_temp",
        "events_count",
        "patrol_cars",
        "ped_activity",
    ]

    # Identify the most recent week for inference.
    latest_week = panel["week_start"].max()

    # 3. Train/validation split (time-aware)
    train_df = panel[panel["week_start"] < latest_week].copy()
    val_df = panel[
        panel["week_start"] >= latest_week - pd.Timedelta(days=28)
    ].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["hotspot"]
    X_val = val_df[feature_cols]
    y_val = val_df["hotspot"]

    # 4. Build and train the model
    # Using a simple pipeline:
    #  - StandardScaler: normalize features
    #  - GradientBoostingClassifier: tree-based model for classification
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=42)),
        ]
    )

    pipe.fit(X_train, y_train)

    # 5. Evaluate using Average Precision (PR-AUC)
    val_pred = pipe.predict_proba(X_val)[:, 1]
    ap = average_precision_score(y_val, val_pred)

    # 6. Save the trained model
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    model_path = models_path / "baseline_ml.pkl"
    joblib.dump(pipe, model_path)

    # 7. Score the latest week (BEFORE intervention)
    latest_df = panel[panel["week_start"] == latest_week].copy()
    latest_df["score_before"] = pipe.predict_proba(
        latest_df[feature_cols]
    )[:, 1]

    # 8. Apply a simple intervention and re-score (AFTER)
    after_df = latest_df.copy()

    # Example intervention: +10 patrol cars and +5% pedestrian activity
    after_df["patrol_cars"] = after_df["patrol_cars"] + 10.0
    after_df["ped_activity"] = after_df["ped_activity"] * 1.05

    after_df["score_after"] = pipe.predict_proba(
        after_df[feature_cols]
    )[:, 1]

    # ------------------------------------------------------
    # 9. Save per-cell scores to CSV (for mapping)
    # ------------------------------------------------------
    pred_path = Path(pred_dir)
    pred_path.mkdir(parents=True, exist_ok=True)

    before_csv = pred_path / "latest_scores_before.csv"
    after_csv = pred_path / "latest_scores_after.csv"

    latest_df[
        ["grid_id", "lat", "lon", "crime_count", "lag1", "lag2", "lag3", "score_before"]
    ].to_csv(before_csv, index=False)

    after_df[
        ["grid_id", "lat", "lon", "crime_count", "lag1", "lag2", "lag3", "score_after"]
    ].to_csv(after_csv, index=False)

    print(
        {
            "validation_AP": float(ap),
            "model_path": str(model_path),
            "before_csv": str(before_csv),
            "after_csv": str(after_csv),
            "latest_week": str(latest_week.date()),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames",
        default="data/processed/frames.csv",
        help="Path to preprocessed panel CSV",
    )
    parser.add_argument(
        "--models_dir",
        default="models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--pred_dir",
        default="predictions",
        help="Directory to save per-cell scores",
    )
    args = parser.parse_args()

    train_ml(args.frames, args.models_dir, args.pred_dir)