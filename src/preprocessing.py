"""
Preprocessing pipeline for CrimeHotspotSim
Author: Emmanuel Bautista

This script:
1. Loads the official crime dataset: Crime_Data_from_2020_to_Present.csv
2. Extracts LOCATION + DATE information.
3. Converts raw latitude/longitude into a regular grid (0.5 km size).
4. Aggregates into WEEKLY crime counts per grid cell.
5. Creates lag features (lag1, lag2, lag3).
6. Adds placeholder intervention/context fields (patrol_cars, ped_activity, etc.).
7. Saves:
   - frames.csv (full weekly dataset for model training)
   - inference_frame.csv (latest week's frame for hotspot mapping)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def main(input_csv, outdir, grid_step=0.005):
    """
    grid_step = 0.005 degrees ≈ 0.5 km
    This gives us meaningful spatial resolution while keeping compute light.
    """

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Load the crime dataset (only needed columns)
    # ---------------------------------------------------------------
    df = pd.read_csv(input_csv, low_memory=False,
                     usecols=["DATE OCC", "LAT", "LON"])

    # Convert DATE OCC column into a datetime format
    df["DATE OCC"] = pd.to_datetime(df["DATE OCC"], errors="coerce")

    # Remove any rows WITHOUT dates or coordinates
    df = df.dropna(subset=["DATE OCC", "LAT", "LON"])


    # ---------------------------------------------------------------
    # 2. Build a spatial grid by rounding lat/lon into bins
    # ---------------------------------------------------------------
    # Example:
    # LAT = 34.0522 → lat_bin = round(34.0522 / 0.005) = 6810
    df["lat_bin"] = (df["LAT"] / grid_step).round().astype(int)
    df["lon_bin"] = (df["LON"] / grid_step).round().astype(int)

    # Unique grid cell identifier
    df["grid_id"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)


    # ---------------------------------------------------------------
    # 3. Convert raw timestamps into WEEK START dates
    # ---------------------------------------------------------------
    # Standard trick: subtract weekday → aligns all events to Monday of that week
    df["week_start"] = df["DATE OCC"] - pd.to_timedelta(df["DATE OCC"].dt.weekday, unit="D")
    df["week_start"] = df["week_start"].dt.normalize()


    # ---------------------------------------------------------------
    # 4. Weekly aggregation: crime_count per (grid, week)
    # ---------------------------------------------------------------
    agg = (
        df.groupby(["grid_id", "lat_bin", "lon_bin", "week_start"])
          .size()
          .reset_index(name="crime_count")
    )


    # ---------------------------------------------------------------
    # 5. Create a FULL PANEL (every grid × every week)
    # This guarantees clean time series → needed for lag features.
    # ---------------------------------------------------------------
    all_weeks = pd.date_range(
        start=agg["week_start"].min(),
        end=agg["week_start"].max(),
        freq="W-MON"
    )

    grids = agg[["grid_id", "lat_bin", "lon_bin"]].drop_duplicates()

    panel = (
        grids.assign(key=1)
        .merge(
            pd.DataFrame({"week_start": all_weeks, "key": 1}),
            on="key"
        )
        .drop(columns="key")
        .merge(agg, on=["grid_id", "lat_bin", "lon_bin", "week_start"], how="left")
        .fillna({"crime_count": 0})
        .sort_values(["grid_id", "week_start"])
    )


    # ---------------------------------------------------------------
    # 6. Add lag features (1–3 weeks)
    # ---------------------------------------------------------------
    for k in [1, 2, 3]:
        panel[f"lag{k}"] = panel.groupby("grid_id")["crime_count"].shift(k).fillna(0)


    # ---------------------------------------------------------------
    # 7. Add geographic centroids (inverse of grid bin)
    # ---------------------------------------------------------------
    panel["lat"] = panel["lat_bin"] * grid_step
    panel["lon"] = panel["lon_bin"] * grid_step


    # ---------------------------------------------------------------
    # 8. Add placeholder intervention/context features
    # These will be real later (pedestrian activity, patrol cars, POI density)
    # ---------------------------------------------------------------
    context_cols = [
        "poi_density", "nightlight", "income",
        "pop_density", "weather_temp", "events_count",
        "patrol_cars", "ped_activity"
    ]

    panel["poi_density"] = 0.0
    panel["nightlight"] = 0.0
    panel["income"] = 0.0
    panel["pop_density"] = 0.0
    panel["weather_temp"] = 0.0
    panel["events_count"] = 0.0
    panel["patrol_cars"] = 0.0
    panel["ped_activity"] = 1.0   # baseline = 1.0; intervention = multiply by 1.05


    # ---------------------------------------------------------------
    # 9. Save outputs
    # ---------------------------------------------------------------
    frames_csv = out / "frames.csv"
    panel.to_csv(frames_csv, index=False)

    latest_week = panel["week_start"].max()
    inference_frame = panel[panel["week_start"] == latest_week]
    inference_frame.to_csv(out / "inference_frame.csv", index=False)

    print(f"Saved:\n- {frames_csv}\n- {out / 'inference_frame.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Path to Crime_Data_from_2020_to_Present.csv")
    parser.add_argument("--outdir", default="data/processed")
    args = parser.parse_args()

    main(args.input, args.outdir)