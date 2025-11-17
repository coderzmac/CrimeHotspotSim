import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_panel(input_csv: str, outdir: str, grid_step: float = 0.005):

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load only the columns we need: date + coordinates
    df = pd.read_csv(
        input_csv,
        low_memory=False,
        usecols=["DATE OCC", "LAT", "LON"],  # keep it minimal
    )

    # Convert the "DATE OCC" text into a proper datetime.
    # errors='coerce' turns bad strings into NaT (missing).
    df["DATE OCC"] = pd.to_datetime(df["DATE OCC"], errors="coerce")

    # Drop any rows where date or coordinates are missing.
    df = df.dropna(subset=["DATE OCC", "LAT", "LON"])


    # 2. Build a simple spatial grid

    df["lat_bin"] = (df["LAT"] / grid_step).round().astype(int)
    df["lon_bin"] = (df["LON"] / grid_step).round().astype(int)

    # Combine lat_bin and lon_bin to form a unique grid_id (string).
    df["grid_id"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)


    # 3. Assign each crime to a weekly time bin

    df["week_start"] = df["DATE OCC"] - pd.to_timedelta(
        df["DATE OCC"].dt.weekday, unit="D"
    )
    df["week_start"] = df["week_start"].dt.normalize()  # drop time-of-day


    # 4. Aggregate crimes: weekly counts per grid cell

    agg = (
        df.groupby(["grid_id", "lat_bin", "lon_bin", "week_start"])
        .size()
        .reset_index(name="crime_count")
    )


    # 5. Build a complete panel (all grids x all weeks)

    all_weeks = pd.date_range(
        agg["week_start"].min(), agg["week_start"].max(), freq="W-MON"
    )
    grids = agg[["grid_id", "lat_bin", "lon_bin"]].drop_duplicates()

    # Create cartesian product of all grids and all weeks
    panel = (
        grids.assign(key=1)
        .merge(
            pd.DataFrame({"week_start": all_weeks, "key": 1}),
            on="key",
        )
        .drop(columns="key")
        # Left join to the aggregated counts (fill missing with 0)
        .merge(
            agg,
            on=["grid_id", "lat_bin", "lon_bin", "week_start"],
            how="left",
        )
        .fillna({"crime_count": 0})
        .sort_values(["grid_id", "week_start"])
        .reset_index(drop=True)
    )


    # 6. Create lag features (crime in previous weeks)

    for k in [1, 2, 3]:
        panel[f"lag{k}"] = (
            panel.groupby("grid_id")["crime_count"]
            .shift(k)          # shift k weeks back
            .fillna(0)        # first k weeks have no history
        )

    # 7. Approximate lat/lon for each grid cell (for mapping)
    # Reverse our binning to get a coordinate at the center of each cell.
    panel["lat"] = panel["lat_bin"] * grid_step
    panel["lon"] = panel["lon_bin"] * grid_step


    # 8. Add placeholder context & intervention features

    for col in [
        "poi_density",
        "nightlight",
        "income",
        "pop_density",
        "weather_temp",
        "events_count",
        "patrol_cars",    # intervention: number of patrol units
    ]:
        panel[col] = 0.0


    panel["ped_activity"] = 1.0


    # 9. Save the full panel and the latest-week slice
    frames_csv = out_path / "frames.csv"
    panel.to_csv(frames_csv, index=False)

    latest_week = panel["week_start"].max()
    inference_frame = panel[panel["week_start"] == latest_week].copy()
    inference_csv = out_path / "inference_frame.csv"
    inference_frame.to_csv(inference_csv, index=False)

    print(
        {
            "frames_csv": str(frames_csv),
            "inference_csv": str(inference_csv),
            "weeks_range": (
                str(panel["week_start"].min().date()),
                str(panel["week_start"].max().date()),
            ),
            "n_grids": int(panel["grid_id"].nunique()),
            "latest_week": str(latest_week.date()),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw crime CSV")
    parser.add_argument(
        "--outdir",
        default="data/processed",
        help="Directory to save processed CSVs",
    )
    args = parser.parse_args()

    build_panel(args.input, args.outdir)