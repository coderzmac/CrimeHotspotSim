"""
Preprocessing pipeline for CrimeHotspotSim (Cleveland only)
Authors: Emmanuel Bautista, Julian Fennema

This script:

1. Calls the Cleveland Crime_Incidents ArcGIS FeatureService API.
   - URL: https://services3.arcgis.com/dty2kHktVXHrqO8i/ArcGIS/rest/services/Crime_Incidents/FeatureServer/0/query
   - Uses fields: OffenseDate (date), LAT, LON.
2. Renames OffenseDate → DATE OCC so it plugs into the grid/aggregation pipeline.
3. Converts raw latitude/longitude into a regular grid (0.5 km size by default).
4. Aggregates into WEEKLY crime counts per grid cell.
5. Creates lag features (lag1, lag2, lag3).
6. Adds placeholder intervention/context fields (patrol_cars, ped_activity, etc.).
7. Saves:
   - frames.csv (full weekly dataset for model training)
   - inference_frame.csv (latest week's frame for hotspot mapping)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------
# 1. Load the crime dataset (only needed columns)
# ---------------------------------------------------------------
# Default Cleveland Crime Incidents FeatureServer layer 0 query URL
CLEVELAND_FEATURE_URL = (
    "https://services3.arcgis.com/dty2kHktVXHrqO8i/ArcGIS/rest/services/"
    "Crime_Incidents/FeatureServer/0/query"
)


# Helper: Cleveland API loader → DataFrame with DATE OCC, LAT, LON
def load_cleveland_crime_from_api(
    api_url: str = CLEVELAND_FEATURE_URL,
    grid_fields=("OffenseDate", "LAT", "LON"),
    batch_size: int = 2000,
) -> pd.DataFrame:
    """
    Fetches Cleveland crime incidents from the ArcGIS FeatureService API.
    Returns a DataFrame with at least DATE OCC, LAT, LON columns.

    Notes:
    - MaxRecordCount for this service is ~2000; we page using resultOffset.
    - OffenseDate is an esriFieldTypeDate (ms since epoch) on this service.
    """
    offense_date_field, lat_field, lon_field = grid_fields

    records = []
    offset = 0
    total = 0

    while True:
        params = {
            "where": "1=1",  # no filter – pull everything; filter later if needed
            "outFields": f"{offense_date_field},{lat_field},{lon_field}",
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": batch_size,
        }

        resp = requests.get(api_url, params=params)
        resp.raise_for_status()
        data = resp.json()

        features = data.get("features", [])
        n = len(features)
        if n == 0:
            # No more records
            break

        for feat in features:
            attrs = feat.get("attributes", {})
            records.append(attrs)

        total += n
        offset += n  # move by number of records actually returned

        # Progress print
        print(f"Fetched {n} records (total so far: {total})")

    if not records:
        raise RuntimeError("No records returned from Cleveland Crime_Incidents API.")

    df = pd.DataFrame.from_records(records)

    # Standardize column names so the core pipeline can stay the same
    rename_map = {
        offense_date_field: "DATE OCC",
        lat_field: "LAT",
        lon_field: "LON",
    }
    df = df.rename(columns=rename_map)

    # Convert date column; ArcGIS often returns ms since epoch, but it may also be
    # ISO strings. We handle both cases.
    date_series = df["DATE OCC"]
    if np.issubdtype(date_series.dtype, np.number):
        df["DATE OCC"] = pd.to_datetime(df["DATE OCC"], unit="ms", errors="coerce")
    else:
        df["DATE OCC"] = pd.to_datetime(df["DATE OCC"], errors="coerce")

    # Drop rows with missing core fields
    df = df.dropna(subset=["DATE OCC", "LAT", "LON"])

    # Drop all records from 2015 and earlier
    df = df[df["DATE OCC"].dt.year >= 2016]

    # Simple sanity check: how many unique weeks and min/max dates?
    print(f"Total incidents after cleaning: {len(df)}")
    print(f"DATE OCC range: {df['DATE OCC'].min()} → {df['DATE OCC'].max()}")

    return df[["DATE OCC", "LAT", "LON"]]




# ---------------------------------------------------------------------------
# Core panel-building pipeline
# ---------------------------------------------------------------------------
def build_panel(df: pd.DataFrame, outdir: str, grid_step: float = 0.005) -> None:
    """
    Shared pipeline that assumes df has columns:
    - DATE OCC (datetime-like)
    - LAT (float), LON (float)

    grid_step = 0.005 degrees ≈ 0.5 km
    This gives us meaningful spatial resolution while keeping compute light.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Ensure datetime
    df["DATE OCC"] = pd.to_datetime(df["DATE OCC"], errors="coerce")
    df = df.dropna(subset=["DATE OCC", "LAT", "LON"])

    # ---------------------------------------------------------------
    # 2. Build a spatial grid by rounding lat/lon into bins
    # ---------------------------------------------------------------
    df["lat_bin"] = (df["LAT"] / grid_step).round().astype(int)
    df["lon_bin"] = (df["LON"] / grid_step).round().astype(int)

    # Unique grid cell identifier
    df["grid_id"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)

    # ---------------------------------------------------------------
    # 3. Convert raw timestamps into WEEK START dates (Mondays)
    # ---------------------------------------------------------------
    df["week_start"] = df["DATE OCC"] - pd.to_timedelta(
        df["DATE OCC"].dt.weekday, unit="D"
    )
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
        freq="W-MON",
    )

    grids = agg[["grid_id", "lat_bin", "lon_bin"]].drop_duplicates()

    panel = (
        grids.assign(key=1)
        .merge(pd.DataFrame({"week_start": all_weeks, "key": 1}), on="key")
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
    # ---------------------------------------------------------------
    context_cols = [
        "poi_density",
        "nightlight",
        "income",
        "pop_density",
        "weather_temp",
        "events_count",
        "patrol_cars",
        "ped_activity",
    ]

    panel["poi_density"] = 0.0
    panel["nightlight"] = 0.0
    panel["income"] = 0.0
    panel["pop_density"] = 0.0
    panel["weather_temp"] = 0.0
    panel["events_count"] = 0.0
    panel["patrol_cars"] = 0.0
    panel["ped_activity"] = 1.0  # baseline = 1.0; intervention = multiply by 1.05

    # ---------------------------------------------------------------
    # 9. Save outputs
    # ---------------------------------------------------------------
    frames_csv = out / "frames.csv"
    panel.to_csv(frames_csv, index=False)

    latest_week = panel["week_start"].max()
    inference_frame = panel[panel["week_start"] == latest_week]
    inference_frame.to_csv(out / "inference_frame.csv", index=False)

    print(f"Saved:\n- {frames_csv}\n- {out / 'inference_frame.csv'}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=CLEVELAND_FEATURE_URL,
        help="Cleveland Crime_Incidents FeatureService query URL ",
    )
    parser.add_argument(
        "--outdir",
        default="data/processed",
        help="Output directory for frames.csv and inference_frame.csv",
    )
    parser.add_argument(
        "--grid_step",
        type=float,
        default=0.005,
        help="Grid step in degrees (~0.005 ≈ 0.5km).",
    )
    args = parser.parse_args()

    # Cleveland API load → harmonized DATE OCC / LAT / LON
    df = load_cleveland_crime_from_api(api_url=args.input)
    build_panel(df, args.outdir, grid_step=args.grid_step)


if __name__ == "__main__":
    main()
